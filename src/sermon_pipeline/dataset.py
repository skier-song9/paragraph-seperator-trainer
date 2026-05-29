from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .extractors.datalab import parse_datalab_json
from .extractors.docx import parse_docx
from .extractors.hwp import load_libhwp_reader, parse_hwp
from .io import write_json
from .models import PreparedDocument


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _document_id(source_type: str, path: Path, root: Path) -> str:
    digest = hashlib.sha1(_relative_path(path, root).encode("utf-8")).hexdigest()[:12]
    return f"{source_type}.{digest}"


def discover_dataset_sources(root: Path) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for path in sorted((root / "datas" / "datalab_parsed").glob("**/*.json")):
        sources.append(("datalab_parsed_json", path))
    for path in sorted((root / "datas" / "docx").glob("*.docx")):
        sources.append(("docx", path))
    for path in sorted((root / "datas" / "hwps").glob("*.hwp")):
        sources.append(("hwp", path))
    return sources


def _prepare_document(
    source_type: str,
    path: Path,
    root: Path,
    max_sentences_per_document: int | None,
    hwp_reader_factory: Callable[[str], Any],
) -> PreparedDocument:
    document_id = _document_id(source_type, path, root)
    if source_type == "datalab_parsed_json":
        return parse_datalab_json(
            path,
            root=root,
            document_id=document_id,
            max_sentences=max_sentences_per_document,
        )
    if source_type == "docx":
        return parse_docx(
            path,
            root=root,
            document_id=document_id,
            max_sentences=max_sentences_per_document,
        )
    if source_type == "hwp":
        return parse_hwp(
            path,
            root=root,
            document_id=document_id,
            reader_factory=hwp_reader_factory,
            max_sentences=max_sentences_per_document,
        )
    raise ValueError(f"Unsupported source_type={source_type}")


def _document_row(document: PreparedDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "source_type": document.source_type,
        "document_kind": document.document_kind,
        "source_path": document.source_path,
        "reasoning_effort": document.reasoning_effort,
        "effort_label": document.effort_label,
        "block_count": len(document.blocks),
        "sentence_count": len(document.sentences),
        "removed_foreign_paragraphs": document.removed_foreign_paragraphs,
        "extraction_notes": list(document.extraction_notes),
        "status": "ok",
    }


def _sentence_rows(document: PreparedDocument) -> Iterable[dict[str, Any]]:
    for index, sentence in enumerate(document.sentences):
        row = sentence.to_payload()
        row.update(
            {
                "document_id": document.document_id,
                "document_source_type": document.source_type,
                "document_kind": document.document_kind,
                "source_path": document.source_path,
                "sentence_index": index,
            }
        )
        yield row


def _failure_row(
    source_type: str,
    path: Path,
    root: Path,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "source_type": source_type,
        "source_path": _relative_path(path, root),
        "exception_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _serialize_jsonl_line(row: dict[str, Any]) -> str:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    line.encode("utf-8")
    return line


def _write_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(_serialize_jsonl_line(row))
    handle.flush()


def _write_failures_markdown(path: Path, failures: list[dict[str, Any]]) -> None:
    lines = [
        "# Dataset Build Failures",
        "",
        "| source_type | source_path | exception_type | error |",
        "|---|---|---|---|",
    ]
    for failure in failures:
        error = str(failure["error"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| {source_type} | `{source_path}` | {exception_type} | {error} |".format(
                source_type=failure["source_type"],
                source_path=failure["source_path"],
                exception_type=failure["exception_type"],
                error=error[:300],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    root: Path,
    out_dir: Path,
    max_sentences_per_document: int | None = None,
    hwp_reader_factory: Callable[[str], Any] = load_libhwp_reader,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_dataset_sources(root)
    started_at = datetime.now().isoformat(timespec="seconds")
    failures: list[dict[str, Any]] = []
    logs = [
        f"{started_at} dataset_build_start root={root} total_files={len(sources)}"
    ]
    by_source_type: dict[str, dict[str, int]] = {}
    succeeded = 0
    total_sentences = 0

    documents_path = out_dir / "documents.jsonl"
    sentences_path = out_dir / "sentences.jsonl"
    failures_path = out_dir / "failures.jsonl"

    with (
        documents_path.open("w", encoding="utf-8") as documents_handle,
        sentences_path.open("w", encoding="utf-8") as sentences_handle,
        failures_path.open("w", encoding="utf-8") as failures_handle,
    ):
        for source_type, path in sources:
            stats = by_source_type.setdefault(
                source_type,
                {"total": 0, "succeeded": 0, "failed": 0, "sentences": 0},
            )
            stats["total"] += 1
            rel_path = _relative_path(path, root)
            try:
                document = _prepare_document(
                    source_type,
                    path,
                    root,
                    max_sentences_per_document,
                    hwp_reader_factory,
                )
                if not document.sentences:
                    raise ValueError("document produced zero sentences")
                document_line = _serialize_jsonl_line(_document_row(document))
                sentence_lines = [
                    _serialize_jsonl_line(sentence_row)
                    for sentence_row in _sentence_rows(document)
                ]
                documents_handle.write(document_line)
                documents_handle.flush()
                for sentence_line in sentence_lines:
                    sentences_handle.write(sentence_line)
                sentences_handle.flush()
                succeeded += 1
                sentence_count = len(document.sentences)
                total_sentences += sentence_count
                stats["succeeded"] += 1
                stats["sentences"] += sentence_count
                logs.append(f"OK {source_type} {rel_path} sentences={sentence_count}")
            except Exception as exc:
                failure = _failure_row(source_type, path, root, exc)
                failures.append(failure)
                _write_jsonl_line(failures_handle, failure)
                stats["failed"] += 1
                logs.append(
                    "FAIL {source_type} {rel_path} {exc_type}: {error}".format(
                        source_type=source_type,
                        rel_path=rel_path,
                        exc_type=type(exc).__name__,
                        error=str(exc).replace("\n", " ")[:500],
                    )
                )

    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "root": str(root),
        "out_dir": str(out_dir),
        "total_files": len(sources),
        "succeeded": succeeded,
        "failed": len(failures),
        "total_sentences": total_sentences,
        "by_source_type": by_source_type,
        "output_files": {
            "documents": str(documents_path),
            "sentences": str(sentences_path),
            "failures": str(failures_path),
            "failure_report": str(out_dir / "failures.md"),
            "log": str(out_dir / "dataset.log"),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    _write_failures_markdown(out_dir / "failures.md", failures)
    logs.append(
        f"{finished_at} dataset_build_finish succeeded={succeeded} "
        f"failed={len(failures)} total_sentences={total_sentences}"
    )
    (out_dir / "dataset.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-sentences-per-document", type=int)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        run_id = datetime.now().strftime("dataset_%Y%m%d_%H%M%S")
        out_dir = root / "tests" / "results" / run_id
    summary = build_dataset(
        root=root,
        out_dir=out_dir,
        max_sentences_per_document=args.max_sentences_per_document,
    )
    print(summary["out_dir"])
    print(
        "total_files={total_files} succeeded={succeeded} "
        "failed={failed} total_sentences={total_sentences}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
