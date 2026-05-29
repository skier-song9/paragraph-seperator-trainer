from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .extractors.datalab import parse_datalab_json
from .extractors.docx import parse_docx
from .extractors.hwp import load_libhwp_reader, parse_hwp
from .io import load_dotenv, render_comparison, summarize_annotation, write_json
from .models import PreparedDocument
from .openai_client import call_openai
from .teacher import build_payload
from .validation import teacher_output_to_training_rows, validate_teacher_output


def select_datalab_json(root: Path) -> Path:
    for path in sorted((root / "datas" / "datalab_parsed").glob("**/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("success") is True and data.get("html"):
            return path
    raise FileNotFoundError("No usable datalab_parsed JSON found.")


def select_docx(root: Path) -> Path:
    docx_dir = root / "datas" / "docx"
    for path in sorted(docx_dir.glob("*.docx")):
        normalized_name = unicodedata.normalize("NFC", path.name)
        if "세마포" in normalized_name and "10" in normalized_name:
            return path
    candidates = sorted(docx_dir.glob("*.docx"))
    if not candidates:
        raise FileNotFoundError("No DOCX files found.")
    return candidates[0]


def select_hwp(root: Path) -> Path:
    candidates = sorted((root / "datas" / "hwps").glob("*.hwp"))
    if not candidates:
        raise FileNotFoundError("No HWP files found.")
    return candidates[0]


def prepare_sample_documents(
    root: Path,
    max_sentences: int,
    hwp_reader_factory: Callable[[str], Any] = load_libhwp_reader,
) -> list[PreparedDocument]:
    return [
        parse_datalab_json(
            select_datalab_json(root),
            root=root,
            document_id="datalab_sample",
            max_sentences=max_sentences,
        ),
        parse_docx(
            select_docx(root),
            root=root,
            document_id="docx_sample",
            max_sentences=max_sentences,
        ),
        parse_hwp(
            select_hwp(root),
            root=root,
            document_id="hwp_sample",
            reader_factory=hwp_reader_factory,
            max_sentences=max_sentences,
        ),
    ]


def _case_id(document: PreparedDocument) -> str:
    if document.source_type == "datalab_parsed_json":
        return "datalab_xhigh"
    if document.source_type == "docx":
        return "docx_high"
    return "hwp_high"


def _row_for_document(case_id: str, document: PreparedDocument) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source_type": document.source_type,
        "source_path": document.source_path,
        "reasoning_effort": document.reasoning_effort,
        "effort_label": document.effort_label,
        "sentence_count": len(document.sentences),
        "extraction_notes": list(document.extraction_notes),
        "removed_foreign_paragraphs": document.removed_foreign_paragraphs,
        "status": "prepared",
    }


def run(
    root: Path,
    out_dir: Path,
    model: str,
    max_sentences: int,
    max_output_tokens: int,
    timeout: int,
    dry_run: bool,
    hwp_reader_factory: Callable[[str], Any] = load_libhwp_reader,
) -> int:
    input_dir = out_dir / "inputs"
    response_dir = out_dir / "responses"
    annotation_dir = out_dir / "annotations"
    rows_dir = out_dir / "rows"
    out_dir.mkdir(parents=True, exist_ok=True)

    documents = prepare_sample_documents(root, max_sentences, hwp_reader_factory)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    rows: list[dict[str, Any]] = []

    for document in documents:
        case_id = _case_id(document)
        payload = build_payload(
            document,
            model=model,
            max_output_tokens=max_output_tokens,
        )
        write_json(input_dir / f"{case_id}.payload.json", payload)
        write_json(
            input_dir / f"{case_id}.sentences.json",
            {
                "source_path": document.source_path,
                "source_type": document.source_type,
                "reasoning_effort": document.reasoning_effort,
                "sentences": [
                    sentence.to_payload() for sentence in document.sentences
                ],
            },
        )

        row = _row_for_document(case_id, document)
        if dry_run or not api_key:
            row["status"] = "skipped"
            row["error"] = "dry_run" if dry_run else "missing_OPENAI_API_KEY"
            rows.append(row)
            continue

        try:
            response, output_text, elapsed = call_openai(payload, api_key, timeout)
            write_json(response_dir / f"{case_id}.raw_response.json", response)
            (response_dir / f"{case_id}.output_text.txt").write_text(
                output_text,
                encoding="utf-8",
            )
            if not output_text.strip():
                incomplete = response.get("incomplete_details") or {}
                raise RuntimeError(
                    "empty output_text; "
                    f"response_status={response.get('status')}; "
                    f"incomplete_reason={incomplete.get('reason')}; "
                    f"usage={response.get('usage')}"
                )

            annotation = json.loads(output_text)
            write_json(response_dir / f"{case_id}.output.json", annotation)
            write_json(annotation_dir / f"{case_id}.annotation.json", annotation)
            issues = validate_teacher_output(document, annotation)
            training_rows = teacher_output_to_training_rows(document, annotation)
            write_json(
                rows_dir / f"{case_id}.training_rows.json",
                [item.to_payload() for item in training_rows],
            )

            row.update(summarize_annotation(annotation))
            row["status"] = "ok" if not issues else "error"
            row["usage"] = response.get("usage", {})
            row["elapsed_seconds"] = elapsed
            if issues:
                row["error"] = "; ".join(issue.message for issue in issues[:5])
                row["validation_issues"] = [issue.to_payload() for issue in issues]
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

    write_json(out_dir / "run_summary.json", rows)
    (out_dir / "comparison.md").write_text(
        render_comparison(rows, model),
        encoding="utf-8",
    )
    return 0 if all(row["status"] in {"ok", "skipped"} for row in rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-sentences", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "tests" / "results" / run_id
    status = run(
        root=root,
        out_dir=out_dir,
        model=args.model,
        max_sentences=args.max_sentences,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    print(out_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
