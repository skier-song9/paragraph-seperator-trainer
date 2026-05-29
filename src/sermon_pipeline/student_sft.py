from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import BOUNDARY_TYPES
from .io import write_json

STUDENT_LABELS = tuple(label for label in BOUNDARY_TYPES if label != "none")
SPLITS = ("train", "validation", "test")


def _jsonl(row: dict[str, Any]) -> str:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    text.encode("utf-8")
    return text


def _load_jsonl_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(f"JSONL row {line_number} must include custom_id")
            rows[custom_id] = row
    return rows


def render_student_input(sentences: list[tuple[str, str]]) -> str:
    lines = [
        "<TASK>",
        "Identify sermon paragraph boundaries after the listed sentences.",
        "Return one line per boundary as: LOCAL_SID boundary_type.",
        "If there is no boundary, return NO_BOUNDARY.",
        f"Allowed labels: {', '.join(STUDENT_LABELS)}",
        "",
        "<TEXT>",
    ]
    lines.extend(f"<{local_sid}> {text}" for local_sid, text in sentences)
    return "\n".join(lines)


def _boundary_lines(annotation: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in annotation.get("boundary_annotations", []):
        if not isinstance(item, dict):
            continue
        boundary_type = str(item.get("boundary_type", "none"))
        if item.get("split_after") is True and boundary_type != "none":
            lines.append(f"{item['local_sid']} {boundary_type}")
    return lines


def sparse_boundary_output(annotation: dict[str, Any]) -> str:
    lines = _boundary_lines(annotation)
    return "\n".join(lines) if lines else "NO_BOUNDARY"


def first_boundary_output(annotation: dict[str, Any]) -> str:
    lines = _boundary_lines(annotation)
    return lines[0] if lines else "NO_BOUNDARY"


def split_for_document(document_id: str) -> str:
    bucket = int(hashlib.sha1(document_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def _sid_number(local_sid: str) -> int:
    if local_sid.startswith("S") and local_sid[1:].isdigit():
        return int(local_sid[1:])
    return 0


def _sentences_from_mapping(mapping: dict[str, Any]) -> list[tuple[str, str]]:
    target_sids = {
        str(local_sid) for local_sid in mapping.get("target_local_sids", [])
    }
    local_to_source = mapping.get("local_to_source", {})
    if not isinstance(local_to_source, dict):
        raise ValueError("mapping local_to_source must be an object")

    sentences: list[tuple[str, str]] = []
    for local_sid in sorted(target_sids, key=_sid_number):
        item = local_to_source[local_sid]
        if not isinstance(item, dict):
            raise ValueError(f"mapping for {local_sid} must be an object")
        sentences.append((local_sid, str(item["text"])))
    return sentences


def _example(
    example_id: str,
    document_id: str,
    input_text: str,
    output_text: str,
    target_type: str,
) -> dict[str, str]:
    return {
        "example_id": example_id,
        "document_id": document_id,
        "target_type": target_type,
        "input": input_text,
        "output": output_text,
    }


def build_sft_datasets(
    annotations_path: Path,
    windows_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings = _load_jsonl_by_custom_id(windows_path)
    annotations = _load_jsonl_by_custom_id(annotations_path)

    dataset_dirs = {
        "sparse_multi_boundary": out_dir / "sparse_multi_boundary",
        "first_boundary": out_dir / "first_boundary",
    }
    handles = {}
    try:
        for family, family_dir in dataset_dirs.items():
            family_dir.mkdir(parents=True, exist_ok=True)
            for split in SPLITS:
                handles[(family, split)] = (family_dir / f"{split}.jsonl").open(
                    "w", encoding="utf-8"
                )

        split_counts = {
            family: {split: 0 for split in SPLITS} for family in dataset_dirs
        }
        mapping_rows: list[dict[str, Any]] = []
        example_count = 0

        for custom_id in sorted(annotations):
            annotation = annotations[custom_id]
            mapping = mappings[custom_id]
            document_id = str(mapping["document_id"])
            split = split_for_document(document_id)
            input_text = render_student_input(_sentences_from_mapping(mapping))
            sparse_example_id = f"sparse:{custom_id}"
            first_example_id = f"first:{custom_id}"

            sparse_example = _example(
                sparse_example_id,
                document_id,
                input_text,
                sparse_boundary_output(annotation),
                "sparse_multi_boundary",
            )
            first_example = _example(
                first_example_id,
                document_id,
                input_text,
                first_boundary_output(annotation),
                "first_boundary",
            )
            handles[("sparse_multi_boundary", split)].write(_jsonl(sparse_example))
            handles[("first_boundary", split)].write(_jsonl(first_example))
            split_counts["sparse_multi_boundary"][split] += 1
            split_counts["first_boundary"][split] += 1
            example_count += 1

            mapping_rows.append(
                {
                    "sparse_example_id": sparse_example_id,
                    "first_boundary_example_id": first_example_id,
                    "custom_id": custom_id,
                    "document_id": document_id,
                    "source_path": mapping.get("source_path"),
                    "local_to_source": mapping.get("local_to_source"),
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    with (out_dir / "mappings.jsonl").open("w", encoding="utf-8") as handle:
        for row in mapping_rows:
            handle.write(_jsonl(row))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "annotations_path": str(annotations_path),
        "windows_path": str(windows_path),
        "out_dir": str(out_dir),
        "example_count": example_count,
        "split_counts": split_counts,
    }
    write_json(out_dir / "sft_run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    summary = build_sft_datasets(
        annotations_path=args.annotations,
        windows_path=args.windows,
        out_dir=args.out_dir,
    )
    print(f"out_dir: {summary['out_dir']}")
    print(f"example_count: {summary['example_count']}")
    print(f"split_counts: {json.dumps(summary['split_counts'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
