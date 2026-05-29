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
            if custom_id in rows:
                raise ValueError(
                    f"{path}: duplicate custom_id at line {line_number}: {custom_id}"
                )
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
    boundaries: list[tuple[str, str]] = []
    custom_id = annotation.get("custom_id", "<unknown>")
    for item in annotation.get("boundary_annotations", []):
        if not isinstance(item, dict):
            continue
        if item.get("split_after") is not True:
            continue
        local_sid = item.get("local_sid")
        boundary_type = item.get("boundary_type")
        if boundary_type not in STUDENT_LABELS:
            raise ValueError(
                f"{custom_id}: invalid boundary_type for {local_sid}: {boundary_type!r}"
            )
        boundaries.append((local_sid, boundary_type))
    boundaries.sort(key=lambda item: _sid_number(item[0]))
    return [f"{local_sid} {boundary_type}" for local_sid, boundary_type in boundaries]


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
    custom_id = mapping.get("custom_id", "<unknown>")
    target_local_sids = mapping.get("target_local_sids")
    if not isinstance(target_local_sids, list):
        raise ValueError(f"{custom_id}: missing target_local_sids")
    target_sids = {str(local_sid) for local_sid in target_local_sids}
    local_to_source = mapping.get("local_to_source")
    if not isinstance(local_to_source, dict):
        raise ValueError(f"{custom_id}: missing local_to_source")

    sentences: list[tuple[str, str]] = []
    for local_sid in sorted(target_sids, key=_sid_number):
        if local_sid not in local_to_source:
            raise ValueError(f"{custom_id}: missing local_to_source entry for {local_sid}")
        item = local_to_source[local_sid]
        if not isinstance(item, dict):
            raise ValueError(f"{custom_id}: mapping for {local_sid} must be an object")
        if "text" not in item:
            raise ValueError(f"{custom_id}: missing text for {local_sid}")
        text = item["text"]
        if not isinstance(text, str):
            raise ValueError(f"{custom_id}: text for {local_sid} must be a string")
        sentences.append((local_sid, text))
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
    mappings = _load_jsonl_by_custom_id(windows_path)
    annotations = _load_jsonl_by_custom_id(annotations_path)

    dataset_dirs = {
        "sparse_multi_boundary": out_dir / "sparse_multi_boundary",
        "first_boundary": out_dir / "first_boundary",
    }
    split_counts = {
        family: {split: 0 for split in SPLITS} for family in dataset_dirs
    }
    examples_by_family_split: dict[tuple[str, str], list[dict[str, str]]] = {
        (family, split): [] for family in dataset_dirs for split in SPLITS
    }
    mapping_rows: list[dict[str, Any]] = []

    for custom_id in sorted(annotations):
        if custom_id not in mappings:
            raise ValueError(f"missing mapping for {custom_id}")

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
        examples_by_family_split[("sparse_multi_boundary", split)].append(
            sparse_example
        )
        examples_by_family_split[("first_boundary", split)].append(first_example)
        split_counts["sparse_multi_boundary"][split] += 1
        split_counts["first_boundary"][split] += 1

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

    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        for family, family_dir in dataset_dirs.items():
            family_dir.mkdir(parents=True, exist_ok=True)
            for split in SPLITS:
                handles[(family, split)] = (family_dir / f"{split}.jsonl").open(
                    "w", encoding="utf-8"
                )

        for key, examples in examples_by_family_split.items():
            for example in examples:
                handles[key].write(_jsonl(example))
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
        "example_count": len(annotations),
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
