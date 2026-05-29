from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import write_json
from .student_sft import _sid_number
from .teacher_ingest import validate_annotation

SMOKE_QUALITY_FLAG = "silver_smoke_labels_not_for_model_evaluation"


def _jsonl(row: dict[str, Any]) -> str:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    text.encode("utf-8")
    return text


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} must be a JSON object")
            rows.append(row)
    return rows


_SCRIPTURE_REF_RE = re.compile(r"^\([^)]*(?:요|마|막|눅|롬|고전|고후|창|출|시|사)[^)]*\d[^)]*\)$")


def silver_boundary_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "none"
    if stripped == "기도하겠습니다.":
        return "prayer_or_closing"
    if stripped in {"아멘.", "아멘"} or stripped.endswith("기도합니다."):
        return "prayer_or_closing"
    if stripped.startswith("Chapter ") or stripped.startswith("제"):
        return "topic_shift"
    if _SCRIPTURE_REF_RE.match(stripped):
        return "scripture_reading_start"
    if "다음과 같이 표현합니다" in stripped:
        return "scripture_reading_start"
    return "none"


def annotation_from_window(window: dict[str, Any]) -> dict[str, Any]:
    custom_id = window.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise ValueError("window missing custom_id")
    target_local_sids = window.get("target_local_sids")
    if not isinstance(target_local_sids, list):
        raise ValueError(f"{custom_id}: missing target_local_sids")
    local_to_source = window.get("local_to_source")
    if not isinstance(local_to_source, dict):
        raise ValueError(f"{custom_id}: missing local_to_source")

    boundary_annotations: list[dict[str, Any]] = []
    for local_sid in sorted((str(value) for value in target_local_sids), key=_sid_number):
        source = local_to_source.get(local_sid)
        if not isinstance(source, dict):
            raise ValueError(f"{custom_id}: missing local_to_source entry for {local_sid}")
        source_sentence_id = source.get("source_sentence_id")
        text = source.get("text")
        if not isinstance(source_sentence_id, str) or not isinstance(text, str):
            raise ValueError(f"{custom_id}: invalid source mapping for {local_sid}")
        boundary_type = silver_boundary_type(text)
        split_after = boundary_type != "none"
        boundary_annotations.append(
            {
                "local_sid": local_sid,
                "source_sentence_id": source_sentence_id,
                "split_after": split_after,
                "boundary_type": boundary_type,
                "confidence": 0.35 if split_after else 0.6,
                "rationale": (
                    "Deterministic smoke label from obvious lexical cue."
                    if split_after
                    else ""
                ),
            }
        )

    return {
        "custom_id": custom_id,
        "boundary_annotations": boundary_annotations,
        "quality_flags": [SMOKE_QUALITY_FLAG],
    }


def build_silver_annotations(
    windows_path: Path,
    out_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    windows = _iter_jsonl(windows_path)
    if limit is not None:
        windows = windows[:limit]

    annotations = [annotation_from_window(window) for window in windows]
    failures: list[dict[str, Any]] = []
    for window, annotation in zip(windows, annotations):
        issues = validate_annotation(window, annotation)
        if issues:
            failures.append(
                {
                    "custom_id": window.get("custom_id"),
                    "issues": issues,
                }
            )
    if failures:
        raise ValueError(f"silver annotation validation failed: {failures}")

    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "teacher_annotations.jsonl"
    with annotations_path.open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            handle.write(_jsonl(annotation))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "windows_path": str(windows_path),
        "out_dir": str(out_dir),
        "annotations_path": str(annotations_path),
        "annotation_count": len(annotations),
        "quality_flag": SMOKE_QUALITY_FLAG,
    }
    write_json(out_dir / "silver_run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    summary = build_silver_annotations(
        windows_path=args.windows,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    print(f"out_dir: {summary['out_dir']}")
    print(f"annotation_count: {summary['annotation_count']}")
    print(f"quality_flag: {summary['quality_flag']}")
    return 0
