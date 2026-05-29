from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .constants import BOUNDARY_TYPES
from .io import extract_output_text, write_json


def _jsonl(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    payload.encode("utf-8")
    return payload


def _load_jsonl_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(f"Missing custom_id in {path} line {line_number}")
            if custom_id in rows:
                raise ValueError(
                    f"Duplicate custom_id {custom_id!r} in {path} line {line_number}"
                )
            rows[custom_id] = row
    return rows


def _issue(code: str, **fields: Any) -> dict[str, Any]:
    issue = {"code": code}
    issue.update(fields)
    return issue


def validate_annotation(
    mapping: dict[str, Any], annotation: dict[str, Any]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    mapping_custom_id = mapping.get("custom_id")
    annotation_custom_id = annotation.get("custom_id")
    if annotation_custom_id != mapping_custom_id:
        issues.append(
            _issue(
                "custom_id_mismatch",
                expected=mapping_custom_id,
                actual=annotation_custom_id,
            )
        )

    expected = set(mapping.get("target_local_sids", []))
    local_to_source = mapping.get("local_to_source", {})
    boundary_types = set(BOUNDARY_TYPES)
    seen: set[Any] = set()

    annotation_items = annotation.get("boundary_annotations", [])
    if not isinstance(annotation_items, list):
        return [_issue("invalid_boundary_annotations")]

    for index, item in enumerate(annotation_items):
        if not isinstance(item, dict):
            issues.append(_issue("invalid_boundary_annotation_item", index=index))
            continue

        local_sid = item.get("local_sid")
        if local_sid in seen:
            issues.append(_issue("duplicate_local_sid", local_sid=local_sid))
        seen.add(local_sid)

        if local_sid not in expected:
            issues.append(_issue("unknown_or_context_local_sid", local_sid=local_sid))

        source_meta = (
            local_to_source.get(local_sid) if isinstance(local_to_source, dict) else None
        )
        expected_source_id = (
            source_meta.get("source_sentence_id")
            if isinstance(source_meta, dict)
            else None
        )
        actual_source_id = item.get("source_sentence_id")
        if expected_source_id is not None and actual_source_id != expected_source_id:
            issues.append(
                _issue(
                    "source_sentence_id_mismatch",
                    local_sid=local_sid,
                    expected=expected_source_id,
                    actual=actual_source_id,
                )
            )

        boundary_type = item.get("boundary_type")
        if boundary_type not in boundary_types:
            issues.append(
                _issue(
                    "unknown_boundary_type",
                    local_sid=local_sid,
                    boundary_type=boundary_type,
                )
            )

        split_after = item.get("split_after")
        if not isinstance(split_after, bool):
            issues.append(
                _issue(
                    "invalid_split_after",
                    local_sid=local_sid,
                    split_after=split_after,
                )
            )
        else:
            if split_after is False and boundary_type != "none":
                issues.append(
                    _issue(
                        "split_false_boundary_not_none",
                        local_sid=local_sid,
                        boundary_type=boundary_type,
                    )
                )
            if split_after is True and boundary_type == "none":
                issues.append(_issue("split_true_boundary_none", local_sid=local_sid))

        confidence = item.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or confidence < 0
            or confidence > 1
        ):
            issues.append(
                _issue(
                    "invalid_confidence",
                    local_sid=local_sid,
                    confidence=confidence,
                )
            )

    for local_sid in sorted(expected - seen):
        issues.append(_issue("missing_target_annotation", local_sid=local_sid))

    return issues


def _extract_batch_annotation(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError("Batch row response must be an object")
    status_code = response.get("status_code")
    if status_code != 200:
        raise RuntimeError(f"Batch response status_code={status_code}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError("Batch response body must be an object")
    output_text = extract_output_text(body)
    if not output_text:
        raise ValueError("Batch response body has no output text")
    annotation = json.loads(output_text)
    if not isinstance(annotation, dict):
        raise ValueError("Batch output text must decode to a JSON object")
    return annotation


def _needs_review(annotation: dict[str, Any], threshold: float) -> bool:
    if annotation.get("quality_flags"):
        return True
    for item in annotation.get("boundary_annotations", []):
        if not isinstance(item, dict):
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return True
        if confidence < threshold:
            return True
    return False


def ingest_batch_results(
    windows_path: Path,
    batch_output_path: Path,
    out_dir: Path,
    review_confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    mappings = _load_jsonl_by_custom_id(windows_path)
    annotation_path = out_dir / "teacher_annotations.jsonl"
    issue_path = out_dir / "teacher_validation_issues.jsonl"
    failure_path = out_dir / "teacher_failures.jsonl"
    review_path = out_dir / "needs_human_review.jsonl"

    ok_count = 0
    issue_count = 0
    failure_count = 0
    review_count = 0
    seen_custom_ids: set[str] = set()

    with (
        batch_output_path.open("r", encoding="utf-8") as batch_handle,
        annotation_path.open("w", encoding="utf-8") as annotation_handle,
        issue_path.open("w", encoding="utf-8") as issue_handle,
        failure_path.open("w", encoding="utf-8") as failure_handle,
        review_path.open("w", encoding="utf-8") as review_handle,
    ):
        for line_number, line in enumerate(batch_handle, start=1):
            if not line.strip():
                continue
            custom_id: Any = None
            try:
                row = json.loads(line)
                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    raise ValueError("Batch row missing custom_id")
                if custom_id in seen_custom_ids:
                    raise ValueError(f"Duplicate batch custom_id {custom_id!r}")
                seen_custom_ids.add(custom_id)
                mapping = mappings.get(custom_id)
                if mapping is None:
                    raise ValueError(f"No window mapping for custom_id {custom_id!r}")

                annotation = _extract_batch_annotation(row)
                issues = validate_annotation(mapping, annotation)
                if issues:
                    issue_handle.write(
                        _jsonl(
                            {
                                "custom_id": custom_id,
                                "line_number": line_number,
                                "issues": issues,
                            }
                        )
                    )
                    issue_count += 1
                    continue

                annotation_handle.write(_jsonl(annotation))
                ok_count += 1
                if _needs_review(annotation, review_confidence_threshold):
                    review_handle.write(
                        _jsonl(
                            {
                                "custom_id": custom_id,
                                "annotation": annotation,
                                "review_reasons": {
                                    "has_quality_flags": bool(
                                        annotation.get("quality_flags")
                                    ),
                                    "confidence_threshold": review_confidence_threshold,
                                },
                            }
                        )
                    )
                    review_count += 1
            except Exception as exc:
                failure_handle.write(
                    _jsonl(
                        {
                            "custom_id": custom_id,
                            "line_number": line_number,
                            "exception_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                )
                failure_count += 1

        for custom_id in sorted(set(mappings) - seen_custom_ids):
            failure_handle.write(
                _jsonl(
                    {
                        "custom_id": custom_id,
                        "line_number": None,
                        "exception_type": "MissingBatchResult",
                        "error": f"Missing batch result for custom_id {custom_id!r}",
                    }
                )
            )
            failure_count += 1

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "windows_path": str(windows_path),
        "batch_output_path": str(batch_output_path),
        "out_dir": str(out_dir),
        "ok_count": ok_count,
        "issue_count": issue_count,
        "failure_count": failure_count,
        "review_count": review_count,
    }
    write_json(out_dir / "labeling_run_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest OpenAI Batch results for teacher annotation."
    )
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--batch-output", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--review-confidence-threshold", type=float, default=0.75)
    args = parser.parse_args(argv)

    summary = ingest_batch_results(
        windows_path=args.windows,
        batch_output_path=args.batch_output,
        out_dir=args.out_dir,
        review_confidence_threshold=args.review_confidence_threshold,
    )
    print(summary["out_dir"])
    print(
        f"ok_count={summary['ok_count']} "
        f"issue_count={summary['issue_count']} "
        f"failure_count={summary['failure_count']} "
        f"review_count={summary['review_count']}"
    )
    return 0 if summary["failure_count"] == 0 and summary["issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
