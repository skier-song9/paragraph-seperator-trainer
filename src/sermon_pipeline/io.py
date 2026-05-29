from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    texts: list[str] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("type") in {"output_text", "text"} and isinstance(
                obj.get("text"), str
            ):
                texts.append(obj["text"])
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(response.get("output", []))
    return "\n".join(texts).strip()


def summarize_annotation(annotation: dict[str, Any] | None) -> dict[str, Any]:
    if not annotation:
        return {"split_count": None, "boundary_types": {}, "quality_flags": []}

    boundary_types: dict[str, int] = {}
    split_count = 0
    for item in annotation.get("boundary_annotations", []):
        boundary_type = item.get("boundary_type", "unknown")
        boundary_types[boundary_type] = boundary_types.get(boundary_type, 0) + 1
        if item.get("split_after"):
            split_count += 1

    return {
        "split_count": split_count,
        "boundary_types": boundary_types,
        "quality_flags": annotation.get("quality_flags", []),
    }


def render_comparison(rows: list[dict[str, Any]], model: str) -> str:
    lines = [
        "# OpenAI Preprocessing Comparison",
        "",
        f"- model: `{model}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "| case | source | effort | status | sentences | splits | boundary types | tokens | seconds |",
        "|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        boundary_types = row.get("boundary_types") or {}
        btxt = (
            ", ".join(
                f"{key}:{value}" for key, value in sorted(boundary_types.items())
            )
            or "-"
        )
        usage = row.get("usage") or {}
        tokens = usage.get("total_tokens") or usage.get("total") or "-"
        elapsed = row.get("elapsed_seconds")
        seconds = f"{elapsed:.1f}" if isinstance(elapsed, (int, float)) else "-"
        lines.append(
            "| {case_id} | {source_type}<br>`{source_path}` | {effort} | {status} | "
            "{sentences} | {splits} | {btypes} | {tokens} | {seconds} |".format(
                case_id=row["case_id"],
                source_type=row["source_type"],
                source_path=row["source_path"],
                effort=row["reasoning_effort"],
                status=row["status"],
                sentences=row["sentence_count"],
                splits=row.get("split_count", "-"),
                btypes=btxt.replace("|", "\\|"),
                tokens=tokens,
                seconds=seconds,
            )
        )

    lines.extend(["", "## Notes", ""])
    for row in rows:
        lines.append(f"### {row['case_id']}")
        lines.append("")
        if row.get("error"):
            lines.append(f"- error: `{row['error']}`")
        lines.append(f"- extraction notes: {'; '.join(row.get('extraction_notes', []))}")
        if row.get("removed_foreign_paragraphs"):
            lines.append(
                f"- removed_foreign_paragraphs: {row['removed_foreign_paragraphs']}"
            )
        for flag in row.get("quality_flags", []):
            lines.append(f"- quality flag: {flag}")
        lines.append("")
    return "\n".join(lines)
