from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import load_dotenv, write_json
from .student_sft import SPLITS

SFT_FAMILIES = ("sparse_multi_boundary", "first_boundary")
WANDB_MODES = ("disabled", "offline", "online")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _boundary_line_count(output_text: str) -> int:
    stripped = output_text.strip()
    if not stripped or stripped == "NO_BOUNDARY":
        return 0
    return len([line for line in stripped.splitlines() if line.strip()])


def summarize_sft_dataset(dataset_dir: Path) -> dict[str, Any]:
    by_family_split = {
        family: {split: 0 for split in SPLITS} for family in SFT_FAMILIES
    }
    by_target_type = {family: 0 for family in SFT_FAMILIES}
    total_examples = 0
    no_boundary_count = 0
    boundary_line_count = 0
    input_char_total = 0
    output_char_total = 0

    for family in SFT_FAMILIES:
        for split in SPLITS:
            path = dataset_dir / family / f"{split}.jsonl"
            rows = _iter_jsonl(path)
            by_family_split[family][split] = len(rows)
            by_target_type[family] += len(rows)
            for row in rows:
                output_text = row.get("output")
                input_text = row.get("input")
                if not isinstance(output_text, str) or not isinstance(input_text, str):
                    raise ValueError(f"{path}: examples must include string input/output")
                total_examples += 1
                input_char_total += len(input_text)
                output_char_total += len(output_text)
                lines = _boundary_line_count(output_text)
                boundary_line_count += lines
                if lines == 0:
                    no_boundary_count += 1

    avg_input_chars = input_char_total / total_examples if total_examples else 0.0
    avg_output_chars = output_char_total / total_examples if total_examples else 0.0
    return {
        "dataset_dir": str(dataset_dir),
        "total_examples": total_examples,
        "by_family_split": by_family_split,
        "by_target_type": by_target_type,
        "boundary_example_count": total_examples - no_boundary_count,
        "no_boundary_count": no_boundary_count,
        "boundary_line_count": boundary_line_count,
        "avg_input_chars": avg_input_chars,
        "avg_output_chars": avg_output_chars,
    }


def flatten_wandb_metrics(summary: dict[str, Any]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[f"dataset/{key}"] = value

    by_family_split = summary.get("by_family_split", {})
    if isinstance(by_family_split, dict):
        for family, split_counts in by_family_split.items():
            if not isinstance(family, str) or not isinstance(split_counts, dict):
                continue
            for split, count in split_counts.items():
                if isinstance(split, str) and isinstance(count, int):
                    metrics[f"dataset/{family}/{split}_examples"] = count
    return metrics


def log_wandb_summary(
    summary: dict[str, Any],
    project: str,
    run_name: str,
    mode: str,
    backend: Any | None = None,
) -> dict[str, Any]:
    if mode not in WANDB_MODES:
        raise ValueError(f"unknown W&B mode: {mode}")
    api_key = os.environ.get("WANDB_API_KEY")
    status = {
        "mode": mode,
        "project": project,
        "run_name": run_name,
        "api_key_present": bool(api_key),
    }
    if mode == "disabled":
        status["enabled"] = False
        return status

    if backend is None:
        try:
            import wandb as backend  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "wandb is not installed. Run with the training extra: "
                "uv run --extra train sermon-sft-train-smoke ..."
            ) from exc

    if mode == "online":
        if not api_key:
            raise RuntimeError("WANDB_API_KEY is required for online W&B logging")
        backend.login(key=api_key, relogin=False)

    run = backend.init(
        project=project,
        name=run_name,
        mode=mode,
        config={
            "dataset_dir": summary.get("dataset_dir"),
            "sft_families": list(SFT_FAMILIES),
            "splits": list(SPLITS),
        },
    )
    run.log(flatten_wandb_metrics(summary), step=0)
    run.finish()
    status["enabled"] = True
    return status


def write_training_summary(
    dataset_dir: Path,
    out_dir: Path,
    wandb_mode: str,
    wandb_project: str,
    run_name: str | None = None,
    env_file: Path | None = None,
) -> dict[str, Any]:
    if env_file is not None:
        load_dotenv(env_file)
    resolved_run_name = run_name or f"sft-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dataset_summary = summarize_sft_dataset(dataset_dir)
    wandb_status = log_wandb_summary(
        summary=dataset_summary,
        project=wandb_project,
        run_name=resolved_run_name,
        mode=wandb_mode,
    )
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_status": "dry_run_dataset_logging_only",
        "dataset": dataset_summary,
        "wandb": wandb_status,
    }
    write_json(out_dir / "train_run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default="offline")
    parser.add_argument("--wandb-project", default="sermon-boundary-sft")
    parser.add_argument("--run-name")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)

    summary = write_training_summary(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        env_file=args.env_file,
    )
    print(f"out_dir: {args.out_dir}")
    print(f"training_status: {summary['training_status']}")
    print(f"total_examples: {summary['dataset']['total_examples']}")
    print(f"wandb_mode: {summary['wandb']['mode']}")
    print(f"wandb_api_key_present: {summary['wandb']['api_key_present']}")
    return 0
