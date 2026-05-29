from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.train_sft import (
    flatten_wandb_metrics,
    log_wandb_summary,
    summarize_sft_dataset,
    write_training_summary,
)


def _write_example(path: Path, output: str, target_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "example_id": f"{target_type}:doc-a",
        "document_id": "doc-a",
        "target_type": target_type,
        "input": "<TASK>\n<TEXT>\n<S1> 문장.",
        "output": output,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.finished = False

    def log(self, metrics: dict[str, object], step: int | None = None) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class _FakeWandb:
    def __init__(self) -> None:
        self.login_key: str | None = None
        self.init_kwargs: dict[str, object] | None = None
        self.run = _FakeRun()

    def login(self, key: str, relogin: bool = False) -> None:
        self.login_key = key

    def init(self, **kwargs: object) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run


class TrainSftTests(unittest.TestCase):
    def test_summarize_sft_dataset_counts_examples_and_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_example(
                dataset_dir / "sparse_multi_boundary" / "train.jsonl",
                "S1 topic_shift\nS3 prayer_or_closing",
                "sparse_multi_boundary",
            )
            _write_example(
                dataset_dir / "first_boundary" / "validation.jsonl",
                "NO_BOUNDARY",
                "first_boundary",
            )

            summary = summarize_sft_dataset(dataset_dir)

        self.assertEqual(summary["total_examples"], 2)
        self.assertEqual(summary["boundary_line_count"], 2)
        self.assertEqual(summary["no_boundary_count"], 1)
        self.assertEqual(
            summary["by_family_split"]["sparse_multi_boundary"]["train"], 1
        )
        self.assertEqual(summary["by_family_split"]["first_boundary"]["validation"], 1)

    def test_flatten_wandb_metrics_uses_numeric_values_only(self) -> None:
        summary = {
            "total_examples": 2,
            "boundary_line_count": 1,
            "by_family_split": {
                "sparse_multi_boundary": {"train": 2, "validation": 0, "test": 0}
            },
        }

        metrics = flatten_wandb_metrics(summary)

        self.assertEqual(metrics["dataset/total_examples"], 2)
        self.assertEqual(metrics["dataset/boundary_line_count"], 1)
        self.assertEqual(metrics["dataset/sparse_multi_boundary/train_examples"], 2)

    def test_log_wandb_summary_online_uses_env_key_without_returning_secret(self) -> None:
        fake = _FakeWandb()
        old_key = os.environ.get("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = "secret-key"
        try:
            status = log_wandb_summary(
                summary={"total_examples": 1},
                project="sermon-test",
                run_name="run-a",
                mode="online",
                backend=fake,
            )
        finally:
            if old_key is None:
                os.environ.pop("WANDB_API_KEY", None)
            else:
                os.environ["WANDB_API_KEY"] = old_key

        self.assertEqual(fake.login_key, "secret-key")
        self.assertEqual(fake.init_kwargs["project"], "sermon-test")
        self.assertEqual(fake.init_kwargs["name"], "run-a")
        self.assertEqual(fake.init_kwargs["mode"], "online")
        self.assertEqual(fake.run.logged[0][0]["dataset/total_examples"], 1)
        self.assertTrue(fake.run.finished)
        self.assertNotIn("secret-key", json.dumps(status))
        self.assertTrue(status["api_key_present"])

    def test_write_training_summary_logs_disabled_wandb_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            out_dir = root / "train"
            _write_example(
                dataset_dir / "sparse_multi_boundary" / "test.jsonl",
                "NO_BOUNDARY",
                "sparse_multi_boundary",
            )

            summary = write_training_summary(
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                wandb_mode="disabled",
                wandb_project="sermon-test",
                run_name="dry-run",
            )

            written = json.loads(
                (out_dir / "train_run_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["dataset"]["total_examples"], 1)
        self.assertEqual(written["wandb"]["mode"], "disabled")


if __name__ == "__main__":
    unittest.main()
