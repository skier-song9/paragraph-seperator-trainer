from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.gradio_demo import (
    build_browser_state,
    build_example_detail,
    load_model,
    render_metrics_html,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _example(output: str = "S2 topic_shift") -> dict[str, object]:
    return {
        "example_id": "sparse:teacher:doc-a:w0000:0-3",
        "document_id": "doc-a",
        "target_type": "sparse_multi_boundary",
        "input": (
            "<TASK>\n"
            "Identify sermon paragraph boundaries after the listed sentences.\n"
            "\n"
            "<TEXT>\n"
            "<S1> 첫 문장입니다.\n"
            "<S2> 둘째 문장입니다.\n"
            "<S3> 셋째 문장입니다."
        ),
        "output": output,
    }


def _write_dataset(root: Path) -> Path:
    dataset_dir = root / "student_sft"
    row = _example()
    _write_jsonl(dataset_dir / "sparse_multi_boundary" / "train.jsonl", [row])
    _write_jsonl(dataset_dir / "sparse_multi_boundary" / "validation.jsonl", [])
    _write_jsonl(dataset_dir / "sparse_multi_boundary" / "test.jsonl", [])
    _write_jsonl(dataset_dir / "first_boundary" / "train.jsonl", [])
    _write_jsonl(dataset_dir / "first_boundary" / "validation.jsonl", [])
    _write_jsonl(dataset_dir / "first_boundary" / "test.jsonl", [])
    _write_jsonl(
        dataset_dir / "mappings.jsonl",
        [
            {
                "sparse_example_id": row["example_id"],
                "first_boundary_example_id": "first:teacher:doc-a:w0000:0-3",
                "custom_id": "teacher:doc-a:w0000:0-3",
                "document_id": "doc-a",
                "source_path": "datas/doc-a.json",
                "local_to_source": {
                    "S1": {"global_sentence_index": 0, "role": "target"},
                    "S2": {"global_sentence_index": 1, "role": "target"},
                    "S3": {"global_sentence_index": 2, "role": "target"},
                },
            }
        ],
    )
    return dataset_dir


def _write_model(train_dir: Path) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "model_type": "linear_boundary_classifier",
        "labels": ["none", "topic_shift"],
        "weights": {
            "none": {},
            "topic_shift": {"bias": 3.0},
        },
        "metadata": {"family": "sparse_multi_boundary"},
    }
    (train_dir / "model.json").write_text(
        json.dumps(model, ensure_ascii=False), encoding="utf-8"
    )
    _write_jsonl(
        train_dir / "metrics.jsonl",
        [
            {
                "epoch": 1,
                "train/loss": 0.5,
                "validation/loss": 0.4,
                "validation/f1": 0.7,
                "validation/tolerance1_f1": 0.8,
            }
        ],
    )
    (train_dir / "train_run_summary.json").write_text(
        json.dumps(
            {
                "training": {
                    "final_validation": {
                        "validation/f1": 0.7,
                        "validation/tolerance1_f1": 0.8,
                    },
                    "final_test": {
                        "test/f1": 0.6,
                        "test/tolerance1_f1": 0.9,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


class GradioDemoTests(unittest.TestCase):
    def test_build_browser_state_filters_and_summarizes_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = _write_dataset(Path(temp_dir))

            state = build_browser_state(
                dataset_dir=dataset_dir,
                family="sparse_multi_boundary",
                split="train",
                query="둘째",
                boundary_filter="boundary_only",
                label_filter="topic_shift",
            )

        self.assertEqual(len(state["examples"]), 1)
        self.assertEqual(state["overview_rows"][0][3], 1)
        self.assertIn("topic_shift", state["summary_html"])

    def test_build_example_detail_parses_gold_and_model_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = _write_dataset(root)
            train_dir = root / "train"
            _write_model(train_dir)
            state = build_browser_state(
                dataset_dir=dataset_dir,
                family="sparse_multi_boundary",
                split="train",
            )

            detail = build_example_detail(
                state,
                state["choice_labels"][0],
                train_dir,
            )

        self.assertEqual(detail["metadata"]["custom_id"], "teacher:doc-a:w0000:0-3")
        self.assertEqual(detail["boundary_rows"], [["S2", 2, "topic_shift"]])
        self.assertEqual(len(detail["sentence_rows"]), 3)
        self.assertEqual(detail["sentence_rows"][0][4], "topic_shift")
        self.assertIn("Mismatches", detail["model_summary_html"])

    def test_load_model_and_metrics_html_tolerate_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dir = Path(temp_dir) / "train"
            _write_model(train_dir)

            model, status = load_model(train_dir)
            metrics_html = render_metrics_html(train_dir)

        self.assertIsNotNone(model)
        self.assertIn("loaded model", status)
        self.assertIn("Validation F1", metrics_html)
        self.assertIn("Boundary F1 by epoch", metrics_html)


if __name__ == "__main__":
    unittest.main()
