from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.student_sft import (
    build_sft_datasets,
    first_boundary_output,
    render_student_input,
    sparse_boundary_output,
)


def _mapping() -> dict[str, object]:
    return {
        "custom_id": "teacher:doc-a:w0000:0-3",
        "document_id": "doc-a",
        "source_path": "datas/doc-a.json",
        "source_type": "datalab_parsed_json",
        "document_kind": "book_chapter",
        "target_local_sids": ["S1", "S2", "S3"],
        "local_to_source": {
            "S1": {
                "source_sentence_id": "doc-a.s0000",
                "global_sentence_index": 0,
                "role": "target",
                "text": "첫 문장.",
            },
            "S2": {
                "source_sentence_id": "doc-a.s0001",
                "global_sentence_index": 1,
                "role": "target",
                "text": "둘째 문장.",
            },
            "S3": {
                "source_sentence_id": "doc-a.s0002",
                "global_sentence_index": 2,
                "role": "target",
                "text": "셋째 문장.",
            },
        },
    }


def _annotation() -> dict[str, object]:
    return {
        "custom_id": "teacher:doc-a:w0000:0-3",
        "boundary_annotations": [
            {
                "local_sid": "S1",
                "source_sentence_id": "doc-a.s0000",
                "split_after": False,
                "boundary_type": "none",
                "confidence": 0.9,
                "rationale": "",
            },
            {
                "local_sid": "S2",
                "source_sentence_id": "doc-a.s0001",
                "split_after": True,
                "boundary_type": "topic_shift",
                "confidence": 0.8,
                "rationale": "",
            },
            {
                "local_sid": "S3",
                "source_sentence_id": "doc-a.s0002",
                "split_after": True,
                "boundary_type": "application_start",
                "confidence": 0.8,
                "rationale": "",
            },
        ],
        "quality_flags": [],
    }


class StudentSftTests(unittest.TestCase):
    def test_render_student_input_contains_only_task_and_text(self) -> None:
        text = render_student_input([("S1", "첫 문장."), ("S2", "둘째 문장.")])

        self.assertIn("<TASK>", text)
        self.assertIn("<TEXT>", text)
        self.assertIn("<S1> 첫 문장.", text)
        self.assertNotIn("source_sentence_id", text)
        self.assertNotIn("document_id", text)

    def test_sparse_and_first_boundary_outputs(self) -> None:
        annotation = _annotation()

        self.assertEqual(
            sparse_boundary_output(annotation),
            "S2 topic_shift\nS3 application_start",
        )
        self.assertEqual(first_boundary_output(annotation), "S2 topic_shift")

    def test_no_boundary_outputs(self) -> None:
        annotation = _annotation()
        for item in annotation["boundary_annotations"]:
            item["split_after"] = False
            item["boundary_type"] = "none"

        self.assertEqual(sparse_boundary_output(annotation), "NO_BOUNDARY")
        self.assertEqual(first_boundary_output(annotation), "NO_BOUNDARY")

    def test_build_sft_datasets_writes_two_dataset_families(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotations_path = root / "teacher_annotations.jsonl"
            windows_path = root / "windows.jsonl"
            out_dir = root / "sft"
            annotations_path.write_text(
                json.dumps(_annotation(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            windows_path.write_text(
                json.dumps(_mapping(), ensure_ascii=False) + "\n", encoding="utf-8"
            )

            summary = build_sft_datasets(
                annotations_path=annotations_path,
                windows_path=windows_path,
                out_dir=out_dir,
            )

            sparse_files = list((out_dir / "sparse_multi_boundary").glob("*.jsonl"))
            first_files = list((out_dir / "first_boundary").glob("*.jsonl"))
            mapping_lines = (out_dir / "mappings.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(summary["example_count"], 1)
        self.assertEqual(len(sparse_files), 3)
        self.assertEqual(len(first_files), 3)
        self.assertEqual(len(mapping_lines), 1)


if __name__ == "__main__":
    unittest.main()
