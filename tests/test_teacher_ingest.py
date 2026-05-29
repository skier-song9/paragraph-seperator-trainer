from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.teacher_ingest import ingest_batch_results, validate_annotation


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
            },
            "S2": {
                "source_sentence_id": "doc-a.s0001",
                "global_sentence_index": 1,
                "role": "target",
            },
            "S3": {
                "source_sentence_id": "doc-a.s0002",
                "global_sentence_index": 2,
                "role": "target",
            },
            "S4": {
                "source_sentence_id": "doc-a.s0003",
                "global_sentence_index": 3,
                "role": "context",
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
                "rationale": "same unit",
            },
            {
                "local_sid": "S2",
                "source_sentence_id": "doc-a.s0001",
                "split_after": True,
                "boundary_type": "topic_shift",
                "confidence": 0.8,
                "rationale": "topic changes",
            },
            {
                "local_sid": "S3",
                "source_sentence_id": "doc-a.s0002",
                "split_after": False,
                "boundary_type": "none",
                "confidence": 0.7,
                "rationale": "continues",
            },
        ],
        "quality_flags": [],
    }


class TeacherIngestTests(unittest.TestCase):
    def test_validate_annotation_accepts_valid_output(self) -> None:
        issues = validate_annotation(_mapping(), _annotation())
        self.assertEqual(issues, [])

    def test_validate_annotation_rejects_missing_target_and_bad_label(self) -> None:
        annotation = _annotation()
        annotation["boundary_annotations"] = annotation["boundary_annotations"][:2]
        annotation["boundary_annotations"][1]["boundary_type"] = "bad_label"

        issues = validate_annotation(_mapping(), annotation)
        codes = [issue["code"] for issue in issues]

        self.assertIn("missing_target_annotation", codes)
        self.assertIn("unknown_boundary_type", codes)

    def test_ingest_batch_results_writes_annotations_review_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            windows_path = root / "windows.jsonl"
            batch_output_path = root / "batch_output.jsonl"
            out_dir = root / "ingested"
            windows_path.write_text(
                json.dumps(_mapping(), ensure_ascii=False) + "\n", encoding="utf-8"
            )
            response_body = {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(_annotation(), ensure_ascii=False),
                            }
                        ]
                    }
                ]
            }
            batch_output_path.write_text(
                json.dumps(
                    {
                        "custom_id": "teacher:doc-a:w0000:0-3",
                        "response": {"status_code": 200, "body": response_body},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = ingest_batch_results(
                windows_path=windows_path,
                batch_output_path=batch_output_path,
                out_dir=out_dir,
                review_confidence_threshold=0.75,
            )

            annotations = (out_dir / "teacher_annotations.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            review = (out_dir / "needs_human_review.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(len(review), 1)


if __name__ == "__main__":
    unittest.main()
