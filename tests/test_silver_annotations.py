from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.silver_annotations import build_silver_annotations
from sermon_pipeline.teacher_ingest import validate_annotation


def _window() -> dict[str, object]:
    return {
        "custom_id": "teacher:doc-a:w0000:0-4",
        "document_id": "doc-a",
        "source_path": "datas/doc-a.json",
        "source_type": "datalab_parsed_json",
        "document_kind": "book_chapter",
        "target_local_sids": ["S1", "S2", "S3", "S4"],
        "local_to_source": {
            "S1": {
                "source_sentence_id": "doc-a.s0000",
                "global_sentence_index": 0,
                "role": "target",
                "text": "기도하겠습니다.",
            },
            "S2": {
                "source_sentence_id": "doc-a.s0001",
                "global_sentence_index": 1,
                "role": "target",
                "text": "하나님 아버지, 은혜를 주옵소서.",
            },
            "S3": {
                "source_sentence_id": "doc-a.s0002",
                "global_sentence_index": 2,
                "role": "target",
                "text": "Chapter 9",
            },
            "S4": {
                "source_sentence_id": "doc-a.s0003",
                "global_sentence_index": 3,
                "role": "target",
                "text": "본문 설명입니다.",
            },
            "S5": {
                "source_sentence_id": "doc-a.s0004",
                "global_sentence_index": 4,
                "role": "context",
                "text": "문맥 문장입니다.",
            },
        },
    }


class SilverAnnotationTests(unittest.TestCase):
    def test_build_silver_annotations_writes_valid_teacher_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            windows_path = root / "windows.jsonl"
            out_dir = root / "silver"
            windows_path.write_text(
                json.dumps(_window(), ensure_ascii=False) + "\n", encoding="utf-8"
            )

            summary = build_silver_annotations(
                windows_path=windows_path,
                out_dir=out_dir,
                limit=1,
            )

            rows = [
                json.loads(line)
                for line in (out_dir / "teacher_annotations.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(summary["annotation_count"], 1)
        self.assertEqual(len(rows), 1)
        annotation = rows[0]
        self.assertEqual(annotation["custom_id"], "teacher:doc-a:w0000:0-4")
        self.assertEqual(
            [item["local_sid"] for item in annotation["boundary_annotations"]],
            ["S1", "S2", "S3", "S4"],
        )
        self.assertNotIn("S5", {item["local_sid"] for item in annotation["boundary_annotations"]})
        self.assertFalse(validate_annotation(_window(), annotation))
        split_labels = {
            item["local_sid"]: item["boundary_type"]
            for item in annotation["boundary_annotations"]
            if item["split_after"]
        }
        self.assertEqual(split_labels["S1"], "prayer_or_closing")
        self.assertEqual(split_labels["S3"], "topic_shift")


if __name__ == "__main__":
    unittest.main()
