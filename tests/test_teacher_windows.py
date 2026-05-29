from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.teacher_windows import (
    build_teacher_windows,
    load_sentence_records,
)


def _row(document_id: str, index: int, text: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "sentence_id": f"{document_id}.s{index:04d}",
        "sentence_index": index,
        "text": text,
        "source_path": f"datas/{document_id}.json",
        "document_source_type": "datalab_parsed_json",
        "document_kind": "book_chapter",
        "block_type": "paragraph",
        "source_tag": "p",
        "page_id": "1",
        "paragraph_index": index // 2,
        "heading_context": ["heading"],
        "html_boundary_before": index % 2 == 0,
    }


def load_sentence_records_from_rows(rows: list[dict[str, object]]):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "sentences.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return load_sentence_records(path)


class TeacherWindowTests(unittest.TestCase):
    def test_load_sentence_records_normalizes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sentences.jsonl"
            rows = [_row("doc-a", 0, "첫 문장."), _row("doc-a", 1, "둘째 문장.")]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            records = load_sentence_records(path)

        self.assertEqual(
            [record.sentence_id for record in records], ["doc-a.s0000", "doc-a.s0001"]
        )
        self.assertEqual(records[0].source_type, "datalab_parsed_json")
        self.assertEqual(records[0].heading_context, ("heading",))

    def test_build_teacher_windows_uses_target_and_context(self) -> None:
        records = [_row("doc-a", index, f"{index}번 문장.") for index in range(6)]

        windows = build_teacher_windows(
            load_sentence_records_from_rows(records),
            target_size=3,
            left_context=1,
            right_context=1,
        )

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].custom_id, "teacher:doc-a:w0000:0-3")
        self.assertEqual(
            [item.local_sid for item in windows[0].sentences], ["S1", "S2", "S3", "S4"]
        )
        self.assertEqual(windows[0].target_local_sids, ("S1", "S2", "S3"))
        self.assertEqual(windows[1].custom_id, "teacher:doc-a:w0001:3-6")
        self.assertEqual(
            [item.local_sid for item in windows[1].sentences], ["S1", "S2", "S3", "S4"]
        )
        self.assertEqual(windows[1].target_local_sids, ("S2", "S3"))
        self.assertEqual(windows[1].sentences[1].source_sentence_id, "doc-a.s0003")

    def test_window_mapping_excludes_student_input_metadata(self) -> None:
        records = [_row("doc-a", index, f"{index}번 문장.") for index in range(4)]
        window = build_teacher_windows(load_sentence_records_from_rows(records), target_size=4)[
            0
        ]

        mapping = window.to_mapping()
        task = window.to_teacher_task()

        self.assertIn("local_to_source", mapping)
        self.assertEqual(
            mapping["local_to_source"]["S1"]["source_sentence_id"], "doc-a.s0000"
        )
        self.assertEqual(task["sentences"][0]["local_sid"], "S1")
        self.assertEqual(task["sentences"][0]["text"], "0번 문장.")
        self.assertIn("hints", task["sentences"][0])


if __name__ == "__main__":
    unittest.main()
