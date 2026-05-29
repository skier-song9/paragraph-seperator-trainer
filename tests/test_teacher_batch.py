from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.teacher_batch import build_batch_requests
from tests.test_teacher_windows import _row


class TeacherBatchTests(unittest.TestCase):
    def test_build_batch_requests_writes_jsonl_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sentences_path = root / "sentences.jsonl"
            out_dir = root / "out"
            rows = [_row("doc-a", index, f"{index}번 문장.") for index in range(5)]
            sentences_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = build_batch_requests(
                sentences_path=sentences_path,
                out_dir=out_dir,
                model="gpt-5.5",
                target_size=3,
                left_context=1,
                right_context=1,
                max_output_tokens=4096,
                limit_windows=None,
            )

            request_lines = (out_dir / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines()
            mapping_lines = (out_dir / "windows.jsonl").read_text(encoding="utf-8").splitlines()
            first_request = json.loads(request_lines[0])

        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(len(request_lines), 2)
        self.assertEqual(len(mapping_lines), 2)
        self.assertEqual(first_request["method"], "POST")
        self.assertEqual(first_request["url"], "/v1/responses")
        self.assertEqual(first_request["body"]["model"], "gpt-5.5")
        self.assertTrue(first_request["custom_id"].startswith("teacher:doc-a:w0000"))

    def test_limit_windows_caps_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sentences_path = root / "sentences.jsonl"
            out_dir = root / "out"
            rows = [_row("doc-a", index, f"{index}번 문장.") for index in range(10)]
            sentences_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = build_batch_requests(
                sentences_path=sentences_path,
                out_dir=out_dir,
                target_size=2,
                limit_windows=1,
            )

            request_lines = (out_dir / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["window_count"], 1)
        self.assertEqual(len(request_lines), 1)


if __name__ == "__main__":
    unittest.main()
