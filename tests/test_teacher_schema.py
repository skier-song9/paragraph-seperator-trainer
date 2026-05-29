from __future__ import annotations

import unittest

from sermon_pipeline.constants import BOUNDARY_TYPES
import sermon_pipeline.teacher_schema as teacher_schema
from sermon_pipeline.teacher_schema import build_teacher_payload, teacher_response_schema
from sermon_pipeline.teacher_windows import build_teacher_windows
from tests.test_teacher_windows import load_sentence_records_from_rows, _row


class TeacherSchemaTests(unittest.TestCase):
    def test_module_exports_only_specified_public_names(self) -> None:
        public_names = {
            name for name in dir(teacher_schema) if name.startswith("TEACHER")
        }

        self.assertEqual(public_names, {"TEACHER_SYSTEM_PROMPT"})

    def test_response_schema_uses_boundary_type_enum(self) -> None:
        schema = teacher_response_schema()
        annotation = schema["properties"]["boundary_annotations"]["items"]

        self.assertEqual(
            annotation["properties"]["boundary_type"]["enum"], list(BOUNDARY_TYPES)
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(annotation["additionalProperties"])
        self.assertEqual(
            schema["required"],
            ["custom_id", "boundary_annotations", "quality_flags"],
        )
        self.assertEqual(
            annotation["required"],
            [
                "local_sid",
                "source_sentence_id",
                "split_after",
                "boundary_type",
                "confidence",
                "rationale",
            ],
        )
        self.assertEqual(annotation["properties"]["confidence"]["minimum"], 0)
        self.assertEqual(annotation["properties"]["confidence"]["maximum"], 1)
        self.assertEqual(
            schema["properties"]["quality_flags"]["items"]["type"], "string"
        )

    def test_payload_uses_responses_api_structured_output(self) -> None:
        records = load_sentence_records_from_rows(
            [_row("doc-a", index, f"{index}번 문장.") for index in range(4)]
        )
        window = build_teacher_windows(records, target_size=4)[0]

        payload = build_teacher_payload(
            window, model="gpt-5.5", max_output_tokens=4096
        )

        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertEqual(payload["input"][0]["role"], "system")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertIn("target_local_sids", payload["input"][1]["content"])
        self.assertIn("<S1>", payload["input"][1]["content"])


if __name__ == "__main__":
    unittest.main()
