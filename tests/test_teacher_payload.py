import json
import unittest

from sermon_pipeline.constants import BOUNDARY_TYPES, SYSTEM_PROMPT
from sermon_pipeline.models import PreparedDocument, SentenceUnit, SourceBlock
from sermon_pipeline.teacher import build_payload, response_schema, source_instructions


class TeacherPayloadTests(unittest.TestCase):
    def make_document(self, source_type):
        block = SourceBlock(
            block_id=f"{source_type}.b0000",
            text="제10장 세마포",
            block_type="paragraph",
            source_tag=f"{source_type}_paragraph",
            paragraph_index=0,
        )
        sentence = SentenceUnit(
            sentence_id="doc.s0000",
            text="제10장 세마포",
            block_id=block.block_id,
            block_type=block.block_type,
            source_tag=block.source_tag,
            paragraph_index=0,
        )
        return PreparedDocument(
            document_id="sample",
            source_type=source_type,
            document_kind="sermon",
            source_path=f"datas/{source_type}/sample",
            reasoning_effort="high",
            effort_label="high",
            extraction_notes=["parsed"],
            removed_foreign_paragraphs=0,
            blocks=[block],
            sentences=[sentence],
        )

    def test_source_instructions_include_common_and_docx_delta(self):
        instructions = source_instructions("docx")
        self.assertIn(
            "Return one boundary_annotations item for every provided sentence.",
            instructions,
        )
        self.assertIn(
            "Do not infer boundaries from removed foreign paragraphs.",
            instructions,
        )

    def test_response_schema_is_strict_and_uses_boundary_taxonomy(self):
        schema = response_schema()
        boundary_enum = schema["properties"]["boundary_annotations"]["items"][
            "properties"
        ]["boundary_type"]["enum"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(boundary_enum, list(BOUNDARY_TYPES))

    def test_build_payload_serializes_teacher_task_as_json_content(self):
        document = self.make_document("datalab_parsed_json")
        payload = build_payload(document, model="gpt-5.5", max_output_tokens=8192)
        user_content = payload["input"][1]["content"]
        task = json.loads(user_content)

        self.assertEqual(payload["input"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(task["task"], "annotate_sentence_boundaries")
        self.assertIn(
            "HTML tag and heading_context are boundary hints, not gold labels.",
            task["instructions"],
        )
        self.assertEqual(
            payload["text"]["format"]["schema"]["required"][0],
            "document_id",
        )


if __name__ == "__main__":
    unittest.main()
