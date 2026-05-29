import unittest

from sermon_pipeline.constants import BOUNDARY_TYPES, SYSTEM_PROMPT
from sermon_pipeline.models import PreparedDocument, SourceBlock, SentenceUnit


class ModelTests(unittest.TestCase):
    def test_boundary_types_are_stable(self):
        self.assertEqual(
            BOUNDARY_TYPES,
            [
                "none",
                "topic_shift",
                "scripture_reading_start",
                "scripture_explanation_start",
                "illustration_start",
                "application_start",
                "prayer_or_closing",
                "enumeration_start",
            ],
        )

    def test_system_prompt_forbids_rewriting(self):
        self.assertIn("Do not rewrite", SYSTEM_PROMPT)
        self.assertIn("sentence_id", SYSTEM_PROMPT)

    def test_prepared_document_to_teacher_task(self):
        block = SourceBlock(
            block_id="docx.b0000",
            text="제10장 세마포",
            block_type="paragraph",
            source_tag="docx_paragraph",
            paragraph_index=0,
        )
        sentence = SentenceUnit(
            sentence_id="doc.s0000",
            text="제10장 세마포",
            block_id="docx.b0000",
            block_type="paragraph",
            source_tag="docx_paragraph",
            paragraph_index=0,
        )
        document = PreparedDocument(
            document_id="docx_high",
            source_type="docx",
            document_kind="sermon",
            source_path="datas/docx/세마포__설교 10장.docx",
            reasoning_effort="high",
            effort_label="high",
            extraction_notes=["DOCX parsed by paragraph."],
            removed_foreign_paragraphs=81,
            blocks=[block],
            sentences=[sentence],
        )

        task = document.to_teacher_task(["Return one item per sentence."])

        self.assertEqual(task["document_id"], "docx_high")
        self.assertEqual(task["source_type"], "docx")
        self.assertEqual(task["allowed_boundary_types"], BOUNDARY_TYPES)
        self.assertEqual(task["instructions"], ["Return one item per sentence."])
        self.assertEqual(task["sentences"][0]["sentence_id"], "doc.s0000")


if __name__ == "__main__":
    unittest.main()
