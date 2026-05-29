import unittest

from sermon_pipeline.constants import BOUNDARY_TYPES, SYSTEM_PROMPT
from sermon_pipeline.models import (
    PreparedDocument,
    SentenceUnit,
    SourceBlock,
    TrainingRow,
    ValidationIssue,
)


class ModelTests(unittest.TestCase):
    def test_boundary_types_are_stable(self):
        self.assertEqual(
            list(BOUNDARY_TYPES),
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

    def test_boundary_types_are_immutable(self):
        self.assertIsInstance(BOUNDARY_TYPES, tuple)
        with self.assertRaises(AttributeError):
            BOUNDARY_TYPES.append("new_boundary")

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
        self.assertEqual(task["allowed_boundary_types"], list(BOUNDARY_TYPES))
        self.assertIsInstance(task["allowed_boundary_types"], list)
        self.assertEqual(task["instructions"], ["Return one item per sentence."])
        self.assertEqual(task["sentences"][0]["sentence_id"], "doc.s0000")

    def test_mutating_constructor_inputs_does_not_change_model_state(self):
        heading_context = ["Intro"]
        script_counts = {"hangul": 3}
        extraction_notes = ["DOCX parsed by paragraph."]
        features = {"same_block": True, "scores": [1, 2]}

        block = SourceBlock(
            block_id="docx.b0000",
            text="제10장 세마포",
            block_type="paragraph",
            source_tag="docx_paragraph",
            heading_context=heading_context,
            script_counts=script_counts,
        )
        sentence = SentenceUnit(
            sentence_id="doc.s0000",
            text="제10장 세마포",
            block_id="docx.b0000",
            heading_context=heading_context,
        )
        blocks = [block]
        sentences = [sentence]
        document = PreparedDocument(
            document_id="docx_high",
            source_type="docx",
            document_kind="sermon",
            source_path="datas/docx/세마포__설교 10장.docx",
            reasoning_effort="high",
            effort_label="high",
            extraction_notes=extraction_notes,
            removed_foreign_paragraphs=81,
            blocks=blocks,
            sentences=sentences,
        )
        row = TrainingRow(
            left_sentence_id="doc.s0000",
            right_sentence_id=None,
            split_after_left=True,
            boundary_type="topic_shift",
            teacher_confidence=0.9,
            source_type="docx",
            document_kind="sermon",
            features=features,
            review_status="accepted",
        )

        heading_context.append("Body")
        script_counts["latin"] = 1
        extraction_notes.append("Changed")
        blocks.clear()
        sentences.clear()
        features["same_block"] = False
        features["scores"].append(3)

        self.assertEqual(block.heading_context, ("Intro",))
        self.assertEqual(dict(block.script_counts), {"hangul": 3})
        self.assertEqual(sentence.heading_context, ("Intro",))
        self.assertEqual(document.extraction_notes, ("DOCX parsed by paragraph.",))
        self.assertEqual(document.blocks, (block,))
        self.assertEqual(document.sentences, (sentence,))
        self.assertEqual(
            row.features,
            {"same_block": True, "scores": (1, 2)},
        )

    def test_mutating_returned_payload_does_not_change_model_state(self):
        block = SourceBlock(
            block_id="docx.b0000",
            text="제10장 세마포",
            block_type="paragraph",
            source_tag="docx_paragraph",
            heading_context=["Intro"],
            script_counts={"hangul": 3},
        )
        sentence = SentenceUnit(
            sentence_id="doc.s0000",
            text="제10장 세마포",
            block_id="docx.b0000",
            heading_context=["Intro"],
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
        row = TrainingRow(
            left_sentence_id="doc.s0000",
            right_sentence_id=None,
            split_after_left=True,
            boundary_type="topic_shift",
            teacher_confidence=0.9,
            source_type="docx",
            document_kind="sermon",
            features={"scores": [1, 2]},
            review_status="accepted",
        )

        block_payload = block.to_payload()
        sentence_payload = sentence.to_payload()
        task_payload = document.to_teacher_task(["Return one item per sentence."])
        row_payload = row.to_payload()
        block_payload["heading_context"].append("Body")
        block_payload["script_counts"]["latin"] = 1
        sentence_payload["heading_context"].append("Body")
        task_payload["extraction_notes"].append("Changed")
        task_payload["sentences"][0]["heading_context"].append("Body")
        row_payload["features"]["scores"].append(3)

        self.assertEqual(block.heading_context, ("Intro",))
        self.assertEqual(dict(block.script_counts), {"hangul": 3})
        self.assertEqual(sentence.heading_context, ("Intro",))
        self.assertEqual(document.extraction_notes, ("DOCX parsed by paragraph.",))
        self.assertEqual(document.sentences[0].heading_context, ("Intro",))
        self.assertEqual(row.features["scores"], (1, 2))

    def test_source_block_payload_omits_none_fields(self):
        block = SourceBlock(
            block_id="docx.b0000",
            text="제10장 세마포",
            block_type="paragraph",
            source_tag="docx_paragraph",
        )

        payload = block.to_payload()

        self.assertNotIn("page_id", payload)
        self.assertNotIn("paragraph_index", payload)
        self.assertEqual(payload["heading_context"], [])
        self.assertEqual(payload["script_counts"], {})

    def test_validation_issue_to_payload_returns_json_compatible_dict(self):
        issue = ValidationIssue(
            code="missing_sentence",
            message="Sentence was not annotated.",
            sentence_id="doc.s0000",
        )

        self.assertEqual(
            issue.to_payload(),
            {
                "code": "missing_sentence",
                "message": "Sentence was not annotated.",
                "sentence_id": "doc.s0000",
            },
        )

    def test_training_row_to_payload_returns_json_compatible_dict(self):
        row = TrainingRow(
            left_sentence_id="doc.s0000",
            right_sentence_id="doc.s0001",
            split_after_left=True,
            boundary_type="topic_shift",
            teacher_confidence=0.9,
            source_type="docx",
            document_kind="sermon",
            features={"scores": [1, 2], "metadata": {"same_block": True}},
            review_status="accepted",
        )

        payload = row.to_payload()

        self.assertEqual(
            payload,
            {
                "left_sentence_id": "doc.s0000",
                "right_sentence_id": "doc.s0001",
                "split_after_left": True,
                "boundary_type": "topic_shift",
                "teacher_confidence": 0.9,
                "source_type": "docx",
                "document_kind": "sermon",
                "features": {
                    "scores": [1, 2],
                    "metadata": {"same_block": True},
                },
                "review_status": "accepted",
            },
        )
        self.assertIsInstance(payload["features"], dict)
        self.assertIsInstance(payload["features"]["scores"], list)
        self.assertIsInstance(payload["features"]["metadata"], dict)


if __name__ == "__main__":
    unittest.main()
