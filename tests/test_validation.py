import unittest

from sermon_pipeline.models import PreparedDocument, SentenceUnit, SourceBlock
from sermon_pipeline.validation import (
    teacher_output_to_training_rows,
    validate_teacher_output,
)


class ValidationTests(unittest.TestCase):
    def make_document(self):
        block = SourceBlock(
            block_id="docx.b0000",
            text="제10장 세마포",
            block_type="paragraph",
            source_tag="docx_paragraph",
            paragraph_index=0,
        )
        sentences = [
            SentenceUnit(
                "doc.s0000",
                "제10장 세마포",
                "docx.b0000",
                "paragraph",
                "docx_paragraph",
                paragraph_index=0,
            ),
            SentenceUnit(
                "doc.s0001",
                "성경 본문입니다.",
                "docx.b0001",
                "paragraph",
                "docx_paragraph",
                paragraph_index=1,
            ),
            SentenceUnit(
                "doc.s0002",
                "(계 19:7, 8)",
                "docx.b0001",
                "paragraph",
                "docx_paragraph",
                paragraph_index=1,
            ),
        ]
        return PreparedDocument(
            document_id="docx_high",
            source_type="docx",
            document_kind="sermon",
            source_path="datas/docx/sample.docx",
            reasoning_effort="high",
            effort_label="high",
            extraction_notes=["parsed"],
            removed_foreign_paragraphs=0,
            blocks=[block],
            sentences=sentences,
        )

    def test_validate_teacher_output_detects_coverage_and_boundary_consistency(self):
        document = self.make_document()
        output = {
            "document_id": "docx_high",
            "source_type": "docx",
            "reasoning_effort": "high",
            "preprocessing_observations": [],
            "boundary_annotations": [
                {
                    "sentence_id": "doc.s0000",
                    "text_excerpt": "제10장 세마포",
                    "split_after": False,
                    "boundary_type": "topic_shift",
                    "confidence": 0.9,
                    "rationale": "bad consistency",
                }
            ],
            "proposed_atomic_paragraphs": [],
            "quality_flags": [],
        }

        issues = validate_teacher_output(document, output)
        codes = [issue.code for issue in issues]

        self.assertIn("coverage_mismatch", codes)
        self.assertIn("split_false_boundary_type_not_none", codes)
        self.assertIn("missing_sentence_annotation", codes)

    def test_teacher_output_to_training_rows_maps_neighbor_features(self):
        document = self.make_document()
        output = {
            "document_id": "docx_high",
            "source_type": "docx",
            "reasoning_effort": "high",
            "preprocessing_observations": [],
            "boundary_annotations": [
                {
                    "sentence_id": "doc.s0000",
                    "text_excerpt": "제10장 세마포",
                    "split_after": True,
                    "boundary_type": "scripture_reading_start",
                    "confidence": 0.93,
                    "rationale": "heading then scripture",
                },
                {
                    "sentence_id": "doc.s0001",
                    "text_excerpt": "성경 본문입니다.",
                    "split_after": False,
                    "boundary_type": "none",
                    "confidence": 0.98,
                    "rationale": "keep reference",
                },
                {
                    "sentence_id": "doc.s0002",
                    "text_excerpt": "(계 19:7, 8)",
                    "split_after": False,
                    "boundary_type": "none",
                    "confidence": 0.72,
                    "rationale": "terminal",
                },
            ],
            "proposed_atomic_paragraphs": [],
            "quality_flags": [],
        }

        rows = teacher_output_to_training_rows(document, output)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].left_sentence_id, "doc.s0000")
        self.assertEqual(rows[0].right_sentence_id, "doc.s0001")
        self.assertTrue(rows[0].split_after_left)
        self.assertEqual(rows[0].review_status, "teacher_only")
        self.assertEqual(rows[2].review_status, "needs_review")
        self.assertEqual(rows[0].features["same_original_paragraph"], False)


if __name__ == "__main__":
    unittest.main()
