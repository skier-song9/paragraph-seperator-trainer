import unittest

from sermon_pipeline.models import SourceBlock
from sermon_pipeline.sentence_splitter import split_blocks_into_sentences


class SentenceSplitterTests(unittest.TestCase):
    def test_splits_sentence_punctuation_and_preserves_metadata(self):
        block = SourceBlock(
            block_id="html.b0003",
            text="오래간만에 여러분을 뵙게 되어서 정말 반갑습니다. 오늘부터 시작되는 이번 학기에는 율법과 복음을 공부합니다.",
            block_type="paragraph",
            source_tag="p",
            page_id="1",
            heading_context=["율법과 복음"],
            html_boundary_before=True,
        )

        sentences = split_blocks_into_sentences(
            [block], "datalab_sample", max_sentences=8
        )

        self.assertEqual(
            [s.sentence_id for s in sentences],
            ["datalab_sample.s0000", "datalab_sample.s0001"],
        )
        self.assertEqual(sentences[0].page_id, "1")
        self.assertEqual(list(sentences[0].heading_context), ["율법과 복음"])
        self.assertTrue(sentences[0].html_boundary_before)

    def test_long_korean_block_fallback_splits_on_sentence_endings(self):
        block = SourceBlock(
            block_id="hwp.b0001",
            text=("오늘 본문 말씀은 요한일서 2장 1절입니다 " * 25).strip(),
            block_type="paragraph",
            source_tag="hwp_paragraph",
            paragraph_index=1,
        )

        sentences = split_blocks_into_sentences([block], "hwp_sample", max_sentences=3)

        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0].sentence_id, "hwp_sample.s0000")
        self.assertEqual(sentences[2].sentence_id, "hwp_sample.s0002")


if __name__ == "__main__":
    unittest.main()
