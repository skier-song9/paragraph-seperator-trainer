import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from sermon_pipeline.extractors.datalab import parse_datalab_json
from sermon_pipeline.extractors.docx import extract_docx_paragraphs, parse_docx


class DatalabExtractorTests(unittest.TestCase):
    def test_parse_datalab_json_keeps_heading_page_and_boundary_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chapter.json"
            path.write_text(
                json.dumps(
                    {
                        "success": True,
                        "page_count": 2,
                        "html": (
                            '<div data-page-id="0">'
                            "<h2>인간의 기준은 인간의 양심이고 하나님의 기준은 하나님이 양심이다</h2>"
                            "<p>그러므로 율법의 행위로 그의 앞에 의롭다 하심을 얻을 육체가 없나니.</p>"
                            "</div>"
                            '<div data-page-id="1">'
                            "<h2>인간의 기준은 인간의 양심이고 하나님의 기준은 하나님이 양심이다</h2>"
                            "<p>오래간만에 여러분을 뵙게 되어서 정말 반갑습니다.</p>"
                            "</div>"
                        ),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            document = parse_datalab_json(
                path, root=Path(tmp), document_id="datalab_sample", max_sentences=8
            )

        self.assertEqual(document.source_type, "datalab_parsed_json")
        self.assertEqual(document.reasoning_effort, "xhigh")
        self.assertEqual(document.extraction_notes[-1], "page_count=2")
        self.assertEqual(document.blocks[0].source_tag, "h2")
        self.assertEqual(document.blocks[0].page_id, "0")
        self.assertTrue(document.blocks[0].html_boundary_before)
        self.assertEqual(document.sentences[0].sentence_id, "datalab_sample.s0000")
        self.assertEqual(
            list(document.sentences[1].heading_context),
            ["인간의 기준은 인간의 양심이고 하나님의 기준은 하나님이 양심이다"],
        )

    def test_parse_datalab_json_preserves_nested_kept_tag_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.json"
            path.write_text(
                json.dumps(
                    {
                        "success": True,
                        "page_count": 1,
                        "html": (
                            '<div data-page-id="0"><ul><li>바깥 설명 '
                            "<span>중간</span><li>안쪽 설명입니다.</li> 끝입니다."
                            "</li></ul></div>"
                        ),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            document = parse_datalab_json(
                path, root=Path(tmp), document_id="nested_sample", max_sentences=8
            )

        block_texts = [block.text for block in document.blocks]
        self.assertTrue(
            any("바깥 설명" in text and "끝입니다" in text for text in block_texts)
        )
        self.assertTrue(any("안쪽 설명입니다" in text for text in block_texts))
        self.assertEqual(document.sentences[0].sentence_id, "nested_sample.s0000")
        self.assertEqual(document.blocks[0].page_id, "0")
        self.assertTrue(document.blocks[0].html_boundary_before)


class DocxExtractorTests(unittest.TestCase):
    def test_extract_docx_paragraphs_reads_word_document_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.docx"
            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>제10장 세마포</w:t></w:r></w:p>
    <w:p><w:r><w:t>English translation paragraph should be removed.</w:t></w:r></w:p>
    <w:p><w:r><w:t>이제 성소가 완성되는 과정에서 남은 것은 세마포와 은받침입니다.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            paragraphs = extract_docx_paragraphs(path)
            document = parse_docx(
                path, root=Path(tmp), document_id="docx_sample", max_sentences=8
            )

        self.assertEqual(paragraphs[0], "제10장 세마포")
        self.assertEqual(document.source_type, "docx")
        self.assertEqual(document.removed_foreign_paragraphs, 1)
        self.assertEqual(len(document.blocks), 2)
        self.assertEqual(document.blocks[0].paragraph_index, 0)
        self.assertEqual(document.blocks[1].paragraph_index, 2)
        self.assertEqual(document.blocks[0].language_filter_reason, "hangul_present")
        self.assertEqual(document.sentences[0].sentence_id, "docx_sample.s0000")


if __name__ == "__main__":
    unittest.main()
