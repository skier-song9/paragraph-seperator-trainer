import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.extractors.datalab import parse_datalab_json


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


if __name__ == "__main__":
    unittest.main()
