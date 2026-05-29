import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from sermon_pipeline.cli import run


class FakeChar:
    def __init__(self, code):
        self.kind = "char_code"
        self.code = code


class FakeParagraph:
    def __init__(self, text):
        self.chars = [FakeChar(ord(ch)) for ch in text]


class FakeSection:
    def __init__(self, paragraphs):
        self.paragraphs = [FakeParagraph(text) for text in paragraphs]


class FakeReader:
    version = "5.0.3.0"

    def __init__(self, path):
        self.path = path
        self.sections = [
            FakeSection(
                [
                    "20141113 목요찬양예배 파라클레토스입니다.",
                    "오늘 본문 말씀은 요한일서 2장 1절입니다.",
                ]
            )
        ]


class CliSmokeTests(unittest.TestCase):
    def test_run_dry_run_writes_payloads_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datalab_path = root / "datas" / "datalab_parsed" / "book" / "chapter.json"
            datalab_path.parent.mkdir(parents=True)
            datalab_path.write_text(
                json.dumps(
                    {
                        "success": True,
                        "page_count": 1,
                        "html": (
                            '<div data-page-id="0">'
                            "<h2>제10장 세마포</h2>"
                            "<p>오늘 본문은 출애굽기 말씀입니다. 하나님께서 길을 여십니다.</p>"
                            "</div>"
                        ),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            docx_path = root / "datas" / "docx" / "세마포__설교 10장.docx"
            docx_path.parent.mkdir(parents=True)
            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>제10장 세마포</w:t></w:r></w:p>
    <w:p><w:r><w:t>성소가 완성되는 과정에서 남은 것은 세마포와 은받침입니다.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
            with zipfile.ZipFile(docx_path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            hwp_path = root / "datas" / "hwps" / "sample.hwp"
            hwp_path.parent.mkdir(parents=True)
            hwp_path.write_bytes(b"HWP fixture bytes")

            out_dir = root / "out"
            status = run(
                root=root,
                out_dir=out_dir,
                model="gpt-5.5",
                max_sentences=4,
                max_output_tokens=8192,
                timeout=30,
                dry_run=True,
                hwp_reader_factory=FakeReader,
            )

            self.assertEqual(status, 0)
            self.assertTrue((out_dir / "inputs" / "datalab_xhigh.payload.json").exists())
            self.assertTrue((out_dir / "inputs" / "docx_high.payload.json").exists())
            self.assertTrue((out_dir / "inputs" / "hwp_high.payload.json").exists())

            comparison = (out_dir / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("OpenAI Preprocessing Comparison", comparison)
            self.assertIn("dry_run", comparison)


if __name__ == "__main__":
    unittest.main()
