import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from sermon_pipeline.dataset import build_dataset


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
            FakeSection(["오늘 본문 말씀입니다.", "두 번째 문장입니다."])
        ]


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class DatasetBuildTests(unittest.TestCase):
    def test_build_dataset_writes_success_rows_and_traceable_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datalab = root / "datas" / "datalab_parsed" / "book" / "chapter.json"
            datalab.parent.mkdir(parents=True)
            datalab.write_text(
                json.dumps(
                    {
                        "success": True,
                        "page_count": 1,
                        "html": "<h2>제목</h2><p>첫 문장입니다. 둘째 문장입니다.</p>",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            docx_dir = root / "datas" / "docx"
            docx_dir.mkdir(parents=True)
            valid_docx = docx_dir / "valid.docx"
            document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>설교 문장입니다.</w:t></w:r></w:p></w:body>
</w:document>"""
            with zipfile.ZipFile(valid_docx, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            (docx_dir / "broken.docx").write_bytes(b"not a zip")

            hwp = root / "datas" / "hwps" / "sample.hwp"
            hwp.parent.mkdir(parents=True)
            hwp.write_bytes(b"HWP fixture")

            out_dir = root / "out"
            summary = build_dataset(
                root=root,
                out_dir=out_dir,
                max_sentences_per_document=5,
                hwp_reader_factory=FakeReader,
            )

            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["succeeded"], 3)
            self.assertGreater(summary["total_sentences"], 0)

            documents = read_jsonl(out_dir / "documents.jsonl")
            sentences = read_jsonl(out_dir / "sentences.jsonl")
            failures = read_jsonl(out_dir / "failures.jsonl")

            self.assertEqual(len(documents), 3)
            self.assertGreaterEqual(len(sentences), 3)
            self.assertEqual(failures[0]["source_type"], "docx")
            self.assertEqual(failures[0]["source_path"], "datas/docx/broken.docx")
            self.assertIn("exception_type", failures[0])
            self.assertIn("traceback", failures[0])

            saved_summary = json.loads(
                (out_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_summary["failed"], 1)


if __name__ == "__main__":
    unittest.main()
