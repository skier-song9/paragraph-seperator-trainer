# Teacher Annotation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable source-aware pipeline that extracts DATAlab HTML JSON, DOCX, and HWP documents into stable sentence units, builds strict LLM teacher annotation payloads, validates teacher outputs, and converts them into classifier training rows.

**Architecture:** Split the current smoke-test script into a small `src/sermon_pipeline` package with focused modules for models, text normalization, extraction, sentence splitting, payload construction, validation, and CLI orchestration. Keep `tests/openai_preprocessing_comparison.py` as a thin compatibility runner after package modules exist. All source types produce the same `PreparedDocument` contract before LLM payload generation.

**Tech Stack:** Python 3.11+ standard library, `unittest`, optional `libhwp==0.2.0` for HWP extraction, OpenAI Responses API via `urllib.request`.

---

## Scope Check

This plan implements one subsystem: teacher annotation data preparation and output validation. It does not train the boundary classifier, build embeddings, or create RAG chunks beyond converting teacher output into sentence-pair training rows. Those should be separate plans after this pipeline is stable.

## File Structure

- Create: `src/sermon_pipeline/__init__.py`  
  Package exports and version string.
- Create: `src/sermon_pipeline/constants.py`  
  Shared boundary taxonomy and prompt text constants.
- Create: `src/sermon_pipeline/models.py`  
  Dataclasses for blocks, sentences, prepared documents, teacher annotations, validation issues, and training rows.
- Create: `src/sermon_pipeline/text.py`  
  Whitespace normalization, layout-noise detection, script counting, and Korean paragraph classification.
- Create: `src/sermon_pipeline/sentence_splitter.py`  
  Deterministic sentence-unit generation with stable IDs.
- Create: `src/sermon_pipeline/extractors/__init__.py`  
  Extractor namespace.
- Create: `src/sermon_pipeline/extractors/datalab.py`  
  DATAlab JSON HTML block parsing and document preparation.
- Create: `src/sermon_pipeline/extractors/docx.py`  
  DOCX paragraph extraction, Korean filtering, and document preparation.
- Create: `src/sermon_pipeline/extractors/hwp.py`  
  HWP extraction through injectable `libhwp` reader adapter.
- Create: `src/sermon_pipeline/teacher.py`  
  System prompt, source-specific user instructions, response schema, and OpenAI payload builder.
- Create: `src/sermon_pipeline/validation.py`  
  Teacher output validation and sentence-pair training row conversion.
- Create: `src/sermon_pipeline/io.py`  
  JSON writing, dotenv loading, output-text extraction, and markdown comparison rendering.
- Create: `src/sermon_pipeline/openai_client.py`  
  Minimal Responses API caller.
- Create: `src/sermon_pipeline/cli.py`  
  CLI that prepares three sample cases, supports dry run, and optionally calls OpenAI.
- Modify: `tests/openai_preprocessing_comparison.py`  
  Replace embedded logic with a call into `sermon_pipeline.cli.main`.
- Create: `tests/test_text.py`  
  Unit tests for normalization and language filtering.
- Create: `tests/test_sentence_splitter.py`  
  Unit tests for stable sentence IDs and metadata retention.
- Create: `tests/test_extractors.py`  
  Unit tests for DATAlab HTML parsing, DOCX XML paragraph extraction, and HWP reader injection.
- Create: `tests/test_teacher_payload.py`  
  Unit tests for prompt construction, source-specific instructions, strict schema, and payload JSON.
- Create: `tests/test_validation.py`  
  Unit tests for output validation and training-row conversion.
- Modify: `tests/README.md`  
  Document new test commands and dry-run CLI.

## Implementation Notes

- Use `PYTHONPATH=src python3 -m unittest` for all tests. No new test dependency is required.
- Do not call OpenAI in unit tests.
- Do not import `libhwp` at module import time. Import it only inside the HWP adapter path so tests can run without installing it.
- Preserve existing `tests/results/20260527_185341` as fixture-like historical output; do not edit it.
- Preserve user-created review documents under `docs/reviews`.

### Task 1: Shared Package Models And Constants

**Files:**
- Create: `src/sermon_pipeline/__init__.py`
- Create: `src/sermon_pipeline/constants.py`
- Create: `src/sermon_pipeline/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_models -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline'`.

- [ ] **Step 3: Create package constants and models**

Create `src/sermon_pipeline/__init__.py`:

```python
"""Source-aware sermon boundary annotation pipeline."""

__version__ = "0.1.0"
```

Create `src/sermon_pipeline/constants.py`:

```python
BOUNDARY_TYPES = [
    "none",
    "topic_shift",
    "scripture_reading_start",
    "scripture_explanation_start",
    "illustration_start",
    "application_start",
    "prayer_or_closing",
    "enumeration_start",
]

SYSTEM_PROMPT = (
    "You annotate Korean sermon discourse boundaries for RAG preprocessing. "
    "Do not rewrite, summarize, translate, or normalize the sermon text. "
    "Use only sentence_id references when proposing boundaries. "
    "Return split_after for every provided sentence. "
    "Prefer atomic semantic paragraphs, preserve scripture quotations with their references, "
    "and mark boundary_type from the allowed taxonomy. "
    "If metadata and semantics conflict, explain the conflict in rationale or quality_flags."
)
```

Create `src/sermon_pipeline/models.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import BOUNDARY_TYPES


@dataclass(frozen=True)
class SourceBlock:
    block_id: str
    text: str
    block_type: str
    source_tag: str
    page_id: str | None = None
    paragraph_index: int | None = None
    heading_context: list[str] = field(default_factory=list)
    html_boundary_before: bool = False
    language_filter_reason: str | None = None
    script_counts: dict[str, int] = field(default_factory=dict)
    section_id: int | None = None

    def to_payload(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class SentenceUnit:
    sentence_id: str
    text: str
    block_id: str
    block_type: str | None = None
    source_tag: str | None = None
    page_id: str | None = None
    paragraph_index: int | None = None
    heading_context: list[str] = field(default_factory=list)
    html_boundary_before: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "block_id": self.block_id,
            "block_type": self.block_type,
            "source_tag": self.source_tag,
            "page_id": self.page_id,
            "paragraph_index": self.paragraph_index,
            "heading_context": list(self.heading_context),
            "html_boundary_before": self.html_boundary_before,
        }


@dataclass(frozen=True)
class PreparedDocument:
    document_id: str
    source_type: str
    document_kind: str
    source_path: str
    reasoning_effort: str
    effort_label: str
    extraction_notes: list[str]
    removed_foreign_paragraphs: int
    blocks: list[SourceBlock]
    sentences: list[SentenceUnit]

    def to_teacher_task(self, instructions: list[str]) -> dict[str, Any]:
        return {
            "task": "annotate_sentence_boundaries",
            "document_id": self.document_id,
            "source_type": self.source_type,
            "document_kind": self.document_kind,
            "reasoning_effort": self.reasoning_effort,
            "effort_label": self.effort_label,
            "source_path": self.source_path,
            "extraction_notes": list(self.extraction_notes),
            "removed_foreign_paragraphs": self.removed_foreign_paragraphs,
            "allowed_boundary_types": list(BOUNDARY_TYPES),
            "instructions": list(instructions),
            "sentences": [sentence.to_payload() for sentence in self.sentences],
        }


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    sentence_id: str | None = None


@dataclass(frozen=True)
class TrainingRow:
    left_sentence_id: str
    right_sentence_id: str | None
    split_after_left: bool
    boundary_type: str
    teacher_confidence: float
    source_type: str
    document_kind: str
    features: dict[str, Any]
    review_status: str
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_models -v
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/__init__.py src/sermon_pipeline/constants.py src/sermon_pipeline/models.py tests/test_models.py
git commit -m "feat: add pipeline models"
```

### Task 2: Text Normalization And Language Filtering

**Files:**
- Create: `src/sermon_pipeline/text.py`
- Test: `tests/test_text.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_text.py`:

```python
import unittest

from sermon_pipeline.text import classify_korean_paragraph, is_layout_noise, normalize_ws, script_counts


class TextTests(unittest.TestCase):
    def test_normalize_ws_collapses_spaces_and_nbsp(self):
        self.assertEqual(normalize_ws("  요한\u00a0  일서\n2장\t1절  "), "요한 일서 2장 1절")

    def test_layout_noise_filters_page_numbers_and_marks(self):
        self.assertTrue(is_layout_noise("12"))
        self.assertTrue(is_layout_noise("•"))
        self.assertFalse(is_layout_noise("오늘 본문 말씀은 요한일서 2장 1절입니다."))

    def test_script_counts_detects_mixed_theology_terms(self):
        counts = script_counts("파라클레토스 παράκλητος logos דבר")
        self.assertGreater(counts["hangul"], 0)
        self.assertGreater(counts["greek"], 0)
        self.assertGreater(counts["latin"], 0)
        self.assertGreater(counts["hebrew"], 0)

    def test_classify_korean_paragraph_keeps_korean_and_scripture_reference(self):
        keep, counts, reason = classify_korean_paragraph("(계 19:7, 8)")
        self.assertTrue(keep)
        self.assertEqual(reason, "scripture_reference_or_short_context")
        self.assertEqual(counts["hangul"], 1)

    def test_classify_korean_paragraph_removes_foreign_translation(self):
        keep, counts, reason = classify_korean_paragraph("This paragraph is an English translation with many latin letters.")
        self.assertFalse(keep)
        self.assertEqual(reason, "latin_dominant_foreign")
        self.assertGreater(counts["latin"], 20)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_text -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.text'`.

- [ ] **Step 3: Implement text helpers**

Create `src/sermon_pipeline/text.py`:

```python
from __future__ import annotations

import re


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def is_layout_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"[0-9]{1,3}", stripped):
        return True
    if stripped in {"•", "-", "—", "_"}:
        return True
    return False


def script_counts(text: str) -> dict[str, int]:
    counts = {"hangul": 0, "han": 0, "latin": 0, "greek": 0, "hebrew": 0}
    for ch in text:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            counts["hangul"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["han"] += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            counts["latin"] += 1
        elif 0x0370 <= cp <= 0x03FF:
            counts["greek"] += 1
        elif 0x0590 <= cp <= 0x05FF:
            counts["hebrew"] += 1
    return counts


def classify_korean_paragraph(text: str) -> tuple[bool, dict[str, int], str]:
    counts = script_counts(text)
    hangul = counts["hangul"]
    han = counts["han"]
    latin = counts["latin"]

    if hangul >= 2:
        return True, counts, "hangul_present"
    if re.search(r"\([^)]+[0-9]+[:：][0-9]+", text):
        return True, counts, "scripture_reference_or_short_context"
    if hangul == 0 and han >= 2:
        return False, counts, "han_dominant_foreign"
    if hangul == 0 and latin >= 20:
        return False, counts, "latin_dominant_foreign"
    return False, counts, "no_korean_signal"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_text -v
```

Expected: PASS with 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/text.py tests/test_text.py
git commit -m "feat: add text normalization helpers"
```

### Task 3: Stable Sentence Splitting

**Files:**
- Create: `src/sermon_pipeline/sentence_splitter.py`
- Test: `tests/test_sentence_splitter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sentence_splitter.py`:

```python
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

        sentences = split_blocks_into_sentences([block], "datalab_sample", max_sentences=8)

        self.assertEqual([s.sentence_id for s in sentences], ["datalab_sample.s0000", "datalab_sample.s0001"])
        self.assertEqual(sentences[0].page_id, "1")
        self.assertEqual(sentences[0].heading_context, ["율법과 복음"])
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_sentence_splitter -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.sentence_splitter'`.

- [ ] **Step 3: Implement sentence splitter**

Create `src/sermon_pipeline/sentence_splitter.py`:

```python
from __future__ import annotations

import re
from collections.abc import Iterable

from .models import SentenceUnit, SourceBlock
from .text import normalize_ws


def _split_text(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    if len(parts) == 1 and len(text) > 450:
        parts = re.split(r"(?<=[다요죠까라니함됨음])\s+", text)
    return [normalize_ws(part) for part in parts if len(normalize_ws(part)) >= 2]


def split_blocks_into_sentences(
    blocks: Iterable[SourceBlock],
    document_id: str,
    max_sentences: int | None = None,
) -> list[SentenceUnit]:
    sentences: list[SentenceUnit] = []
    for block in blocks:
        for sent in _split_text(block.text):
            sentence_id = f"{document_id}.s{len(sentences):04d}"
            sentences.append(
                SentenceUnit(
                    sentence_id=sentence_id,
                    text=sent,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    source_tag=block.source_tag,
                    page_id=block.page_id,
                    paragraph_index=block.paragraph_index,
                    heading_context=list(block.heading_context),
                    html_boundary_before=block.html_boundary_before,
                )
            )
            if max_sentences is not None and len(sentences) >= max_sentences:
                return sentences
    return sentences
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_sentence_splitter -v
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/sentence_splitter.py tests/test_sentence_splitter.py
git commit -m "feat: add stable sentence splitter"
```

### Task 4: DATAlab HTML JSON Extractor

**Files:**
- Create: `src/sermon_pipeline/extractors/__init__.py`
- Create: `src/sermon_pipeline/extractors/datalab.py`
- Test: `tests/test_extractors.py`

- [ ] **Step 1: Write the failing DATAlab extractor test**

Create `tests/test_extractors.py` with this initial content:

```python
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

            document = parse_datalab_json(path, root=Path(tmp), document_id="datalab_sample", max_sentences=8)

        self.assertEqual(document.source_type, "datalab_parsed_json")
        self.assertEqual(document.reasoning_effort, "xhigh")
        self.assertEqual(document.extraction_notes[-1], "page_count=2")
        self.assertEqual(document.blocks[0].source_tag, "h2")
        self.assertEqual(document.blocks[0].page_id, "0")
        self.assertTrue(document.blocks[0].html_boundary_before)
        self.assertEqual(document.sentences[0].sentence_id, "datalab_sample.s0000")
        self.assertEqual(document.sentences[1].heading_context, ["인간의 기준은 인간의 양심이고 하나님의 기준은 하나님이 양심이다"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_extractors.DatalabExtractorTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.extractors'`.

- [ ] **Step 3: Implement DATAlab extractor**

Create `src/sermon_pipeline/extractors/__init__.py`:

```python
"""Document extractors for supported source formats."""
```

Create `src/sermon_pipeline/extractors/datalab.py`:

```python
from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

from sermon_pipeline.models import PreparedDocument, SourceBlock
from sermon_pipeline.sentence_splitter import split_blocks_into_sentences
from sermon_pipeline.text import is_layout_noise, normalize_ws


class DatalabHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[SourceBlock] = []
        self.page_id: str | None = None
        self.heading_context: list[str] = []
        self.current: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "div" and attrs_dict.get("data-page-id") is not None:
            self.page_id = attrs_dict.get("data-page-id")
        if tag in {"h1", "h2", "h3", "p", "li", "td", "th"}:
            self.current = {
                "tag": tag,
                "page_id": self.page_id,
                "heading_context": list(self.heading_context[-3:]),
                "parts": [],
            }
        if tag == "br" and self.current is not None:
            self.current["parts"].append("\n")

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current["parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is None or tag != self.current["tag"]:
            return
        parts = self.current["parts"]
        text = normalize_ws(" ".join(str(part) for part in parts))
        source_tag = str(self.current["tag"])
        if text and not is_layout_noise(text):
            block = SourceBlock(
                block_id=f"html.b{len(self.blocks):04d}",
                text=text,
                block_type="heading" if source_tag.startswith("h") else "paragraph",
                source_tag=source_tag,
                page_id=self.current["page_id"],
                heading_context=list(self.current["heading_context"]),
            )
            self.blocks.append(block)
            if source_tag.startswith("h"):
                self.heading_context.append(text)
        self.current = None


def _mark_html_boundaries(blocks: list[SourceBlock]) -> list[SourceBlock]:
    marked: list[SourceBlock] = []
    for idx, block in enumerate(blocks):
        previous_tag = blocks[idx - 1].source_tag if idx else None
        boundary = idx == 0 or previous_tag in {"h1", "h2", "h3"} or block.source_tag in {"h1", "h2", "h3"}
        marked.append(
            SourceBlock(
                block_id=block.block_id,
                text=block.text,
                block_type=block.block_type,
                source_tag=block.source_tag,
                page_id=block.page_id,
                paragraph_index=block.paragraph_index,
                heading_context=list(block.heading_context),
                html_boundary_before=boundary,
            )
        )
    return marked


def parse_datalab_json(
    path: Path,
    root: Path,
    document_id: str,
    max_sentences: int | None = None,
) -> PreparedDocument:
    data = json.loads(path.read_text(encoding="utf-8"))
    parser = DatalabHTMLParser()
    parser.feed(data.get("html") or "")
    blocks = _mark_html_boundaries(parser.blocks)
    sentences = split_blocks_into_sentences(blocks, document_id, max_sentences=max_sentences)
    return PreparedDocument(
        document_id=document_id,
        source_type="datalab_parsed_json",
        document_kind="book_chapter",
        source_path=str(path.relative_to(root)),
        reasoning_effort="xhigh",
        effort_label="최상",
        extraction_notes=[
            "JSON html parsed with HTMLParser.",
            "h1/h2/p/page metadata retained as label hints.",
            f"page_count={data.get('page_count')}",
        ],
        removed_foreign_paragraphs=0,
        blocks=blocks,
        sentences=sentences,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_extractors.DatalabExtractorTests -v
```

Expected: PASS with 1 test.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/extractors/__init__.py src/sermon_pipeline/extractors/datalab.py tests/test_extractors.py
git commit -m "feat: add datalab html extractor"
```

### Task 5: DOCX Extractor

**Files:**
- Create: `src/sermon_pipeline/extractors/docx.py`
- Modify: `tests/test_extractors.py`

- [ ] **Step 1: Add failing DOCX extractor tests**

Append this class to `tests/test_extractors.py` before the `if __name__ == "__main__"` block:

```python
import zipfile

from sermon_pipeline.extractors.docx import extract_docx_paragraphs, parse_docx


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
            document = parse_docx(path, root=Path(tmp), document_id="docx_sample", max_sentences=8)

        self.assertEqual(paragraphs[0], "제10장 세마포")
        self.assertEqual(document.source_type, "docx")
        self.assertEqual(document.removed_foreign_paragraphs, 1)
        self.assertEqual(len(document.blocks), 2)
        self.assertEqual(document.blocks[0].paragraph_index, 0)
        self.assertEqual(document.blocks[1].paragraph_index, 2)
        self.assertEqual(document.blocks[0].language_filter_reason, "hangul_present")
        self.assertEqual(document.sentences[0].sentence_id, "docx_sample.s0000")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_extractors.DocxExtractorTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.extractors.docx'`.

- [ ] **Step 3: Implement DOCX extractor**

Create `src/sermon_pipeline/extractors/docx.py`:

```python
from __future__ import annotations

import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from sermon_pipeline.models import PreparedDocument, SourceBlock
from sermon_pipeline.sentence_splitter import split_blocks_into_sentences
from sermon_pipeline.text import classify_korean_paragraph, normalize_ws


def extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for para in root.iter(f"{ns}p"):
        parts: list[str] = []
        for node in para.iter():
            if node.tag == f"{ns}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{ns}tab":
                parts.append(" ")
            elif node.tag == f"{ns}br":
                parts.append("\n")
        text = normalize_ws("".join(parts))
        if text:
            paragraphs.append(text)
    return paragraphs


def infer_document_kind_from_name(path: Path) -> str:
    name = unicodedata.normalize("NFC", path.name)
    if "기도" in name:
        return "prayer"
    if "논문" in name:
        return "thesis"
    if "공지" in name:
        return "notice"
    return "sermon"


def parse_docx(
    path: Path,
    root: Path,
    document_id: str,
    max_sentences: int | None = None,
) -> PreparedDocument:
    paragraphs = extract_docx_paragraphs(path)
    blocks: list[SourceBlock] = []
    removed_foreign = 0
    for idx, text in enumerate(paragraphs):
        keep, counts, reason = classify_korean_paragraph(text)
        if not keep:
            removed_foreign += 1
            continue
        blocks.append(
            SourceBlock(
                block_id=f"docx.b{len(blocks):04d}",
                text=text,
                block_type="paragraph",
                source_tag="docx_paragraph",
                paragraph_index=idx,
                language_filter_reason=reason,
                script_counts=counts,
            )
        )
    sentences = split_blocks_into_sentences(blocks, document_id, max_sentences=max_sentences)
    return PreparedDocument(
        document_id=document_id,
        source_type="docx",
        document_kind=infer_document_kind_from_name(path),
        source_path=str(path.relative_to(root)),
        reasoning_effort="high",
        effort_label="high",
        extraction_notes=[
            "DOCX word/document.xml parsed by paragraph.",
            "Foreign translation paragraphs removed by script-ratio heuristic.",
            f"raw_paragraphs={len(paragraphs)}",
        ],
        removed_foreign_paragraphs=removed_foreign,
        blocks=blocks,
        sentences=sentences,
    )
```

- [ ] **Step 4: Run DOCX tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_extractors.DocxExtractorTests -v
```

Expected: PASS with 1 test.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/extractors/docx.py tests/test_extractors.py
git commit -m "feat: add docx extractor"
```

### Task 6: HWP Extractor With Injectable Reader

**Files:**
- Create: `src/sermon_pipeline/extractors/hwp.py`
- Modify: `tests/test_extractors.py`

- [ ] **Step 1: Add failing HWP extractor tests**

Append this class to `tests/test_extractors.py` before the `if __name__ == "__main__"` block:

```python
from sermon_pipeline.extractors.hwp import parse_hwp


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
                    "20141113 목요찬양예배-파라클레토스(παράκλητος)",
                    "오늘 본문 말씀은 요한일서 2장 1절입니다.",
                ]
            )
        ]


class HwpExtractorTests(unittest.TestCase):
    def test_parse_hwp_uses_injected_reader_and_keeps_version_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.hwp"
            path.write_bytes(b"HWP fixture bytes")

            document = parse_hwp(
                path,
                root=Path(tmp),
                document_id="hwp_sample",
                reader_factory=FakeReader,
                max_sentences=8,
            )

        self.assertEqual(document.source_type, "hwp")
        self.assertEqual(document.reasoning_effort, "high")
        self.assertIn("hwp_version=5.0.3.0", document.extraction_notes)
        self.assertEqual(len(document.blocks), 2)
        self.assertEqual(document.blocks[0].source_tag, "hwp_paragraph")
        self.assertEqual(document.sentences[0].sentence_id, "hwp_sample.s0000")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_extractors.HwpExtractorTests -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.extractors.hwp'`.

- [ ] **Step 3: Implement HWP extractor**

Create `src/sermon_pipeline/extractors/hwp.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sermon_pipeline.models import PreparedDocument, SourceBlock
from sermon_pipeline.sentence_splitter import split_blocks_into_sentences
from sermon_pipeline.text import normalize_ws


def load_libhwp_reader(path: str) -> Any:
    import libhwp

    return libhwp.HWPReader(path)


def extract_hwp_paragraphs(path: Path, reader_factory: Callable[[str], Any] = load_libhwp_reader) -> tuple[list[str], str]:
    reader = reader_factory(str(path))
    paragraphs: list[str] = []
    for section in reader.sections:
        for para in section.paragraphs:
            chars: list[str] = []
            for ch in para.chars:
                if getattr(ch, "kind", None) == "char_code" and getattr(ch, "code", 0):
                    chars.append(chr(ch.code))
            text = normalize_ws("".join(chars))
            if text:
                paragraphs.append(text)
    return paragraphs, str(reader.version)


def parse_hwp(
    path: Path,
    root: Path,
    document_id: str,
    reader_factory: Callable[[str], Any] = load_libhwp_reader,
    max_sentences: int | None = None,
) -> PreparedDocument:
    paragraphs, version = extract_hwp_paragraphs(path, reader_factory)
    blocks = [
        SourceBlock(
            block_id=f"hwp.b{idx:04d}",
            text=text,
            block_type="paragraph",
            source_tag="hwp_paragraph",
            paragraph_index=idx,
        )
        for idx, text in enumerate(paragraphs)
    ]
    sentences = split_blocks_into_sentences(blocks, document_id, max_sentences=max_sentences)
    return PreparedDocument(
        document_id=document_id,
        source_type="hwp",
        document_kind="sermon",
        source_path=str(path.relative_to(root)),
        reasoning_effort="high",
        effort_label="high",
        extraction_notes=[
            "HWP parsed with libhwp.",
            f"hwp_version={version}",
            f"paragraphs={len(paragraphs)}",
        ],
        removed_foreign_paragraphs=0,
        blocks=blocks,
        sentences=sentences,
    )
```

- [ ] **Step 4: Run extractor tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_extractors -v
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/extractors/hwp.py tests/test_extractors.py
git commit -m "feat: add hwp extractor"
```

### Task 7: Teacher Payload Builder And Strict Schema

**Files:**
- Create: `src/sermon_pipeline/teacher.py`
- Test: `tests/test_teacher_payload.py`

- [ ] **Step 1: Write the failing payload tests**

Create `tests/test_teacher_payload.py`:

```python
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
        self.assertIn("Return one boundary_annotations item for every provided sentence.", instructions)
        self.assertIn("Do not infer boundaries from removed foreign paragraphs.", instructions)

    def test_response_schema_is_strict_and_uses_boundary_taxonomy(self):
        schema = response_schema()
        boundary_enum = schema["properties"]["boundary_annotations"]["items"]["properties"]["boundary_type"]["enum"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(boundary_enum, BOUNDARY_TYPES)

    def test_build_payload_serializes_teacher_task_as_json_content(self):
        document = self.make_document("datalab_parsed_json")
        payload = build_payload(document, model="gpt-5.5", max_output_tokens=8192)
        user_content = payload["input"][1]["content"]
        task = json.loads(user_content)

        self.assertEqual(payload["input"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(task["task"], "annotate_sentence_boundaries")
        self.assertIn("HTML tag and heading_context are boundary hints, not gold labels.", task["instructions"])
        self.assertEqual(payload["text"]["format"]["schema"]["required"][0], "document_id")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_teacher_payload -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.teacher'`.

- [ ] **Step 3: Implement payload builder**

Create `src/sermon_pipeline/teacher.py`:

```python
from __future__ import annotations

import json
from typing import Any

from .constants import BOUNDARY_TYPES, SYSTEM_PROMPT
from .models import PreparedDocument


COMMON_INSTRUCTIONS = [
    "Return one boundary_annotations item for every provided sentence.",
    "Use boundary_type='none' when split_after=false.",
    "Keep Bible quotation spans together with verse references where possible.",
    "Treat headings, original paragraphs, and page boundaries as hints, not gold labels.",
    "Do not infer a split after the final sentence unless the boundary is explicit inside the excerpt.",
    "Report extraction noise, duplicate headings, truncation, and uncertain language filtering in quality_flags.",
]

SOURCE_INSTRUCTION_DELTAS = {
    "datalab_parsed_json": [
        "HTML tag and heading_context are boundary hints, not gold labels.",
        "Detect repeated page headings and report them in quality_flags.",
        "Keep scripture text and verse reference in the same atomic paragraph where possible.",
    ],
    "docx": [
        "Do not infer boundaries from removed foreign paragraphs.",
        "Treat rhetorical question and immediate answer as one semantic unit unless a new topic begins.",
        "Preserve verse quotation and verse reference as one scripture_reading paragraph.",
    ],
    "hwp": [
        "Do not repair possible extraction noise; flag it.",
        "If the excerpt ends mid-discourse, do not invent a following boundary.",
        "Separate scripture-reading setup, scripture quotation, and explanation cue when the transition is explicit.",
    ],
}


def source_instructions(source_type: str) -> list[str]:
    return list(COMMON_INSTRUCTIONS) + list(SOURCE_INSTRUCTION_DELTAS.get(source_type, []))


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_id": {"type": "string"},
            "source_type": {"type": "string"},
            "reasoning_effort": {"type": "string"},
            "preprocessing_observations": {"type": "array", "items": {"type": "string"}},
            "boundary_annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sentence_id": {"type": "string"},
                        "text_excerpt": {"type": "string"},
                        "split_after": {"type": "boolean"},
                        "boundary_type": {"type": "string", "enum": BOUNDARY_TYPES},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "sentence_id",
                        "text_excerpt",
                        "split_after",
                        "boundary_type",
                        "confidence",
                        "rationale",
                    ],
                },
            },
            "proposed_atomic_paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "paragraph_id": {"type": "string"},
                        "sentence_ids": {"type": "array", "items": {"type": "string"}},
                        "paragraph_role": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["paragraph_id", "sentence_ids", "paragraph_role", "reason"],
                },
            },
            "quality_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "document_id",
            "source_type",
            "reasoning_effort",
            "preprocessing_observations",
            "boundary_annotations",
            "proposed_atomic_paragraphs",
            "quality_flags",
        ],
    }


def build_payload(document: PreparedDocument, model: str, max_output_tokens: int) -> dict[str, Any]:
    task = document.to_teacher_task(source_instructions(document.source_type))
    return {
        "model": model,
        "reasoning": {"effort": document.reasoning_effort},
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(task, ensure_ascii=False, indent=2)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sermon_boundary_annotation",
                "schema": response_schema(),
                "strict": True,
            }
        },
        "max_output_tokens": max_output_tokens,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_teacher_payload -v
```

Expected: PASS with 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/teacher.py tests/test_teacher_payload.py
git commit -m "feat: add teacher payload builder"
```

### Task 8: Teacher Output Validation And Training Rows

**Files:**
- Create: `src/sermon_pipeline/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing validation tests**

Create `tests/test_validation.py`:

```python
import unittest

from sermon_pipeline.models import PreparedDocument, SentenceUnit, SourceBlock
from sermon_pipeline.validation import teacher_output_to_training_rows, validate_teacher_output


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
            SentenceUnit("doc.s0000", "제10장 세마포", "docx.b0000", "paragraph", "docx_paragraph", paragraph_index=0),
            SentenceUnit("doc.s0001", "성경 본문입니다.", "docx.b0001", "paragraph", "docx_paragraph", paragraph_index=1),
            SentenceUnit("doc.s0002", "(계 19:7, 8)", "docx.b0001", "paragraph", "docx_paragraph", paragraph_index=1),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_validation -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.validation'`.

- [ ] **Step 3: Implement validation and training conversion**

Create `src/sermon_pipeline/validation.py`:

```python
from __future__ import annotations

from typing import Any

from .constants import BOUNDARY_TYPES
from .models import PreparedDocument, SentenceUnit, TrainingRow, ValidationIssue


def validate_teacher_output(document: PreparedDocument, output: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    sentence_ids = [sentence.sentence_id for sentence in document.sentences]
    expected = set(sentence_ids)
    annotations = output.get("boundary_annotations", [])
    annotated = [item.get("sentence_id") for item in annotations]
    annotated_set = set(annotated)

    if len(annotations) != len(sentence_ids):
        issues.append(
            ValidationIssue(
                code="coverage_mismatch",
                message=f"expected {len(sentence_ids)} annotations, got {len(annotations)}",
            )
        )
    for sentence_id in sorted(expected - annotated_set):
        issues.append(ValidationIssue("missing_sentence_annotation", "sentence was not annotated", sentence_id))
    for sentence_id in sorted(annotated_set - expected):
        issues.append(ValidationIssue("unknown_sentence_id", "annotation referenced an unknown sentence", sentence_id))

    for item in annotations:
        sentence_id = item.get("sentence_id")
        split_after = item.get("split_after")
        boundary_type = item.get("boundary_type")
        if boundary_type not in BOUNDARY_TYPES:
            issues.append(ValidationIssue("unknown_boundary_type", f"invalid boundary_type={boundary_type}", sentence_id))
        if split_after is False and boundary_type != "none":
            issues.append(
                ValidationIssue(
                    "split_false_boundary_type_not_none",
                    "split_after=false requires boundary_type='none'",
                    sentence_id,
                )
            )
        if split_after is True and boundary_type == "none":
            issues.append(
                ValidationIssue(
                    "split_true_boundary_type_none",
                    "split_after=true requires a non-none boundary_type",
                    sentence_id,
                )
            )
    return issues


def _feature_map(left: SentenceUnit, right: SentenceUnit | None) -> dict[str, Any]:
    return {
        "left_block_type": left.block_type,
        "right_block_type": right.block_type if right else None,
        "html_boundary_before_right": right.html_boundary_before if right else False,
        "same_original_paragraph": (
            right is not None
            and left.paragraph_index is not None
            and left.paragraph_index == right.paragraph_index
            and left.block_id == right.block_id
        ),
        "heading_context_depth": len(left.heading_context),
    }


def teacher_output_to_training_rows(document: PreparedDocument, output: dict[str, Any]) -> list[TrainingRow]:
    annotations_by_id = {item["sentence_id"]: item for item in output.get("boundary_annotations", [])}
    quality_flags = output.get("quality_flags", [])
    rows: list[TrainingRow] = []
    for index, sentence in enumerate(document.sentences):
        annotation = annotations_by_id[sentence.sentence_id]
        right = document.sentences[index + 1] if index + 1 < len(document.sentences) else None
        confidence = float(annotation["confidence"])
        review_status = "teacher_only"
        if confidence < 0.75 or quality_flags:
            review_status = "needs_review"
        rows.append(
            TrainingRow(
                left_sentence_id=sentence.sentence_id,
                right_sentence_id=right.sentence_id if right else None,
                split_after_left=bool(annotation["split_after"]),
                boundary_type=str(annotation["boundary_type"]),
                teacher_confidence=confidence,
                source_type=document.source_type,
                document_kind=document.document_kind,
                features=_feature_map(sentence, right),
                review_status=review_status,
            )
        )
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_validation -v
```

Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sermon_pipeline/validation.py tests/test_validation.py
git commit -m "feat: validate teacher annotations"
```

### Task 9: IO Helpers, OpenAI Client, And CLI Runner

**Files:**
- Create: `src/sermon_pipeline/io.py`
- Create: `src/sermon_pipeline/openai_client.py`
- Create: `src/sermon_pipeline/cli.py`
- Modify: `tests/openai_preprocessing_comparison.py`
- Modify: `tests/README.md`
- Test: `tests/test_cli_smoke.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/test_cli_smoke.py`:

```python
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from sermon_pipeline.cli import run


class CliSmokeTests(unittest.TestCase):
    def test_run_dry_run_writes_payloads_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datalab_dir = root / "datas" / "datalab_parsed" / "book"
            docx_dir = root / "datas" / "docx"
            hwp_dir = root / "datas" / "hwps"
            datalab_dir.mkdir(parents=True)
            docx_dir.mkdir(parents=True)
            hwp_dir.mkdir(parents=True)

            (datalab_dir / "chapter.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "page_count": 1,
                        "html": "<div data-page-id='0'><h2>율법과 복음</h2><p>오래간만에 여러분을 뵙게 되어서 정말 반갑습니다.</p></div>",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>제10장 세마포</w:t></w:r></w:p>
    <w:p><w:r><w:t>성경 본문입니다.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
            with zipfile.ZipFile(docx_dir / "세마포__설교 10장.docx", "w") as archive:
                archive.writestr("word/document.xml", document_xml)

            class FakeChar:
                def __init__(self, code):
                    self.kind = "char_code"
                    self.code = code

            class FakeParagraph:
                def __init__(self, text):
                    self.chars = [FakeChar(ord(ch)) for ch in text]

            class FakeSection:
                paragraphs = [FakeParagraph("오늘 본문 말씀은 요한일서 2장 1절입니다.")]

            class FakeReader:
                version = "5.0.3.0"
                sections = [FakeSection()]

                def __init__(self, path):
                    self.path = path

            (hwp_dir / "sample.hwp").write_bytes(b"hwp")
            out_dir = root / "tests" / "results" / "dryrun"

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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_cli_smoke -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sermon_pipeline.cli'`.

- [ ] **Step 3: Implement IO helpers**

Create `src/sermon_pipeline/io.py`:

```python
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    texts: list[str] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("type") in {"output_text", "text"} and isinstance(obj.get("text"), str):
                texts.append(obj["text"])
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(response.get("output", []))
    return "\n".join(texts).strip()


def summarize_annotation(annotation: dict[str, Any] | None) -> dict[str, Any]:
    if not annotation:
        return {"split_count": None, "boundary_types": {}, "quality_flags": []}
    boundary_types: dict[str, int] = {}
    split_count = 0
    for item in annotation.get("boundary_annotations", []):
        boundary_type = item.get("boundary_type", "unknown")
        boundary_types[boundary_type] = boundary_types.get(boundary_type, 0) + 1
        if item.get("split_after"):
            split_count += 1
    return {
        "split_count": split_count,
        "boundary_types": boundary_types,
        "quality_flags": annotation.get("quality_flags", []),
    }


def render_comparison(rows: list[dict[str, Any]], model: str) -> str:
    lines = [
        "# OpenAI Preprocessing Comparison",
        "",
        f"- model: `{model}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "| case | source | effort | status | sentences | splits | boundary types | tokens | seconds |",
        "|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        boundary_types = row.get("boundary_types") or {}
        btxt = ", ".join(f"{key}:{value}" for key, value in sorted(boundary_types.items())) or "-"
        usage = row.get("usage") or {}
        tokens = usage.get("total_tokens") or usage.get("total") or "-"
        elapsed = row.get("elapsed_seconds")
        seconds = f"{elapsed:.1f}" if isinstance(elapsed, (int, float)) else "-"
        lines.append(
            "| {case_id} | {source_type}<br>`{source_path}` | {effort} | {status} | {sentences} | {splits} | {btypes} | {tokens} | {seconds} |".format(
                case_id=row["case_id"],
                source_type=row["source_type"],
                source_path=row["source_path"],
                effort=row["reasoning_effort"],
                status=row["status"],
                sentences=row["sentence_count"],
                splits=row.get("split_count", "-"),
                btypes=btxt.replace("|", "\\|"),
                tokens=tokens,
                seconds=seconds,
            )
        )
    lines.extend(["", "## Notes", ""])
    for row in rows:
        lines.append(f"### {row['case_id']}")
        lines.append("")
        if row.get("error"):
            lines.append(f"- error: `{row['error']}`")
        lines.append(f"- extraction notes: {'; '.join(row.get('extraction_notes', []))}")
        if row.get("removed_foreign_paragraphs"):
            lines.append(f"- removed_foreign_paragraphs: {row['removed_foreign_paragraphs']}")
        for flag in row.get("quality_flags", []):
            lines.append(f"- quality flag: {flag}")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Implement OpenAI client**

Create `src/sermon_pipeline/openai_client.py`:

```python
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .io import extract_output_text


def call_openai(payload: dict[str, Any], api_key: str, timeout: int) -> tuple[dict[str, Any], str, float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {error_body[:3000]}") from exc
    elapsed = time.time() - start
    return data, extract_output_text(data), elapsed
```

- [ ] **Step 5: Implement CLI runner**

Create `src/sermon_pipeline/cli.py`:

```python
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .extractors.datalab import parse_datalab_json
from .extractors.docx import parse_docx
from .extractors.hwp import load_libhwp_reader, parse_hwp
from .io import load_dotenv, render_comparison, summarize_annotation, write_json
from .models import PreparedDocument
from .openai_client import call_openai
from .teacher import build_payload


def select_datalab_json(root: Path) -> Path:
    for path in sorted((root / "datas" / "datalab_parsed").glob("**/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("success") is True and data.get("html"):
            return path
    raise FileNotFoundError("No usable datalab_parsed JSON found.")


def select_docx(root: Path) -> Path:
    docx_dir = root / "datas" / "docx"
    for path in sorted(docx_dir.glob("*.docx")):
        if "세마포" in path.name and "10" in path.name:
            return path
    candidates = sorted(docx_dir.glob("*.docx"))
    if not candidates:
        raise FileNotFoundError("No DOCX files found.")
    return candidates[0]


def select_hwp(root: Path) -> Path:
    candidates = sorted((root / "datas" / "hwps").glob("*.hwp"))
    if not candidates:
        raise FileNotFoundError("No HWP files found.")
    return candidates[0]


def prepare_sample_documents(
    root: Path,
    max_sentences: int,
    hwp_reader_factory: Callable[[str], Any] = load_libhwp_reader,
) -> list[PreparedDocument]:
    return [
        parse_datalab_json(select_datalab_json(root), root=root, document_id="datalab_sample", max_sentences=max_sentences),
        parse_docx(select_docx(root), root=root, document_id="docx_sample", max_sentences=max_sentences),
        parse_hwp(select_hwp(root), root=root, document_id="hwp_sample", reader_factory=hwp_reader_factory, max_sentences=max_sentences),
    ]


def _case_id(document: PreparedDocument) -> str:
    if document.source_type == "datalab_parsed_json":
        return "datalab_xhigh"
    if document.source_type == "docx":
        return "docx_high"
    return "hwp_high"


def run(
    root: Path,
    out_dir: Path,
    model: str,
    max_sentences: int,
    max_output_tokens: int,
    timeout: int,
    dry_run: bool,
    hwp_reader_factory: Callable[[str], Any] = load_libhwp_reader,
) -> int:
    input_dir = out_dir / "inputs"
    response_dir = out_dir / "responses"
    annotation_dir = out_dir / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    documents = prepare_sample_documents(root, max_sentences, hwp_reader_factory)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    rows: list[dict[str, Any]] = []

    for document in documents:
        case_id = _case_id(document)
        payload = build_payload(document, model=model, max_output_tokens=max_output_tokens)
        write_json(input_dir / f"{case_id}.payload.json", payload)
        write_json(
            input_dir / f"{case_id}.sentences.json",
            {
                "source_path": document.source_path,
                "source_type": document.source_type,
                "reasoning_effort": document.reasoning_effort,
                "sentences": [sentence.to_payload() for sentence in document.sentences],
            },
        )
        row: dict[str, Any] = {
            "case_id": case_id,
            "source_type": document.source_type,
            "source_path": document.source_path,
            "reasoning_effort": document.reasoning_effort,
            "effort_label": document.effort_label,
            "sentence_count": len(document.sentences),
            "extraction_notes": document.extraction_notes,
            "removed_foreign_paragraphs": document.removed_foreign_paragraphs,
            "status": "prepared",
        }
        if dry_run or not api_key:
            row["status"] = "skipped"
            row["error"] = "dry_run" if dry_run else "missing_OPENAI_API_KEY"
            rows.append(row)
            continue
        try:
            response, output_text, elapsed = call_openai(payload, api_key, timeout)
            write_json(response_dir / f"{case_id}.raw_response.json", response)
            (response_dir / f"{case_id}.output_text.txt").write_text(output_text, encoding="utf-8")
            if not output_text.strip():
                incomplete = response.get("incomplete_details") or {}
                raise RuntimeError(
                    "empty output_text; "
                    f"response_status={response.get('status')}; "
                    f"incomplete_reason={incomplete.get('reason')}; "
                    f"usage={response.get('usage')}"
                )
            annotation = json.loads(output_text)
            write_json(response_dir / f"{case_id}.output.json", annotation)
            write_json(annotation_dir / f"{case_id}.annotation.json", annotation)
            row.update(summarize_annotation(annotation))
            row["status"] = "ok"
            row["usage"] = response.get("usage", {})
            row["elapsed_seconds"] = elapsed
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

    write_json(out_dir / "run_summary.json", rows)
    (out_dir / "comparison.md").write_text(render_comparison(rows, model), encoding="utf-8")
    return 0 if all(row["status"] in {"ok", "skipped"} for row in rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-sentences", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "tests" / "results" / run_id
    status = run(
        root=root,
        out_dir=out_dir,
        model=args.model,
        max_sentences=args.max_sentences,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    print(out_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 6: Replace compatibility smoke script**

Replace `tests/openai_preprocessing_comparison.py` with:

```python
#!/usr/bin/env python3
"""Compatibility runner for the source-aware teacher annotation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sermon_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 7: Update test README**

Replace `tests/README.md` with:

````markdown
# OpenAI preprocessing comparison test

This project now exposes reusable preprocessing code under `src/sermon_pipeline`.

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Prepare payloads without calling OpenAI:

```bash
PYTHONPATH=src python3 tests/openai_preprocessing_comparison.py --dry-run --max-sentences 16
```

Run the OpenAI smoke comparison:

```bash
PYTHONPATH=src python3 tests/openai_preprocessing_comparison.py --max-sentences 16
```

The script loads `OPENAI_API_KEY` from `.env` when the variable is not already present.
It writes payloads, raw responses, parsed annotations, and `comparison.md` under `tests/results/<timestamp>/`.
````

- [ ] **Step 8: Run CLI smoke test**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_cli_smoke -v
```

Expected: PASS with 1 test.

- [ ] **Step 9: Run full unit test suite**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

Expected: PASS for `test_models`, `test_text`, `test_sentence_splitter`, `test_extractors`, `test_teacher_payload`, `test_validation`, and `test_cli_smoke`.

- [ ] **Step 10: Commit**

```bash
git add src/sermon_pipeline/io.py src/sermon_pipeline/openai_client.py src/sermon_pipeline/cli.py tests/openai_preprocessing_comparison.py tests/README.md tests/test_cli_smoke.py
git commit -m "feat: add annotation pipeline cli"
```

### Task 10: Local Dry Run Against Repository Data

**Files:**
- Modify: no source files unless a test failure exposes a concrete bug.
- Generated: `tests/results/<timestamp>/inputs/*.payload.json`
- Generated: `tests/results/<timestamp>/comparison.md`

- [ ] **Step 1: Run dry run against real local data**

Run:

```bash
PYTHONPATH=src python3 tests/openai_preprocessing_comparison.py --dry-run --max-sentences 8
```

Expected: command prints a path like `tests/results/20260528_153000`, exits `0`, and writes these files:

```text
tests/results/<timestamp>/inputs/datalab_xhigh.payload.json
tests/results/<timestamp>/inputs/docx_high.payload.json
tests/results/<timestamp>/inputs/hwp_high.payload.json
tests/results/<timestamp>/comparison.md
tests/results/<timestamp>/run_summary.json
```

- [ ] **Step 2: Inspect DATAlab payload for strategy contract**

Run:

```bash
export LATEST="$(ls -td tests/results/* | head -1)"
python3 - <<'PY'
import json
import os
from pathlib import Path

latest = Path(os.environ["LATEST"])
payload = json.loads((latest / "inputs" / "datalab_xhigh.payload.json").read_text(encoding="utf-8"))
task = json.loads(payload["input"][1]["content"])
print(task["source_type"])
print(task["instructions"])
print(task["sentences"][0].keys())
PY
```

Expected output includes:

```text
datalab_parsed_json
HTML tag and heading_context are boundary hints, not gold labels.
dict_keys
```

- [ ] **Step 3: Inspect DOCX payload for removed paragraph count**

Run:

```bash
export LATEST="$(ls -td tests/results/* | head -1)"
python3 - <<'PY'
import json
import os
from pathlib import Path

latest = Path(os.environ["LATEST"])
payload = json.loads((latest / "inputs" / "docx_high.payload.json").read_text(encoding="utf-8"))
task = json.loads(payload["input"][1]["content"])
print(task["source_type"])
print(task["removed_foreign_paragraphs"])
print(task["extraction_notes"])
PY
```

Expected output includes:

```text
docx
```

The second printed line should be an integer greater than or equal to `0`.

- [ ] **Step 4: Inspect HWP payload for extraction notes**

Run:

```bash
export LATEST="$(ls -td tests/results/* | head -1)"
python3 - <<'PY'
import json
import os
from pathlib import Path

latest = Path(os.environ["LATEST"])
payload = json.loads((latest / "inputs" / "hwp_high.payload.json").read_text(encoding="utf-8"))
task = json.loads(payload["input"][1]["content"])
print(task["source_type"])
print(task["extraction_notes"])
PY
```

Expected output includes:

```text
hwp
HWP parsed with libhwp.
```

- [ ] **Step 5: Commit dry-run artifacts only if the team wants a new checked-in fixture**

Default action: do not commit `tests/results/<timestamp>` generated by this run. Commit only source and docs changes:

```bash
git status --short
```

Expected: generated `tests/results/<timestamp>` appears untracked or modified only if the run produced files. Leave it uncommitted unless a reviewer asks for a fixture update.

## Self-Review

**Spec coverage:**  
- Document-type input strategy is covered by Tasks 4, 5, 6, 7, and 10.  
- LLM instruction construction is covered by Task 7.  
- Strict output contract is covered by Task 7.  
- Output validation and training rows are covered by Task 8.  
- Existing smoke runner compatibility is covered by Task 9.  
- Real-data dry-run verification is covered by Task 10.

**Placeholder scan:**  
This plan uses concrete file paths, function names, test cases, commands, and expected outcomes. It avoids deferred implementation markers and broad unspecified error handling.

**Type consistency:**  
`PreparedDocument`, `SourceBlock`, `SentenceUnit`, `ValidationIssue`, and `TrainingRow` are introduced in Task 1 and reused with the same names and fields in later tasks. `build_payload`, `source_instructions`, `response_schema`, `validate_teacher_output`, and `teacher_output_to_training_rows` are referenced only after their defining tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-28-teacher-annotation-pipeline.md`. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, fast iteration

2. Inline Execution - execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
