from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from sermon_pipeline.models import PreparedDocument, SourceBlock
from sermon_pipeline.sentence_splitter import split_blocks_into_sentences
from sermon_pipeline.text import is_layout_noise, normalize_ws


class DatalabHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[SourceBlock] = []
        self.page_id: str | None = None
        self.heading_context: list[str] = []
        self.stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "div" and attrs_dict.get("data-page-id") is not None:
            self.page_id = attrs_dict.get("data-page-id")
        if tag in {"h1", "h2", "h3", "p", "li", "td", "th"}:
            self.stack.append(
                {
                    "tag": tag,
                    "page_id": self.page_id,
                    "heading_context": list(self.heading_context[-3:]),
                    "parts": [],
                }
            )
        if tag == "br" and self.stack:
            self.stack[-1]["parts"].append("\n")

    def handle_data(self, data: str) -> None:
        if self.stack:
            self.stack[-1]["parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or tag != self.stack[-1]["tag"]:
            return
        current = self.stack.pop()
        parts = current["parts"]
        text = normalize_ws(" ".join(str(part) for part in parts))
        source_tag = str(current["tag"])
        if text and not is_layout_noise(text):
            block = SourceBlock(
                block_id=f"html.b{len(self.blocks):04d}",
                text=text,
                block_type="heading" if source_tag.startswith("h") else "paragraph",
                source_tag=source_tag,
                page_id=current["page_id"],
                heading_context=list(current["heading_context"]),
            )
            self.blocks.append(block)
            if source_tag.startswith("h"):
                self.heading_context.append(text)
        if self.stack:
            self.stack[-1]["parts"].append(text)


def _mark_html_boundaries(blocks: list[SourceBlock]) -> list[SourceBlock]:
    marked: list[SourceBlock] = []
    for idx, block in enumerate(blocks):
        previous_tag = blocks[idx - 1].source_tag if idx else None
        boundary = (
            idx == 0
            or previous_tag in {"h1", "h2", "h3"}
            or block.source_tag in {"h1", "h2", "h3"}
        )
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
    sentences = split_blocks_into_sentences(
        blocks, document_id, max_sentences=max_sentences
    )
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
