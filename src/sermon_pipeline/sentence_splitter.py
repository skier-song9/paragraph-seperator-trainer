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
            if max_sentences is not None and len(sentences) >= max_sentences:
                return sentences
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
    return sentences
