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
    sentences = split_blocks_into_sentences(
        blocks, document_id, max_sentences=max_sentences
    )
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
