from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sermon_pipeline.models import PreparedDocument, SourceBlock
from sermon_pipeline.sentence_splitter import split_blocks_into_sentences
from sermon_pipeline.text import normalize_ws


def load_libhwp_reader(path: str) -> Any:
    import libhwp

    return libhwp.HWPReader(path)


def extract_hwp_paragraphs(
    path: Path, reader_factory: Callable[[str], Any] = load_libhwp_reader
) -> tuple[list[str], str]:
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
    sentences = split_blocks_into_sentences(
        blocks, document_id, max_sentences=max_sentences
    )
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
