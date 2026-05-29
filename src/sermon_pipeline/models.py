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
