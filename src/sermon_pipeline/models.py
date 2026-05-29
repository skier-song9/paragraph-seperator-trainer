from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .constants import BOUNDARY_TYPES


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _payload_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _payload_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload_value(item) for item in value]
    return value


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading_context", tuple(self.heading_context))
        object.__setattr__(
            self, "script_counts", _freeze_json_value(self.script_counts)
        )

    def to_payload(self) -> dict[str, Any]:
        data = {
            "block_id": self.block_id,
            "text": self.text,
            "block_type": self.block_type,
            "source_tag": self.source_tag,
            "page_id": self.page_id,
            "paragraph_index": self.paragraph_index,
            "heading_context": self.heading_context,
            "html_boundary_before": self.html_boundary_before,
            "language_filter_reason": self.language_filter_reason,
            "script_counts": self.script_counts,
            "section_id": self.section_id,
        }
        return {
            key: _payload_value(value)
            for key, value in data.items()
            if value is not None
        }


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading_context", tuple(self.heading_context))

    def to_payload(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "text": self.text,
            "block_id": self.block_id,
            "block_type": self.block_type,
            "source_tag": self.source_tag,
            "page_id": self.page_id,
            "paragraph_index": self.paragraph_index,
            "heading_context": _payload_value(self.heading_context),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "extraction_notes", tuple(self.extraction_notes))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "sentences", tuple(self.sentences))

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

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "sentence_id": self.sentence_id,
        }


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _freeze_json_value(self.features))

    def to_payload(self) -> dict[str, Any]:
        return {
            "left_sentence_id": self.left_sentence_id,
            "right_sentence_id": self.right_sentence_id,
            "split_after_left": self.split_after_left,
            "boundary_type": self.boundary_type,
            "teacher_confidence": self.teacher_confidence,
            "source_type": self.source_type,
            "document_kind": self.document_kind,
            "features": _payload_value(self.features),
            "review_status": self.review_status,
        }
