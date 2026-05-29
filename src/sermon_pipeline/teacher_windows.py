from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable


@dataclass(frozen=True)
class SentenceRecord:
    document_id: str
    sentence_id: str
    sentence_index: int
    text: str
    source_path: str
    source_type: str
    document_kind: str
    block_type: str
    source_tag: str
    page_id: str | None
    paragraph_index: int | None
    heading_context: tuple[str, ...]
    html_boundary_before: bool

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SentenceRecord:
        source_type = row.get("document_source_type", row.get("source_type"))
        heading_context = row.get("heading_context", ())
        if heading_context is None:
            heading_context = ()
        return cls(
            document_id=str(row["document_id"]),
            sentence_id=str(row["sentence_id"]),
            sentence_index=int(row["sentence_index"]),
            text=str(row["text"]),
            source_path=str(row["source_path"]),
            source_type=str(source_type),
            document_kind=str(row["document_kind"]),
            block_type=str(row["block_type"]),
            source_tag=str(row["source_tag"]),
            page_id=None if row.get("page_id") is None else str(row.get("page_id")),
            paragraph_index=(
                None
                if row.get("paragraph_index") is None
                else int(row.get("paragraph_index"))
            ),
            heading_context=tuple(str(item) for item in heading_context),
            html_boundary_before=bool(row.get("html_boundary_before", False)),
        )


@dataclass(frozen=True)
class WindowSentence:
    local_sid: str
    source_sentence_id: str
    global_sentence_index: int
    text: str
    role: str
    hints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hints", _freeze_hints(self.hints))

    def to_teacher_payload(self) -> dict[str, Any]:
        return {
            "local_sid": self.local_sid,
            "source_sentence_id": self.source_sentence_id,
            "role": self.role,
            "text": self.text,
            "hints": _json_copy(self.hints),
        }

    def to_mapping_payload(self) -> dict[str, Any]:
        return {
            "source_sentence_id": self.source_sentence_id,
            "global_sentence_index": self.global_sentence_index,
            "role": self.role,
        }


@dataclass(frozen=True)
class TeacherWindow:
    custom_id: str
    document_id: str
    source_path: str
    source_type: str
    document_kind: str
    window_index: int
    target_start: int
    target_end: int
    visible_start: int
    visible_end: int
    sentences: tuple[WindowSentence, ...]
    target_local_sids: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "document_kind": self.document_kind,
            "window_index": self.window_index,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "visible_start": self.visible_start,
            "visible_end": self.visible_end,
            "target_local_sids": list(self.target_local_sids),
            "local_to_source": {
                sentence.local_sid: sentence.to_mapping_payload()
                for sentence in self.sentences
            },
        }

    def to_teacher_task(self) -> dict[str, Any]:
        return {
            "task": "annotate_sentence_boundaries",
            "custom_id": self.custom_id,
            "document": {
                "document_id": self.document_id,
                "source_path": self.source_path,
                "source_type": self.source_type,
                "document_kind": self.document_kind,
            },
            "annotation_scope": {
                "target_local_sids": list(self.target_local_sids),
                "rule": (
                    "Annotate split_after only for target sentences whose immediate "
                    "next source sentence is visible in this window."
                ),
            },
            "sentences": [
                sentence.to_teacher_payload() for sentence in self.sentences
            ],
        }


def load_sentence_records(path: Path) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("row must be a JSON object")
                records.append(SentenceRecord.from_row(row))
            except Exception as exc:
                raise ValueError(
                    f"Invalid sentence row at line {line_number}: {exc}"
                ) from exc
    return records


def build_teacher_windows(
    records: Iterable[SentenceRecord],
    target_size: int = 160,
    left_context: int = 20,
    right_context: int = 20,
) -> list[TeacherWindow]:
    if target_size <= 0:
        raise ValueError("target_size must be greater than zero")
    if left_context < 0:
        raise ValueError("left_context must be greater than or equal to zero")
    if right_context < 0:
        raise ValueError("right_context must be greater than or equal to zero")

    by_document: dict[str, list[SentenceRecord]] = defaultdict(list)
    for record in records:
        by_document[record.document_id].append(record)

    windows: list[TeacherWindow] = []
    for document_id in sorted(by_document):
        document_records = sorted(
            by_document[document_id], key=lambda record: record.sentence_index
        )
        for window_index, target_start in enumerate(
            range(0, len(document_records), target_size)
        ):
            target_end = min(target_start + target_size, len(document_records))
            visible_start = max(0, target_start - left_context)
            visible_end = min(len(document_records), target_end + right_context)
            visible_records = document_records[visible_start:visible_end]
            target_local_sids: list[str] = []
            sentences: list[WindowSentence] = []

            for offset, record in enumerate(visible_records):
                source_position = visible_start + offset
                local_index = offset + 1
                local_sid = f"S{local_index}"
                is_target = target_start <= source_position < target_end
                sentence = WindowSentence(
                    local_sid=local_sid,
                    source_sentence_id=record.sentence_id,
                    global_sentence_index=record.sentence_index,
                    text=record.text,
                    role="target" if is_target else "context",
                    hints={
                        "block_type": record.block_type,
                        "source_tag": record.source_tag,
                        "page_id": record.page_id,
                        "paragraph_index": record.paragraph_index,
                        "heading_context": list(record.heading_context),
                        "html_boundary_before": record.html_boundary_before,
                    },
                )
                sentences.append(sentence)
                if is_target and source_position + 1 < visible_end:
                    target_local_sids.append(local_sid)

            first_record = document_records[0]
            custom_id = (
                f"teacher:{document_id}:w{window_index:04d}:"
                f"{target_start}-{target_end}"
            )
            windows.append(
                TeacherWindow(
                    custom_id=custom_id,
                    document_id=document_id,
                    source_path=first_record.source_path,
                    source_type=first_record.source_type,
                    document_kind=first_record.document_kind,
                    window_index=window_index,
                    target_start=target_start,
                    target_end=target_end,
                    visible_start=visible_start,
                    visible_end=visible_end,
                    sentences=tuple(sentences),
                    target_local_sids=tuple(target_local_sids),
                )
            )
    return windows


def _freeze_hints(hints: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(value) for key, value in hints.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


def _json_copy(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_copy(item) for item in value]
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    return copy.deepcopy(value)
