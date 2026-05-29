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
    return list(COMMON_INSTRUCTIONS) + list(
        SOURCE_INSTRUCTION_DELTAS.get(source_type, [])
    )


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_id": {"type": "string"},
            "source_type": {"type": "string"},
            "reasoning_effort": {"type": "string"},
            "preprocessing_observations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "boundary_annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sentence_id": {"type": "string"},
                        "text_excerpt": {"type": "string"},
                        "split_after": {"type": "boolean"},
                        "boundary_type": {
                            "type": "string",
                            "enum": list(BOUNDARY_TYPES),
                        },
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
                    "required": [
                        "paragraph_id",
                        "sentence_ids",
                        "paragraph_role",
                        "reason",
                    ],
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


def build_payload(
    document: PreparedDocument,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    task = document.to_teacher_task(source_instructions(document.source_type))
    return {
        "model": model,
        "reasoning": {"effort": document.reasoning_effort},
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(task, ensure_ascii=False, indent=2),
            },
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
