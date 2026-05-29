from __future__ import annotations

import json
from typing import Any

from .constants import BOUNDARY_TYPES
from .teacher_windows import TeacherWindow


TEACHER_SYSTEM_PROMPT = (
    "You are a Korean sermon discourse-boundary annotation teacher. "
    "Annotate only target_local_sids. Metadata is available only as hints, "
    "not gold labels. Do not rewrite, summarize, translate, or normalize the "
    "sermon text. A boundary means a split after the sentence. Return strict "
    "JSON that matches the provided schema."
)

_TEACHER_INSTRUCTIONS = [
    "Return one annotation for every target_local_sid.",
    "Use boundary_type='none' when split_after=false.",
    "Use a non-none boundary_type when split_after=true.",
    "Keep scripture quotation/reference together.",
    "Treat metadata hints as hints, not gold labels.",
    "Do not annotate context sentences.",
    "Do not annotate a boundary when the next source sentence is not visible.",
]


def teacher_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "custom_id": {"type": "string"},
            "boundary_annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "local_sid": {"type": "string"},
                        "source_sentence_id": {"type": "string"},
                        "split_after": {"type": "boolean"},
                        "boundary_type": {
                            "type": "string",
                            "enum": list(BOUNDARY_TYPES),
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "local_sid",
                        "source_sentence_id",
                        "split_after",
                        "boundary_type",
                        "confidence",
                        "rationale",
                    ],
                },
            },
            "quality_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["custom_id", "boundary_annotations", "quality_flags"],
    }


def _render_teacher_user_content(window: TeacherWindow) -> str:
    task = window.to_teacher_task()
    task["anchored_text"] = [
        f"<{sentence.local_sid}> {sentence.text}" for sentence in window.sentences
    ]
    task["allowed_boundary_types"] = list(BOUNDARY_TYPES)
    task["instructions"] = list(_TEACHER_INSTRUCTIONS)
    return json.dumps(task, ensure_ascii=False, indent=2)


def build_teacher_payload(
    window: TeacherWindow,
    model: str = "gpt-5.5",
    max_output_tokens: int = 8192,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": _render_teacher_user_content(window)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "sermon_teacher_boundary_annotation",
                "schema": teacher_response_schema(),
                "strict": True,
            }
        },
        "max_output_tokens": max_output_tokens,
    }
