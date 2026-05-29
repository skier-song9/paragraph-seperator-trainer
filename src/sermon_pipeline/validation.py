from __future__ import annotations

from typing import Any

from .constants import BOUNDARY_TYPES
from .models import PreparedDocument, SentenceUnit, TrainingRow, ValidationIssue


def validate_teacher_output(
    document: PreparedDocument, output: dict[str, Any]
) -> list[ValidationIssue]:
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
        issues.append(
            ValidationIssue(
                "missing_sentence_annotation",
                "sentence was not annotated",
                sentence_id,
            )
        )
    for sentence_id in sorted(annotated_set - expected):
        issues.append(
            ValidationIssue(
                "unknown_sentence_id",
                "annotation referenced an unknown sentence",
                sentence_id,
            )
        )

    for item in annotations:
        sentence_id = item.get("sentence_id")
        split_after = item.get("split_after")
        boundary_type = item.get("boundary_type")
        if boundary_type not in BOUNDARY_TYPES:
            issues.append(
                ValidationIssue(
                    "unknown_boundary_type",
                    f"invalid boundary_type={boundary_type}",
                    sentence_id,
                )
            )
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


def teacher_output_to_training_rows(
    document: PreparedDocument, output: dict[str, Any]
) -> list[TrainingRow]:
    annotations_by_id = {
        item["sentence_id"]: item for item in output.get("boundary_annotations", [])
    }
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
