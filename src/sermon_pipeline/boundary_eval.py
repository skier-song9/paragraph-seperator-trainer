from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .student_sft import STUDENT_LABELS


@dataclass(frozen=True)
class BoundaryPrediction:
    local_sid: str
    sentence_number: int
    boundary_type: str


@dataclass(frozen=True)
class ParsedOutput:
    boundaries: list[BoundaryPrediction]
    issues: list[dict[str, Any]]


def parse_student_output(output: str, valid_local_sids: set[str]) -> ParsedOutput:
    text = output.strip()
    if not text:
        return ParsedOutput(
            boundaries=[],
            issues=[{"code": "invalid_line", "line_number": 1, "line": output}],
        )
    if text == "NO_BOUNDARY":
        return ParsedOutput(boundaries=[], issues=[])

    boundaries: list[BoundaryPrediction] = []
    issues: list[dict[str, Any]] = []
    seen_local_sids: set[str] = set()
    previous_sentence_number: int | None = None

    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            issues.append(
                {"code": "invalid_line", "line_number": line_number, "line": raw_line}
            )
            continue

        local_sid, boundary_type = parts
        sentence_number = _parse_local_sid(local_sid)
        if sentence_number is None:
            issues.append(
                {
                    "code": "invalid_line",
                    "line_number": line_number,
                    "line": raw_line,
                    "local_sid": local_sid,
                }
            )

        known_sid = local_sid in valid_local_sids
        if not known_sid:
            issues.append(
                {
                    "code": "unknown_local_sid",
                    "line_number": line_number,
                    "local_sid": local_sid,
                }
            )

        valid_label = boundary_type in STUDENT_LABELS
        if not valid_label:
            issues.append(
                {
                    "code": "invalid_label",
                    "line_number": line_number,
                    "boundary_type": boundary_type,
                }
            )

        duplicate = local_sid in seen_local_sids
        if duplicate:
            issues.append(
                {
                    "code": "duplicate_local_sid",
                    "line_number": line_number,
                    "local_sid": local_sid,
                }
            )

        if (
            sentence_number is not None
            and previous_sentence_number is not None
            and sentence_number <= previous_sentence_number
        ):
            issues.append(
                {
                    "code": "not_ascending",
                    "line_number": line_number,
                    "local_sid": local_sid,
                }
            )

        if sentence_number is not None:
            previous_sentence_number = sentence_number
        seen_local_sids.add(local_sid)

        if sentence_number is None or not known_sid or not valid_label or duplicate:
            continue
        boundaries.append(
            BoundaryPrediction(
                local_sid=local_sid,
                sentence_number=sentence_number,
                boundary_type=boundary_type,
            )
        )

    return ParsedOutput(boundaries=boundaries, issues=issues)


def boundary_f1(
    gold: set[int], predicted: set[int], tolerance: int = 0
) -> dict[str, float | int]:
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    unmatched_gold = sorted(gold)
    true_positive = 0
    for boundary in sorted(predicted):
        match_index = _closest_match_index(boundary, unmatched_gold, tolerance)
        if match_index is None:
            continue
        true_positive += 1
        unmatched_gold.pop(match_index)

    false_positive = len(predicted) - true_positive
    false_negative = len(gold) - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    if not gold and not predicted:
        precision = 1.0
        recall = 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def pk_score(sentence_count: int, gold: set[int], predicted: set[int]) -> float:
    if sentence_count <= 1:
        return 0.0

    window_size = _window_size(sentence_count, gold)
    comparisons = sentence_count - window_size
    if comparisons <= 0:
        return 0.0

    errors = 0
    for start in range(1, comparisons + 1):
        end = start + window_size
        gold_same = _boundary_count_between(gold, start, end) == 0
        predicted_same = _boundary_count_between(predicted, start, end) == 0
        if gold_same != predicted_same:
            errors += 1
    return errors / comparisons


def windowdiff_score(sentence_count: int, gold: set[int], predicted: set[int]) -> float:
    if sentence_count <= 1:
        return 0.0

    window_size = _window_size(sentence_count, gold)
    comparisons = sentence_count - window_size
    if comparisons <= 0:
        return 0.0

    errors = 0
    for start in range(1, comparisons + 1):
        end = start + window_size
        if _boundary_count_between(gold, start, end) != _boundary_count_between(
            predicted, start, end
        ):
            errors += 1
    return errors / comparisons


def _parse_local_sid(local_sid: str) -> int | None:
    if len(local_sid) < 2 or local_sid[0] != "S":
        return None
    suffix = local_sid[1:]
    if not suffix.isdecimal():
        return None
    return int(suffix)


def _closest_match_index(
    predicted: int, unmatched_gold: list[int], tolerance: int
) -> int | None:
    best_index: int | None = None
    best_distance: int | None = None
    for index, boundary in enumerate(unmatched_gold):
        distance = abs(predicted - boundary)
        if distance > tolerance:
            continue
        if best_distance is None or distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _window_size(sentence_count: int, gold: set[int]) -> int:
    segment_count = len(_valid_boundaries(sentence_count, gold)) + 1
    average_segment_length = sentence_count / segment_count
    return max(1, min(sentence_count - 1, round(average_segment_length / 2)))


def _boundary_count_between(boundaries: set[int], start: int, end: int) -> int:
    return sum(1 for boundary in boundaries if start <= boundary < end)


def _valid_boundaries(sentence_count: int, boundaries: set[int]) -> set[int]:
    return {boundary for boundary in boundaries if 1 <= boundary < sentence_count}
