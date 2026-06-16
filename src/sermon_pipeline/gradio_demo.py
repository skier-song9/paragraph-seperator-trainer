from __future__ import annotations

import argparse
import html
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .boundary_eval import parse_student_output
from .constants import BOUNDARY_TYPES
from .student_sft import SPLITS, STUDENT_LABELS
from .train_sft import (
    SFT_FAMILIES,
    LinearBoundaryClassifier,
    _extract_sentences,
    _feature_dict,
    _softmax,
    summarize_sft_dataset,
)

DEFAULT_DATASET_DIR = Path("outputs/verify-training-100epoch/student_sft")
DEFAULT_TRAIN_OUT_DIR = Path("outputs/verify-training-100epoch/train_sparse_100epoch")
OVERVIEW_HEADERS = [
    "#",
    "example_id",
    "document_id",
    "boundaries",
    "labels",
    "input_chars",
]
SENTENCE_HEADERS = [
    "SID",
    "Global",
    "Role",
    "Gold after",
    "Pred after",
    "Confidence",
    "Status",
    "Sentence",
]
BOUNDARY_HEADERS = ["SID", "Sentence #", "Boundary type"]
PREDICTION_HEADERS = [
    "SID",
    "Gold",
    "Pred",
    "Confidence",
    "Status",
    "Sentence",
]
COLOR_BY_LABEL = {
    "topic_shift": "#2563eb",
    "scripture_reading_start": "#7c3aed",
    "scripture_explanation_start": "#0891b2",
    "illustration_start": "#c2410c",
    "application_start": "#15803d",
    "prayer_or_closing": "#be123c",
    "enumeration_start": "#a16207",
    "none": "#64748b",
}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _path_signature(path: Path) -> tuple[str, int | None, int | None]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), None, None)
    return (str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=24)
def _load_examples_cached(
    path: str, mtime_ns: int | None, size: int | None
) -> tuple[dict[str, Any], ...]:
    del mtime_ns, size
    return tuple(_iter_jsonl(Path(path)))


def load_examples(dataset_dir: Path, family: str, split: str) -> list[dict[str, Any]]:
    if family not in SFT_FAMILIES:
        raise ValueError(f"unknown SFT family: {family}")
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    path = dataset_dir / family / f"{split}.jsonl"
    return [dict(row) for row in _load_examples_cached(*_path_signature(path))]


@lru_cache(maxsize=8)
def _load_mappings_cached(
    path: str, mtime_ns: int | None, size: int | None
) -> dict[str, dict[str, Any]]:
    del mtime_ns, size
    mappings: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(Path(path)):
        for key in ("sparse_example_id", "first_boundary_example_id"):
            example_id = row.get(key)
            if isinstance(example_id, str):
                mappings[example_id] = row
    return mappings


def load_mappings(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    path = dataset_dir / "mappings.jsonl"
    return dict(_load_mappings_cached(*_path_signature(path)))


@lru_cache(maxsize=4)
def _load_model_cached(
    path: str, mtime_ns: int | None, size: int | None
) -> tuple[LinearBoundaryClassifier | None, str]:
    del mtime_ns, size
    model_path = Path(path)
    if not model_path.exists():
        return None, f"model.json not found: {model_path}"
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "linear_boundary_classifier":
        return None, f"unsupported model_type: {payload.get('model_type')!r}"
    labels = payload.get("labels", list(BOUNDARY_TYPES))
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        return None, "model labels must be a string list"
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        return None, "model weights must be an object"
    model = LinearBoundaryClassifier(labels=tuple(labels))
    model.weights = {
        label: {
            str(feature): float(value)
            for feature, value in feature_weights.items()
            if isinstance(value, (int, float))
        }
        for label, feature_weights in weights.items()
        if isinstance(label, str) and isinstance(feature_weights, dict)
    }
    for label in model.labels:
        model.weights.setdefault(label, {})
    family = payload.get("metadata", {}).get("family", "unknown")
    return model, f"loaded model: {model_path} (family={family})"


def load_model(train_out_dir: Path) -> tuple[LinearBoundaryClassifier | None, str]:
    path = train_out_dir / "model.json"
    return _load_model_cached(*_path_signature(path))


def _sid_number(local_sid: str) -> int:
    if local_sid.startswith("S") and local_sid[1:].isdigit():
        return int(local_sid[1:])
    return 0


def _valid_sids(input_text: str) -> set[str]:
    return {local_sid for local_sid, _ in _extract_sentences(input_text)}


def _labels_from_output(output_text: str, valid_local_sids: set[str]) -> dict[str, str]:
    parsed = parse_student_output(output_text, valid_local_sids)
    labels = {local_sid: "none" for local_sid in valid_local_sids}
    for boundary in parsed.boundaries:
        labels[boundary.local_sid] = boundary.boundary_type
    return labels


def _boundary_rows(output_text: str, valid_local_sids: set[str]) -> list[list[Any]]:
    parsed = parse_student_output(output_text, valid_local_sids)
    return [
        [item.local_sid, item.sentence_number, item.boundary_type]
        for item in parsed.boundaries
    ]


def _boundary_labels(row: dict[str, Any]) -> list[str]:
    input_text = str(row.get("input", ""))
    output_text = str(row.get("output", ""))
    parsed = parse_student_output(output_text, _valid_sids(input_text))
    return [item.boundary_type for item in parsed.boundaries]


def _example_label(index: int, row: dict[str, Any]) -> str:
    labels = _boundary_labels(row)
    label_text = ",".join(sorted(set(labels))) if labels else "NO_BOUNDARY"
    document_id = str(row.get("document_id", ""))
    example_id = str(row.get("example_id", ""))
    return f"{index:04d} | {document_id} | {label_text} | {example_id}"


def _filter_examples(
    examples: list[dict[str, Any]],
    query: str,
    boundary_filter: str,
    label_filter: str,
    sort_order: str,
) -> list[dict[str, Any]]:
    query_text = query.strip().casefold()
    filtered: list[dict[str, Any]] = []
    for row in examples:
        labels = _boundary_labels(row)
        has_boundary = bool(labels)
        if boundary_filter == "boundary_only" and not has_boundary:
            continue
        if boundary_filter == "no_boundary_only" and has_boundary:
            continue
        if label_filter != "all" and label_filter not in labels:
            continue
        if query_text:
            haystack = "\n".join(
                str(row.get(key, ""))
                for key in ("example_id", "document_id", "input", "output")
            ).casefold()
            if query_text not in haystack:
                continue
        filtered.append(row)

    if sort_order == "input_chars_desc":
        filtered.sort(key=lambda row: len(str(row.get("input", ""))), reverse=True)
    elif sort_order == "input_chars_asc":
        filtered.sort(key=lambda row: len(str(row.get("input", ""))))
    elif sort_order == "boundary_count_desc":
        filtered.sort(key=lambda row: len(_boundary_labels(row)), reverse=True)
    elif sort_order == "boundary_count_asc":
        filtered.sort(key=lambda row: len(_boundary_labels(row)))
    else:
        filtered.sort(key=lambda row: str(row.get("example_id", "")))
    return filtered


def build_browser_state(
    dataset_dir: str | Path,
    family: str,
    split: str,
    query: str = "",
    boundary_filter: str = "all",
    label_filter: str = "all",
    sort_order: str = "example_id",
) -> dict[str, Any]:
    root = Path(dataset_dir)
    examples = load_examples(root, family, split)
    filtered = _filter_examples(
        examples=examples,
        query=query,
        boundary_filter=boundary_filter,
        label_filter=label_filter,
        sort_order=sort_order,
    )
    labels = [_example_label(index, row) for index, row in enumerate(filtered)]
    overview_rows = [
        [
            index,
            str(row.get("example_id", "")),
            str(row.get("document_id", "")),
            len(_boundary_labels(row)),
            ", ".join(sorted(set(_boundary_labels(row)))) or "NO_BOUNDARY",
            len(str(row.get("input", ""))),
        ]
        for index, row in enumerate(filtered)
    ]
    return {
        "dataset_dir": str(root),
        "family": family,
        "split": split,
        "examples": filtered,
        "choice_labels": labels,
        "overview_rows": overview_rows,
        "summary_html": render_dataset_summary(root, family, split, examples, filtered),
    }


def _metadata_for_example(
    dataset_dir: Path, row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    example_id = str(row.get("example_id", ""))
    mapping = load_mappings(dataset_dir).get(example_id)
    metadata = {
        "example_id": row.get("example_id"),
        "document_id": row.get("document_id"),
        "target_type": row.get("target_type"),
        "input_chars": len(str(row.get("input", ""))),
        "output_chars": len(str(row.get("output", ""))),
    }
    if mapping is not None:
        metadata.update(
            {
                "custom_id": mapping.get("custom_id"),
                "source_path": mapping.get("source_path"),
            }
        )
    return metadata, mapping


def _predict_labels(
    model: LinearBoundaryClassifier | None,
    sentences: list[tuple[str, str]],
) -> tuple[dict[str, str], dict[str, float]]:
    if model is None:
        return {}, {}
    predictions: dict[str, str] = {}
    confidence_by_sid: dict[str, float] = {}
    for local_sid, sentence in sentences:
        features = _feature_dict(sentence, _sid_number(local_sid))
        probabilities = _softmax(model.scores(features))
        predicted = max(model.labels, key=lambda label: probabilities[label])
        predictions[local_sid] = predicted
        confidence_by_sid[local_sid] = probabilities[predicted]
    return predictions, confidence_by_sid


def _status_for(gold: str, predicted: str | None) -> str:
    if predicted is None:
        return "-"
    if gold == predicted:
        return "match"
    if gold == "none":
        return "FP"
    if predicted == "none":
        return "FN"
    return "label_mismatch"


def _sentence_source(mapping: dict[str, Any] | None, local_sid: str) -> dict[str, Any]:
    if mapping is None:
        return {}
    local_to_source = mapping.get("local_to_source")
    if not isinstance(local_to_source, dict):
        return {}
    source = local_to_source.get(local_sid)
    return source if isinstance(source, dict) else {}


def build_example_detail(
    state: dict[str, Any],
    selected_choice: str | None,
    train_out_dir: str | Path,
) -> dict[str, Any]:
    examples = state.get("examples")
    if not isinstance(examples, list) or not examples:
        return _empty_detail("No examples match the current filters.")
    labels = state.get("choice_labels")
    if not isinstance(labels, list):
        labels = []
    try:
        selected_index = labels.index(selected_choice) if selected_choice else 0
    except ValueError:
        selected_index = 0
    row = examples[selected_index]
    dataset_dir = Path(str(state.get("dataset_dir", DEFAULT_DATASET_DIR)))
    metadata, mapping = _metadata_for_example(dataset_dir, row)

    input_text = str(row.get("input", ""))
    output_text = str(row.get("output", ""))
    sentences = _extract_sentences(input_text)
    valid_local_sids = {local_sid for local_sid, _ in sentences}
    gold_labels = _labels_from_output(output_text, valid_local_sids)
    model, model_status = load_model(Path(train_out_dir))
    predicted_labels, confidence_by_sid = _predict_labels(model, sentences)

    sentence_rows: list[list[Any]] = []
    prediction_rows: list[list[Any]] = []
    mismatch_count = 0
    for local_sid, sentence in sentences:
        source = _sentence_source(mapping, local_sid)
        gold = gold_labels.get(local_sid, "none")
        predicted = predicted_labels.get(local_sid)
        status = _status_for(gold, predicted)
        if status not in ("match", "-"):
            mismatch_count += 1
        confidence = confidence_by_sid.get(local_sid)
        confidence_text = f"{confidence:.3f}" if confidence is not None else ""
        sentence_rows.append(
            [
                local_sid,
                source.get("global_sentence_index", ""),
                source.get("role", ""),
                gold,
                predicted or "",
                confidence_text,
                status,
                sentence,
            ]
        )
        prediction_rows.append(
            [
                local_sid,
                gold,
                predicted or "",
                confidence_text,
                status,
                sentence,
            ]
        )

    boundary_rows = _boundary_rows(output_text, valid_local_sids)
    model_summary = render_model_summary(
        model_status=model_status,
        mismatch_count=mismatch_count,
        sentence_count=len(sentences),
        boundary_count=len(boundary_rows),
    )
    return {
        "metadata": metadata,
        "sentence_rows": sentence_rows,
        "sentence_html": render_sentence_html(sentence_rows),
        "boundary_rows": boundary_rows,
        "raw_input": input_text,
        "raw_output": output_text,
        "model_summary_html": model_summary,
        "prediction_rows": prediction_rows,
    }


def _empty_detail(message: str) -> dict[str, Any]:
    return {
        "metadata": {"message": message},
        "sentence_rows": [],
        "sentence_html": f"<p class='empty'>{html.escape(message)}</p>",
        "boundary_rows": [],
        "raw_input": "",
        "raw_output": "",
        "model_summary_html": f"<p class='empty'>{html.escape(message)}</p>",
        "prediction_rows": [],
    }


def render_dataset_summary(
    dataset_dir: Path,
    family: str,
    split: str,
    all_examples: list[dict[str, Any]],
    filtered_examples: list[dict[str, Any]],
) -> str:
    try:
        summary = summarize_sft_dataset(dataset_dir)
    except Exception as exc:
        summary = {"error": str(exc)}
    label_counts: dict[str, int] = {label: 0 for label in STUDENT_LABELS}
    no_boundary = 0
    for row in all_examples:
        labels = _boundary_labels(row)
        if labels:
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        else:
            no_boundary += 1

    total = len(all_examples)
    filtered = len(filtered_examples)
    cards = [
        _metric_card("Current split", f"{family} / {split}"),
        _metric_card("Examples", f"{filtered} / {total}"),
        _metric_card("No boundary", str(no_boundary)),
    ]
    if "total_examples" in summary:
        cards.extend(
            [
                _metric_card("All SFT rows", str(summary["total_examples"])),
                _metric_card("Avg input chars", f"{summary['avg_input_chars']:.0f}"),
                _metric_card("Boundary lines", str(summary["boundary_line_count"])),
            ]
        )
    else:
        cards.append(_metric_card("Summary error", str(summary.get("error", "unknown"))))

    bars = []
    max_count = max(label_counts.values(), default=0)
    for label, count in label_counts.items():
        width = 0 if max_count == 0 else int(count / max_count * 100)
        color = COLOR_BY_LABEL.get(label, "#475569")
        bars.append(
            "<div class='bar-row'>"
            f"<span>{html.escape(label)}</span>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width}%;"
            f"background:{color}'></div></div>"
            f"<strong>{count}</strong>"
            "</div>"
        )
    return (
        "<div class='summary-grid'>"
        + "".join(cards)
        + "</div>"
        + "<div class='summary-bars'>"
        + "".join(bars)
        + "</div>"
    )


def render_model_summary(
    model_status: str,
    mismatch_count: int,
    sentence_count: int,
    boundary_count: int,
) -> str:
    cards = [
        _metric_card("Model", model_status),
        _metric_card("Sentences", str(sentence_count)),
        _metric_card("Gold boundaries", str(boundary_count)),
        _metric_card("Mismatches", str(mismatch_count)),
    ]
    return "<div class='summary-grid'>" + "".join(cards) + "</div>"


def _metric_card(label: str, value: str) -> str:
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{html.escape(label)}</div>"
        f"<div class='metric-value'>{html.escape(value)}</div>"
        "</div>"
    )


def _badge(label: str, css_class: str = "") -> str:
    color = COLOR_BY_LABEL.get(label, "#475569")
    return (
        f"<span class='badge {css_class}' style='border-color:{color};"
        f"color:{color}'>{html.escape(label)}</span>"
    )


def render_sentence_html(sentence_rows: list[list[Any]]) -> str:
    if not sentence_rows:
        return "<p class='empty'>No sentence rows.</p>"
    rows = []
    for item in sentence_rows:
        local_sid, global_index, role, gold, predicted, confidence, status, sentence = item
        classes = ["sentence-row"]
        if status == "FP":
            classes.append("status-fp")
        elif status == "FN":
            classes.append("status-fn")
        elif status == "label_mismatch":
            classes.append("status-mismatch")
        elif gold != "none":
            classes.append("status-boundary")
        badges = [_badge(str(gold), "gold") if gold != "none" else ""]
        if predicted and predicted != gold:
            badges.append(_badge(str(predicted), "pred"))
        meta = f"{local_sid}"
        if global_index != "":
            meta += f" | global {global_index}"
        if role:
            meta += f" | {role}"
        if confidence:
            meta += f" | p={confidence}"
        rows.append(
            f"<div class='{' '.join(classes)}'>"
            f"<div class='sentence-meta'>{html.escape(meta)}</div>"
            f"<div class='sentence-text'>{html.escape(str(sentence))}</div>"
            f"<div class='sentence-badges'>{''.join(badges)}</div>"
            "</div>"
        )
    return "<div class='sentence-list'>" + "".join(rows) + "</div>"


def load_metrics(train_out_dir: Path) -> list[dict[str, Any]]:
    return _iter_jsonl(train_out_dir / "metrics.jsonl")


def load_training_summary(train_out_dir: Path) -> dict[str, Any]:
    path = train_out_dir / "train_run_summary.json"
    if not path.exists():
        return {"error": f"train_run_summary.json not found: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def render_metrics_html(train_out_dir: str | Path) -> str:
    root = Path(train_out_dir)
    rows = load_metrics(root)
    summary = load_training_summary(root)
    final_training = summary.get("training", {}) if isinstance(summary, dict) else {}
    final_validation = final_training.get("final_validation", {})
    final_test = final_training.get("final_test", {})
    cards: list[str] = []
    for label, metrics, prefix in (
        ("Validation F1", final_validation, "validation/f1"),
        ("Validation tol1 F1", final_validation, "validation/tolerance1_f1"),
        ("Test F1", final_test, "test/f1"),
        ("Test tol1 F1", final_test, "test/tolerance1_f1"),
    ):
        value = metrics.get(prefix) if isinstance(metrics, dict) else None
        cards.append(_metric_card(label, "-" if value is None else f"{float(value):.4f}"))
    return (
        "<div class='summary-grid'>"
        + "".join(cards)
        + "</div>"
        + _svg_chart(rows, ["train/loss", "validation/loss"], "Loss by epoch")
        + _svg_chart(rows, ["validation/f1", "validation/tolerance1_f1"], "Boundary F1 by epoch")
    )


def _svg_chart(rows: list[dict[str, Any]], keys: list[str], title: str) -> str:
    if not rows:
        return f"<p class='empty'>No metrics rows for {html.escape(title)}.</p>"
    width = 720
    height = 240
    left = 48
    right = 16
    top = 34
    bottom = 32
    plot_width = width - left - right
    plot_height = height - top - bottom
    epochs = [float(row.get("epoch", index + 1)) for index, row in enumerate(rows)]
    values = [
        float(row[key])
        for row in rows
        for key in keys
        if isinstance(row.get(key), (int, float))
    ]
    if not values:
        return f"<p class='empty'>No plottable series for {html.escape(title)}.</p>"
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        min_value -= 0.5
        max_value += 0.5

    def point(epoch: float, value: float) -> tuple[float, float]:
        if math.isclose(min_epoch, max_epoch):
            x = left + plot_width / 2
        else:
            x = left + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_width
        y = top + (max_value - value) / (max_value - min_value) * plot_height
        return x, y

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    lines = []
    legends = []
    for index, key in enumerate(keys):
        points = [
            point(float(row.get("epoch", pos + 1)), float(row[key]))
            for pos, row in enumerate(rows)
            if isinstance(row.get(key), (int, float))
        ]
        if not points:
            continue
        color = colors[index % len(colors)]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2.5' "
            f"points='{point_text}' />"
        )
        legends.append(
            f"<span class='legend-item'><span style='background:{color}'></span>"
            f"{html.escape(key)}</span>"
        )

    return (
        "<div class='chart-card'>"
        f"<div class='chart-title'>{html.escape(title)}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img'>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' class='axis' />"
        f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' y2='{top + plot_height}' class='axis' />"
        f"<text x='{left}' y='20' class='tick'>{max_value:.4f}</text>"
        f"<text x='{left}' y='{height - 8}' class='tick'>{min_value:.4f}</text>"
        + "".join(lines)
        + "</svg>"
        + "<div class='legend'>"
        + "".join(legends)
        + "</div></div>"
    )


def _refresh_ui(
    dataset_dir: str,
    train_out_dir: str,
    family: str,
    split: str,
    query: str,
    boundary_filter: str,
    label_filter: str,
    sort_order: str,
) -> tuple[Any, ...]:
    import gradio as gr

    try:
        state = build_browser_state(
            dataset_dir=dataset_dir,
            family=family,
            split=split,
            query=query,
            boundary_filter=boundary_filter,
            label_filter=label_filter,
            sort_order=sort_order,
        )
        choices = state["choice_labels"]
        selected = choices[0] if choices else None
        detail = build_example_detail(state, selected, train_out_dir)
        metrics_html = render_metrics_html(train_out_dir)
        training_summary = load_training_summary(Path(train_out_dir))
        return (
            state,
            state["summary_html"],
            gr.update(choices=choices, value=selected),
            state["overview_rows"],
            detail["metadata"],
            detail["sentence_html"],
            detail["sentence_rows"],
            detail["boundary_rows"],
            detail["raw_input"],
            detail["raw_output"],
            detail["model_summary_html"],
            detail["prediction_rows"],
            metrics_html,
            training_summary,
        )
    except Exception as exc:
        empty = _empty_detail(str(exc))
        return (
            {},
            f"<p class='empty error'>{html.escape(str(exc))}</p>",
            gr.update(choices=[], value=None),
            [],
            empty["metadata"],
            empty["sentence_html"],
            [],
            [],
            "",
            "",
            empty["model_summary_html"],
            [],
            f"<p class='empty error'>{html.escape(str(exc))}</p>",
            {"error": str(exc)},
        )


def _select_ui(
    state: dict[str, Any],
    selected_choice: str | None,
    train_out_dir: str,
) -> tuple[Any, ...]:
    try:
        detail = build_example_detail(state, selected_choice, train_out_dir)
        return (
            detail["metadata"],
            detail["sentence_html"],
            detail["sentence_rows"],
            detail["boundary_rows"],
            detail["raw_input"],
            detail["raw_output"],
            detail["model_summary_html"],
            detail["prediction_rows"],
        )
    except Exception as exc:
        empty = _empty_detail(str(exc))
        return (
            empty["metadata"],
            empty["sentence_html"],
            [],
            [],
            "",
            "",
            empty["model_summary_html"],
            [],
        )


def create_app() -> Any:
    import gradio as gr

    with gr.Blocks(
        title="Sermon Boundary Dataset Browser",
        fill_width=True,
    ) as demo:
        state = gr.State({})
        gr.Markdown("# Sermon Boundary Dataset Browser")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                dataset_dir = gr.Textbox(
                    label="SFT dataset dir",
                    value=str(DEFAULT_DATASET_DIR),
                    lines=1,
                )
                train_out_dir = gr.Textbox(
                    label="Training output dir",
                    value=str(DEFAULT_TRAIN_OUT_DIR),
                    lines=1,
                )
                family = gr.Radio(
                    choices=list(SFT_FAMILIES),
                    value="sparse_multi_boundary",
                    label="Dataset family",
                )
                split = gr.Radio(choices=list(SPLITS), value="train", label="Split")
                query = gr.Textbox(label="Search", placeholder="example_id, document_id, text")
                boundary_filter = gr.Dropdown(
                    choices=[
                        ("All", "all"),
                        ("Boundary only", "boundary_only"),
                        ("NO_BOUNDARY only", "no_boundary_only"),
                    ],
                    value="all",
                    label="Boundary filter",
                )
                label_filter = gr.Dropdown(
                    choices=[("All labels", "all")]
                    + [(label, label) for label in STUDENT_LABELS],
                    value="all",
                    label="Label filter",
                )
                sort_order = gr.Dropdown(
                    choices=[
                        ("Example ID", "example_id"),
                        ("Input length desc", "input_chars_desc"),
                        ("Input length asc", "input_chars_asc"),
                        ("Boundary count desc", "boundary_count_desc"),
                        ("Boundary count asc", "boundary_count_asc"),
                    ],
                    value="example_id",
                    label="Sort",
                )
                refresh = gr.Button("Reload", variant="primary")

            with gr.Column(scale=3):
                summary_html = gr.HTML()
                example_choice = gr.Dropdown(label="Example", choices=[])
                overview = gr.Dataframe(
                    headers=OVERVIEW_HEADERS,
                    datatype=["number", "str", "str", "number", "str", "number"],
                    interactive=False,
                    wrap=True,
                    label="Filtered examples",
                )
                with gr.Tabs():
                    with gr.Tab("Training Data"):
                        metadata = gr.JSON(label="Example metadata")
                        sentence_html = gr.HTML(label="Sentence view")
                        sentence_table = gr.Dataframe(
                            headers=SENTENCE_HEADERS,
                            interactive=False,
                            wrap=True,
                            label="Sentence table",
                        )
                        boundary_table = gr.Dataframe(
                            headers=BOUNDARY_HEADERS,
                            interactive=False,
                            wrap=True,
                            label="Parsed output boundaries",
                        )
                        with gr.Row():
                            raw_input = gr.Textbox(
                                label="Raw input",
                                lines=18,
                                max_lines=28,
                                buttons=["copy"],
                            )
                            raw_output = gr.Textbox(
                                label="Raw output",
                                lines=18,
                                max_lines=28,
                                buttons=["copy"],
                            )
                    with gr.Tab("Model Result"):
                        model_summary = gr.HTML()
                        prediction_table = gr.Dataframe(
                            headers=PREDICTION_HEADERS,
                            interactive=False,
                            wrap=True,
                            label="Gold vs prediction",
                        )
                    with gr.Tab("Metrics"):
                        metrics_html = gr.HTML()
                        training_summary = gr.JSON(label="train_run_summary.json")

        refresh_outputs = [
            state,
            summary_html,
            example_choice,
            overview,
            metadata,
            sentence_html,
            sentence_table,
            boundary_table,
            raw_input,
            raw_output,
            model_summary,
            prediction_table,
            metrics_html,
            training_summary,
        ]
        refresh_inputs = [
            dataset_dir,
            train_out_dir,
            family,
            split,
            query,
            boundary_filter,
            label_filter,
            sort_order,
        ]
        refresh.click(_refresh_ui, inputs=refresh_inputs, outputs=refresh_outputs)
        demo.load(_refresh_ui, inputs=refresh_inputs, outputs=refresh_outputs)
        for component in (family, split, boundary_filter, label_filter, sort_order):
            component.change(_refresh_ui, inputs=refresh_inputs, outputs=refresh_outputs)
        query.submit(_refresh_ui, inputs=refresh_inputs, outputs=refresh_outputs)
        example_choice.change(
            _select_ui,
            inputs=[state, example_choice, train_out_dir],
            outputs=[
                metadata,
                sentence_html,
                sentence_table,
                boundary_table,
                raw_input,
                raw_output,
                model_summary,
                prediction_table,
            ],
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    import gradio as gr

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)
    app = create_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        prevent_thread_lock=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        close = getattr(app, "close", None)
        if callable(close):
            close()
    return 0


CSS = """
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin: 6px 0 12px;
}
.metric-card {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  padding: 10px 12px;
  background: #ffffff;
}
.metric-label {
  color: #64748b;
  font-size: 12px;
  line-height: 1.2;
}
.metric-value {
  color: #0f172a;
  font-size: 15px;
  font-weight: 650;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.summary-bars {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  background: #ffffff;
  padding: 10px 12px;
  margin-bottom: 12px;
}
.bar-row {
  display: grid;
  grid-template-columns: minmax(170px, 230px) 1fr 48px;
  gap: 10px;
  align-items: center;
  font-size: 13px;
  margin: 6px 0;
}
.bar-track {
  height: 8px;
  background: #e5e7eb;
  border-radius: 999px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
}
.sentence-list {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  background: #ffffff;
  max-height: 620px;
  overflow: auto;
}
.sentence-row {
  display: grid;
  grid-template-columns: 160px 1fr minmax(180px, 260px);
  gap: 12px;
  border-bottom: 1px solid #edf2f7;
  padding: 9px 12px;
  align-items: start;
}
.sentence-row:last-child {
  border-bottom: 0;
}
.sentence-meta {
  color: #64748b;
  font-size: 12px;
  line-height: 1.35;
}
.sentence-text {
  color: #0f172a;
  line-height: 1.55;
  overflow-wrap: anywhere;
}
.sentence-badges {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.status-boundary {
  background: #f8fafc;
}
.status-fp {
  background: #fff7ed;
}
.status-fn {
  background: #fef2f2;
}
.status-mismatch {
  background: #f5f3ff;
}
.badge {
  border: 1px solid;
  border-radius: 999px;
  padding: 2px 7px;
  font-size: 12px;
  font-weight: 650;
  white-space: nowrap;
}
.badge.pred {
  border-style: dashed;
}
.empty {
  color: #64748b;
  border: 1px solid #d8dee8;
  border-radius: 8px;
  padding: 12px;
  background: #ffffff;
}
.error {
  color: #b91c1c;
}
.chart-card {
  border: 1px solid #d8dee8;
  border-radius: 8px;
  background: #ffffff;
  padding: 12px;
  margin: 12px 0;
}
.chart-title {
  font-weight: 650;
  color: #0f172a;
  margin-bottom: 6px;
}
.axis {
  stroke: #94a3b8;
  stroke-width: 1;
}
.tick {
  fill: #64748b;
  font-size: 11px;
}
.legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  color: #475569;
  font-size: 13px;
}
.legend-item {
  display: inline-flex;
  gap: 6px;
  align-items: center;
}
.legend-item span {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}
@media (max-width: 760px) {
  .sentence-row {
    grid-template-columns: 1fr;
  }
  .bar-row {
    grid-template-columns: 1fr;
  }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
