from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .boundary_eval import boundary_f1, parse_student_output
from .constants import BOUNDARY_TYPES
from .io import load_dotenv, write_json
from .student_sft import SPLITS

SFT_FAMILIES = ("sparse_multi_boundary", "first_boundary")
WANDB_MODES = ("disabled", "offline", "online")
MODEL_LABELS = BOUNDARY_TYPES
TEXT_LINE_RE = re.compile(r"^<(?P<sid>S\d+)>\s*(?P<text>.*)$")


class WandbRunHandle:
    def __init__(self, run: Any | None, status: dict[str, Any]) -> None:
        self.run = run
        self.status = status

    def log(self, metrics: dict[str, int | float], step: int | None = None) -> None:
        if self.run is not None and metrics:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


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


def _boundary_line_count(output_text: str) -> int:
    stripped = output_text.strip()
    if not stripped or stripped == "NO_BOUNDARY":
        return 0
    return len([line for line in stripped.splitlines() if line.strip()])


def summarize_sft_dataset(dataset_dir: Path) -> dict[str, Any]:
    by_family_split = {
        family: {split: 0 for split in SPLITS} for family in SFT_FAMILIES
    }
    by_target_type = {family: 0 for family in SFT_FAMILIES}
    total_examples = 0
    no_boundary_count = 0
    boundary_line_count = 0
    input_char_total = 0
    output_char_total = 0

    for family in SFT_FAMILIES:
        for split in SPLITS:
            path = dataset_dir / family / f"{split}.jsonl"
            rows = _iter_jsonl(path)
            by_family_split[family][split] = len(rows)
            by_target_type[family] += len(rows)
            for row in rows:
                output_text = row.get("output")
                input_text = row.get("input")
                if not isinstance(output_text, str) or not isinstance(input_text, str):
                    raise ValueError(f"{path}: examples must include string input/output")
                total_examples += 1
                input_char_total += len(input_text)
                output_char_total += len(output_text)
                lines = _boundary_line_count(output_text)
                boundary_line_count += lines
                if lines == 0:
                    no_boundary_count += 1

    avg_input_chars = input_char_total / total_examples if total_examples else 0.0
    avg_output_chars = output_char_total / total_examples if total_examples else 0.0
    return {
        "dataset_dir": str(dataset_dir),
        "total_examples": total_examples,
        "by_family_split": by_family_split,
        "by_target_type": by_target_type,
        "boundary_example_count": total_examples - no_boundary_count,
        "no_boundary_count": no_boundary_count,
        "boundary_line_count": boundary_line_count,
        "avg_input_chars": avg_input_chars,
        "avg_output_chars": avg_output_chars,
    }


def flatten_wandb_metrics(summary: dict[str, Any]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[f"dataset/{key}"] = value

    by_family_split = summary.get("by_family_split", {})
    if isinstance(by_family_split, dict):
        for family, split_counts in by_family_split.items():
            if not isinstance(family, str) or not isinstance(split_counts, dict):
                continue
            for split, count in split_counts.items():
                if isinstance(split, str) and isinstance(count, int):
                    metrics[f"dataset/{family}/{split}_examples"] = count
    return metrics


def _wandb_backend(backend: Any | None = None) -> Any:
    if backend is not None:
        return backend
    try:
        import wandb as imported_backend
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Run with the training extra: "
            "uv run --extra train sermon-sft-train-smoke ..."
        ) from exc
    return imported_backend


def start_wandb_run(
    dataset_summary: dict[str, Any],
    project: str,
    run_name: str,
    mode: str,
    config: dict[str, Any] | None = None,
    backend: Any | None = None,
) -> WandbRunHandle:
    if mode not in WANDB_MODES:
        raise ValueError(f"unknown W&B mode: {mode}")
    api_key = os.environ.get("WANDB_API_KEY")
    status = {
        "mode": mode,
        "project": project,
        "run_name": run_name,
        "api_key_present": bool(api_key),
    }
    if mode == "disabled":
        status["enabled"] = False
        return WandbRunHandle(run=None, status=status)

    backend = _wandb_backend(backend)

    if mode == "online":
        if not api_key:
            raise RuntimeError("WANDB_API_KEY is required for online W&B logging")
        backend.login(key=api_key, relogin=False)

    run = backend.init(
        project=project,
        name=run_name,
        mode=mode,
        config={
            "dataset_dir": dataset_summary.get("dataset_dir"),
            "sft_families": list(SFT_FAMILIES),
            "splits": list(SPLITS),
            **(config or {}),
        },
    )
    status["enabled"] = True
    return WandbRunHandle(run=run, status=status)


def log_wandb_summary(
    summary: dict[str, Any],
    project: str,
    run_name: str,
    mode: str,
    backend: Any | None = None,
) -> dict[str, Any]:
    handle = start_wandb_run(
        dataset_summary=summary,
        project=project,
        run_name=run_name,
        mode=mode,
        backend=backend,
    )
    handle.log(flatten_wandb_metrics(summary), step=0)
    handle.finish()
    return handle.status


def _sid_number(local_sid: str) -> int:
    if local_sid.startswith("S") and local_sid[1:].isdigit():
        return int(local_sid[1:])
    return 0


def _extract_sentences(input_text: str) -> list[tuple[str, str]]:
    in_text = False
    sentences: list[tuple[str, str]] = []
    for raw_line in input_text.splitlines():
        line = raw_line.strip()
        if line == "<TEXT>":
            in_text = True
            continue
        if not in_text:
            continue
        match = TEXT_LINE_RE.match(line)
        if match is None:
            continue
        sentences.append((match.group("sid"), match.group("text")))
    sentences.sort(key=lambda item: _sid_number(item[0]))
    return sentences


def _target_labels(output_text: str, valid_local_sids: set[str]) -> dict[str, str]:
    parsed = parse_student_output(output_text, valid_local_sids)
    if parsed.issues:
        raise ValueError(f"invalid SFT output {output_text!r}: {parsed.issues}")
    labels = {local_sid: "none" for local_sid in valid_local_sids}
    for boundary in parsed.boundaries:
        labels[boundary.local_sid] = boundary.boundary_type
    return labels


def _load_examples(dataset_dir: Path, family: str, split: str) -> list[dict[str, Any]]:
    if family not in SFT_FAMILIES:
        raise ValueError(f"unknown SFT family: {family}")
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    return _iter_jsonl(dataset_dir / family / f"{split}.jsonl")


def _load_candidates(dataset_dir: Path, family: str, split: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in _load_examples(dataset_dir, family, split):
        input_text = row.get("input")
        output_text = row.get("output")
        example_id = row.get("example_id")
        if not isinstance(input_text, str) or not isinstance(output_text, str):
            raise ValueError(f"{dataset_dir}: examples must include string input/output")
        if not isinstance(example_id, str):
            raise ValueError(f"{dataset_dir}: examples must include string example_id")
        sentences = _extract_sentences(input_text)
        if not sentences:
            raise ValueError(f"{example_id}: input contains no <Sx> sentence lines")
        valid_sids = {local_sid for local_sid, _ in sentences}
        labels = _target_labels(output_text, valid_sids)
        for local_sid, text in sentences:
            candidates.append(
                {
                    "example_id": example_id,
                    "document_id": str(row.get("document_id", "")),
                    "local_sid": local_sid,
                    "sentence_number": _sid_number(local_sid),
                    "text": text,
                    "label": labels[local_sid],
                }
            )
    return candidates


def _feature_dict(text: str, sentence_number: int) -> dict[str, float]:
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    features: dict[str, float] = {
        "bias": 1.0,
        f"sid_bucket:{min(sentence_number, 20)}": 1.0,
        f"len_bucket:{min(len(normalized) // 20, 10)}": 1.0,
    }
    if sentence_number == 1:
        features["sid:first"] = 1.0
    if normalized.startswith("제"):
        features["cue:starts_je"] = 1.0
    if lowered.startswith("chapter "):
        features["cue:starts_chapter"] = 1.0
    for cue in ("기도", "아멘", "다음과 같이 표현합니다", "말씀", "본문"):
        if cue in normalized:
            features[f"cue:contains:{cue}"] = 1.0

    chars = normalized[:256]
    for char in chars:
        if char.isspace():
            continue
        key = f"char:{char}"
        features[key] = min(features.get(key, 0.0) + 1.0, 4.0)
    for index in range(max(0, len(chars) - 1)):
        bigram = chars[index : index + 2]
        if bigram.strip():
            key = f"bigram:{bigram}"
            features[key] = min(features.get(key, 0.0) + 1.0, 3.0)
    return features


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exp_scores.values())
    return {label: score / total for label, score in exp_scores.items()}


class LinearBoundaryClassifier:
    def __init__(self, labels: tuple[str, ...] = MODEL_LABELS) -> None:
        self.labels = labels
        self.weights: dict[str, dict[str, float]] = {label: {} for label in labels}

    def scores(self, features: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for label in self.labels:
            weights = self.weights[label]
            score = 0.0
            for feature, value in features.items():
                score += weights.get(feature, 0.0) * value
            scores[label] = score
        return scores

    def predict_label(self, features: dict[str, float]) -> str:
        scores = self.scores(features)
        return max(self.labels, key=lambda label: (scores[label], label == "none"))

    def train_candidate(
        self,
        features: dict[str, float],
        gold_label: str,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[float, str]:
        if gold_label not in self.weights:
            raise ValueError(f"unknown training label: {gold_label}")
        probabilities = _softmax(self.scores(features))
        prediction = max(self.labels, key=lambda label: probabilities[label])
        loss = -math.log(max(probabilities[gold_label], 1e-12))
        for label in self.labels:
            diff = (1.0 if label == gold_label else 0.0) - probabilities[label]
            weights = self.weights[label]
            for feature, value in features.items():
                if weight_decay:
                    weights[feature] = weights.get(feature, 0.0) * (
                        1.0 - learning_rate * weight_decay
                    )
                weights[feature] = weights.get(feature, 0.0) + learning_rate * diff * value
        return loss, prediction

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_type": "linear_boundary_classifier",
            "labels": list(self.labels),
            "weights": self.weights,
        }


def _evaluate_candidates(
    model: LinearBoundaryClassifier,
    candidates: list[dict[str, Any]],
    prefix: str,
) -> dict[str, int | float]:
    if not candidates:
        return {
            f"{prefix}/candidate_count": 0,
            f"{prefix}/loss": 0.0,
            f"{prefix}/accuracy": 0.0,
            f"{prefix}/precision": 0.0,
            f"{prefix}/recall": 0.0,
            f"{prefix}/f1": 0.0,
            f"{prefix}/tolerance1_f1": 0.0,
            f"{prefix}/parse_success_rate": 1.0,
        }

    loss = 0.0
    correct = 0
    by_example: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        features = _feature_dict(candidate["text"], int(candidate["sentence_number"]))
        probabilities = _softmax(model.scores(features))
        gold_label = str(candidate["label"])
        predicted_label = max(model.labels, key=lambda label: probabilities[label])
        loss += -math.log(max(probabilities[gold_label], 1e-12))
        if predicted_label == gold_label:
            correct += 1
        example = by_example.setdefault(
            str(candidate["example_id"]),
            {"gold": set(), "predicted": set(), "valid_sids": set()},
        )
        sentence_number = int(candidate["sentence_number"])
        local_sid = str(candidate["local_sid"])
        example["valid_sids"].add(local_sid)
        if gold_label != "none":
            example["gold"].add(sentence_number)
        if predicted_label != "none":
            example["predicted"].add(sentence_number)

    exact_counts = {"tp": 0, "fp": 0, "fn": 0}
    tolerance_counts = {"tp": 0, "fp": 0, "fn": 0}
    parse_success = 0
    for example in by_example.values():
        exact = boundary_f1(example["gold"], example["predicted"])
        tolerance = boundary_f1(example["gold"], example["predicted"], tolerance=1)
        for key in exact_counts:
            exact_counts[key] += int(exact[key])
            tolerance_counts[key] += int(tolerance[key])
        if _counts_to_f1(exact["tp"], exact["fp"], exact["fn"]) >= 0.0:
            parse_success += 1

    precision, recall, f1 = _precision_recall_f1(**exact_counts)
    _, _, tolerance_f1 = _precision_recall_f1(**tolerance_counts)
    return {
        f"{prefix}/candidate_count": len(candidates),
        f"{prefix}/example_count": len(by_example),
        f"{prefix}/loss": loss / len(candidates),
        f"{prefix}/accuracy": correct / len(candidates),
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/f1": f1,
        f"{prefix}/tolerance1_f1": tolerance_f1,
        f"{prefix}/parse_success_rate": parse_success / len(by_example),
    }


def _counts_to_f1(tp: int | float, fp: int | float, fn: int | float) -> float:
    _, _, f1 = _precision_recall_f1(int(tp), int(fp), int(fn))
    return f1


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = 1.0 if tp + fp == 0 and fn == 0 else _safe_ratio(tp, tp + fp)
    recall = 1.0 if tp + fn == 0 and fp == 0 else _safe_ratio(tp, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def train_boundary_classifier(
    dataset_dir: Path,
    out_dir: Path,
    family: str = "sparse_multi_boundary",
    epochs: int = 100,
    learning_rate: float = 0.05,
    weight_decay: float = 0.0,
    seed: int = 13,
    wandb_run: WandbRunHandle | None = None,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    train_candidates = _load_candidates(dataset_dir, family, "train")
    validation_candidates = _load_candidates(dataset_dir, family, "validation")
    test_candidates = _load_candidates(dataset_dir, family, "test")
    if not train_candidates:
        raise ValueError(f"{dataset_dir / family / 'train.jsonl'} contains no examples")

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    log_path = out_dir / "train.log"
    model = LinearBoundaryClassifier()
    rng = random.Random(seed)
    metrics_rows: list[dict[str, int | float]] = []
    logs = [
        (
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"train_start family={family} epochs={epochs} "
            f"train_candidates={len(train_candidates)} "
            f"validation_candidates={len(validation_candidates)}"
        )
    ]

    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for epoch in range(1, epochs + 1):
            epoch_candidates = list(train_candidates)
            rng.shuffle(epoch_candidates)
            total_loss = 0.0
            correct = 0
            for candidate in epoch_candidates:
                features = _feature_dict(candidate["text"], int(candidate["sentence_number"]))
                loss, prediction = model.train_candidate(
                    features=features,
                    gold_label=str(candidate["label"]),
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
                total_loss += loss
                if prediction == candidate["label"]:
                    correct += 1

            row: dict[str, int | float] = {
                "epoch": epoch,
                "train/loss": total_loss / len(epoch_candidates),
                "train/accuracy": correct / len(epoch_candidates),
                "train/candidate_count": len(epoch_candidates),
            }
            row.update(_evaluate_candidates(model, validation_candidates, "validation"))
            metrics_rows.append(row)
            metrics_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            metrics_handle.flush()
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            logs.append(
                "epoch={epoch} train_loss={loss:.6f} train_accuracy={accuracy:.4f} "
                "validation_f1={f1:.4f}".format(
                    epoch=epoch,
                    loss=row["train/loss"],
                    accuracy=row["train/accuracy"],
                    f1=row["validation/f1"],
                )
            )

    final_validation = _evaluate_candidates(model, validation_candidates, "validation")
    final_test = _evaluate_candidates(model, test_candidates, "test")
    model_payload = model.to_payload()
    model_payload["metadata"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "family": family,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    write_json(out_dir / "model.json", model_payload)
    logs.append(
        f"{datetime.now().isoformat(timespec='seconds')} train_finish "
        f"epochs={epochs} validation_f1={final_validation['validation/f1']:.4f} "
        f"test_f1={final_test['test/f1']:.4f}"
    )
    log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")
    return {
        "model_type": "linear_boundary_classifier",
        "family": family,
        "epochs_requested": epochs,
        "epochs_completed": len(metrics_rows),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "train_candidate_count": len(train_candidates),
        "validation_candidate_count": len(validation_candidates),
        "test_candidate_count": len(test_candidates),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "model_path": str(out_dir / "model.json"),
        "final_train": metrics_rows[-1] if metrics_rows else {},
        "final_validation": final_validation,
        "final_test": final_test,
    }


def write_training_summary(
    dataset_dir: Path,
    out_dir: Path,
    wandb_mode: str,
    wandb_project: str,
    run_name: str | None = None,
    env_file: Path | None = None,
    family: str = "sparse_multi_boundary",
    epochs: int = 100,
    learning_rate: float = 0.05,
    weight_decay: float = 0.0,
    seed: int = 13,
    wandb_backend: Any | None = None,
) -> dict[str, Any]:
    if env_file is not None:
        load_dotenv(env_file)
    resolved_run_name = run_name or f"sft-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dataset_summary = summarize_sft_dataset(dataset_dir)
    wandb_run = start_wandb_run(
        dataset_summary=dataset_summary,
        project=wandb_project,
        run_name=resolved_run_name,
        mode=wandb_mode,
        config={
            "trainer": "linear_boundary_classifier",
            "family": family,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        backend=wandb_backend,
    )
    wandb_run.log(flatten_wandb_metrics(dataset_summary), step=0)
    try:
        training = train_boundary_classifier(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            family=family,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            wandb_run=wandb_run,
        )
    finally:
        wandb_run.finish()
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_status": "trained_linear_boundary_classifier",
        "dataset": dataset_summary,
        "training": training,
        "wandb": wandb_run.status,
    }
    write_json(out_dir / "train_run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default="offline")
    parser.add_argument("--wandb-project", default="sermon-boundary-sft")
    parser.add_argument("--run-name")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--family", choices=SFT_FAMILIES, default="sparse_multi_boundary")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    summary = write_training_summary(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        env_file=args.env_file,
        family=args.family,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    print(f"out_dir: {args.out_dir}")
    print(f"training_status: {summary['training_status']}")
    print(f"model_type: {summary['training']['model_type']}")
    print(f"epochs_completed: {summary['training']['epochs_completed']}")
    print(f"total_examples: {summary['dataset']['total_examples']}")
    print(f"wandb_mode: {summary['wandb']['mode']}")
    print(f"wandb_api_key_present: {summary['wandb']['api_key_present']}")
    return 0
