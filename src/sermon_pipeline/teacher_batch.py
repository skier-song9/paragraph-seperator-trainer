from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .teacher_schema import build_teacher_payload
from .teacher_windows import build_teacher_windows, load_sentence_records


def _jsonl(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    payload.encode("utf-8")
    return payload


def _batch_request(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_batch_requests(
    sentences_path: Path,
    out_dir: Path,
    model: str = "gpt-5.5",
    target_size: int = 160,
    left_context: int = 20,
    right_context: int = 20,
    max_output_tokens: int = 8192,
    limit_windows: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_sentence_records(sentences_path)
    windows = build_teacher_windows(
        records,
        target_size=target_size,
        left_context=left_context,
        right_context=right_context,
    )
    if limit_windows is not None:
        windows = windows[:limit_windows]

    request_path = out_dir / "batch_requests.jsonl"
    mapping_path = out_dir / "windows.jsonl"
    failure_path = out_dir / "failures.jsonl"
    summary_path = out_dir / "run_summary.json"

    window_count = 0
    failure_count = 0

    with (
        request_path.open("w", encoding="utf-8") as request_handle,
        mapping_path.open("w", encoding="utf-8") as mapping_handle,
        failure_path.open("w", encoding="utf-8") as failure_handle,
    ):
        for window in windows:
            try:
                body = build_teacher_payload(
                    window,
                    model=model,
                    max_output_tokens=max_output_tokens,
                )
                request_handle.write(_jsonl(_batch_request(window.custom_id, body)))
                mapping_handle.write(_jsonl(window.to_mapping()))
                window_count += 1
            except Exception as exc:
                failure_handle.write(
                    _jsonl(
                        {
                            "custom_id": window.custom_id,
                            "document_id": window.document_id,
                            "exception_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                )
                failure_count += 1

    summary = {
        "sentences_path": str(sentences_path),
        "out_dir": str(out_dir),
        "request_path": str(request_path),
        "mapping_path": str(mapping_path),
        "failure_path": str(failure_path),
        "window_count": window_count,
        "failure_count": failure_count,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build OpenAI Batch JSONL requests for teacher annotation."
    )
    parser.add_argument("--sentences", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--target-size", type=int, default=160)
    parser.add_argument("--left-context", type=int, default=20)
    parser.add_argument("--right-context", type=int, default=20)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--limit-windows", type=int)
    args = parser.parse_args(argv)

    summary = build_batch_requests(
        sentences_path=args.sentences,
        out_dir=args.out_dir,
        model=args.model,
        target_size=args.target_size,
        left_context=args.left_context,
        right_context=args.right_context,
        max_output_tokens=args.max_output_tokens,
        limit_windows=args.limit_windows,
    )
    print(summary["out_dir"])
    print(
        f"window_count={summary['window_count']} "
        f"failure_count={summary['failure_count']}"
    )
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
