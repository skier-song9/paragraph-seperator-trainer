# 설교 Teacher 라벨링 및 SFT 데이터셋 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `datas/`에서 추출된 문장 JSONL을 GPT-5.5 teacher 라벨링 요청으로 만들고, 응답을 검증한 뒤 small student model용 Sparse Multi-Boundary / First-Boundary SFT 데이터셋으로 변환한다.

**Architecture:** 기존 extraction pipeline은 유지한다. 새 코드는 `sentences.jsonl`을 입력으로 받아 teacher window sidecar, OpenAI Batch 요청 JSONL, teacher annotation 검증 결과, student SFT JSONL을 단계별 파일로 생성한다. Student input은 metadata 없는 `<S1> sentence text` 형식만 사용하고, source `sentence_id`는 sidecar mapping에만 남긴다.

**Tech Stack:** Python 3.11, stdlib `json`/`argparse`/`dataclasses`/`hashlib`, existing `uv`, `unittest`, OpenAI Responses API payload shape, OpenAI Batch JSONL request shape.

---

## 스코프 체크

이 계획은 Phase 1 구현이다.

포함:

- GPT-5.5 teacher window 생성
- OpenAI Batch 요청 JSONL 생성
- Batch 결과 ingest와 teacher annotation 검증
- Sparse Multi-Boundary / First-Boundary student SFT 데이터셋 생성
- boundary output parser와 boundary-level metric 유틸
- CLI entrypoint와 README 사용법

제외:

- OpenAI Batch job 업로드/다운로드 자동화
- LoRA/QLoRA training runner
- RAG retrieval/generation metric runner
- 3-4B escalation training

위 제외 항목은 SFT 데이터셋 파일이 실제로 생성된 뒤 별도 implementation plan으로 진행한다.

## 파일 구조

- Create: `src/sermon_pipeline/teacher_windows.py`
  - `sentences.jsonl` 로드, document별 정렬, target/context window 생성, local `<S1>` mapping 생성.

- Create: `src/sermon_pipeline/teacher_schema.py`
  - GPT-5.5 teacher system prompt, strict JSON schema, Responses API body 생성.

- Create: `src/sermon_pipeline/teacher_batch.py`
  - Batch 요청 JSONL과 `windows.jsonl` sidecar를 쓰는 CLI.

- Create: `src/sermon_pipeline/teacher_ingest.py`
  - Batch output JSONL을 읽고 annotation parse/validation/failure logging 수행.

- Create: `src/sermon_pipeline/student_sft.py`
  - 검증된 teacher annotations를 student SFT examples로 변환.

- Create: `src/sermon_pipeline/boundary_eval.py`
  - student output parser, exact/tolerance F1, Pk, WindowDiff 계산.

- Modify: `pyproject.toml`
  - CLI entrypoint 추가.

- Modify: `README.md`
  - 새 pipeline 명령 추가.

- Test: `tests/test_teacher_windows.py`
- Test: `tests/test_teacher_schema.py`
- Test: `tests/test_teacher_batch.py`
- Test: `tests/test_teacher_ingest.py`
- Test: `tests/test_student_sft.py`
- Test: `tests/test_boundary_eval.py`
- Modify Test: `tests/test_cli_smoke.py`

## Task 1: Teacher Window 생성

**Files:**
- Create: `src/sermon_pipeline/teacher_windows.py`
- Test: `tests/test_teacher_windows.py`

- [ ] **Step 1: failing test 작성**

`tests/test_teacher_windows.py` 생성:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.teacher_windows import (
    build_teacher_windows,
    load_sentence_records,
)


def _row(document_id: str, index: int, text: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "sentence_id": f"{document_id}.s{index:04d}",
        "sentence_index": index,
        "text": text,
        "source_path": f"datas/{document_id}.json",
        "document_source_type": "datalab_parsed_json",
        "document_kind": "book_chapter",
        "block_type": "paragraph",
        "source_tag": "p",
        "page_id": "1",
        "paragraph_index": index // 2,
        "heading_context": ["heading"],
        "html_boundary_before": index % 2 == 0,
    }


class TeacherWindowTests(unittest.TestCase):
    def test_load_sentence_records_normalizes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sentences.jsonl"
            rows = [_row("doc-a", 0, "첫 문장."), _row("doc-a", 1, "둘째 문장.")]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            records = load_sentence_records(path)

        self.assertEqual([record.sentence_id for record in records], ["doc-a.s0000", "doc-a.s0001"])
        self.assertEqual(records[0].source_type, "datalab_parsed_json")
        self.assertEqual(records[0].heading_context, ("heading",))

    def test_build_teacher_windows_uses_target_and_context(self) -> None:
        records = [_row("doc-a", index, f"{index}번 문장.") for index in range(6)]

        windows = build_teacher_windows(
            load_sentence_records_from_rows(records),
            target_size=3,
            left_context=1,
            right_context=1,
        )

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].custom_id, "teacher:doc-a:w0000:0-3")
        self.assertEqual([item.local_sid for item in windows[0].sentences], ["S1", "S2", "S3", "S4"])
        self.assertEqual(windows[0].target_local_sids, ("S1", "S2", "S3"))
        self.assertEqual(windows[1].custom_id, "teacher:doc-a:w0001:3-6")
        self.assertEqual([item.local_sid for item in windows[1].sentences], ["S1", "S2", "S3", "S4"])
        self.assertEqual(windows[1].target_local_sids, ("S2", "S3"))
        self.assertEqual(windows[1].sentences[1].source_sentence_id, "doc-a.s0003")

    def test_window_mapping_excludes_student_input_metadata(self) -> None:
        records = [_row("doc-a", index, f"{index}번 문장.") for index in range(4)]
        window = build_teacher_windows(load_sentence_records_from_rows(records), target_size=4)[0]

        mapping = window.to_mapping()
        task = window.to_teacher_task()

        self.assertIn("local_to_source", mapping)
        self.assertEqual(mapping["local_to_source"]["S1"]["source_sentence_id"], "doc-a.s0000")
        self.assertEqual(task["sentences"][0]["local_sid"], "S1")
        self.assertEqual(task["sentences"][0]["text"], "0번 문장.")
        self.assertIn("hints", task["sentences"][0])


def load_sentence_records_from_rows(rows: list[dict[str, object]]):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "sentences.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return load_sentence_records(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_windows -v
```

Expected:

```text
ImportError: No module named 'sermon_pipeline.teacher_windows'
```

- [ ] **Step 3: 최소 구현 작성**

`src/sermon_pipeline/teacher_windows.py` 생성:

```python
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
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
    block_type: str | None
    source_tag: str | None
    page_id: str | None
    paragraph_index: int | None
    heading_context: tuple[str, ...]
    html_boundary_before: bool

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SentenceRecord":
        return cls(
            document_id=str(row["document_id"]),
            sentence_id=str(row["sentence_id"]),
            sentence_index=int(row["sentence_index"]),
            text=str(row["text"]),
            source_path=str(row["source_path"]),
            source_type=str(row.get("document_source_type") or row.get("source_type") or ""),
            document_kind=str(row.get("document_kind") or ""),
            block_type=row.get("block_type"),
            source_tag=row.get("source_tag"),
            page_id=row.get("page_id"),
            paragraph_index=row.get("paragraph_index"),
            heading_context=tuple(row.get("heading_context") or ()),
            html_boundary_before=bool(row.get("html_boundary_before", False)),
        )


@dataclass(frozen=True)
class WindowSentence:
    local_sid: str
    source_sentence_id: str
    global_sentence_index: int
    text: str
    role: str
    hints: dict[str, Any]

    def to_teacher_payload(self) -> dict[str, Any]:
        return {
            "local_sid": self.local_sid,
            "source_sentence_id": self.source_sentence_id,
            "role": self.role,
            "text": self.text,
            "hints": self.hints,
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
                item.local_sid: item.to_mapping_payload() for item in self.sentences
            },
        }

    def to_teacher_task(self) -> dict[str, Any]:
        return {
            "task": "annotate_sermon_sentence_boundaries",
            "custom_id": self.custom_id,
            "document": {
                "document_id": self.document_id,
                "source_path": self.source_path,
                "source_type": self.source_type,
                "document_kind": self.document_kind,
            },
            "annotation_scope": {
                "target_local_sids": list(self.target_local_sids),
                "rule": "Return one annotation for each target_local_sid only.",
            },
            "sentences": [item.to_teacher_payload() for item in self.sentences],
        }


def load_sentence_records(path: Path) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(SentenceRecord.from_row(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid sentence row line={line_number}: {exc}") from exc
    return records


def _group_by_document(records: Iterable[SentenceRecord]) -> dict[str, list[SentenceRecord]]:
    grouped: dict[str, list[SentenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.document_id].append(record)
    for document_records in grouped.values():
        document_records.sort(key=lambda item: item.sentence_index)
    return dict(sorted(grouped.items()))


def _hints(record: SentenceRecord) -> dict[str, Any]:
    return {
        "block_type": record.block_type,
        "source_tag": record.source_tag,
        "page_id": record.page_id,
        "paragraph_index": record.paragraph_index,
        "heading_context": list(record.heading_context),
        "html_boundary_before": record.html_boundary_before,
    }


def build_teacher_windows(
    records: Iterable[SentenceRecord],
    target_size: int = 160,
    left_context: int = 20,
    right_context: int = 20,
) -> list[TeacherWindow]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if left_context < 0 or right_context < 0:
        raise ValueError("context sizes must be non-negative")

    windows: list[TeacherWindow] = []
    for document_id, document_records in _group_by_document(records).items():
        if not document_records:
            continue
        for window_index, target_start in enumerate(range(0, len(document_records), target_size)):
            target_end = min(len(document_records), target_start + target_size)
            visible_start = max(0, target_start - left_context)
            visible_end = min(len(document_records), target_end + right_context)
            visible_records = document_records[visible_start:visible_end]
            target_positions = set(range(target_start, target_end))
            target_local_sids: list[str] = []
            window_sentences: list[WindowSentence] = []

            for offset, record in enumerate(visible_records, start=1):
                position = visible_start + offset - 1
                local_sid = f"S{offset}"
                role = "target" if position in target_positions else "context"
                if role == "target" and position + 1 < visible_end:
                    target_local_sids.append(local_sid)
                window_sentences.append(
                    WindowSentence(
                        local_sid=local_sid,
                        source_sentence_id=record.sentence_id,
                        global_sentence_index=record.sentence_index,
                        text=record.text,
                        role=role,
                        hints=_hints(record),
                    )
                )

            first = document_records[0]
            windows.append(
                TeacherWindow(
                    custom_id=f"teacher:{document_id}:w{window_index:04d}:{target_start}-{target_end}",
                    document_id=document_id,
                    source_path=first.source_path,
                    source_type=first.source_type,
                    document_kind=first.document_kind,
                    window_index=window_index,
                    target_start=target_start,
                    target_end=target_end,
                    visible_start=visible_start,
                    visible_end=visible_end,
                    sentences=tuple(window_sentences),
                    target_local_sids=tuple(target_local_sids),
                )
            )
    return windows
```

- [ ] **Step 4: 테스트 통과 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_windows -v
```

Expected:

```text
Ran 3 tests
OK
```

- [ ] **Step 5: 커밋**

```bash
git add src/sermon_pipeline/teacher_windows.py tests/test_teacher_windows.py
git commit -m "feat: add teacher window generation"
```

## Task 2: GPT-5.5 Teacher Schema와 Responses API payload

**Files:**
- Create: `src/sermon_pipeline/teacher_schema.py`
- Test: `tests/test_teacher_schema.py`

- [ ] **Step 1: failing test 작성**

`tests/test_teacher_schema.py` 생성:

```python
from __future__ import annotations

import unittest

from sermon_pipeline.constants import BOUNDARY_TYPES
from sermon_pipeline.teacher_schema import build_teacher_payload, teacher_response_schema
from sermon_pipeline.teacher_windows import build_teacher_windows
from tests.test_teacher_windows import load_sentence_records_from_rows, _row


class TeacherSchemaTests(unittest.TestCase):
    def test_response_schema_uses_boundary_type_enum(self) -> None:
        schema = teacher_response_schema()
        annotation = schema["properties"]["boundary_annotations"]["items"]

        self.assertEqual(annotation["properties"]["boundary_type"]["enum"], list(BOUNDARY_TYPES))
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("custom_id", schema["required"])

    def test_payload_uses_responses_api_structured_output(self) -> None:
        records = load_sentence_records_from_rows([_row("doc-a", index, f"{index}번 문장.") for index in range(4)])
        window = build_teacher_windows(records, target_size=4)[0]

        payload = build_teacher_payload(window, model="gpt-5.5", max_output_tokens=4096)

        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertEqual(payload["input"][0]["role"], "system")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertIn("target_local_sids", payload["input"][1]["content"])
        self.assertIn("<S1>", payload["input"][1]["content"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_schema -v
```

Expected:

```text
ImportError: No module named 'sermon_pipeline.teacher_schema'
```

- [ ] **Step 3: schema/payload 구현**

`src/sermon_pipeline/teacher_schema.py` 생성:

```python
from __future__ import annotations

import json
from typing import Any

from .constants import BOUNDARY_TYPES
from .teacher_windows import TeacherWindow


TEACHER_SYSTEM_PROMPT = (
    "You are a Korean sermon discourse-boundary annotation teacher. "
    "Annotate only target_local_sids. "
    "Use metadata only as hints. "
    "Do not rewrite, summarize, translate, or normalize sermon text. "
    "A boundary line means the discourse should split after that sentence. "
    "Return strict JSON that matches the provided schema."
)


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
                        "boundary_type": {"type": "string", "enum": list(BOUNDARY_TYPES)},
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
    anchored_lines = [
        f"<{item.local_sid}> {item.text}" for item in window.sentences
    ]
    task["anchored_text"] = "\n".join(anchored_lines)
    task["allowed_boundary_types"] = list(BOUNDARY_TYPES)
    task["instructions"] = [
        "Return one boundary_annotations item for every target_local_sid.",
        "Use boundary_type='none' when split_after=false.",
        "Use a non-none boundary_type when split_after=true.",
        "Keep scripture quotation and verse reference together when possible.",
        "Treat headings, source tags, page ids, and paragraph hints as hints, not gold labels.",
        "Do not annotate context-only sentences.",
        "Do not annotate a boundary after a sentence whose next source sentence is not visible.",
    ]
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_schema -v
```

Expected:

```text
Ran 2 tests
OK
```

- [ ] **Step 5: 커밋**

```bash
git add src/sermon_pipeline/teacher_schema.py tests/test_teacher_schema.py
git commit -m "feat: add teacher structured output schema"
```

## Task 3: OpenAI Batch 요청 JSONL 생성 CLI

**Files:**
- Create: `src/sermon_pipeline/teacher_batch.py`
- Modify: `pyproject.toml`
- Test: `tests/test_teacher_batch.py`
- Modify Test: `tests/test_cli_smoke.py`

- [ ] **Step 1: failing test 작성**

`tests/test_teacher_batch.py` 생성:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.teacher_batch import build_batch_requests
from tests.test_teacher_windows import _row


class TeacherBatchTests(unittest.TestCase):
    def test_build_batch_requests_writes_jsonl_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sentences_path = root / "sentences.jsonl"
            out_dir = root / "out"
            rows = [_row("doc-a", index, f"{index}번 문장.") for index in range(5)]
            sentences_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = build_batch_requests(
                sentences_path=sentences_path,
                out_dir=out_dir,
                model="gpt-5.5",
                target_size=3,
                left_context=1,
                right_context=1,
                max_output_tokens=4096,
                limit_windows=None,
            )

            request_lines = (out_dir / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines()
            mapping_lines = (out_dir / "windows.jsonl").read_text(encoding="utf-8").splitlines()
            first_request = json.loads(request_lines[0])

        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(len(request_lines), 2)
        self.assertEqual(len(mapping_lines), 2)
        self.assertEqual(first_request["method"], "POST")
        self.assertEqual(first_request["url"], "/v1/responses")
        self.assertEqual(first_request["body"]["model"], "gpt-5.5")
        self.assertTrue(first_request["custom_id"].startswith("teacher:doc-a:w0000"))

    def test_limit_windows_caps_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sentences_path = root / "sentences.jsonl"
            out_dir = root / "out"
            rows = [_row("doc-a", index, f"{index}번 문장.") for index in range(10)]
            sentences_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = build_batch_requests(
                sentences_path=sentences_path,
                out_dir=out_dir,
                target_size=2,
                limit_windows=1,
            )

            request_lines = (out_dir / "batch_requests.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["window_count"], 1)
        self.assertEqual(len(request_lines), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_batch -v
```

Expected:

```text
ImportError: No module named 'sermon_pipeline.teacher_batch'
```

- [ ] **Step 3: batch builder 구현**

`src/sermon_pipeline/teacher_batch.py` 생성:

```python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import write_json
from .teacher_schema import build_teacher_payload
from .teacher_windows import build_teacher_windows, load_sentence_records


def _jsonl(row: dict[str, Any]) -> str:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    line.encode("utf-8")
    return line


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
    windows_path = out_dir / "windows.jsonl"
    failures_path = out_dir / "failures.jsonl"
    failures: list[dict[str, Any]] = []

    with (
        request_path.open("w", encoding="utf-8") as request_handle,
        windows_path.open("w", encoding="utf-8") as windows_handle,
        failures_path.open("w", encoding="utf-8") as failures_handle,
    ):
        for window in windows:
            try:
                payload = build_teacher_payload(
                    window,
                    model=model,
                    max_output_tokens=max_output_tokens,
                )
                request_handle.write(_jsonl(_batch_request(window.custom_id, payload)))
                windows_handle.write(_jsonl(window.to_mapping()))
            except Exception as exc:
                failure = {
                    "custom_id": window.custom_id,
                    "document_id": window.document_id,
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                failures_handle.write(_jsonl(failure))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sentences_path": str(sentences_path),
        "out_dir": str(out_dir),
        "model": model,
        "target_size": target_size,
        "left_context": left_context,
        "right_context": right_context,
        "max_output_tokens": max_output_tokens,
        "window_count": len(windows) - len(failures),
        "failure_count": len(failures),
        "output_files": {
            "batch_requests": str(request_path),
            "windows": str(windows_path),
            "failures": str(failures_path),
            "summary": str(out_dir / "run_summary.json"),
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
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
    print(f"window_count={summary['window_count']} failure_count={summary['failure_count']}")
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: `pyproject.toml` entrypoint 추가**

`[project.scripts]`에 한 줄 추가:

```toml
sermon-teacher-batch-build = "sermon_pipeline.teacher_batch:main"
```

- [ ] **Step 5: CLI smoke test 추가**

`tests/test_cli_smoke.py`에 import와 test method 추가:

```python
from sermon_pipeline.teacher_batch import main as teacher_batch_main
```

```python
    def test_teacher_batch_build_dry_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sentences_path = root / "sentences.jsonl"
            out_dir = root / "teacher_batch"
            rows = [
                {
                    "document_id": "doc-a",
                    "sentence_id": f"doc-a.s{index:04d}",
                    "sentence_index": index,
                    "text": f"{index}번 문장.",
                    "source_path": "datas/doc-a.json",
                    "document_source_type": "datalab_parsed_json",
                    "document_kind": "book_chapter",
                }
                for index in range(3)
            ]
            sentences_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            status = teacher_batch_main(
                [
                    "--sentences",
                    str(sentences_path),
                    "--out-dir",
                    str(out_dir),
                    "--target-size",
                    "2",
                    "--limit-windows",
                    "1",
                ]
            )

        self.assertEqual(status, 0)
        self.assertTrue((out_dir / "batch_requests.jsonl").exists())
```

- [ ] **Step 6: 테스트 통과 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_batch tests.test_cli_smoke -v
```

Expected:

```text
OK
```

- [ ] **Step 7: 커밋**

```bash
git add src/sermon_pipeline/teacher_batch.py tests/test_teacher_batch.py tests/test_cli_smoke.py pyproject.toml
git commit -m "feat: build teacher batch requests"
```

## Task 4: Batch 결과 ingest와 teacher validation

**Files:**
- Create: `src/sermon_pipeline/teacher_ingest.py`
- Modify: `pyproject.toml`
- Test: `tests/test_teacher_ingest.py`

- [ ] **Step 1: failing test 작성**

`tests/test_teacher_ingest.py` 생성:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.teacher_ingest import ingest_batch_results, validate_annotation


def _mapping() -> dict[str, object]:
    return {
        "custom_id": "teacher:doc-a:w0000:0-3",
        "document_id": "doc-a",
        "source_path": "datas/doc-a.json",
        "source_type": "datalab_parsed_json",
        "document_kind": "book_chapter",
        "target_local_sids": ["S1", "S2", "S3"],
        "local_to_source": {
            "S1": {"source_sentence_id": "doc-a.s0000", "global_sentence_index": 0, "role": "target"},
            "S2": {"source_sentence_id": "doc-a.s0001", "global_sentence_index": 1, "role": "target"},
            "S3": {"source_sentence_id": "doc-a.s0002", "global_sentence_index": 2, "role": "target"},
            "S4": {"source_sentence_id": "doc-a.s0003", "global_sentence_index": 3, "role": "context"},
        },
    }


def _annotation() -> dict[str, object]:
    return {
        "custom_id": "teacher:doc-a:w0000:0-3",
        "boundary_annotations": [
            {
                "local_sid": "S1",
                "source_sentence_id": "doc-a.s0000",
                "split_after": False,
                "boundary_type": "none",
                "confidence": 0.9,
                "rationale": "same unit",
            },
            {
                "local_sid": "S2",
                "source_sentence_id": "doc-a.s0001",
                "split_after": True,
                "boundary_type": "topic_shift",
                "confidence": 0.8,
                "rationale": "topic changes",
            },
            {
                "local_sid": "S3",
                "source_sentence_id": "doc-a.s0002",
                "split_after": False,
                "boundary_type": "none",
                "confidence": 0.7,
                "rationale": "continues",
            },
        ],
        "quality_flags": [],
    }


class TeacherIngestTests(unittest.TestCase):
    def test_validate_annotation_accepts_valid_output(self) -> None:
        issues = validate_annotation(_mapping(), _annotation())
        self.assertEqual(issues, [])

    def test_validate_annotation_rejects_missing_target_and_bad_label(self) -> None:
        annotation = _annotation()
        annotation["boundary_annotations"] = annotation["boundary_annotations"][:2]
        annotation["boundary_annotations"][1]["boundary_type"] = "bad_label"

        issues = validate_annotation(_mapping(), annotation)
        codes = [issue["code"] for issue in issues]

        self.assertIn("missing_target_annotation", codes)
        self.assertIn("unknown_boundary_type", codes)

    def test_ingest_batch_results_writes_annotations_review_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            windows_path = root / "windows.jsonl"
            batch_output_path = root / "batch_output.jsonl"
            out_dir = root / "ingested"
            windows_path.write_text(json.dumps(_mapping(), ensure_ascii=False) + "\n", encoding="utf-8")
            response_body = {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(_annotation(), ensure_ascii=False),
                            }
                        ]
                    }
                ]
            }
            batch_output_path.write_text(
                json.dumps(
                    {
                        "custom_id": "teacher:doc-a:w0000:0-3",
                        "response": {"status_code": 200, "body": response_body},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = ingest_batch_results(
                windows_path=windows_path,
                batch_output_path=batch_output_path,
                out_dir=out_dir,
                review_confidence_threshold=0.75,
            )

            annotations = (out_dir / "teacher_annotations.jsonl").read_text(encoding="utf-8").splitlines()
            review = (out_dir / "needs_human_review.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(len(review), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_ingest -v
```

Expected:

```text
ImportError: No module named 'sermon_pipeline.teacher_ingest'
```

- [ ] **Step 3: ingest/validation 구현**

`src/sermon_pipeline/teacher_ingest.py` 생성:

```python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import BOUNDARY_TYPES
from .io import extract_output_text, write_json


def _jsonl(row: dict[str, Any]) -> str:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    line.encode("utf-8")
    return line


def _load_jsonl_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = str(row["custom_id"])
            if custom_id in rows:
                raise ValueError(f"duplicate custom_id={custom_id} in {path} line={line_number}")
            rows[custom_id] = row
    return rows


def validate_annotation(mapping: dict[str, Any], annotation: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    custom_id = str(mapping["custom_id"])
    expected = set(mapping["target_local_sids"])
    local_to_source = mapping["local_to_source"]
    annotations = annotation.get("boundary_annotations", [])
    seen: set[str] = set()

    if annotation.get("custom_id") != custom_id:
        issues.append({"custom_id": custom_id, "code": "custom_id_mismatch", "message": "annotation custom_id differs"})

    for item in annotations:
        local_sid = item.get("local_sid")
        if local_sid in seen:
            issues.append({"custom_id": custom_id, "code": "duplicate_local_sid", "local_sid": local_sid, "message": "duplicate annotation"})
        seen.add(local_sid)
        if local_sid not in expected:
            issues.append({"custom_id": custom_id, "code": "unknown_or_context_local_sid", "local_sid": local_sid, "message": "annotation is not in target_local_sids"})
            continue
        expected_source = local_to_source[local_sid]["source_sentence_id"]
        if item.get("source_sentence_id") != expected_source:
            issues.append({"custom_id": custom_id, "code": "source_sentence_id_mismatch", "local_sid": local_sid, "message": "source sentence id differs"})
        boundary_type = item.get("boundary_type")
        split_after = item.get("split_after")
        confidence = item.get("confidence")
        if boundary_type not in BOUNDARY_TYPES:
            issues.append({"custom_id": custom_id, "code": "unknown_boundary_type", "local_sid": local_sid, "message": f"invalid boundary_type={boundary_type}"})
        if split_after is False and boundary_type != "none":
            issues.append({"custom_id": custom_id, "code": "split_false_boundary_not_none", "local_sid": local_sid, "message": "split_after=false requires none"})
        if split_after is True and boundary_type == "none":
            issues.append({"custom_id": custom_id, "code": "split_true_boundary_none", "local_sid": local_sid, "message": "split_after=true requires non-none"})
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            issues.append({"custom_id": custom_id, "code": "invalid_confidence", "local_sid": local_sid, "message": "confidence must be in [0, 1]"})

    for local_sid in sorted(expected - seen):
        issues.append({"custom_id": custom_id, "code": "missing_target_annotation", "local_sid": local_sid, "message": "target local sid was not annotated"})
    return issues


def _extract_batch_annotation(row: dict[str, Any]) -> dict[str, Any]:
    custom_id = str(row.get("custom_id", ""))
    response = row.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"missing response for custom_id={custom_id}")
    if int(response.get("status_code", 0)) != 200:
        raise ValueError(f"non-200 response for custom_id={custom_id}: {response.get('status_code')}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"missing response body for custom_id={custom_id}")
    output_text = extract_output_text(body)
    if not output_text.strip():
        raise ValueError(f"empty output_text for custom_id={custom_id}")
    return json.loads(output_text)


def _needs_review(annotation: dict[str, Any], threshold: float) -> bool:
    if annotation.get("quality_flags"):
        return True
    for item in annotation.get("boundary_annotations", []):
        if float(item.get("confidence", 0.0)) < threshold:
            return True
    return False


def ingest_batch_results(
    windows_path: Path,
    batch_output_path: Path,
    out_dir: Path,
    review_confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings = _load_jsonl_by_custom_id(windows_path)
    ok_count = 0
    issue_count = 0
    failure_count = 0
    review_count = 0

    annotations_path = out_dir / "teacher_annotations.jsonl"
    issues_path = out_dir / "teacher_validation_issues.jsonl"
    failures_path = out_dir / "teacher_failures.jsonl"
    review_path = out_dir / "needs_human_review.jsonl"

    with (
        batch_output_path.open("r", encoding="utf-8") as input_handle,
        annotations_path.open("w", encoding="utf-8") as annotations_handle,
        issues_path.open("w", encoding="utf-8") as issues_handle,
        failures_path.open("w", encoding="utf-8") as failures_handle,
        review_path.open("w", encoding="utf-8") as review_handle,
    ):
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = str(row.get("custom_id", ""))
            try:
                if custom_id not in mappings:
                    raise ValueError(f"unknown custom_id={custom_id}")
                annotation = _extract_batch_annotation(row)
                issues = validate_annotation(mappings[custom_id], annotation)
                if issues:
                    issue_count += len(issues)
                    for issue in issues:
                        issues_handle.write(_jsonl(issue))
                    continue
                annotations_handle.write(_jsonl(annotation))
                ok_count += 1
                if _needs_review(annotation, review_confidence_threshold):
                    review_handle.write(_jsonl({"custom_id": custom_id, "annotation": annotation, "mapping": mappings[custom_id]}))
                    review_count += 1
            except Exception as exc:
                failure_count += 1
                failures_handle.write(
                    _jsonl(
                        {
                            "custom_id": custom_id,
                            "line_number": line_number,
                            "exception_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "windows_path": str(windows_path),
        "batch_output_path": str(batch_output_path),
        "out_dir": str(out_dir),
        "ok_count": ok_count,
        "issue_count": issue_count,
        "failure_count": failure_count,
        "review_count": review_count,
    }
    write_json(out_dir / "labeling_run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--batch-output", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--review-confidence-threshold", type=float, default=0.75)
    args = parser.parse_args(argv)
    summary = ingest_batch_results(
        windows_path=args.windows,
        batch_output_path=args.batch_output,
        out_dir=args.out_dir,
        review_confidence_threshold=args.review_confidence_threshold,
    )
    print(summary["out_dir"])
    print(f"ok_count={summary['ok_count']} issue_count={summary['issue_count']} failure_count={summary['failure_count']} review_count={summary['review_count']}")
    return 0 if summary["failure_count"] == 0 and summary["issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: `pyproject.toml` entrypoint 추가**

```toml
sermon-teacher-batch-ingest = "sermon_pipeline.teacher_ingest:main"
```

- [ ] **Step 5: 테스트 통과 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_ingest -v
```

Expected:

```text
Ran 3 tests
OK
```

- [ ] **Step 6: 커밋**

```bash
git add src/sermon_pipeline/teacher_ingest.py tests/test_teacher_ingest.py pyproject.toml
git commit -m "feat: ingest teacher batch results"
```

## Task 5: Student SFT 변환

**Files:**
- Create: `src/sermon_pipeline/student_sft.py`
- Modify: `pyproject.toml`
- Test: `tests/test_student_sft.py`

- [ ] **Step 1: failing test 작성**

`tests/test_student_sft.py` 생성:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sermon_pipeline.student_sft import (
    build_sft_datasets,
    first_boundary_output,
    render_student_input,
    sparse_boundary_output,
)


def _mapping() -> dict[str, object]:
    return {
        "custom_id": "teacher:doc-a:w0000:0-3",
        "document_id": "doc-a",
        "source_path": "datas/doc-a.json",
        "source_type": "datalab_parsed_json",
        "document_kind": "book_chapter",
        "target_local_sids": ["S1", "S2", "S3"],
        "local_to_source": {
            "S1": {"source_sentence_id": "doc-a.s0000", "global_sentence_index": 0, "role": "target", "text": "첫 문장."},
            "S2": {"source_sentence_id": "doc-a.s0001", "global_sentence_index": 1, "role": "target", "text": "둘째 문장."},
            "S3": {"source_sentence_id": "doc-a.s0002", "global_sentence_index": 2, "role": "target", "text": "셋째 문장."},
        },
    }


def _annotation() -> dict[str, object]:
    return {
        "custom_id": "teacher:doc-a:w0000:0-3",
        "boundary_annotations": [
            {"local_sid": "S1", "source_sentence_id": "doc-a.s0000", "split_after": False, "boundary_type": "none", "confidence": 0.9, "rationale": ""},
            {"local_sid": "S2", "source_sentence_id": "doc-a.s0001", "split_after": True, "boundary_type": "topic_shift", "confidence": 0.8, "rationale": ""},
            {"local_sid": "S3", "source_sentence_id": "doc-a.s0002", "split_after": True, "boundary_type": "application_start", "confidence": 0.8, "rationale": ""},
        ],
        "quality_flags": [],
    }


class StudentSftTests(unittest.TestCase):
    def test_render_student_input_contains_only_task_and_text(self) -> None:
        text = render_student_input([("S1", "첫 문장."), ("S2", "둘째 문장.")])

        self.assertIn("<TASK>", text)
        self.assertIn("<TEXT>", text)
        self.assertIn("<S1> 첫 문장.", text)
        self.assertNotIn("source_sentence_id", text)
        self.assertNotIn("document_id", text)

    def test_sparse_and_first_boundary_outputs(self) -> None:
        annotation = _annotation()

        self.assertEqual(
            sparse_boundary_output(annotation),
            "S2 topic_shift\nS3 application_start",
        )
        self.assertEqual(first_boundary_output(annotation), "S2 topic_shift")

    def test_no_boundary_outputs(self) -> None:
        annotation = _annotation()
        for item in annotation["boundary_annotations"]:
            item["split_after"] = False
            item["boundary_type"] = "none"

        self.assertEqual(sparse_boundary_output(annotation), "NO_BOUNDARY")
        self.assertEqual(first_boundary_output(annotation), "NO_BOUNDARY")

    def test_build_sft_datasets_writes_two_dataset_families(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotations_path = root / "teacher_annotations.jsonl"
            windows_path = root / "windows.jsonl"
            out_dir = root / "sft"
            annotations_path.write_text(json.dumps(_annotation(), ensure_ascii=False) + "\n", encoding="utf-8")
            windows_path.write_text(json.dumps(_mapping(), ensure_ascii=False) + "\n", encoding="utf-8")

            summary = build_sft_datasets(
                annotations_path=annotations_path,
                windows_path=windows_path,
                out_dir=out_dir,
            )

            sparse_files = list((out_dir / "sparse_multi_boundary").glob("*.jsonl"))
            first_files = list((out_dir / "first_boundary").glob("*.jsonl"))
            mapping_lines = (out_dir / "mappings.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["example_count"], 1)
        self.assertEqual(len(sparse_files), 3)
        self.assertEqual(len(first_files), 3)
        self.assertEqual(len(mapping_lines), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
uv run python -m unittest tests.test_student_sft -v
```

Expected:

```text
ImportError: No module named 'sermon_pipeline.student_sft'
```

- [ ] **Step 3: SFT 변환 구현**

`src/sermon_pipeline/student_sft.py` 생성:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import BOUNDARY_TYPES
from .io import write_json


STUDENT_LABELS = tuple(label for label in BOUNDARY_TYPES if label != "none")


def _jsonl(row: dict[str, Any]) -> str:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    line.encode("utf-8")
    return line


def _load_jsonl_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["custom_id"])] = row
    return rows


def render_student_input(sentences: list[tuple[str, str]]) -> str:
    allowed = ", ".join(STUDENT_LABELS)
    lines = [
        "<TASK>",
        "Find sermon discourse split boundaries.",
        "Return only boundary lines. Use NO_BOUNDARY if no split is needed.",
        "Each boundary line means split after that sentence.",
        f"Allowed labels: {allowed}",
        "",
        "<TEXT>",
    ]
    lines.extend(f"<{local_sid}> {text}" for local_sid, text in sentences)
    return "\n".join(lines)


def _boundary_lines(annotation: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in annotation.get("boundary_annotations", []):
        if item.get("split_after") is True and item.get("boundary_type") != "none":
            lines.append(f"{item['local_sid']} {item['boundary_type']}")
    return lines


def sparse_boundary_output(annotation: dict[str, Any]) -> str:
    lines = _boundary_lines(annotation)
    return "\n".join(lines) if lines else "NO_BOUNDARY"


def first_boundary_output(annotation: dict[str, Any]) -> str:
    lines = _boundary_lines(annotation)
    return lines[0] if lines else "NO_BOUNDARY"


def split_for_document(document_id: str) -> str:
    bucket = int(hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def _sentences_from_mapping(mapping: dict[str, Any]) -> list[tuple[str, str]]:
    sentences: list[tuple[str, str]] = []
    local_to_source = mapping["local_to_source"]
    for local_sid in sorted(local_to_source, key=lambda sid: int(sid[1:])):
        if local_sid not in mapping["target_local_sids"]:
            continue
        item = local_to_source[local_sid]
        text = str(item.get("text") or "")
        sentences.append((local_sid, text))
    return sentences


def _example(example_id: str, document_id: str, input_text: str, output_text: str, target_type: str) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "document_id": document_id,
        "target_type": target_type,
        "input": input_text,
        "output": output_text,
    }


def build_sft_datasets(
    annotations_path: Path,
    windows_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings = _load_jsonl_by_custom_id(windows_path)
    annotations = _load_jsonl_by_custom_id(annotations_path)

    for family in ("sparse_multi_boundary", "first_boundary"):
        (out_dir / family).mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            (out_dir / family / f"{split}.jsonl").write_text("", encoding="utf-8")

    mapping_path = out_dir / "mappings.jsonl"
    example_count = 0
    split_counts = {"train": 0, "validation": 0, "test": 0}

    with mapping_path.open("w", encoding="utf-8") as mapping_handle:
        for custom_id, annotation in sorted(annotations.items()):
            mapping = mappings[custom_id]
            document_id = str(mapping["document_id"])
            split = split_for_document(document_id)
            sentences = _sentences_from_mapping(mapping)
            input_text = render_student_input(sentences)
            sparse_id = custom_id.replace("teacher:", "sft:sparse:")
            first_id = custom_id.replace("teacher:", "sft:first:")

            sparse_row = _example(sparse_id, document_id, input_text, sparse_boundary_output(annotation), "sparse_multi_boundary")
            first_row = _example(first_id, document_id, input_text, first_boundary_output(annotation), "first_boundary")

            with (out_dir / "sparse_multi_boundary" / f"{split}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(_jsonl(sparse_row))
            with (out_dir / "first_boundary" / f"{split}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(_jsonl(first_row))

            mapping_handle.write(
                _jsonl(
                    {
                        "sparse_example_id": sparse_id,
                        "first_boundary_example_id": first_id,
                        "custom_id": custom_id,
                        "document_id": document_id,
                        "source_path": mapping["source_path"],
                        "local_to_source": mapping["local_to_source"],
                    }
                )
            )
            example_count += 1
            split_counts[split] += 1

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "annotations_path": str(annotations_path),
        "windows_path": str(windows_path),
        "out_dir": str(out_dir),
        "example_count": example_count,
        "split_counts": split_counts,
    }
    write_json(out_dir / "sft_run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_sft_datasets(
        annotations_path=args.annotations,
        windows_path=args.windows,
        out_dir=args.out_dir,
    )
    print(summary["out_dir"])
    print(f"example_count={summary['example_count']} split_counts={summary['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

주의: Task 1의 `window.to_mapping()`은 아직 `text`를 sidecar에 쓰지 않는다. 이 task에서 `src/sermon_pipeline/teacher_windows.py`의 `WindowSentence.to_mapping_payload()`에 `"text": self.text`를 추가한다.

```python
    def to_mapping_payload(self) -> dict[str, Any]:
        return {
            "source_sentence_id": self.source_sentence_id,
            "global_sentence_index": self.global_sentence_index,
            "role": self.role,
            "text": self.text,
        }
```

- [ ] **Step 4: `pyproject.toml` entrypoint 추가**

```toml
sermon-sft-build = "sermon_pipeline.student_sft:main"
```

- [ ] **Step 5: 테스트 통과 확인**

Run:

```bash
uv run python -m unittest tests.test_teacher_windows tests.test_student_sft -v
```

Expected:

```text
OK
```

- [ ] **Step 6: 커밋**

```bash
git add src/sermon_pipeline/student_sft.py src/sermon_pipeline/teacher_windows.py tests/test_student_sft.py tests/test_teacher_windows.py pyproject.toml
git commit -m "feat: build student sft datasets"
```

## Task 6: Student output parser와 boundary metrics

**Files:**
- Create: `src/sermon_pipeline/boundary_eval.py`
- Test: `tests/test_boundary_eval.py`

- [ ] **Step 1: failing test 작성**

`tests/test_boundary_eval.py` 생성:

```python
from __future__ import annotations

import unittest

from sermon_pipeline.boundary_eval import (
    boundary_f1,
    parse_student_output,
    pk_score,
    windowdiff_score,
)


class BoundaryEvalTests(unittest.TestCase):
    def test_parse_no_boundary(self) -> None:
        parsed = parse_student_output("NO_BOUNDARY", valid_local_sids={"S1", "S2"})
        self.assertEqual(parsed.boundaries, [])
        self.assertEqual(parsed.issues, [])

    def test_parse_sparse_boundaries(self) -> None:
        parsed = parse_student_output("S2 topic_shift\nS5 application_start", valid_local_sids={"S1", "S2", "S5"})
        self.assertEqual([(item.local_sid, item.boundary_type) for item in parsed.boundaries], [("S2", "topic_shift"), ("S5", "application_start")])
        self.assertEqual(parsed.issues, [])

    def test_parse_reports_invalid_lines(self) -> None:
        parsed = parse_student_output("S2 none\nS9 topic_shift\nbad", valid_local_sids={"S1", "S2"})
        codes = [issue["code"] for issue in parsed.issues]
        self.assertIn("invalid_label", codes)
        self.assertIn("unknown_local_sid", codes)
        self.assertIn("invalid_line", codes)

    def test_boundary_f1_exact_and_tolerance(self) -> None:
        exact = boundary_f1(gold={2, 7}, predicted={2, 8}, tolerance=0)
        tolerant = boundary_f1(gold={2, 7}, predicted={2, 8}, tolerance=1)

        self.assertEqual(exact["true_positive"], 1)
        self.assertAlmostEqual(exact["f1"], 0.5)
        self.assertEqual(tolerant["true_positive"], 2)
        self.assertAlmostEqual(tolerant["f1"], 1.0)

    def test_pk_and_windowdiff_perfect_zero(self) -> None:
        self.assertEqual(pk_score(sentence_count=6, gold={2, 4}, predicted={2, 4}), 0.0)
        self.assertEqual(windowdiff_score(sentence_count=6, gold={2, 4}, predicted={2, 4}), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run:

```bash
uv run python -m unittest tests.test_boundary_eval -v
```

Expected:

```text
ImportError: No module named 'sermon_pipeline.boundary_eval'
```

- [ ] **Step 3: parser/metrics 구현**

`src/sermon_pipeline/boundary_eval.py` 생성:

```python
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
    if text == "NO_BOUNDARY":
        return ParsedOutput(boundaries=[], issues=[])

    boundaries: list[BoundaryPrediction] = []
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_number = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 2 or not parts[0].startswith("S") or not parts[0][1:].isdigit():
            issues.append({"code": "invalid_line", "line_number": line_number, "line": line})
            continue
        local_sid, boundary_type = parts
        sentence_number = int(local_sid[1:])
        if local_sid not in valid_local_sids:
            issues.append({"code": "unknown_local_sid", "line_number": line_number, "local_sid": local_sid})
        if boundary_type not in STUDENT_LABELS:
            issues.append({"code": "invalid_label", "line_number": line_number, "boundary_type": boundary_type})
        if local_sid in seen:
            issues.append({"code": "duplicate_local_sid", "line_number": line_number, "local_sid": local_sid})
        if sentence_number < previous_number:
            issues.append({"code": "not_ascending", "line_number": line_number, "local_sid": local_sid})
        seen.add(local_sid)
        previous_number = sentence_number
        if local_sid in valid_local_sids and boundary_type in STUDENT_LABELS:
            boundaries.append(BoundaryPrediction(local_sid, sentence_number, boundary_type))
    return ParsedOutput(boundaries=boundaries, issues=issues)


def boundary_f1(gold: set[int], predicted: set[int], tolerance: int = 0) -> dict[str, float | int]:
    matched_gold: set[int] = set()
    true_positive = 0
    for pred in sorted(predicted):
        candidates = [item for item in sorted(gold) if item not in matched_gold and abs(item - pred) <= tolerance]
        if candidates:
            matched_gold.add(candidates[0])
            true_positive += 1
    false_positive = len(predicted) - true_positive
    false_negative = len(gold) - true_positive
    precision = true_positive / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = true_positive / len(gold) if gold else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _segment_id(sentence_position: int, boundaries: set[int]) -> int:
    return sum(1 for boundary in boundaries if boundary < sentence_position)


def _window_size(sentence_count: int, boundaries: set[int]) -> int:
    segment_count = len(boundaries) + 1
    average_segment_length = max(1, round(sentence_count / segment_count))
    return max(1, round(average_segment_length / 2))


def pk_score(sentence_count: int, gold: set[int], predicted: set[int]) -> float:
    if sentence_count <= 1:
        return 0.0
    k = _window_size(sentence_count, gold)
    total = max(1, sentence_count - k)
    errors = 0
    for start in range(1, sentence_count - k + 1):
        gold_same = _segment_id(start, gold) == _segment_id(start + k, gold)
        pred_same = _segment_id(start, predicted) == _segment_id(start + k, predicted)
        if gold_same != pred_same:
            errors += 1
    return errors / total


def _boundary_count_between(start: int, end: int, boundaries: set[int]) -> int:
    return sum(1 for boundary in boundaries if start <= boundary < end)


def windowdiff_score(sentence_count: int, gold: set[int], predicted: set[int]) -> float:
    if sentence_count <= 1:
        return 0.0
    k = _window_size(sentence_count, gold)
    total = max(1, sentence_count - k)
    errors = 0
    for start in range(1, sentence_count - k + 1):
        end = start + k
        if _boundary_count_between(start, end, gold) != _boundary_count_between(start, end, predicted):
            errors += 1
    return errors / total
```

- [ ] **Step 4: 테스트 통과 확인**

Run:

```bash
uv run python -m unittest tests.test_boundary_eval -v
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 5: 커밋**

```bash
git add src/sermon_pipeline/boundary_eval.py tests/test_boundary_eval.py
git commit -m "feat: add boundary output evaluation utilities"
```

## Task 7: README 사용법 추가

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README에 Phase 1 명령 추가**

`README.md`의 기존 명령 아래에 추가:

````markdown
Build GPT-5.5 teacher Batch requests from extracted sentences:

```bash
uv run sermon-teacher-batch-build \
  --sentences tests/results/dataset_final_check_20260529_1431/sentences.jsonl \
  --out-dir tests/results/teacher_batch_$(date +%Y%m%d_%H%M%S) \
  --model gpt-5.5 \
  --target-size 160 \
  --left-context 20 \
  --right-context 20
```

After downloading OpenAI Batch results, ingest and validate them:

```bash
uv run sermon-teacher-batch-ingest \
  --windows tests/results/teacher_batch_20260529_000000/windows.jsonl \
  --batch-output tests/results/teacher_batch_20260529_000000/openai_batch_output.jsonl \
  --out-dir tests/results/teacher_ingest_20260529_000000
```

Build student SFT datasets:

```bash
uv run sermon-sft-build \
  --annotations tests/results/teacher_ingest_20260529_000000/teacher_annotations.jsonl \
  --windows tests/results/teacher_batch_20260529_000000/windows.jsonl \
  --out-dir tests/results/student_sft_20260529_000000
```
````

- [ ] **Step 2: README grep 검증**

Run:

```bash
rg -n "sermon-teacher-batch-build|sermon-teacher-batch-ingest|sermon-sft-build" README.md
```

Expected:

```text
uv run sermon-teacher-batch-build \
uv run sermon-teacher-batch-ingest \
uv run sermon-sft-build \
```

- [ ] **Step 3: 커밋**

```bash
git add README.md
git commit -m "docs: document teacher labeling dataset commands"
```

## Task 8: 전체 검증

**Files:**
- No code changes.

- [ ] **Step 1: 전체 unittest 실행**

Run:

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

Expected:

```text
OK
```

- [ ] **Step 2: 실제 extraction dataset으로 작은 batch dry run**

Run:

```bash
uv run sermon-teacher-batch-build \
  --sentences tests/results/dataset_final_check_20260529_1431/sentences.jsonl \
  --out-dir tests/results/teacher_batch_smoke \
  --model gpt-5.5 \
  --target-size 20 \
  --left-context 3 \
  --right-context 3 \
  --limit-windows 2
```

Expected:

```text
tests/results/teacher_batch_smoke
window_count=2 failure_count=0
```

- [ ] **Step 3: batch JSONL 구조 확인**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("tests/results/teacher_batch_smoke/batch_requests.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
assert len(rows) == 2
assert rows[0]["method"] == "POST"
assert rows[0]["url"] == "/v1/responses"
assert rows[0]["body"]["model"] == "gpt-5.5"
assert rows[0]["body"]["text"]["format"]["type"] == "json_schema"
print("batch_jsonl_ok")
PY
```

Expected:

```text
batch_jsonl_ok
```

- [ ] **Step 4: git 상태 확인**

Run:

```bash
git status --short
```

Expected:

```text
 M .gitignore
?? .codex/
?? docs/research/
?? docs/reviews/sermon_data_preprocessing_strategy_260526.html
?? docs/reviews/sermon_rag_segmentation_prior_work_260526.html
```

The expected dirty files are pre-existing unrelated workspace files. If implementation files appear, commit the intended files before completion.

## 구현 후 실행 순서

1. `uv run sermon-dataset-build --out-dir tests/results/dataset_final_check_YYYYMMDD_HHMM`
2. `uv run sermon-teacher-batch-build --sentences tests/results/dataset_final_check_YYYYMMDD_HHMM/sentences.jsonl --out-dir tests/results/teacher_batch_YYYYMMDD_HHMM --model gpt-5.5`
3. Upload `tests/results/teacher_batch_YYYYMMDD_HHMM/batch_requests.jsonl` to OpenAI Batch API for `/v1/responses`.
4. Download Batch output to `tests/results/teacher_batch_YYYYMMDD_HHMM/openai_batch_output.jsonl`.
5. `uv run sermon-teacher-batch-ingest --windows tests/results/teacher_batch_YYYYMMDD_HHMM/windows.jsonl --batch-output tests/results/teacher_batch_YYYYMMDD_HHMM/openai_batch_output.jsonl --out-dir tests/results/teacher_ingest_YYYYMMDD_HHMM`
6. Human review `needs_human_review.jsonl`; replace or merge reviewed labels into `teacher_annotations.jsonl`.
7. `uv run sermon-sft-build --annotations tests/results/teacher_ingest_YYYYMMDD_HHMM/teacher_annotations.jsonl --windows tests/results/teacher_batch_YYYYMMDD_HHMM/windows.jsonl --out-dir tests/results/student_sft_YYYYMMDD_HHMM`

## Self-Review

- Spec coverage:
  - GPT-5.5 teacher windowing: Task 1.
  - Batch request generation: Task 2, Task 3.
  - Batch result ingestion and failure logs: Task 4.
  - Teacher-to-SFT conversion: Task 5.
  - Sparse-only and First-Boundary dataset families: Task 5.
  - `N=` 제거: Task 5 output functions never emit counts.
  - Metadata-free student input: Task 5 `render_student_input`.
  - Sidecar mapping: Task 1 and Task 5.
  - Boundary parser and metrics: Task 6.
  - README command path: Task 7.

- Placeholder scan:
  - 금지 문자열 없음.
  - No undefined task names.
  - Every code-writing task includes file path, test, implementation snippet, command, expected result, commit.

- Type consistency:
  - `custom_id` is the join key across `windows.jsonl`, `batch_requests.jsonl`, `teacher_annotations.jsonl`, and SFT mappings.
  - `local_sid` uses `S1` format without angle brackets in outputs.
  - Student output labels use `BOUNDARY_TYPES` excluding `none`.
  - Teacher output labels use full `BOUNDARY_TYPES` including `none`.
