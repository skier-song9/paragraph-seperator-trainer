from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .io import extract_output_text


def call_openai(
    payload: dict[str, Any], api_key: str, timeout: int
) -> tuple[dict[str, Any], str, float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {error_body[:3000]}") from exc
    elapsed = time.time() - start
    return data, extract_output_text(data), elapsed
