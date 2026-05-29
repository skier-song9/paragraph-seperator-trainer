from __future__ import annotations

import re


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def is_layout_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"[0-9]{1,3}", stripped):
        return True
    if stripped in {"•", "-", "—", "_"}:
        return True
    return False


def script_counts(text: str) -> dict[str, int]:
    counts = {"hangul": 0, "han": 0, "latin": 0, "greek": 0, "hebrew": 0}
    for ch in text:
        cp = ord(ch)
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            counts["hangul"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["han"] += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            counts["latin"] += 1
        elif 0x0370 <= cp <= 0x03FF:
            counts["greek"] += 1
        elif 0x0590 <= cp <= 0x05FF:
            counts["hebrew"] += 1
    return counts


def classify_korean_paragraph(text: str) -> tuple[bool, dict[str, int], str]:
    counts = script_counts(text)
    hangul = counts["hangul"]
    han = counts["han"]
    latin = counts["latin"]

    if hangul >= 2:
        return True, counts, "hangul_present"
    if re.search(r"\([^)]+[0-9]+[:：][0-9]+", text):
        return True, counts, "scripture_reference_or_short_context"
    if hangul == 0 and han >= 2:
        return False, counts, "han_dominant_foreign"
    if hangul == 0 and latin >= 20:
        return False, counts, "latin_dominant_foreign"
    return False, counts, "no_korean_signal"
