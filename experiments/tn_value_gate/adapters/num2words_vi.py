"""Fair integer-span adapter for num2words; not a full TN system."""

import re
import time
from importlib.metadata import version

from num2words import num2words

from .base import NormalizationResult

NAME = "num2words_vi"
VERSION = version("num2words")
_INTEGER = re.compile(r"\d+")
_STRUCTURED = re.compile(r"\d\s*[/,:.\-–—]\s*\d|[$€₫¥%]|\d[A-Za-zÀ-ỹ]+\b")


def normalize(text: str) -> NormalizationResult:
    started = time.perf_counter()
    attempts = 0
    errors = []

    def replace(match):
        nonlocal attempts
        attempts += 1
        try:
            return num2words(int(match.group()), lang="vi")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            return match.group()

    output = _INTEGER.sub(replace, text)
    latency = (time.perf_counter() - started) * 1000
    if not attempts:
        return NormalizationResult(
            text,
            False,
            "unsupported",
            None,
            "no supported integer span",
            latency,
            {"version": VERSION, "attempted_spans": 0},
        )
    if errors:
        return NormalizationResult(
            output,
            False,
            "partial",
            "SpanError",
            "; ".join(errors),
            latency,
            {"version": VERSION, "attempted_spans": attempts},
        )
    structured = bool(_STRUCTURED.search(text))
    return NormalizationResult(
        output,
        not structured,
        "partial" if structured else "success",
        None,
        "generic integer substitution inside unsupported structure"
        if structured
        else None,
        latency,
        {"version": VERSION, "attempted_spans": attempts, "full_tn_system": False},
    )
