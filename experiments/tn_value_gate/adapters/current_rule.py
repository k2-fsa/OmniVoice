"""Adapter for the actual executable rule in the dirty workspace."""

import time

from omnivoice.utils.text import normalize_text

from .base import NormalizationResult

NAME = "current_rule"
VERSION = "workspace-frozen-20260722"


def normalize(text: str) -> NormalizationResult:
    started = time.perf_counter()
    try:
        output = normalize_text(text, "vi")
        return NormalizationResult(
            output,
            True,
            "success",
            None,
            None,
            (time.perf_counter() - started) * 1000,
            {
                "version": VERSION,
                "implementation": "omnivoice.utils.text.normalize_text",
            },
        )
    except Exception as error:
        return NormalizationResult("", False, "runtime_error", type(error).__name__, str(error),
                                   (time.perf_counter() - started) * 1000, {"version": VERSION})
