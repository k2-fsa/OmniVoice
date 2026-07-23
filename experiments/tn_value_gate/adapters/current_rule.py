"""Adapter for the actual executable rule in the dirty workspace."""

import time

from omnivoice.utils.vietnamese_normalization import normalize_with_trace

from .base import NormalizationResult

NAME = "current_rule"
VERSION = "workspace-frozen-20260722"


def normalize(text: str) -> NormalizationResult:
    started = time.perf_counter()
    try:
        traced = normalize_with_trace(text)
        uncertain = any(item.uncertain or not item.supported for item in traced.decisions)
        return NormalizationResult(
            traced.text,
            not uncertain,
            "partial" if uncertain else "success",
            None,
            None,
            (time.perf_counter() - started) * 1000,
            {"version": VERSION, "implementation": "omnivoice.utils.vietnamese_normalization",
             "decisions": [item.__dict__ | {"semiotic_class": item.semiotic_class.value}
                           for item in traced.decisions]},
        )
    except Exception as error:
        return NormalizationResult("", False, "runtime_error", type(error).__name__, str(error),
                                   (time.perf_counter() - started) * 1000, {"version": VERSION})
