"""Safe orchestration for externally supplied NER candidate spans."""

import logging
import math
import re
from collections.abc import Iterable

from .parsers import parse
from .types import (
    CandidateSpan,
    Diagnostic,
    EntityLabel,
    NormalizationResult,
)
from .verbalize import verbalize

logger = logging.getLogger(__name__)
ID_CONTEXT_RE = re.compile(
    r"(?:mã sinh viên|mã số|số tài khoản|cccd|mã đơn hàng|phòng)\s*"
    r"(?:của[^:]{0,30})?"
    r"(?:là|:)?\s*$",
    re.IGNORECASE,
)
ORDINAL_CONTEXT_RE = re.compile(r"(?:thứ|hạng|lần thứ)\s*$", re.IGNORECASE)


def _effective_label(text: str, start: int, label: EntityLabel) -> tuple[EntityLabel, str]:
    context = text[max(0, start - 80) : start]
    if label in {
        EntityLabel.PHONE,
        EntityLabel.CARDINAL,
        EntityLabel.ORDINAL,
    } and ID_CONTEXT_RE.search(context):
        return EntityLabel.IDENTIFIER, (
            f"{label.value} overridden by explicit identifier context"
        )
    return label, "model label accepted"


def _normalize_candidates(
    text: str, candidates: Iterable[CandidateSpan | dict]
) -> NormalizationResult:
    diagnostics: list[Diagnostic] = []
    valid: list[tuple[CandidateSpan, int, int, str, int, float]] = []
    for input_order, raw in enumerate(candidates):
        try:
            candidate = raw if isinstance(raw, CandidateSpan) else CandidateSpan(
                start=int(raw["start"]), end=int(raw["end"]),
                label=str(raw.get("label", raw.get("entity_group"))),
                text=raw.get("text"), score=raw.get("score"),
            )
        except (KeyError, TypeError, ValueError) as error:
            placeholder = CandidateSpan(-1, -1, "UNKNOWN")
            diagnostics.append(Diagnostic(placeholder, "preserved", f"malformed candidate: {error}"))
            continue
        if not 0 <= candidate.start < candidate.end <= len(text):
            diagnostics.append(Diagnostic(candidate, "preserved", "invalid span bounds"))
            continue
        source = text[candidate.start : candidate.end]
        if candidate.text is not None and candidate.text != source:
            diagnostics.append(Diagnostic(candidate, "preserved", "candidate text/source mismatch"))
            continue
        left = len(source) - len(source.lstrip())
        right = len(source.rstrip())
        semantic = source[left:right]
        try:
            score = float(candidate.score) if candidate.score is not None else 0.0
        except (TypeError, ValueError):
            diagnostics.append(Diagnostic(candidate, "preserved", "invalid confidence score"))
            continue
        if not math.isfinite(score):
            diagnostics.append(Diagnostic(candidate, "preserved", "invalid confidence score"))
            continue
        valid.append(
            (
                candidate,
                candidate.start + left,
                candidate.start + right,
                semantic,
                input_order,
                score,
            )
        )

    # Highest score wins; then longer span; then earlier input order. Exact
    # duplicates collapse. Conflicting losers are explicitly diagnosed.
    accepted: list[tuple[CandidateSpan, int, int, str, int, float]] = []
    ranked = sorted(
        valid,
        key=lambda item: (-item[5], -(item[2] - item[1]), item[4]),
    )
    for item in ranked:
        candidate, start, end, _, _, _ = item
        if any(
            start < other_end and end > other_start
            for _, other_start, other_end, _, _, _ in accepted
        ):
            diagnostics.append(Diagnostic(candidate, "preserved", "overlap lost deterministic conflict"))
        else:
            accepted.append(item)

    replacements: list[tuple[int, int, str]] = []
    for candidate, start, end, semantic, _, _ in sorted(
        accepted, key=lambda item: item[1]
    ):
        try:
            label = EntityLabel(candidate.label.upper())
        except ValueError:
            diagnostics.append(Diagnostic(candidate, "preserved", "unsupported label"))
            continue
        effective, reason = _effective_label(text, start, label)
        # Do not duplicate context words excluded from ordinal/date spans.
        try:
            if effective == EntityLabel.CARDINAL and re.fullmatch(
                r"[+\-−]?\d[.,]\d{3}", semantic
            ):
                raise ValueError("ambiguous decimal/grouping form")
            value = parse(semantic, effective)
            replacement = verbalize(value, effective, surface=semantic)
            if effective == EntityLabel.ORDINAL and not ORDINAL_CONTEXT_RE.search(text[:start]):
                replacement = f"thứ {replacement}"
            if effective == EntityLabel.DATE:
                prefix = text[max(0, start - 12) : start].lower()
                if re.search(r"ngày\s*$", prefix):
                    replacement = replacement.removeprefix("ngày ")
            replacements.append((start, end, replacement))
            diagnostics.append(Diagnostic(candidate, "normalized", reason, effective.value,
                                          replacement, value))
        except (ValueError, OverflowError) as error:
            logger.debug("Preserving %r: %s", semantic, error)
            diagnostics.append(Diagnostic(candidate, "preserved", str(error), effective.value))

    output = text
    for start, end, replacement in sorted(replacements, reverse=True):
        output = output[:start] + replacement + output[end:]
    return NormalizationResult(output, tuple(diagnostics))


def normalize_candidates(
    text: str, candidates: Iterable[CandidateSpan | dict]
) -> NormalizationResult:
    """Normalize candidates atomically and preserve the full input on failure."""
    try:
        return _normalize_candidates(text, candidates)
    except Exception as error:
        logger.warning("Candidate normalization failed; preserving input: %s", error)
        return NormalizationResult(
            text,
            (
                Diagnostic(
                    CandidateSpan(0, len(text), "UNKNOWN"),
                    "preserved",
                    f"pipeline failure: {error}",
                ),
            ),
        )


def normalize_from_ner(text: str, detector) -> NormalizationResult:
    """Run an injected detector at the fail-closed sentence boundary."""
    try:
        return normalize_candidates(text, detector(text))
    # This deliberately broad catch is the outer normalization service
    # boundary: detector libraries and injected implementations may raise
    # arbitrary errors, while the product policy is to preserve the complete
    # original target and never expose a partially rewritten sentence.
    except Exception as error:
        logger.warning("NER normalization failed; preserving input: %s", error)
        return NormalizationResult(
            text, (Diagnostic(CandidateSpan(0, len(text), "UNKNOWN"), "preserved",
                              f"detector failure: {error}"),)
        )
