"""Conservative detect-classify-parse-verbalize orchestration."""

import logging

from .classifier import classify
from .detector import detect
from .types import CandidateLabel, DecisionTrace, NormalizationResult, SemioticClass
from .values import parse_value
from .verbalizers import verbalize

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    SemioticClass.CARDINAL: CandidateLabel.CARDINAL,
    SemioticClass.IDENTIFIER: CandidateLabel.IDENTIFIER,
    SemioticClass.PHONE: CandidateLabel.IDENTIFIER,
    SemioticClass.DATE: CandidateLabel.DATE,
    SemioticClass.TIME: CandidateLabel.TIME,
    SemioticClass.FRACTION: CandidateLabel.FRACTION,
    SemioticClass.MONEY: CandidateLabel.CURRENCY,
}


def normalize_with_trace(text: str) -> NormalizationResult:
    """Normalize only confidently classified spans and expose all abstentions."""
    chunks, traces, last = [], [], 0
    for span in detect(text):
        decision = classify(text, span)
        supported = decision.semiotic_class != SemioticClass.UNSUPPORTED
        output = span.surface
        reason = decision.reason
        if supported:
            label = _LABEL_MAP[decision.semiotic_class]
            try:
                output = verbalize(parse_value(span.surface, label), label)
            except ValueError as error:
                supported = False
                reason = f"parse failed ({error}); preserved raw span"
                logger.warning(
                    "Vietnamese normalization could not parse %r as %s; "
                    "keeping it unchanged.",
                    span.surface,
                    label.value,
                )
        chunks.extend((text[last : span.start], output))
        traces.append(
            DecisionTrace(
                span.start,
                span.end,
                span.surface,
                decision.semiotic_class,
                output,
                output != span.surface,
                decision.uncertain,
                supported,
                reason,
            )
        )
        last = span.end
    chunks.append(text[last:])
    return NormalizationResult("".join(chunks), tuple(traces))


def normalize(text: str) -> str:
    return normalize_with_trace(text).text
