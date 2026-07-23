"""Common adapters for text-normalization benchmark methods."""

from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from typing import Any, Callable

from num2words import num2words

from omnivoice.utils.text import _num2words_segment, normalize_text
from omnivoice.utils.vietnamese_normalization import normalize_with_trace


@dataclass(frozen=True)
class NormalizationResult:
    output_text: str
    available: bool = True
    changed: bool = False
    uncertain: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def identity(text: str, language: str = "vi") -> NormalizationResult:
    return NormalizationResult(
        text, metadata={"version": "identity", "language": language}
    )


def omnivoice_fallback(text: str, language: str = "vi") -> NormalizationResult:
    """Call the exact generic wrapper used by OmniVoice before VI routing."""
    output = _num2words_segment(text, language)
    return NormalizationResult(
        output,
        changed=output != text,
        metadata={
            "version": version("num2words"),
            "call": "omnivoice.utils.text._num2words_segment",
        },
    )


def direct_num2words(text: str, language: str = "vi") -> NormalizationResult:
    import re

    errors = []

    def replace(match):
        try:
            return num2words(int(match.group()), lang=language)
        except Exception as error:  # package behavior is benchmark output
            errors.append(f"{type(error).__name__}: {error}")
            return match.group()

    output = re.sub(r"\d+", replace, text)
    return NormalizationResult(
        output,
        changed=output != text,
        uncertain=bool(errors),
        error="; ".join(errors) or None,
        metadata={
            "version": version("num2words"),
            "call": "num2words(int(span), lang='vi')",
        },
    )


def contextual_rule(text: str, language: str = "vi") -> NormalizationResult:
    traced = normalize_with_trace(text)
    uncertain = any(item.uncertain or not item.supported for item in traced.decisions)
    return NormalizationResult(
        traced.text,
        changed=traced.text != text,
        uncertain=uncertain,
        metadata={
            "version": "frozen-poc-20260722",
            "classes": [item.semiotic_class.value for item in traced.decisions],
            "decisions": [
                item.__dict__ | {"semiotic_class": item.semiotic_class.value}
                for item in traced.decisions
            ],
        },
    )


def omnivoice_current(text: str, language: str = "vi") -> NormalizationResult:
    """Call the public API in the current dirty workspace."""
    output = normalize_text(text, language)
    return NormalizationResult(
        output,
        changed=output != text,
        metadata={
            "version": "workspace",
            "call": "omnivoice.utils.text.normalize_text",
        },
    )


METHODS: dict[str, Callable[[str, str], NormalizationResult]] = {
    "identity": identity,
    "omnivoice_current": omnivoice_current,
    "omnivoice_fallback": omnivoice_fallback,
    "direct_num2words": direct_num2words,
    "contextual_rule": contextual_rule,
}


def unavailable(error: str, version_value: str | None = None) -> NormalizationResult:
    return NormalizationResult(
        "",
        available=False,
        uncertain=True,
        error=error,
        metadata={"version": version_value},
    )
