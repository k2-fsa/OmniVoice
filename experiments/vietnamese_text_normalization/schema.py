"""JSONL schemas and strict validators for Vietnamese TN development data."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from omnivoice.utils.vietnamese_normalization.types import CandidateLabel

TAXONOMY_V1 = frozenset(label.value for label in CandidateLabel)


@dataclass(frozen=True)
class GoldSpan:
    start: int
    end: int
    surface: str
    label: str
    spoken: Optional[str] = None
    acceptable_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizationCase:
    """One sentence and its smallest-complete, non-overlapping gold spans."""

    id: str
    text: str
    spans: list[GoldSpan]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_label(label: Any, field_name: str) -> str:
    if not isinstance(label, str) or label not in TAXONOMY_V1:
        allowed = ", ".join(sorted(TAXONOMY_V1))
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return label


def validate_case(value: dict[str, Any]) -> NormalizationCase:
    """Validate offsets, surfaces, taxonomy, ordering, and JSON-compatible fields."""
    if not isinstance(value, dict):
        raise ValueError("case must be an object")
    for name in ("id", "text", "source"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"{name} must be a nonempty string")
    raw_spans = value.get("spans")
    if not isinstance(raw_spans, list):
        raise ValueError("spans must be a list")
    spans = []
    previous_end = 0
    for index, raw in enumerate(raw_spans):
        if not isinstance(raw, dict):
            raise ValueError(f"spans[{index}] must be an object")
        start, end, surface = raw.get("start"), raw.get("end"), raw.get("surface")
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError(f"spans[{index}].start must be a nonnegative integer")
        if not isinstance(end, int) or isinstance(end, bool) or end <= start:
            raise ValueError(f"spans[{index}].end must be greater than start")
        if end > len(value["text"]):
            raise ValueError(f"spans[{index}] is outside text")
        if not isinstance(surface, str) or value["text"][start:end] != surface:
            raise ValueError(
                f"spans[{index}] violates text[start:end] == surface"
            )
        if start < previous_end:
            raise ValueError("spans must be sorted and non-overlapping")
        label = _validate_label(raw.get("label"), f"spans[{index}].label")
        acceptable = raw.get("acceptable_labels", [])
        if not isinstance(acceptable, list):
            raise ValueError(f"spans[{index}].acceptable_labels must be a list")
        acceptable = [
            _validate_label(item, f"spans[{index}].acceptable_labels")
            for item in acceptable
        ]
        spoken = raw.get("spoken")
        if spoken is not None and not isinstance(spoken, str):
            raise ValueError(f"spans[{index}].spoken must be null or a string")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"spans[{index}].metadata must be an object")
        spans.append(
            GoldSpan(start, end, surface, label, spoken, acceptable, metadata)
        )
        previous_end = end
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    return NormalizationCase(
        value["id"], value["text"], spans, value["source"], metadata
    )


def load_cases(path: Path) -> list[NormalizationCase]:
    """Load a dev-only JSONL file and reject duplicate ids or malformed lines."""
    cases, ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = validate_case(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if case.id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {case.id!r}")
            ids.add(case.id)
            cases.append(case)
    return cases


@dataclass(frozen=True)
class CanonicalRecord:
    id: str
    text: str
    numeric_span: str
    semiotic_class: str
    domain: Optional[str]
    expected_spoken_text: str
    acceptable_variants: list[str]
    source: str
    template_group: Optional[str]
    audio_path: Optional[str]
    human_rating: Optional[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_record(value: dict[str, Any]) -> CanonicalRecord:
    """Validate one canonical record without coercing malformed values."""
    required_strings = (
        "id",
        "text",
        "numeric_span",
        "semiotic_class",
        "expected_spoken_text",
        "source",
    )
    for name in required_strings:
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"{name} must be a nonempty string")
    variants = value.get("acceptable_variants")
    if not isinstance(variants, list) or not all(
        isinstance(item, str) for item in variants
    ):
        raise ValueError("acceptable_variants must be a list of strings")
    if value["numeric_span"] not in value["text"]:
        raise ValueError("numeric_span must occur in text")
    for nullable in ("domain", "template_group", "audio_path"):
        if value.get(nullable) is not None and not isinstance(value[nullable], str):
            raise ValueError(f"{nullable} must be null or a string")
    if not isinstance(value.get("metadata", {}), dict):
        raise ValueError("metadata must be an object")
    return CanonicalRecord(**value)
