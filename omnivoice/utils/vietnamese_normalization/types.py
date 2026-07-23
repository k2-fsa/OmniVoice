"""Small immutable interface between Vietnamese normalization stages."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class CandidateLabel(str, Enum):
    """Closed v1 taxonomy for candidate generation and gold annotation."""

    CARDINAL = "CARDINAL"
    YEAR = "YEAR"
    IDENTIFIER = "IDENTIFIER"
    DECIMAL = "DECIMAL"
    FRACTION = "FRACTION"
    DATE = "DATE"
    TIME = "TIME"
    VERSION = "VERSION"
    SCORE = "SCORE"
    RANGE = "RANGE"
    CURRENCY = "CURRENCY"
    MEASUREMENT = "MEASUREMENT"
    PERCENT = "PERCENT"
    ROMAN = "ROMAN"
    KEEP = "KEEP"


class SemioticClass(str, Enum):
    CARDINAL = "CARDINAL"
    IDENTIFIER = "IDENTIFIER"
    PHONE = "PHONE"
    DATE = "DATE"
    TIME = "TIME"
    FRACTION = "FRACTION"
    MONEY = "MONEY"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class DetectedSpan:
    start: int
    end: int
    surface: str
    syntax: str
    candidate_labels: Tuple[CandidateLabel, ...] = ()


@dataclass(frozen=True)
class Classification:
    semiotic_class: SemioticClass
    uncertain: bool
    reason: str


@dataclass(frozen=True)
class DecisionTrace:
    start: int
    end: int
    surface: str
    semiotic_class: SemioticClass
    output: str
    changed: bool
    uncertain: bool
    supported: bool
    reason: str


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    decisions: Tuple[DecisionTrace, ...]
