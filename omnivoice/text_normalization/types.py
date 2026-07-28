"""Typed boundaries for the conservative Vietnamese normalization pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EntityLabel(str, Enum):
    CARDINAL = "CARDINAL"
    DECIMAL = "DECIMAL"
    YEAR = "YEAR"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    MEASUREMENT = "MEASUREMENT"
    PERCENT = "PERCENT"
    PHONE = "PHONE"
    IDENTIFIER = "IDENTIFIER"
    FRACTION = "FRACTION"
    SCORE = "SCORE"
    ORDINAL = "ORDINAL"


@dataclass(frozen=True)
class CandidateSpan:
    start: int
    end: int
    label: str
    text: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class NumberValue:
    sign: str
    integer: str
    fractional: str | None = None


@dataclass(frozen=True)
class DateValue:
    day: int
    month: int
    year: int


@dataclass(frozen=True)
class TimeValue:
    hour: int
    minute: int | None
    second: int | None
    period: str | None = None


@dataclass(frozen=True)
class PairValue:
    left: str
    right: str


@dataclass(frozen=True)
class MoneyValue:
    amount: NumberValue
    multiplier: str | None
    currency: str


@dataclass(frozen=True)
class MeasurementValue:
    amount: NumberValue
    unit: str


@dataclass(frozen=True)
class DigitSequenceValue:
    digits: str


CanonicalValue = (
    NumberValue
    | DateValue
    | TimeValue
    | PairValue
    | MoneyValue
    | MeasurementValue
    | DigitSequenceValue
)


@dataclass(frozen=True)
class Diagnostic:
    candidate: CandidateSpan
    action: str
    reason: str
    effective_label: str | None = None
    replacement: str | None = None
    value: Any = None


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    diagnostics: tuple[Diagnostic, ...]
