"""Lossless parsed values for deterministic Vietnamese verbalization."""

from dataclasses import dataclass
import re
from typing import Union

from .types import CandidateLabel


@dataclass(frozen=True)
class CardinalValue:
    sign: str
    digits: str


@dataclass(frozen=True)
class YearValue:
    digits: str


@dataclass(frozen=True)
class IdentifierValue:
    groups: tuple[str, ...]


@dataclass(frozen=True)
class DecimalValue:
    sign: str
    integer: str
    fractional: str
    separator: str


@dataclass(frozen=True)
class FractionValue:
    numerator: str
    denominator: str
    separator: str


@dataclass(frozen=True)
class DateValue:
    day: str
    month: str
    year: str | None
    separator: str
    order: str


@dataclass(frozen=True)
class TimeValue:
    hour: str
    minute: str
    second: str | None


@dataclass(frozen=True)
class VersionValue:
    prefix: str
    parts: tuple[str, ...]
    separator: str


@dataclass(frozen=True)
class ScoreValue:
    left: str
    right: str
    separator: str


@dataclass(frozen=True)
class RangeValue:
    start: str
    end: str
    separator: str


@dataclass(frozen=True)
class CurrencyValue:
    amount: CardinalValue | DecimalValue
    currency: str
    marker_first: bool


@dataclass(frozen=True)
class MeasurementValue:
    number: CardinalValue | DecimalValue
    unit: str


@dataclass(frozen=True)
class PercentValue:
    number: CardinalValue | DecimalValue
    symbol: str


@dataclass(frozen=True)
class RomanValue:
    numeral: str


@dataclass(frozen=True)
class KeepValue:
    raw: str


ParsedValue = Union[
    CardinalValue,
    YearValue,
    IdentifierValue,
    DecimalValue,
    FractionValue,
    DateValue,
    TimeValue,
    VersionValue,
    ScoreValue,
    RangeValue,
    CurrencyValue,
    MeasurementValue,
    PercentValue,
    RomanValue,
    KeepValue,
]

_CURRENCY_MARKERS = (
    "HK$",
    "US$",
    "USD",
    "HKD",
    "VND",
    "đồng",
    "euro",
    "₫",
    "$",
    "€",
    "¥",
    "đ",
)
_UNITS = (
    "km/h",
    "m/s",
    "kWh",
    "MHz",
    "GHz",
    "km",
    "cm",
    "mm",
    "kg",
    "mg",
    "ml",
    "GB",
    "MB",
    "TB",
    "°C",
    "°F",
    "m",
    "g",
    "l",
    "W",
    "V",
    "Hz",
)


def _cardinal(surface: str) -> CardinalValue:
    match = re.fullmatch(r"([+-]?)(\d+(?:[ .]\d{3})*)", surface.strip())
    if not match:
        raise ValueError("malformed cardinal")
    sign, grouped = match.groups()
    return CardinalValue(sign, re.sub(r"[ .]", "", grouped))


def _decimal(surface: str) -> DecimalValue:
    match = re.fullmatch(r"([+-]?)(\d+)([.,])(\d+)", surface.strip())
    if not match:
        raise ValueError("malformed decimal")
    sign, integer, separator, fractional = match.groups()
    return DecimalValue(sign, integer, fractional, separator)


def _number(surface: str) -> CardinalValue | DecimalValue:
    return _decimal(surface) if re.search(r"\d[.,]\d", surface) else _cardinal(surface)


def parse_value(surface: str, label: CandidateLabel) -> ParsedValue:
    """Parse one already-classified span without lossy numeric conversion."""
    if not isinstance(surface, str) or not surface:
        raise ValueError("surface must be a nonempty string")
    stripped = surface.strip()
    if label == CandidateLabel.KEEP:
        return KeepValue(surface)
    if label == CandidateLabel.CARDINAL:
        return _cardinal(stripped)
    if label == CandidateLabel.YEAR:
        if not re.fullmatch(r"\d{1,4}", stripped):
            raise ValueError("malformed year")
        return YearValue(stripped)
    if label == CandidateLabel.IDENTIFIER:
        groups = tuple(re.findall(r"\d+", stripped))
        if not groups or re.sub(r"[\d\s().+-]", "", stripped):
            raise ValueError("malformed identifier")
        return IdentifierValue(groups)
    if label == CandidateLabel.DECIMAL:
        return _decimal(stripped)
    if label == CandidateLabel.FRACTION:
        match = re.fullmatch(r"(\d+)(\s*/\s*)(\d+)", stripped)
        if not match or int(match.group(3)) == 0:
            raise ValueError("malformed fraction")
        numerator, separator, denominator = match.groups()
        return FractionValue(numerator, denominator, separator)
    if label == CandidateLabel.DATE:
        parts = re.split(r"([./-])", stripped)
        if len(parts) == 3:
            day, separator, month = parts
            year, order = None, "DM"
        elif len(parts) == 5 and parts[1] == parts[3]:
            first, separator, second, _, third = parts
            if len(first) == 4:
                year, month, day, order = first, second, third, "YMD"
            else:
                day, month, year, order = first, second, third, "DMY"
        else:
            raise ValueError("malformed date")
        components = (day, month) if year is None else (day, month, year)
        if not all(item.isdigit() for item in components):
            raise ValueError("malformed date")
        if not 1 <= int(month) <= 12 or not 1 <= int(day) <= 31:
            raise ValueError("invalid date")
        return DateValue(day, month, year, separator, order)
    if label == CandidateLabel.TIME:
        parts = stripped.split(":")
        if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
            raise ValueError("malformed time")
        hour, minute = parts[:2]
        second = parts[2] if len(parts) == 3 else None
        if int(hour) > 23 or int(minute) > 59 or (
            second is not None and int(second) > 59
        ):
            raise ValueError("invalid time")
        return TimeValue(hour, minute, second)
    if label == CandidateLabel.VERSION:
        match = re.fullmatch(r"([vV]?)(\d+(?:([._-])\d+)+)", stripped)
        if not match:
            raise ValueError("malformed version")
        prefix, body, separator = match.groups()
        if len(set(re.findall(r"[._-]", body))) != 1:
            raise ValueError("mixed version separators")
        return VersionValue(prefix, tuple(body.split(separator)), separator)
    if label in (CandidateLabel.SCORE, CandidateLabel.RANGE):
        match = re.fullmatch(r"([+-]?\d+)(\s*[-–—]\s*)([+-]?\d+)", stripped)
        if not match:
            raise ValueError(f"malformed {label.value.lower()}")
        value_type = ScoreValue if label == CandidateLabel.SCORE else RangeValue
        left, separator, right = match.groups()
        return value_type(left, right, separator)
    if label == CandidateLabel.CURRENCY:
        marker_pattern = "|".join(
            re.escape(marker) for marker in sorted(_CURRENCY_MARKERS, key=len, reverse=True)
        )
        match = re.fullmatch(
            rf"\s*(?:(?P<before>{marker_pattern})\s*(?P<a>[\d.,]+)"
            rf"|(?P<b>[\d.,]+)\s*(?P<after>{marker_pattern}))\s*",
            surface,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("unsupported or malformed currency")
        amount = match.group("a") or match.group("b")
        marker = match.group("before") or match.group("after")
        parsed_amount = (
            _cardinal(amount)
            if re.fullmatch(r"\d{1,3}(?:[ .]\d{3})+", amount)
            else _number(amount)
        )
        return CurrencyValue(
            parsed_amount, marker, bool(match.group("before"))
        )
    if label == CandidateLabel.MEASUREMENT:
        unit_pattern = "|".join(
            re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True)
        )
        match = re.fullmatch(
            rf"\s*(?P<number>[+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>{unit_pattern})\s*",
            surface,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("unsupported or malformed measurement")
        return MeasurementValue(_number(match.group("number")), match.group("unit"))
    if label == CandidateLabel.PERCENT:
        match = re.fullmatch(r"\s*([+-]?\d+(?:[.,]\d+)?)\s*(%|‰)\s*", surface)
        if not match:
            raise ValueError("malformed percent")
        return PercentValue(_number(match.group(1)), match.group(2))
    if label == CandidateLabel.ROMAN:
        if not re.fullmatch(
            r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})",
            stripped.upper(),
        ):
            raise ValueError("malformed Roman numeral")
        return RomanValue(stripped)
    raise ValueError(f"unsupported label: {label}")
