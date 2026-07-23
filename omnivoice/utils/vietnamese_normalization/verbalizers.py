"""One deterministic Vietnamese verbalizer for every v1 candidate label."""

from functools import singledispatch

from .types import CandidateLabel
from .values import (
    CardinalValue,
    CurrencyValue,
    DateValue,
    DecimalValue,
    FractionValue,
    IdentifierValue,
    KeepValue,
    MeasurementValue,
    ParsedValue,
    PercentValue,
    RangeValue,
    RomanValue,
    ScoreValue,
    TimeValue,
    VersionValue,
    YearValue,
)

_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_SCALES = ((10**9, "tỷ"), (10**6, "triệu"), (10**3, "nghìn"))
_CURRENCIES = {
    "HK$": "đô la Hồng Kông",
    "HKD": "đô la Hồng Kông",
    "US$": "đô la Mỹ",
    "USD": "đô la Mỹ",
    "VND": "đồng",
    "ĐỒNG": "đồng",
    "Đ": "đồng",
    "₫": "đồng",
    "$": "đô la",
    "€": "euro",
    "EURO": "euro",
    "¥": "yên",
}
_UNITS = {
    "km/h": "ki lô mét trên giờ",
    "m/s": "mét trên giây",
    "kwh": "ki lô oát giờ",
    "mhz": "mê ga héc",
    "ghz": "gi ga héc",
    "km": "ki lô mét",
    "cm": "xen ti mét",
    "mm": "mi li mét",
    "kg": "ki lô gam",
    "mg": "mi li gam",
    "ml": "mi li lít",
    "gb": "gi ga bai",
    "mb": "mê ga bai",
    "tb": "tê ra bai",
    "°c": "độ xê",
    "°f": "độ ép",
    "m": "mét",
    "g": "gam",
    "l": "lít",
    "w": "oát",
    "v": "vôn",
    "hz": "héc",
}


def digits(value: str) -> str:
    """Read every digit in order, preserving all zeroes."""
    if not value or not value.isdigit():
        raise ValueError("digits expects one or more decimal digits")
    return " ".join(_DIGITS[int(char)] for char in value)


def _under_thousand(number: int, force_hundreds: bool = False) -> str:
    words = []
    hundreds, rest = divmod(number, 100)
    if hundreds or force_hundreds:
        words.extend((_DIGITS[hundreds], "trăm"))
        if 0 < rest < 10:
            words.append("linh")
    tens, unit = divmod(rest, 10)
    if tens >= 2:
        words.extend((_DIGITS[tens], "mươi"))
    elif tens == 1:
        words.append("mười")
    if unit:
        if tens >= 2 and unit == 1:
            words.append("mốt")
        elif tens >= 2 and unit == 4:
            words.append("tư")
        elif tens >= 1 and unit == 5:
            words.append("lăm")
        else:
            words.append(_DIGITS[unit])
    return " ".join(words)


def cardinal(number: int, full_lower_groups: bool = False) -> str:
    if number == 0:
        return _DIGITS[0]
    if number < 0:
        return "âm " + cardinal(-number, full_lower_groups)
    words = []
    remainder = number
    for scale, name in _SCALES:
        group, remainder = divmod(remainder, scale)
        if group:
            words.extend((cardinal(group), name))
    if remainder:
        force = full_lower_groups and number >= 1000 and remainder < 100
        words.append(_under_thousand(remainder, force))
    return " ".join(words)


def _number(value: CardinalValue | DecimalValue) -> str:
    if isinstance(value, DecimalValue):
        sign = "âm " if value.sign == "-" else ""
        return (
            f"{sign}{cardinal(int(value.integer), True)} phẩy "
            f"{digits(value.fractional)}"
        )
    sign = "âm " if value.sign == "-" else ""
    return sign + cardinal(int(value.digits), True)


def decimal(surface: str) -> str:
    """Compatibility helper retaining the original public utility."""
    from .values import parse_value

    return _number(parse_value(surface, CandidateLabel.DECIMAL))  # type: ignore[arg-type]


def _calendar_day(value: int) -> str:
    if 21 <= value <= 29:
        unit = (
            "mốt"
            if value == 21
            else "tư"
            if value == 24
            else "lăm"
            if value == 25
            else _DIGITS[value % 10]
        )
        return "hai " + unit
    return cardinal(value)


def _roman_to_int(numeral: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result, previous = 0, 0
    for char in reversed(numeral.upper()):
        current = values[char]
        result += -current if current < previous else current
        previous = current
    return result


@singledispatch
def _render(value: ParsedValue) -> str:
    raise TypeError(f"no verbalizer for {type(value).__name__}")


@_render.register
def _(value: CardinalValue) -> str:
    return _number(value)


@_render.register
def _(value: YearValue) -> str:
    return cardinal(int(value.digits), True)


@_render.register
def _(value: IdentifierValue) -> str:
    return " ".join(digits(group) for group in value.groups)


@_render.register
def _(value: DecimalValue) -> str:
    return _number(value)


@_render.register
def _(value: FractionValue) -> str:
    return f"{cardinal(int(value.numerator))} phần {cardinal(int(value.denominator))}"


@_render.register
def _(value: DateValue) -> str:
    year = f" năm {cardinal(int(value.year), True)}" if value.year else ""
    if value.order == "YMD":
        return (
            f"năm {digits(value.year or '')} tháng {cardinal(int(value.month))} "
            f"ngày {_calendar_day(int(value.day))}"
        )
    return f"{cardinal(int(value.day))} tháng {cardinal(int(value.month))}{year}"


@_render.register
def _(value: TimeValue) -> str:
    result = f"{cardinal(int(value.hour))} giờ {cardinal(int(value.minute))} phút"
    if value.second is not None:
        result += f" {cardinal(int(value.second))} giây"
    return result


@_render.register
def _(value: VersionValue) -> str:
    prefix = "vê " if value.prefix else ""
    return prefix + " chấm ".join(cardinal(int(part)) for part in value.parts)


@_render.register
def _(value: ScoreValue) -> str:
    return f"{cardinal(int(value.left))} {cardinal(int(value.right))}"


@_render.register
def _(value: RangeValue) -> str:
    return f"{cardinal(int(value.start))} đến {cardinal(int(value.end))}"


@_render.register
def _(value: CurrencyValue) -> str:
    return f"{_number(value.amount)} {_CURRENCIES[value.currency.upper()]}"


@_render.register
def _(value: MeasurementValue) -> str:
    return f"{_number(value.number)} {_UNITS[value.unit.lower()]}"


@_render.register
def _(value: PercentValue) -> str:
    suffix = "phần trăm" if value.symbol == "%" else "phần nghìn"
    return f"{_number(value.number)} {suffix}"


@_render.register
def _(value: RomanValue) -> str:
    return cardinal(_roman_to_int(value.numeral))


@_render.register
def _(value: KeepValue) -> str:
    return value.raw


_EXPECTED_TYPES = {
    CandidateLabel.CARDINAL: CardinalValue,
    CandidateLabel.YEAR: YearValue,
    CandidateLabel.IDENTIFIER: IdentifierValue,
    CandidateLabel.DECIMAL: DecimalValue,
    CandidateLabel.FRACTION: FractionValue,
    CandidateLabel.DATE: DateValue,
    CandidateLabel.TIME: TimeValue,
    CandidateLabel.VERSION: VersionValue,
    CandidateLabel.SCORE: ScoreValue,
    CandidateLabel.RANGE: RangeValue,
    CandidateLabel.CURRENCY: CurrencyValue,
    CandidateLabel.MEASUREMENT: MeasurementValue,
    CandidateLabel.PERCENT: PercentValue,
    CandidateLabel.ROMAN: RomanValue,
    CandidateLabel.KEEP: KeepValue,
}


def verbalize(parsed_value: ParsedValue, label: CandidateLabel) -> str:
    """Route one explicit label to exactly one typed deterministic renderer."""
    expected = _EXPECTED_TYPES[label]
    if not isinstance(parsed_value, expected):
        raise TypeError(
            f"{label.value} requires {expected.__name__}, "
            f"got {type(parsed_value).__name__}"
        )
    return _render(parsed_value)
