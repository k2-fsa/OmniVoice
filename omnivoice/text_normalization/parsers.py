"""Label-specific, lossless parsers and validators."""

import calendar
import re

from .types import (
    DateValue,
    DigitSequenceValue,
    EntityLabel,
    MeasurementValue,
    MoneyValue,
    NumberValue,
    PairValue,
    TimeValue,
)

SPACE = r"[\s\u00a0\u202f]"
SIGN = r"[+\-−]?"
GROUPED_INTEGER_RE = re.compile(rf"({SIGN})(\d{{1,3}}(?:{SPACE}\d{{3}})+|\d+)")
DOT_GROUPED_RE = re.compile(rf"({SIGN})(\d{{1,3}}(?:\.\d{{3}})+)")
PLAIN_INTEGER_RE = re.compile(rf"({SIGN})(\d+)")
DECIMAL_RE = re.compile(rf"({SIGN})(\d[\d.,{SPACE}]*)([.,])(\d+)")
DATE_RE = re.compile(r"(\d{1,4})([/.\-–—])(\d{1,2})\2(\d{1,4})")
TIME_RE = re.compile(
    r"(\d{1,2})(?:"
    r"(?::|[hH]|[ \t]+(?:giờ|h)[ \t]*)(\d{1,2})(?::(\d{1,2}))?"
    r"|[ \t]+giờ"
    r")?[ \t]*(AM|PM)?",
    re.IGNORECASE,
)
PAIR_RE = re.compile(r"([+-]?\d+)\s*([-–—:])\s*([+-]?\d+)")
FRACTION_RE = re.compile(r"([−-]?\d+)\s*/\s*(\d+)")
PHONE_ALLOWED_RE = re.compile(r"[+\d\s\u00a0\u202f().-]+")
MONEY_RE = re.compile(
    rf"(?P<prefix>₫|\$|€)?{SPACE}*(?P<number>{SIGN}\d[\d.,\s\u00a0\u202f]*)"
    rf"{SPACE}*(?P<multiplier>triệu|tỷ)?{SPACE}*"
    r"(?P<currency>đồng|VND|₫|USD|\$|EUR|€|đ)?",
    re.IGNORECASE,
)
MEASUREMENT_RE = re.compile(
    rf"(?P<number>{SIGN}\d[\d.,\s\u00a0\u202f]*){SPACE}*"
    r"(?P<unit>km/h|m/s|kWh|MHz|GHz|km|cm|mm|kg|mg|ml|GB|MB|TB|°C|°F|m|g|l|W|V|Hz)",
    re.IGNORECASE,
)

UNITS = {
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

MAX_DIGITS = 18


class CurrencyMarkerMissingError(ValueError):
    """A money-shaped value contained no currency marker."""


def _sign(raw: str) -> str:
    return "-" if raw in {"-", "−"} else raw


def parse_cardinal(surface: str) -> NumberValue:
    """Parse integers; dot grouping is accepted only when structurally complete."""
    match = GROUPED_INTEGER_RE.fullmatch(surface) or DOT_GROUPED_RE.fullmatch(surface)
    match = match or PLAIN_INTEGER_RE.fullmatch(surface)
    if not match:
        raise ValueError("malformed or ambiguous integer grouping")
    sign, body = match.groups()
    digits = re.sub(r"[\s\u00a0\u202f.]", "", body)
    if len(digits) > MAX_DIGITS:
        raise ValueError(f"integer exceeds {MAX_DIGITS}-digit limit")
    return NumberValue(_sign(sign), digits)


def parse_decimal(surface: str) -> NumberValue:
    """Parse mixed-locale decimals using the rightmost separator as decimal mark.

    A single separator followed by exactly three digits (for example ``1.234``)
    is deliberately ambiguous and rejected.
    """
    raw = surface.strip()
    separators = [index for index, char in enumerate(raw) if char in ".,"]
    if not separators:
        raise ValueError("decimal separator missing")
    decimal_at = separators[-1]
    left, fractional = raw[:decimal_at], raw[decimal_at + 1 :]
    if not fractional.isdigit():
        raise ValueError("malformed fractional component")
    if len(separators) == 1 and len(fractional) == 3:
        raise ValueError("ambiguous decimal/grouping form")
    sign = ""
    if left[:1] in "+-−":
        sign, left = _sign(left[0]), left[1:]
    grouping = "," if raw[decimal_at] == "." else "."
    if grouping in left:
        groups = left.split(grouping)
        if not groups[0].isdigit() or not 1 <= len(groups[0]) <= 3:
            raise ValueError("malformed grouping")
        if not all(len(group) == 3 and group.isdigit() for group in groups[1:]):
            raise ValueError("malformed grouping")
        integer = "".join(groups)
    else:
        integer = re.sub(r"[\s\u00a0\u202f]", "", left)
        if not integer.isdigit():
            raise ValueError("malformed integer component")
    if len(integer) > MAX_DIGITS:
        raise ValueError("decimal integer component too large")
    return NumberValue(sign, integer, fractional)


def parse_number(surface: str) -> NumberValue:
    """Parse one complete plain integer, grouped integer, or decimal."""
    if re.search(r"[.,]", surface) and not re.fullmatch(
        r"[+\-−]?\d{1,3}(?:[.\s\u00a0\u202f]\d{3})+",
        surface,
    ):
        return parse_decimal(surface)
    return parse_cardinal(surface)


def parse_year(surface: str) -> NumberValue:
    if not re.fullmatch(r"\d{4}", surface):
        raise ValueError("only four-digit years are supported")
    year = int(surface)
    if not 1000 <= year <= 2999:
        raise ValueError("year outside supported range")
    return NumberValue("", surface)


def parse_date(surface: str) -> DateValue:
    match = DATE_RE.fullmatch(surface)
    if not match:
        raise ValueError("malformed date")
    first, _, second, third = match.groups()
    if len(first) == 4:
        year, month, day = map(int, (first, second, third))
    else:
        day, month, year = map(int, (first, second, third))
        if len(third) == 2:
            year += 2000  # documented bounded policy: 00..99 => 2000..2099
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        raise ValueError("invalid calendar date")
    if not 1 <= day <= calendar.monthrange(year, month)[1]:
        raise ValueError("invalid calendar date")
    return DateValue(day, month, year)


def parse_time(surface: str) -> TimeValue:
    match = TIME_RE.fullmatch(surface)
    if not match:
        raise ValueError("malformed time")
    hour_raw, minute_raw, second_raw, period = match.groups()
    if minute_raw is None and second_raw is not None:
        raise ValueError("seconds require minutes")
    hour = int(hour_raw)
    minute = int(minute_raw) if minute_raw is not None else None
    second = int(second_raw) if second_raw is not None else None
    if period:
        if not 1 <= hour <= 12:
            raise ValueError("invalid 12-hour clock")
    elif not 0 <= hour <= 23:
        raise ValueError("invalid hour")
    if minute is not None and not 0 <= minute <= 59:
        raise ValueError("invalid minute")
    if second is not None and not 0 <= second <= 59:
        raise ValueError("invalid second")
    return TimeValue(hour, minute, second, period.upper() if period else None)


def parse_fraction(surface: str) -> PairValue:
    match = FRACTION_RE.fullmatch(surface)
    if not match or int(match.group(2)) == 0:
        raise ValueError("invalid fraction")
    return PairValue(match.group(1).replace("−", "-"), match.group(2))


def parse_score(surface: str) -> PairValue:
    match = PAIR_RE.fullmatch(surface)
    if not match:
        raise ValueError("malformed score")
    return PairValue(match.group(1), match.group(3))


def parse_digits(surface: str) -> DigitSequenceValue:
    if not PHONE_ALLOWED_RE.fullmatch(surface):
        raise ValueError("unsupported digit sequence characters")
    digits = "".join(re.findall(r"\d", surface))
    if not digits:
        raise ValueError("empty digit sequence")
    return DigitSequenceValue(digits)


def parse_money(surface: str) -> MoneyValue:
    match = MONEY_RE.fullmatch(surface)
    if not match:
        raise ValueError("malformed or unsupported money")
    prefix, number, multiplier, suffix = (
        match.group("prefix"),
        match.group("number").strip(),
        match.group("multiplier"),
        match.group("currency"),
    )
    marker = prefix or suffix
    if not marker:
        raise CurrencyMarkerMissingError("currency marker missing")
    amount = parse_number(number)
    currencies = {
        "₫": "đồng",
        "đ": "đồng",
        "vnd": "đồng",
        "đồng": "đồng",
        "$": "đô la Mỹ",
        "usd": "đô la Mỹ",
        "€": "euro",
        "eur": "euro",
    }
    return MoneyValue(
        amount, multiplier.lower() if multiplier else None, currencies[marker.lower()]
    )


def parse_measurement(surface: str) -> MeasurementValue:
    match = MEASUREMENT_RE.fullmatch(surface)
    if not match:
        raise ValueError("malformed or unsupported measurement")
    number = match.group("number").strip()
    amount = (
        parse_decimal(number)
        if re.search(r"[.,]", number)
        and not (re.fullmatch(r"[+\-−]?\d{1,3}(?:[.\s\u00a0\u202f]\d{3})+", number))
        else parse_cardinal(number)
    )
    return MeasurementValue(amount, UNITS[match.group("unit").lower()])


def parse(surface: str, label: EntityLabel):
    parsers = {
        EntityLabel.CARDINAL: parse_cardinal,
        EntityLabel.DECIMAL: parse_decimal,
        EntityLabel.YEAR: parse_year,
        EntityLabel.DATE: parse_date,
        EntityLabel.TIME: parse_time,
        EntityLabel.MONEY: parse_money,
        EntityLabel.MEASUREMENT: parse_measurement,
        EntityLabel.PERCENT: lambda value: parse_decimal(value.rstrip(" %"))
        if any(char in value.rstrip(" %") for char in ".,")
        else parse_cardinal(value.rstrip(" %")),
        EntityLabel.PHONE: parse_digits,
        EntityLabel.IDENTIFIER: parse_digits,
        EntityLabel.FRACTION: parse_fraction,
        EntityLabel.SCORE: parse_score,
        EntityLabel.ORDINAL: parse_cardinal,
    }
    return parsers[label](surface)
