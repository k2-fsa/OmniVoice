"""Vietnamese verbalization of validated canonical values."""

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
from .verbalizers.cardinal import DIGITS, verbalize_cardinal


def _number(value: NumberValue) -> str:
    spoken = verbalize_cardinal(value.integer)
    if value.fractional is not None:
        spoken += " phẩy " + " ".join(DIGITS[int(d)] for d in value.fractional)
    return f"âm {spoken}" if value.sign == "-" else spoken


def verbalize(value, label: EntityLabel, *, surface: str = "") -> str:
    if isinstance(value, NumberValue):
        spoken = _number(value)
        if label == EntityLabel.PERCENT:
            return f"{spoken} phần trăm"
        if label == EntityLabel.ORDINAL:
            return spoken
        return spoken
    if isinstance(value, DigitSequenceValue):
        return " ".join(DIGITS[int(digit)] for digit in value.digits)
    if isinstance(value, DateValue):
        return (
            f"ngày {verbalize_cardinal(value.day)} tháng "
            f"{verbalize_cardinal(value.month)} năm {verbalize_cardinal(value.year)}"
        )
    if isinstance(value, TimeValue):
        hour = value.hour
        if value.period:
            if value.period == "PM" and hour != 12:
                hour += 12
            if value.period == "AM" and hour == 12:
                hour = 0
        spoken = f"{verbalize_cardinal(hour)} giờ"
        if value.minute is not None:
            spoken += f" {verbalize_cardinal(value.minute)} phút"
        if value.second is not None:
            spoken += f" {verbalize_cardinal(value.second)} giây"
        return spoken
    if isinstance(value, MoneyValue):
        spoken = _number(value.amount)
        if value.multiplier:
            spoken += f" {value.multiplier}"
        return f"{spoken} {value.currency}"
    if isinstance(value, MeasurementValue):
        return f"{_number(value.amount)} {value.unit}"
    if isinstance(value, PairValue):
        left = verbalize_cardinal(value.left)
        right = verbalize_cardinal(value.right)
        if label == EntityLabel.FRACTION:
            return f"{left} phần {right}"
        return f"{left} {right}"
    raise ValueError(f"unsupported canonical value: {type(value).__name__}")
