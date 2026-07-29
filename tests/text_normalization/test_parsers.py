import pytest

from omnivoice.text_normalization.parsers import (
    parse_cardinal,
    parse_date,
    parse_decimal,
    parse_fraction,
    parse_measurement,
    parse_time,
)


@pytest.mark.parametrize(
    "surface", ["250 000", "250\u00a0000", "250\u202f000", "250.000"]
)
def test_cardinal_grouping(surface):
    assert parse_cardinal(surface).integer == "250000"


@pytest.mark.parametrize("surface", ["12.34.567", "12k", "1,5 tỷ"])
def test_cardinal_rejects_unsupported(surface):
    with pytest.raises(ValueError):
        parse_cardinal(surface)


@pytest.mark.parametrize(
    ("surface", "integer", "fractional"),
    [
        ("27,5", "27", "5"),
        ("-0,05", "0", "05"),
        ("12,50", "12", "50"),
        ("1.234,56", "1234", "56"),
        ("1,234.56", "1234", "56"),
    ],
)
def test_decimal_locale_policy(surface, integer, fractional):
    value = parse_decimal(surface)
    assert (value.integer, value.fractional) == (integer, fractional)


@pytest.mark.parametrize("surface", ["1.234", "1,234", "1,2,3", "12,"])
def test_decimal_ambiguity_and_malformed_preserved_by_parser(surface):
    with pytest.raises(ValueError):
        parse_decimal(surface)


@pytest.mark.parametrize(
    "surface",
    [
        "27/07/2026",
        "27/7/2026",
        "27-07-2026",
        "27.07.2026",
        "2026-07-27",
        "29/02/2024",
        "27/07/26",
    ],
)
def test_valid_dates(surface):
    assert parse_date(surface).month == 7 or surface == "29/02/2024"


@pytest.mark.parametrize(
    "surface", ["31/02/2026", "29/02/2025", "0/7/2026", "1/13/2026"]
)
def test_invalid_dates(surface):
    with pytest.raises(ValueError):
        parse_date(surface)


@pytest.mark.parametrize(
    "surface",
    [
        "08:30",
        "8:30",
        "08:30:15",
        "8h30",
        "8 h 30",
        "8 giờ 30",
        "8 giờ",
        "8:30 PM",
        "8:30 AM",
    ],
)
def test_valid_times(surface):
    parse_time(surface)


@pytest.mark.parametrize("surface", ["25:00", "12:70", "08:30:99"])
def test_invalid_times(surface):
    with pytest.raises(ValueError):
        parse_time(surface)


def test_fraction_zero_denominator():
    with pytest.raises(ValueError):
        parse_fraction("1/0")


@pytest.mark.parametrize(
    ("surface", "integer", "fractional", "unit"),
    [
        ("2.5 kg", "2", "5", "ki lô gam"),
        ("37°C", "37", None, "độ xê"),
        ("250\u202f000 km", "250000", None, "ki lô mét"),
    ],
)
def test_measurements(surface, integer, fractional, unit):
    value = parse_measurement(surface)
    assert (value.amount.integer, value.amount.fractional, value.unit) == (
        integer,
        fractional,
        unit,
    )


@pytest.mark.parametrize("surface", ["12 furlong", "kg", "2.5"])
def test_unsupported_measurements(surface):
    with pytest.raises(ValueError):
        parse_measurement(surface)
