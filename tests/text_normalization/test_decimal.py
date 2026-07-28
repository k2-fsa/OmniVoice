import pytest

from omnivoice.text_normalization.verbalizers.decimal import (
    verbalize_decimal,
)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("27,5", "hai mươi bảy phẩy năm"),
        ("12.5", "mười hai phẩy năm"),
        ("0,25", "không phẩy hai năm"),
        ("3,1415", "ba phẩy một bốn một năm"),
        ("2,05", "hai phẩy không năm"),
        ("12,50", "mười hai phẩy năm không"),
        ("-2,05", "âm hai phẩy không năm"),
    ],
)
def test_verbalize_decimal(input_value, expected):
    assert verbalize_decimal(input_value) == expected


@pytest.mark.parametrize(
    "invalid_value",
    [
        "12",
        "1,2,3",
        "abc",
        "12,",
    ],
)
def test_invalid_decimal(invalid_value):
    with pytest.raises(ValueError):
        verbalize_decimal(invalid_value)