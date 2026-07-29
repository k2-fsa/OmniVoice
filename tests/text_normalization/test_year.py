import pytest

from omnivoice.text_normalization.verbalizers.year import (
    verbalize_year,
)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (
            "1999",
            "một nghìn chín trăm chín mươi chín",
        ),
        ("2000", "hai nghìn"),
        (
            "2005",
            "hai nghìn không trăm linh năm",
        ),
        (
            "2024",
            "hai nghìn không trăm hai mươi tư",
        ),
        (
            "2026",
            "hai nghìn không trăm hai mươi sáu",
        ),
        (
            " 2026 ",
            "hai nghìn không trăm hai mươi sáu",
        ),
    ],
)
def test_verbalize_year(input_value, expected):
    assert verbalize_year(input_value) == expected


@pytest.mark.parametrize(
    "invalid_value",
    [
        "99",
        "10000",
        "20a6",
    ],
)
def test_invalid_year(invalid_value):
    with pytest.raises(ValueError):
        verbalize_year(invalid_value)
