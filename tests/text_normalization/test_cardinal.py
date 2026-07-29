import pytest

from omnivoice.text_normalization.verbalizers.cardinal import (
    verbalize_cardinal,
)


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("0", "không"),
        ("5", "năm"),
        ("10", "mười"),
        ("15", "mười lăm"),
        ("21", "hai mươi mốt"),
        ("24", "hai mươi tư"),
        ("25", "hai mươi lăm"),
        ("104", "một trăm linh bốn"),
        ("115", "một trăm mười lăm"),
        (
            "2026",
            "hai nghìn không trăm hai mươi sáu",
        ),
        (
            "1000025",
            "một triệu không trăm hai mươi lăm",
        ),
        (
            "-25",
            "âm hai mươi lăm",
        ),
    ],
)
def test_verbalize_cardinal(input_value, expected):
    assert verbalize_cardinal(input_value) == expected


def test_invalid_cardinal():
    with pytest.raises(ValueError):
        verbalize_cardinal("12,5")
