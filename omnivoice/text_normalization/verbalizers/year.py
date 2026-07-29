from omnivoice.text_normalization.verbalizers.cardinal import (
    verbalize_cardinal,
)


def verbalize_year(value: str | int) -> str:
    raw_value = str(value).strip()

    if not raw_value.isdigit():
        raise ValueError(f"YEAR không hợp lệ: {value!r}")

    if len(raw_value) != 4:
        raise ValueError(f"YEAR hiện chỉ hỗ trợ năm có 4 chữ số: {value!r}")

    return verbalize_cardinal(raw_value)
