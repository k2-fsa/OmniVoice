from omnivoice.text_normalization.verbalizers.cardinal import (
    DIGITS,
    verbalize_cardinal,
)


def verbalize_decimal(value: str | float) -> str:
    raw_value = str(value).strip()

    is_negative = raw_value.startswith("-")

    if raw_value[:1] in {"+", "-"}:
        raw_value = raw_value[1:]

    separator_count = (
        raw_value.count(",")
        + raw_value.count(".")
    )

    if separator_count != 1:
        raise ValueError(
            f"DECIMAL phải có đúng một dấu phẩy hoặc dấu chấm: "
            f"{value!r}"
        )

    separator = "," if "," in raw_value else "."
    integer_part, fractional_part = raw_value.split(separator)

    if (
        not integer_part
        or not fractional_part
        or not integer_part.isdigit()
        or not fractional_part.isdigit()
    ):
        raise ValueError(f"DECIMAL không hợp lệ: {value!r}")

    integer_words = verbalize_cardinal(integer_part)

    # Đọc từng chữ số để bảo toàn số 0:
    # 2,05 → hai phẩy không năm
    fractional_words = " ".join(
        DIGITS[int(digit)]
        for digit in fractional_part
    )

    spoken = f"{integer_words} phẩy {fractional_words}"

    if is_negative:
        spoken = f"âm {spoken}"

    return spoken