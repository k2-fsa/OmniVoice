DIGITS = [
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
]

SCALES = [
    "",
    "nghìn",
    "triệu",
    "tỷ",
    "nghìn tỷ",
    "triệu tỷ",
]


def _read_three_digits(
    number: int,
    *,
    force_hundreds: bool = False,
) -> str:
    hundreds = number // 100
    tens = (number % 100) // 10
    ones = number % 10

    words = []

    if hundreds:
        words.extend([DIGITS[hundreds], "trăm"])
    elif force_hundreds and (tens or ones):
        words.extend(["không", "trăm"])

    if tens >= 2:
        words.extend([DIGITS[tens], "mươi"])

        if ones == 1:
            words.append("mốt")
        elif ones == 4:
            words.append("tư")
        elif ones == 5:
            words.append("lăm")
        elif ones:
            words.append(DIGITS[ones])

    elif tens == 1:
        words.append("mười")

        if ones == 5:
            words.append("lăm")
        elif ones:
            words.append(DIGITS[ones])

    elif ones:
        if hundreds or force_hundreds:
            words.append("linh")

        words.append(DIGITS[ones])

    return " ".join(words)


def verbalize_cardinal(value: str | int) -> str:
    raw_value = str(value).strip()

    is_negative = raw_value.startswith("-")

    if raw_value[:1] in {"+", "-"}:
        raw_value = raw_value[1:]

    if not raw_value or not raw_value.isdigit():
        raise ValueError(
            f"CARDINAL phải chứa một số nguyên: {value!r}"
        )

    number = int(raw_value)

    if number == 0:
        return "không"

    groups = []

    while number:
        groups.append(number % 1000)
        number //= 1000

    if len(groups) > len(SCALES):
        raise ValueError("Số vượt quá phạm vi đang hỗ trợ")

    result = []

    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]

        if group == 0:
            continue

        has_higher_group = any(groups[index + 1 :])

        result.append(
            _read_three_digits(
                group,
                force_hundreds=has_higher_group,
            )
        )

        if SCALES[index]:
            result.append(SCALES[index])

    spoken = " ".join(result)

    if is_negative:
        spoken = f"âm {spoken}"

    return spoken