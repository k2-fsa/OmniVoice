"""Malayalam text normalization for TTS preprocessing.

Converts numbers, currency (₹), percentages (%), and measurement units
into their Malayalam spoken-word equivalents. Uses regex word-boundary
matching to avoid false positives on partial number matches.

Supports numbers from 0 to 99,99,999 (Indian numbering: up to 99 lakh).

Example::

    >>> normalize_malayalam("എനിക്ക് ₹100 ഉണ്ട്")
    'എനിക്ക് നൂറ് രൂപ ഉണ്ട്'

    >>> normalize_malayalam("5% കുറവ്")
    'അഞ്ച് ശതമാനം കുറവ്'
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Malayalam number words
# ---------------------------------------------------------------------------

_ONES = {
    0: "പൂജ്യം",
    1: "ഒന്ന്",
    2: "രണ്ട്",
    3: "മൂന്ന്",
    4: "നാല്",
    5: "അഞ്ച്",
    6: "ആറ്",
    7: "ഏഴ്",
    8: "എട്ട്",
    9: "ഒമ്പത്",
}

_TEENS = {
    10: "പത്ത്",
    11: "പതിനൊന്ന്",
    12: "പന്ത്രണ്ട്",
    13: "പതിമൂന്ന്",
    14: "പതിനാല്",
    15: "പതിനഞ്ച്",
    16: "പതിനാറ്",
    17: "പതിനേഴ്",
    18: "പതിനെട്ട്",
    19: "പത്തൊമ്പത്",
}

_TENS = {
    20: "ഇരുപത്",
    30: "മുപ്പത്",
    40: "നാല്പത്",
    50: "അമ്പത്",
    60: "അറുപത്",
    70: "എഴുപത്",
    80: "എൺപത്",
    90: "തൊണ്ണൂറ്",
}

_HUNDREDS_PREFIX = {
    1: "നൂറ്",
    2: "ഇരുനൂറ്",
    3: "മുന്നൂറ്",
    4: "നാനൂറ്",
    5: "അഞ്ഞൂറ്",
    6: "അറുനൂറ്",
    7: "എഴുനൂറ്",
    8: "എണ്ണൂറ്",
    9: "തൊള്ളായിരം",
}

# Connective form used when hundreds are followed by tens/ones
_HUNDREDS_CONNECTIVE = {
    1: "നൂറ്റി",
    2: "ഇരുനൂറ്റി",
    3: "മുന്നൂറ്റി",
    4: "നാനൂറ്റി",
    5: "അഞ്ഞൂറ്റി",
    6: "അറുനൂറ്റി",
    7: "എഴുനൂറ്റി",
    8: "എണ്ണൂറ്റി",
    9: "തൊള്ളായിരത്തി",
}

_THOUSANDS_PREFIX = {
    1: "ആയിരം",
    2: "രണ്ടായിരം",
    3: "മൂവ്വായിരം",
    4: "നാലായിരം",
    5: "അയ്യായിരം",
    6: "ആറായിരം",
    7: "ഏഴായിരം",
    8: "എട്ടായിരം",
    9: "ഒമ്പതിനായിരം",
}

_THOUSANDS_CONNECTIVE = {
    1: "ആയിരത്തി",
    2: "രണ്ടായിരത്തി",
    3: "മൂവ്വായിരത്തി",
    4: "നാലായിരത്തി",
    5: "അയ്യായിരത്തി",
    6: "ആറായിരത്തി",
    7: "ഏഴായിരത്തി",
    8: "എട്ടായിരത്തി",
    9: "ഒമ്പതിനായിരത്തി",
}

# Connective form for tens when followed by a unit digit
_TENS_CONNECTIVE = {
    20: "ഇരുപത്തി",
    30: "മുപ്പത്തി",
    40: "നാല്പത്തി",
    50: "അമ്പത്തി",
    60: "അറുപത്തി",
    70: "എഴുപത്തി",
    80: "എൺപത്തി",
    90: "തൊണ്ണൂറ്റി",
}

# ---------------------------------------------------------------------------
# Lakh (1,00,000) support — Indian numbering
# ---------------------------------------------------------------------------

_LAKH_PREFIXES = {
    1: "ഒരു ലക്ഷം",
    2: "രണ്ട് ലക്ഷം",
    3: "മൂന്ന് ലക്ഷം",
    4: "നാല് ലക്ഷം",
    5: "അഞ്ച് ലക്ഷം",
    6: "ആറ് ലക്ഷം",
    7: "ഏഴ് ലക്ഷം",
    8: "എട്ട് ലക്ഷം",
    9: "ഒമ്പത് ലക്ഷം",
}

_LAKH_CONNECTIVE = {
    1: "ഒരു ലക്ഷത്തി",
    2: "രണ്ട് ലക്ഷത്തി",
    3: "മൂന്ന് ലക്ഷത്തി",
    4: "നാല് ലക്ഷത്തി",
    5: "അഞ്ച് ലക്ഷത്തി",
    6: "ആറ് ലക്ഷത്തി",
    7: "ഏഴ് ലക്ഷത്തി",
    8: "എട്ട് ലക്ഷത്തി",
    9: "ഒമ്പത് ലക്ഷത്തി",
}

# ---------------------------------------------------------------------------
# Unit suffixes for measurement normalization
# ---------------------------------------------------------------------------

_UNIT_MAP = {
    "km": "കിലോമീറ്റർ",
    "kg": "കിലോഗ്രാം",
    "mg": "മില്ലിഗ്രാം",
    "cm": "സെന്റീമീറ്റർ",
    "mm": "മില്ലിമീറ്റർ",
    "ml": "മില്ലിലിറ്റർ",
    "m": "മീറ്റർ",
    "g": "ഗ്രാം",
    "l": "ലിറ്റർ",
    "L": "ലിറ്റർ",
}

# ---------------------------------------------------------------------------
# Fractions & English letters mapping
# ---------------------------------------------------------------------------

_FRACTIONS = {
    "1/4": "കാൽ",
    "1/2": "അര",
    "3/4": "മുക്കാൽ",
}

_ENGLISH_LETTERS = {
    "A": "എ", "B": "ബി", "C": "സി", "D": "ഡി", "E": "ഇ", "F": "എഫ്", "G": "ജി",
    "H": "എച്ച്", "I": "ഐ", "J": "ജെ", "K": "കെ", "L": "എൽ", "M": "എം", "N": "എൻ",
    "O": "ഒ", "P": "പി", "Q": "ക്യൂ", "R": "ആർ", "S": "എസ്", "T": "ടി", "U": "യു",
    "V": "വി", "W": "ഡബ്ല്യൂ", "X": "എക്സ്", "Y": "വൈ", "Z": "സെഡ്"
}


# ---------------------------------------------------------------------------
# Number to Malayalam word conversion
# ---------------------------------------------------------------------------


def _number_to_malayalam(n: int) -> Optional[str]:
    """Convert an integer (0 to 9999999) to Malayalam words.

    Args:
        n: Non-negative integer.

    Returns:
        Malayalam word string, or ``None`` if the number is out of range.
    """
    if n < 0 or n > 9999999:
        return None

    if n == 0:
        return _ONES[0]

    parts = []

    # --- Lakhs (1,00,000 – 99,00,000) ---
    if n >= 100000:
        lakh_part = n // 100000
        remainder = n % 100000

        if lakh_part <= 9:
            if remainder == 0:
                parts.append(_LAKH_PREFIXES[lakh_part])
            else:
                parts.append(_LAKH_CONNECTIVE[lakh_part])
        else:
            # 10-99 lakhs: compose the tens/ones for the lakh multiplier
            tens_word = _number_to_malayalam(lakh_part)
            if tens_word is None:
                return None
            if remainder == 0:
                parts.append(f"{tens_word} ലക്ഷം")
            else:
                parts.append(f"{tens_word} ലക്ഷത്തി")

        n = remainder

    # --- Thousands (1000 – 99,999) ---
    if n >= 1000:
        t = n // 1000
        remainder = n % 1000

        if t <= 9:
            # Single-digit thousands: use lookup tables
            if remainder == 0:
                parts.append(_THOUSANDS_PREFIX[t])
            else:
                parts.append(_THOUSANDS_CONNECTIVE[t])
        else:
            # Multi-digit thousands (10–99): compose the word recursively
            t_word = _number_to_malayalam(t)
            if t_word is None:
                return None
            if remainder == 0:
                parts.append(f"{t_word} ആയിരം")
            else:
                parts.append(f"{t_word} ആയിരത്തി")

        n = remainder

    # --- Hundreds (100 – 999) ---
    if n >= 100:
        h = n // 100
        remainder = n % 100
        if remainder == 0:
            parts.append(_HUNDREDS_PREFIX[h])
        else:
            parts.append(_HUNDREDS_CONNECTIVE[h])
        n = remainder

    # --- Tens and ones (0 – 99) ---
    if n > 0:
        if n < 10:
            parts.append(_ONES[n])
        elif n < 20:
            parts.append(_TEENS[n])
        else:
            tens = (n // 10) * 10
            ones = n % 10
            if ones == 0:
                parts.append(_TENS[tens])
            else:
                parts.append(_TENS_CONNECTIVE[tens])
                parts.append(_ONES[ones])

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sub-normalizers
# ---------------------------------------------------------------------------

# Match ₹ followed by a number (with optional commas): ₹1,00,000 or ₹250
_CURRENCY_RE = re.compile(r"₹\s*([\d,]+)")

# Match number followed by % sign
_PERCENT_RE = re.compile(r"(\d[\d,]*)\s*%")

# Match number followed by a unit (longest match first to avoid "m" eating "mm")
# Sort keys by length descending so "km" matches before "m"
_UNIT_KEYS_SORTED = sorted(_UNIT_MAP.keys(), key=len, reverse=True)
_UNIT_PATTERN = "|".join(re.escape(u) for u in _UNIT_KEYS_SORTED)
_UNIT_RE = re.compile(rf"(\d[\d,]*)\s*({_UNIT_PATTERN})\b")

# Match decimals
_DECIMAL_RE = re.compile(r"(\d[\d,]*)\.(\d+)")

# Match time
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")

# Match fractions
_FRACTION_RE = re.compile(r"(?<!\d)(1/4|1/2|3/4)(?!\d)")
_MIXED_FRACTION_RE = re.compile(r"(\d[\d,]*)\s+(1/4|1/2|3/4)")

# Match ordinals
_ORDINAL_RE = re.compile(r"\b(\d[\d,]*)(st|nd|rd|th)\b")

# Match acronyms (2+ consecutive uppercase English letters)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")

# Match standalone numbers (word boundary): avoids matching "abc123def"
_NUMBER_RE = re.compile(r"(?<!\d)(\d[\d,]*)(?!\d)")


def _parse_number_str(s: str) -> Optional[int]:
    """Parse a number string that may contain commas (Indian format)."""
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return None


def _normalize_currency(text: str) -> str:
    """Replace ₹{number} with Malayalam number + രൂപ."""

    def _replace(m: re.Match) -> str:
        n = _parse_number_str(m.group(1))
        if n is None:
            return m.group(0)
        word = _number_to_malayalam(n)
        if word is None:
            return m.group(0)
        return f"{word} രൂപ"

    return _CURRENCY_RE.sub(_replace, text)


def _normalize_percentages(text: str) -> str:
    """Replace {number}% with Malayalam number + ശതമാനം."""

    def _replace(m: re.Match) -> str:
        n = _parse_number_str(m.group(1))
        if n is None:
            return m.group(0)
        word = _number_to_malayalam(n)
        if word is None:
            return m.group(0)
        return f"{word} ശതമാനം"

    return _PERCENT_RE.sub(_replace, text)


def _normalize_units(text: str) -> str:
    """Replace {number}{unit} with Malayalam number + unit name."""

    def _replace(m: re.Match) -> str:
        n = _parse_number_str(m.group(1))
        unit = m.group(2)
        if n is None:
            return m.group(0)
        word = _number_to_malayalam(n)
        if word is None:
            return m.group(0)
        ml_unit = _UNIT_MAP.get(unit, unit)
        return f"{word} {ml_unit}"

    return _UNIT_RE.sub(_replace, text)


def _digit_by_digit(s: str) -> str:
    """Read a string of digits one by one."""
    return " ".join(_ONES[int(char)] for char in s if char.isdigit())


def _normalize_decimals(text: str) -> str:
    """Replace 10.5 with പത്ത് പോയിന്റ് അഞ്ച്."""

    def _replace(m: re.Match) -> str:
        n = _parse_number_str(m.group(1))
        if n is None:
            return m.group(0)
        word = _number_to_malayalam(n)
        if word is None:
            word = _digit_by_digit(m.group(1))
        frac = _digit_by_digit(m.group(2))
        return f"{word} പോയിന്റ് {frac}"

    return _DECIMAL_RE.sub(_replace, text)


def _normalize_time(text: str) -> str:
    """Replace 5:30 with അഞ്ച് മുപ്പത്, 10:00 with പത്ത് മണി."""

    def _replace(m: re.Match) -> str:
        hour = m.group(1)
        mins = m.group(2)
        if mins == "00":
            return f"{hour} മണി"
        
        # Strip leading zeros so "05" becomes "5" to avoid digit-by-digit fallback
        mins_stripped = mins.lstrip("0")
        return f"{hour} {mins_stripped}"

    return _TIME_RE.sub(_replace, text)


def _normalize_fractions(text: str) -> str:
    """Replace 1/2, 1/4, 3/4 and mixed fractions."""

    def _replace_mixed(m: re.Match) -> str:
        n = _parse_number_str(m.group(1))
        if n is None:
            return m.group(0)
        word = _number_to_malayalam(n)
        frac = _FRACTIONS.get(m.group(2))
        return f"{word} {frac}"

    text = _MIXED_FRACTION_RE.sub(_replace_mixed, text)

    def _replace_simple(m: re.Match) -> str:
        return _FRACTIONS.get(m.group(1), m.group(0))

    return _FRACTION_RE.sub(_replace_simple, text)


def _normalize_ordinals(text: str) -> str:
    """Replace 1st, 2nd, 10th with ഒന്നാമത്തെ, രണ്ടാമത്തെ, പത്താമത്തെ."""

    def _replace(m: re.Match) -> str:
        n = _parse_number_str(m.group(1))
        if n is None:
            return m.group(0)
        word = _number_to_malayalam(n)
        if word is None:
            return m.group(0)
        
        if word.endswith("്") or word.endswith("ം"):
            return word[:-1] + "ാമത്തെ"
        return word + "ാമത്തെ"

    return _ORDINAL_RE.sub(_replace, text)


def _normalize_acronyms(text: str) -> str:
    """Replace AICTE with എ ഐ സി ടി ഇ."""

    def _replace(m: re.Match) -> str:
        chars = list(m.group(0))
        words = [_ENGLISH_LETTERS.get(c, c) for c in chars]
        return " ".join(words)

    return _ACRONYM_RE.sub(_replace, text)


def _normalize_numbers(text: str) -> str:
    """Replace standalone numbers with Malayalam words."""

    def _replace(m: re.Match) -> str:
        n_str = m.group(1)
        # Fallback for phone numbers or leading-zero numbers
        if n_str.startswith("0") and len(n_str) > 1:
            return _digit_by_digit(n_str)

        n = _parse_number_str(n_str)
        if n is None:
            return m.group(0)
        
        word = _number_to_malayalam(n)
        if word is None:
            return _digit_by_digit(n_str)
        return word

    return _NUMBER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def normalize_malayalam(text: str) -> str:
    """Normalize Malayalam text for TTS.

    Applies the following transformations in order:

    1. Currency (₹) → Malayalam number + രൂപ
    2. Percentages (%) → Malayalam number + ശതമാനം
    3. Units (km, kg, etc.) → Malayalam number + unit name
    4. Standalone numbers → Malayalam words

    The order matters: currency/percent/unit patterns are matched first so
    that their special suffixes are preserved. Remaining standalone numbers
    are then converted.

    Args:
        text: Input Malayalam text.

    Returns:
        Text with numbers/symbols replaced by Malayalam words.
    """
    text = _normalize_acronyms(text)
    text = _normalize_fractions(text)
    text = _normalize_decimals(text)
    text = _normalize_ordinals(text)
    text = _normalize_currency(text)
    text = _normalize_percentages(text)
    text = _normalize_units(text)
    text = _normalize_time(text)
    text = _normalize_numbers(text)
    return text
