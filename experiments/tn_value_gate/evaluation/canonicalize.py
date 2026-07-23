"""Formatting-only canonicalization used alongside strict matching."""

import re
import unicodedata

_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")


def canonicalize(text: str) -> str:
    value = unicodedata.normalize("NFC", text).strip()
    value = _SPACE.sub(" ", value)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", value)
