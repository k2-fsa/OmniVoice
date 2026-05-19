"""Text normalization utilities for TTS preprocessing.

Provides language-aware text normalization that converts numbers, currency,
percentages, and units into spoken-form text. This improves TTS pronunciation
quality for non-Latin scripts where raw digits produce poor speech output.

Currently supported languages:
- Malayalam (``ml``)

Usage::

    from omnivoice.utils.text_normalization import normalize_text

    text = normalize_text("എനിക്ക് ₹100 ഉണ്ട്", language="ml")
    # Returns: "എനിക്ക് നൂറ് രൂപ ഉണ്ട്"
"""

from typing import Optional


def normalize_text(text: str, language: Optional[str] = None) -> str:
    """Normalize text for TTS based on the target language.

    Converts numbers, currency symbols, percentages, and measurement units
    into their spoken-word equivalents in the target language.

    Args:
        text: Input text that may contain raw numbers, currency, etc.
        language: ISO 639 language code (e.g., ``"ml"`` for Malayalam).
            If ``None`` or unsupported, the text is returned unchanged.

    Returns:
        Normalized text with numbers/symbols replaced by words.
    """
    if language is None or not text:
        return text

    if language == "ml":
        from omnivoice.utils.text_normalization.malayalam import normalize_malayalam

        return normalize_malayalam(text)

    # Unsupported languages pass through unchanged
    return text
