#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text processing utilities for TTS inference.

Provides:
- ``chunk_text_punctuation()``: Splits long text into model-friendly chunks at
  sentence boundaries, with abbreviation-aware punctuation splitting.
- ``add_punctuation()``: Appends missing end punctuation (Chinese or English).
- ``normalize_text()``: Optional text normalization (numbers, dates, currency,
  etc.) into their spoken form, while preserving inline control syntax.
"""

import logging
import re
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


SPLIT_PUNCTUATION = set(".,;:!?。，；：！？")
CLOSING_MARKS = set("\"'“”‘’）]》>」】")

END_PUNCTUATION = {
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "…",
    ")",
    "]",
    "}",
    '"',
    "'",
    "“",
    "”",
    "‘",
    "’",
    "；",
    "：",
    "，",
    "。",
    "！",
    "？",
    "、",
    "……",
    "）",
    "】",
}


ABBREVIATIONS = {
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "Rev.",
    "Fr.",
    "Hon.",
    "Pres.",
    "Gov.",
    "Capt.",
    "Gen.",
    "Sen.",
    "Rep.",
    "Col.",
    "Maj.",
    "Lt.",
    "Cmdr.",
    "Sgt.",
    "Cpl.",
    "Co.",
    "Corp.",
    "Inc.",
    "Ltd.",
    "Est.",
    "Dept.",
    "St.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "Mt.",
    "Ft.",
    "No.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "i.e.",
    "e.g.",
    "vs.",
    "Vs.",
    "Etc.",
    "approx.",
    "fig.",
    "def.",
}


def chunk_text_punctuation(
    text: str,
    chunk_len: int,
    min_chunk_len: Optional[int] = None,
) -> List[str]:
    """
    Splits the input tokens list into chunks according to punctuations,
    avoiding splits on common abbreviations (e.g., Mr., No.).
    """

    # 1. Split the tokens according to punctuations.
    sentences = []
    current_sentence = []

    tokens_list = list(text)

    for token in tokens_list:
        # If the first token of current sentence is punctuation,
        # append it to the end of the previous sentence.
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in SPLIT_PUNCTUATION or token in CLOSING_MARKS)
        ):
            sentences[-1].append(token)
        # Otherwise, append the current token to the current sentence.
        else:
            current_sentence.append(token)

            # Split the sentence in positions of punctuations.
            if token in SPLIT_PUNCTUATION:
                is_abbreviation = False

                if token == ".":
                    temp_str = "".join(current_sentence).strip()
                    if temp_str:
                        last_word = temp_str.split()[-1]
                        if last_word in ABBREVIATIONS:
                            is_abbreviation = True

                if not is_abbreviation:
                    sentences.append(current_sentence)
                    current_sentence = []
    # Assume the last few tokens are also a sentence
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    # 2. Merge short sentences.
    merged_chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_len:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                merged_chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        merged_chunks.append(current_chunk)

    # 4. Post-process: Check for undersized chunks and merge them
    #  with the previous chunk or next chunk (if it's the first chunk).
    if min_chunk_len is not None:
        first_chunk_short_flag = (
            len(merged_chunks) > 0 and len(merged_chunks[0]) < min_chunk_len
        )
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            if i == 1 and first_chunk_short_flag:
                final_chunks[-1].extend(chunk)
            else:
                if len(chunk) >= min_chunk_len:
                    final_chunks.append(chunk)
                else:
                    if len(final_chunks) == 0:
                        final_chunks.append(chunk)
                    else:
                        final_chunks[-1].extend(chunk)
    else:
        final_chunks = merged_chunks

    chunk_strings = [
        "".join(chunk).strip() for chunk in final_chunks if "".join(chunk).strip()
    ]
    return chunk_strings


def add_punctuation(text: str):
    """Add punctuation if there is not in the end of text"""
    text = text.strip()

    if not text:
        return text

    if text[-1] not in END_PUNCTUATION:
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)

        text += "。" if is_chinese else "."

    return text


# ---------------------------------------------------------------------------
# Optional text normalization (opt-in via ``generate(normalize_text=True)``)
# ---------------------------------------------------------------------------
#
# Arabic numerals, dates, currency, etc. are converted into their spoken form
# so the model reads them correctly (e.g. "2345" -> "twenty three forty five",
# "199" -> the Chinese reading). Chinese/English go through WeTextProcessing;
# any other language falls back to ``num2words`` for bare integers when
# available.
#
# The OmniVoice inline control syntax must survive normalization:
#   * bracketed non-verbal tags, e.g. ``[laughter]``, ``[sigh]``;
#   * bracketed CMU pronunciation overrides, e.g. ``[B EY1 S]`` -- the stress
#     digit would otherwise be read as a number;
#   * Chinese pinyin tone markers (uppercase pinyin + tone digit) -- likewise.
# Protected spans are held out and re-inserted verbatim around normalization.

# Any ``[...]`` span covers both non-verbal tags and CMU pronunciation.
_BRACKET_TAG_RE = re.compile(r"\[[^\[\]]*\]")
# Uppercase pinyin followed by a tone digit 1-5 (Chinese pronunciation control).
_PINYIN_TONE_RE = re.compile(r"[A-Z]+[1-5]")
_CJK_RE = re.compile("[一-鿿]")

_TN_INSTALL_MSG = (
    "Text normalization (normalize_text=True) requires WeTextProcessing, which "
    "is not installed.\n"
    "  pip install WeTextProcessing         # or:  pip install 'omnivoice[tn]'\n"
    "WeTextProcessing depends on pynini, which has no prebuilt wheel for macOS "
    "arm64 (Apple Silicon). On macOS, install pynini from conda-forge first:\n"
    "  conda install -c conda-forge pynini\n"
    "then:  pip install WeTextProcessing"
)

# Normalizer construction builds FSTs and is comparatively slow, so instances
# are cached per language for the lifetime of the process.
_ZH_NORMALIZER = None
_EN_NORMALIZER = None


def _get_zh_normalizer():
    global _ZH_NORMALIZER
    if _ZH_NORMALIZER is None:
        try:
            from tn.chinese.normalizer import Normalizer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(_TN_INSTALL_MSG) from e
        # Conservative flags: normalize numbers/symbols only. Keep interjections
        # and erhua (they are spoken), keep the user's original characters, and
        # do not delete or rewrite anything beyond numeric/symbolic tokens.
        _ZH_NORMALIZER = Normalizer(
            remove_interjections=False,
            remove_erhua=False,
            traditional_to_simple=False,
            remove_puncts=False,
            full_to_half=False,
        )
    return _ZH_NORMALIZER


def _get_en_normalizer():
    global _EN_NORMALIZER
    if _EN_NORMALIZER is None:
        try:
            from tn.english.normalizer import Normalizer
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(_TN_INSTALL_MSG) from e
        _EN_NORMALIZER = Normalizer()
    return _EN_NORMALIZER


def _resolve_lang_code(language: Optional[str], text: str) -> str:
    """Map a language name/code to ``"zh"``/``"en"``/other code.

    When ``language`` is ``None`` (or unrecognized), fall back to detecting
    Chinese vs. English by the presence of CJK characters.
    """
    if language is not None:
        code = language.strip().lower()
        if code and code != "none":
            if code in ("zh", "en"):
                return code
            try:
                from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

                if language in LANG_IDS:
                    return language
                if code in LANG_NAME_TO_ID:
                    return LANG_NAME_TO_ID[code]
            except Exception:  # pragma: no cover - lang_map should be importable
                pass
            return code  # assume it is already a language id, e.g. "ja", "de"
    return "zh" if _CJK_RE.search(text) else "en"


def _num2words_segment(text: str, lang: str) -> str:
    """Best-effort integer-to-words fallback for non zh/en languages."""
    try:
        from num2words import num2words
    except ImportError:
        return text  # fallback is best-effort; silently skip when unavailable

    def _repl(match):
        try:
            return num2words(int(match.group()), lang=lang)
        except Exception:
            return match.group()  # unsupported language / value: leave as-is

    return re.sub(r"\d+", _repl, text)


def _normalize_segment(fn: Callable[[str], str], segment: str) -> str:
    """Normalize one non-protected segment, never raising on bad input.

    Leading/trailing whitespace is preserved explicitly because the underlying
    normalizers strip it, which would otherwise glue words to an adjacent
    protected span (e.g. ``the [B EY1 S] guitar`` -> ``the[B EY1 S]guitar``).
    """
    if not segment.strip():
        return segment
    lead = segment[: len(segment) - len(segment.lstrip())]
    trail = segment[len(segment.rstrip()) :]
    try:
        core = fn(segment.strip())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Text normalization failed on a segment (%s); keeping it unchanged.",
            type(e).__name__,
        )
        return segment
    return lead + core + trail


def _apply_with_protection(
    text: str, fn: Callable[[str], str], protect_pinyin: bool
) -> str:
    """Run ``fn`` on ``text`` while holding out protected control spans."""
    spans = [m.span() for m in _BRACKET_TAG_RE.finditer(text)]
    if protect_pinyin:
        spans += [m.span() for m in _PINYIN_TONE_RE.finditer(text)]
    if not spans:
        return _normalize_segment(fn, text)

    # Merge overlapping/adjacent protected spans, then normalize the gaps.
    spans.sort()
    merged: List[List[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out: List[str] = []
    last = 0
    for start, end in merged:
        if start > last:
            out.append(_normalize_segment(fn, text[last:start]))
        out.append(text[start:end])  # protected span, verbatim
        last = end
    if last < len(text):
        out.append(_normalize_segment(fn, text[last:]))
    return "".join(out)


def normalize_text(text: str, language: Optional[str] = None) -> str:
    """Normalize numbers, dates, currency, etc. into their spoken form.

    Chinese is routed to WeTextProcessing's ``ZhNormalizer`` and English to its
    ``EnNormalizer`` (configured to only rewrite numeric/symbolic tokens). Any
    other language falls back to ``num2words`` for bare integers when it is
    installed, otherwise the text is returned unchanged.

    Inline OmniVoice control syntax is preserved: bracketed non-verbal tags
    (``[laughter]``) and CMU pronunciation overrides (``[B EY1 S]``) are passed
    through untouched, and Chinese pinyin tone markers (uppercase pinyin +
    tone digit) are protected so the tone digit is not read as a number.

    Args:
        text: Input text.
        language: Language code (``"en"``/``"zh"``) or full name (``"English"``).
            ``None`` auto-detects Chinese vs. English by script.

    Returns:
        The normalized text.

    Raises:
        ImportError: For Chinese/English when the optional ``omnivoice[tn]``
            dependency (WeTextProcessing) is not installed.
    """
    if not text or not text.strip():
        return text

    code = _resolve_lang_code(language, text)
    if code == "zh":
        normalizer = _get_zh_normalizer()
        return _apply_with_protection(text, normalizer.normalize, protect_pinyin=True)
    if code == "en":
        normalizer = _get_en_normalizer()
        return _apply_with_protection(text, normalizer.normalize, protect_pinyin=False)
    # Other languages: best-effort integer conversion via num2words.
    return _apply_with_protection(
        text, lambda s: _num2words_segment(s, code), protect_pinyin=False
    )
