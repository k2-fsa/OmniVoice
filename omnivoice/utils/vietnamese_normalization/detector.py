"""High-recall candidate-span detection for Vietnamese text normalization.

The detector deliberately does not choose a final reading.  Regex precedence
only determines non-overlapping boundaries; ``candidate_labels`` records every
v1 label that is structurally plausible and leaves contextual disambiguation to
the classifier.
"""

import re
from typing import Iterable

from .types import CandidateLabel, DetectedSpan

_TOKEN_RE = re.compile(
    r"(?P<interval>[\[(]\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*[\])])"
    r"|(?P<url>(?i:https?://|www\.)[^\s<>\[\]]+)"
    r"|(?P<email>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(?P<code>(?<!\w)(?=[A-Za-z0-9._-]*\d)[A-Za-z]+[A-Za-z0-9]*"
    r"(?:[._-][A-Za-z0-9]+)+(?!\w))"
    r"|(?P<iso_date>(?<!\d)(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])(?!\d))"
    r"|(?P<date>(?<!\d)(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}(?!\d))"
    r"|(?P<time>(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?!\d))"
    r"|(?P<money>(?i:HK\$|US\$|USD|VND|[$€₫¥])\s*\d+(?:[.,]\d{1,3})*"
    r"|\d+(?:[.,]\d{1,3})*\s*(?i:USD|VND|HKD|đồng|euro|đ|₫|¥))"
    r"|(?P<percent>[+-]?\d+(?:[.,]\d+)?\s*(?:%|‰))"
    r"|(?P<measurement>[+-]?\d+(?:[.,]\d+)?(?:\s*[–—-]\s*[+-]?\d+(?:[.,]\d+)?)?"
    r"\s*(?i:km/h|m/s|kwh|mhz|ghz|km|cm|mm|kg|mg|ml|gb|mb|tb|°c|°f|m|g|l|w|v|hz)(?!\w))"
    r"|(?P<phone>(?<!\w)(?:\+84|0)\d{2,3}(?:[\s.-]?\d{2,4}){2,4}(?!\w))"
    r"|(?P<numeric_chain>(?<!\w)\d+(?:\s*[/:–—-]\s*\d+)+(?!\w))"
    r"|(?P<decimal>[+-]?\d+[.,]\d+)"
    r"|(?P<roman>(?<!\w)(?=[IVXLCDM]+\b)[IVXLCDM]{2,}(?!\w))"
    r"|(?P<integer>(?<!\w)[+-]?\d+(?!\w))"
)

_LABELS = {
    "interval": (CandidateLabel.RANGE, CandidateLabel.KEEP),
    "url": (CandidateLabel.KEEP,),
    "email": (CandidateLabel.KEEP,),
    "code": (CandidateLabel.KEEP,),
    "iso_date": (CandidateLabel.DATE, CandidateLabel.KEEP),
    "date": (CandidateLabel.DATE, CandidateLabel.FRACTION, CandidateLabel.KEEP),
    "time": (CandidateLabel.TIME, CandidateLabel.KEEP),
    "money": (CandidateLabel.CURRENCY, CandidateLabel.KEEP),
    "percent": (CandidateLabel.PERCENT, CandidateLabel.KEEP),
    "measurement": (CandidateLabel.MEASUREMENT, CandidateLabel.KEEP),
    "phone": (CandidateLabel.IDENTIFIER, CandidateLabel.KEEP),
    "numeric_chain": (
        CandidateLabel.FRACTION,
        CandidateLabel.SCORE,
        CandidateLabel.RANGE,
        CandidateLabel.IDENTIFIER,
        CandidateLabel.KEEP,
    ),
    "decimal": (
        CandidateLabel.DECIMAL,
        CandidateLabel.VERSION,
        CandidateLabel.KEEP,
    ),
    "roman": (CandidateLabel.ROMAN, CandidateLabel.KEEP),
    "integer": (
        CandidateLabel.CARDINAL,
        CandidateLabel.YEAR,
        CandidateLabel.IDENTIFIER,
        CandidateLabel.KEEP,
    ),
}


def detect(text: str) -> Iterable[DetectedSpan]:
    """Yield high-recall, non-overlapping candidates in source-text order."""
    for match in _TOKEN_RE.finditer(text):
        syntax = match.lastgroup or "unknown"
        yield DetectedSpan(
            match.start(),
            match.end(),
            match.group(),
            syntax,
            _LABELS.get(syntax, (CandidateLabel.KEEP,)),
        )
