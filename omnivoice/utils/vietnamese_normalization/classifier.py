"""Limited, inspectable contextual classification for the prototype."""

import re

from .types import Classification, DetectedSpan, SemioticClass

_IDENTIFIER_CONTEXT = re.compile(r"\b(?:mã|phòng|buồng|đơn hàng|mã đơn|pin|seri)\b")
_PHONE_CONTEXT = re.compile(
    r"\b(?:điện thoại|liên hệ|số máy|hotline|tổng đài|đường dây)\b"
)
_EMERGENCY_CONTEXT = re.compile(r"\b(?:cấp cứu|khẩn cấp|cứu hỏa|công an)\b")
_MEDICAL_RATIO_CONTEXT = re.compile(r"\b(?:thị lực|huyết áp)\b")
_DATE_CONTEXT = re.compile(r"\b(?:ngày|mùng|hôm|sinh|hạn dùng|hết hạn)\b")
_FRACTION_CONTEXT = re.compile(
    r"\b(?:phân số|phần|tỷ lệ|tỉ lệ|số người|số phiếu)\b"
)
_SERVICE_CODES = {"110", "111", "112", "113", "114", "115", "1080"}


def classify(text: str, span: DetectedSpan) -> Classification:
    """Choose one of seven classes, preserving expressions outside scope."""
    context = text[max(0, span.start - 45) : min(len(text), span.end + 45)].lower()
    numeric = re.sub(r"\D", "", span.surface)
    if span.syntax in {"date", "iso_date"}:
        return Classification(
            SemioticClass.DATE, False, f"valid {span.syntax.replace('_', ' ')} syntax"
        )
    if span.syntax == "time":
        return Classification(SemioticClass.TIME, False, "valid 24-hour clock syntax")
    if span.syntax == "money":
        return Classification(SemioticClass.MONEY, False, "recognized currency marker")
    if span.syntax == "numeric_chain" and "/" in span.surface:
        if _MEDICAL_RATIO_CONTEXT.search(context):
            return Classification(
                SemioticClass.UNSUPPORTED,
                True,
                "medical ratio is outside prototype scope",
            )
        if span.surface.count("/") == 1:
            if _DATE_CONTEXT.search(context):
                return Classification(
                    SemioticClass.DATE, False, "single slash with explicit date cue"
                )
            if _FRACTION_CONTEXT.search(context):
                return Classification(
                    SemioticClass.FRACTION,
                    False,
                    "single slash with explicit fraction cue",
                )
            return Classification(
                SemioticClass.UNSUPPORTED,
                True,
                "single slash is ambiguous between date, fraction, and ratio",
            )
    if span.syntax == "phone":
        if _PHONE_CONTEXT.search(context) or len(numeric) >= 9:
            return Classification(
                SemioticClass.PHONE,
                False,
                "phone-like grouping with phone context or length",
            )
        return Classification(
            SemioticClass.UNSUPPORTED,
            True,
            "formatted digit sequence lacks phone evidence",
        )
    if span.syntax == "integer":
        if _IDENTIFIER_CONTEXT.search(context):
            return Classification(
                SemioticClass.IDENTIFIER, False, "nearby identifier noun"
            )
        if (_EMERGENCY_CONTEXT.search(context) and numeric in _SERVICE_CODES) or (
            _PHONE_CONTEXT.search(context)
            and (len(numeric) >= 7 or numeric in _SERVICE_CODES)
        ):
            return Classification(
                SemioticClass.PHONE,
                False,
                "service/phone context and compatible length",
            )
        if text[max(0, span.start - 8) : span.start].lower().rstrip().endswith("gọi"):
            return Classification(
                SemioticClass.UNSUPPORTED, True, "short number after 'gọi' is ambiguous"
            )
        if numeric.startswith("0") and len(numeric) > 1:
            return Classification(
                SemioticClass.IDENTIFIER,
                False,
                "leading zero requires digit preservation",
            )
        return Classification(
            SemioticClass.CARDINAL, False, "default integer quantity reading"
        )
    return Classification(
        SemioticClass.UNSUPPORTED, True, f"{span.syntax} is outside prototype scope"
    )
