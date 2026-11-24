#!/usr/bin/env python3
# Copyright    2026  Phuc Nguyen
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

"""Zero-dependency Vietnamese text normalization for TTS inference.

Rewrites numeric, temporal, currency, measurement, and symbolic expressions
into natural spoken Vietnamese words while leaving ordinary alphabetic text
(including mixed English loanwords and brand names) untouched.
"""

import re
from typing import Union

ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

# Precomposed characters exclusive to Vietnamese orthography (excludes characters
# shared with French/Spanish/German/Portuguese/Romanian/Croatian/Yoruba/Igbo/Pinyin
# such as bare é, à, â, ô, ă, đ, bare underdots ạ/ẹ/ị/ọ/ụ/ỵ, or carons ǎ/ǐ/ǒ/ǔ).
VI_SPECIFIC_CHAR_RE = re.compile(
    r"[ắằẳẵặấầẩẫậếềểễệốồổỗộơớờởỡợưứừửữựảẻỉỏủỷỹ]",
    re.IGNORECASE,
)
_VI_D_WITH_ACCENT_RE = re.compile(
    r"(?=.*[đĐ])(?=.*[àáâãèéêìíòóôõùúýăạẹịọụ])",
    re.IGNORECASE,
)

_ROMAN_VALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_VALID_ROMAN_RE = re.compile(r"^(?:XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")

_RES_MAP = {
    "360": "ba sáu mươi",
    "480": "bốn tám mươi",
    "720": "bảy hai mươi",
    "1080": "một không tám không",
    "1440": "một bốn bốn không",
    "2160": "hai một sáu không",
}


def has_vietnamese_diacritics(text: str) -> bool:
    """Return True if ``text`` contains precomposed characters specific to Vietnamese."""
    if not text:
        return False
    return bool(VI_SPECIFIC_CHAR_RE.search(text) or _VI_D_WITH_ACCENT_RE.search(text))


def read_two_digits(val: int, has_hundred: bool = False) -> str:
    """Read a two-digit integer (0-99) in natural Vietnamese."""
    chuc = val // 10
    donvi = val % 10
    res = []

    if chuc == 0:
        if has_hundred and donvi > 0:
            res.append("lẻ " + ONES[donvi])
        elif not has_hundred:
            res.append(ONES[donvi])
    elif chuc == 1:
        res.append("mười")
        if donvi == 1:
            res.append("một")
        elif donvi == 5:
            res.append("lăm")
        elif donvi > 0:
            res.append(ONES[donvi])
    else:
        res.append(ONES[chuc] + " mươi")
        if donvi == 1:
            res.append("mốt")
        elif donvi == 4:
            res.append("tư")
        elif donvi == 5:
            res.append("lăm")
        elif donvi > 0:
            res.append(ONES[donvi])

    return " ".join(res)


def read_three_digits(val: int, is_highest: bool = False) -> str:
    """Read a three-digit group (0-999) in Vietnamese."""
    tram = val // 100
    du = val % 100
    res = []

    if tram > 0 or not is_highest:
        res.append(ONES[tram] + " trăm")
    if du > 0:
        res.append(read_two_digits(du, has_hundred=(tram > 0 or not is_highest)))

    return " ".join(res)


def _read_below_billion(val: int, has_higher: bool = False) -> str:
    """Read an integer in 1..999_999_999, preserving 'không trăm' when preceded by a higher scale."""
    units = ["", "nghìn", "triệu"]
    groups = [val % 1000, (val // 1000) % 1000, (val // 1_000_000) % 1000]
    words = []
    seen_nonzero = has_higher
    for i in (2, 1, 0):
        g = groups[i]
        if g == 0:
            continue
        is_highest = not seen_nonzero
        seen_nonzero = True
        g_words = read_three_digits(g, is_highest=is_highest)
        unit = units[i]
        words.append(f"{g_words} {unit}" if unit else g_words)
    return " ".join(words).strip()


def read_int(n: Union[int, str]) -> str:
    """Read an arbitrary-size integer into Vietnamese words."""
    try:
        val = int(n)
    except (ValueError, TypeError):
        return str(n)

    if val == 0:
        return "không"
    if val < 0:
        return "âm " + read_int(-val)

    if val >= 1_000_000_000:
        high, low = divmod(val, 1_000_000_000)
        head = f"{read_int(high)} tỷ"
        if low == 0:
            return head
        return f"{head} {_read_below_billion(low, has_higher=True)}"

    return _read_below_billion(val, has_higher=False)


def read_digits_verbatim(digits_str: str) -> str:
    """Read each digit individually (for decimals, OTPs, phone numbers)."""
    return " ".join(ONES[int(d)] for d in digits_str if d.isdigit())


def _read_digit_sequence_grouped(digits_str: str) -> str:
    """Read long identifier/phone/account digit strings in 3-4 digit prosodic groups."""
    clean = "".join(ch for ch in digits_str if ch.isdigit())
    n = len(clean)
    if n <= 6:
        return read_digits_verbatim(clean)
    if n in (10, 11):
        chunks = [clean[:4], clean[4:7], clean[7:]]
    elif n == 12:
        chunks = [clean[:4], clean[4:8], clean[8:]]
    else:
        chunks = [clean[i : i + 4] for i in range(0, n, 4)]
        if len(chunks) >= 2 and len(chunks[-1]) == 1:
            tail5 = chunks[-2] + chunks[-1]
            chunks = chunks[:-2] + [tail5[:3], tail5[3:]]
    return ", ".join(read_digits_verbatim(c) for c in chunks if c)


def _parse_roman(roman_str: str) -> int:
    s = roman_str.upper().strip()
    if not s or not _VALID_ROMAN_RE.fullmatch(s):
        return 0
    total = 0
    prev = 0
    for ch in reversed(s):
        v = _ROMAN_VALS.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def _resolve_sign_prefix(sign_ch: str, full_str: str, match_start: int) -> str:
    """Convert leading '-' or '+' before a number into 'âm ' or 'cộng ' without doubling."""
    if not sign_ch:
        return ""
    before = full_str[:match_start].rstrip()
    last_word = before.rsplit(None, 1)[-1].lower() if before else ""
    if sign_ch == "-":
        return "" if last_word in ("âm", "trừ") else "âm "
    if sign_ch == "+":
        return "" if last_word in ("cộng", "dương") else "cộng "
    return ""


def _repl_hms_shorthand_match(m: re.Match) -> str:
    h = int(m.group(1))
    m_str = m.group(2)
    p_tok = m.group(3)
    s_str = m.group(4)
    if h > 72 and not p_tok:
        return m.group(0)
    parts = [f"{read_int(h)} giờ"]
    if m_str is not None:
        mv = int(m_str)
        if mv > 59:
            return m.group(0)
        if mv > 0 or s_str is not None:
            parts.append(f"{read_int(mv)} phút")
    if s_str is not None:
        sv = int(s_str)
        if sv > 59:
            return m.group(0)
        if sv > 0:
            parts.append(f"{read_int(sv)} giây")
    return " ".join(parts)


def normalize_vietnamese_text(text: str) -> str:
    """Normalize numeric, temporal, currency, unit, and symbolic tokens in Vietnamese text.

    Follows OmniVoice's conservative normalization contract: only numeric and
    symbolic tokens are expanded into Vietnamese words; surrounding Vietnamese
    and English words are kept as-is.
    """
    if not text or not isinstance(text, str):
        return text

    s = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2212", "-")

    # 1. Context-aware Roman numerals (Quý III, Quý IV -> Quý tư, Thế kỷ XXI, Chương IV)
    def _repl_roman_ctx(m: re.Match) -> str:
        label = m.group(1)
        val = _parse_roman(m.group(2))
        if val <= 0:
            return m.group(0)
        if label.strip().lower() == "quý" and val == 4:
            return f"{label} tư"
        return f"{label} {read_int(val)}"

    s = re.sub(
        r"(?<!\w)(quý|thế\s+kỷ|chương|phần|tập|mục|cấp|bậc|hạng|loại|vành\s+đai|khóa|lần\s+thứ|đại\s+hội)\s+([IVXL]{1,8})\b",
        _repl_roman_ctx,
        s,
        flags=re.IGNORECASE,
    )

    # 2. Phone numbers, switchboard hotlines & explicit identifier digit strings
    def _repl_intl_phone(m: re.Match) -> str:
        groups = [g for g in re.findall(r"\d+", m.group(1)) if g]
        if sum(len(g) for g in groups) < 8:
            return m.group(0)
        if len(groups) == 1:
            return "cộng tám tư, " + _read_digit_sequence_grouped(groups[0])
        return "cộng tám tư, " + ", ".join(read_digits_verbatim(g) for g in groups)

    s = re.sub(
        r"(?:(?<=\s)|^)\+84([\s.\-]?\(?\d{2,4}\)?(?:[\s.\-]?\d{3,4}){2,3})\b",
        _repl_intl_phone,
        s,
    )

    def _repl_hotline(m: re.Match) -> str:
        p1 = m.group(1)
        p2 = re.sub(r"\D", "", m.group(2))
        return f"{read_digits_verbatim(p1)}, {read_digits_verbatim(p2)}"

    s = re.sub(r"\b(1800|1900)[\s.\-](\d{4,6})\b", _repl_hotline, s)
    s = re.sub(r"\b(1800|1900)(\d{4})\b", _repl_hotline, s)

    def _repl_grouped_phone(m: re.Match) -> str:
        raw_p = m.group(0)
        groups = re.findall(r"\d+", raw_p)
        total_d = sum(len(g) for g in groups)
        if total_d < 9 or total_d > 12:
            return raw_p
        return ", ".join(read_digits_verbatim(g) for g in groups)

    s = re.sub(
        r"(?<![\w.,\-])\(?0\d{1,3}\)?(?:[\s.\-]\d{3,5}){2,3}\b",
        _repl_grouped_phone,
        s,
    )

    # Normalize space-separated thousands (e.g. '1 500 000', '25 000') to dot-thousands
    s = re.sub(
        r"(?<![\d.,])\b(\d{1,3}(?: \d{3}){2,}|\d{1,3} 000)\b(?![\s.\-]\d)",
        lambda m: m.group(1).replace(" ", "."),
        s,
    )

    s = re.sub(
        r"((?<!\w)(?:mã\s+xác\s+thực\s+otp|mã\s+otp|otp|mã\s+pin|mã\s+xác\s+thực|số\s+căn\s+cước|căn\s+cước\s+số|căn\s+cước|cccd|cmnd|số\s+tài\s+khoản|tài\s+khoản\s+số|tài\s+khoản|stk|số\s+thẻ)(?:\s+của\s+[\w\s]{1,20}?)?\s+(?:là\s+)?)\b(\d{4,19})\b",
        lambda m: f"{m.group(1)}{_read_digit_sequence_grouped(m.group(2))}",
        s,
        flags=re.IGNORECASE,
    )

    # 3. Dates, 24/7 availability, aspect ratios & clock times
    # Handle full dates and context-prefixed day/month BEFORE 24/7 so dates like
    # "24/7/2026" and "ngày 24/7" are read as calendar dates rather than "24 trên 7".
    def _read_month_vi(mo: int) -> str:
        return "tư" if mo == 4 else read_int(mo)

    def _repl_full_date(m: re.Match) -> str:
        prefix = m.group(1) or "ngày"
        d, mo, y = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return (
                f"{prefix} {read_int(d)} tháng {_read_month_vi(mo)} năm {read_int(y)}"
            )
        return m.group(0)

    s = re.sub(
        r"(?:(?<!\w)(ngày)\s+)?\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
        _repl_full_date,
        s,
        flags=re.IGNORECASE,
    )

    def _repl_month_year(m: re.Match) -> str:
        prefix = m.group(1)
        mo, y = int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12:
            return f"{prefix} {_read_month_vi(mo)} năm {read_int(y)}"
        return m.group(0)

    s = re.sub(
        r"(?<!\w)(tháng)\s+(\d{1,2})[/-](\d{4})\b",
        _repl_month_year,
        s,
        flags=re.IGNORECASE,
    )

    def _repl_day_month(m: re.Match) -> str:
        prefix = m.group(1)
        d, mo = int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{prefix} {read_int(d)} tháng {_read_month_vi(mo)}"
        return m.group(0)

    s = re.sub(
        r"(?<!\w)(ngày)\s+(\d{1,2})[/-](\d{1,2})\b(?![/-]\d)",
        _repl_day_month,
        s,
        flags=re.IGNORECASE,
    )

    s = re.sub(r"(?<![/\d])\b24\s*/\s*7\b(?![/\d])", "hai mươi tư trên bảy", s)
    s = re.sub(r"(?<![/\d])\b24\s*/\s*24\b(?![/\d])", "hai mươi tư trên hai mươi tư", s)

    # Aspect ratios ('tỷ lệ 16:9') before HH:MM
    s = re.sub(
        r"((?<!\w)t[ỷỉ]\s+lệ\s+)(\d{1,3})\s*:\s*(\d{1,3})\b",
        lambda m: f"{m.group(1)}{read_int(int(m.group(2)))} {read_int(int(m.group(3)))}",
        s,
        flags=re.IGNORECASE,
    )

    s = re.sub(
        r"\b(\d{1,2})h(?:(\d{1,2})(p|phút|')?(?:(\d{1,2})(?:s|giây|\"))?)?(?![a-zA-Zà-ỹÀ-Ỹ0-9])",
        _repl_hms_shorthand_match,
        s,
        flags=re.IGNORECASE,
    )

    def _repl_colon_hms(m: re.Match) -> str:
        h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
        parts = [f"{read_int(h)} giờ"]
        if mi > 0 or sec > 0:
            parts.append(f"{read_int(mi)} phút")
        if sec > 0:
            parts.append(f"{read_int(sec)} giây")
        return " ".join(parts)

    s = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)\b", _repl_colon_hms, s)

    def _repl_colon_hm(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if mi == 0:
            return f"{read_int(h)} giờ"
        return f"{read_int(h)} giờ {read_int(mi)} phút"

    s = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _repl_colon_hm, s)

    # Leading-zero digit codes (0901234567, 048291, 007) and >=12-digit serials
    s = re.sub(
        r"(?<![.,:/\d])\b0\d+\b(?![.,:/]\d)",
        lambda m: _read_digit_sequence_grouped(m.group(0)),
        s,
    )
    s = re.sub(
        r"(?<![.,:/\d])\b\d{12,}\b(?![.,:/]\d)",
        lambda m: _read_digit_sequence_grouped(m.group(0)),
        s,
    )

    # 4. IPv4 addresses & multi-part software version numbers
    def _read_ip_octet(o_str: str) -> str:
        val = int(o_str)
        return read_int(val) if val < 100 else read_digits_verbatim(o_str)

    def _repl_ipv4(m: re.Match) -> str:
        parts = m.group(1).split(".")
        if all(len(p) == 3 for p in parts[1:]):
            before = s[: m.start()].lower()[-25:]
            if not any(
                k in before for k in ("ip", "dns", "host", "máy chủ", "địa chỉ")
            ):
                return m.group(0)
        if any(len(p) > 1 and p.startswith("0") for p in parts):
            return m.group(0)
        return " chấm ".join(_read_ip_octet(p) for p in parts)

    s = re.sub(
        r"\b((?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3})\b",
        _repl_ipv4,
        s,
    )

    def _repl_prefixed_version(m: re.Match) -> str:
        ver_parts = m.group(1).split(".")
        spoken = " chấm ".join(
            read_int(int(p)) if idx == 0 or len(p) <= 2 else read_digits_verbatim(p)
            for idx, p in enumerate(ver_parts)
        )
        return f"phiên bản {spoken}"

    s = re.sub(
        r"(?<!\w)(?:phiên\s+bản\s*v|v|ver)\s*(\d+(?:\.\d+)+)\b",
        _repl_prefixed_version,
        s,
        flags=re.IGNORECASE,
    )

    def _repl_semver(m: re.Match) -> str:
        tok = m.group(0)
        parts = tok.split(".")
        if len(parts[1]) == 3 and len(parts[2]) == 3:
            return tok
        return " chấm ".join(read_int(int(p)) for p in parts)

    s = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,4}\b", _repl_semver, s)

    # 5. Vietnamese price shorthands, video resolutions & mobile generation tags
    def _repl_tr_compound(m: re.Match) -> str:
        tr_part = read_int(int(m.group(1)))
        sub_d = int(m.group(2))
        if sub_d == 5:
            return f"{tr_part} triệu rưỡi"
        if sub_d == 0:
            return f"{tr_part} triệu"
        return f"{tr_part} triệu {ONES[sub_d]} trăm nghìn"

    s = re.sub(
        r"\b(\d{1,4})\s*tr(?:iệu)?\s*(\d)\b",
        _repl_tr_compound,
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"(\d(?:[\d.,]*\d)?)\s*tr(?:iệu)?\b", r"\1 triệu", s, flags=re.IGNORECASE
    )
    s = re.sub(r"(\d(?:[\d.,]*\d)?)\s*[kK]\b", r"\1 ca", s)
    s = re.sub(r"\b([23456])G\b", lambda m: f"{ONES[int(m.group(1))]} gờ", s)
    s = re.sub(
        r"\b(360|480|720|1080|1440|2160)p\b",
        lambda m: f"{_RES_MAP[m.group(1)]} pê",
        s,
        flags=re.IGNORECASE,
    )

    # 6. Physical, area/volume & digital measurement units attached to numbers
    s = re.sub(r"[Øø]\s*(?=\d)", "đường kính ", s)
    s = re.sub(r"(?<=\d)\s*(?:N[\s·.\-]*cm|NCM)\b", " niu tơn xen ti mét", s)
    s = re.sub(r"(?<=\d)\s*(?:N[·.]m|Nm)\b", " niu tơn mét", s)
    s = re.sub(r"(?<=\d)\s*RPM\b", " vòng một phút", s, flags=re.IGNORECASE)
    s = re.sub(r"\bvòng\s*/\s*phút\b", "vòng một phút", s, flags=re.IGNORECASE)

    s = re.sub(r"(?<=\d)\s*km(?:²|2\b)", " ki lô mét vuông", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*cm(?:²|2\b)", " xen ti mét vuông", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*mm(?:²|2\b)", " mi li mét vuông", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*m(?:²|2\b)", " mét vuông", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*km(?:³|3\b)", " ki lô mét khối", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*cm(?:³|3\b)", " xen ti mét khối", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*mm(?:³|3\b)", " mi li mét khối", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*m(?:³|3\b)", " mét khối", s, flags=re.IGNORECASE)
    s = re.sub(r"\bm²", "mét vuông", s)
    s = re.sub(r"\bm³", "mét khối", s)

    s = re.sub(r"(?<=\d)\s*km\s*/\s*h\b", " ki lô mét trên giờ", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*m\s*/\s*s\b", " mét trên giây", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*Gbps\b", " gi ga bít trên giây", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*Mbps\b", " mê ga bít trên giây", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*Kbps\b", " ki lô bít trên giây", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*fps\b", " ép pê ét", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*GHz\b", " gi ga héc", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*MHz\b", " mê ga héc", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*kHz\b", " ki lô héc", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*(?:Hz|hertz)\b", " héc", s, flags=re.IGNORECASE)

    s = re.sub(r"(?<=\d)\s*TB\b", " tê ra bai", s)
    s = re.sub(r"(?<=\d)\s*GB\b", " gi ga bai", s)
    s = re.sub(r"(?<=\d)\s*MB\b", " mê ga bai", s)
    s = re.sub(r"(?<=\d)\s*KB\b", " ki lô bai", s)

    s = re.sub(r"(?<=\d)\s*[µμ]m\b", " mi crô mét", s)
    s = re.sub(r"(?<=\d)\s*mm\b", " mi li mét", s)
    s = re.sub(r"(?<=\d)\s*cm\b", " xen ti mét", s)
    s = re.sub(r"(?<=\d)\s*km\b", " ki lô mét", s)
    s = re.sub(r"(?<=\d)\s*nm\b", " na nô mét", s)
    s = re.sub(r"(?<=\d)\s*mg\b", " mi li gam", s)
    s = re.sub(r"(?<=\d)\s*kg\b", " ki lô gam", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*m[lL]\b", " mi li lít", s)
    s = re.sub(r"(?<=\d)\s*mAh\b", " mi li am pe giờ", s)
    s = re.sub(r"(?<=\d)\s*kWh\b", " ki lô oát giờ", s)
    s = re.sub(r"(?<=\d)\s*kW\b", " ki lô oát", s)
    s = re.sub(r"(?<=\d)\s*g\b", " gam", s)
    s = re.sub(r"(?<=\d)\s*[lL]\b", " lít", s)
    s = re.sub(r"(?<=\d)\s*m\b", " mét", s)
    s = re.sub(r"(?<=\d)\s*°\s*C\b", " độ xê", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*°\s*F\b", " độ ép", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*°(?![A-Za-z])", " độ", s)

    # Percentages & currencies
    num_tok = r"\d(?:[\d.,]*\d)?"
    s = re.sub(rf"({num_tok})\s*%", r"\1 phần trăm", s)

    def _normalize_intl_comma_thousands(num_s: str) -> str:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", num_s):
            return num_s.replace(",", ".")
        return num_s

    def _repl_usd_prefix(m: re.Match) -> str:
        sign = m.group(1) or ""
        return f"{sign}{_normalize_intl_comma_thousands(m.group(2))} đô la"

    s = re.sub(rf"([+-])?\$\s*({num_tok})", _repl_usd_prefix, s)
    s = re.sub(
        rf"([+-])?€\s*({num_tok})", lambda m: f"{m.group(1) or ''}{m.group(2)} ơ-rô", s
    )
    s = re.sub(
        rf"([+-])?¥\s*({num_tok})", lambda m: f"{m.group(1) or ''}{m.group(2)} yên", s
    )
    s = re.sub(
        rf"([+-])?£\s*({num_tok})",
        lambda m: f"{m.group(1) or ''}{m.group(2)} bảng Anh",
        s,
    )
    s = re.sub(
        rf"({num_tok})\s*\$",
        lambda m: f"{_normalize_intl_comma_thousands(m.group(1))} đô la",
        s,
    )
    s = re.sub(
        rf"({num_tok})\s*USD\b",
        lambda m: f"{_normalize_intl_comma_thousands(m.group(1))} đô la Mỹ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(rf"({num_tok})\s*EUR\b", r"\1 ơ-rô", s, flags=re.IGNORECASE)
    s = re.sub(rf"({num_tok})\s*(?:đ|vnđ|vnd)\b", r"\1 đồng", s, flags=re.IGNORECASE)

    # Per-unit slash expressions ('/năm', '/tháng', '/ngày', '/người')
    s = re.sub(
        r"\s*/\s*(năm|tháng|tuần|ngày|đêm|giờ|phút|giây|người|lượt|chiếc|cái|suất|kỳ|gói|vé)\b",
        r" một \1",
        s,
        flags=re.IGNORECASE,
    )

    # 7. Scores, ranges, multiplications & fractions
    s = re.sub(
        r"((?<!\w)(?:t[ỷỉ]\s+số|chung\s+cuộc|hòa)\s+)(\d{1,2})\s*[-–—]\s*(\d{1,2})\b",
        lambda m: f"{m.group(1)}{read_int(int(m.group(2)))} {read_int(int(m.group(3)))}",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"({num_tok}\s+(?:phần trăm|triệu|tỷ|nghìn|ca|đồng|đô la|đô la Mỹ|độ|độ xê))\s*[-–—]\s*(?=\d)",
        r"\1 đến ",
        s,
    )
    s = re.sub(
        rf"(?<![-\w.,])({num_tok})\s*[-–—]\s*({num_tok})(?![.,]\d)",
        r"\1 đến \2",
        s,
    )
    s = re.sub(rf"({num_tok})\s*[xX×]\s*({num_tok})", r"\1 nhân \2", s)

    def _repl_fraction_or_score(m: re.Match) -> str:
        num = int(m.group(1))
        den = int(m.group(2))
        after = s[m.end() : m.end() + 20].lstrip().lower()
        before = s[max(0, m.start() - 20) : m.start()].lower()
        is_score = num >= den or (
            den in (10, 100)
            and (
                after.startswith(("điểm", "câu", "trận", "sao", "tiêu chí"))
                or "đạt" in before
            )
        )
        if is_score:
            return f"{read_int(num)} trên {read_int(den)}"
        den_words = "tư" if (den == 4 and num < 4) else read_int(den)
        return f"{read_int(num)} phần {den_words}"

    s = re.sub(
        r"(?<![/\w.,])\b(\d{1,3})\s*/\s*(\d{1,3})\b(?![/\w.,])",
        _repl_fraction_or_score,
        s,
    )

    # 8. Atomic signed & unsigned numeric forms
    # Allow punctuation such as ':', ';', quotes, and brackets before complete numbers
    # while excluding '.' and '/' so partial IP/version fragments are never matched.
    left_bnd = r"(?:(?<=^)|(?<=[\s(\[{:;\"'“”‘’]))"

    def _repl_vn_mixed(m: re.Match) -> str:
        sign = _resolve_sign_prefix(m.group(1), s, m.start())
        int_w = read_int(int(m.group(2).replace(".", "")))
        dec_w = read_digits_verbatim(m.group(3))
        return f"{sign}{int_w} phẩy {dec_w}"

    s = re.sub(
        rf"{left_bnd}([+-])?(\d{{1,3}}(?:\.\d{{3}})+),(\d+)\b",
        _repl_vn_mixed,
        s,
    )

    def _repl_en_mixed(m: re.Match) -> str:
        sign = _resolve_sign_prefix(m.group(1), s, m.start())
        int_w = read_int(int(m.group(2).replace(",", "")))
        dec_w = read_digits_verbatim(m.group(3))
        return f"{sign}{int_w} chấm {dec_w}"

    s = re.sub(
        rf"{left_bnd}([+-])?(\d{{1,3}}(?:,\d{{3}})+)\.(\d+)\b",
        _repl_en_mixed,
        s,
    )

    def _repl_en_thousands(m: re.Match) -> str:
        sign = _resolve_sign_prefix(m.group(1), s, m.start())
        return f"{sign}{read_int(int(m.group(2).replace(',', '')))}"

    s = re.sub(
        rf"{left_bnd}([+-])?(\d{{1,3}}(?:,\d{{3}}){{2,}})\b",
        _repl_en_thousands,
        s,
    )

    def _repl_vn_thousands(m: re.Match) -> str:
        sign = _resolve_sign_prefix(m.group(1), s, m.start())
        return f"{sign}{read_int(int(m.group(2).replace('.', '')))}"

    s = re.sub(
        rf"{left_bnd}([+-])?(\d{{1,3}}(?:\.\d{{3}})+)\b(?![.,]\d)",
        _repl_vn_thousands,
        s,
    )

    def _repl_comma_dec(m: re.Match) -> str:
        sign = _resolve_sign_prefix(m.group(1), s, m.start())
        int_w = read_int(int(m.group(2)))
        dec_w = read_digits_verbatim(m.group(3))
        return f"{sign}{int_w} phẩy {dec_w}"

    s = re.sub(
        rf"{left_bnd}([+-])?(\d+),(\d+)\b",
        _repl_comma_dec,
        s,
    )

    def _repl_dot_dec(m: re.Match) -> str:
        sign = _resolve_sign_prefix(m.group(1), s, m.start())
        int_w = read_int(int(m.group(2)))
        dec_w = read_digits_verbatim(m.group(3))
        return f"{sign}{int_w} chấm {dec_w}"

    s = re.sub(
        rf"{left_bnd}([+-])?(\d+)\.(\d+)\b",
        _repl_dot_dec,
        s,
    )

    # 9. Remaining signed & unsigned standalone integers
    def _repl_signed_int(m: re.Match) -> str:
        sign = _resolve_sign_prefix(m.group(1), s, m.start())
        return f"{sign}{read_int(int(m.group(2)))}"

    s = re.sub(rf"{left_bnd}([+-])?(\d+)\b", _repl_signed_int, s)

    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    orig_first = text.lstrip()[0] if text.lstrip() else ""
    if s and orig_first.isalpha() and orig_first.isupper() and s[0].isalpha():
        s = s[0].upper() + s[1:]
    return s
