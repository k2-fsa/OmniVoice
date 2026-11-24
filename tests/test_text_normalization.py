#!/usr/bin/env python3
# Copyright    2026  Phuc Nguyen
#
# See ../LICENSE for clarification regarding multiple authors
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

"""Unit tests for Vietnamese text normalization and language auto-detection."""

from omnivoice.utils.text import _resolve_lang_code, normalize_text
from omnivoice.utils.vi_normalizer import normalize_vietnamese_text, read_int


def test_vietnamese_integer_rules():
    assert read_int(0) == "không"
    assert read_int(5) == "năm"
    assert read_int(14) == "mười bốn"
    assert read_int(15) == "mười lăm"
    assert read_int(21) == "hai mươi mốt"
    assert read_int(24) == "hai mươi tư"
    assert read_int(25) == "hai mươi lăm"
    assert read_int(105) == "một trăm lẻ năm"
    assert read_int(1005) == "một nghìn không trăm lẻ năm"
    assert read_int(1_000_050_000) == "một tỷ không trăm năm mươi nghìn"
    assert read_int(1_000_000_000_000) == "một nghìn tỷ"
    assert read_int(1_500_000_000_000) == "một nghìn năm trăm tỷ"
    assert read_int(1_000_000_000_000_000) == "một triệu tỷ"
    assert read_int(1_000_000_000_000_000_000) == "một tỷ tỷ"


def test_mixed_thousands_decimals_and_signs():
    assert (
        normalize_text("1.284,65 điểm", language="vi")
        == "một nghìn hai trăm tám mươi tư phẩy sáu năm điểm"
    )
    assert (
        normalize_text("18.450,75 tỷ đồng", language="Vietnamese")
        == "mười tám nghìn bốn trăm năm mươi phẩy bảy năm tỷ đồng"
    )
    assert (
        normalize_text("1,234.56 USD", language="vi")
        == "một nghìn hai trăm ba mươi tư chấm năm sáu đô la Mỹ"
    )
    assert (
        normalize_text("1,200,000 lượt xem", language="vi")
        == "một triệu hai trăm nghìn lượt xem"
    )
    assert (
        normalize_text("1 500 000 người", language="vi")
        == "một triệu năm trăm nghìn người"
    )
    assert normalize_text("Nhiệt độ -5°C đến 22,5°C", language="vi") == (
        "Nhiệt độ âm năm độ xê đến hai mươi hai phẩy năm độ xê"
    )
    # Avoid duplicate 'âm âm' when user already wrote the word 'âm'
    assert (
        normalize_text("Lợi nhuận âm -125 tỷ", language="vi")
        == "Lợi nhuận âm một trăm hai mươi lăm tỷ"
    )
    assert normalize_text("Tăng +0,45% và lạm phát -0,15%", language="vi") == (
        "Tăng cộng không phẩy bốn năm phần trăm và lạm phát âm không phẩy một năm phần trăm"
    )
    assert normalize_text("Lãi suất từ 4,8% - 6,25%/năm", language="vi") == (
        "Lãi suất từ bốn phẩy tám phần trăm đến sáu phẩy hai năm phần trăm một năm"
    )


def test_phones_hotlines_otp_and_leading_zeros():
    assert (
        normalize_text("Gọi 0901234567", language="vi")
        == "Gọi không chín không một, hai ba bốn, năm sáu bảy"
    )
    assert (
        normalize_text("Liên hệ 0987.654.321", language="vi")
        == "Liên hệ không chín tám bảy, sáu năm bốn, ba hai một"
    )
    assert normalize_text("Hotline 1900 1080 hoặc 028 3822 5678", language="vi") == (
        "Hotline một chín không không, một không tám không hoặc "
        "không hai tám, ba tám hai hai, năm sáu bảy tám"
    )
    assert normalize_text("Số +84 912 345 678", language="vi") == (
        "Số cộng tám tư, chín một hai, ba bốn năm, sáu bảy tám"
    )
    assert (
        normalize_text("Mã OTP của bạn là 048291", language="vi")
        == "Mã OTP của bạn là không bốn tám hai chín một"
    )
    assert normalize_text("Điệp viên 007", language="vi") == "Điệp viên không không bảy"
    assert normalize_text("Căn cước 079095001234", language="vi") == (
        "Căn cước không bảy chín không, chín năm không không, một hai ba bốn"
    )


def test_dates_times_roman_numerals_units_and_versions():
    # Avoid duplicate 'ngày ngày' when input already includes 'ngày'
    assert normalize_text("Ngày 24/11/2026", language="vi") == (
        "Ngày hai mươi tư tháng mười một năm hai nghìn không trăm hai mươi sáu"
    )
    assert normalize_text("20/11/2026", language="vi") == (
        "ngày hai mươi tháng mười một năm hai nghìn không trăm hai mươi sáu"
    )
    assert normalize_text(
        "Báo cáo Quý III và Quý IV lúc 8h30p sáng ngày 30/4 và 14:05:30",
        language="vi",
    ) == (
        "Báo cáo Quý ba và Quý tư lúc tám giờ ba mươi phút sáng "
        "ngày ba mươi tháng tư và mười bốn giờ năm phút ba mươi giây"
    )
    assert normalize_text("Check-in lúc 14h00 hỗ trợ 24/7", language="vi") == (
        "Check-in lúc mười bốn giờ hỗ trợ hai mươi tư trên bảy"
    )
    assert normalize_text(
        "Phiên bản v2.4.1 tối ưu 2/3 băng thông tại IP 192.168.1.1 cho phòng 120 m²",
        language="vi",
    ) == (
        "Phiên bản hai chấm bốn chấm một tối ưu hai phần ba băng thông "
        "tại IP một chín hai chấm một sáu tám chấm một chấm một cho phòng một trăm hai mươi mét vuông"
    )
    assert normalize_text(
        "Combo 1.650.000đ giảm còn 199k hoặc mẫu 2tr5, màn hình 4K 60fps tỷ lệ 16:9",
        language="vi",
    ) == (
        "Combo một triệu sáu trăm năm mươi nghìn đồng giảm còn một trăm chín mươi chín ca "
        "hoặc mẫu hai triệu rưỡi, màn hình bốn ca sáu mươi ép pê ét tỷ lệ mười sáu chín"
    )
    assert normalize_vietnamese_text("Đội nhà thắng với tỷ số 2-1") == (
        "Đội nhà thắng với tỷ số hai một"
    )
    # Regression checks for 24/7 dates, spaced metres, Roman XL, postfix 1,234$, and colon-prefixed numbers
    assert normalize_text("Sự kiện ngày 24/7/2026 và ngày 24/7", language="vi") == (
        "Sự kiện ngày hai mươi tư tháng bảy năm hai nghìn không trăm hai mươi sáu và ngày hai mươi tư tháng bảy"
    )
    assert normalize_text("Dài 5 m và rộng 3m", language="vi") == (
        "Dài năm mét và rộng ba mét"
    )
    assert normalize_text("Chương XL gồm 1,234$ (Giá:100)", language="vi") == (
        "Chương bốn mươi gồm một nghìn hai trăm ba mươi tư đô la (Giá:một trăm)"
    )
    assert normalize_text("Giá:1.500đ", language="vi") == "Giá:một nghìn năm trăm đồng"
    # Structured dotted tokens not matched by IPv4/semver must not be partially rewritten
    assert normalize_text("IP:192.168.100.01", language="vi") == "IP:192.168.100.01"
    assert normalize_text("Phiên bản:2026.1.2", language="vi") == "Phiên bản:2026.1.2"


def test_inline_control_tags_and_english_loanwords_preserved():
    raw = "[laughter] Hệ thống Cloudflare REST API lúc 8h30 giá 1.500đ [B EY1 S] [sigh]"
    out = normalize_text(raw, language="vi")
    assert out == (
        "[laughter] Hệ thống Cloudflare REST API lúc tám giờ ba mươi phút "
        "giá một nghìn năm trăm đồng [B EY1 S] [sigh]"
    )


def test_language_autodetection_for_vietnamese_without_false_positives():
    # Auto-detects Vietnamese when language=None (works without WeTextProcessing installed)
    assert _resolve_lang_code(None, "Giá sản phẩm là 1.500đ") == "vi"
    assert (
        normalize_text("Giá sản phẩm là 1.500đ", language=None)
        == "Giá sản phẩm là một nghìn năm trăm đồng"
    )
    assert _resolve_lang_code(None, "今天是2026年") == "zh"

    # Non-Vietnamese accented Latin scripts & Pinyin must NOT be misclassified as 'vi'
    for non_vi in (
        "Café résumé à Paris 2026",  # French
        "El niño mañana 2026",  # Spanish
        "Übergröße München 2026",  # German
        "São Paulo coração 2026",  # Portuguese
        "România școală 2026",  # Romanian (bare 'ă')
        "Đorđe Petrović 2026",  # Serbian/Croatian (bare 'Đ')
        "Ọjọ́ 2026",  # Yoruba (underdotted 'ọ')
        "Ụlọ 2026",  # Igbo (underdotted 'ụ', 'ọ')
        "nǐ hǎo běijīng 2026",  # Pinyin carons
    ):
        assert _resolve_lang_code(None, non_vi) == "en", f"False positive on: {non_vi}"
