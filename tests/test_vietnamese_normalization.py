"""Behavioral tests for contextual Vietnamese text normalization."""

import unittest
from unittest.mock import patch

from omnivoice.utils.text import normalize_text
from omnivoice.utils.vietnamese_normalization import (
    normalize as custom_normalize,
    normalize_with_trace,
)
from omnivoice.utils.vietnamese_normalization.types import SemioticClass
from omnivoice.utils.vietnamese_normalization.verbalizers import (
    cardinal,
    decimal,
    digits,
)


REGRESSIONS = (
    (
        "Ngày 29/02/2024 là ngày nhuận.",
        "Ngày hai mươi chín tháng hai năm hai nghìn không trăm hai mươi tư là ngày nhuận.",
    ),
    ("Khi cần cấp cứu, hãy gọi 115.", "Khi cần cấp cứu, hãy gọi một một năm."),
    ("Có 2/3 số người đồng ý.", "Có hai phần ba số người đồng ý."),
    ("Mã xác nhận là 105.", "Mã xác nhận là một không năm."),
    ("Khoảng tin cậy nằm trong đoạn [0,1].", "Khoảng tin cậy nằm trong đoạn [0,1]."),
    (
        "Hệ thống lưu ngày theo định dạng 2026-07-21.",
        "Hệ thống lưu ngày theo định dạng năm hai không hai sáu tháng bảy ngày hai mốt.",
    ),
    ("Thị lực của cô ấy đạt 10/10.", "Thị lực của cô ấy đạt 10/10."),
    ("Món đồ này có giá HK$500.", "Món đồ này có giá năm trăm đô la Hồng Kông."),
    ("Đơn hàng mang mã 105.", "Đơn hàng mang mã một không năm."),
    ("Tôi đang ở phòng 105.", "Tôi đang ở phòng một không năm."),
    ("Hãy gọi đường dây 1080.", "Hãy gọi đường dây một không tám không."),
    (
        "Cửa hàng đóng cửa lúc 23:59.",
        "Cửa hàng đóng cửa lúc hai mươi ba giờ năm mươi chín phút.",
    ),
)


class MorphologyTest(unittest.TestCase):
    def test_cardinals(self):
        cases = {
            0: "không",
            5: "năm",
            15: "mười lăm",
            21: "hai mươi mốt",
            24: "hai mươi tư",
            105: "một trăm linh năm",
            2024: "hai nghìn không trăm hai mươi tư",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(cardinal(value, value >= 1000), expected)

    def test_digit_and_decimal_readings(self):
        self.assertEqual(digits("00105"), "không không một không năm")
        self.assertEqual(decimal("-12,04"), "âm mười hai phẩy không bốn")


class RegressionTest(unittest.TestCase):
    """Research-baseline regression tests; not the production routing test."""

    def test_supplied_regressions(self):
        for raw, expected in REGRESSIONS:
            with self.subTest(raw=raw):
                self.assertEqual(custom_normalize(raw), expected)

    def test_same_surface_is_disambiguated(self):
        expected = {
            "Kho hiện có 105 hộp.": "Kho hiện có một trăm linh năm hộp.",
            "Mã xác nhận là 105.": "Mã xác nhận là một không năm.",
            "Tôi đang ở phòng 105.": "Tôi đang ở phòng một không năm.",
        }
        for raw, spoken in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(custom_normalize(raw), spoken)

    def test_selected_classes_and_multiple_spans(self):
        cases = {
            "Gọi số 0903120508 lúc 07:30.": "Gọi số không chín không ba một hai không năm không tám lúc bảy giờ ba mươi phút.",
            "Tôi trả $25 cho 2 món.": "Tôi trả hai mươi lăm đô la cho hai món.",
            "Mã 007 và mã 105.": "Mã không không bảy và mã một không năm.",
        }
        for raw, spoken in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(custom_normalize(raw), spoken)

    def test_invalid_and_incomplete_input_is_safe(self):
        class Stub:
            def normalize(self, text):
                return custom_normalize(text)

        with patch("omnivoice.utils.text._get_vi_normalizer", return_value=Stub()):
            for raw in (
                "Mã là +.",
                "Ngày 99/99/9999.",
                "Khoảng [0,].",
                "abc123def",
                "",
                "Không có chữ số.",
                "Gọi 105 ngay.",
                "Thị lực đạt 10/10.",
                "Giảm 10%.",
                "Quãng đường 12km.",
            ):
                with self.subTest(raw=raw):
                    self.assertEqual(normalize_text(raw, "vi"), raw)

    def test_ambiguous_slash_abstains_without_context(self):
        self.assertEqual(custom_normalize("Chọn 1/2."), "Chọn 1/2.")
        self.assertEqual(
            custom_normalize("Hẹn ngày 1/2."),
            "Hẹn ngày một tháng hai.",
        )

    def test_protected_spans_and_spacing(self):
        raw = "Có 2 hộp [laughter] và mã [B EY1 S] 105."
        class Stub:
            def normalize(self, text):
                return custom_normalize(text)

        with patch("omnivoice.utils.text._get_vi_normalizer", return_value=Stub()):
            self.assertEqual(
                normalize_text(raw, "vi"),
                "Có hai hộp [laughter] và mã [B EY1 S] một không năm.",
            )

    def test_trace_is_deterministic_and_inspectable(self):
        first = normalize_with_trace("Mã xác nhận là 105.")
        self.assertEqual(first, normalize_with_trace("Mã xác nhận là 105."))
        self.assertEqual(first.decisions[0].semiotic_class, SemioticClass.IDENTIFIER)
        self.assertFalse(first.decisions[0].uncertain)
        unsupported = normalize_with_trace("Thị lực đạt 10/10.").decisions[0]
        self.assertFalse(unsupported.supported)
        self.assertTrue(unsupported.uncertain)


class RoutingCompatibilityTest(unittest.TestCase):
    class Stub:
        def __init__(self, prefix):
            self.prefix = prefix

        def normalize(self, text):
            return self.prefix + text

    def test_english_and_chinese_still_use_existing_routes(self):
        with patch(
            "omnivoice.utils.text._get_en_normalizer", return_value=self.Stub("EN:")
        ):
            self.assertEqual(normalize_text("  12 [tag] ", "en"), "  EN:12 [tag] ")
        with patch(
            "omnivoice.utils.text._get_zh_normalizer", return_value=self.Stub("ZH:")
        ):
            self.assertEqual(normalize_text("数字 12", "zh"), "ZH:数字 12")

    def test_other_language_fallback_is_unchanged(self):
        with patch(
            "omnivoice.utils.text._num2words_segment",
            side_effect=lambda text, lang: f"{lang}:{text}",
        ):
            self.assertEqual(normalize_text("12", "fr"), "fr:12")


class RealVietNormalizerIntegrationTest(unittest.TestCase):
    def setUp(self):
        import omnivoice.utils.text as text_utils

        text_utils._VI_NORMALIZER = None

    def tearDown(self):
        import omnivoice.utils.text as text_utils

        text_utils._VI_NORMALIZER = None

    def test_real_numeric_backend_preserves_boundary_and_reaches_public_route(self):
        self.assertEqual(
            normalize_text("Hôm nay tôi nhận được 105 đơn hàng.", "vi"),
            "Hôm nay tôi nhận được một trăm lẻ năm đơn hàng.",
        )
        self.assertEqual(
            normalize_text("Gói hàng nặng 2.5 kg.", "vi"),
            "Gói hàng nặng hai phẩy năm ki-lô-gam.",
        )


if __name__ == "__main__":
    unittest.main()
