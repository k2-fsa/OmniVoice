"""Deterministic v1 Vietnamese parser and verbalizer coverage."""

import unittest

from omnivoice.utils.vietnamese_normalization import (
    CandidateLabel,
    parse_value,
    verbalize,
)


class VerbalizerTest(unittest.TestCase):
    def render(self, surface, label):
        label = CandidateLabel(label)
        return verbalize(parse_value(surface, label), label)

    def test_every_v1_label(self):
        cases = (
            ("21", "CARDINAL", "hai mươi mốt"),
            ("2026", "YEAR", "hai nghìn không trăm hai mươi sáu"),
            ("007", "IDENTIFIER", "không không bảy"),
            ("1,05", "DECIMAL", "một phẩy không năm"),
            ("1/2", "FRACTION", "một phần hai"),
            ("20/11/2026", "DATE", "hai mươi tháng mười một năm hai nghìn không trăm hai mươi sáu"),
            ("1/2", "DATE", "một tháng hai"),
            ("07:05", "TIME", "bảy giờ năm phút"),
            ("v2.5", "VERSION", "vê hai chấm năm"),
            ("2-1", "SCORE", "hai một"),
            ("2–5", "RANGE", "hai đến năm"),
            ("500.000 đồng", "CURRENCY", "năm trăm nghìn đồng"),
            ("2.5 kg", "MEASUREMENT", "hai phẩy năm ki lô gam"),
            ("15%", "PERCENT", "mười lăm phần trăm"),
            ("IV", "ROMAN", "bốn"),
            ("model X100-Pro", "KEEP", "model X100-Pro"),
        )
        for surface, label, expected in cases:
            with self.subTest(label=label, surface=surface):
                self.assertEqual(self.render(surface, label), expected)

    def test_lossless_decimal_and_identifier_components(self):
        self.assertEqual(self.render("2.50", "DECIMAL"), "hai phẩy năm không")
        self.assertEqual(
            self.render("0903 123 456", "IDENTIFIER"),
            "không chín không ba một hai ba bốn năm sáu",
        )

    def test_unicode_and_additional_structures(self):
        cases = (
            ("2026-11-20", "DATE", "năm hai không hai sáu tháng mười một ngày hai mươi"),
            ("23:59:08", "TIME", "hai mươi ba giờ năm mươi chín phút tám giây"),
            ("US$1,05", "CURRENCY", "một phẩy không năm đô la Mỹ"),
            ("37°C", "MEASUREMENT", "ba mươi bảy độ xê"),
            ("1,5‰", "PERCENT", "một phẩy năm phần nghìn"),
        )
        for surface, label, expected in cases:
            with self.subTest(surface=surface):
                self.assertEqual(self.render(surface, label), expected)

    def test_malformed_and_unsupported_values_fail_closed(self):
        cases = (
            ("1,2.3", "DECIMAL"),
            ("1/0", "FRACTION"),
            ("99/99/2026", "DATE"),
            ("25:61", "TIME"),
            ("v2.-5", "VERSION"),
            ("10 BTC", "CURRENCY"),
            ("12 furlong", "MEASUREMENT"),
            ("IIV", "ROMAN"),
        )
        for surface, label in cases:
            with self.subTest(surface=surface), self.assertRaises(ValueError):
                parse_value(surface, CandidateLabel(label))

    def test_label_type_mismatch_is_rejected(self):
        parsed = parse_value("007", CandidateLabel.IDENTIFIER)
        with self.assertRaises(TypeError):
            verbalize(parsed, CandidateLabel.CARDINAL)


if __name__ == "__main__":
    unittest.main()
