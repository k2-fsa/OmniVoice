"""Schema and high-recall detector tests for Vietnamese TN v1."""

import unittest

from experiments.vietnamese_text_normalization.schema import validate_case
from omnivoice.utils.vietnamese_normalization.detector import detect
from omnivoice.utils.vietnamese_normalization.types import CandidateLabel


class SchemaTest(unittest.TestCase):
    def test_half_open_offsets_and_taxonomy(self):
        text = "Nặng 2.5 kg"
        case = validate_case(
            {
                "id": "measurement-1",
                "text": text,
                "source": "unit-test",
                "spans": [
                    {
                        "start": 5,
                        "end": 11,
                        "surface": "2.5 kg",
                        "label": "MEASUREMENT",
                    }
                ],
            }
        )
        self.assertEqual(case.text[case.spans[0].start : case.spans[0].end], "2.5 kg")

    def test_rejects_bad_surface_unknown_label_and_overlap(self):
        base = {"id": "x", "text": "Có 12 kg", "source": "unit-test"}
        for spans, message in (
            ([{"start": 3, "end": 5, "surface": "99", "label": "CARDINAL"}], "surface"),
            ([{"start": 3, "end": 5, "surface": "12", "label": "PHONE"}], "one of"),
            (
                [
                    {"start": 3, "end": 5, "surface": "12", "label": "CARDINAL"},
                    {"start": 4, "end": 8, "surface": "2 kg", "label": "MEASUREMENT"},
                ],
                "non-overlapping",
            ),
        ):
            with self.subTest(spans=spans), self.assertRaisesRegex(ValueError, message):
                validate_case(base | {"spans": spans})


class DetectorTest(unittest.TestCase):
    def assert_detection(self, text, surface, label):
        spans = list(detect(text))
        matches = [span for span in spans if span.surface == surface]
        self.assertEqual(len(matches), 1, spans)
        span = matches[0]
        self.assertEqual(text[span.start : span.end], surface)
        self.assertIn(CandidateLabel(label), span.candidate_labels)

    def test_smallest_complete_boundaries(self):
        cases = (
            ("Phiên bản 2.5", "2.5", "VERSION"),
            ("ngày 20/11/2026", "20/11/2026", "DATE"),
            ("Nặng 2.5 kg", "2.5 kg", "MEASUREMENT"),
            ("Giá 500.000 đồng", "500.000 đồng", "CURRENCY"),
            ("Gọi 0903 123 456", "0903 123 456", "IDENTIFIER"),
            ("Tăng 15%", "15%", "PERCENT"),
            ("Tỉ số 2-1", "2-1", "SCORE"),
        )
        for text, surface, label in cases:
            with self.subTest(text=text):
                self.assert_detection(text, surface, label)

    def test_keep_candidates(self):
        self.assert_detection("Xem https://x.vn/v2.5", "https://x.vn/v2.5", "KEEP")
        self.assert_detection("Dùng model X100-Pro", "X100-Pro", "KEEP")


if __name__ == "__main__":
    unittest.main()
