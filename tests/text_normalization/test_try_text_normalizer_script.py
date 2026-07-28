import io
import unittest
from unittest.mock import patch

from omnivoice.text_normalization.types import CandidateSpan
from scripts import try_text_normalizer


class TryTextNormalizerCliTest(unittest.TestCase):
    def test_main_prints_result_and_supports_no_diagnostics(self):
        class FakeDetector:
            def __call__(self, _text):
                return [
                    CandidateSpan(7, 8, "CARDINAL", "2", 0.91),
                ]

        stdout = io.StringIO()
        with (
            patch.object(try_text_normalizer, "build_detector", return_value=FakeDetector()),
            patch("sys.stdout", stdout),
        ):
            exit_code = try_text_normalizer.main(["Tôi có 2 quyển sách", "--model-path", "custom-model", "--no-diagnostics"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Original text:", stdout.getvalue())
        self.assertIn("surface='2'", stdout.getvalue())
        self.assertIn("Tôi có hai quyển sách", stdout.getvalue())
        self.assertIn("Normalized text:", stdout.getvalue())
        self.assertNotIn("Structured diagnostics:", stdout.getvalue())

    def test_failure_preserves_input_and_returns_nonzero(self):
        stdout = io.StringIO()
        with (
            patch.object(
                try_text_normalizer,
                "build_detector",
                side_effect=RuntimeError("model missing"),
            ),
            patch("sys.stdout", stdout),
        ):
            exit_code = try_text_normalizer.main(["Giữ 007"])
        self.assertEqual(exit_code, 2)
        self.assertIn("Normalized text: Giữ 007", stdout.getvalue())
        self.assertIn("model missing", stdout.getvalue())
