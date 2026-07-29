import json
import tempfile
import unittest
from pathlib import Path

from experiments.tn_value_gate.adapters.base import NormalizationResult
from experiments.tn_value_gate.adapters.num2words_vi import normalize
from experiments.tn_value_gate.evaluation.canonicalize import canonicalize
from experiments.tn_value_gate.run_all import load_dataset


class CanonicalizationTests(unittest.TestCase):
    def test_only_unicode_and_formatting_are_canonicalized(self):
        self.assertEqual(canonicalize("  Việt  Nam .  "), "Việt Nam.")
        self.assertNotEqual(canonicalize("Việt Nam"), canonicalize("việt nam"))


class DatasetTests(unittest.TestCase):
    def test_jsonl_parser_and_duplicate_guard(self):
        row = {
            "id": "x",
            "raw_text": "Có 12",
            "number": "12",
            "role": "cardinal",
            "preferred_spoken": ["Có mười hai"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            self.assertEqual(load_dataset(path), [row])
            path.write_text(
                "\n".join([json.dumps(row), json.dumps(row)]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_dataset(path)


class AdapterTests(unittest.TestCase):
    def test_result_contract(self):
        result = NormalizationResult(
            output_text="x",
            supported=True,
            status="success",
            error_type=None,
            error_message=None,
            latency_ms=1.0,
        )
        self.assertEqual(result.to_dict()["error_type"], None)

    def test_num2words_marks_structured_context_partial(self):
        result = normalize("Tôi chuyển 150.000đ lúc 08:30.")
        self.assertEqual(result.status, "partial")
        self.assertFalse(result.supported)
        self.assertGreater(result.metadata["attempted_spans"], 1)

    def test_num2words_handles_plain_integer_span(self):
        result = normalize("Kho có 105 hộp")
        self.assertEqual(result.status, "success")
        self.assertIn("một trăm lẻ năm", result.output_text)


if __name__ == "__main__":
    unittest.main()
