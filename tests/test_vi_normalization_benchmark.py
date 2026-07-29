"""Focused integrity tests for the Vietnamese TN benchmark harness."""

import unittest
from unittest.mock import patch

from experiments.vietnamese_text_normalization.adapters import (
    contextual_rule,
    direct_num2words,
    identity,
    omnivoice_fallback,
)
from experiments.vietnamese_text_normalization.schema import validate_record
from experiments.vietnamese_text_normalization.split_dataset import (
    assign_splits,
    canonicalize,
)


def source_record(case_id, text, number, role="QUANTITY"):
    return {
        "id": case_id,
        "raw_text": text,
        "number": number,
        "role": role,
        "preferred_spoken": [text.replace(number, "một")],
    }


class SchemaAndSplitTest(unittest.TestCase):
    def test_schema_rejects_missing_and_nonoccurring_spans(self):
        with self.assertRaises(ValueError):
            validate_record({})
        records = canonicalize([source_record("a", "Có 1 hộp.", "1")], "test")
        value = records[0].to_dict() | {"numeric_span": "9"}
        with self.assertRaisesRegex(ValueError, "must occur"):
            validate_record(value)

    def test_split_is_deterministic_and_keeps_template_groups_together(self):
        raw = [
            source_record(f"q{i}", f"Kho {i} có {i + 10} hộp.", str(i + 10))
            for i in range(10)
        ]
        records = canonicalize(raw, "test")
        first = assign_splits(records)
        self.assertEqual(first, assign_splits(records))
        groups = {}
        for record in records:
            groups.setdefault(record.template_group, set()).add(first[record.id])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))


class AdapterTest(unittest.TestCase):
    def test_identity_and_num2words_adapters(self):
        self.assertFalse(identity("Có 2 hộp.").changed)
        direct = direct_num2words("Có 2 hộp.")
        wrapped = omnivoice_fallback("Có 2 hộp.")
        self.assertTrue(direct.available)
        self.assertEqual(direct.output_text, wrapped.output_text)

    def test_contextual_adapter_reports_abstention(self):
        def fake_detector(text):
            if "105" in text:
                start = text.index("105")
                return [
                    {
                        "start": start,
                        "end": start + 3,
                        "label": "CARDINAL",
                        "text": "105",
                    }
                ]
            start = text.index("10/10")
            return [
                {
                    "start": start,
                    "end": start + 5,
                    "label": "RANGE",
                    "text": "10/10",
                }
            ]

        with patch(
            "experiments.vietnamese_text_normalization.adapters.get_bamibert_detector",
            return_value=fake_detector,
        ):
            supported = contextual_rule("Phòng 105.")
            self.assertEqual(supported.output_text, "Phòng một không năm.")
            self.assertFalse(supported.uncertain)
            unsupported = contextual_rule("Thị lực đạt 10/10.")
            self.assertEqual(unsupported.output_text, "Thị lực đạt 10/10.")
            self.assertTrue(unsupported.uncertain)


if __name__ == "__main__":
    unittest.main()
