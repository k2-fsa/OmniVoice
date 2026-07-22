import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.unicode_audio_tests.run_experiment import BASE_TEXT, run_experiment


class _FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [np.zeros(240, dtype=np.float32)]


class UnicodeExperimentTest(unittest.TestCase):
    def test_experiment_uses_existing_model_and_records_all_cases(self):
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = run_experiment(model, output_dir=temp_dir, seed=123)
            records = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(len(model.calls), 8)
            self.assertEqual(len(records), 8)
            self.assertEqual(
                {record["normalized_text"] for record in records}, {BASE_TEXT}
            )
            self.assertEqual({record["seed"] for record in records}, {123})
            self.assertTrue(all("ref_audio" not in call for call in model.calls))
            self.assertTrue(all("ref_text" not in call for call in model.calls))
            self.assertTrue(
                all(
                    (Path(temp_dir) / record["output_file"]).is_file()
                    for record in records
                )
            )


if __name__ == "__main__":
    unittest.main()
