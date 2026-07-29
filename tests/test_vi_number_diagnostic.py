import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.vi_number_normalization.diagnostic import (
    GenerationSettings,
    build_variants,
    load_role_map,
    read_jsonl,
    run_audio_generation,
    select_subset,
    stable_audio_filename,
    summarize_evaluations,
    validate_records,
    write_evaluation_csv,
    write_jsonl,
)


def _record(record_id="one", number="1", role="QUANTITY", preferred=None):
    return {
        "id": record_id,
        "raw_text": f"Giá trị là {number}.",
        "number": number,
        "role": role,
        "preferred_spoken": preferred or [f"Giá trị là {number} đọc."],
    }


class ValidationTest(unittest.TestCase):
    def test_jsonl_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text('{"id": "ok"}\nnot-json\n', encoding="utf-8")
            records, errors = read_jsonl(path)
        self.assertEqual(len(records), 1)
        self.assertRegex(errors[0], "line 2: invalid JSON")

    def test_duplicate_ids_are_invalid(self):
        records = [_record("duplicate"), _record("duplicate", "2")]
        errors, _, _ = validate_records(records, {"QUANTITY": "quantity"})
        self.assertTrue(any("duplicate id" in error for error in errors))

    def test_invalid_preferred_spoken_is_rejected(self):
        record = _record()
        record["preferred_spoken"] = [""]
        errors, _, _ = validate_records([record], {"QUANTITY": "quantity"})
        self.assertTrue(any("nonempty strings" in error for error in errors))

    def test_present_acceptable_spoken_must_be_a_list(self):
        record = _record()
        record["acceptable_spoken"] = None
        errors, _, _ = validate_records([record], {"QUANTITY": "quantity"})
        self.assertTrue(
            any("acceptable_spoken must be a list" in error for error in errors)
        )

    def test_unknown_role_is_reported(self):
        errors, _, _ = validate_records([_record(role="UNKNOWN")], {})
        self.assertTrue(any("unknown role" in error for error in errors))

    def test_repository_role_mapping_is_unambiguous(self):
        mapping_path = (
            Path(__file__).parents[1]
            / "experiments"
            / "vi_number_normalization"
            / "role_groups.json"
        )
        role_map, errors = load_role_map(mapping_path)
        self.assertEqual(errors, [])
        self.assertEqual(role_map["PHONE"], "phone")
        self.assertEqual(role_map["DATE"], "date/year")

    def test_repository_dataset_validates(self):
        experiment_dir = (
            Path(__file__).parents[1] / "experiments" / "vi_number_normalization"
        )
        records, parse_errors = read_jsonl(experiment_dir / "data" / "pilot.jsonl")
        role_map, mapping_errors = load_role_map(experiment_dir / "role_groups.json")
        validation_errors, warnings, roles = validate_records(records, role_map)
        self.assertEqual(parse_errors + mapping_errors + validation_errors, [])
        self.assertEqual(len(records), 200)
        self.assertTrue(warnings)
        self.assertEqual(sum(roles.values()), 200)


class SelectionAndVariantTest(unittest.TestCase):
    def setUp(self):
        self.role_map = {
            "QUANTITY": "quantity",
            "PHONE": "phone",
            "DATE": "date/year",
        }
        self.records = [
            _record("quantity_same", "12", "QUANTITY", ["mười hai"]),
            _record("phone_same", "12", "PHONE", ["một hai"]),
            _record("date", "2026", "DATE", ["năm hai nghìn không trăm hai sáu"]),
            _record("quantity_other", "3", "QUANTITY", ["ba"]),
            _record("phone_other", "4", "PHONE", ["bốn"]),
        ]

    def test_selection_is_deterministic_and_keeps_contrastive_pair(self):
        first = select_subset(self.records, self.role_map, sample_size=4, seed=7)
        second = select_subset(self.records, self.role_map, sample_size=4, seed=7)
        self.assertEqual(
            [item["id"] for item in first], [item["id"] for item in second]
        )
        selected_ids = {item["id"] for item in first}
        self.assertIn("quantity_same", selected_ids)
        self.assertIn("phone_same", selected_ids)
        self.assertEqual(
            {self.role_map[item["role"]] for item in first},
            {"quantity", "phone", "date/year"},
        )

    def test_variants_use_real_normalizer_once_and_never_double_normalize(self):
        calls = []

        def normalizer(text, language):
            calls.append((text, language))
            return "CURRENT"

        selection = {
            "records": [
                {
                    **_record(preferred=["ORACLE"]),
                    "broad_group": "quantity",
                }
            ]
        }
        variants = build_variants(
            selection, Path("/tmp/output"), GenerationSettings(), normalizer
        )
        self.assertEqual(calls, [("Giá trị là 1.", "vi")])
        self.assertEqual(
            {item["system"]: item["text_sent"] for item in variants},
            {"raw": "Giá trị là 1.", "current": "CURRENT", "oracle": "ORACLE"},
        )
        self.assertTrue(all(item["normalize_text"] is False for item in variants))

    def test_filenames_are_stable_safe_and_collision_free(self):
        names = {
            stable_audio_filename("unsafe id/1", system)
            for system in ("raw", "current", "oracle")
        }
        self.assertEqual(len(names), 3)
        self.assertTrue(
            all("/" not in name and name.endswith(".wav") for name in names)
        )

    def test_manifest_and_evaluation_paths_match(self):
        selection = {"records": [{**_record(), "broad_group": "quantity"}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            variants = build_variants(
                selection,
                output_dir,
                GenerationSettings(),
                lambda text, language: text,
            )
            manifest = output_dir / "variants.jsonl"
            evaluation = output_dir / "evaluation.csv"
            write_jsonl(manifest, variants)
            write_evaluation_csv(evaluation, variants)
            with evaluation.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["audio_path"], variants[0]["audio_path"])
        self.assertTrue(all(not row["number_correct"] for row in rows))


class _FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.generate_calls = []
        self.prompt_calls = []

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return "fixed-prompt"

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[0.0, 0.0]]


class GenerationAndEvaluationTest(unittest.TestCase):
    def test_generation_loads_once_resumes_and_disables_normalization(self):
        fake_model = _FakeModel()
        factory_calls = []

        def factory(*args, **kwargs):
            factory_calls.append((args, kwargs))
            return fake_model

        def writer(path, audio, sampling_rate):
            Path(path).write_bytes(b"wav")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            variants = []
            for system in ("raw", "current", "oracle"):
                variants.append(
                    {
                        "id": "one",
                        "system": system,
                        "text_sent": system.upper(),
                        "audio_path": str(output_dir / f"{system}.wav"),
                    }
                )
            Path(variants[0]["audio_path"]).write_bytes(b"existing")
            summary = run_audio_generation(
                variants,
                output_dir,
                "checkpoint",
                "reference.wav",
                "reference text",
                "cpu",
                GenerationSettings(seed=9),
                model_factory=factory,
                audio_writer=writer,
                seed_setter=lambda seed: None,
            )
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(len(fake_model.prompt_calls), 1)
        self.assertEqual(summary, {"skipped": 1, "generated": 2})
        self.assertEqual(len(fake_model.generate_calls), 2)
        self.assertTrue(
            all(call["normalize_text"] is False for call in fake_model.generate_calls)
        )
        self.assertTrue(
            all(
                call["voice_clone_prompt"] == "fixed-prompt"
                for call in fake_model.generate_calls
            )
        )

    def test_incomplete_evaluation_rows_are_safe(self):
        rows = [
            {
                "id": "one",
                "system": "raw",
                "broad_group": "quantity",
                "number_correct": "",
                "pronunciation_clear": "",
                "naturalness_1_to_5": "",
                "error_type": "",
            },
            {
                "id": "one",
                "system": "current",
                "broad_group": "quantity",
                "number_correct": "no",
                "pronunciation_clear": "yes",
                "naturalness_1_to_5": "4",
                "error_type": "substituted-number",
            },
            {
                "id": "one",
                "system": "oracle",
                "broad_group": "quantity",
                "number_correct": "yes",
                "pronunciation_clear": "yes",
                "naturalness_1_to_5": "5",
                "error_type": "",
            },
        ]
        summary = summarize_evaluations(rows)
        self.assertIsNone(summary["systems"]["raw"]["numeric_correctness_rate"])
        self.assertEqual(summary["systems"]["current"]["numeric_correctness_rate"], 0)
        self.assertEqual(summary["systems"]["oracle"]["mean_naturalness"], 5)
        self.assertEqual(summary["paired_records_evaluated"], 0)
        self.assertEqual(summary["paired_results"]["numeric_correctness"]["records"], 0)

    def test_paired_summary_identifies_oracle_outcomes(self):
        rows = []
        for record_id, results in (
            ("oracle_fixes", {"raw": "no", "current": "no", "oracle": "yes"}),
            ("oracle_fails", {"raw": "no", "current": "yes", "oracle": "no"}),
        ):
            for system, result in results.items():
                rows.append(
                    {
                        "id": record_id,
                        "system": system,
                        "broad_group": "quantity",
                        "number_correct": result,
                        "pronunciation_clear": "yes",
                        "naturalness_1_to_5": "3",
                        "error_type": "",
                    }
                )
        summary = summarize_evaluations(rows)
        self.assertEqual(summary["paired_results"]["numeric_correctness"]["records"], 2)
        self.assertEqual(
            summary["paired_results"]["numeric_correctness"]["current"], 0.5
        )
        self.assertEqual(
            summary["oracle_succeeds_where_raw_or_current_fails"], ["oracle_fixes"]
        )
        self.assertEqual(summary["oracle_also_fails"], ["oracle_fails"])


if __name__ == "__main__":
    unittest.main()
