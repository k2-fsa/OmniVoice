"""CLI, batch CLI, and Gradio forwarding for target normalization."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from omnivoice.cli import demo, infer, infer_batch


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.generate_calls = []
        self.prompt_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        texts = kwargs["text"] if isinstance(kwargs["text"], list) else [kwargs["text"]]
        return [np.zeros(24, dtype=np.float32) for _ in texts]

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return object()


class SingleCliTest(unittest.TestCase):
    def test_parser_defaults_off_and_flag_enables(self):
        parser = infer.get_parser()
        base = ["--text", "Có 2 hộp", "--output", "out.wav"]
        self.assertFalse(parser.parse_args(base).normalize_text)
        self.assertTrue(parser.parse_args(base + ["--normalize-text"]).normalize_text)
        self.assertFalse(parser.parse_args(base + ["--no-normalize-text"]).normalize_text)
        self.assertIsNone(parser.parse_args(base).bamibert_model_path)

    def test_main_forwards_boolean_without_touching_other_text_fields(self):
        for enabled in (False, True):
            model = FakeModel()
            argv = [
                "omnivoice-infer",
                "--text",
                "Có 2 hộp",
                "--ref_text",
                "Mẫu có 1 hộp",
                "--instruct",
                "female, low pitch",
                "--output",
                "out.wav",
                "--device",
                "cpu",
            ]
            if enabled:
                argv.append("--normalize-text")
            else:
                argv.append("--no-normalize-text")
            with (
                self.subTest(enabled=enabled),
                patch.object(sys, "argv", argv),
                patch.object(infer.OmniVoice, "from_pretrained", return_value=model),
                patch.object(infer.sf, "write"),
            ):
                infer.main()
            call = model.generate_calls[-1]
            self.assertIs(call["normalize_text"], enabled)
            self.assertEqual(call["text"], "Có 2 hộp")
            self.assertEqual(call["ref_text"], "Mẫu có 1 hộp")
            self.assertEqual(call["instruct"], "female, low pitch")


class ConfigurationConsistencyTest(unittest.TestCase):
    def test_all_entrypoints_share_defaults_and_have_no_backend_selector(self):
        parsers_and_args = (
            (
                infer.get_parser(),
                ["--text", "Có 2 hộp", "--output", "out.wav"],
            ),
            (
                infer_batch.get_parser(),
                ["--test_list", "input.jsonl", "--res_dir", "results"],
            ),
            (demo.build_parser(), []),
        )
        for parser, argv in parsers_and_args:
            with self.subTest(prog=parser.prog):
                args = parser.parse_args(argv)
                self.assertFalse(args.normalize_text)
                self.assertIsNone(args.bamibert_model_path)
                self.assertIsNone(args.bamibert_device)
                self.assertFalse(
                    any("backend" in action.dest for action in parser._actions)
                )


class BatchCliTest(unittest.TestCase):
    def test_parser_defaults_off_and_flag_enables(self):
        parser = infer_batch.get_parser()
        base = ["--test_list", "input.jsonl", "--res_dir", "results"]
        self.assertFalse(parser.parse_args(base).normalize_text)
        self.assertTrue(parser.parse_args(base + ["--normalize-text"]).normalize_text)
        self.assertFalse(parser.parse_args(base + ["--no-normalize-text"]).normalize_text)
        self.assertIsNone(parser.parse_args(base).bamibert_device)

    def test_worker_forwards_boolean_without_touching_other_text_fields(self):
        sample = (
            "case-1",
            "Mẫu có 1 hộp",
            "ref.wav",
            "Có 2 hộp",
            "vi",
            None,
            None,
            "female",
        )
        original = infer_batch.worker_model
        try:
            for enabled in (False, True):
                model = FakeModel()
                infer_batch.worker_model = model
                with (
                    self.subTest(enabled=enabled),
                    patch.object(infer_batch.sf, "write"),
                ):
                    infer_batch.run_inference_batch(
                        [sample], "results", normalize_text=enabled
                    )
                call = model.generate_calls[-1]
                self.assertIs(call["normalize_text"], enabled)
                self.assertEqual(call["text"], ["Có 2 hộp"])
                self.assertEqual(call["ref_text"], ["Mẫu có 1 hộp"])
                self.assertEqual(call["instruct"], ["female"])
        finally:
            infer_batch.worker_model = original

    def test_multiple_batch_items_forward_exact_fields_consistently(self):
        samples = [
            (
                "case-1",
                "Mẫu\u00a0có 1 hộp",
                "ref-1.wav",
                "  Có 2 hộp  ",
                "vi",
                None,
                None,
                "female",
            ),
            (
                "case-2",
                "Mẫu có 3 túi",
                "ref-2.wav",
                "Có 4 túi",
                "vi",
                None,
                None,
                "low pitch",
            ),
        ]
        original = infer_batch.worker_model
        try:
            model = FakeModel()
            infer_batch.worker_model = model
            with patch.object(infer_batch.sf, "write"):
                infer_batch.run_inference_batch(
                    samples,
                    "results",
                    normalize_text=True,
                )
            call = model.generate_calls[-1]
            self.assertEqual(call["text"], ["  Có 2 hộp  ", "Có 4 túi"])
            self.assertEqual(
                call["ref_text"],
                ["Mẫu\u00a0có 1 hộp", "Mẫu có 3 túi"],
            )
            self.assertEqual(call["instruct"], ["female", "low pitch"])
            self.assertIs(call["normalize_text"], True)
        finally:
            infer_batch.worker_model = original


class GradioTest(unittest.TestCase):
    def test_parser_default_and_flags(self):
        parser = demo.build_parser()
        self.assertFalse(parser.parse_args([]).normalize_text)
        self.assertTrue(parser.parse_args(["--normalize-text"]).normalize_text)
        self.assertFalse(parser.parse_args(["--no-normalize-text"]).normalize_text)

    def test_parser_reads_normalization_default_from_environment(self):
        with patch.dict(
            os.environ, {"OMNIVOICE_NORMALIZE_TEXT": "true"}, clear=False
        ):
            self.assertTrue(demo.build_parser().parse_args([]).normalize_text)

    def test_checkbox_defaults_off_and_clone_event_forwards_value(self):
        model = FakeModel()
        app = demo.build_demo(model, "test")
        checkboxes = [
            component
            for component in app.config["components"]
            if component.get("props", {}).get("label") == "Chuẩn hóa tiếng Việt"
        ]
        self.assertEqual(len(checkboxes), 2)
        self.assertTrue(
            all(component["props"].get("value") is False for component in checkboxes)
        )

        clone_fn = next(
            block_fn.fn
            for block_fn in app.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "_clone_fn"
        )
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                clone_fn(
                    "Có 2 hộp",
                    "vi",
                    "ref.wav",
                    "Mẫu có 1 hộp",
                    "female",
                    32,
                    2.0,
                    True,
                    1.0,
                    None,
                    True,
                    True,
                    enabled,
                )
                call = model.generate_calls[-1]
                self.assertIs(call["normalize_text"], enabled)
                self.assertEqual(call["text"], "Có 2 hộp")
                self.assertEqual(call["instruct"], "female")
                self.assertEqual(
                    model.prompt_calls[-1]["ref_text"], "Mẫu có 1 hộp"
                )

    def test_clone_prompt_errors_are_returned_to_the_ui(self):
        model = FakeModel()
        app = demo.build_demo(model, "test")
        clone_fn = next(
            block_fn.fn
            for block_fn in app.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "_clone_fn"
        )
        with patch.object(
            model,
            "create_voice_clone_prompt",
            side_effect=RuntimeError("ASR model unavailable"),
        ):
            audio, status = clone_fn(
                "Có 2 hộp",
                "vi",
                "ref.wav",
                None,
                "female",
                32,
                2.0,
                True,
                1.0,
                None,
                True,
                True,
                False,
            )
        self.assertIsNone(audio)
        self.assertEqual(status, "Error: RuntimeError: ASR model unavailable")

    def test_invalid_environment_configuration_exits_cleanly(self):
        with (
            patch.dict(os.environ, {"OMNIVOICE_PORT": "invalid"}, clear=False),
            self.assertLogs(level="ERROR") as logs,
        ):
            self.assertEqual(demo.main([]), 2)
        self.assertIn("Invalid demo configuration", "\n".join(logs.output))

    def test_configured_checkbox_default_is_enabled(self):
        app = demo.build_demo(
            FakeModel(),
            "test",
            normalize_text_default=True,
        )
        checkboxes = [
            component
            for component in app.config["components"]
            if component.get("props", {}).get("label") == "Chuẩn hóa tiếng Việt"
        ]
        self.assertEqual(len(checkboxes), 2)
        self.assertTrue(all(item["props"]["value"] is True for item in checkboxes))

    def test_disabled_gradio_semantic_normalization_still_applies_unicode_cleanup(
        self,
    ):
        model = FakeModel()
        app = demo.build_demo(model, "test")
        design_fn = next(
            block_fn.fn
            for block_fn in app.fns.values()
            if getattr(block_fn.fn, "__name__", "") == "_design_fn"
        )
        design_fn(
            "  Có 2 hộp  ",
            "vi",
            32,
            2.0,
            True,
            1.0,
            None,
            True,
            True,
            False,
        )
        self.assertEqual(model.generate_calls[-1]["text"], "Có 2 hộp")


if __name__ == "__main__":
    unittest.main()
