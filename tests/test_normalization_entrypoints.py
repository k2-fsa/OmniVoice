"""CLI, batch CLI, and Gradio forwarding for target normalization."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

from omnivoice.cli import demo, infer, infer_batch


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.generate_calls = []
        self.prompt_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [np.zeros(24, dtype=np.float32)]

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return object()


class SingleCliTest(unittest.TestCase):
    def test_parser_defaults_off_and_flag_enables(self):
        parser = infer.get_parser()
        base = ["--text", "Có 2 hộp", "--output", "out.wav"]
        self.assertFalse(parser.parse_args(base).normalize_text)
        self.assertTrue(parser.parse_args(base + ["--normalize-text"]).normalize_text)

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


class BatchCliTest(unittest.TestCase):
    def test_parser_defaults_off_and_flag_enables(self):
        parser = infer_batch.get_parser()
        base = ["--test_list", "input.jsonl", "--res_dir", "results"]
        self.assertFalse(parser.parse_args(base).normalize_text)
        self.assertTrue(parser.parse_args(base + ["--normalize-text"]).normalize_text)

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


class GradioTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
