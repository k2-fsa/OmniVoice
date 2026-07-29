"""Normalization policy and tokenizer-boundary data-flow tests."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.text import normalize_for_inference, normalize_text


class PolicyTest(unittest.TestCase):
    def test_target_reference_instruction_policy(self):
        with patch(
            "omnivoice.utils.text.normalize_text", return_value="TARGET"
        ) as call:
            self.assertEqual(
                normalize_for_inference("Có 2 hộp", language="vi", field="target"),
                "TARGET",
            )
            self.assertEqual(
                normalize_for_inference("Có 2 hộp", language="vi", field="reference"),
                "Có 2 hộp",
            )
            self.assertEqual(
                normalize_for_inference("pitch 2", language="vi", field="instruction"),
                "pitch 2",
            )
            self.assertEqual(call.call_count, 1)

    def test_disabled_and_invalid_policy(self):
        with patch("omnivoice.utils.text._get_bamibert_detector") as detector:
            self.assertEqual(
                normalize_for_inference("Có 2 hộp", language="vi", enabled=False),
                "Có 2 hộp",
            )
        detector.assert_not_called()
        with self.assertRaises(ValueError):
            normalize_for_inference("text", field="unknown")  # type: ignore[arg-type]

    def test_english_and_chinese_routes_remain_available(self):
        class Stub:
            def __init__(self, prefix):
                self.prefix = prefix

            def normalize(self, text):
                return self.prefix + text

        with patch(
            "omnivoice.utils.text._get_en_normalizer",
            return_value=Stub("EN:"),
        ):
            self.assertEqual(normalize_text("  12 [tag] ", "en"), "  EN:12 [tag] ")
        with patch(
            "omnivoice.utils.text._get_zh_normalizer",
            return_value=Stub("ZH:"),
        ):
            self.assertEqual(normalize_text("数字 12", "zh"), "ZH:数字 12")


class InferenceDataFlowTest(unittest.TestCase):
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, text, **kwargs):
            self.calls.append(text)
            input_ids = (
                torch.tensor([[1, 2]], dtype=torch.long)
                if kwargs.get("return_tensors")
                else [1, 2]
            )
            return SimpleNamespace(input_ids=input_ids)

    def make_model(self):
        model = OmniVoice.__new__(OmniVoice)
        torch.nn.Module.__init__(model)
        model.register_parameter("_test_parameter", torch.nn.Parameter(torch.zeros(1)))
        model.text_tokenizer = self.Tokenizer()
        model.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=12.5))
        model.config = SimpleNamespace(num_audio_codebook=2, audio_mask_id=99)
        return model

    def test_normalized_target_reaches_tokenizer_exactly_once(self):
        model = self.make_model()
        with (
            patch(
                "omnivoice.models.omnivoice.normalize_for_inference",
                return_value="Có hai hộp",
            ) as normalize,
            patch.object(model, "_estimate_target_tokens", return_value=8),
        ):
            task = model._preprocess_all(
                text="Có 2 hộp",
                language="vi",
                normalize_text=True,
            )
            self.assertEqual(task.texts, ["Có hai hộp"])
            normalize.assert_called_once_with(
                "Có 2 hộp", language="vi", enabled=True, field="target"
            )
            model._prepare_inference_inputs(
                task.texts[0],
                task.target_lens[0],
                task.ref_texts[0],
                task.ref_audio_tokens[0],
                task.langs[0],
                task.instructs[0],
            )
        tokenized = " ".join(model.text_tokenizer.calls)
        self.assertIn("<|text_start|>Có hai hộp<|text_end|>", tokenized)
        self.assertNotIn("Có 2 hộp", tokenized)

    def test_disabled_path_does_not_call_boundary(self):
        model = self.make_model()
        with (
            patch("omnivoice.models.omnivoice.normalize_for_inference") as normalize,
            patch.object(model, "_estimate_target_tokens", return_value=8),
        ):
            task = model._preprocess_all(
                text="Có 2 hộp",
                language="vi",
                normalize_text=False,
            )
        normalize.assert_not_called()
        self.assertEqual(task.texts, ["Có 2 hộp"])

    def test_real_candidate_pipeline_handoff_preserves_reference_and_instruction(self):
        model = self.make_model()
        prompt = SimpleNamespace(
            ref_text="Mẫu có 1 hộp",
            ref_audio_tokens=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            ref_rms=None,
        )

        def detector(text):
            start = text.index("25")
            return [
                {
                    "start": start,
                    "end": start + 2,
                    "label": "CARDINAL",
                    "text": "25",
                }
            ]

        with (
            patch(
                "omnivoice.utils.text._get_bamibert_detector",
                return_value=detector,
            ) as getter,
            patch.object(model, "_estimate_target_tokens", return_value=8),
        ):
            task = model._preprocess_all(
                text="Tôi có 25 quyển sách.",
                language="vi",
                voice_clone_prompt=prompt,
                instruct="female, low pitch",
                normalize_text=True,
            )
        getter.assert_called_once_with(None, None)
        self.assertEqual(task.texts, ["Tôi có hai mươi lăm quyển sách."])
        self.assertEqual(task.ref_texts, ["Mẫu có 1 hộp"])
        self.assertEqual(task.instructs, ["female, low pitch"])

    def test_detector_failure_cannot_handoff_partially_modified_target(self):
        model = self.make_model()

        def broken(_text):
            raise KeyError("malformed model output")

        with (
            patch(
                "omnivoice.utils.text._get_bamibert_detector",
                return_value=broken,
            ),
            patch.object(model, "_estimate_target_tokens", return_value=8),
        ):
            task = model._preprocess_all(
                text="Có 25 hộp và 30 túi.",
                language="vi",
                normalize_text=True,
            )
        self.assertEqual(task.texts, ["Có 25 hộp và 30 túi."])

    def test_generate_returns_fake_tts_output_unchanged_at_real_boundary(self):
        model = self.make_model()
        prompt = SimpleNamespace(
            ref_text="Mẫu\u00a0có 1 hộp",
            ref_audio_tokens=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            ref_rms=None,
        )
        fake_audio = object()
        observed_task = None

        def detector(text):
            start = text.index("25")
            return [
                {
                    "start": start,
                    "end": start + 2,
                    "label": "CARDINAL",
                    "text": "25",
                }
            ]

        def fake_generate(task, _config):
            nonlocal observed_task
            observed_task = task
            return [object()]

        with (
            patch(
                "omnivoice.utils.text._get_bamibert_detector",
                return_value=detector,
            ),
            patch.object(model, "_estimate_target_tokens", return_value=8),
            patch.object(model, "_generate_iterative", side_effect=fake_generate),
            patch.object(model, "_decode_and_post_process", return_value=fake_audio),
        ):
            output = model.generate(
                text="Tôi có 25 quyển sách.",
                language="vi",
                voice_clone_prompt=prompt,
                instruct="female, low pitch",
                normalize_text=True,
            )

        self.assertIs(output[0], fake_audio)
        self.assertIsNotNone(observed_task)
        self.assertEqual(observed_task.texts, ["Tôi có hai mươi lăm quyển sách."])
        self.assertEqual(observed_task.ref_texts, ["Mẫu có 1 hộp"])
        self.assertEqual(observed_task.instructs, ["female, low pitch"])


if __name__ == "__main__":
    unittest.main()
