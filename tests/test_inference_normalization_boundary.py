"""Normalization policy and tokenizer-boundary data-flow tests."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.text import normalize_for_inference
from omnivoice.utils.vietnamese_normalization.adapters import VietNormalizerAdapter


class PolicyTest(unittest.TestCase):
    def test_target_reference_instruction_policy(self):
        with patch("omnivoice.utils.text.normalize_text", return_value="TARGET") as call:
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
        self.assertEqual(
            normalize_for_inference("Có 2 hộp", language="vi", enabled=False),
            "Có 2 hộp",
        )
        with self.assertRaises(ValueError):
            normalize_for_inference("text", field="unknown")  # type: ignore[arg-type]


class AdapterTest(unittest.TestCase):
    class Broken:
        def normalize_numeric(self, text):
            raise ValueError("bad input")

    class ProgrammingBug:
        def normalize_numeric(self, text):
            raise AttributeError("bug")

    def test_vietnormalizer_known_failure_keeps_complete_raw_target(self):
        adapter = VietNormalizerAdapter(self.Broken())
        with self.assertLogs(
            "omnivoice.utils.vietnamese_normalization.adapters", level="WARNING"
        ):
            self.assertEqual(adapter.normalize("Giữ 007"), "Giữ 007")

    def test_vietnormalizer_does_not_hide_programming_errors(self):
        adapter = VietNormalizerAdapter(self.ProgrammingBug())
        with self.assertRaises(AttributeError):
            adapter.normalize("Giữ 007")

    def test_adapter_calls_numeric_mode_once(self):
        class Recording:
            def __init__(self):
                self.calls = []

            def normalize_numeric(self, text):
                self.calls.append(text)
                return "Có hai hộp"

        implementation = Recording()
        adapter = VietNormalizerAdapter(implementation)
        self.assertEqual(adapter.normalize("Có 2 hộp"), "Có hai hộp")
        self.assertEqual(implementation.calls, ["Có 2 hộp"])

    def test_upstream_only_api_is_rejected_without_custom_fallback(self):
        class UpstreamOnly:
            def normalize(self, text):
                return "upstream"

        with self.assertRaisesRegex(RuntimeError, "does not provide normalize_numeric"):
            VietNormalizerAdapter(UpstreamOnly())


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
        model.audio_tokenizer = SimpleNamespace(
            config=SimpleNamespace(frame_rate=12.5)
        )
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

    def test_real_vietnormalizer_runs_once_and_only_changes_target(self):
        import omnivoice.utils.text as text_utils
        from vietnormalizer import VietnameseNormalizer

        class RecordingBackend:
            def __init__(self):
                self.backend = VietnameseNormalizer(enable_transliteration=False)
                self.calls = []

            def normalize_numeric(self, text):
                self.calls.append(text)
                return self.backend.normalize_numeric(text)

        recording = RecordingBackend()
        previous = text_utils._VI_NORMALIZER
        text_utils._VI_NORMALIZER = VietNormalizerAdapter(recording)
        model = self.make_model()
        prompt = SimpleNamespace(
            ref_text="Mẫu có 1 hộp",
            ref_audio_tokens=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            ref_rms=None,
        )
        try:
            with patch.object(model, "_estimate_target_tokens", return_value=8):
                task = model._preprocess_all(
                    text="Hôm nay tôi nhận được 105 đơn hàng.",
                    language="vi",
                    voice_clone_prompt=prompt,
                    instruct="female",
                    normalize_text=True,
                )
                model._prepare_inference_inputs(
                    task.texts[0],
                    task.target_lens[0],
                    task.ref_texts[0],
                    task.ref_audio_tokens[0],
                    task.langs[0],
                    task.instructs[0],
                )
        finally:
            text_utils._VI_NORMALIZER = previous

        self.assertEqual(recording.calls, ["Hôm nay tôi nhận được 105 đơn hàng."])
        self.assertEqual(
            task.texts,
            ["Hôm nay tôi nhận được một trăm lẻ năm đơn hàng."],
        )
        self.assertEqual(task.ref_texts, ["Mẫu có 1 hộp"])
        self.assertEqual(task.instructs, ["female"])
        tokenized = " ".join(model.text_tokenizer.calls)
        self.assertIn(
            "<|text_start|>Mẫu có 1 hộp Hôm nay tôi nhận được một trăm lẻ năm "
            "đơn hàng.<|text_end|>",
            tokenized,
        )
        self.assertNotIn("105 đơn hàng", tokenized)


if __name__ == "__main__":
    unittest.main()
