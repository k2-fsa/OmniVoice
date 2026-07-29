import unittest
from types import SimpleNamespace

import torch

from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt


class _DurationEstimator:
    def estimate_duration(self, text, ref_text, num_ref_audio_tokens):
        return 10


class _FakeInferenceModel:
    _ensure_list = OmniVoice._ensure_list
    _estimate_target_tokens = OmniVoice._estimate_target_tokens
    _preprocess_all = OmniVoice._preprocess_all

    audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=25))
    duration_estimator = _DurationEstimator()


class UnicodeInferenceBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.model = _FakeInferenceModel()

    def test_invisible_only_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty after Unicode normalization"):
            self.model._preprocess_all("\u200b\u2060\ufeff")

    def test_target_and_saved_prompt_transcript_are_normalized(self):
        prompt = VoiceClonePrompt(
            ref_audio_tokens=torch.zeros((8, 10), dtype=torch.long),
            ref_text="Khánh\u00a0Hu\u200byền",
            ref_rms=0.1,
        )
        task = self.model._preprocess_all(
            text="Khánh\u00a0Hu\u200byền",
            voice_clone_prompt=prompt,
        )

        self.assertEqual(task.texts, ["Khánh Huyền"])
        self.assertEqual(task.ref_texts, ["Khánh Huyền"])


if __name__ == "__main__":
    unittest.main()
