#!/usr/bin/env python3
# Copyright    2026  Phuc Nguyen
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for empty/whitespace ref_text handling in voice cloning."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from omnivoice.models.omnivoice import OmniVoice, _combine_text


def _make_stub_model(sampling_rate: int = 24000) -> OmniVoice:
    """Build a minimal OmniVoice instance without loading pretrained weights."""
    model = OmniVoice.__new__(OmniVoice)
    model.sampling_rate = sampling_rate
    model._asr_pipe = object()  # already "loaded" so load_asr_model() is skipped

    class _DummyAudioTokenizer:
        config = SimpleNamespace(hop_length=480)
        device = torch.device("cpu")

        def encode(self, wav_tensor: torch.Tensor):
            num_frames = max(1, wav_tensor.shape[-1] // self.config.hop_length)
            return SimpleNamespace(
                audio_codes=torch.zeros((1, 8, num_frames), dtype=torch.long)
            )

    model.audio_tokenizer = _DummyAudioTokenizer()
    model.transcribe = MagicMock(return_value="Auto transcribed prompt")
    return model


@pytest.mark.parametrize("blank_ref_text", [None, "", "   ", "\t\n  "])
def test_create_voice_clone_prompt_treats_blank_ref_text_as_none(blank_ref_text):
    model = _make_stub_model()
    sr = model.sampling_rate
    # 2 seconds of non-silent synthetic sine audio
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    wav = torch.from_numpy(0.2 * np.sin(2 * np.pi * 440.0 * t))

    with (
        patch(
            "omnivoice.models.omnivoice.trim_long_audio",
            side_effect=lambda x, *a, **k: x,
        ) as mock_trim,
        patch(
            "omnivoice.models.omnivoice.remove_silence",
            side_effect=lambda x, *a, **k: x,
        ),
    ):
        prompt = model.create_voice_clone_prompt(
            ref_audio=(wav, sr),
            ref_text=blank_ref_text,
            preprocess_prompt=True,
        )

    mock_trim.assert_called_once()
    model.transcribe.assert_called_once()
    assert prompt.ref_text == "Auto transcribed prompt."


def test_create_voice_clone_prompt_preserves_non_empty_ref_text_without_asr():
    model = _make_stub_model()
    sr = model.sampling_rate
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    wav = torch.from_numpy(0.2 * np.sin(2 * np.pi * 440.0 * t))

    with (
        patch(
            "omnivoice.models.omnivoice.trim_long_audio",
            side_effect=lambda x, *a, **k: x,
        ) as mock_trim,
        patch(
            "omnivoice.models.omnivoice.remove_silence",
            side_effect=lambda x, *a, **k: x,
        ),
    ):
        prompt = model.create_voice_clone_prompt(
            ref_audio=(wav, sr),
            ref_text="  Custom transcript  ",
            preprocess_prompt=True,
        )

    mock_trim.assert_not_called()
    model.transcribe.assert_not_called()
    assert prompt.ref_text == "Custom transcript."


def test_combine_text_and_duration_estimation_handle_whitespace_ref_text():
    assert _combine_text("Hello world.", "   ") == "Hello world."
    assert _combine_text("Hello world.", "") == "Hello world."
    assert _combine_text("Hello world.", "  Prompt.  ") == "Prompt. Hello world."

    model = _make_stub_model()
    model.duration_estimator = MagicMock()
    model.duration_estimator.estimate_duration.return_value = 50
    tokens = model._estimate_target_tokens(
        text="Hello world.", ref_text="   ", num_ref_audio_tokens=100, speed=1.0
    )
    assert tokens == 50
    # Must fall back to default placeholder ref_text when whitespace-only
    args = model.duration_estimator.estimate_duration.call_args[0]
    assert args[1] == "Nice to meet you."
