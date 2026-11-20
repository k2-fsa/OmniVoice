#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
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

"""Tests for audio processing utilities."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

import omnivoice.utils.audio as audio_utils
from omnivoice.utils.audio import (
    fade_and_pad_audio,
    limit_audio_peak,
    match_edge_silence,
    remove_silence,
)


SAMPLE_RATE = 24_000
SEEK_STEP_MS = 10
QUANTIZATION_ATOL = 1.0 / 32768.0


def _samples(milliseconds: int, sample_rate: int = SAMPLE_RATE) -> int:
    return round(milliseconds * sample_rate / 1000)


def _tone(
    milliseconds: int,
    levels: tuple[float, ...] = (0.5,),
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    return np.repeat(
        np.asarray(levels, dtype=np.float32)[:, np.newaxis],
        _samples(milliseconds, sample_rate),
        axis=1,
    )


def _silence(
    milliseconds: int,
    channels: int = 1,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    return np.zeros((channels, _samples(milliseconds, sample_rate)), dtype=np.float32)


def _audio_with_internal_silence(
    silence_ms: int,
    levels: tuple[float, ...] = (0.5,),
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    return np.concatenate(
        [
            _tone(200, levels, sample_rate),
            _silence(silence_ms, len(levels), sample_rate),
            _tone(200, levels, sample_rate),
        ],
        axis=-1,
    )


def _longest_silent_run_ms(audio: np.ndarray, sample_rate: int) -> float:
    silent = np.all(np.abs(audio) <= QUANTIZATION_ATOL, axis=0)
    transitions = np.diff(np.concatenate(([False], silent, [False])).astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    longest = np.max(ends - starts, initial=0)
    return longest * 1000 / sample_rate


def _edge_silence_ms(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    silent = np.all(np.abs(audio) <= QUANTIZATION_ATOL, axis=0)
    leading = 0
    for value in silent:
        if not value:
            break
        leading += 1
    trailing = 0
    for value in silent[::-1]:
        if not value:
            break
        trailing += 1
    return leading * 1000 / sample_rate, trailing * 1000 / sample_rate


def test_remove_silence_caps_internal_gap_at_detection_threshold():
    audio = _audio_with_internal_silence(1200)

    result = remove_silence(
        audio,
        SAMPLE_RATE,
        mid_sil=300,
        lead_sil=0,
        trail_sil=0,
    )

    assert _longest_silent_run_ms(result, SAMPLE_RATE) <= 300 + SEEK_STEP_MS


@pytest.mark.parametrize("keep_mid_sil", [0, 60, 100])
def test_remove_silence_honors_total_internal_silence_limit(keep_mid_sil):
    audio = _audio_with_internal_silence(1200)

    result = remove_silence(
        audio,
        SAMPLE_RATE,
        mid_sil=300,
        lead_sil=0,
        trail_sil=0,
        keep_mid_sil=keep_mid_sil,
    )

    assert _longest_silent_run_ms(result, SAMPLE_RATE) <= keep_mid_sil + SEEK_STEP_MS


def test_keep_mid_sil_does_not_change_detection_threshold():
    audio = _audio_with_internal_silence(250)

    result = remove_silence(
        audio,
        SAMPLE_RATE,
        mid_sil=300,
        lead_sil=0,
        trail_sil=0,
        keep_mid_sil=0,
    )

    assert result.shape == audio.shape
    np.testing.assert_allclose(result, audio, atol=QUANTIZATION_ATOL, rtol=0)


def test_remove_silence_rejects_negative_internal_silence_limit():
    audio = _audio_with_internal_silence(1200)

    with pytest.raises(ValueError, match="keep_mid_sil"):
        remove_silence(
            audio,
            SAMPLE_RATE,
            mid_sil=300,
            keep_mid_sil=-1,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("mid_sil", True),
        ("mid_sil", 300.0),
        ("lead_sil", "100"),
        ("trail_sil", False),
        ("keep_mid_sil", 100.0),
    ],
)
def test_remove_silence_rejects_non_integer_millisecond_limits(argument, value):
    audio = _audio_with_internal_silence(1200)

    with pytest.raises(TypeError, match=argument):
        remove_silence(audio, SAMPLE_RATE, **{argument: value})


@pytest.mark.parametrize("argument", ["mid_sil", "lead_sil", "trail_sil"])
def test_remove_silence_rejects_negative_existing_limits(argument):
    audio = _audio_with_internal_silence(1200)
    kwargs = {argument: -1}

    with pytest.raises(ValueError, match=argument):
        remove_silence(audio, SAMPLE_RATE, **kwargs)


def test_remove_silence_trims_edges_independently_of_internal_silence():
    audio = np.concatenate([_silence(500), _tone(200), _silence(700)], axis=-1)

    result = remove_silence(
        audio,
        SAMPLE_RATE,
        mid_sil=0,
        lead_sil=100,
        trail_sil=300,
    )

    leading_ms, trailing_ms = _edge_silence_ms(result, SAMPLE_RATE)
    assert leading_ms == pytest.approx(100, abs=SEEK_STEP_MS)
    assert trailing_ms == pytest.approx(300, abs=SEEK_STEP_MS)
    assert result.shape == (1, _samples(600))
    assert result.dtype == np.float32


def test_remove_silence_preserves_multichannel_layout_and_levels():
    audio = _audio_with_internal_silence(1200, levels=(0.25, -0.5))

    result = remove_silence(
        audio,
        SAMPLE_RATE,
        mid_sil=300,
        lead_sil=0,
        trail_sil=0,
    )

    assert result.ndim == 2
    assert result.shape[0] == 2
    assert result.dtype == np.float32
    assert _longest_silent_run_ms(result, SAMPLE_RATE) <= 300 + SEEK_STEP_MS
    non_silent = np.any(np.abs(result) > QUANTIZATION_ATOL, axis=0)
    np.testing.assert_allclose(
        result[1, non_silent],
        -2 * result[0, non_silent],
        atol=2 * QUANTIZATION_ATOL,
        rtol=0,
    )


def test_remove_silence_preserves_float_peaks_without_pcm16_clipping():
    audio = _audio_with_internal_silence(1200, levels=(1.25, -1.1))

    result = remove_silence(
        audio,
        SAMPLE_RATE,
        mid_sil=300,
        lead_sil=0,
        trail_sil=0,
        keep_mid_sil=100,
    )

    assert float(np.max(result[0])) == pytest.approx(1.25)
    assert float(np.min(result[1])) == pytest.approx(-1.1)
    non_silent = np.any(np.abs(result) > QUANTIZATION_ATOL, axis=0)
    np.testing.assert_array_equal(result[0, non_silent], audio[0, 0])
    np.testing.assert_array_equal(result[1, non_silent], audio[1, 0])


@pytest.mark.parametrize("channels", [1, 2])
def test_remove_silence_preserves_nonsilent_audio_shape_and_dtype(channels):
    levels = (0.25,) if channels == 1 else (0.25, -0.5)
    audio = _tone(1000, levels)

    result = remove_silence(audio, SAMPLE_RATE)

    assert result.shape == audio.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, audio, atol=QUANTIZATION_ATOL, rtol=0)


def test_remove_silence_preserves_sample_count_below_11025_hz():
    sample_rate = 8000
    audio = _tone(1000, sample_rate=sample_rate)

    result = remove_silence(audio, sample_rate)

    assert result.shape == audio.shape
    assert result.dtype == np.float32


@pytest.mark.parametrize("samples", [0, SAMPLE_RATE])
def test_remove_silence_preserves_channels_for_empty_or_silent_audio(samples):
    audio = np.zeros((2, samples), dtype=np.float32)

    result = remove_silence(audio, SAMPLE_RATE)

    assert result.shape == (2, 0)
    assert result.dtype == np.float32


def test_match_edge_silence_replaces_both_edges_with_exact_targets():
    audio = np.concatenate([_silence(500), _tone(200), _silence(700)], axis=-1)

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=250,
        target_trail_silence_ms=100,
    )

    leading_ms, trailing_ms = _edge_silence_ms(result, SAMPLE_RATE)
    assert leading_ms == pytest.approx(250, abs=1 / SAMPLE_RATE)
    assert trailing_ms == pytest.approx(100, abs=1 / SAMPLE_RATE)
    assert result.shape == (1, _samples(550))
    np.testing.assert_array_equal(
        result[..., _samples(250) : _samples(450)],
        _tone(200),
    )


@pytest.mark.parametrize(
    ("source_lead_ms", "source_trail_ms"),
    [(501, 701), (503, 703), (505, 705), (509, 709)],
)
def test_match_edge_silence_is_sample_accurate_for_non_ten_millisecond_edges(
    source_lead_ms,
    source_trail_ms,
):
    tone = _tone(200)
    audio = np.concatenate(
        [_silence(source_lead_ms), tone, _silence(source_trail_ms)],
        axis=-1,
    )

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=250,
        target_trail_silence_ms=100,
    )

    leading_ms, trailing_ms = _edge_silence_ms(result, SAMPLE_RATE)
    assert leading_ms == pytest.approx(250, abs=1 / SAMPLE_RATE)
    assert trailing_ms == pytest.approx(100, abs=1 / SAMPLE_RATE)
    assert result.shape == (1, _samples(550))
    np.testing.assert_array_equal(
        result[..., _samples(250) : _samples(450)],
        tone,
    )


def test_match_edge_silence_preserves_an_unspecified_edge():
    audio = np.concatenate([_silence(503), _tone(200), _silence(707)], axis=-1)

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=250,
        target_trail_silence_ms=None,
    )

    leading_ms, trailing_ms = _edge_silence_ms(result, SAMPLE_RATE)
    assert leading_ms == pytest.approx(250, abs=1 / SAMPLE_RATE)
    assert trailing_ms == pytest.approx(707, abs=1 / SAMPLE_RATE)
    assert result.shape == (1, _samples(1157))


def test_match_edge_silence_preserves_an_unspecified_leading_edge():
    audio = np.concatenate([_silence(503), _tone(200), _silence(707)], axis=-1)

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=None,
        target_trail_silence_ms=100,
    )

    leading_ms, trailing_ms = _edge_silence_ms(result, SAMPLE_RATE)
    assert leading_ms == pytest.approx(503, abs=1 / SAMPLE_RATE)
    assert trailing_ms == pytest.approx(100, abs=1 / SAMPLE_RATE)
    assert result.shape == (1, _samples(803))


def test_match_edge_silence_preserves_pcm_representable_low_level_attacks():
    low_lead = _tone(25, levels=(0.002,))
    loud = _tone(200)
    low_trail = _tone(37, levels=(-0.002,))
    core = np.concatenate([low_lead, loud, low_trail], axis=-1)
    audio = np.concatenate(
        [_silence(503), core, _silence(707)],
        axis=-1,
    )

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=250,
        target_trail_silence_ms=100,
    )

    assert result.shape == (1, _samples(612))
    np.testing.assert_array_equal(
        result[..., _samples(250) : _samples(512)],
        core,
    )


def test_match_edge_silence_zero_targets_remove_both_edges():
    audio = np.concatenate([_silence(500), _tone(200), _silence(700)], axis=-1)

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=0,
        target_trail_silence_ms=0,
    )

    np.testing.assert_array_equal(result, _tone(200))


def test_match_edge_silence_scans_chunks_without_materializing_active_indices(
    monkeypatch,
):
    audio = np.asarray(
        [[0.0, 0.0, 0.25, 0.5, 0.0, 0.0, -0.25, 0.0, 0.0]],
        dtype=np.float32,
    )

    def reject_flatnonzero(*_args, **_kwargs):
        raise AssertionError("edge scanning must not materialize all active indices")

    monkeypatch.setattr(audio_utils, "_PCM16_EDGE_SCAN_CHUNK_SAMPLES", 3)
    monkeypatch.setattr(audio_utils.np, "flatnonzero", reject_flatnonzero)

    result = match_edge_silence(
        audio,
        sampling_rate=1000,
        target_lead_silence_ms=0,
        target_trail_silence_ms=0,
    )

    assert result.shape == (1, 5)
    assert result[0].tolist() == [0.25, 0.5, 0.0, 0.0, -0.25]


def test_match_edge_silence_none_targets_return_original_array():
    audio = np.concatenate([_silence(100), _tone(200)], axis=-1)

    result = match_edge_silence(audio, SAMPLE_RATE)

    assert result is audio


def test_match_edge_silence_preserves_float_multichannel_voiced_samples():
    tone = _tone(200, levels=(1.25, -1.1))
    audio = np.concatenate(
        [_silence(300, channels=2), tone, _silence(400, channels=2)],
        axis=-1,
    )

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=125,
        target_trail_silence_ms=75,
    )

    assert result.dtype == np.float32
    assert result.shape == (2, _samples(400))
    np.testing.assert_array_equal(
        result[..., _samples(125) : _samples(325)],
        tone,
    )


def test_match_edge_silence_honors_targets_below_11025_hz():
    sample_rate = 8000
    audio = np.concatenate(
        [
            _silence(300, sample_rate=sample_rate),
            _tone(200, sample_rate=sample_rate),
            _silence(400, sample_rate=sample_rate),
        ],
        axis=-1,
    )

    result = match_edge_silence(
        audio,
        sample_rate,
        target_lead_silence_ms=125,
        target_trail_silence_ms=75,
    )

    assert result.shape == (1, _samples(400, sample_rate))


@pytest.mark.parametrize("samples", [0, SAMPLE_RATE])
def test_match_edge_silence_keeps_empty_or_silent_audio_invalid(samples):
    audio = np.zeros((2, samples), dtype=np.float32)

    result = match_edge_silence(
        audio,
        SAMPLE_RATE,
        target_lead_silence_ms=250,
        target_trail_silence_ms=100,
    )

    assert result.shape == (2, 0)
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    ("argument", "value", "error_type"),
    [
        ("target_lead_silence_ms", -1, ValueError),
        ("target_lead_silence_ms", 1.5, TypeError),
        ("target_lead_silence_ms", True, TypeError),
        ("target_trail_silence_ms", "100", TypeError),
        ("target_trail_silence_ms", -1, ValueError),
    ],
)
def test_match_edge_silence_rejects_invalid_targets(
    argument,
    value,
    error_type,
):
    with pytest.raises(error_type, match=argument):
        match_edge_silence(
            _tone(100),
            SAMPLE_RATE,
            **{argument: value},
        )


@pytest.mark.parametrize(
    ("argument", "value", "error_type"),
    [
        ("pad_duration", -0.1, ValueError),
        ("pad_duration", float("nan"), ValueError),
        ("pad_duration", float("inf"), ValueError),
        ("fade_duration", -0.1, ValueError),
        ("fade_duration", True, TypeError),
        ("fade_duration", "0.1", TypeError),
    ],
)
def test_fade_and_pad_rejects_invalid_durations(argument, value, error_type):
    audio = _tone(100)

    with pytest.raises(error_type, match=argument):
        fade_and_pad_audio(audio, **{argument: value})


def test_limit_audio_peak_scales_only_when_needed():
    audio = np.asarray([[1.25, -0.625, 0.0]], dtype=np.float32)

    limited = limit_audio_peak(audio, 0.8)
    unchanged = limit_audio_peak(audio, None)

    assert float(np.max(np.abs(limited))) == pytest.approx(0.8)
    assert limited[0, 1] / limited[0, 0] == pytest.approx(-0.5)
    np.testing.assert_array_equal(unchanged, audio)


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (0, ValueError),
        (-0.1, ValueError),
        (1.01, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.9", TypeError),
    ],
)
def test_limit_audio_peak_rejects_invalid_limits(value, error_type):
    with pytest.raises(error_type, match="peak_limit"):
        limit_audio_peak(_tone(100), value)


def test_generation_config_forwards_output_silence_controls(monkeypatch):
    import omnivoice.models.omnivoice as omnivoice_module
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceGenerationConfig,
    )

    received = {}
    received_targets = {}

    def capture_remove_silence(audio, sampling_rate, **kwargs):
        received["sampling_rate"] = sampling_rate
        received.update(kwargs)
        return audio

    monkeypatch.setattr(omnivoice_module, "remove_silence", capture_remove_silence)
    monkeypatch.setattr(
        omnivoice_module,
        "fade_and_pad_audio",
        lambda audio, **kwargs: audio,
    )
    monkeypatch.setattr(
        omnivoice_module,
        "match_edge_silence",
        lambda audio, sampling_rate, **kwargs: (
            received_targets.update({"sampling_rate": sampling_rate, **kwargs}) or audio
        ),
    )

    config = OmniVoiceGenerationConfig(
        output_min_silence_ms=450,
        output_keep_silence_ms=90,
        output_lead_silence_ms=40,
        output_trail_silence_ms=60,
        output_peak_limit=None,
        output_target_lead_silence_ms=250,
        output_target_trail_silence_ms=70,
        pad_duration=0,
        fade_duration=0,
    )
    model_stub = SimpleNamespace(sampling_rate=SAMPLE_RATE)
    audio = _audio_with_internal_silence(1200)

    result = OmniVoice._post_process_audio(
        model_stub,
        audio.copy(),
        ref_rms=0.1,
        gen_config=config,
    )

    np.testing.assert_array_equal(result, audio)
    assert received == {
        "sampling_rate": SAMPLE_RATE,
        "mid_sil": 450,
        "lead_sil": 40,
        "trail_sil": 60,
        "keep_mid_sil": 90,
        "preserve_active_edges": False,
    }
    assert received_targets == {
        "sampling_rate": SAMPLE_RATE,
        "target_lead_silence_ms": 250,
        "target_trail_silence_ms": 70,
    }


def test_generation_config_edge_targets_override_generic_padding():
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceGenerationConfig,
    )

    config = OmniVoiceGenerationConfig(
        postprocess_output=False,
        pad_duration=0.1,
        fade_duration=0,
        output_target_lead_silence_ms=250,
        output_target_trail_silence_ms=75,
    )
    model_stub = SimpleNamespace(sampling_rate=SAMPLE_RATE)
    audio = np.concatenate([_silence(500), _tone(200), _silence(700)], axis=-1)

    result = OmniVoice._post_process_audio(
        model_stub,
        audio,
        ref_rms=0.1,
        gen_config=config,
    )

    leading_ms, trailing_ms = _edge_silence_ms(result, SAMPLE_RATE)
    assert leading_ms == pytest.approx(250, abs=1 / SAMPLE_RATE)
    assert trailing_ms == pytest.approx(75, abs=1 / SAMPLE_RATE)


def test_generation_config_default_preserves_historical_per_side_keep_silence():
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceGenerationConfig,
    )

    config = OmniVoiceGenerationConfig(pad_duration=0, fade_duration=0)
    model_stub = SimpleNamespace(sampling_rate=SAMPLE_RATE)
    tone = _tone(1000)
    audio = np.concatenate((tone, _silence(2000), tone), axis=-1)

    result = OmniVoice._post_process_audio(
        model_stub,
        audio,
        ref_rms=0.1,
        gen_config=config,
    )

    assert result.shape == (1, 72_000)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[..., :24_000], tone)
    np.testing.assert_array_equal(result[..., -24_000:], tone)


def test_generation_config_explicit_keep_silence_remains_a_total_gap_limit():
    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig

    config = OmniVoiceGenerationConfig(
        output_min_silence_ms=500,
        output_keep_silence_ms=500,
        pad_duration=0,
        fade_duration=0,
    )
    model_stub = SimpleNamespace(sampling_rate=SAMPLE_RATE)
    audio = np.concatenate((_tone(1000), _silence(2000), _tone(1000)), axis=-1)

    result = OmniVoice._post_process_audio(
        model_stub,
        audio,
        ref_rms=0.1,
        gen_config=config,
    )

    assert _longest_silent_run_ms(result, SAMPLE_RATE) <= 500 + SEEK_STEP_MS


@pytest.mark.parametrize("samples", [0, SAMPLE_RATE])
def test_generation_config_preserves_empty_audio_without_peak_reduction(samples):
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceGenerationConfig,
    )

    config = OmniVoiceGenerationConfig(pad_duration=0, fade_duration=0)
    model_stub = SimpleNamespace(sampling_rate=SAMPLE_RATE)
    audio = np.zeros((1, samples), dtype=np.float32)

    result = OmniVoice._post_process_audio(
        model_stub,
        audio,
        ref_rms=None,
        gen_config=config,
    )

    assert result.shape == (1, 0)
    assert result.dtype == np.float32


def test_generation_config_accepts_controls_from_keyword_dictionary():
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    config = OmniVoiceGenerationConfig.from_dict(
        {
            "output_min_silence_ms": 420,
            "output_keep_silence_ms": 80,
            "output_lead_silence_ms": 30,
            "output_trail_silence_ms": 50,
            "output_peak_limit": 0.95,
            "output_target_lead_silence_ms": 250,
            "output_target_trail_silence_ms": 75,
            "output_mode": "raw_codec",
            "unknown_key": "ignored",
        }
    )

    assert config.output_min_silence_ms == 420
    assert config.output_keep_silence_ms == 80
    assert config.output_lead_silence_ms == 30
    assert config.output_trail_silence_ms == 50
    assert config.output_peak_limit == 0.95
    assert config.output_target_lead_silence_ms == 250
    assert config.output_target_trail_silence_ms == 75
    assert config.output_mode == "raw_codec"
    assert not hasattr(config, "unknown_key")


@pytest.mark.parametrize(
    ("argument", "value", "error_type"),
    [
        ("output_min_silence_ms", -1, ValueError),
        ("output_keep_silence_ms", 10.5, TypeError),
        ("output_lead_silence_ms", True, TypeError),
        ("output_trail_silence_ms", "100", TypeError),
        ("pad_duration", float("nan"), ValueError),
        ("fade_duration", -0.1, ValueError),
        ("output_peak_limit", 0, ValueError),
        ("output_peak_limit", 1.1, ValueError),
        ("output_peak_limit", True, TypeError),
        ("output_target_lead_silence_ms", -1, ValueError),
        ("output_target_lead_silence_ms", 1.5, TypeError),
        ("output_target_trail_silence_ms", True, TypeError),
        ("output_target_trail_silence_ms", "100", TypeError),
        ("output_mode", "raw", ValueError),
        ("output_mode", None, TypeError),
    ],
)
def test_generation_config_rejects_invalid_postprocessing_values(
    argument, value, error_type
):
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    with pytest.raises(error_type, match=argument):
        OmniVoiceGenerationConfig(**{argument: value})


def test_prompt_preprocessing_preserves_legacy_internal_silence_limit(monkeypatch):
    import torch

    import omnivoice.models.omnivoice as omnivoice_module
    from omnivoice.models.omnivoice import OmniVoice

    received = {}

    def capture_remove_silence(audio, sampling_rate, **kwargs):
        received["sampling_rate"] = sampling_rate
        received.update(kwargs)
        return audio

    class AudioTokenizerStub:
        config = SimpleNamespace(hop_length=1)
        device = torch.device("cpu")

        @staticmethod
        def encode(audio):
            return SimpleNamespace(audio_codes=torch.zeros((1, 1, 1), dtype=torch.long))

    monkeypatch.setattr(omnivoice_module, "remove_silence", capture_remove_silence)
    model_stub = SimpleNamespace(
        sampling_rate=SAMPLE_RATE,
        audio_tokenizer=AudioTokenizerStub(),
    )

    OmniVoice.create_voice_clone_prompt(
        model_stub,
        (torch.ones(SAMPLE_RATE), SAMPLE_RATE),
        ref_text="Reference text.",
    )

    assert received == {
        "sampling_rate": SAMPLE_RATE,
        "mid_sil": 200,
        "lead_sil": 100,
        "trail_sil": 200,
        "keep_mid_sil": 400,
    }


def test_generation_config_appends_new_fields_for_positional_compatibility():
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    names = [field.name for field in fields(OmniVoiceGenerationConfig)]

    assert names[-9:] == [
        "output_min_silence_ms",
        "output_keep_silence_ms",
        "output_lead_silence_ms",
        "output_trail_silence_ms",
        "output_peak_limit",
        "output_target_lead_silence_ms",
        "output_target_trail_silence_ms",
        "output_mode",
        "output_preserve_active_edges",
    ]
