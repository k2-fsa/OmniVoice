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

"""Tests for raw codec output and opt-in generation telemetry."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Barrier, Event
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

import omnivoice.models.omnivoice as omnivoice_module
from omnivoice import (
    OmniVoiceCudaTelemetry,
    OmniVoiceFinalDurationTelemetry,
    OmniVoiceFramingObservation,
    OmniVoiceGenerationTelemetry,
)
from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoice,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
)


class AudioTokenizerStub:
    device = torch.device("cpu")
    config = SimpleNamespace(frame_rate=25)

    def __init__(self, waveforms):
        self.waveforms = waveforms

    def decode(self, tokens):
        token_id = int(tokens.flatten()[0])
        waveform = torch.from_numpy(self.waveforms[token_id].copy())
        return SimpleNamespace(audio_values=[waveform])


class DecodeHarness:
    sampling_rate = 24_000
    _decode_audio_tokens = OmniVoice._decode_audio_tokens
    _decode_and_post_process = OmniVoice._decode_and_post_process
    _post_process_audio = OmniVoice._post_process_audio

    def __init__(self, waveforms):
        self.audio_tokenizer = AudioTokenizerStub(waveforms)


class GenerateHarness(DecodeHarness):
    device = torch.device("cpu")
    text_tokenizer = object()

    def __init__(self, waveform):
        super().__init__({1: waveform})
        self.evaluated = False

    def eval(self):
        self.evaluated = True

    def _preprocess_all(self, **kwargs):
        return GenerationTask(
            batch_size=1,
            texts=[kwargs["text"]],
            target_lens=[1],
            langs=[None],
            instructs=[None],
            ref_texts=[None],
            ref_audio_tokens=[None],
            ref_rms=[0.1],
        )

    @staticmethod
    def _generate_iterative(task, gen_config):
        assert task.batch_size == 1
        return [torch.tensor([[1]], dtype=torch.long)]

    @staticmethod
    def _generate_chunked(task, gen_config):
        raise AssertionError("The short test input must not use chunked generation")


def _call_generate(model, **kwargs):
    generate = getattr(OmniVoice.generate, "__wrapped__", OmniVoice.generate)
    return generate(model, **kwargs)


def test_processed_default_preserves_decode_crossfade_and_postprocessing(monkeypatch):
    first = np.array([[0.1, 0.2]], dtype=np.float32)
    second = np.array([[0.3, 0.4]], dtype=np.float32)
    model = DecodeHarness({1: first, 2: second})
    calls = []

    def crossfade_stub(chunks, sample_rate):
        calls.append(("crossfade", chunks, sample_rate))
        return np.concatenate(chunks, axis=-1)

    def postprocess_stub(self, audio, ref_rms, gen_config):
        calls.append(("postprocess", audio.copy(), ref_rms, gen_config.output_mode))
        return audio + np.float32(0.25)

    monkeypatch.setattr(omnivoice_module, "cross_fade_chunks", crossfade_stub)
    model._post_process_audio = MethodType(postprocess_stub, model)

    config = OmniVoiceGenerationConfig()
    result = model._decode_and_post_process(
        [torch.tensor([[1]]), torch.tensor([[2]])],
        rms=0.07,
        gen_config=config,
    )

    assert config.output_mode == "processed"
    np.testing.assert_array_equal(
        result,
        np.array([0.35, 0.45, 0.55, 0.65], dtype=np.float32),
    )
    assert calls[0][0] == "crossfade"
    assert calls[0][2] == model.sampling_rate
    assert calls[1][0] == "postprocess"
    assert calls[1][2:] == (0.07, "processed")


def test_historical_decode_override_signature_works_for_processed_and_raw(monkeypatch):
    first = np.array([[0.1, 0.2]], dtype=np.float32)
    second = np.array([[0.3, 0.4]], dtype=np.float32)

    class HistoricalDecodeHarness(DecodeHarness):
        def _decode_audio_tokens(self, tokens):
            assert isinstance(tokens, list)
            return [first, second]

    model = HistoricalDecodeHarness({})

    def concatenate_chunks(chunks, sample_rate):
        assert sample_rate == 24_000
        return np.concatenate(chunks, axis=-1)

    monkeypatch.setattr(omnivoice_module, "cross_fade_chunks", concatenate_chunks)
    model._post_process_audio = MethodType(
        lambda self, audio, ref_rms, gen_config: audio,
        model,
    )
    tokens = [torch.tensor([[1]]), torch.tensor([[2]])]

    processed = model._decode_and_post_process(
        tokens,
        rms=0.1,
        gen_config=OmniVoiceGenerationConfig(output_mode="processed"),
    )
    raw = model._decode_and_post_process(
        tokens,
        rms=0.1,
        gen_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
    )

    expected = np.concatenate([first, second], axis=-1).squeeze(0)
    np.testing.assert_array_equal(processed, expected)
    np.testing.assert_array_equal(raw, expected)


def test_raw_long_form_join_is_counted_as_codec_decode_time(monkeypatch):
    first = np.array([[0.1, 0.2]], dtype=np.float32)
    second = np.array([[0.3, 0.4]], dtype=np.float32)
    model = DecodeHarness({1: first, 2: second})
    events = []
    original_concatenate = np.concatenate

    def clock():
        events.append("clock")
        return float(len(events))

    def concatenate(chunks, axis):
        events.append("raw_join")
        return original_concatenate(chunks, axis=axis)

    monkeypatch.setattr(omnivoice_module.time, "perf_counter", clock)
    monkeypatch.setattr(omnivoice_module.np, "concatenate", concatenate)
    telemetry_state = omnivoice_module._GenerationTelemetryState.start(model)

    result = model._decode_and_post_process(
        [torch.tensor([[1]]), torch.tensor([[2]])],
        rms=0.1,
        gen_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        _telemetry_state=telemetry_state,
    )

    assert events == ["clock", "clock", "raw_join", "clock"]
    assert telemetry_state.codec_decode_seconds == 2.0
    np.testing.assert_array_equal(
        result,
        original_concatenate([first, second], axis=-1).squeeze(0),
    )


def test_decode_preserves_historical_positional_telemetry_argument():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = DecodeHarness({1: waveform})

    result = model._decode_and_post_process(
        torch.tensor([[1]]),
        0.1,
        OmniVoiceGenerationConfig(
            postprocess_output=False,
            pad_duration=0.0,
            fade_duration=0.0,
        ),
        None,
    )

    np.testing.assert_array_equal(result, waveform.squeeze(0))


@pytest.mark.parametrize("chunked", [False, True])
def test_raw_codec_bypasses_every_output_transformation(monkeypatch, chunked):
    first = np.array([[0.0, 1.2, -1.1, 0.0]], dtype=np.float32)
    second = np.array([[0.0, 0.25, 0.0]], dtype=np.float32)
    model = DecodeHarness({1: first, 2: second})

    def unexpected_call(*args, **kwargs):
        raise AssertionError("raw_codec must bypass all output post-processing")

    monkeypatch.setattr(omnivoice_module, "cross_fade_chunks", unexpected_call)
    model._post_process_audio = MethodType(unexpected_call, model)
    config = OmniVoiceGenerationConfig(
        output_mode="raw_codec",
        postprocess_output=True,
        output_peak_limit=0.5,
        output_target_lead_silence_ms=0,
        output_target_trail_silence_ms=0,
        pad_duration=1.0,
        fade_duration=1.0,
    )
    tokens = (
        [torch.tensor([[1]]), torch.tensor([[2]])] if chunked else torch.tensor([[1]])
    )

    result = model._decode_and_post_process(tokens, rms=0.01, gen_config=config)

    expected = np.concatenate([first, second], axis=-1) if chunked else first
    np.testing.assert_array_equal(result, expected.squeeze(0))


def test_processed_framing_runs_after_postprocessing(monkeypatch):
    waveform = np.array([[0.25, -0.5]], dtype=np.float32)
    model = DecodeHarness({1: waveform})

    def postprocess_stub(self, audio, ref_rms, gen_config):
        del self, ref_rms, gen_config
        return np.concatenate((audio, np.zeros((1, 2), dtype=np.float32)), axis=-1)

    model._post_process_audio = MethodType(postprocess_stub, model)
    target = omnivoice_module._FinalDurationTarget(
        samples=2,
        source="integer_samples",
        requested_seconds=None,
    )

    result = model._decode_and_post_process(
        torch.tensor([[1]]),
        rms=0.1,
        gen_config=OmniVoiceGenerationConfig(),
        final_duration_target=target,
    )

    np.testing.assert_array_equal(result, waveform.squeeze(0))


def test_framing_observer_receives_read_only_padding_views_and_exact_mapping():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = DecodeHarness({1: waveform})
    target = omnivoice_module._FinalDurationTarget(
        samples=5,
        source="integer_samples",
        requested_seconds=None,
    )
    observations = []

    result = model._decode_and_post_process(
        torch.tensor([[1]]),
        rms=0.1,
        gen_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_target=target,
        item_index=4,
        framing_observer=observations.append,
    )

    assert len(observations) == 1
    observation = observations[0]
    assert isinstance(observation, OmniVoiceFramingObservation)
    assert observation.framing.item_index == 4
    assert observation.framing.operation == "padded"
    assert observation.source_waveform.shape == (1, 3)
    assert observation.final_waveform.shape == (1, 5)
    assert not observation.source_waveform.flags.writeable
    assert not observation.final_waveform.flags.writeable
    assert (
        observation.retained_source_start_sample,
        observation.retained_source_end_sample,
        observation.retained_final_start_sample,
        observation.retained_final_end_sample,
    ) == (0, 3, 0, 3)
    np.testing.assert_array_equal(
        observation.source_waveform[:, 0:3],
        observation.final_waveform[:, 0:3],
    )
    np.testing.assert_array_equal(observation.final_waveform[:, 3:], 0.0)
    np.testing.assert_array_equal(result, observation.final_waveform.squeeze(0))
    with pytest.raises(ValueError, match="read-only"):
        observation.final_waveform[0, 0] = 0.0
    for waveform_snapshot in (
        observation.source_waveform,
        observation.final_waveform,
    ):
        owner = waveform_snapshot
        while isinstance(owner, np.ndarray):
            assert not owner.flags.writeable
            with pytest.raises(ValueError, match="WRITEABLE"):
                owner.setflags(write=True)
            if owner.size:
                with pytest.raises(ValueError, match="read-only"):
                    owner.flat[0] = 0.0
            owner = owner.base
        assert isinstance(owner, bytes)


def test_framing_observer_reports_zero_only_trim_and_preserved_core():
    waveform = np.array([[0.0, 0.0, 0.1, -0.2, 0.0, 0.0]], dtype=np.float32)
    model = DecodeHarness({1: waveform})
    target = omnivoice_module._FinalDurationTarget(
        samples=3,
        source="integer_samples",
        requested_seconds=None,
    )
    observations = []

    result = model._decode_and_post_process(
        torch.tensor([[1]]),
        rms=0.1,
        gen_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_target=target,
        framing_observer=observations.append,
    )

    observation = observations[0]
    assert observation.framing.operation == "trimmed"
    assert observation.framing.trimmed_leading_samples == 1
    assert observation.framing.trimmed_trailing_samples == 2
    assert (
        observation.retained_source_start_sample,
        observation.retained_source_end_sample,
        observation.retained_final_start_sample,
        observation.retained_final_end_sample,
    ) == (1, 4, 0, 3)
    np.testing.assert_array_equal(
        observation.source_waveform[:, 1:4],
        observation.final_waveform[:, 0:3],
    )
    np.testing.assert_array_equal(result, waveform[:, 1:4].squeeze(0))


def test_framing_observer_accepts_signed_zero_edges_as_digital_silence():
    waveform = np.array([[-0.0, 0.25, -0.0]], dtype=np.float32)
    model = GenerateHarness(waveform)
    observations = []

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=1,
        framing_observer=observations.append,
    )

    np.testing.assert_array_equal(result[0], np.array([0.25], dtype=np.float32))
    assert len(observations) == 1
    assert observations[0].framing.trimmed_leading_samples == 1
    assert observations[0].framing.trimmed_trailing_samples == 1


def test_raw_framing_ignores_processed_edge_targets_and_only_adds_zeros():
    waveform = np.array([[0.25, -0.5]], dtype=np.float32)
    model = DecodeHarness({1: waveform})
    target = omnivoice_module._FinalDurationTarget(
        samples=4,
        source="integer_samples",
        requested_seconds=None,
    )
    config = OmniVoiceGenerationConfig(
        output_mode="raw_codec",
        output_target_lead_silence_ms=100,
        output_target_trail_silence_ms=100,
    )

    result = model._decode_and_post_process(
        torch.tensor([[1]]),
        rms=0.1,
        gen_config=config,
        final_duration_target=target,
    )

    np.testing.assert_array_equal(result[:2], waveform.squeeze(0))
    np.testing.assert_array_equal(result[2:], np.zeros(2, dtype=np.float32))


def test_raw_framing_fails_closed_before_cutting_active_samples():
    waveform = np.array([[0.25, -0.5]], dtype=np.float32)
    model = DecodeHarness({1: waveform})
    target = omnivoice_module._FinalDurationTarget(
        samples=1,
        source="integer_samples",
        requested_seconds=None,
    )

    with pytest.raises(ValueError, match="active samples is forbidden"):
        model._decode_and_post_process(
            torch.tensor([[1]]),
            rms=0.1,
            gen_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_target=target,
        )


@pytest.mark.parametrize("output_mode", ["processed", "raw_codec"])
def test_long_form_framing_runs_after_the_mode_specific_chunk_join(
    monkeypatch, output_mode
):
    first = np.array([[0.25, 0.0]], dtype=np.float32)
    second = np.array([[-0.5, 0.0]], dtype=np.float32)
    model = DecodeHarness({1: first, 2: second})
    monkeypatch.setattr(
        omnivoice_module,
        "cross_fade_chunks",
        lambda chunks, sample_rate: np.concatenate(chunks, axis=-1),
    )
    model._post_process_audio = MethodType(
        lambda self, audio, ref_rms, gen_config: audio,
        model,
    )
    target = omnivoice_module._FinalDurationTarget(
        samples=3,
        source="integer_samples",
        requested_seconds=None,
    )

    result = model._decode_and_post_process(
        [torch.tensor([[1]]), torch.tensor([[2]])],
        rms=0.1,
        gen_config=OmniVoiceGenerationConfig(output_mode=output_mode),
        final_duration_target=target,
    )

    np.testing.assert_array_equal(
        result,
        np.array([0.25, 0.0, -0.5], dtype=np.float32),
    )


def test_processed_framing_never_consumes_an_exact_trailing_edge_target():
    waveform = np.array([[0.25, -0.5, 0.0, 0.0]], dtype=np.float32)
    model = DecodeHarness({1: waveform})
    model._post_process_audio = MethodType(
        lambda self, audio, ref_rms, gen_config: audio,
        model,
    )
    target = omnivoice_module._FinalDurationTarget(
        samples=3,
        source="integer_samples",
        requested_seconds=None,
    )

    with pytest.raises(ValueError, match="only 0 removable"):
        model._decode_and_post_process(
            torch.tensor([[1]]),
            rms=0.1,
            gen_config=OmniVoiceGenerationConfig(output_target_trail_silence_ms=100),
            final_duration_target=target,
        )


def test_generate_without_telemetry_avoids_timers_and_cuda_queries(monkeypatch):
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = GenerateHarness(waveform)
    observations = []

    def unexpected_call(*args, **kwargs):
        raise AssertionError("telemetry-off generation must not collect metrics")

    monkeypatch.setattr(omnivoice_module.time, "perf_counter", unexpected_call)
    monkeypatch.setattr(torch.cuda, "is_available", unexpected_call)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_call)
    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected_call)
    monkeypatch.setattr(torch.mps, "synchronize", unexpected_call)
    monkeypatch.setattr(torch.xpu, "is_available", unexpected_call)
    monkeypatch.setattr(torch.xpu, "synchronize", unexpected_call)

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=3,
        framing_observer=observations.append,
    )

    assert model.evaluated
    assert len(observations) == 1
    np.testing.assert_array_equal(result[0], waveform.squeeze(0))


def test_generate_default_preserves_historical_private_method_signatures():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class LegacyPrivateSignatureHarness(GenerateHarness):
        def _preprocess_all(
            self,
            text,
            language=None,
            ref_text=None,
            ref_audio=None,
            voice_clone_prompt=None,
            instruct=None,
            preprocess_prompt=True,
            speed=None,
            duration=None,
            normalize_text=False,
            _telemetry_state=None,
        ):
            class LegacyTask:
                batch_size = 1
                ref_rms = [0.1]

                @staticmethod
                def get_indices(gen_config, frame_rate):
                    return [0], []

                def slice_task(self, indices):
                    return self

            return LegacyTask()

        def _decode_and_post_process(
            self,
            tokens,
            rms,
            gen_config,
            _telemetry_state=None,
        ):
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                _telemetry_state,
            )

    model = LegacyPrivateSignatureHarness(waveform)

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
    )

    np.testing.assert_array_equal(result[0], waveform.squeeze(0))


def test_generate_consumes_mixed_physical_duration_generator_only_once():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    decode_kwargs = []
    observations = []

    class LegacyBatchPreprocessorHarness(GenerateHarness):
        def _preprocess_all(
            self,
            text,
            language=None,
            ref_text=None,
            ref_audio=None,
            voice_clone_prompt=None,
            instruct=None,
            preprocess_prompt=True,
            speed=None,
            duration=None,
            normalize_text=False,
            _telemetry_state=None,
        ):
            assert text == ["first", "second", "third"]
            return GenerationTask(
                batch_size=3,
                texts=text,
                target_lens=[1, 1, 1],
                langs=[None, None, None],
                instructs=[None, None, None],
                ref_texts=[None, None, None],
                ref_audio_tokens=[None, None, None],
                ref_rms=[0.1, 0.1, 0.1],
            )

        @staticmethod
        def _generate_iterative(task, gen_config):
            assert task.batch_size == 3
            return [torch.tensor([[1]], dtype=torch.long) for _ in range(3)]

        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            decode_kwargs.append(kwargs.copy())
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )

    model = LegacyBatchPreprocessorHarness(waveform)
    one_shot_targets = (value for value in [3, None, 3])

    result = _call_generate(
        model,
        text=["first", "second", "third"],
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=one_shot_targets,
        framing_observer=observations.append,
    )

    assert [audio.shape[0] for audio in result] == [3, 3, 3]
    assert "final_duration_target" in decode_kwargs[0]
    assert decode_kwargs[1] == {}
    assert "final_duration_target" in decode_kwargs[2]
    assert [observation.framing.item_index for observation in observations] == [0, 2]


@pytest.mark.parametrize(
    ("output_mode", "long_form"),
    [("processed", False), ("raw_codec", True), ("processed", True)],
)
def test_framing_observer_covers_processed_raw_and_long_form_paths(
    output_mode,
    long_form,
):
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class PathHarness(GenerateHarness):
        def _preprocess_all(self, **kwargs):
            task = super()._preprocess_all(**kwargs)
            task.target_lens = [26 if long_form else 1]
            return task

        def _generate_iterative(self, task, gen_config):
            assert not long_form
            return [torch.tensor([[1]], dtype=torch.long)]

        def _generate_chunked(self, task, gen_config):
            assert long_form
            return [
                [
                    torch.tensor([[1]], dtype=torch.long),
                    torch.tensor([[1]], dtype=torch.long),
                ]
            ]

        def _post_process_audio(self, audio, ref_rms, gen_config):
            assert output_mode == "processed"
            return audio

    model = PathHarness(waveform)
    observations = []
    target_samples = 5 if not long_form else 8 if output_mode == "raw_codec" else 2_408

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(
            output_mode=output_mode,
            audio_chunk_threshold=1,
        ),
        final_duration_samples=target_samples,
        framing_observer=observations.append,
    )

    assert result[0].shape == (target_samples,)
    assert len(observations) == 1
    assert observations[0].framing.operation == "padded"
    np.testing.assert_array_equal(result[0], observations[0].final_waveform[0])


def test_processed_both_edge_anchors_allow_trailing_outer_container_fill():
    lead = np.zeros((1, 24), dtype=np.float32)
    core = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    trail = np.zeros((1, 48), dtype=np.float32)
    waveform = np.concatenate((lead, core, trail), axis=-1)

    class EdgeAlignedHarness(GenerateHarness):
        @staticmethod
        def _post_process_audio(audio, ref_rms, gen_config):
            return audio

    model = EdgeAlignedHarness(waveform)
    observations = []

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(
            output_mode="processed",
            output_target_lead_silence_ms=1,
            output_target_trail_silence_ms=2,
        ),
        final_duration_samples=waveform.shape[-1] + 7,
        framing_observer=observations.append,
    )

    np.testing.assert_array_equal(result[0][: waveform.shape[-1]], waveform[0])
    np.testing.assert_array_equal(result[0][waveform.shape[-1] :], 0.0)
    evidence = observations[0].framing
    assert evidence.protected_leading_edge
    assert evidence.protected_trailing_edge
    assert evidence.padded_leading_samples == 0
    assert evidence.padded_trailing_samples == 7
    assert observations[0].retained_final_end_sample == waveform.shape[-1]


def test_processed_trailing_anchor_underfill_still_appends_without_moving_onset():
    core = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    trail = np.zeros((1, 48), dtype=np.float32)
    waveform = np.concatenate((core, trail), axis=-1)

    class EdgeAlignedHarness(GenerateHarness):
        @staticmethod
        def _post_process_audio(audio, ref_rms, gen_config):
            return audio

    model = EdgeAlignedHarness(waveform)
    observations = []

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(
            output_mode="processed",
            output_target_trail_silence_ms=2,
        ),
        final_duration_samples=waveform.shape[-1] + 7,
        framing_observer=observations.append,
    )

    np.testing.assert_array_equal(result[0][: waveform.shape[-1]], waveform[0])
    np.testing.assert_array_equal(result[0][waveform.shape[-1] :], 0.0)
    evidence = observations[0].framing
    assert evidence.protected_leading_edge is False
    assert evidence.protected_trailing_edge is True
    assert evidence.padded_leading_samples == 0
    assert evidence.padded_trailing_samples == 7
    assert observations[0].retained_final_start_sample == 0
    assert observations[0].retained_final_end_sample == waveform.shape[-1]


def test_generate_rejects_decoder_that_ignores_physical_duration_authority():
    waveform = np.array([[0.1, -0.2, 0.3, 0.4]], dtype=np.float32)

    class DroppingDecoderHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **_kwargs):
            return waveform.squeeze(0)

    model = DroppingDecoderHarness(waveform)

    with pytest.raises(RuntimeError, match="expected 3 samples, got 4"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=3,
        )


def test_generate_rejects_empty_decoder_waveform_instead_of_padding_it():
    model = GenerateHarness(np.empty((1, 0), dtype=np.float32))

    with pytest.raises(RuntimeError, match="must contain at least one sample"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=3,
        )


def test_generate_rejects_missing_framing_telemetry_before_callback():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class DroppingDecoderHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **_kwargs):
            return waveform.squeeze(0)

    model = DroppingDecoderHarness(waveform)
    records = []

    with pytest.raises(RuntimeError, match="telemetry evidence is incomplete"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=3,
            telemetry_callback=records.append,
        )

    assert records == []


def test_generate_rejects_decoder_that_ignores_requested_framing_observer():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class DroppingDecoderHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **_kwargs):
            return waveform.squeeze(0)

    model = DroppingDecoderHarness(waveform)
    observations = []

    with pytest.raises(RuntimeError, match="framing_observer evidence is incomplete"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=3,
            framing_observer=observations.append,
        )

    assert observations == []


def test_generate_rejects_decoder_that_returns_content_different_from_observer():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class AlteringDecoderHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            result = OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            ).copy()
            result[1] = np.nextafter(result[1], np.float32(0.0))
            return result

    model = AlteringDecoderHarness(waveform)
    observations = []

    with pytest.raises(RuntimeError, match="does not match the returned output"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=3,
            framing_observer=observations.append,
        )

    assert observations == []


def test_generate_rejects_observer_evidence_with_wrong_protected_edge_flag():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class AlteringEvidenceHarness(GenerateHarness):
        @staticmethod
        def _post_process_audio(audio, ref_rms, gen_config):
            return audio

        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            original_observer = kwargs["framing_observer"]

            def alter_evidence(observation):
                original_observer(
                    replace(
                        observation,
                        framing=replace(
                            observation.framing,
                            protected_leading_edge=False,
                        ),
                    )
                )

            kwargs["framing_observer"] = alter_evidence
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )

    model = AlteringEvidenceHarness(waveform)

    with pytest.raises(RuntimeError, match="does not match the authority"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(
                output_mode="processed",
                output_target_lead_silence_ms=0,
            ),
            final_duration_samples=3,
            framing_observer=lambda _observation: None,
        )


@pytest.mark.parametrize(
    ("forgery", "message"),
    [
        ("integer_dtype", "mono real-floating"),
        ("non_finite", "non-finite"),
        ("modified_core", "retained waveform content was modified"),
        ("active_padding", "non-zero trimmed or padded region"),
        ("wrong_operation", "operation metadata is inconsistent"),
        ("separate_backing", "same immutable backing snapshot"),
        ("disjoint_shared_owner", "overlapping views"),
    ],
)
def test_generate_rejects_forged_framing_observer_waveforms(forgery, message):
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class ForgingEvidenceHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            original_observer = kwargs["framing_observer"]

            def freeze(array):
                return omnivoice_module._immutable_waveform_snapshot(array)

            def replace_from_final(observation, final, source_slice=slice(0, 3)):
                shared = freeze(final)
                return replace(
                    observation,
                    source_waveform=shared[..., source_slice],
                    final_waveform=shared,
                )

            def forge(observation):
                if forgery == "integer_dtype":
                    observation = replace_from_final(
                        observation,
                        observation.final_waveform.astype(np.int16),
                    )
                elif forgery == "non_finite":
                    final = observation.final_waveform.copy()
                    final[0, 0] = np.nan
                    observation = replace_from_final(observation, final)
                elif forgery == "modified_core":
                    final = observation.final_waveform.copy()
                    observation = replace_from_final(
                        observation,
                        final,
                        source_slice=slice(1, 4),
                    )
                elif forgery == "active_padding":
                    final = observation.final_waveform.copy()
                    final[0, -1] = np.float32(0.1)
                    observation = replace_from_final(observation, final)
                elif forgery == "wrong_operation":
                    observation = replace(
                        observation,
                        framing=replace(observation.framing, operation="unchanged"),
                    )
                elif forgery == "separate_backing":
                    observation = replace(
                        observation,
                        source_waveform=freeze(observation.source_waveform.copy()),
                        final_waveform=freeze(observation.final_waveform.copy()),
                    )
                elif forgery == "disjoint_shared_owner":
                    raw = observation.source_waveform.tobytes(
                        order="C"
                    ) + observation.final_waveform.tobytes(order="C")
                    combined = np.frombuffer(
                        raw,
                        dtype=observation.source_waveform.dtype,
                    )
                    source_samples = observation.source_waveform.shape[-1]
                    observation = replace(
                        observation,
                        source_waveform=combined[:source_samples].reshape(
                            observation.source_waveform.shape
                        ),
                        final_waveform=combined[source_samples:].reshape(
                            observation.final_waveform.shape
                        ),
                    )
                else:
                    raise AssertionError(f"unknown forgery: {forgery}")
                original_observer(observation)

            kwargs["framing_observer"] = forge
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )

    model = ForgingEvidenceHarness(waveform)
    observations = []

    with pytest.raises(RuntimeError, match=message):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=5,
            framing_observer=observations.append,
        )

    assert observations == []


def test_generate_rejects_forged_all_silent_protected_edge_evidence():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class SilentEvidenceHarness(GenerateHarness):
        @staticmethod
        def _post_process_audio(audio, ref_rms, gen_config):
            return audio

        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            original_observer = kwargs["framing_observer"]

            def forge(observation):
                silent = omnivoice_module._immutable_waveform_snapshot(
                    np.zeros_like(observation.source_waveform)
                )
                original_observer(
                    replace(
                        observation,
                        source_waveform=silent,
                        final_waveform=silent,
                    )
                )

            kwargs["framing_observer"] = forge
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )

    model = SilentEvidenceHarness(waveform)

    with pytest.raises(RuntimeError, match="all-silent source waveform"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(
                output_mode="processed",
                output_target_lead_silence_ms=0,
            ),
            final_duration_samples=3,
            framing_observer=lambda _observation: None,
        )


def test_observer_and_telemetry_metadata_are_detached_from_callback_aliases():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class AliasMutationHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            collector = kwargs["framing_observer"]

            def mutate_after_collection(observation):
                collector(observation)
                object.__setattr__(observation.framing, "operation", "trimmed")
                object.__setattr__(
                    observation,
                    "retained_final_end_sample",
                    999,
                )

            kwargs["framing_observer"] = mutate_after_collection
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )

    model = AliasMutationHarness(waveform)
    callback_values = []
    telemetry = []

    def mutate_public_copy(observation):
        callback_values.append(
            (
                observation.framing.operation,
                observation.retained_final_end_sample,
            )
        )
        object.__setattr__(observation.framing, "operation", "trimmed")
        object.__setattr__(observation, "retained_final_end_sample", 999)

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=5,
        framing_observer=mutate_public_copy,
        telemetry_callback=telemetry.append,
    )

    assert result[0].shape == (5,)
    assert callback_values == [("padded", 3)]
    assert telemetry[0].final_duration_items[0].operation == "padded"
    assert telemetry[0].final_duration_items[0].final_samples == 5


def test_generate_rejects_internally_inconsistent_framing_telemetry():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class CorruptingTelemetryHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            result = OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )
            evidence = kwargs["_telemetry_state"].final_duration_items[-1]
            object.__setattr__(evidence, "operation", "trimmed")
            return result

    records = []

    with pytest.raises(RuntimeError, match="operation metadata is inconsistent"):
        _call_generate(
            CorruptingTelemetryHarness(waveform),
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=5,
            telemetry_callback=records.append,
        )

    assert records == []


def test_generate_binds_telemetry_to_canonical_observer_evidence():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class CoherentlyCorruptingTelemetryHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            result = OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )
            evidence = kwargs["_telemetry_state"].final_duration_items[-1]
            object.__setattr__(evidence, "source_samples", 2)
            object.__setattr__(evidence, "padded_trailing_samples", 3)
            return result

    records = []
    observations = []

    with pytest.raises(RuntimeError, match="canonical observer evidence"):
        _call_generate(
            CoherentlyCorruptingTelemetryHarness(waveform),
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=5,
            telemetry_callback=records.append,
            framing_observer=observations.append,
        )

    assert records == []
    assert observations == []


def test_public_framing_callback_cannot_rewrite_validated_telemetry_evidence():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class RetainingTelemetryStateHarness(GenerateHarness):
        retained_telemetry_state = None

        def _decode_and_post_process(self, tokens, rms, gen_config, **kwargs):
            self.retained_telemetry_state = kwargs["_telemetry_state"]
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                **kwargs,
            )

    model = RetainingTelemetryStateHarness(waveform)
    records = []

    def corrupt_retained_state(_observation):
        evidence = model.retained_telemetry_state.final_duration_items[0]
        object.__setattr__(evidence, "operation", "trimmed")
        object.__setattr__(evidence, "source_samples", 999)

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=5,
        framing_observer=corrupt_retained_state,
        telemetry_callback=records.append,
    )

    assert result[0].shape == (5,)
    assert records[0].final_duration_items[0].operation == "padded"
    assert records[0].final_duration_items[0].source_samples == 3

    retained = model.retained_telemetry_state.final_duration_items[0]
    object.__setattr__(retained, "final_samples", 999)
    assert records[0].final_duration_items[0].final_samples == 5


def test_framing_observer_exception_propagates_before_generation_returns():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = GenerateHarness(waveform)

    def fail_observation(_observation):
        raise RuntimeError("independent framing audit failed")

    with pytest.raises(RuntimeError, match="independent framing audit failed"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=3,
            framing_observer=fail_observation,
        )


def test_generate_validates_authoritative_output_shape_without_cli_boundary():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class InvalidShapeDecoderHarness(GenerateHarness):
        def _decode_and_post_process(self, tokens, rms, gen_config, **_kwargs):
            return waveform

    model = InvalidShapeDecoderHarness(waveform)

    with pytest.raises(RuntimeError, match="one-dimensional numpy array"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            final_duration_samples=3,
        )


def test_prompt_preparation_is_measured_only_when_built_inside_generate():
    class PromptHarness:
        device = torch.device("cpu")
        audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=25))
        _ensure_list = OmniVoice._ensure_list

        @staticmethod
        def create_voice_clone_prompt(ref_audio, ref_text, preprocess_prompt):
            assert ref_audio == "reference.wav"
            return VoiceClonePrompt(
                ref_audio_tokens=torch.ones((1, 1), dtype=torch.long),
                ref_text=ref_text,
                ref_rms=0.1,
            )

        @staticmethod
        def _estimate_target_tokens(*args, **kwargs):
            return 1

    model = PromptHarness()
    built_state = omnivoice_module._GenerationTelemetryState.start(model)

    task = OmniVoice._preprocess_all(
        model,
        text="target",
        ref_text="reference",
        ref_audio="reference.wav",
        _telemetry_state=built_state,
    )

    assert task.ref_texts == ["reference"]
    assert built_state.prompt_preparation_seconds is not None
    assert built_state.prompt_preparation_seconds >= 0

    reused_state = omnivoice_module._GenerationTelemetryState.start(model)
    OmniVoice._preprocess_all(
        model,
        text="target",
        voice_clone_prompt=VoiceClonePrompt(
            ref_audio_tokens=torch.ones((1, 1), dtype=torch.long),
            ref_text="reference",
            ref_rms=0.1,
        ),
        _telemetry_state=reused_state,
    )

    assert reused_state.prompt_preparation_seconds is None


def test_generate_emits_one_immutable_telemetry_record():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = GenerateHarness(waveform)
    records = []

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        telemetry_callback=records.append,
    )

    np.testing.assert_array_equal(result[0], waveform.squeeze(0))
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, OmniVoiceGenerationTelemetry)
    assert record.batch_size == 1
    assert record.output_count == 1
    assert record.output_mode == "raw_codec"
    assert record.prompt_preparation_seconds is None
    assert record.input_preparation_seconds >= 0
    assert record.token_generation_seconds >= 0
    assert record.codec_decode_seconds >= 0
    assert record.postprocessing_seconds == 0
    assert record.wall_seconds >= (
        record.input_preparation_seconds
        + record.token_generation_seconds
        + record.codec_decode_seconds
    )
    assert record.cuda is None
    assert record.output_framing_seconds == 0
    assert record.final_duration_items == ()
    with pytest.raises(FrozenInstanceError):
        record.batch_size = 2


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (np.empty((1, 0), dtype=np.float32), "at least one sample"),
        (np.array([[np.nan]], dtype=np.float32), "non-finite sample"),
        (np.array([0.25], dtype=np.float32), "channels-first mono shape"),
        (np.array([[0.25 + 0.5j]], dtype=np.complex64), "real floating-point dtype"),
    ],
)
def test_raw_codec_without_target_rejects_invalid_decoder_waveform(
    waveform,
    message,
):
    model = GenerateHarness(waveform)
    records = []

    with pytest.raises(RuntimeError, match=message):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            telemetry_callback=records.append,
        )

    assert records == []


def test_raw_codec_without_target_preserves_valid_samples_bit_exactly():
    waveform = np.array(
        [[-0.0, 0.0, np.nextafter(np.float32(0.25), np.float32(1.0))]],
        dtype=np.float32,
    )
    model = GenerateHarness(waveform)

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
    )[0]

    assert result.dtype == waveform.dtype
    np.testing.assert_array_equal(
        result.view(np.uint8),
        waveform[0].view(np.uint8),
    )


@pytest.mark.parametrize(
    "generation_config",
    [
        OmniVoiceGenerationConfig(pad_duration=1e300),
        OmniVoiceGenerationConfig(output_target_lead_silence_ms=np.iinfo(np.intp).max),
        OmniVoiceGenerationConfig(output_target_trail_silence_ms=np.iinfo(np.intp).max),
    ],
)
def test_postprocessing_allocation_bounds_fail_before_generation(generation_config):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))

    with pytest.raises(ValueError, match="waveform-allocation limit"):
        _call_generate(
            model,
            text="test",
            generation_config=generation_config,
        )

    assert model.evaluated is False


def test_postprocessing_preflight_bounds_combined_target_edges(monkeypatch):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.sampling_rate = 1000
    monkeypatch.setattr(omnivoice_module, "_MAX_SAFE_FLOAT_WAVEFORM_SAMPLES", 10)

    with pytest.raises(ValueError, match="combined.*waveform-allocation limit"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(
                pad_duration=0.0,
                output_target_lead_silence_ms=5,
                output_target_trail_silence_ms=5,
            ),
        )

    assert model.evaluated is False


def test_postprocessing_preflight_bounds_target_plus_generic_pad(monkeypatch):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.sampling_rate = 1000
    monkeypatch.setattr(omnivoice_module, "_MAX_SAFE_FLOAT_WAVEFORM_SAMPLES", 10)

    with pytest.raises(ValueError, match="combined.*waveform-allocation limit"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(
                pad_duration=0.004,
                output_target_lead_silence_ms=6,
            ),
        )

    assert model.evaluated is False


def test_raw_codec_does_not_preflight_ignored_postprocessing_allocations():
    waveform = np.array([[0.1]], dtype=np.float32)
    model = GenerateHarness(waveform)

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(
            output_mode="raw_codec",
            pad_duration=1e300,
            output_target_lead_silence_ms=np.iinfo(np.intp).max,
        ),
    )

    np.testing.assert_array_equal(result[0], waveform[0])


def test_all_none_physical_targets_keep_default_path_and_empty_telemetry():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    class HistoricalDecoderHarness(GenerateHarness):
        def _decode_and_post_process(
            self,
            tokens,
            rms,
            gen_config,
            _telemetry_state=None,
        ):
            return OmniVoice._decode_and_post_process(
                self,
                tokens,
                rms,
                gen_config,
                _telemetry_state,
            )

    model = HistoricalDecoderHarness(waveform)
    records = []
    observations = []

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=[None],
        telemetry_callback=records.append,
        framing_observer=observations.append,
    )

    np.testing.assert_array_equal(result[0], waveform.squeeze(0))
    assert records[0].output_framing_seconds == 0
    assert records[0].framing_observer_seconds == 0
    assert records[0].final_duration_items == ()
    assert observations == []


def test_generate_emits_immutable_per_item_physical_framing_telemetry():
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = GenerateHarness(waveform)
    records = []
    observations = []

    result = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=5,
        telemetry_callback=records.append,
        framing_observer=observations.append,
    )

    np.testing.assert_array_equal(result[0][:3], waveform.squeeze(0))
    np.testing.assert_array_equal(result[0][3:], np.zeros(2, dtype=np.float32))
    record = records[0]
    assert record.output_framing_seconds >= 0
    assert record.framing_observer_seconds >= 0
    assert len(observations) == 1
    assert len(record.final_duration_items) == 1
    item = record.final_duration_items[0]
    assert item == OmniVoiceFinalDurationTelemetry(
        item_index=0,
        target_source="integer_samples",
        requested_seconds=None,
        operation="padded",
        protected_leading_edge=False,
        protected_trailing_edge=False,
        sample_rate=24_000,
        target_samples=5,
        source_samples=3,
        final_samples=5,
        padded_leading_samples=0,
        padded_trailing_samples=2,
        trimmed_leading_samples=0,
        trimmed_trailing_samples=0,
    )
    with pytest.raises(FrozenInstanceError):
        item.final_samples = 4
    assert record.wall_seconds >= (
        record.input_preparation_seconds
        + record.token_generation_seconds
        + record.codec_decode_seconds
        + record.postprocessing_seconds
        + record.output_framing_seconds
        + record.framing_observer_seconds
    )


def test_cuda_telemetry_is_collected_only_for_an_available_cuda_device(monkeypatch):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("cuda:0")
    synchronize_calls = []
    allocated_values = iter((10, 20))
    reserved_values = iter((30, 40))
    reset_calls = []

    def reset_peak_memory_stats(device):
        assert omnivoice_module._cuda_telemetry_lease_held_by_current_thread(device)
        reset_calls.append(str(device))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronize_calls.append(str(device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: next(allocated_values),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: next(reserved_values),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        reset_peak_memory_stats,
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 50)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 60)
    records = []

    _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        telemetry_callback=records.append,
    )

    assert synchronize_calls
    assert all(device == "cuda:0" for device in synchronize_calls)
    assert records[0].cuda == OmniVoiceCudaTelemetry(
        device="cuda:0",
        memory_allocated_start_bytes=10,
        memory_allocated_end_bytes=20,
        memory_reserved_start_bytes=30,
        memory_reserved_end_bytes=40,
        memory_allocated_peak_bytes=50,
        memory_reserved_peak_bytes=60,
        peak_stats_reset_at_call_start=True,
    )
    assert reset_calls == ["cuda:0"]


def test_cuda_peak_counters_are_untouched_without_telemetry_callback(monkeypatch):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("cuda:0")

    def unexpected_reset(device):
        raise AssertionError(f"unexpected peak reset on {device}")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", unexpected_reset)
    audios = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
    )
    assert len(audios) == 1


def test_cuda_telemetry_canonicalizes_an_implicit_device_index(monkeypatch):
    _mock_cuda_telemetry_runtime(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    reset_devices = []
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_devices.append(str(device)),
    )
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("cuda")
    records = []

    _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        telemetry_callback=records.append,
    )

    assert reset_devices == ["cuda:2"]
    assert records[0].cuda is not None
    assert records[0].cuda.device == "cuda:2"


def test_uninstrumented_cuda_generation_never_acquires_a_telemetry_lease(
    monkeypatch,
):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def unexpected_acquire(device):
        raise AssertionError(f"unexpected telemetry lease on {device}")

    monkeypatch.setattr(
        omnivoice_module,
        "_acquire_cuda_telemetry_lease",
        unexpected_acquire,
    )

    audios = _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
    )

    assert len(audios) == 1


def test_cuda_telemetry_lease_serializes_same_device_threads():
    first_acquired = Event()
    release_first = Event()
    second_acquired = Event()

    def first_worker():
        lease = omnivoice_module._acquire_cuda_telemetry_lease(torch.device("cuda:0"))
        try:
            first_acquired.set()
            assert release_first.wait(timeout=2)
        finally:
            lease.release()

    def second_worker():
        assert first_acquired.wait(timeout=2)
        lease = omnivoice_module._acquire_cuda_telemetry_lease(torch.device("cuda:0"))
        try:
            second_acquired.set()
        finally:
            lease.release()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_worker)
        second = executor.submit(second_worker)
        assert first_acquired.wait(timeout=2)
        assert not second_acquired.wait(timeout=0.1)
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert second_acquired.is_set()


def test_cuda_telemetry_leases_allow_distinct_devices_concurrently():
    both_acquired = Barrier(2)

    def worker(index):
        lease = omnivoice_module._acquire_cuda_telemetry_lease(
            torch.device(f"cuda:{index}")
        )
        try:
            both_acquired.wait(timeout=2)
        finally:
            lease.release()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, index) for index in (0, 1)]
        for future in futures:
            future.result(timeout=2)


def test_cuda_telemetry_lease_rejects_internal_same_thread_reentrancy():
    device = torch.device("cuda:0")
    lease = omnivoice_module._acquire_cuda_telemetry_lease(device)
    try:
        with pytest.raises(RuntimeError, match="reentrant telemetry-enabled"):
            omnivoice_module._acquire_cuda_telemetry_lease(device)
    finally:
        lease.release()


def _mock_cuda_telemetry_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 10)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 20)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 40)


def test_cuda_telemetry_lease_is_released_when_generation_fails(monkeypatch):
    _mock_cuda_telemetry_runtime(monkeypatch)

    class FailingHarness(GenerateHarness):
        def _preprocess_all(self, **_kwargs):
            raise RuntimeError("synthetic generation failure")

    model = FailingHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("cuda:0")

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        _call_generate(model, text="test", telemetry_callback=lambda _record: None)

    assert not omnivoice_module._cuda_telemetry_lease_held_by_current_thread(
        torch.device("cuda:0")
    )
    lease = omnivoice_module._acquire_cuda_telemetry_lease(torch.device("cuda:0"))
    lease.release()


def test_public_callbacks_run_after_cuda_lease_release_and_can_reenter(monkeypatch):
    _mock_cuda_telemetry_runtime(monkeypatch)
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = GenerateHarness(waveform)
    model.device = torch.device("cuda:0")
    nested_records = []
    callback_order = []

    def framing_callback(_observation):
        assert not omnivoice_module._cuda_telemetry_lease_held_by_current_thread(
            torch.device("cuda:0")
        )
        callback_order.append("framing")

    def telemetry_callback(_record):
        assert not omnivoice_module._cuda_telemetry_lease_held_by_current_thread(
            torch.device("cuda:0")
        )
        callback_order.append("telemetry")
        _call_generate(
            model,
            text="nested",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            telemetry_callback=nested_records.append,
        )

    _call_generate(
        model,
        text="outer",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        final_duration_samples=3,
        framing_observer=framing_callback,
        telemetry_callback=telemetry_callback,
    )

    assert callback_order == ["framing", "telemetry"]
    assert len(nested_records) == 1


def test_mps_telemetry_synchronizes_stage_boundaries(monkeypatch):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("mps")
    synchronize_calls = []

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.mps,
        "synchronize",
        lambda: synchronize_calls.append("mps"),
    )
    records = []

    _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        telemetry_callback=records.append,
    )

    assert synchronize_calls
    assert set(synchronize_calls) == {"mps"}
    assert records[0].cuda is None


def test_xpu_telemetry_synchronizes_stage_boundaries(monkeypatch):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("xpu:1")
    synchronize_calls = []

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.xpu,
        "synchronize",
        lambda device: synchronize_calls.append(str(device)),
    )
    records = []

    _call_generate(
        model,
        text="test",
        generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
        telemetry_callback=records.append,
    )

    assert synchronize_calls
    assert set(synchronize_calls) == {"xpu:1"}
    assert records[0].cuda is None


def test_telemetry_callback_exceptions_propagate_after_generation():
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))

    def failing_callback(record):
        assert record.output_count == 1
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        _call_generate(
            model,
            text="test",
            generation_config=OmniVoiceGenerationConfig(output_mode="raw_codec"),
            telemetry_callback=failing_callback,
        )


def test_concurrent_telemetry_callbacks_do_not_hold_a_global_lock():
    record = OmniVoiceGenerationTelemetry(
        batch_size=1,
        output_count=1,
        output_mode="processed",
        prompt_preparation_seconds=None,
        input_preparation_seconds=0.0,
        token_generation_seconds=0.0,
        codec_decode_seconds=0.0,
        postprocessing_seconds=0.0,
        wall_seconds=0.0,
        cuda=None,
    )
    callback_barrier = Barrier(2)

    def callback(received):
        assert received is record
        callback_barrier.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                omnivoice_module._emit_generation_telemetry,
                callback,
                record,
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result()


def test_cuda_telemetry_type_is_public_and_immutable():
    cuda = OmniVoiceCudaTelemetry(
        device="cuda:0",
        memory_allocated_start_bytes=1,
        memory_allocated_end_bytes=2,
        memory_reserved_start_bytes=3,
        memory_reserved_end_bytes=4,
        memory_allocated_peak_bytes=5,
        memory_reserved_peak_bytes=6,
        peak_stats_reset_at_call_start=True,
    )

    assert cuda.device == "cuda:0"
    with pytest.raises(FrozenInstanceError):
        cuda.device = "cpu"

    with pytest.raises(ValueError, match="allocated peak"):
        OmniVoiceCudaTelemetry(
            device="cuda:0",
            memory_allocated_start_bytes=10,
            memory_allocated_end_bytes=20,
            memory_reserved_start_bytes=30,
            memory_reserved_end_bytes=40,
            memory_allocated_peak_bytes=19,
            memory_reserved_peak_bytes=40,
            peak_stats_reset_at_call_start=True,
        )
