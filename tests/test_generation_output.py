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
from dataclasses import FrozenInstanceError
from threading import Barrier
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

import omnivoice.models.omnivoice as omnivoice_module
from omnivoice import (
    OmniVoiceCudaTelemetry,
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


def test_generate_without_telemetry_avoids_timers_and_cuda_queries(monkeypatch):
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)
    model = GenerateHarness(waveform)

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
    )

    assert model.evaluated
    np.testing.assert_array_equal(result[0], waveform.squeeze(0))


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
    with pytest.raises(FrozenInstanceError):
        record.batch_size = 2


def test_cuda_telemetry_is_collected_only_for_an_available_cuda_device(monkeypatch):
    model = GenerateHarness(np.array([[0.1]], dtype=np.float32))
    model.device = torch.device("cuda:0")
    synchronize_calls = []
    allocated_values = iter((10, 20))
    reserved_values = iter((30, 40))

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
    )


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
    )

    assert cuda.device == "cuda:0"
    with pytest.raises(FrozenInstanceError):
        cuda.device = "cpu"
