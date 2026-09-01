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

"""Tests for generated-audio post-processing CLI controls."""

import numpy as np
import pytest
import soundfile as sf

import omnivoice.cli.infer as infer_module
import omnivoice.cli.infer_batch as infer_batch_module
from omnivoice.cli.infer import get_parser as get_infer_parser
from omnivoice.cli.infer_batch import get_parser as get_batch_parser


EXPECTED_DEFAULTS = {
    "output_mode": "processed",
    "output_min_silence_ms": 500,
    "output_keep_silence_ms": None,
    "output_lead_silence_ms": 100,
    "output_trail_silence_ms": 100,
    "output_peak_limit": None,
    "output_target_lead_silence_ms": None,
    "output_target_trail_silence_ms": None,
    "pad_duration": 0.1,
    "fade_duration": 0.1,
}

OVERRIDE_ARGUMENTS = [
    "--output_mode",
    "raw_codec",
    "--output_min_silence_ms",
    "420",
    "--output_keep_silence_ms",
    "80",
    "--output_lead_silence_ms",
    "30",
    "--output_trail_silence_ms",
    "50",
    "--output_peak_limit",
    "0.98",
    "--output_target_lead_silence_ms",
    "250",
    "--output_target_trail_silence_ms",
    "75",
    "--pad_duration",
    "0",
    "--fade_duration",
    "0.02",
]

INVALID_ARGUMENTS = [
    ["--output_mode", "raw"],
    ["--output_min_silence_ms", "-1"],
    ["--output_keep_silence_ms", "1.5"],
    ["--output_lead_silence_ms", "-1"],
    ["--output_trail_silence_ms", "-1"],
    ["--pad_duration", "nan"],
    ["--pad_duration", "inf"],
    ["--fade_duration", "-0.1"],
    ["--output_peak_limit", "0"],
    ["--output_peak_limit", "1.01"],
    ["--output_peak_limit", "nan"],
    ["--output_target_lead_silence_ms", "-1"],
    ["--output_target_lead_silence_ms", "1.5"],
    ["--output_target_trail_silence_ms", "-1"],
    ["--output_target_trail_silence_ms", "1.5"],
]

CLI_WAV_WRITERS = [
    pytest.param(infer_module._write_output_wav, id="single"),
    pytest.param(infer_batch_module._write_output_wav, id="batch"),
]


def _assert_defaults(namespace):
    for name, expected in EXPECTED_DEFAULTS.items():
        assert getattr(namespace, name) == expected


def _assert_overrides(namespace):
    assert namespace.output_mode == "raw_codec"
    assert namespace.output_min_silence_ms == 420
    assert namespace.output_keep_silence_ms == 80
    assert namespace.output_lead_silence_ms == 30
    assert namespace.output_trail_silence_ms == 50
    assert namespace.output_peak_limit == 0.98
    assert namespace.output_target_lead_silence_ms == 250
    assert namespace.output_target_trail_silence_ms == 75
    assert namespace.pad_duration == 0
    assert namespace.fade_duration == 0.02


def test_single_inference_parser_exposes_postprocessing_defaults():
    args = get_infer_parser().parse_args(["--text", "test", "--output", "out.wav"])
    _assert_defaults(args)


def test_single_inference_parser_accepts_postprocessing_overrides():
    args = get_infer_parser().parse_args(
        ["--text", "test", "--output", "out.wav", *OVERRIDE_ARGUMENTS]
    )
    _assert_overrides(args)


def test_batch_inference_parser_exposes_postprocessing_defaults():
    args = get_batch_parser().parse_args(
        ["--test_list", "test.jsonl", "--res_dir", "results"]
    )
    _assert_defaults(args)


def test_batch_inference_parser_accepts_postprocessing_overrides():
    args = get_batch_parser().parse_args(
        [
            "--test_list",
            "test.jsonl",
            "--res_dir",
            "results",
            *OVERRIDE_ARGUMENTS,
        ]
    )
    _assert_overrides(args)


def test_batch_runtime_forwards_postprocessing_controls(monkeypatch, tmp_path):
    received = {}

    class WorkerModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(**kwargs):
            received.update(kwargs)
            return [np.zeros(240, dtype=np.float32)]

    args = get_batch_parser().parse_args(
        [
            "--test_list",
            "test.jsonl",
            "--res_dir",
            str(tmp_path),
            *OVERRIDE_ARGUMENTS,
        ]
    )
    gen_kwargs = vars(args).copy()
    gen_kwargs.pop("res_dir")
    monkeypatch.setattr(infer_batch_module, "worker_model", WorkerModelStub())
    monkeypatch.setattr(infer_batch_module.sf, "write", lambda *args, **kwargs: None)

    infer_batch_module.run_inference_batch(
        [("sample", None, None, "test", None, None, None, None)],
        str(tmp_path),
        **gen_kwargs,
    )

    for name in EXPECTED_DEFAULTS:
        assert received[name] == getattr(args, name)


@pytest.mark.parametrize("writer", CLI_WAV_WRITERS)
def test_raw_codec_cli_wav_round_trip_preserves_float32_over_range(writer, tmp_path):
    waveform = np.array([0.0, 1.25, -1.5, 0.125], dtype=np.float32)
    path = tmp_path / "raw.wav"

    writer(path, waveform, 24_000, "raw_codec")

    info = sf.info(path)
    decoded, sample_rate = sf.read(path, dtype="float32")
    assert info.format == "WAV"
    assert info.subtype == "FLOAT"
    assert sample_rate == 24_000
    np.testing.assert_array_equal(decoded, waveform)
    assert np.max(np.abs(decoded)) > 1.0


@pytest.mark.parametrize("writer", CLI_WAV_WRITERS)
def test_processed_cli_wav_keeps_backward_compatible_default_subtype(writer, tmp_path):
    path = tmp_path / "processed.wav"

    writer(path, np.array([0.0, 0.25], dtype=np.float32), 24_000, "processed")

    assert sf.info(path).subtype == "PCM_16"


@pytest.mark.parametrize("invalid_arguments", INVALID_ARGUMENTS)
def test_single_inference_parser_rejects_invalid_postprocessing_values(
    invalid_arguments,
):
    with pytest.raises(SystemExit):
        get_infer_parser().parse_args(
            ["--text", "test", "--output", "out.wav", *invalid_arguments]
        )


@pytest.mark.parametrize("invalid_arguments", INVALID_ARGUMENTS)
def test_batch_inference_parser_rejects_invalid_postprocessing_values(
    invalid_arguments,
):
    with pytest.raises(SystemExit):
        get_batch_parser().parse_args(
            [
                "--test_list",
                "test.jsonl",
                "--res_dir",
                "results",
                *invalid_arguments,
            ]
        )
