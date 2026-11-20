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

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import omnivoice.cli.infer as infer_module
import omnivoice.cli.infer_batch as infer_batch_module
import omnivoice.utils.audio as audio_utils
from omnivoice.cli.infer import get_parser as get_infer_parser
from omnivoice.cli.infer_batch import get_parser as get_batch_parser


EXPECTED_DEFAULTS = {
    "final_duration": None,
    "final_duration_samples": None,
    "output_mode": "processed",
    "output_min_silence_ms": 500,
    "output_keep_silence_ms": None,
    "output_preserve_active_edges": False,
    "output_lead_silence_ms": 100,
    "output_trail_silence_ms": 100,
    "output_peak_limit": None,
    "output_target_lead_silence_ms": None,
    "output_target_trail_silence_ms": None,
    "pad_duration": 0.1,
    "fade_duration": 0.1,
}

OVERRIDE_ARGUMENTS = [
    "--output_preserve_active_edges",
    "--final_duration_samples",
    "64301",
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
    ["--final_duration", "0"],
    ["--final_duration", "nan"],
    ["--final_duration", "inf"],
    ["--final_duration_samples", "0"],
    ["--final_duration_samples", "1.5"],
]

CLI_WAV_WRITERS = [pytest.param(audio_utils.write_output_wav, id="shared")]


def test_clis_use_the_shared_output_wav_writer():
    assert infer_module.write_output_wav is audio_utils.write_output_wav
    assert infer_batch_module.write_output_wav is audio_utils.write_output_wav


def _assert_defaults(namespace):
    for name, expected in EXPECTED_DEFAULTS.items():
        assert getattr(namespace, name) == expected


def _assert_overrides(namespace):
    assert namespace.output_preserve_active_edges is True
    assert namespace.final_duration is None
    assert namespace.final_duration_samples == 64301
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
            return [np.zeros(64_301, dtype=np.float32)]

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

    for name in EXPECTED_DEFAULTS.keys() - {
        "final_duration",
        "final_duration_samples",
    }:
        assert received[name] == getattr(args, name)

    assert "final_duration" not in received
    assert received["final_duration_samples"] == [64301]


@pytest.mark.parametrize(
    ("parser", "required"),
    [
        (get_infer_parser, ["--text", "test", "--output", "out.wav"]),
        (get_batch_parser, ["--test_list", "test.jsonl", "--res_dir", "results"]),
    ],
)
def test_cli_duration_authorities_are_mutually_exclusive(parser, required):
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                *required,
                "--final_duration",
                "1.0",
                "--final_duration_samples",
                "24000",
            ]
        )


def test_single_cli_rejects_sample_count_overflow_before_model_loading(monkeypatch):
    class ParserStub:
        @staticmethod
        def parse_args():
            return SimpleNamespace(
                final_duration=None,
                final_duration_samples=np.iinfo(np.intp).max + 1,
            )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("physical target validation must precede model loading")

    monkeypatch.setattr(infer_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_module, "get_best_device", forbidden)
    monkeypatch.setattr(infer_module.OmniVoice, "from_pretrained", forbidden)

    with pytest.raises(ValueError, match="platform sample-count limit"):
        infer_module.main()


def test_single_cli_rejects_missing_output_parent_before_model_loading(
    monkeypatch, tmp_path
):
    output = tmp_path / "missing" / "output.wav"
    args = get_infer_parser().parse_args(["--text", "test", "--output", str(output)])

    class ParserStub:
        @staticmethod
        def parse_args():
            return args

    def forbidden(*_args, **_kwargs):
        raise AssertionError("output validation must precede model loading")

    monkeypatch.setattr(infer_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_module, "get_best_device", forbidden)
    monkeypatch.setattr(infer_module.OmniVoice, "from_pretrained", forbidden)

    with pytest.raises(ValueError, match="output parent directory"):
        infer_module.main()

    assert not output.parent.exists()


def test_single_cli_rejects_output_directory_before_model_loading(
    monkeypatch, tmp_path
):
    output = tmp_path / "output.wav"
    output.mkdir()
    args = get_infer_parser().parse_args(["--text", "test", "--output", str(output)])

    class ParserStub:
        @staticmethod
        def parse_args():
            return args

    def forbidden(*_args, **_kwargs):
        raise AssertionError("output validation must precede model loading")

    monkeypatch.setattr(infer_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_module, "get_best_device", forbidden)
    monkeypatch.setattr(infer_module.OmniVoice, "from_pretrained", forbidden)

    with pytest.raises(ValueError, match="output path.*not a regular file"):
        infer_module.main()

    assert output.is_dir()


def test_single_cli_rejects_physical_length_mismatch_before_writing(
    monkeypatch, tmp_path
):
    output = tmp_path / "output.wav"
    args = get_infer_parser().parse_args(
        [
            "--text",
            "test",
            "--output",
            str(output),
            "--final_duration_samples",
            "5",
        ]
    )

    class ParserStub:
        @staticmethod
        def parse_args():
            return args

    class ModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(**kwargs):
            return [np.zeros(4, dtype=np.float32)]

    writes = []
    monkeypatch.setattr(infer_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_module, "get_best_device", lambda: "cpu")
    monkeypatch.setattr(
        infer_module.OmniVoice,
        "from_pretrained",
        lambda *args, **kwargs: ModelStub(),
    )
    monkeypatch.setattr(
        infer_module,
        "write_output_wav",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="expected 5 samples, got 4"):
        infer_module.main()

    assert writes == []
    assert not output.exists()


@pytest.mark.parametrize(
    "invalid_audio",
    [
        np.array([], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([1], dtype=np.int16),
    ],
)
def test_single_cli_rejects_invalid_waveform_before_writing(
    monkeypatch, tmp_path, invalid_audio
):
    output = tmp_path / "output.wav"
    args = get_infer_parser().parse_args(["--text", "test", "--output", str(output)])

    class ParserStub:
        @staticmethod
        def parse_args():
            return args

    class ModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(**kwargs):
            return [invalid_audio]

    writes = []
    monkeypatch.setattr(infer_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_module, "get_best_device", lambda: "cpu")
    monkeypatch.setattr(
        infer_module.OmniVoice,
        "from_pretrained",
        lambda *args, **kwargs: ModelStub(),
    )
    monkeypatch.setattr(
        infer_module,
        "write_output_wav",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError):
        infer_module.main()

    assert writes == []
    assert not output.exists()


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


@pytest.mark.parametrize("writer", CLI_WAV_WRITERS)
def test_cli_atomic_wav_write_preserves_existing_destination_on_failure(
    monkeypatch, writer, tmp_path
):
    path = tmp_path / "existing.wav"
    original = b"existing-destination"
    path.write_bytes(original)

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("simulated writer failure")

    monkeypatch.setattr(sf, "write", fail_write)

    with pytest.raises(RuntimeError, match="simulated writer failure"):
        writer(path, np.zeros(2, dtype=np.float32), 24_000, "processed")

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".omnivoice-*.tmp.wav")) == []


@pytest.mark.parametrize("writer", CLI_WAV_WRITERS)
def test_cli_atomic_wav_commit_failure_preserves_existing_destination(
    monkeypatch, writer, tmp_path
):
    path = tmp_path / "existing.wav"
    original = b"existing-destination"
    path.write_bytes(original)

    def fail_commit(_temporary, _destination):
        raise OSError("simulated atomic commit failure")

    monkeypatch.setattr(audio_utils, "_replace_wav_staging_file", fail_commit)

    with pytest.raises(OSError, match="simulated atomic commit failure"):
        writer(path, np.zeros(2, dtype=np.float32), 24_000, "processed")

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".omnivoice-*.tmp.wav")) == []


@pytest.mark.parametrize("writer", CLI_WAV_WRITERS)
def test_cli_atomic_wav_write_overwrites_existing_destination(writer, tmp_path):
    path = tmp_path / "existing.wav"
    writer(path, np.array([0.1, -0.1], dtype=np.float32), 24_000, "processed")
    writer(path, np.array([0.2, -0.2, 0.3], dtype=np.float32), 24_000, "processed")

    decoded, sample_rate = sf.read(path, dtype="float32")
    assert sample_rate == 24_000
    assert decoded.shape == (3,)
    np.testing.assert_allclose(decoded, [0.2, -0.2, 0.3], atol=1 / 32768)
    assert list(tmp_path.glob(".omnivoice-*.tmp.wav")) == []


@pytest.mark.parametrize("invalid_arguments", INVALID_ARGUMENTS)
def test_single_inference_parser_rejects_invalid_postprocessing_values(
    invalid_arguments,
):
    with pytest.raises(SystemExit):
        get_infer_parser().parse_args(
            ["--text", "test", "--output", "out.wav", *invalid_arguments]
        )


@pytest.mark.parametrize("name", ["--speed", "--duration"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_single_inference_parser_rejects_invalid_speed_and_duration(name, value):
    with pytest.raises(SystemExit):
        get_infer_parser().parse_args(
            ["--text", "test", "--output", "out.wav", name, value]
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
