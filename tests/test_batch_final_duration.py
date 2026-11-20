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

"""Batch and JSONL coverage for physical output-duration controls."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import omnivoice.cli.infer_batch as infer_batch_module
from omnivoice.utils.audio import _write_wav_atomic
from omnivoice.utils.data_utils import JsonlTestListError, read_test_list
from omnivoice.utils.duration import RuleDurationEstimator


def test_jsonl_reader_preserves_both_physical_duration_authorities(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "sample",
                "text": "test",
                "final_duration": 2.6791875,
                "final_duration_samples": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = read_test_list(path)

    assert samples[0]["final_duration"] == 2.6791875
    assert samples[0]["final_duration_samples"] is None


def test_jsonl_reader_fails_closed_on_malformed_line_after_valid_item(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps({"id": "valid", "text": "test"}) + "\n{" + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlTestListError, match=r"line 2: invalid JSON"):
        read_test_list(path)


def test_jsonl_reader_rejects_duplicate_ids_with_both_line_numbers(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps({"id": "same", "text": "first"})
        + "\n"
        + json.dumps({"id": "same", "text": "second"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlTestListError, match=r"line 2: duplicate.*line 1"):
        read_test_list(path)


def test_jsonl_reader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id":"first","id":"second","text":"test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlTestListError, match=r"line 1: duplicate JSON key 'id'"):
        read_test_list(path)


def test_jsonl_reader_rejects_nonstandard_nonfinite_numbers(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"id":"sample","text":"test","final_duration":NaN}\n',
        encoding="utf-8",
    )

    with pytest.raises(JsonlTestListError, match=r"line 1: non-standard.*NaN"):
        read_test_list(path)


@pytest.mark.parametrize(
    "unsafe_id",
    [
        "../escape",
        "..\\escape",
        "C:\\escape",
        "folder/name",
        "CON",
        "CONIN$",
        "CONOUT$",
        "COM¹",
        "LPT³",
        "CON .txt",
        "NUL .log",
        "name:ads",
        "LONGFI~1",
        "x" * 300,
    ],
)
def test_output_ids_cannot_escape_or_alias_the_result_directory(unsafe_id, tmp_path):
    with pytest.raises(ValueError, match="unsafe output id"):
        infer_batch_module._resolve_output_paths([unsafe_id], str(tmp_path))


def test_output_ids_are_unique_under_portable_case_folding(tmp_path):
    with pytest.raises(ValueError, match="duplicate WAV destination"):
        infer_batch_module._resolve_output_paths(
            ["Michael", "michael"],
            str(tmp_path),
        )


def test_existing_symlink_destination_is_rejected(tmp_path):
    target = tmp_path / "target.wav"
    target.touch()
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    try:
        first.symlink_to(target)
        second.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        infer_batch_module._resolve_output_paths(["first", "second"], str(tmp_path))


def test_existing_hard_link_destination_is_rejected(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.touch()
    try:
        second.hardlink_to(first)
    except OSError as exc:
        pytest.skip(f"hard-link creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="hard-linked"):
        infer_batch_module._resolve_output_paths(["first", "second"], str(tmp_path))


def test_cli_rejects_existing_file_result_directory_before_runtime_side_effects(
    monkeypatch, tmp_path
):
    test_list = tmp_path / "test.jsonl"
    test_list.write_text(
        json.dumps({"id": "sample", "text": "test"}) + "\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "results"
    result_path.write_text("keep", encoding="utf-8")

    class ParserStub:
        @staticmethod
        def parse_args():
            return SimpleNamespace(
                test_list=str(test_list),
                res_dir=str(result_path),
                final_duration=None,
                final_duration_samples=None,
            )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("result-directory validation must precede startup")

    monkeypatch.setattr(infer_batch_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_batch_module.mp, "set_start_method", forbidden)
    monkeypatch.setattr(infer_batch_module.os, "makedirs", forbidden)
    monkeypatch.setattr(infer_batch_module, "get_best_device_with_count", forbidden)
    monkeypatch.setattr(infer_batch_module, "ProcessPoolExecutor", forbidden)

    with pytest.raises(ValueError, match="result directory.*not a directory"):
        infer_batch_module.main()

    assert result_path.read_text(encoding="utf-8") == "keep"


def test_cli_validates_entire_jsonl_before_creating_workers_or_output_directory(
    monkeypatch, tmp_path
):
    test_list = tmp_path / "invalid.jsonl"
    test_list.write_text(
        json.dumps({"id": "valid", "text": "test"}) + "\n{" + "\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "results"

    class ParserStub:
        @staticmethod
        def parse_args():
            return SimpleNamespace(
                test_list=str(test_list),
                res_dir=str(result_dir),
                final_duration=None,
                final_duration_samples=None,
            )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("validation must finish before any runtime side effect")

    monkeypatch.setattr(infer_batch_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(
        infer_batch_module.mp, "set_start_method", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(infer_batch_module.os, "makedirs", forbidden)
    monkeypatch.setattr(infer_batch_module, "get_best_device_with_count", forbidden)
    monkeypatch.setattr(infer_batch_module, "ProcessPoolExecutor", forbidden)

    with pytest.raises(JsonlTestListError, match=r"line 2: invalid JSON"):
        infer_batch_module.main()

    assert not result_dir.exists()


@pytest.mark.parametrize(
    "physical_fields",
    [
        {"final_duration": 1.0, "final_duration_samples": 24_000},
        {"final_duration_samples": -1},
        {"final_duration": "1.0"},
    ],
)
def test_cli_rejects_invalid_physical_targets_before_runtime_side_effects(
    monkeypatch, tmp_path, physical_fields
):
    test_list = tmp_path / "invalid-target.jsonl"
    test_list.write_text(
        json.dumps({"id": "sample", "text": "test", **physical_fields}) + "\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "results"

    class ParserStub:
        @staticmethod
        def parse_args():
            return SimpleNamespace(
                test_list=str(test_list),
                res_dir=str(result_dir),
                final_duration=None,
                final_duration_samples=None,
            )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target validation must precede every runtime side effect")

    monkeypatch.setattr(infer_batch_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_batch_module.mp, "set_start_method", forbidden)
    monkeypatch.setattr(infer_batch_module.os, "makedirs", forbidden)
    monkeypatch.setattr(infer_batch_module, "get_best_device_with_count", forbidden)
    monkeypatch.setattr(infer_batch_module, "ProcessPoolExecutor", forbidden)

    with pytest.raises((TypeError, ValueError)):
        infer_batch_module.main()

    assert not result_dir.exists()


def test_cli_rejects_unknown_physical_authority_typo_before_runtime_startup(
    monkeypatch, tmp_path
):
    test_list = tmp_path / "typo.jsonl"
    test_list.write_text(
        json.dumps({"id": "sample", "text": "test", "final_duration_sample": 24_000})
        + "\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "results"

    class ParserStub:
        @staticmethod
        def parse_args():
            return SimpleNamespace(test_list=str(test_list), res_dir=str(result_dir))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("strict JSONL validation must precede runtime startup")

    monkeypatch.setattr(infer_batch_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_batch_module.mp, "set_start_method", forbidden)
    monkeypatch.setattr(infer_batch_module.os, "makedirs", forbidden)
    monkeypatch.setattr(infer_batch_module, "get_best_device_with_count", forbidden)
    monkeypatch.setattr(infer_batch_module, "ProcessPoolExecutor", forbidden)

    with pytest.raises(JsonlTestListError, match="final_duration_sample"):
        infer_batch_module.main()

    assert not result_dir.exists()


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"duration": "x"},
        {"speed": "fast"},
        {"language_id": 1},
        {"instruct": 1},
        {"ref_audio": False},
        {"final_duration_samples": 1.5},
    ],
)
def test_cli_rejects_invalid_known_optional_fields_before_runtime_startup(
    monkeypatch, tmp_path, invalid_field
):
    test_list = tmp_path / "invalid-known-field.jsonl"
    test_list.write_text(
        json.dumps({"id": "sample", "text": "test", **invalid_field}) + "\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "results"

    class ParserStub:
        @staticmethod
        def parse_args():
            return SimpleNamespace(test_list=str(test_list), res_dir=str(result_dir))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("known-field validation must precede runtime startup")

    monkeypatch.setattr(infer_batch_module, "get_parser", lambda: ParserStub())
    monkeypatch.setattr(infer_batch_module.mp, "set_start_method", forbidden)
    monkeypatch.setattr(infer_batch_module.os, "makedirs", forbidden)
    monkeypatch.setattr(infer_batch_module, "get_best_device_with_count", forbidden)
    monkeypatch.setattr(infer_batch_module, "ProcessPoolExecutor", forbidden)

    with pytest.raises(JsonlTestListError):
        infer_batch_module.main()

    assert not result_dir.exists()


def test_batch_runtime_forwards_mixed_per_item_physical_targets(monkeypatch, tmp_path):
    received = {}

    class WorkerModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(**kwargs):
            received.update(kwargs)
            return [
                np.zeros(64_301, dtype=np.float32),
                np.zeros(12_345, dtype=np.float32),
            ]

    monkeypatch.setattr(infer_batch_module, "worker_model", WorkerModelStub())
    monkeypatch.setattr(infer_batch_module.sf, "write", lambda *args, **kwargs: None)
    samples = [
        (
            "seconds",
            None,
            None,
            "first",
            None,
            None,
            None,
            None,
            2.6791875,
            None,
        ),
        (
            "samples",
            None,
            None,
            "second",
            None,
            None,
            None,
            None,
            None,
            12_345,
        ),
    ]

    result = infer_batch_module.run_inference_batch(samples, str(tmp_path))

    assert received["final_duration"] == [2.6791875, None]
    assert received["final_duration_samples"] == [None, 12_345]
    assert [item[0] for item in result] == ["seconds", "samples"]


def test_batch_seconds_target_uses_the_worker_models_actual_sample_rate(
    monkeypatch, tmp_path
):
    received = {}

    class WorkerModelStub:
        sampling_rate = 48_000

        @staticmethod
        def generate(**kwargs):
            received.update(kwargs)
            return [np.zeros(1, dtype=np.float32)]

    monkeypatch.setattr(infer_batch_module, "worker_model", WorkerModelStub())
    monkeypatch.setattr(infer_batch_module.sf, "write", lambda *args, **kwargs: None)
    sample = (
        "sample",
        None,
        None,
        "test",
        None,
        None,
        None,
        None,
        0.000015,
        None,
    )

    result = infer_batch_module.run_inference_batch([sample], str(tmp_path))

    assert received["final_duration"] == [0.000015]
    assert result[0][2] == pytest.approx(1 / 48_000)


def test_batch_default_does_not_inject_physical_kwargs_into_legacy_adapter(
    monkeypatch, tmp_path
):
    class WorkerModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(
            text,
            language,
            ref_audio,
            ref_text,
            duration,
            speed,
            instruct,
        ):
            return [np.zeros(24_000, dtype=np.float32)]

    monkeypatch.setattr(infer_batch_module, "worker_model", WorkerModelStub())
    monkeypatch.setattr(infer_batch_module.sf, "write", lambda *args, **kwargs: None)

    result = infer_batch_module.run_inference_batch(
        [("sample", None, None, "test", None, None, None, None)],
        str(tmp_path),
    )

    assert result[0][0] == "sample"


def test_batch_rejects_output_cardinality_mismatch_before_writing(
    monkeypatch, tmp_path
):
    writes = []

    class WorkerModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(**kwargs):
            return [np.zeros(24_000, dtype=np.float32)]

    monkeypatch.setattr(infer_batch_module, "worker_model", WorkerModelStub())
    monkeypatch.setattr(
        infer_batch_module.sf,
        "write",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    samples = [
        ("first", None, None, "first", None, None, None, None, None, None),
        ("second", None, None, "second", None, None, None, None, None, None),
    ]

    with pytest.raises(RuntimeError, match="expected 2, got 1"):
        infer_batch_module.run_inference_batch(samples, str(tmp_path))

    assert writes == []


def test_batch_rejects_physical_length_mismatch_before_writing(monkeypatch, tmp_path):
    writes = []

    class WorkerModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(**kwargs):
            return [np.zeros(4, dtype=np.float32)]

    monkeypatch.setattr(infer_batch_module, "worker_model", WorkerModelStub())
    monkeypatch.setattr(
        infer_batch_module.sf,
        "write",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    samples = [
        ("sample", None, None, "test", None, None, None, None, None, 5),
    ]

    with pytest.raises(RuntimeError, match="expected 5 samples, got 4"):
        infer_batch_module.run_inference_batch(samples, str(tmp_path))

    assert writes == []


@pytest.mark.parametrize(
    "invalid_audio",
    [
        np.array([], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([1], dtype=np.int16),
    ],
)
def test_batch_rejects_invalid_waveform_before_writing(
    monkeypatch, tmp_path, invalid_audio
):
    writes = []

    class WorkerModelStub:
        sampling_rate = 24_000

        @staticmethod
        def generate(**kwargs):
            return [invalid_audio]

    monkeypatch.setattr(infer_batch_module, "worker_model", WorkerModelStub())
    monkeypatch.setattr(
        infer_batch_module.sf,
        "write",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError):
        infer_batch_module.run_inference_batch(
            [("sample", None, None, "test", None, None, None, None)],
            str(tmp_path),
        )

    assert writes == []


def test_portable_component_limit_allows_255_but_rejects_256_units(tmp_path):
    accepted_id = "x" * 251
    [accepted_path] = infer_batch_module._resolve_output_paths(
        [accepted_id],
        str(tmp_path),
    )

    def writer(path, _audio, _sampling_rate, **_kwargs):
        Path(path).write_bytes(b"staged-wav")

    _write_wav_atomic(
        accepted_path,
        np.zeros(1, dtype=np.float32),
        24_000,
        writer=writer,
    )

    assert Path(accepted_path).name == accepted_id + ".wav"
    assert Path(accepted_path).read_bytes() == b"staged-wav"
    with pytest.raises(ValueError, match="255-unit component limit"):
        infer_batch_module._resolve_output_paths(["x" * 252], str(tmp_path))


def test_gpu_batch_cost_sorting_uses_generation_duration_not_final_duration():
    estimator = RuleDurationEstimator()
    short_generation_huge_output = (
        "short",
        None,
        None,
        "short text",
        None,
        1.0,
        None,
        None,
        None,
        240_000,
    )
    long_generation_short_output = (
        "long",
        None,
        None,
        "longer text",
        None,
        5.0,
        None,
        None,
        None,
        24_000,
    )

    sorted_samples = infer_batch_module._sort_samples_by_duration(
        [short_generation_huge_output, long_generation_short_output],
        estimator,
    )

    assert [sample[0][0] for sample in sorted_samples] == ["long", "short"]


def test_batch_failures_are_propagated_instead_of_reporting_success():
    original = ValueError("unsafe active-sample overflow")

    with pytest.raises(RuntimeError, match=r"1 inference batch\(es\) failed") as exc:
        infer_batch_module._raise_if_batch_failures([original])

    assert exc.value.__cause__ is original


def test_empty_failure_list_is_a_noop():
    assert infer_batch_module._raise_if_batch_failures([]) is None
