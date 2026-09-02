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

"""Adversarial tests for exact physical output-duration framing."""

import inspect
import math
from decimal import ROUND_DOWN, localcontext

import numpy as np
import pytest

import omnivoice.utils.audio as audio_utils
from omnivoice.models.omnivoice import (
    OmniVoice,
    OmniVoiceGenerationTelemetry,
    _final_duration_seconds_to_samples,
    _normalize_final_duration_targets,
)
from omnivoice.utils.audio import fit_audio_to_target_samples


SAMPLE_RATE = 24_000


def test_exact_active_edge_scan_crosses_bounded_chunks(monkeypatch):
    monkeypatch.setattr(audio_utils, "_PCM16_EDGE_SCAN_CHUNK_SAMPLES", 3)
    audio = np.zeros((2, 11), dtype=np.float32)
    audio[0, 4] = np.float32(0.25)
    audio[1, 8] = np.float32(-0.5)

    assert audio_utils._find_exact_active_edge(audio, from_end=False) == 4
    assert audio_utils._find_exact_active_edge(audio, from_end=True) == 8


@pytest.mark.parametrize(
    ("seconds", "expected_samples"),
    [
        (2.6791875, 64_301),
        (2.2180625, 53_234),
        (2.6584375, 63_803),
        (2.0606875, 49_457),
        (1.2319375, 29_567),
        (1.1341875, 27_221),
        (2.3666875, 56_801),
    ],
)
def test_decimal_half_up_corrects_physical_half_frame_boundaries(
    seconds, expected_samples
):
    _, samples = _final_duration_seconds_to_samples(seconds, SAMPLE_RATE)

    assert samples == expected_samples


def test_decimal_half_up_does_not_use_the_exact_binary_float_ratio():
    seconds = 2.6791875
    assert seconds * SAMPLE_RATE < 64_300.5

    _, samples = _final_duration_seconds_to_samples(seconds, SAMPLE_RATE)

    assert samples == 64_301


def test_decimal_half_up_is_independent_of_the_ambient_decimal_context():
    with localcontext() as context:
        context.prec = 5
        context.rounding = ROUND_DOWN

        _, samples = _final_duration_seconds_to_samples(2.6791875, SAMPLE_RATE)

    assert samples == 64_301


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("1.0", TypeError),
    ],
)
def test_invalid_final_duration_seconds_fail_explicitly(value, error_type):
    with pytest.raises(error_type, match="final_duration"):
        _final_duration_seconds_to_samples(value, SAMPLE_RATE)


def test_seconds_that_round_below_one_sample_are_rejected():
    with pytest.raises(ValueError, match="fewer than one sample"):
        _final_duration_seconds_to_samples(0.000001, SAMPLE_RATE)


def test_scalar_and_mixed_per_item_targets_are_normalized():
    targets = _normalize_final_duration_targets(
        final_duration=[2.6791875, None, 1.0],
        final_duration_samples=[None, 12_345, None],
        batch_size=3,
        sampling_rate=SAMPLE_RATE,
    )

    assert targets is not None
    assert [target.samples if target else None for target in targets] == [
        64_301,
        12_345,
        24_000,
    ]
    assert [target.source if target else None for target in targets] == [
        "decimal_seconds",
        "integer_samples",
        "decimal_seconds",
    ]


def test_scalar_authoritative_samples_repeat_across_batch():
    targets = _normalize_final_duration_targets(
        None,
        np.int64(64_301),
        batch_size=2,
        sampling_rate=SAMPLE_RATE,
    )

    assert targets is not None
    assert [target.samples for target in targets if target] == [64_301, 64_301]


@pytest.mark.parametrize("invalid", [True, 1.5, "10", 0, -1])
def test_invalid_authoritative_sample_targets_identify_batch_index(invalid):
    with pytest.raises((TypeError, ValueError), match=r"final_duration_samples\[1\]"):
        _normalize_final_duration_targets(
            None,
            [10, invalid],
            batch_size=2,
            sampling_rate=SAMPLE_RATE,
        )


def test_authoritative_sample_target_rejects_platform_overflow():
    with pytest.raises(ValueError, match="platform sample-count limit"):
        _normalize_final_duration_targets(
            None,
            np.iinfo(np.intp).max + 1,
            batch_size=1,
            sampling_rate=SAMPLE_RATE,
        )


def test_seconds_and_samples_are_mutually_exclusive_per_item():
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        _normalize_final_duration_targets(
            [1.0, None],
            [24_000, 10],
            batch_size=2,
            sampling_rate=SAMPLE_RATE,
        )


@pytest.mark.parametrize("name", ["final_duration", "final_duration_samples"])
def test_target_list_length_must_match_batch(name):
    kwargs = {"final_duration": None, "final_duration_samples": None}
    kwargs[name] = [1]

    with pytest.raises(ValueError, match="exactly one value per text item"):
        _normalize_final_duration_targets(
            batch_size=2,
            sampling_rate=SAMPLE_RATE,
            **kwargs,
        )


def test_exact_length_is_a_true_noop():
    audio = np.array([[0.0, 0.25, -0.5, 0.0]], dtype=np.float32)

    result = fit_audio_to_target_samples(audio, audio.shape[-1])

    assert result[0] is audio
    assert result[1:] == (0, 0, 0, 0)


def test_underflow_appends_digital_silence_to_the_free_trailing_edge():
    audio = np.array([[0.25, -0.5]], dtype=np.float32)

    fitted, pad_lead, pad_trail, trim_lead, trim_trail = fit_audio_to_target_samples(
        audio, 5
    )

    np.testing.assert_array_equal(fitted[..., :2], audio)
    np.testing.assert_array_equal(fitted[..., 2:], np.zeros((1, 3), np.float32))
    assert (pad_lead, pad_trail, trim_lead, trim_trail) == (0, 3, 0, 0)


def test_unprotected_underflow_avoids_activity_scan_and_full_index_allocations(
    monkeypatch,
):
    audio = np.array([[0.25, -0.5]], dtype=np.float32)

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "the unprotected underflow path must not scan or concatenate"
        )

    monkeypatch.setattr(audio_utils.np, "any", forbidden)
    monkeypatch.setattr(audio_utils.np, "flatnonzero", forbidden)
    monkeypatch.setattr(audio_utils.np, "concatenate", forbidden)

    fitted, pad_lead, pad_trail, trim_lead, trim_trail = fit_audio_to_target_samples(
        audio, 5
    )

    assert fitted.shape == (1, 5)
    assert float(fitted[0, 0]) == pytest.approx(0.25)
    assert float(fitted[0, 1]) == pytest.approx(-0.5)
    assert fitted[0, 2:].tolist() == [0.0, 0.0, 0.0]
    assert (pad_lead, pad_trail, trim_lead, trim_trail) == (0, 3, 0, 0)


def test_underflow_never_moves_onset_when_trailing_target_is_protected():
    audio = np.array([[0.25, -0.5, 0.0]], dtype=np.float32)

    fitted, pad_lead, pad_trail, trim_lead, trim_trail = fit_audio_to_target_samples(
        audio, 5, protect_trailing_edge=True
    )

    np.testing.assert_array_equal(fitted[..., : audio.shape[-1]], audio)
    np.testing.assert_array_equal(fitted[..., audio.shape[-1] :], 0.0)
    assert (pad_lead, pad_trail, trim_lead, trim_trail) == (0, 2, 0, 0)


@pytest.mark.parametrize(
    ("protect_leading_edge", "protect_trailing_edge"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_underflow_always_appends_without_displacing_source(
    protect_leading_edge,
    protect_trailing_edge,
):
    audio = np.array([[0.25, -0.5, 0.0]], dtype=np.float32)

    fitted, pad_lead, pad_trail, trim_lead, trim_trail = fit_audio_to_target_samples(
        audio,
        7,
        protect_leading_edge=protect_leading_edge,
        protect_trailing_edge=protect_trailing_edge,
    )

    np.testing.assert_array_equal(fitted[..., : audio.shape[-1]], audio)
    np.testing.assert_array_equal(fitted[..., audio.shape[-1] :], 0.0)
    assert (pad_lead, pad_trail, trim_lead, trim_trail) == (0, 4, 0, 0)


def test_underflow_with_both_edge_anchors_appends_outer_container_fill():
    audio = np.array([[0.25, -0.5]], dtype=np.float32)

    fitted, pad_lead, pad_trail, trim_lead, trim_trail = fit_audio_to_target_samples(
        audio,
        3,
        protect_leading_edge=True,
        protect_trailing_edge=True,
    )

    np.testing.assert_array_equal(fitted[..., : audio.shape[-1]], audio)
    np.testing.assert_array_equal(fitted[..., audio.shape[-1] :], 0.0)
    assert (pad_lead, pad_trail, trim_lead, trim_trail) == (0, 1, 0, 0)


def test_overflow_trims_exact_trailing_zeros_before_leading_zeros():
    audio = np.array([[0.0, 0.0, 0.25, -0.5, 0.0, 0.0, 0.0]], dtype=np.float32)

    fitted, pad_lead, pad_trail, trim_lead, trim_trail = fit_audio_to_target_samples(
        audio, 3
    )

    np.testing.assert_array_equal(
        fitted,
        np.array([[0.0, 0.25, -0.5]], dtype=np.float32),
    )
    assert (pad_lead, pad_trail, trim_lead, trim_trail) == (0, 0, 1, 3)


def test_overflow_never_cuts_an_active_or_sub_pcm_sample():
    audio = np.array([[1e-12, 0.25, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="trimming 1 active samples is forbidden"):
        fit_audio_to_target_samples(audio, 1)


def test_multichannel_frame_is_active_when_any_channel_is_nonzero():
    audio = np.array([[0.0, 0.0], [1e-12, 0.0]], dtype=np.float32)

    fitted, *_ = fit_audio_to_target_samples(audio, 1)

    np.testing.assert_array_equal(fitted, audio[..., :1])


def test_overflow_cannot_consume_a_protected_edge():
    audio = np.array([[0.25, -0.5, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="only 0 removable"):
        fit_audio_to_target_samples(audio, 3, protect_trailing_edge=True)


def test_all_silent_audio_can_be_resized_without_edge_targets():
    audio = np.zeros((2, 5), dtype=np.float32)

    shorter, *_, trimmed_trailing = fit_audio_to_target_samples(audio, 3)
    longer, padded_leading, padded_trailing, *_ = fit_audio_to_target_samples(audio, 7)

    assert shorter.shape == (2, 3)
    assert trimmed_trailing == 2
    assert longer.shape == (2, 7)
    assert (padded_leading, padded_trailing) == (0, 2)


def test_all_silent_audio_cannot_claim_an_exact_edge_target():
    with pytest.raises(ValueError, match="without an active sample"):
        fit_audio_to_target_samples(
            np.zeros((1, 2), dtype=np.float32),
            3,
            protect_leading_edge=True,
        )


@pytest.mark.parametrize(
    ("protect_leading_edge", "protect_trailing_edge"),
    [(True, False), (False, True), (True, True)],
)
def test_exact_length_all_silent_audio_cannot_claim_a_protected_edge(
    protect_leading_edge, protect_trailing_edge
):
    audio = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="without an active sample"):
        fit_audio_to_target_samples(
            audio,
            audio.shape[-1],
            protect_leading_edge=protect_leading_edge,
            protect_trailing_edge=protect_trailing_edge,
        )


def test_new_public_parameters_are_appended_for_positional_compatibility():
    signature = inspect.signature(OmniVoice.generate)
    names = list(signature.parameters)

    assert names[-4:] == [
        "final_duration",
        "final_duration_samples",
        "framing_observer",
        "kwargs",
    ]
    assert names.index("duration") < names.index("generation_config")


def test_telemetry_observer_field_preserves_existing_positional_construction():
    record = OmniVoiceGenerationTelemetry(
        1,
        1,
        "raw_codec",
        None,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        0.25,
        (),
    )

    assert record.output_framing_seconds == 0.25
    assert record.final_duration_items == ()
    assert record.framing_observer_seconds == 0.0


def test_invalid_sample_rate_is_rejected_without_float_coercion():
    for invalid in (True, 0, -1, 24_000.0):
        with pytest.raises((TypeError, ValueError), match="sampling_rate"):
            _final_duration_seconds_to_samples(1.0, invalid)


def test_positive_infinite_target_is_rejected_before_decimal_conversion():
    with pytest.raises(ValueError, match="finite"):
        _final_duration_seconds_to_samples(math.inf, SAMPLE_RATE)
