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

"""Tests for converting duration seconds to audio-token budgets."""

import math
from types import SimpleNamespace

import pytest

from omnivoice.models.omnivoice import OmniVoice, _duration_to_target_tokens


FRAME_RATE = 25


class _PreprocessStub:
    """Minimal receiver for exercising ``OmniVoice._preprocess_all`` on CPU."""

    audio_tokenizer = SimpleNamespace(config=SimpleNamespace(frame_rate=FRAME_RATE))
    _ensure_list = OmniVoice._ensure_list

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        del text, ref_text, num_ref_audio_tokens, speed
        return 40


def _preprocess(text, duration):
    return OmniVoice._preprocess_all(
        _PreprocessStub(),
        text=text,
        duration=duration,
    )


def test_duration_ratio_on_integer_boundary_keeps_all_29_tokens():
    duration = 29 / FRAME_RATE
    scaled = duration * FRAME_RATE

    assert scaled < 29
    assert 29 - scaled == math.ulp(scaled)
    assert _duration_to_target_tokens(duration, FRAME_RATE) == 29


def test_exact_duration_ratio_keeps_all_30_tokens():
    assert _duration_to_target_tokens(30 / FRAME_RATE, FRAME_RATE) == 30


def test_duration_more_than_one_ulp_below_boundary_preserves_floor_semantics():
    duration = math.nextafter(29 / FRAME_RATE, -math.inf)
    scaled = duration * FRAME_RATE

    assert 29 - scaled > math.ulp(scaled)
    assert _duration_to_target_tokens(duration, FRAME_RATE) == 28


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.001, 1),
        (1.1, 27),
        (1.18, 29),
        (2.999, 74),
    ],
)
def test_genuinely_fractional_token_budgets_still_use_floor(duration, expected):
    assert _duration_to_target_tokens(duration, FRAME_RATE) == expected


def test_per_item_duration_list_uses_boundary_fix_and_preserves_none():
    task = _preprocess(
        ["twenty nine", "thirty", "estimated"],
        [29 / FRAME_RATE, 30 / FRAME_RATE, None],
    )

    assert task.target_lens == [29, 30, 40]
    assert task.speed == [40 / 29, 40 / 30, 1.0]


def test_scalar_duration_is_repeated_across_batch():
    task = _preprocess(["first", "second"], 29 / FRAME_RATE)

    assert task.target_lens == [29, 29]
    assert task.speed == [40 / 29, 40 / 29]


@pytest.mark.parametrize(
    ("duration", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
        (True, TypeError),
        ("1.16", TypeError),
        (object(), TypeError),
    ],
)
def test_invalid_scalar_durations_fail_explicitly(duration, error_type):
    with pytest.raises(error_type, match="duration"):
        _duration_to_target_tokens(duration, FRAME_RATE)


@pytest.mark.parametrize(
    ("frame_rate", "error_type"),
    [
        (0, ValueError),
        (-25, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("25", TypeError),
    ],
)
def test_invalid_frame_rates_fail_explicitly(frame_rate, error_type):
    with pytest.raises(error_type, match="frame_rate"):
        _duration_to_target_tokens(1.0, frame_rate)


def test_batch_duration_length_must_match_text_count():
    with pytest.raises(ValueError, match="exactly one value per text item"):
        _preprocess(["first", "second"], [29 / FRAME_RATE])


@pytest.mark.parametrize("invalid", [True, "1.2", float("nan"), -1.0])
def test_invalid_per_item_duration_identifies_its_batch_index(invalid):
    with pytest.raises((TypeError, ValueError), match=r"duration\[1\]"):
        _preprocess(["first", "second"], [29 / FRAME_RATE, invalid])
