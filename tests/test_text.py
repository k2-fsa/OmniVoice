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

"""Tests for long-text chunking in ``omnivoice.utils.text``."""

from omnivoice.utils.text import chunk_text_punctuation


def test_chunking_keeps_numbers_intact():
    text = (
        "Revenue grew by 12.5 percent to 3,400 million dollars. "
        "Doors open at 10:30 and the dividend is 0.75 dollars per share."
    )
    for chunk_len in (10, 20, 30, 40, 60, 80):
        chunks = chunk_text_punctuation(text, chunk_len=chunk_len, min_chunk_len=3)
        for number in ("12.5", "3,400", "10:30", "0.75"):
            assert any(number in chunk for chunk in chunks), (chunk_len, chunks)


def test_chunking_still_splits_after_numbers():
    chunks = chunk_text_punctuation(
        "It costs 5. Then it costs 6, or 7.", chunk_len=5, min_chunk_len=None
    )
    assert chunks == ["It costs 5.", "Then it costs 6,", "or 7."]
