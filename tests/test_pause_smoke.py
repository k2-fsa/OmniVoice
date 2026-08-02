from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.pause_smoke import (
    find_boundary_anchors,
    low_energy_measurement,
    merge_windows,
    normalized_words,
)


def test_normalized_words_preserves_contractions_and_order():
    assert normalized_words("It's ordinary. But timing matters!") == [
        "it's",
        "ordinary",
        "but",
        "timing",
        "matters",
    ]


def test_boundary_anchor_mapping_uses_expected_word_sequence():
    text = "The first result. The second test."
    offset = text.index(". ") + 1
    asr = {
        "words": [
            {"normalized": word, "start": i * 0.2, "end": i * 0.2 + 0.1}
            for i, word in enumerate(normalized_words(text))
        ]
    }
    anchors = find_boundary_anchors(text, offset, asr)
    assert anchors["expected_preceding"] == "result"
    assert anchors["expected_following"] == "the"
    assert anchors["preceding_word"]["normalized"] == "result"
    assert anchors["following_word"]["normalized"] == "the"


def test_low_energy_measurement_finds_longest_10ms_run(tmp_path: Path):
    sample_rate = 24000
    audio = np.concatenate(
        (
            np.full(int(0.2 * sample_rate), 0.1, dtype=np.float32),
            np.zeros(int(0.4 * sample_rate), dtype=np.float32),
            np.full(int(0.2 * sample_rate), 0.1, dtype=np.float32),
        )
    )
    path = tmp_path / "pause.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    anchors = {
        "preceding_word": {"end": 0.1},
        "following_word": {"start": 0.7},
    }
    measurement = low_energy_measurement(path, anchors)
    assert measurement["longest_low_energy_seconds"] == pytest.approx(0.4)
    assert measurement["asr_word_gap_seconds"] == pytest.approx(0.6)


def test_transition_windows_merge_overlaps_and_touching_edges():
    assert merge_windows([(8, 12), (3, 5), (5, 9), (20, 20)]) == [(3, 12)]
