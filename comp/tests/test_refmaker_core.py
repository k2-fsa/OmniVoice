"""Unit tests for refmaker_core — the pure pipeline functions.

Synthesizes a WAV of alternating tone + silence so candidate detection and
clip export can be verified without ffmpeg/demucs (those are integration-tested
with a real file manually).

Run from the OmniVoice root:
    .venv/Scripts/python.exe -m pytest tests/test_refmaker_core.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "gui"))
import refmaker_core as core  # noqa: E402

SR = 16000


def _tone(seconds: float, freq: float = 200.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


@pytest.fixture
def speech_wav(tmp_path: Path) -> Path:
    """30s file: 10s tone, 2s silence, 10s tone, 2s silence, 6s tone."""
    y = np.concatenate([_tone(10), _silence(2), _tone(10), _silence(2), _tone(6)])
    path = tmp_path / "speech.wav"
    sf.write(str(path), y, SR, subtype="PCM_16")
    return path


def test_find_candidates_locates_tone_regions(speech_wav: Path):
    cands = core.find_candidates(speech_wav, clip_len_s=8.0)
    assert cands, "expected at least one candidate"
    # Every candidate must sit inside a tone region, never inside the silences
    # at [10,12] or [22,24].
    for c in cands:
        mid = (c.start_s + c.end_s) / 2
        assert not (10 < mid < 12), f"candidate centered in silence: {c}"
        assert not (22 < mid < 24), f"candidate centered in silence: {c}"
    # Best candidate should be (near-)fully voiced -> high score.
    assert cands[0].score > 0.8


def test_find_candidates_returns_sorted_desc(speech_wav: Path):
    cands = core.find_candidates(speech_wav, clip_len_s=8.0)
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_find_candidates_empty_for_short_input(tmp_path: Path):
    path = tmp_path / "short.wav"
    sf.write(str(path), _tone(3), SR, subtype="PCM_16")
    assert core.find_candidates(path, clip_len_s=8.0) == []


def test_export_clip_is_mono_target_sr_and_length(speech_wav: Path, tmp_path: Path):
    out = tmp_path / "clip.wav"
    core.export_clip(speech_wav, start_s=1.0, end_s=9.0, out_path=out, target_sr=24000)
    data, sr = sf.read(str(out))
    assert sr == 24000
    assert data.ndim == 1  # mono
    assert abs(len(data) / sr - 8.0) < 0.05  # ~8s


def test_export_clip_peak_normalized(speech_wav: Path, tmp_path: Path):
    out = tmp_path / "clip.wav"
    core.export_clip(speech_wav, 0.0, 8.0, out, target_sr=24000, peak=0.97)
    data, _ = sf.read(str(out))
    assert 0.9 < float(np.max(np.abs(data))) <= 0.971


def test_format_ts():
    assert core.format_ts(0) == "00:00.0"
    assert core.format_ts(83.4) == "01:23.4"


# --------------------------------------------------------------------------- #
# find_speech_intervals / merge_speech
# --------------------------------------------------------------------------- #
def test_find_speech_intervals_three_spans(speech_wav: Path):
    # speech_wav: tone[0,10] sil[10,12] tone[12,22] sil[22,24] tone[24,30].
    # 2s gaps are well above merge_gap_s, so the three spans stay separate.
    ivals = core.find_speech_intervals(speech_wav)
    assert len(ivals) == 3
    starts = [s for s, _ in ivals]
    assert starts == sorted(starts)  # chronological
    # Roughly the right placement (silence edges may shift a touch).
    assert ivals[0][0] < 1.0 and 9.0 < ivals[0][1] < 11.0
    assert 11.0 < ivals[1][0] < 13.0
    assert ivals[2][1] > 29.0


def test_find_speech_intervals_drops_tiny_blip(tmp_path: Path):
    y = np.concatenate([_silence(1), _tone(0.1), _silence(1), _tone(5)])
    path = tmp_path / "blip.wav"
    sf.write(str(path), y, SR, subtype="PCM_16")
    ivals = core.find_speech_intervals(path, min_seg_s=0.3)
    # The 0.1s blip is dropped; only the 5s span survives.
    assert len(ivals) == 1
    assert (ivals[0][1] - ivals[0][0]) > 4.0


def test_merge_speech_capped_stops_near_target(speech_wav: Path, tmp_path: Path):
    out = tmp_path / "merged.wav"
    _, length = core.merge_speech(
        out_path=out, source_path=speech_wav, target_len_s=15.0
    )
    data, sr = sf.read(str(out))
    assert sr == core.DEFAULT_TARGET_SR
    assert data.ndim == 1  # mono
    # Accumulates span 1 (10s) then span 2 (10s) -> ~20s, then stops. Never the
    # full ~26s. Allow headroom for padding minus crossfades.
    assert 15.0 <= length <= 24.0
    assert abs(len(data) / sr - length) < 0.05


def test_merge_speech_full_is_longer(speech_wav: Path, tmp_path: Path):
    capped = tmp_path / "capped.wav"
    full = tmp_path / "full.wav"
    _, capped_len = core.merge_speech(speech_wav, capped, target_len_s=15.0, full=False)
    _, full_len = core.merge_speech(speech_wav, full, target_len_s=15.0, full=True)
    # Full keeps all three spans (~26s of voiced audio); capped stops early.
    assert full_len > capped_len
    assert full_len > 24.0


def test_merge_speech_peak_normalized(speech_wav: Path, tmp_path: Path):
    out = tmp_path / "merged.wav"
    core.merge_speech(speech_wav, out, peak=0.97, full=True)
    data, _ = sf.read(str(out))
    assert 0.9 < float(np.max(np.abs(data))) <= 0.971


def test_merge_speech_uses_supplied_intervals(speech_wav: Path, tmp_path: Path):
    # Pass an explicit span (mimics a VAD result) — only that 3s gets merged,
    # bypassing the energy split that would otherwise find ~26s.
    out = tmp_path / "merged.wav"
    _, length = core.merge_speech(speech_wav, out, intervals=[(2.0, 5.0)], full=True)
    assert 2.5 < length < 3.6  # ~3s span + padding


def test_merge_speech_raises_on_silence(tmp_path: Path):
    path = tmp_path / "silent.wav"
    sf.write(str(path), _silence(5), SR, subtype="PCM_16")
    with pytest.raises(RuntimeError):
        core.merge_speech(path, tmp_path / "out.wav")
