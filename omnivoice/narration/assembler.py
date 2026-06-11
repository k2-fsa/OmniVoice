from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def _as_float_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.reshape(-1)


def apply_edge_fade(audio: np.ndarray, sample_rate: int, fade_ms: int = 8) -> np.ndarray:
    arr = _as_float_mono(audio).copy()
    fade_len = min(len(arr) // 2, int(sample_rate * fade_ms / 1000))
    if fade_len <= 1:
        return arr
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    arr[:fade_len] *= ramp
    arr[-fade_len:] *= ramp[::-1]
    return arr


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    arr = _as_float_mono(audio)
    max_abs = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if max_abs <= 0 or max_abs <= peak:
        return arr
    return arr * (peak / max_abs)


def assemble_segments(
    audio_segments: Sequence[np.ndarray],
    pauses_ms: Sequence[int],
    sample_rate: int,
    fade_ms: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    if len(audio_segments) != len(pauses_ms):
        raise ValueError("audio_segments and pauses_ms must have the same length")

    pieces: List[np.ndarray] = []
    for audio, pause_ms in zip(audio_segments, pauses_ms):
        pieces.append(apply_edge_fade(audio, sample_rate, fade_ms=fade_ms))
        pause_len = max(0, int(sample_rate * int(pause_ms) / 1000))
        if pause_len:
            pieces.append(np.zeros(pause_len, dtype=np.float32))

    if not pieces:
        return np.zeros(0, dtype=np.float32)

    final = np.concatenate(pieces).astype(np.float32)
    return normalize_peak(final) if normalize else final


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    arr = normalize_peak(_as_float_mono(audio), peak=0.999)
    return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)


def audio_duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return len(_as_float_mono(audio)) / float(sample_rate)


def total_pause_seconds(pauses_ms: Iterable[int]) -> float:
    return sum(max(0, int(pause)) for pause in pauses_ms) / 1000.0
