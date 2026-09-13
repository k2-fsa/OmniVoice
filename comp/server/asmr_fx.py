"""ASMR post-processing for synthesized speech (numpy/scipy, no GPU).

The chain mirrors what a commercial companion app's ASMR style sounds like (close-mic whisper,
warm, a touch of room): high-pass rumble cut -> gentle low-pass "warmth" ->
short Schroeder reverb with HF damping -> stereo placement (constant-power
pan or slow L<->R drift) -> optional silence padding at sentence ends.

    from asmr_fx import asmr_pipeline
    y, sr = asmr_pipeline(x, sr, preset="close")

Presets: "close" (intimate, dry), "room" (more reverb), "drift" (slow pan).
Pure functions on float32 arrays in [-1, 1]; mono in, stereo out.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt


def _sos(kind: str, cutoff: float, sr: float, order: int = 2):
    return butter(order, cutoff / (sr / 2.0), btype=kind, output="sos")


def highpass(x: np.ndarray, sr: float, cutoff: float = 70.0) -> np.ndarray:
    return sosfilt(_sos("high", cutoff, sr), x).astype(np.float32)


def lowpass(x: np.ndarray, sr: float, cutoff: float = 7500.0) -> np.ndarray:
    return sosfilt(_sos("low", cutoff, sr), x).astype(np.float32)


def _comb(x: np.ndarray, delay: int, feedback: float, damp: float) -> np.ndarray:
    """Feedback comb with a one-pole low-pass in the loop (HF damping)."""
    y = np.zeros_like(x)
    buf = np.zeros(delay, dtype=np.float32)
    idx = 0
    store = 0.0
    for i in range(len(x)):
        out = buf[idx]
        store = out * (1.0 - damp) + store * damp
        buf[idx] = x[i] + store * feedback
        y[i] = out
        idx = (idx + 1) % delay
    return y


def _allpass(x: np.ndarray, delay: int, g: float = 0.5) -> np.ndarray:
    y = np.zeros_like(x)
    buf = np.zeros(delay, dtype=np.float32)
    idx = 0
    for i in range(len(x)):
        d = buf[idx]
        out = -g * x[i] + d
        buf[idx] = x[i] + g * d
        y[i] = out
        idx = (idx + 1) % delay
    return y


def reverb(
    x: np.ndarray,
    sr: float,
    mix: float = 0.12,
    decay: float = 0.72,
    damp: float = 0.35,
    size: float = 0.6,
) -> np.ndarray:
    """Schroeder: 4 parallel combs + 2 series allpasses. `size` scales the
    delays (0.6 = small room). Returns wet/dry mix, mono."""
    base = np.array([1116, 1188, 1277, 1356]) * (sr / 44100.0) * size
    wet = np.zeros_like(x)
    for d in base:
        wet += _comb(x, max(8, int(d)), decay, damp)
    wet /= len(base)
    for d in (556, 441):
        wet = _allpass(wet, max(8, int(d * sr / 44100.0)))
    return ((1.0 - mix) * x + mix * wet).astype(np.float32)


def to_stereo(x: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """Constant-power pan, -1 (left) .. 1 (right). Returns (n, 2)."""
    theta = (pan + 1.0) * np.pi / 4.0
    return np.stack([x * np.cos(theta), x * np.sin(theta)], axis=1).astype(np.float32)


def drift_stereo(
    x: np.ndarray, sr: float, period_s: float = 9.0, depth: float = 0.6
) -> np.ndarray:
    """Slow left<->right movement, the classic ASMR 'walking around you'."""
    t = np.arange(len(x)) / sr
    pan = depth * np.sin(2 * np.pi * t / period_s)
    theta = (pan + 1.0) * np.pi / 4.0
    return np.stack([x * np.cos(theta), x * np.sin(theta)], axis=1).astype(np.float32)


def normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(y))) or 1.0
    return (y * (peak / m)).astype(np.float32)


PRESETS = {
    "close": dict(
        hp=80.0, lp=6500.0, mix=0.08, decay=0.6, size=0.45, pan=0.0, drift=False
    ),
    "room": dict(
        hp=70.0, lp=7500.0, mix=0.18, decay=0.78, size=0.7, pan=0.0, drift=False
    ),
    "drift": dict(
        hp=80.0, lp=6500.0, mix=0.10, decay=0.65, size=0.5, pan=0.0, drift=True
    ),
}


def asmr_pipeline(
    x: np.ndarray, sr: float, preset: str = "close", gain_db: float = -1.0
) -> tuple[np.ndarray, float]:
    p = PRESETS[preset]
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    y = highpass(x, sr, p["hp"])
    y = lowpass(y, sr, p["lp"])
    y = reverb(y, sr, mix=p["mix"], decay=p["decay"], size=p["size"])
    st = drift_stereo(y, sr) if p["drift"] else to_stereo(y, p["pan"])
    st = normalize(st) * (10 ** (gain_db / 20.0))
    return st.astype(np.float32), sr


if __name__ == "__main__":
    import sys
    import soundfile as sf

    src, dst = sys.argv[1], sys.argv[2]
    preset = sys.argv[3] if len(sys.argv) > 3 else "close"
    x, sr = sf.read(src, dtype="float32")
    y, sr = asmr_pipeline(x, sr, preset)
    sf.write(dst, y, sr)
    print("wrote", dst, preset)
