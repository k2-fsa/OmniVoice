"""Derive an ASMR (whisper) reference clip from a normal reference, and measure
what separates "normal" from "ASMR" delivery on real paired recordings.

    # 1. learn the delta from paired clips (same lines, normal vs whisper)
    python server/asmr_ref.py analyze PAIRS_DIR_NORMAL PAIRS_DIR_ASMR
    # 2. derive a whisper reference from a normal one
    python server/asmr_ref.py make ref/default_ref.wav ref/default_asmr.wav --method lpc|tts|both

Measured features (per clip, then normal->asmr ratio/delta):
  voicing   fraction of frames with a detected F0 (whisper -> ~0)
  f0        median F0 of voiced frames (Hz)
  rms       loudness (dBFS)
  centroid  spectral centroid (Hz) - brightness
  tilt      spectral tilt: energy above 4 kHz vs 0.3-4 kHz (dB) - breathiness
  hnr       harmonic-to-noise proxy (dB) from the autocorrelation peak
  rate      speaking rate proxy: syllable-ish onsets per second
  dur       clip duration (s)

Two derivation methods:
  lpc   classic whisperization - LPC analysis of the normal clip, re-synthesis
        with noise excitation (voicing removed, formants/timbre kept), then the
        measured spectral tilt + loudness delta is applied.
  tts   self-bootstrap - the running OmniVoice server renders a neutral
        passage with instruct "whisper" from the normal ref; that output
        becomes the reference (the model's own idea of this voice whispering).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt, resample_poly

SR = 24000


# ------------------------------------------------------------------ io
def load(path: str, sr: int = SR) -> np.ndarray:
    x, s = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if s != sr:
        from math import gcd

        g = gcd(sr, s)
        x = resample_poly(x, sr // g, s // g).astype(np.float32)
    return x


# ------------------------------------------------------------------ features
def _frames(x: np.ndarray, n: int = 1024, hop: int = 256):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i : i + n] for i in idx]) * np.hanning(n)


def _f0_and_hnr(frame: np.ndarray, sr: int, fmin=70.0, fmax=500.0):
    """Autocorrelation pitch + harmonicity proxy for one frame."""
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0, -60.0
    f = frame - frame.mean()
    ac = np.correlate(f, f, mode="full")[len(f) - 1 :]
    ac /= ac[0] + 1e-9
    lo, hi = int(sr / fmax), int(sr / fmin)
    seg = ac[lo:hi]
    if not len(seg):
        return 0.0, -60.0
    k = int(np.argmax(seg))
    r = float(seg[k])
    hnr = 10 * np.log10(max(r, 1e-6) / max(1 - r, 1e-6))
    f0 = sr / (lo + k) if r > 0.55 else 0.0
    return f0, hnr


def features(x: np.ndarray, sr: int = SR) -> dict:
    fr = _frames(x)
    rms_f = np.sqrt((fr**2).mean(axis=1)) + 1e-9
    active = rms_f > rms_f.max() * 0.08
    f0s, hnrs = [], []
    for i in np.where(active)[0]:
        f0, h = _f0_and_hnr(fr[i], sr)
        f0s.append(f0)
        hnrs.append(h)
    f0s = np.array(f0s)
    hnrs = np.array(hnrs)
    voiced = f0s[f0s > 0]
    spec = np.abs(np.fft.rfft(fr[active], axis=1)) ** 2
    freqs = np.fft.rfftfreq(fr.shape[1], 1 / sr)
    p = spec.mean(axis=0) + 1e-12
    centroid = float((freqs * p).sum() / p.sum())
    lo = p[(freqs >= 300) & (freqs < 4000)].sum()
    hi = p[(freqs >= 4000) & (freqs < 10000)].sum()
    tilt = 10 * np.log10(hi / lo)
    # onsets: rises in the RMS envelope
    env = rms_f / rms_f.max()
    d = np.diff(env)
    onsets = int(((d[1:] > 0.08) & (d[:-1] <= 0.08)).sum())
    dur = len(x) / sr
    return {
        "voicing": round(float((f0s > 0).mean()) if len(f0s) else 0.0, 3),
        "f0": round(float(np.median(voiced)) if len(voiced) else 0.0, 1),
        "rms": round(float(20 * np.log10(np.sqrt((x**2).mean()) + 1e-9)), 1),
        "centroid": round(centroid, 0),
        "tilt": round(float(tilt), 1),
        "hnr": round(float(np.median(hnrs)) if len(hnrs) else -60.0, 1),
        "rate": round(onsets / dur, 2),
        "dur": round(dur, 2),
    }


def analyze(normal_dir: str, asmr_dir: str) -> dict:
    """Pair clips by relative path (same file name = same line)."""
    pairs = []
    for root, _d, files in os.walk(normal_dir):
        for f in files:
            if not f.lower().endswith((".wav", ".m4a", ".mp3", ".flac")):
                continue
            rel = os.path.relpath(os.path.join(root, f), normal_dir)
            other = os.path.join(asmr_dir, rel)
            if os.path.isfile(other):
                pairs.append((os.path.join(root, f), other, rel))
    rows = []
    for a, b, rel in pairs:
        fa, fb = features(_load_any(a)), features(_load_any(b))
        rows.append({"clip": rel, "normal": fa, "asmr": fb})
    keys = ["voicing", "f0", "rms", "centroid", "tilt", "hnr", "rate", "dur"]
    delta = {}
    for k in keys:
        na = np.array([r["normal"][k] for r in rows], dtype=float)
        aa = np.array([r["asmr"][k] for r in rows], dtype=float)
        delta[k] = {
            "normal": round(float(na.mean()), 2),
            "asmr": round(float(aa.mean()), 2),
            "delta": round(float((aa - na).mean()), 2),
        }
    return {"pairs": len(rows), "delta": delta, "rows": rows}


def _load_any(path: str) -> np.ndarray:
    if path.lower().endswith(".m4a") or path.lower().endswith(".mp3"):
        import subprocess
        import tempfile

        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                path,
                "-ar",
                str(SR),
                "-ac",
                "1",
                tmp,
            ],
            check=True,
        )
        x = load(tmp)
        os.unlink(tmp)
        return x
    return load(path)


# ------------------------------------------------------------------ derivation
def _lpc(frame: np.ndarray, order: int) -> np.ndarray:
    """Autocorrelation LPC (Levinson-Durbin). Returns a[0..order], a[0]=1."""
    r = np.correlate(frame, frame, "full")[len(frame) - 1 : len(frame) + order]
    if r[0] < 1e-9:
        return np.r_[1.0, np.zeros(order)]
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])
        k = -acc / e
        a_new = a.copy()
        a_new[1:i] = a[1:i] + k * a[i - 1 : 0 : -1]
        a_new[i] = k
        a = a_new
        e *= 1 - k * k
        if e <= 1e-12:
            break
    return a


def whisperize_lpc(
    x: np.ndarray,
    sr: int = SR,
    order: int = 24,
    n: int = 2048,
    hop: int = 512,
    breath: float = 1.0,
    voice_mix: float = 0.3,
) -> np.ndarray:
    """Replace the voiced excitation with shaped noise; keep the vocal tract
    (formants) so it still sounds like the same person, whispering.

    voice_mix blends a little of the original voiced signal back in (the
    real whisper keeps ~14% voiced frames; pure noise reads as a ghost).
    Longer windows than the analysis ones keep the noise floor from
    crackling frame to frame."""
    win = np.hanning(n)
    out = np.zeros(len(x) + n)
    norm = np.zeros(len(x) + n)
    rng = np.random.default_rng(7)
    pre = lfilter([1, -0.97], [1], x)  # pre-emphasis for a cleaner LPC fit
    for i in range(0, len(x) - n, hop):
        fr = pre[i : i + n] * win
        energy = np.sqrt((fr**2).mean())
        if energy < 1e-5:
            continue
        a = _lpc(fr, order)
        noise = rng.standard_normal(n)
        syn = lfilter([1.0], a, noise)
        syn = lfilter([1.0], [1, -0.97], syn)  # de-emphasis
        syn *= energy * breath / (np.sqrt((syn**2).mean()) + 1e-9)
        out[i : i + n] += syn * win
        norm[i : i + n] += win**2
    out = out[: len(x)] / np.maximum(norm[: len(x)], 1e-3)
    # breath noise lives ~1-8 kHz; tame the top so it is air, not hiss
    out = sosfilt(butter(2, 8500 / (sr / 2), btype="low", output="sos"), out)
    if voice_mix > 0:
        # match loudness before blending so the mix ratio means what it says
        ro = np.sqrt((out**2).mean()) + 1e-9
        rx = np.sqrt((x**2).mean()) + 1e-9
        out = (1.0 - voice_mix) * out + voice_mix * x * (ro / rx)
    return out.astype(np.float32)


def apply_delta(y: np.ndarray, sr: int, delta: dict | None) -> np.ndarray:
    """Match the measured normal->asmr tilt and loudness shift (from analyze)."""
    if not delta:
        return y
    cur = features(y, sr)["tilt"]
    tilt_db = (
        float(delta.get("tilt", {}).get("asmr", cur)) - cur
    )  # reach the measured whisper tilt
    if abs(tilt_db) > 0.5:
        # one shelving stage: boost/cut above 4 kHz by tilt_db
        sos = butter(2, 4000 / (sr / 2), btype="high", output="sos")
        hi = sosfilt(sos, y)
        g = 10 ** (tilt_db / 20.0) - 1.0
        y = (y + g * hi).astype(np.float32)
    # level: noise excitation is spiky, so normalise by RMS (not peak) to the
    # measured whisper loudness, then soft-limit the peaks
    target_db = float(delta.get("rms", {}).get("asmr", -24.0))
    rms = float(np.sqrt((y**2).mean())) + 1e-9
    y = y * (10 ** (target_db / 20.0) / rms)
    y = np.tanh(y * 1.2) / 1.2
    return y.astype(np.float32)


def make_tts_ref(
    normal_ref: str, out_path: str, server: str, text: str, language: str = "ja"
) -> str:
    body = {
        "text": text,
        "ref_audio": normal_ref,
        "instruct": "whisper",
        "language": language,
        "guidance_scale": 2.0,
    }
    req = urllib.request.Request(
        server + "/synthesize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    j = json.load(urllib.request.urlopen(req, timeout=300))
    wav = urllib.request.urlopen(server + j["audio_url"], timeout=120).read()
    open(out_path, "wb").write(wav)
    return out_path


TTS_PASSAGE = {
    "ja": "ねえ、今日はおつかれさま。ここは静かだね。目を閉じて、ゆっくり息をして。眠くなるまで、そばで話していてあげる。",
    "en": "Hey, you did well today. It's quiet here. Close your eyes, breathe slowly. I'll keep talking softly until you drift off.",
}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("normal_dir")
    a.add_argument("asmr_dir")
    a.add_argument("--out")
    m = sub.add_parser("make")
    m.add_argument("normal_ref")
    m.add_argument("out")
    m.add_argument("--method", choices=["lpc", "tts", "both"], default="both")
    m.add_argument("--delta", help="json from analyze")
    m.add_argument("--server", default="http://127.0.0.1:9192")
    m.add_argument("--lang", default="ja")
    m.add_argument("--voice-mix", type=float, default=0.3)
    args = ap.parse_args()
    if args.cmd == "analyze":
        rep = analyze(args.normal_dir, args.asmr_dir)
        print(json.dumps(rep["delta"], indent=1))
        if args.out:
            json.dump(rep, open(args.out, "w"), indent=1)
            print("wrote", args.out)
        return
    delta = json.load(open(args.delta))["delta"] if args.delta else None
    base, ext = os.path.splitext(args.out)
    if args.method in ("lpc", "both"):
        x = load(args.normal_ref)
        y = apply_delta(whisperize_lpc(x, voice_mix=args.voice_mix), SR, delta)
        p = base + ("_lpc" if args.method == "both" else "") + ext
        sf.write(p, y, SR)
        print("wrote", p, "(lpc)")
    if args.method in ("tts", "both"):
        p = base + ("_tts" if args.method == "both" else "") + ext
        make_tts_ref(
            os.path.relpath(
                args.normal_ref,
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ).replace("\\", "/"),
            p,
            args.server,
            TTS_PASSAGE[args.lang],
            args.lang,
        )
        print("wrote", p, "(tts)")


if __name__ == "__main__":
    main()
