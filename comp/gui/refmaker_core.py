"""Pure processing pipeline for the OmniVoice reference clip maker.

No Tkinter imports — fully testable on its own. The GUI (refmaker.pyw) just
orchestrates these functions on background threads.

Pipeline:
    extract_audio(input)            ffmpeg  -> temp 44.1k stereo wav
    isolate_vocals(wav)             demucs  -> vocals.wav  (optional)
    find_candidates(wav, clip_len)  silence-split + sliding window -> [Candidate]
    export_clip(wav, a, b, out)     cut -> mono target SR -> peak-normalize -> wav
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Hide the console window ffmpeg/demucs would otherwise pop when the GUI runs
# under pythonw.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

EXTRACT_SR = 44100  # demucs htdemucs is trained at 44.1k stereo
DEFAULT_TARGET_SR = 24000
DEFAULT_CLIP_LEN_S = 8.0
ANALYSIS_SR = 16000  # plenty for silence/energy analysis, keeps it fast


@dataclass
class Candidate:
    """A proposed reference clip, in seconds, with a quality score in [0, 1]."""

    start_s: float
    end_s: float
    score: float

    @property
    def length_s(self) -> float:
        return self.end_s - self.start_s


# --------------------------------------------------------------------------- #
# subprocess helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command, raising RuntimeError with the stderr tail on failure."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=_CREATE_NO_WINDOW,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        tail = "\n".join(msg.splitlines()[-15:])
        raise RuntimeError(
            f"`{cmd[0]} {cmd[1] if len(cmd) > 1 else ''}` failed:\n{tail}"
        )
    return proc


def check_demucs_available(python_exe: str | None = None) -> bool:
    """True if `demucs` can be imported by the target interpreter."""
    py = python_exe or sys.executable
    try:
        proc = subprocess.run(
            [py, "-c", "import demucs"],
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=_CREATE_NO_WINDOW,
        )
        return proc.returncode == 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# pipeline stages
# --------------------------------------------------------------------------- #
def extract_audio(input_path: str | Path, workdir: str | Path | None = None) -> Path:
    """Decode any video/audio file to a 44.1k stereo WAV via ffmpeg."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="refmaker_"))
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{input_path.stem}_audio.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(EXTRACT_SR),
            str(out),
        ]
    )
    return out


def isolate_vocals(
    wav_path: str | Path,
    workdir: str | Path | None = None,
    python_exe: str | None = None,
) -> Path:
    """Run demucs two-stem vocal isolation, returning the vocals.wav path."""
    wav_path = Path(wav_path)
    out_root = Path(workdir) if workdir else wav_path.parent / "separated"
    py = python_exe or sys.executable
    _run(
        [
            py,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            "htdemucs",
            "-o",
            str(out_root),
            str(wav_path),
        ]
    )
    vocals = out_root / "htdemucs" / wav_path.stem / "vocals.wav"
    if not vocals.exists():
        raise RuntimeError(f"demucs finished but no vocals at {vocals}")
    return vocals


def _coverage(a: float, b: float, intervals: list[tuple[float, float]]) -> float:
    """Fraction of [a, b] covered by the (sorted, disjoint) speech intervals."""
    span = b - a
    if span <= 0:
        return 0.0
    covered = 0.0
    for s, e in intervals:
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            covered += hi - lo
    return covered / span


def _snap(value: float, boundaries: np.ndarray, tol: float) -> float:
    """Snap value to the nearest boundary within tol seconds, else leave it."""
    if len(boundaries) == 0:
        return value
    idx = int(np.argmin(np.abs(boundaries - value)))
    nearest = float(boundaries[idx])
    return nearest if abs(nearest - value) <= tol else value


def _rms_score(rms: float, floor: float = 0.02, high: float = 0.6) -> float:
    """1.0 across a comfortable loudness band; ramps down when too quiet or too hot."""
    if rms < floor:
        return rms / floor  # too quiet -> likely noise floor
    if rms <= high:
        return 1.0
    return max(0.0, 1.0 - (rms - high) / (1.0 - high))  # approaching clipping


def find_candidates(
    audio_path: str | Path,
    clip_len_s: float = DEFAULT_CLIP_LEN_S,
    max_candidates: int = 20,
    top_db: float = 35.0,
    analysis_sr: int = ANALYSIS_SR,
) -> list[Candidate]:
    """Detect clean speech windows and return them ranked best-first.

    Slides a window of clip_len_s across the file, snaps its edges toward nearby
    silence boundaries so clips don't cut mid-word, scores each window, then
    applies greedy overlap suppression so the results are diverse.
    """
    y, sr = librosa.load(str(audio_path), sr=analysis_sr, mono=True)
    total_s = len(y) / sr if sr else 0.0
    if total_s < clip_len_s:
        return []

    # Non-silent regions (in seconds) and their edges as snap boundaries.
    intervals_samp = librosa.effects.split(y, top_db=top_db)
    intervals = [(s / sr, e / sr) for s, e in intervals_samp]
    if not intervals:
        return []
    boundaries = np.array(
        sorted({0.0, total_s} | {pt for iv in intervals for pt in iv})
    )

    tol = clip_len_s * 0.15
    step = clip_len_s / 2.0
    raw: list[Candidate] = []
    t = 0.0
    while t + clip_len_s <= total_s + 1e-6:
        a = _snap(t, boundaries, tol)
        b = _snap(t + clip_len_s, boundaries, tol)
        a = max(0.0, a)
        b = min(total_s, b)
        length = b - a
        # Reject windows the snap distorted too far from the requested length.
        if length < clip_len_s * 0.7 or length > clip_len_s * 1.3:
            t += step
            continue

        i0, i1 = int(a * sr), int(b * sr)
        seg = y[i0:i1]
        if seg.size == 0:
            t += step
            continue
        coverage = _coverage(a, b, intervals)
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(seg)))

        clip_penalty = 0.5 if peak >= 0.99 else 0.0
        score = max(0.0, 0.6 * coverage + 0.4 * _rms_score(rms) - clip_penalty)
        raw.append(Candidate(round(a, 3), round(b, 3), round(score, 4)))
        t += step

    # Greedy non-max suppression: keep highest-scoring, drop heavy overlaps.
    raw.sort(key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for cand in raw:
        if all(_overlap(cand, k) < 0.5 for k in kept):
            kept.append(cand)
        if len(kept) >= max_candidates:
            break
    return kept


def _overlap(a: Candidate, b: Candidate) -> float:
    """Overlap of two candidates as a fraction of the shorter one."""
    lo, hi = max(a.start_s, b.start_s), min(a.end_s, b.end_s)
    inter = max(0.0, hi - lo)
    shorter = min(a.length_s, b.length_s)
    return inter / shorter if shorter > 0 else 0.0


def find_speech_intervals(
    audio_path: str | Path,
    top_db: float = 35.0,
    min_seg_s: float = 0.30,
    merge_gap_s: float = 0.25,
    analysis_sr: int = ANALYSIS_SR,
) -> list[tuple[float, float]]:
    """Return every voiced span [(start_s, end_s), ...] in chronological order.

    Runs librosa silence-splitting, glues neighbours separated by less than
    `merge_gap_s` (keeps natural phrasing together), then drops blips shorter
    than `min_seg_s` (residual clicks / stray SFX the vocal isolation missed).
    """
    y, sr = librosa.load(str(audio_path), sr=analysis_sr, mono=True)
    if y.size == 0 or not sr:
        return []
    raw = librosa.effects.split(y, top_db=top_db)  # sample indices
    if len(raw) == 0:
        return []
    intervals = [[s / sr, e / sr] for s, e in raw]

    merged: list[list[float]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(round(s, 3), round(e, 3)) for s, e in merged if (e - s) >= min_seg_s]


def _append_xfade(buf: np.ndarray, seg: np.ndarray, xfade: int) -> np.ndarray:
    """Concatenate seg onto buf with an `xfade`-sample equal-length crossfade.

    The short linear crossfade hides the click that a hard splice between two
    voiced spans would otherwise make.
    """
    if buf.size == 0:
        return seg.copy()
    n = min(xfade, buf.size, seg.size)
    if n <= 0:
        return np.concatenate([buf, seg])
    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
    blended = buf[-n:] * (1.0 - fade) + seg[:n] * fade
    return np.concatenate([buf[:-n], blended, seg[n:]])


def merge_speech(
    source_path: str | Path,
    out_path: str | Path,
    *,
    target_sr: int = DEFAULT_TARGET_SR,
    top_db: float = 35.0,
    target_len_s: float = 25.0,
    pad_s: float = 0.12,
    crossfade_s: float = 0.015,
    min_seg_s: float = 0.30,
    merge_gap_s: float = 0.25,
    peak: float = 0.97,
    full: bool = False,
    intervals: list[tuple[float, float]] | None = None,
) -> tuple[Path, float]:
    """Cut every voiced span out of source_path and stitch them into one clip.

    Designed for game/VN rips where the character speaks in short bursts between
    unvoiced lines / SFX (run demucs first so the gaps are real silence). Detects
    all speech, concatenates with a tiny crossfade, peak-normalizes, and writes a
    mono PCM_16 WAV.

    capped (default): accumulates voiced spans chronologically until the running
    speech total reaches `target_len_s`, then stops — ideal ~length for cloning.
    full=True: concatenates every detected span.

    `intervals`: pass pre-computed voiced spans (e.g. from a real speech VAD) to
    bypass the built-in energy split. When None, energy splitting is used.

    Returns (out_path, actual_length_s).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if intervals is None:
        intervals = find_speech_intervals(
            source_path, top_db=top_db, min_seg_s=min_seg_s, merge_gap_s=merge_gap_s
        )
    if not intervals:
        raise RuntimeError("No speech detected to merge.")

    y, sr = librosa.load(str(source_path), sr=target_sr, mono=True)
    total_s = len(y) / sr if sr else 0.0
    xfade = max(0, int(crossfade_s * sr))
    pad = max(0.0, pad_s)

    buf = np.zeros(0, dtype=np.float32)
    speech_s = 0.0  # un-padded voiced seconds gathered so far (drives the cap)
    for s, e in intervals:
        a = max(0.0, s - pad)
        b = min(total_s, e + pad)
        seg = y[int(a * sr) : int(b * sr)].astype(np.float32)
        if seg.size == 0:
            continue
        buf = _append_xfade(buf, seg, xfade)
        speech_s += e - s
        if not full and speech_s >= target_len_s:
            break

    # Near-silent buffer => either pure silence in, or demucs found no vocals.
    # Don't write a silent "reference".
    m = float(np.max(np.abs(buf))) if buf.size else 0.0
    if m < 1e-3:
        raise RuntimeError("No speech detected to merge.")
    buf = buf / m * peak
    sf.write(str(out_path), buf, target_sr, subtype="PCM_16")
    return out_path, (len(buf) / target_sr if target_sr else 0.0)


def export_clip(
    audio_path: str | Path,
    start_s: float,
    end_s: float,
    out_path: str | Path,
    target_sr: int = DEFAULT_TARGET_SR,
    peak: float = 0.97,
) -> Path:
    """Cut [start_s, end_s] from audio_path -> mono target_sr -> peak-normalized WAV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_s - start_s)
    y, _ = librosa.load(
        str(audio_path), sr=target_sr, mono=True, offset=start_s, duration=duration
    )
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m > 0:
        y = y / m * peak
    sf.write(str(out_path), y, target_sr, subtype="PCM_16")
    return out_path


def format_ts(seconds: float) -> str:
    """Format seconds as mm:ss.s for display."""
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:04.1f}"
