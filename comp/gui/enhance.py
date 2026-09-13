"""Neural speech cleanup for the reference maker — deep clean + speech VAD.

Two capabilities, both lazily loaded so importing this module stays cheap (the
heavy torch / speechbrain / silero imports only happen on first real use):

  deep_clean(in, out)   SepFormer DNS4 enhancement (speechbrain). Regenerates
                        clean speech, removing noise/SFX that *overlaps* the
                        voice — e.g. the typewriter "blip" a visual-novel plays
                        while text scrolls, which demucs and energy gating leave
                        behind because it sits in the same instant as her words.

  detect_speech(path)   Silero VAD. Returns real speech spans [(start, end)...],
                        far better than energy gating at ignoring non-speech.

Models cache under OmniVoice/models/ (gitignored).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "sepformer-dns4"
ENHANCE_SR = 16000  # the SepFormer DNS4 model is 16 kHz

_MODEL = None  # (separator, device)
_VAD = None


def is_available() -> bool:
    """True if both speechbrain (deep clean) and silero_vad (VAD) are installed."""
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "speechbrain", "silero_vad")
    )


# --------------------------------------------------------------------------- #
# Silero VAD
# --------------------------------------------------------------------------- #
def _get_vad():
    global _VAD
    if _VAD is None:
        from silero_vad import load_silero_vad

        _VAD = load_silero_vad()
    return _VAD


def detect_speech(
    audio_path: str | Path, sr: int = ENHANCE_SR
) -> list[tuple[float, float]]:
    """Return real speech spans [(start_s, end_s), ...] via Silero VAD."""
    from silero_vad import get_speech_timestamps, read_audio

    model = _get_vad()
    wav = read_audio(str(audio_path), sampling_rate=sr)
    ts = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


# --------------------------------------------------------------------------- #
# SepFormer DNS4 deep clean
# --------------------------------------------------------------------------- #
def _get_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from speechbrain.inference.separation import SepformerSeparation as Sep

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        sep = Sep.from_hparams(
            source="speechbrain/sepformer-dns4-16k-enhancement",
            savedir=str(MODEL_DIR),
            run_opts={"device": device},
        )
        _MODEL = (sep, device)
    return _MODEL


def _xfade(buf: np.ndarray, seg: np.ndarray, n: int) -> np.ndarray:
    """Append seg onto buf with an n-sample linear crossfade (hides seams)."""
    if buf.size == 0:
        return seg.copy()
    n = min(n, buf.size, seg.size)
    if n <= 0:
        return np.concatenate([buf, seg])
    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
    blended = buf[-n:] * (1.0 - fade) + seg[:n] * fade
    return np.concatenate([buf[:-n], blended, seg[n:]])


def deep_clean(
    in_path: str | Path,
    out_path: str | Path,
    *,
    target_sr: int = 24000,
    chunk_s: float = 15.0,
    overlap_s: float = 0.5,
    peak: float = 0.97,
) -> tuple[Path, float]:
    """Run SepFormer DNS4 enhancement over in_path, writing a cleaned WAV.

    Processed in overlapping chunks (crossfade-stitched) so arbitrarily long
    audio never OOMs the model. Output is mono PCM_16 at target_sr.
    """
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, device = _get_model()

    y, _ = librosa.load(str(in_path), sr=ENHANCE_SR, mono=True)
    if y.size == 0:
        raise RuntimeError("Nothing to clean (empty audio).")

    hop = max(1, int(chunk_s * ENHANCE_SR))
    ov = max(0, int(overlap_s * ENHANCE_SR))
    out = np.zeros(0, dtype=np.float32)
    i = 0
    while i < len(y):
        chunk = y[i : i + hop + ov]
        mix = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            est = model.separate_batch(mix)[0, :, 0].detach().cpu().numpy()
        out = _xfade(out, est.astype(np.float32), ov if out.size else 0)
        i += hop

    m = float(np.max(np.abs(out))) if out.size else 0.0
    if m < 1e-4:
        raise RuntimeError("Deep clean produced silence — check the input.")
    out = out / m * peak

    if target_sr != ENHANCE_SR:
        out = librosa.resample(out, orig_sr=ENHANCE_SR, target_sr=target_sr)
        m = float(np.max(np.abs(out))) or 0.0
        if m > 0:
            out = out / m * peak

    sf.write(str(out_path), out, target_sr, subtype="PCM_16")
    return out_path, (len(out) / target_sr if target_sr else 0.0)
