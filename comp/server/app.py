"""OmniVoice TTS Server — FastAPI wrapper compatible with the CosyVoice 3 API.

Mirrors the request/response contract of the CosyVoice 3 sidecar server so
avatar-server's tts_service.py can swap engines via a single config flag.

Endpoints:
  POST /synthesize  — text-to-speech with emotion + language control
  GET  /health      — health check
  GET  /docs        — Swagger UI
  GET  /audio/{filename} — serve generated WAV

Usage:
  cd <omnivoice-dir>
  .venv/Scripts/python.exe server/app.py --port 9192 --ref-audio ref/default_ref.wav

Emotion mapping: DISABLED. All emotions -> None (pure neutral synthesis).

Short-text guard: inputs <= SHORT_TEXT_CHARS get a forced minimum duration, and
every short output is checked for degeneration (held tone / flat spectrum) and
regenerated up to MAX_ATTEMPTS times. See the guard block below for the why.

Attempted strategies (rejected by listening test):
  - pitch-shift via instruct=  -> warped voice
  - [happy]/[sad] inline tags  -> pronounced as words (not whitelisted)
  - whitelist tags ([laughter]/[sigh]/[hmph]/[surprise]) -> too theatrical

User opted for consistent neutral voice. Mood field is accepted by the
endpoint for API compatibility but doesn't affect synthesis. If we ever
re-enable: edit EMOTION_TAG dict + flip GUIDANCE_SCALE back to 3.0 (kept
at 2.0 default here so unintended brackets in text don't get pronounced).
"""

import argparse
import hashlib
import logging
import os
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("omnivoice-server")

app = FastAPI(title="OmniVoice TTS", version="1.0")

model = None
_gen_config = None
ref_audio_path: Optional[str] = None
_prompt_cache: dict = {}
sample_rate = 24000
audio_dir = Path("audio")
PROMPT_CACHE_DIR = Path("prompt_cache")

# Client-supplied ref_audio is confined to these directories. The endpoint is
# bound to 0.0.0.0 with no auth, so it is reachable from every device on the
# tailnet -- a raw path would give any of them a filesystem existence oracle
# (404 vs 500) and let them clone the voice out of any audio file on this PC.
# These three cover every path the GUI produces (refmaker writes clips into
# ref-voice-ai/, reads ref-voice-ai-pre-processed/).
REF_ROOTS = [
    ROOT / "ref",
    ROOT / "ref-output",
    ROOT / "ref-input",
    ROOT / "ref-voice-ai",
    ROOT / "ref-voice-ai-pre-processed",
]  # last two: legacy local layout
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


def _resolve_ref(candidate: str) -> str:
    """Resolve a client-supplied reference path, or reject it.

    Errors deliberately do not echo the path back -- that is what turns a
    failed lookup into an existence oracle.
    """
    try:
        resolved = Path(candidate).resolve()
    except (OSError, ValueError):
        raise HTTPException(400, "invalid ref_audio")
    if resolved.suffix.lower() not in AUDIO_EXTS:
        raise HTTPException(400, "ref_audio must be an audio file")
    if not any(resolved.is_relative_to(r.resolve()) for r in REF_ROOTS if r.exists()):
        raise HTTPException(403, "ref_audio outside the allowed reference directories")
    if not resolved.is_file():
        raise HTTPException(400, "ref_audio not found")
    return str(resolved)


# Emotion mapping disabled — all None. Dict + handler plumbing preserved so
# a future strategy slots in without touching request handling.
EMOTION_TAG = {
    "NEUTRAL": None,
    "HAPPY": None,
    "SAD": None,
    "ANGRY": None,
    "SURPRISED": None,
    "THINKING": None,
}

# Back to OmniVoice default. At 3.0 the model pronounces non-whitelisted
# bracket text literally — risky if user-supplied text contains brackets.
GUIDANCE_SCALE = 2.0

# --- Short-text degeneration guard -------------------------------------------
# OmniVoice starves short inputs of frames and collapses into a held tone.
# RuleDurationEstimator runs at a 75 Hz frame rate (24000 / downsample_factor
# 320) with a SOFT floor of low_threshold=50 frames == 0.67s, so a 3-char line
# gets ~47 frames and there is not enough budget to synthesise speech.
# Measured on nijika.wav: 3-char input degenerated 2 of 3 times (F0 std 11 Hz,
# spectral flatness 0.32) while a 34-char line was clean 3 of 3.
# Upstream: issue #229, filed against 0.2.1, closed as duplicate, still unfixed.
#
# Two defences, in order:
#   1. force a minimum frame budget via generate(duration=...) for short text
#   2. verify the returned audio and regenerate if it still came out degenerate
MIN_DURATION_S = 1.6  # forced frame budget floor for short text
SHORT_TEXT_CHARS = 14  # inputs at or below this length get the floor
CHECK_BELOW_S = 3.0  # only verify short outputs; long ones are reliable
# Thresholds recalibrated 2026-08-26 against real short-clip output. The first
# cut used flatness > 0.12, but legitimate SHORT clips measure 0.057-0.137 --
# the threshold sat inside the normal range and rejected good audio, costing a
# full regeneration each time (3 of 6 normal lines were hitting 3 attempts).
# Pitch variation is the clean separator: degenerate tones measure ~11-21,
# healthy short speech 121-298. Flatness survives only as a far-out backstop.
F0STD_MIN = 60.0  # a held tone has near-zero pitch variation
FLATNESS_MAX = 0.22  # backstop for noise bursts; well above healthy speech
CLIP_FRACTION_MAX = 0.02  # degenerate blowups slam into full-scale clipping
MAX_ATTEMPTS = 2  # the detector is accurate now; a 3rd try rarely helped


def _get_prompt(ref_path: str):
    """Voice clone prompt for a reference, built once and cached.

    Without this, every request re-runs Whisper over the reference audio to
    recover ref_text — which cold-loads a second model onto the GPU and, on a
    12 GB card already holding the desktop, thrashes VRAM badly enough to wedge
    a single short request. Build once per ref, reuse forever.
    """
    prompt = _prompt_cache.get(ref_path)
    if prompt is not None:
        return prompt

    from omnivoice.models.omnivoice import VoiceClonePrompt

    src = Path(ref_path)
    key = hashlib.sha1(
        f"{src.resolve()}:{src.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    cache_file = PROMPT_CACHE_DIR / f"{src.stem}-{key}.pt"

    t0 = time.time()
    if cache_file.exists():
        try:
            prompt = VoiceClonePrompt.load(str(cache_file))
            logger.info(f"[prompt] loaded {cache_file.name} in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.warning(f"[prompt] cache load failed ({e}), rebuilding")
            prompt = None
    if prompt is None:
        logger.info(f"[prompt] building voice clone prompt for {ref_path} ...")
        prompt = model.create_voice_clone_prompt(ref_path)
        try:
            PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            prompt.save(str(cache_file))
        except Exception as e:
            logger.warning(f"[prompt] could not persist prompt: {e}")
        logger.info(f"[prompt] built in {time.time() - t0:.1f}s -> {cache_file.name}")

    _prompt_cache[ref_path] = prompt
    return prompt


def _floor_duration(text: str, requested: Optional[float]) -> Optional[float]:
    """Minimum output duration to request, or None to let the model estimate."""
    if requested is not None:
        return requested
    return MIN_DURATION_S if len(text.strip()) <= SHORT_TEXT_CHARS else None


def _degeneracy(audio: np.ndarray, sr: int) -> Optional[str]:
    """Return a reason string if the audio looks degenerate, else None.

    Thresholds separated every good/bad sample in the 2026-08-26 measurement
    run. Only short clips are checked, so the common path pays nothing.
    """
    y = np.asarray(audio, dtype=np.float32).squeeze()
    if y.size == 0:
        return "empty"
    if y.size / sr >= CHECK_BELOW_S:
        return None
    # Judge exactly what _save_wav will write: clipped to +/-1 and quantised to
    # int16. Checking the raw float array instead let clipped-to-1.0 garbage
    # slip through as "clean" while the served WAV was audibly broken.
    clipped = float(np.mean(np.abs(y) >= 0.99))
    y = np.clip(y, -1.0, 1.0)
    y = (y * 32767).astype(np.int16).astype(np.float32) / 32767.0
    if clipped > CLIP_FRACTION_MAX:
        return f"clipped={clipped:.2f}"
    try:
        import librosa

        # yin, not pyin: 12 ms vs 764 ms for the same decision.
        f0 = librosa.yin(
            librosa.resample(y, orig_sr=sr, target_sr=16000),
            fmin=65,
            fmax=800,
            sr=16000,
        )
        f0 = f0[np.isfinite(f0)]
        if f0.size:
            sd = float(np.std(f0))
            if sd < F0STD_MIN:
                return f"f0std={sd:.1f}"
        flat = float(librosa.feature.spectral_flatness(y=y).mean())
        if flat > FLATNESS_MAX:
            return f"flatness={flat:.3f}"
    except Exception as e:  # detection must never break synthesis
        logger.warning(f"[synth] degeneracy check failed: {e}")
    return None


class SynthRequest(BaseModel):
    text: str
    emotion: Optional[str] = "NEUTRAL"
    language: Optional[str] = "en"  # OmniVoice auto-detects; field kept for API compat
    speed: Optional[float] = 1.0
    ref_audio: Optional[str] = (
        None  # path under REF_ROOTS; falls back to startup default
    )
    instruct: Optional[str] = (
        None  # whitelist style hint: "high pitch", "whisper", "young adult", etc.
    )
    guidance_scale: Optional[float] = (
        None  # per-request override; falls back to GUIDANCE_SCALE
    )
    duration: Optional[float] = (
        None  # force output length (s); overrides the short-text floor
    )
    asmr: Optional[str] = (
        None  # post-process preset: close | room | drift (server/asmr_fx.py)
    )


class SynthResponse(BaseModel):
    audio_url: str
    duration: float
    rtf: float
    attempts: int = 1
    degraded: bool = False


@app.on_event("startup")
async def startup():
    global model, ref_audio_path, sample_rate, _gen_config
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    ref_audio_path = os.environ.get("REF_AUDIO", str(ROOT / "ref" / "default_ref.wav"))
    model_id = os.environ.get("MODEL_ID", "k2-fsa/OmniVoice")

    logger.info(f"Loading OmniVoice from {model_id}...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        model_id,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # Whisper stays on CPU: on a 12 GB card already holding the desktop, a
        # GPU-resident whisper-large-v3-turbo pushed us to 11.6/12 GB and
        # wedged synthesis entirely. Needs omnivoice >= 0.2.1 (upstream #224).
        asr_device="cpu",
    )
    _gen_config = OmniVoiceGenerationConfig(guidance_scale=GUIDANCE_SCALE)
    logger.info(
        f"Model loaded in {time.time() - t0:.1f}s. CUDA: {torch.cuda.is_available()}"
    )
    logger.info(f"Reference audio: {ref_audio_path}")
    logger.info(f"Guidance scale: {GUIDANCE_SCALE}")
    audio_dir.mkdir(exist_ok=True)
    try:
        _get_prompt(ref_audio_path)  # pre-warm so request #1 doesn't pay for ASR
    except Exception as e:
        logger.warning(f"Could not pre-warm default ref prompt: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "OmniVoice", "cuda": torch.cuda.is_available()}


def _save_wav(audio: np.ndarray, sr: int, path: Path) -> float:
    """Mono (n,) or stereo (n, 2) float -> 16-bit PCM."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    channels = audio.shape[1] if audio.ndim == 2 else 1
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return len(audio) / sr


@app.post("/synthesize", response_model=SynthResponse)
async def synthesize(req: SynthRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")

    emotion = (req.emotion or "NEUTRAL").upper()
    tag = EMOTION_TAG.get(emotion)
    tagged_text = f"{tag} {text}" if tag else text
    logger.info(
        f"[synth] emotion={emotion} tag={tag!r} guidance={GUIDANCE_SCALE} text={text[:60]}"
    )

    # The startup default is operator-supplied (--ref-audio) and exempt; only a
    # path arriving in the request body is untrusted.
    ref = _resolve_ref(req.ref_audio) if req.ref_audio else ref_audio_path

    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    gscale = req.guidance_scale if req.guidance_scale is not None else GUIDANCE_SCALE
    gen_config = (
        OmniVoiceGenerationConfig(guidance_scale=gscale)
        if gscale != GUIDANCE_SCALE
        else _gen_config
    )

    forced_duration = _floor_duration(text, req.duration)
    logger.info(
        f"[synth] instruct={req.instruct!r} gscale={gscale} duration={forced_duration}"
    )

    t0 = time.time()
    audio = None
    reason = None
    attempts = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        try:
            kwargs = {
                "text": tagged_text,
                "voice_clone_prompt": _get_prompt(ref),
                "generation_config": gen_config,
            }
            if req.speed and req.speed != 1.0:
                kwargs["speed"] = req.speed
            if req.instruct:
                kwargs["instruct"] = req.instruct
            if forced_duration is not None:
                kwargs["duration"] = forced_duration
            result = model.generate(**kwargs)
            audio = result[0] if isinstance(result, list) else result
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
        except Exception as e:
            logger.error(f"[synth] failed: {e}")
            raise HTTPException(500, f"Synthesis failed: {e}")

        reason = _degeneracy(np.asarray(audio), sample_rate)
        if reason is None:
            break
        logger.warning(
            f"[synth] degenerate output ({reason}) on attempt {attempt}/{MAX_ATTEMPTS}, retrying"
        )
        # widen the frame budget each retry — starvation is the usual cause
        forced_duration = (forced_duration or MIN_DURATION_S) + 0.5

    if reason is not None:
        logger.error(
            f"[synth] still degenerate after {MAX_ATTEMPTS} attempts ({reason})"
        )

    elapsed = time.time() - t0
    audio = np.asarray(audio)
    if req.asmr:
        from asmr_fx import PRESETS, asmr_pipeline

        if req.asmr not in PRESETS:
            raise HTTPException(
                400, f"unknown asmr preset; valid: {', '.join(PRESETS)}"
            )
        audio, _ = asmr_pipeline(audio, sample_rate, req.asmr)  # -> stereo
    filename = f"{uuid.uuid4().hex[:12]}.wav"
    filepath = audio_dir / filename
    duration = _save_wav(audio, sample_rate, filepath)
    rtf = elapsed / duration if duration > 0 else 0
    logger.info(
        f"[synth] {duration:.1f}s in {elapsed:.1f}s (RTF={rtf:.2f}): {filename}"
    )

    return SynthResponse(
        audio_url=f"/audio/{filename}",
        duration=round(duration, 2),
        rtf=round(rtf, 2),
        attempts=attempts,
        degraded=reason is not None,
    )


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    filepath = audio_dir / filename
    if not filepath.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(str(filepath), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9192)
    parser.add_argument("--ref-audio", default=str(ROOT / "ref" / "default_ref.wav"))
    parser.add_argument("--model-id", default="k2-fsa/OmniVoice")
    args = parser.parse_args()

    os.environ["REF_AUDIO"] = args.ref_audio
    os.environ["MODEL_ID"] = args.model_id

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
