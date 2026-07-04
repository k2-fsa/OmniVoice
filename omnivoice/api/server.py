#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OmniVoice REST API server.

Production-ready FastAPI server exposing:
  - Voice library management  (list / clone / delete / rename)
  - TTS generation            (sync WAV response)
  - Streaming TTS             (chunked HTTP or WebSocket)

Usage:
    omnivoice-api --model k2-fsa/OmniVoice --port 8000

Or programmatically:
    uvicorn omnivoice.api.server:create_app --factory --port 8000
"""

import argparse
import io
import logging
import os
import struct
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.voice_library import VoiceLibrary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (set by create_app / main)
# ---------------------------------------------------------------------------
_model: Optional[OmniVoice] = None
_library: Optional[VoiceLibrary] = None
_api_key: Optional[str] = None          # None = auth disabled
_executor = ThreadPoolExecutor(max_workers=1)   # one generation at a time
_gen_lock = threading.Lock()            # serialise GPU access


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class VoiceMeta(BaseModel):
    name: str
    safe_name: str
    ref_text: str
    ref_rms: float


class VoiceList(BaseModel):
    voices: list[VoiceMeta]
    count: int


class RenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1, max_length=100)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    voice: Optional[str] = Field(None, description="Saved voice name from library")
    language: Optional[str] = Field(None, description="Language name or ID (None = auto)")
    instruct: Optional[str] = Field(None, description="Voice-design instruction string")
    speed: Optional[float] = Field(None, ge=0.3, le=3.0)
    duration: Optional[float] = Field(None, ge=0.1, le=300.0)
    num_step: int = Field(32, ge=4, le=64)
    guidance_scale: float = Field(2.0, ge=0.0, le=4.0)
    denoise: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = Field(15.0, ge=5.0, le=60.0)
    audio_chunk_threshold: float = Field(30.0, ge=5.0, le=300.0)


class HealthResponse(BaseModel):
    status: str
    sample_rate: int
    device: str
    voices_count: int


class InfoResponse(BaseModel):
    model: str
    sample_rate: int
    device: str
    supported_languages: int
    library_dir: str
    voices: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_model():
    if _model is None:
        raise HTTPException(503, "Model not loaded yet.")
    return _model


def _require_library():
    if _library is None:
        raise HTTPException(503, "Voice library not initialised.")
    return _library


def _check_api_key(x_api_key: Optional[str] = Header(None)):
    if _api_key and x_api_key != _api_key:
        raise HTTPException(401, "Invalid or missing API key. Pass X-Api-Key header.")


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert 1-D float32 array to in-memory WAV bytes (PCM int16)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _build_gen_config(req: GenerateRequest) -> OmniVoiceGenerationConfig:
    return OmniVoiceGenerationConfig(
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        denoise=req.denoise,
        postprocess_output=req.postprocess_output,
        audio_chunk_duration=req.audio_chunk_duration,
        audio_chunk_threshold=req.audio_chunk_threshold,
    )


def _resolve_voice_prompt(req: GenerateRequest, model: OmniVoice, lib: VoiceLibrary):
    """Return VoiceClonePrompt or None based on request params."""
    if req.voice:
        if not lib.exists(req.voice):
            raise HTTPException(404, f"Voice '{req.voice}' not found in library.")
        return lib.load(req.voice)
    return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model: OmniVoice = None,
    library: VoiceLibrary = None,
    api_key: Optional[str] = None,
    cors_origins: list[str] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Can be called programmatically or used as a factory for uvicorn:
        uvicorn omnivoice.api.server:create_app --factory
    """
    global _model, _library, _api_key

    # Allow factory invocation from uvicorn (reads env vars)
    if model is None:
        _init_from_env()
    else:
        _model = model
        _library = library
        _api_key = api_key

    app = FastAPI(
        title="OmniVoice API",
        description=(
            "REST API for OmniVoice TTS — voice library management, "
            "synchronous WAV generation, streaming HTTP and WebSocket."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


def _init_from_env():
    """Load model from env vars (for uvicorn --factory usage)."""
    global _model, _library, _api_key

    checkpoint = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    device = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
    library_dir = os.environ.get("OMNIVOICE_LIBRARY_DIR")
    _api_key = os.environ.get("OMNIVOICE_API_KEY")
    load_asr = os.environ.get("OMNIVOICE_LOAD_ASR", "1") != "0"

    logger.info("Loading OmniVoice from %s on %s …", checkpoint, device)
    _model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        load_asr=load_asr,
    )
    _library = VoiceLibrary(library_dir)
    logger.info("Model ready. Library: %s", _library.library_dir)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def _register_routes(app: FastAPI):

    auth = [Depends(_check_api_key)]

    # ------------------------------------------------------------------ #
    #  System                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health():
        """Check server health and basic stats."""
        m = _require_model()
        lib = _require_library()
        return HealthResponse(
            status="ok",
            sample_rate=m.sampling_rate,
            device=str(m.device),
            voices_count=len(lib.names()),
        )

    @app.get("/info", response_model=InfoResponse, tags=["System"])
    def info():
        """Detailed model and library information."""
        m = _require_model()
        lib = _require_library()
        return InfoResponse(
            model=str(getattr(m.config, "_name_or_path", "OmniVoice")),
            sample_rate=m.sampling_rate,
            device=str(m.device),
            supported_languages=len(m.supported_language_ids()),
            library_dir=str(lib.library_dir),
            voices=lib.names(),
        )

    # ------------------------------------------------------------------ #
    #  Voice Library                                                       #
    # ------------------------------------------------------------------ #

    @app.get("/voices", response_model=VoiceList, tags=["Voices"],
             dependencies=auth)
    def list_voices():
        """Return all saved voice clones with metadata."""
        lib = _require_library()
        voices = [VoiceMeta(**v) for v in lib.list_voices()]
        return VoiceList(voices=voices, count=len(voices))

    @app.get("/voices/{name}", response_model=VoiceMeta, tags=["Voices"],
             dependencies=auth)
    def get_voice(name: str):
        """Get metadata for a single voice by name."""
        lib = _require_library()
        if not lib.exists(name):
            raise HTTPException(404, f"Voice '{name}' not found.")
        meta = next(v for v in lib.list_voices() if v["name"] == name)
        return VoiceMeta(**meta)

    @app.post("/voices/clone", response_model=VoiceMeta,
              status_code=201, tags=["Voices"], dependencies=auth)
    async def clone_voice(
        name: str = Form(..., description="Display name for the new voice"),
        ref_text: Optional[str] = Form(None, description="Transcript (leave empty to auto-transcribe)"),
        preprocess: bool = Form(True, description="Trim silence and normalise volume"),
        audio: UploadFile = File(..., description="Reference audio file (WAV/MP3/FLAC, 3–10s recommended)"),
    ):
        """Clone a voice from uploaded audio and save it to the library.

        The audio is processed once and stored as tokens — no audio file
        is kept on disk after cloning.
        """
        m = _require_model()
        lib = _require_library()

        if lib.exists(name):
            raise HTTPException(409, f"Voice '{name}' already exists. Delete it first or choose a different name.")

        # Save upload to a temp file (model needs a path or numpy array)
        suffix = "." + (audio.filename.rsplit(".", 1)[-1] if "." in audio.filename else "wav")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        try:
            loop = __import__("asyncio").get_event_loop()
            def _do_clone():
                with _gen_lock:
                    return m.create_voice_clone_prompt(
                        ref_audio=tmp_path,
                        ref_text=ref_text or None,
                        preprocess_prompt=preprocess,
                    )
            prompt = await loop.run_in_executor(_executor, _do_clone)
            lib.save(name, prompt)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            logger.exception("Clone failed")
            raise HTTPException(500, f"Cloning failed: {type(e).__name__}: {e}")
        finally:
            os.unlink(tmp_path)

        meta = next(v for v in lib.list_voices() if v["name"] == name)
        return VoiceMeta(**meta)

    @app.delete("/voices/{name}", tags=["Voices"], dependencies=auth)
    def delete_voice(name: str):
        """Delete a saved voice from the library."""
        lib = _require_library()
        if not lib.exists(name):
            raise HTTPException(404, f"Voice '{name}' not found.")
        lib.delete(name)
        return {"deleted": True, "name": name}

    @app.patch("/voices/{name}", response_model=VoiceMeta,
               tags=["Voices"], dependencies=auth)
    def rename_voice(name: str, body: RenameRequest):
        """Rename a saved voice."""
        lib = _require_library()
        if not lib.exists(name):
            raise HTTPException(404, f"Voice '{name}' not found.")
        if lib.exists(body.new_name):
            raise HTTPException(409, f"Voice '{body.new_name}' already exists.")
        lib.rename(name, body.new_name)
        meta = next(v for v in lib.list_voices() if v["name"] == body.new_name)
        return VoiceMeta(**meta)

    # ------------------------------------------------------------------ #
    #  Generation — synchronous (returns complete WAV)                     #
    # ------------------------------------------------------------------ #

    @app.post("/generate", tags=["Generate"], dependencies=auth,
              responses={200: {"content": {"audio/wav": {}}}})
    async def generate(req: GenerateRequest):
        """Generate speech and return a complete WAV file.

        Use this for short texts or when you need the full audio before
        playing. For long texts or real-time playback use `/generate/stream`
        or the WebSocket endpoint.
        """
        m = _require_model()
        lib = _require_library()
        prompt = _resolve_voice_prompt(req, m, lib)
        cfg = _build_gen_config(req)

        kw = dict(
            text=req.text,
            language=req.language,
            voice_clone_prompt=prompt,
            generation_config=cfg,
        )
        if req.instruct:
            kw["instruct"] = req.instruct
        if req.speed is not None:
            kw["speed"] = req.speed
        if req.duration is not None:
            kw["duration"] = req.duration

        loop = __import__("asyncio").get_event_loop()

        def _do_generate():
            with _gen_lock:
                return m.generate(**kw)[0]

        try:
            audio = await loop.run_in_executor(_executor, _do_generate)
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(500, f"Generation failed: {type(e).__name__}: {e}")

        wav_bytes = _audio_to_wav_bytes(audio, m.sampling_rate)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="output.wav"',
                "X-Sample-Rate": str(m.sampling_rate),
                "X-Audio-Duration": f"{len(audio) / m.sampling_rate:.3f}",
            },
        )

    # ------------------------------------------------------------------ #
    #  Generation — streaming HTTP (chunked PCM frames)                    #
    # ------------------------------------------------------------------ #

    @app.post("/generate/stream", tags=["Generate"], dependencies=auth,
              responses={200: {"content": {"application/octet-stream": {}}}})
    async def generate_stream(req: GenerateRequest):
        """Stream TTS audio as it is generated (chunk by chunk).

        **Response format** — binary stream of frames:

        ```
        [4 bytes uint32-LE: frame_length] [frame_length bytes: PCM int16-LE]
        ...repeated for each chunk...
        [4 bytes: 0x00000000]   <- end-of-stream sentinel
        ```

        Each frame is a valid segment of audio at `X-Sample-Rate` Hz, mono,
        int16. Concatenating all frames gives the full audio.

        **Headers returned:**
        - `X-Sample-Rate`: audio sample rate (e.g. 24000)
        - `X-Voice`: name of the voice used (or "uploaded")
        - `Transfer-Encoding`: chunked
        """
        m = _require_model()
        lib = _require_library()
        prompt = _resolve_voice_prompt(req, m, lib)
        cfg = _build_gen_config(req)

        kw = dict(
            text=req.text,
            language=req.language,
            voice_clone_prompt=prompt,
            generation_config=cfg,
        )
        if req.instruct:
            kw["instruct"] = req.instruct
        if req.speed is not None:
            kw["speed"] = req.speed
        if req.duration is not None:
            kw["duration"] = req.duration

        sample_rate = m.sampling_rate
        prev_len = 0

        async def _stream() -> AsyncIterator[bytes]:
            nonlocal prev_len
            loop = __import__("asyncio").get_event_loop()

            # Wrap the synchronous generator in run_in_executor
            gen = m.generate_streaming(**kw)

            def _next():
                try:
                    return next(gen)
                except StopIteration:
                    return None

            with _gen_lock:
                while True:
                    result = await loop.run_in_executor(_executor, _next)
                    if result is None:
                        break
                    audio, _status = result
                    # Only send the NEW portion since last yield
                    new_audio = audio[prev_len:]
                    prev_len = len(audio)
                    if len(new_audio) == 0:
                        continue
                    pcm = (np.clip(new_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    # Frame: [4-byte length][pcm bytes]
                    yield struct.pack("<I", len(pcm)) + pcm

            # End-of-stream sentinel
            yield struct.pack("<I", 0)

        return StreamingResponse(
            _stream(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Voice": req.voice or "none",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
            },
        )

    # ------------------------------------------------------------------ #
    #  Generation — WebSocket                                              #
    # ------------------------------------------------------------------ #

    @app.websocket("/ws/generate")
    async def ws_generate(websocket: WebSocket):
        """WebSocket endpoint for real-time TTS streaming.

        **Client sends** (JSON):
        ```json
        {
          "text": "Hello world",
          "voice": "Alice",
          "language": "English",
          "speed": 1.0,
          "num_step": 32
        }
        ```

        **Server sends** (interleaved):
        - **Binary frames**: raw PCM int16-LE mono at 24 kHz (new audio only)
        - **Text frames** (JSON):
          - `{"type": "status", "message": "Generating... chunk 1/3", "chunk": 1, "total": 3}`
          - `{"type": "done", "duration": 5.23, "sample_rate": 24000}`
          - `{"type": "error", "message": "..."}`

        **Auth**: if API key is configured, send `{"api_key": "..."}` as the
        first JSON message before the generation request.
        """
        await websocket.accept()
        m = _require_model()
        lib = _require_library()
        import asyncio, json as _json

        try:
            # --- Receive request ---
            raw = await websocket.receive_text()
            data = _json.loads(raw)

            # Optional auth via WebSocket message
            if _api_key and data.get("api_key") != _api_key:
                await websocket.send_text(_json.dumps({"type": "error", "message": "Unauthorised"}))
                await websocket.close(code=4001)
                return

            req = GenerateRequest(**{k: v for k, v in data.items() if k != "api_key"})
            prompt = _resolve_voice_prompt(req, m, lib)
            cfg = _build_gen_config(req)

            kw = dict(
                text=req.text,
                language=req.language,
                voice_clone_prompt=prompt,
                generation_config=cfg,
            )
            if req.instruct:
                kw["instruct"] = req.instruct
            if req.speed is not None:
                kw["speed"] = req.speed
            if req.duration is not None:
                kw["duration"] = req.duration

            sample_rate = m.sampling_rate
            prev_len = 0
            total_audio = None

            # --- Stream generation ---
            loop = asyncio.get_event_loop()
            gen = m.generate_streaming(**kw)

            def _next_chunk():
                try:
                    return next(gen)
                except StopIteration:
                    return None

            with _gen_lock:
                while True:
                    result = await loop.run_in_executor(_executor, _next_chunk)
                    if result is None:
                        break

                    audio, status = result
                    total_audio = audio

                    # Send only the new audio portion
                    new_audio = audio[prev_len:]
                    prev_len = len(audio)

                    if len(new_audio) > 0:
                        pcm = (np.clip(new_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                        await websocket.send_bytes(pcm)

                    # Parse chunk info from status string
                    chunk_num, total_chunks = None, None
                    if "chunk" in status:
                        try:
                            parts = status.split()
                            frac = parts[-1]   # e.g. "2/5"
                            chunk_num, total_chunks = map(int, frac.split("/"))
                        except Exception:
                            pass

                    await websocket.send_text(_json.dumps({
                        "type": "status",
                        "message": status,
                        "chunk": chunk_num,
                        "total": total_chunks,
                    }))

            # --- Final done message ---
            duration = len(total_audio) / sample_rate if total_audio is not None else 0.0
            await websocket.send_text(_json.dumps({
                "type": "done",
                "duration": round(duration, 3),
                "sample_rate": sample_rate,
            }))

        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected.")
        except HTTPException as e:
            try:
                import json as _json
                await websocket.send_text(_json.dumps({"type": "error", "message": e.detail}))
                await websocket.close()
            except Exception:
                pass
        except Exception as e:
            logger.exception("WebSocket generation error")
            try:
                import json as _json
                await websocket.send_text(_json.dumps({"type": "error", "message": str(e)}))
                await websocket.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omnivoice-api",
        description="Launch the OmniVoice REST API server.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model", default="k2-fsa/OmniVoice",
                   help="Model checkpoint path or HuggingFace repo id.")
    p.add_argument("--device", default=None,
                   help="Device (cuda / cpu / mps). Auto-detected if omitted.")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    p.add_argument("--library-dir", default=None,
                   help="Voice library directory. Default: ~/.omnivoice/voices/")
    p.add_argument("--api-key", default=None,
                   help="If set, all endpoints require X-Api-Key header.")
    p.add_argument("--no-asr", action="store_true", default=False,
                   help="Skip loading Whisper ASR (auto-transcription disabled).")
    p.add_argument("--asr-model", default="openai/whisper-large-v3-turbo",
                   help="ASR model for reference audio transcription.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of uvicorn worker processes (default: 1).")
    p.add_argument("--cors-origins", default="*",
                   help='Allowed CORS origins, comma-separated. Default: "*".')
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])
    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()
    dtype = torch.float16 if "cuda" in str(device) else torch.float32

    logger.info("Loading OmniVoice from %s on device=%s …", args.model, device)
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=dtype,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    logger.info("Model loaded. Sample rate: %d Hz", model.sampling_rate)

    library = VoiceLibrary(args.library_dir)
    logger.info("Voice library: %s  (%d voices)", library.library_dir, len(library.names()))

    cors = [o.strip() for o in args.cors_origins.split(",")]

    app = create_app(
        model=model,
        library=library,
        api_key=args.api_key or None,
        cors_origins=cors,
    )

    logger.info("Starting API server on http://%s:%d", args.host, args.port)
    logger.info("Interactive docs: http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
