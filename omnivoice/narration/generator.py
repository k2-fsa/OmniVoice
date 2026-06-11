from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from omnivoice.narration.assembler import (
    assemble_segments,
    audio_duration_seconds,
    float_to_int16,
)
from omnivoice.narration.cache import NarrationCache
from omnivoice.narration.schema import NarrationPlan, NarrationSegment


@dataclass
class NarrationGenerationResult:
    plan: NarrationPlan
    audio: np.ndarray
    sample_rate: int
    cache_hits: int
    generated: int
    failed: int

    @property
    def duration_seconds(self) -> float:
        return audio_duration_seconds(self.audio, self.sample_rate)

    def status_text(self) -> str:
        return (
            f"Concluido. segmentos={len(self.plan.segments)}, gerados={self.generated}, "
            f"cache={self.cache_hits}, falhas={self.failed}, "
            f"duracao={self.duration_seconds:.2f}s"
        )


def _identity_hash(value: Optional[str]) -> str:
    if not value:
        return ""
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def plan_to_json(plan: NarrationPlan) -> str:
    return json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)


def plan_from_json(plan_json: str) -> NarrationPlan:
    data = json.loads(plan_json)
    if not isinstance(data, dict):
        raise ValueError("Narration plan must be a JSON object")
    return NarrationPlan.from_dict(data)


def _load_cached_audio(path: str, cache: NarrationCache) -> np.ndarray:
    audio_path = cache.validate_audio_path(path)
    import soundfile as sf

    audio, _sr = sf.read(str(audio_path), dtype="float32")
    return np.asarray(audio, dtype=np.float32)


def _generate_one(
    model,
    segment: NarrationSegment,
    language: Optional[str],
    instruct: Optional[str],
    voice_clone_prompt: Any,
    generation_settings: Dict[str, Any],
) -> np.ndarray:
    kwargs: Dict[str, Any] = {
        "text": segment.text,
        "language": language,
        "speed": segment.speed,
        **generation_settings,
    }
    if voice_clone_prompt is not None:
        kwargs["voice_clone_prompt"] = voice_clone_prompt
    elif instruct:
        kwargs["instruct"] = instruct
    return model.generate(**kwargs)[0]


def generate_narration(
    model,
    plan: NarrationPlan,
    model_name: str,
    language: Optional[str] = None,
    instruct: Optional[str] = None,
    voice_clone_prompt: Any = None,
    voice_mode: str = "design",
    generation_settings: Optional[Dict[str, Any]] = None,
    cache: Optional[NarrationCache] = None,
    force_regenerate: Optional[set[int]] = None,
) -> NarrationGenerationResult:
    cache = cache or NarrationCache()
    generation_settings = dict(generation_settings or {})
    force_regenerate = force_regenerate or set()
    voice_identity = _identity_hash(instruct or str(type(voice_clone_prompt)))

    audios: List[np.ndarray] = []
    cache_hits = 0
    generated = 0
    failed = 0

    for segment in plan.segments:
        try:
            key = cache.make_key(
                segment=segment,
                model_name=model_name,
                voice_mode=voice_mode,
                generation_settings=generation_settings,
                voice_identity=voice_identity,
            )
            segment.cache_key = key
            cached = None if segment.index in force_regenerate else cache.get_cached_segment(key)
            if cached is not None:
                audio, sample_rate, path = cached
                if int(sample_rate) == int(model.sampling_rate):
                    segment.audio_path = str(path)
                    segment.status = "cached"
                    segment.error = None
                    audios.append(np.asarray(audio, dtype=np.float32))
                    cache_hits += 1
                    continue

            audio = _generate_one(
                model=model,
                segment=segment,
                language=language,
                instruct=instruct,
                voice_clone_prompt=voice_clone_prompt,
                generation_settings=generation_settings,
            )
            path = cache.put_cached_segment(
                key=key,
                audio=audio,
                sample_rate=model.sampling_rate,
                metadata={
                    "text": segment.text,
                    "speed": segment.speed,
                    "pause_after_ms": segment.pause_after_ms,
                    "model": model_name,
                    "voice_mode": voice_mode,
                    "generation_settings": generation_settings,
                },
            )
            segment.audio_path = str(path)
            segment.status = "generated"
            segment.error = None
            audios.append(np.asarray(audio, dtype=np.float32))
            generated += 1
        except Exception as exc:
            segment.status = "failed"
            segment.error = f"{type(exc).__name__}: {exc}"
            failed += 1

    if failed:
        failed_items = [f"{s.index}: {s.error}" for s in plan.segments if s.status == "failed"]
        raise RuntimeError("Falha ao gerar segmentos de narracao: " + "; ".join(failed_items))

    final_audio = assemble_segments(
        audios,
        [segment.pause_after_ms for segment in plan.segments],
        sample_rate=model.sampling_rate,
    )
    return NarrationGenerationResult(
        plan=plan,
        audio=final_audio,
        sample_rate=model.sampling_rate,
        cache_hits=cache_hits,
        generated=generated,
        failed=failed,
    )


def assemble_from_plan(
    plan: NarrationPlan,
    sample_rate: int,
    cache: Optional[NarrationCache] = None,
) -> np.ndarray:
    cache = cache or NarrationCache()
    audios: List[np.ndarray] = []
    for segment in plan.segments:
        if not segment.audio_path:
            raise ValueError(f"Segmento {segment.index} nao tem audio em cache")
        audios.append(_load_cached_audio(segment.audio_path, cache))
    return assemble_segments(
        audios,
        [segment.pause_after_ms for segment in plan.segments],
        sample_rate=sample_rate,
    )


def regenerate_segment(
    model,
    plan: NarrationPlan,
    segment_index: int,
    **kwargs,
) -> NarrationGenerationResult:
    return generate_narration(
        model=model,
        plan=plan,
        force_regenerate={int(segment_index)},
        **kwargs,
    )


def result_audio_for_gradio(result: NarrationGenerationResult):
    return result.sample_rate, float_to_int16(result.audio)
