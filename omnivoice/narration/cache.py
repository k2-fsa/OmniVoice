from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from omnivoice._offline import ensure_path_inside
from omnivoice.narration.schema import NarrationSegment


def default_cache_dir() -> Path:
    docker_path = Path("/workspace/.cache/omnivoice/narration")
    if Path("/workspace").exists():
        return docker_path
    return Path(".cache/omnivoice/narration")


class NarrationCache:
    def __init__(self, root: Optional[os.PathLike[str] | str] = None):
        self.root = Path(root) if root else default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        segment: NarrationSegment,
        model_name: str,
        voice_mode: str,
        generation_settings: Dict[str, Any],
        voice_identity: Optional[str] = None,
    ) -> str:
        payload = {
            "text": segment.text,
            "speed": round(float(segment.speed), 4),
            "pause_after_ms": int(segment.pause_after_ms),
            "model": model_name,
            "voice_mode": voice_mode,
            "voice_identity": voice_identity or "",
            "generation_settings": generation_settings,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def audio_path(self, key: str) -> Path:
        return self.root / f"{key}.wav"

    def meta_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def validate_audio_path(self, path: os.PathLike[str] | str) -> Path:
        audio_path = ensure_path_inside(self.root, path)
        if audio_path.suffix.lower() != ".wav":
            raise ValueError(f"Arquivo de cache invalido: {audio_path}")
        return audio_path

    def get_cached_segment(self, key: str) -> Optional[Tuple[Any, int, Path]]:
        audio_path = self.audio_path(key)
        meta_path = self.meta_path(key)
        if not audio_path.exists() or not meta_path.exists():
            return None
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        except Exception:
            return None
        return audio, sample_rate, audio_path

    def put_cached_segment(
        self,
        key: str,
        audio,
        sample_rate: int,
        metadata: Dict[str, Any],
    ) -> Path:
        import soundfile as sf

        audio_path = self.audio_path(key)
        meta_path = self.meta_path(key)
        sf.write(str(audio_path), audio, sample_rate)
        meta = dict(metadata)
        meta.update(
            {
                "cache_key": key,
                "sampling_rate": sample_rate,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "audio_path": str(audio_path),
            }
        )
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return audio_path

    def clear_cache(self) -> int:
        count = 0
        for path in self.root.glob("*"):
            if path.is_file():
                path.unlink()
                count += 1
        return count

    def cache_stats(self) -> Dict[str, Any]:
        wavs = list(self.root.glob("*.wav"))
        bytes_total = sum(path.stat().st_size for path in wavs)
        return {
            "root": str(self.root),
            "segments": len(wavs),
            "bytes": bytes_total,
        }
