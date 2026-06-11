from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PausePreset:
    name: str
    comma_ms: int
    sentence_ms: int
    ellipsis_ms: int
    paragraph_ms: int
    section_ms: int


@dataclass
class NarrationSegment:
    id: str
    index: int
    text: str
    pause_after_ms: int
    speed: float
    section: Optional[str] = None
    status: str = "pending"
    cache_key: Optional[str] = None
    audio_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NarrationSegment":
        return cls(
            id=str(data.get("id") or ""),
            index=int(data.get("index") or 0),
            text=str(data.get("text") or ""),
            pause_after_ms=int(data.get("pause_after_ms") or 0),
            speed=float(data.get("speed") or 1.0),
            section=data.get("section"),
            status=str(data.get("status") or "pending"),
            cache_key=data.get("cache_key"),
            audio_path=data.get("audio_path"),
            error=data.get("error"),
        )


@dataclass
class NarrationPlan:
    preset: str
    segments: List[NarrationSegment] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "settings": self.settings,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NarrationPlan":
        return cls(
            preset=str(data.get("preset") or "Presentation"),
            settings=dict(data.get("settings") or {}),
            segments=[
                NarrationSegment.from_dict(item)
                for item in data.get("segments", [])
                if isinstance(item, dict)
            ],
        )


DEFAULT_PRESETS: Dict[str, PausePreset] = {
    "Natural": PausePreset(
        name="Natural",
        comma_ms=220,
        sentence_ms=520,
        ellipsis_ms=800,
        paragraph_ms=950,
        section_ms=1300,
    ),
    "Presentation": PausePreset(
        name="Presentation",
        comma_ms=300,
        sentence_ms=750,
        ellipsis_ms=1100,
        paragraph_ms=1400,
        section_ms=1800,
    ),
    "Manual": PausePreset(
        name="Manual",
        comma_ms=250,
        sentence_ms=650,
        ellipsis_ms=900,
        paragraph_ms=1000,
        section_ms=1500,
    ),
}
