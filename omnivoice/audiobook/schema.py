from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AudiobookProject:
    title: str
    author: str
    language: str
    genre: str
    source_docx_hash: str
    created_at: str


@dataclass
class VoiceProfile:
    mode: str = "design"
    default_voice: str = "narrator"
    speed: float = 0.92
    style: str = "technical_clear"
    pronunciation_notes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AudiobookQcTargets:
    sample_rate_hz: int = 44100
    peak_dbfs_max: float = -3.0
    target_loudness_lufs: float = -20.0
    allowed_loudness_tolerance: float = 2.0
    max_segment_chars: int = 900
    target_words_per_minute: int = 150


@dataclass
class AudiobookSegment:
    id: str
    text: str
    text_hash: str
    speaker: str
    pause_after_ms: int
    speed: float
    tone: str
    chapter_id: str
    status: str = "pending"
    source_paragraph_index: Optional[int] = None


@dataclass
class AudiobookChapter:
    id: str
    title: str
    order: int
    segments: List[AudiobookSegment] = field(default_factory=list)


@dataclass
class AudiobookPlan:
    project: AudiobookProject
    voice_profile: VoiceProfile
    chapters: List[AudiobookChapter]
    qc_targets: AudiobookQcTargets
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
