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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudiobookProject":
        return cls(
            title=str(data.get("title") or ""),
            author=str(data.get("author") or ""),
            language=str(data.get("language") or "pt-BR"),
            genre=str(data.get("genre") or "technical"),
            source_docx_hash=str(data.get("source_docx_hash") or ""),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass
class VoiceProfile:
    mode: str = "design"
    default_voice: str = "narrator"
    speed: float = 0.92
    style: str = "technical_clear"
    pronunciation_notes: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        notes = data.get("pronunciation_notes") or []
        return cls(
            mode=str(data.get("mode") or "design"),
            default_voice=str(data.get("default_voice") or "narrator"),
            speed=float(data.get("speed") or 0.92),
            style=str(data.get("style") or "technical_clear"),
            pronunciation_notes=[dict(item) for item in notes if isinstance(item, dict)],
        )


@dataclass
class AudiobookQcTargets:
    sample_rate_hz: int = 44100
    peak_dbfs_max: float = -3.0
    target_loudness_lufs: float = -20.0
    allowed_loudness_tolerance: float = 2.0
    max_segment_chars: int = 900
    target_words_per_minute: int = 150

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudiobookQcTargets":
        return cls(
            sample_rate_hz=int(data.get("sample_rate_hz") or 44100),
            peak_dbfs_max=float(data.get("peak_dbfs_max") or -3.0),
            target_loudness_lufs=float(data.get("target_loudness_lufs") or -20.0),
            allowed_loudness_tolerance=float(data.get("allowed_loudness_tolerance") or 2.0),
            max_segment_chars=int(data.get("max_segment_chars") or 900),
            target_words_per_minute=int(data.get("target_words_per_minute") or 150),
        )


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
    audio_path: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudiobookSegment":
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            text_hash=str(data.get("text_hash") or ""),
            speaker=str(data.get("speaker") or "narrator"),
            pause_after_ms=int(data.get("pause_after_ms") or 0),
            speed=float(data.get("speed") or 1.0),
            tone=str(data.get("tone") or "neutral"),
            chapter_id=str(data.get("chapter_id") or ""),
            status=str(data.get("status") or "pending"),
            source_paragraph_index=data.get("source_paragraph_index"),
            audio_path=data.get("audio_path"),
            error=data.get("error"),
        )


@dataclass
class AudiobookChapter:
    id: str
    title: str
    order: int
    segments: List[AudiobookSegment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudiobookChapter":
        segments = data.get("segments") or []
        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or ""),
            order=int(data.get("order") or 0),
            segments=[
                AudiobookSegment.from_dict(item)
                for item in segments
                if isinstance(item, dict)
            ],
        )


@dataclass
class AudiobookPlan:
    project: AudiobookProject
    voice_profile: VoiceProfile
    chapters: List[AudiobookChapter]
    qc_targets: AudiobookQcTargets
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudiobookPlan":
        chapters = data.get("chapters") or []
        return cls(
            project=AudiobookProject.from_dict(dict(data.get("project") or {})),
            voice_profile=VoiceProfile.from_dict(dict(data.get("voice_profile") or {})),
            chapters=[
                AudiobookChapter.from_dict(item)
                for item in chapters
                if isinstance(item, dict)
            ],
            qc_targets=AudiobookQcTargets.from_dict(dict(data.get("qc_targets") or {})),
            settings=dict(data.get("settings") or {}),
        )
