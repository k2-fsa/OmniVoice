from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from omnivoice.audiobook.schema import AudiobookPlan, AudiobookSegment


@dataclass
class SegmentWorkItem:
    chapter_id: str
    segment_id: str
    text: str
    status: str
    audio_path: Optional[str] = None


class AudiobookGenerationJob:
    def __init__(self, plan: AudiobookPlan):
        self.plan = plan

    def iter_segments(self) -> Iterable[AudiobookSegment]:
        for chapter in self.plan.chapters:
            for segment in chapter.segments:
                yield segment

    def pending_segments(self) -> List[SegmentWorkItem]:
        return [
            SegmentWorkItem(
                chapter_id=segment.chapter_id,
                segment_id=segment.id,
                text=segment.text,
                status=segment.status,
            )
            for segment in self.iter_segments()
            if segment.status in {"pending", "failed"}
        ]

    def next_segment(self) -> Optional[SegmentWorkItem]:
        items = self.pending_segments()
        return items[0] if items else None

    def mark_generated(self, segment_id: str, audio_path: str) -> None:
        segment = self._find_segment(segment_id)
        segment.status = "generated"
        segment.audio_path = str(audio_path)
        segment.error = None

    def mark_failed(self, segment_id: str, error: str) -> None:
        segment = self._find_segment(segment_id)
        segment.status = "failed"
        segment.error = error

    def progress(self) -> Dict[str, int]:
        counts = {"total": 0, "pending": 0, "generated": 0, "failed": 0}
        for segment in self.iter_segments():
            counts["total"] += 1
            status = segment.status if segment.status in counts else "pending"
            counts[status] += 1
        return counts

    def _find_segment(self, segment_id: str) -> AudiobookSegment:
        for segment in self.iter_segments():
            if segment.id == segment_id:
                return segment
        raise KeyError(f"Unknown audiobook segment: {segment_id}")


def chapter_audio_paths(plan: AudiobookPlan, chapter_id: str) -> List[Path]:
    paths: List[Path] = []
    for chapter in plan.chapters:
        if chapter.id != chapter_id:
            continue
        for segment in chapter.segments:
            if not segment.audio_path:
                raise ValueError(f"Segment {segment.id} has no generated audio path")
            paths.append(Path(segment.audio_path))
        return paths
    raise KeyError(f"Unknown audiobook chapter: {chapter_id}")


def write_generation_checkpoint(plan: AudiobookPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_generation_checkpoint(path: Path) -> AudiobookPlan:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Audiobook generation checkpoint must be a JSON object")
    return AudiobookPlan.from_dict(data)
