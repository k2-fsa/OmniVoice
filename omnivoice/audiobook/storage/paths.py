from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    slug: str

    @property
    def project_root(self) -> Path:
        return self.root / self.slug

    @property
    def sources(self) -> Path:
        return self.project_root / "sources"

    @property
    def chunks(self) -> Path:
        return self.project_root / "chunks"

    @property
    def plans(self) -> Path:
        return self.project_root / "plans"

    @property
    def audio_raw(self) -> Path:
        return self.project_root / "audio" / "raw"

    @property
    def audio_master(self) -> Path:
        return self.project_root / "audio" / "master"

    @property
    def qc(self) -> Path:
        return self.project_root / "qc"

    @property
    def backups(self) -> Path:
        return self.project_root / "backups"

    def ensure(self) -> "ProjectPaths":
        for path in [
            self.sources,
            self.chunks,
            self.plans,
            self.audio_raw,
            self.audio_master,
            self.qc,
            self.backups,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self

    def segment_audio_name(self, chapter_index: int, segment_index: int, *, role: str = "raw") -> str:
        return f"{self.slug}_ch{chapter_index:03d}_seg{segment_index:04d}_{role}.wav"
