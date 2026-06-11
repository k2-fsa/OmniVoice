from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from omnivoice.audiobook.docx import DocxDocument, DocxParagraph, extract_docx_structure
from omnivoice.audiobook.schema import (
    AudiobookChapter,
    AudiobookPlan,
    AudiobookProject,
    AudiobookQcTargets,
    AudiobookSegment,
    VoiceProfile,
)
from omnivoice.narration.parser import parse_narration_text


_TECHNICAL_STYLE = "technical_clear"
_FICTION_STYLE = "fiction_narrative"
_HEADING_RE = re.compile(r"^(?:heading|titulo|title)[\s_-]*([1-6])?$", re.I)


@dataclass
class AudiobookPlanConfig:
    title: str
    author: str = ""
    language: str = "pt-BR"
    genre: str = "technical"
    voice_mode: str = "design"
    default_voice: str = "narrator"
    speed: float = 0.92
    preset: str = "Presentation"
    max_segment_chars: int = 900
    target_words_per_minute: Optional[int] = None


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _segment_id(chapter_order: int, segment_order: int) -> str:
    return f"seg_{chapter_order:03d}_{segment_order:06d}"


def _chapter_id(order: int) -> str:
    return f"ch_{order:03d}"


def _is_heading(paragraph: DocxParagraph) -> bool:
    style = paragraph.style or ""
    if _HEADING_RE.match(style):
        return True
    text = paragraph.text.strip()
    if len(text) > 90 or text.endswith((".", "?", "!", ";", ":")):
        return False
    return len(text.split()) <= 10 and any(char.isalpha() for char in text)


def _split_chapters(paragraphs: Iterable[DocxParagraph], fallback_title: str) -> List[tuple[str, List[DocxParagraph]]]:
    chapters: List[tuple[str, List[DocxParagraph]]] = []
    current_title = fallback_title or "Chapter 1"
    current: List[DocxParagraph] = []

    for paragraph in paragraphs:
        if _is_heading(paragraph):
            if current:
                chapters.append((current_title, current))
                current = []
            current_title = paragraph.text.strip()
            continue
        current.append(paragraph)

    if current:
        chapters.append((current_title, current))
    if not chapters:
        chapters.append((fallback_title or "Chapter 1", []))
    return chapters


def _tone_for_genre(genre: str) -> str:
    return "narrative" if genre == "fiction" else "neutral"


def _style_for_genre(genre: str) -> str:
    return _FICTION_STYLE if genre == "fiction" else _TECHNICAL_STYLE


def _wpm_for_genre(genre: str, override: Optional[int]) -> int:
    if override:
        return int(override)
    return 158 if genre == "fiction" else 145


def create_audiobook_plan(document: DocxDocument, config: AudiobookPlanConfig) -> AudiobookPlan:
    genre = config.genre if config.genre in {"technical", "fiction"} else "technical"
    chapters: List[AudiobookChapter] = []

    for chapter_order, (title, paragraphs) in enumerate(
        _split_chapters(document.paragraphs, config.title),
        start=1,
    ):
        chapter_text = "\n\n".join(paragraph.text for paragraph in paragraphs)
        narration_plan = parse_narration_text(
            chapter_text,
            preset_name=config.preset,
            global_speed=config.speed,
            remove_slide_labels=True,
            clean_slide_artifacts=True,
        )
        chapter = AudiobookChapter(id=_chapter_id(chapter_order), title=title, order=chapter_order)
        for segment_order, segment in enumerate(narration_plan.segments, start=1):
            chapter.segments.append(
                AudiobookSegment(
                    id=_segment_id(chapter_order, segment_order),
                    text=segment.text,
                    text_hash=_stable_hash(segment.text),
                    speaker="narrator",
                    pause_after_ms=segment.pause_after_ms,
                    speed=segment.speed,
                    tone=_tone_for_genre(genre),
                    chapter_id=chapter.id,
                    source_paragraph_index=None,
                )
            )
        chapters.append(chapter)

    return AudiobookPlan(
        project=AudiobookProject(
            title=config.title,
            author=config.author,
            language=config.language,
            genre=genre,
            source_docx_hash=document.sha256,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        voice_profile=VoiceProfile(
            mode=config.voice_mode,
            default_voice=config.default_voice,
            speed=config.speed,
            style=_style_for_genre(genre),
        ),
        chapters=chapters,
        qc_targets=AudiobookQcTargets(
            max_segment_chars=config.max_segment_chars,
            target_words_per_minute=_wpm_for_genre(genre, config.target_words_per_minute),
        ),
        settings={
            "source_docx_path": document.path,
            "parser": "omnivoice.audiobook.v1",
            "preset": config.preset,
            "offline_required": True,
            "external_provider": None,
        },
    )


def create_audiobook_plan_from_docx(path: str | Path, config: AudiobookPlanConfig) -> AudiobookPlan:
    return create_audiobook_plan(extract_docx_structure(path), config)


def create_audiobook_plan_from_openrouter_results(
    document: DocxDocument,
    config: AudiobookPlanConfig,
    results: Iterable[Dict[str, object]],
    *,
    model: str,
) -> AudiobookPlan:
    genre = config.genre if config.genre in {"technical", "fiction"} else "technical"
    chapters: List[AudiobookChapter] = []
    chapter_order = 0

    for result in results:
        for item in result.get("chapters", []):  # type: ignore[union-attr]
            if not isinstance(item, dict):
                continue
            chapter_order += 1
            chapter = AudiobookChapter(
                id=_chapter_id(chapter_order),
                title=str(item.get("title") or f"Chapter {chapter_order}"),
                order=chapter_order,
            )
            segments = item.get("segments", [])
            if not isinstance(segments, list):
                segments = []
            for segment_order, segment_data in enumerate(segments, start=1):
                if not isinstance(segment_data, dict):
                    continue
                text = str(segment_data.get("text") or "").strip()
                if not text:
                    continue
                chapter.segments.append(
                    AudiobookSegment(
                        id=_segment_id(chapter_order, segment_order),
                        text=text,
                        text_hash=_stable_hash(text),
                        speaker=str(segment_data.get("speaker") or "narrator"),
                        pause_after_ms=int(segment_data.get("pause_after_ms") or 750),
                        speed=float(segment_data.get("speed") or config.speed),
                        tone=str(segment_data.get("tone") or _tone_for_genre(genre)),
                        chapter_id=chapter.id,
                    )
                )
            chapters.append(chapter)

    if not any(chapter.segments for chapter in chapters):
        raise ValueError("OpenRouter results did not contain any valid audiobook segments")

    return AudiobookPlan(
        project=AudiobookProject(
            title=config.title,
            author=config.author,
            language=config.language,
            genre=genre,
            source_docx_hash=document.sha256,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        voice_profile=VoiceProfile(
            mode=config.voice_mode,
            default_voice=config.default_voice,
            speed=config.speed,
            style=_style_for_genre(genre),
        ),
        chapters=chapters,
        qc_targets=AudiobookQcTargets(
            max_segment_chars=config.max_segment_chars,
            target_words_per_minute=_wpm_for_genre(genre, config.target_words_per_minute),
        ),
        settings={
            "source_docx_path": document.path,
            "parser": "omnivoice.audiobook.openrouter.v1",
            "preset": config.preset,
            "offline_required": False,
            "external_provider": "openrouter",
            "openrouter_model": model,
        },
    )


def plan_to_json(plan: AudiobookPlan) -> str:
    return json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)


def write_plan(plan: AudiobookPlan, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan_to_json(plan), encoding="utf-8")
    return path
