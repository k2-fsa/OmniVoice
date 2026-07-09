from __future__ import annotations

from typing import Any, Dict


class StructuredResultContractError(ValueError):
    pass


def audiobook_chunk_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["chapters", "warnings"],
        "properties": {
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "segments"],
                    "properties": {
                        "title": {"type": "string"},
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "text",
                                    "speaker",
                                    "pause_after_ms",
                                    "speed",
                                    "tone",
                                ],
                                "properties": {
                                    "text": {"type": "string", "minLength": 1},
                                    "speaker": {"type": "string"},
                                    "pause_after_ms": {"type": "integer", "minimum": 0},
                                    "speed": {"type": "number", "minimum": 0.5, "maximum": 1.5},
                                    "tone": {"type": "string"},
                                    "pronunciation_notes": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }


def validate_structured_chunk_content(content: Dict[str, Any]) -> None:
    allowed_top = {"chapters", "warnings"}
    extra_top = set(content) - allowed_top
    if extra_top:
        raise StructuredResultContractError(
            f"Structured chunk response has unexpected top-level fields: {sorted(extra_top)}"
        )
    chapters = content.get("chapters")
    warnings = content.get("warnings")
    if not isinstance(chapters, list) or not isinstance(warnings, list):
        raise StructuredResultContractError("Structured chunk response must include list fields: chapters and warnings")
    if not chapters:
        raise StructuredResultContractError("Structured chunk response must include at least one chapter")
    for warning in warnings:
        if not isinstance(warning, str):
            raise StructuredResultContractError("Structured chunk warnings must be strings")
    for chapter in chapters:
        if not isinstance(chapter, dict):
            raise StructuredResultContractError("Each chapter must be an object")
        chapter_extra = set(chapter) - {"title", "segments"}
        if chapter_extra:
            raise StructuredResultContractError(f"Chapter has unexpected fields: {sorted(chapter_extra)}")
        if not isinstance(chapter.get("title"), str) or not chapter.get("title"):
            raise StructuredResultContractError("Chapter title must be a non-empty string")
        segments = chapter.get("segments")
        if not isinstance(segments, list):
            raise StructuredResultContractError("Chapter segments must be a list")
        if not segments:
            raise StructuredResultContractError("Chapter must include at least one segment")
        for segment in segments:
            if not isinstance(segment, dict):
                raise StructuredResultContractError("Each segment must be an object")
            segment_extra = set(segment) - {
                "text",
                "speaker",
                "pause_after_ms",
                "speed",
                "tone",
                "pronunciation_notes",
            }
            if segment_extra:
                raise StructuredResultContractError(f"Segment has unexpected fields: {sorted(segment_extra)}")
            for key in ["text", "speaker", "tone"]:
                if not isinstance(segment.get(key), str) or not segment.get(key):
                    raise StructuredResultContractError(f"Segment {key} must be a non-empty string")
            if not isinstance(segment.get("pause_after_ms"), int) or segment["pause_after_ms"] < 0:
                raise StructuredResultContractError("Segment pause_after_ms must be a non-negative integer")
            if not isinstance(segment.get("speed"), (int, float)) or not 0.5 <= float(segment["speed"]) <= 1.5:
                raise StructuredResultContractError("Segment speed must be numeric between 0.5 and 1.5")
            notes = segment.get("pronunciation_notes")
            if notes is not None and (
                not isinstance(notes, list) or any(not isinstance(item, str) for item in notes)
            ):
                raise StructuredResultContractError("Segment pronunciation_notes must be a string list")
