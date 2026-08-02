"""Structured pause controls for OmniVoice generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PauseSpec:
    after_char: int
    seconds: float


@dataclass(frozen=True)
class PausePlan:
    pauses: tuple[PauseSpec, ...]


@dataclass(frozen=True)
class PauseLayout:
    speech_frames: int
    total_frames: int
    phrase_frames: tuple[int, ...]
    pause_frames: tuple[int, ...]


_PAUSE_CANDIDATE_RE = re.compile(r"<pause:([^<>]*)>")
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _validate_seconds(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Pause duration must be a positive finite number")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pause duration must be a positive finite number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Pause duration must be a positive finite number")
    return seconds


def canonicalize_pause_plan(text: str, plan: PausePlan) -> PausePlan:
    """Validate, sort, and merge pauses that share an insertion offset."""
    if not isinstance(plan, PausePlan):
        raise TypeError("pause_plan items must be PausePlan or None")

    merged: dict[int, float] = {}
    for pause in plan.pauses:
        if not isinstance(pause, PauseSpec):
            raise TypeError("PausePlan.pauses must contain PauseSpec values")
        if isinstance(pause.after_char, bool) or not isinstance(pause.after_char, int):
            raise ValueError("Pause offset must be an integer")
        if pause.after_char <= 0 or pause.after_char >= len(text):
            raise ValueError("Leading and trailing pauses are not supported")
        seconds = _validate_seconds(pause.seconds)
        merged[pause.after_char] = merged.get(pause.after_char, 0.0) + seconds

    pauses = tuple(PauseSpec(offset, merged[offset]) for offset in sorted(merged))
    start = 0
    for pause in pauses:
        if not text[start : pause.after_char].strip():
            raise ValueError("Pause offsets must separate nonempty phrases")
        start = pause.after_char
    if pauses and not text[start:].strip():
        raise ValueError("Pause offsets must separate nonempty phrases")
    return PausePlan(pauses)


def parse_pause_markers(text: str) -> tuple[str, PausePlan]:
    """Strip inline pause markers and return offsets in cleaned text."""
    if not isinstance(text, str):
        raise TypeError("text items must be strings")

    matches = list(_PAUSE_CANDIDATE_RE.finditer(text))
    residue = _PAUSE_CANDIDATE_RE.sub("", text)
    if "<pause" in residue:
        raise ValueError("Malformed <pause...> marker")
    if not matches:
        return text, PausePlan(())

    durations = []
    for match in matches:
        raw = match.group(1)
        if not _NUMBER_RE.fullmatch(raw) and raw.lower() not in {
            "nan",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }:
            raise ValueError(f"Invalid pause duration: {raw!r}")
        durations.append(_validate_seconds(raw))

    segments: list[str] = []
    cursor = 0
    for match in matches:
        segments.append(text[cursor : match.start()])
        cursor = match.end()
    segments.append(text[cursor:])

    cleaned = segments[0]
    pauses: list[PauseSpec] = []
    i = 0
    while i < len(matches):
        seconds = durations[i]
        j = i
        while j + 1 < len(matches) and not segments[j + 1].strip():
            j += 1
            seconds += durations[j]

        right = segments[j + 1]
        had_whitespace = bool(cleaned and cleaned[-1].isspace()) or bool(
            right and right[0].isspace()
        )
        cleaned = cleaned.rstrip()
        if not cleaned:
            raise ValueError("Leading pauses are not supported")
        offset = len(cleaned)
        right = right.lstrip()
        if not right:
            raise ValueError("Trailing pauses are not supported")

        pauses.append(PauseSpec(offset, seconds))
        if had_whitespace:
            cleaned += " "
        cleaned += right
        i = j + 1

    return cleaned, canonicalize_pause_plan(cleaned, PausePlan(tuple(pauses)))


def pause_seconds_to_frames(seconds: float, frame_rate: int = 25) -> int:
    """Convert seconds to codec frames using round-half-up semantics."""
    seconds = _validate_seconds(seconds)
    frames = math.floor(seconds * frame_rate + 0.5)
    if frames < 1:
        raise ValueError("Pause duration is shorter than one codec frame")
    return frames


def create_pause_layout(
    text: str,
    plan: PausePlan,
    estimated_speech_frames: int,
    speed: float,
    duration: float | None,
    frame_rate: int,
    weight_fn: Callable[[str], float],
) -> PauseLayout:
    """Allocate speech phrases and fixed pause spans in codec-frame space."""
    plan = canonicalize_pause_plan(text, plan)
    if not plan.pauses:
        raise ValueError("Pause layout requires at least one pause")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be a positive finite number")

    pause_frames = tuple(
        pause_seconds_to_frames(pause.seconds, frame_rate) for pause in plan.pauses
    )
    if duration is None:
        speech_frames = max(1, int(estimated_speech_frames / speed))
        total_frames = speech_frames + sum(pause_frames)
    else:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be a positive finite number")
        total_frames = max(1, int(duration * frame_rate))
        speech_frames = total_frames - sum(pause_frames)

    offsets = [0, *(pause.after_char for pause in plan.pauses), len(text)]
    phrases = [text[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    phrase_count = len(phrases)
    if speech_frames < phrase_count:
        raise ValueError(
            "duration leaves fewer than one speech frame per nonempty phrase"
        )

    weights = [max(0.0, float(weight_fn(phrase))) for phrase in phrases]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * phrase_count
        total_weight = float(phrase_count)

    boundaries: list[int] = []
    previous = 0
    cumulative = 0.0
    for i, weight in enumerate(weights[:-1]):
        cumulative += weight
        proposed = math.floor(speech_frames * cumulative / total_weight + 0.5)
        remaining_phrases = phrase_count - i - 1
        boundary = max(previous + 1, proposed)
        boundary = min(boundary, speech_frames - remaining_phrases)
        boundaries.append(boundary)
        previous = boundary

    speech_boundaries = [0, *boundaries, speech_frames]
    phrase_frames = tuple(
        speech_boundaries[i + 1] - speech_boundaries[i] for i in range(phrase_count)
    )
    return PauseLayout(
        speech_frames=speech_frames,
        total_frames=total_frames,
        phrase_frames=phrase_frames,
        pause_frames=pause_frames,
    )
