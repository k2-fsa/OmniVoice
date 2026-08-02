"""Single-reference narration controls built on local codec-token inpainting."""

from __future__ import annotations

import difflib
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio

from omnivoice.controls import PausePlan, PauseSpec, parse_pause_markers
from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoice,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
)


@dataclass(frozen=True)
class NarrationControl:
    kind: str
    start_char: int
    end_char: int
    value: str | float
    source: str = "explicit"


@dataclass(frozen=True)
class NarrationPlan:
    text: str
    controls: tuple[NarrationControl, ...]
    pause_plan: PausePlan
    source_text: str


@dataclass
class NarrationResult:
    audio: np.ndarray
    baseline_audio: np.ndarray
    sample_rate: int
    plan: NarrationPlan
    tokens: torch.Tensor
    baseline_tokens: torch.Tensor
    report: dict[str, Any]
    traces: Optional[dict[str, Any]] = None


_CONTROL_TAG_RE = re.compile(
    r"<(?P<closing>/)?(?P<name>emphasis|rate|intonation|aside)"
    r"(?P<attrs>\s+[^<>]*?)?\s*>",
    re.IGNORECASE,
)
_CONTROL_FRAGMENT_RE = re.compile(
    r"</?(?:emphasis|rate|intonation|aside)\b", re.IGNORECASE
)
_ATTR_RE = re.compile(r"([a-zA-Z_][\w-]*)\s*=\s*(['\"])(.*?)\2")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_PAUSE_RE = re.compile(r"<pause:[^<>]*>")
_CMU_VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}
_CMU_CONSONANTS = {
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
}

NARRATION_CAPABILITIES = {
    "format": "omnivoice_narration_capabilities_v1",
    "frame_rate": 25,
    "short_form_only": True,
    "regional_controls": {
        "emphasis": {"values": ["reduced", "moderate", "strong"]},
        "rate": {"min": 0.7, "max": 1.4, "step": 0.05},
        "intonation": {"values": ["rising", "falling"]},
        "aside": {"values": ["parenthetical"]},
    },
    "pause": {"min_seconds": 0.04, "max_seconds": 5.0, "frame_seconds": 0.04},
    "trace": {
        "format": "omnivoice_generation_trace_v1",
        "fields": [
            "unmask_step",
            "token_logprob",
            "entropy",
            "cfg_delta",
            "fixed",
            "pause",
        ],
    },
}


def narration_capabilities() -> dict[str, Any]:
    """Return stable controls/trace schema without loading model weights."""
    import copy

    return copy.deepcopy(NARRATION_CAPABILITIES)


def _expand_shorthand(text: str) -> str:
    def emphasis_replacement(match: re.Match) -> str:
        marker, content = match.group(1), match.group(2)
        strength = "strong" if len(marker) == 3 else "moderate"
        return (
            f'<emphasis strength="{strength}" source="shorthand">{content}</emphasis>'
        )

    text = re.sub(r"(?<!\*)(\*{2,3})([^*\n]+?)\1(?!\*)", emphasis_replacement, text)
    text = re.sub(
        r"\(([^()\n]+)\)",
        r'<aside source="shorthand">\1</aside>',
        text,
    )
    text = re.sub(
        r"—\s*([^—\n]+?)\s*—",
        r'<emphasis strength="strong" source="dash">\1</emphasis>',
        text,
    )
    return text


def _parse_attrs(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    attrs = {}
    consumed = []
    for match in _ATTR_RE.finditer(raw):
        attrs[match.group(1).lower()] = match.group(3)
        consumed.append((match.start(), match.end()))
    residue = list(raw)
    for start, end in consumed:
        residue[start:end] = " " * (end - start)
    if "".join(residue).strip():
        raise ValueError(f"Malformed narration-control attributes: {raw!r}")
    return attrs


def _control_value(kind: str, attrs: dict[str, str]) -> tuple[str | float, str]:
    source = attrs.pop("source", "explicit")
    if kind == "emphasis":
        value = attrs.pop("strength", "moderate").lower()
        if value not in {"reduced", "moderate", "strong"}:
            raise ValueError("emphasis strength must be reduced, moderate, or strong")
    elif kind == "rate":
        raw = attrs.pop("value", None)
        if raw is None:
            raise ValueError("rate control requires value")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError("rate value must be numeric") from exc
        if not math.isfinite(value) or not 0.7 <= value <= 1.4:
            raise ValueError("rate value must be between 0.7 and 1.4")
    elif kind == "intonation":
        value = attrs.pop("type", "").lower()
        if value not in {"rising", "falling"}:
            raise ValueError("intonation type must be rising or falling")
    else:
        value = "parenthetical"
    if attrs:
        raise ValueError(f"Unsupported {kind} attribute(s): {', '.join(sorted(attrs))}")
    return value, source


def _remap_unchanged_span(
    before: str, after: str, start_char: int, end_char: int
) -> tuple[int, int]:
    """Map a control span through pause-marker removal without guessing by text."""
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, before_start, before_end, after_start, _ in matcher.get_opcodes():
        if tag == "equal" and before_start <= start_char and end_char <= before_end:
            return (
                after_start + start_char - before_start,
                after_start + end_char - before_start,
            )
    raise ValueError("Could not map narration control after pause removal")


def _normalize_cmu_pronunciation(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("CMU pronunciation values must be strings")
    raw = value.strip()
    if raw.startswith("[") or raw.endswith("]"):
        if not (raw.startswith("[") and raw.endswith("]")):
            raise ValueError("CMU pronunciation brackets must be balanced")
        raw = raw[1:-1].strip()
    tokens = raw.upper().split()
    if not tokens:
        raise ValueError("CMU pronunciation cannot be empty")
    for token in tokens:
        base = token.rstrip("012")
        stress = token[len(base) :]
        if base in _CMU_VOWELS:
            if stress not in {"0", "1", "2"}:
                raise ValueError(f"CMU vowel requires stress 0, 1, or 2: {token}")
        elif base in _CMU_CONSONANTS:
            if stress:
                raise ValueError(f"CMU consonant cannot have stress: {token}")
        else:
            raise ValueError(f"Unknown CMU phoneme: {token}")
    return f"[{' '.join(tokens)}]"


def _resolve_pronunciations(
    plan: NarrationPlan, pronunciations: Optional[dict[str, str]]
) -> dict[int, str | tuple[tuple[int, int, str], ...]]:
    if pronunciations is None:
        return {}
    if not isinstance(pronunciations, dict):
        raise TypeError("pronunciations must be a dictionary")

    resolved_parts: dict[int, list[tuple[int, int, str]]] = {}
    seen_keys = set()
    ordered = sorted(
        pronunciations.items(), key=lambda item: len(str(item[0]).strip()), reverse=True
    )
    for raw_key, raw_value in ordered:
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise TypeError("pronunciation keys must be nonempty strings")
        key = raw_key.strip().casefold()
        if key in seen_keys:
            raise ValueError(f"Duplicate pronunciation key: {raw_key!r}")
        seen_keys.add(key)
        pronunciation = _normalize_cmu_pronunciation(raw_value)
        pattern = re.compile(r"(?<![A-Za-z0-9'])" + re.escape(key) + r"(?![A-Za-z0-9'])")
        matched = bool(pattern.search(plan.text.casefold()))
        for control_index, control in enumerate(plan.controls):
            surface = plan.text[control.start_char : control.end_char]
            for match in pattern.finditer(surface.casefold()):
                parts = resolved_parts.setdefault(control_index, [])
                start, end = match.span()
                if any(start < old_end and old_start < end for old_start, old_end, _ in parts):
                    continue
                parts.append((start, end, pronunciation))
        if not matched:
            raise ValueError(
                f"Pronunciation key does not match a controlled span: {raw_key!r}"
            )

    resolved: dict[int, str | tuple[tuple[int, int, str], ...]] = {}
    for control_index, parts in resolved_parts.items():
        control = plan.controls[control_index]
        surface = plan.text[control.start_char : control.end_char]
        parts.sort(key=lambda item: item[0])
        if len(parts) == 1 and parts[0][0] == 0 and parts[0][1] == len(surface):
            resolved[control_index] = parts[0][2]
        else:
            resolved[control_index] = tuple(parts)
    return resolved


def _pronunciation_replacements(
    text: str, pronunciations: Optional[dict[str, str]]
) -> list[tuple[int, int, str]]:
    if pronunciations is None:
        return []
    if not isinstance(pronunciations, dict):
        raise TypeError("pronunciations must be a dictionary")
    replacements = []
    occupied: list[tuple[int, int]] = []
    ordered = sorted(
        pronunciations.items(), key=lambda item: len(str(item[0]).strip()), reverse=True
    )
    for raw_key, raw_value in ordered:
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise TypeError("pronunciation keys must be nonempty strings")
        key = raw_key.strip()
        pronunciation = _normalize_cmu_pronunciation(raw_value)
        pattern = re.compile(
            r"(?<![A-Za-z0-9'])" + re.escape(key) + r"(?![A-Za-z0-9'])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < old_end and old_start < end for old_start, old_end in occupied):
                continue
            occupied.append((start, end))
            replacements.append((start, end, pronunciation))
    replacements.sort(key=lambda item: item[0])
    return replacements


def _apply_pronunciations(
    text: str, pronunciations: Optional[dict[str, str]]
) -> str:
    replacements = _pronunciation_replacements(text, pronunciations)
    if not replacements:
        return text
    parts = []
    cursor = 0
    for start, end, pronunciation in replacements:
        parts.extend((text[cursor:start], pronunciation))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _apply_pronunciations_to_pause_plan(
    text: str,
    pause_plan: PausePlan,
    pronunciations: Optional[dict[str, str]],
) -> tuple[str, PausePlan]:
    replacements = _pronunciation_replacements(text, pronunciations)
    if not replacements:
        return text, pause_plan
    conditioned = _apply_pronunciations(text, pronunciations)
    remapped = []
    for pause in pause_plan.pauses:
        delta = 0
        for start, end, pronunciation in replacements:
            if pause.after_char >= end:
                delta += len(pronunciation) - (end - start)
            elif start < pause.after_char < end:
                raise ValueError("Pause cannot split a pronunciation-controlled word")
        remapped.append(PauseSpec(pause.after_char + delta, pause.seconds))
    return conditioned, PausePlan(tuple(remapped))


def parse_narration_controls(text: str, shorthand: bool = False) -> NarrationPlan:
    """Parse narration markup into cleaned text and non-overlapping controls."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    marked_text = _expand_shorthand(text) if shorthand else text
    controls = []
    stack = []
    cleaned_parts = []
    cleaned_length = 0
    cursor = 0

    for match in _CONTROL_TAG_RE.finditer(marked_text):
        chunk = marked_text[cursor : match.start()]
        cleaned_parts.append(chunk)
        cleaned_length += len(chunk)
        name = match.group("name").lower()
        closing = bool(match.group("closing"))
        attrs = _parse_attrs(match.group("attrs"))
        if closing:
            if attrs:
                raise ValueError(
                    "Closing narration-control tags cannot have attributes"
                )
            if not stack or stack[-1][0] != name:
                raise ValueError(f"Mismatched closing tag: {name}")
            _, start_char, value, source = stack.pop()
            if cleaned_length <= start_char:
                raise ValueError(f"{name} control cannot be empty")
            controls.append(
                NarrationControl(name, start_char, cleaned_length, value, source)
            )
        else:
            value, source = _control_value(name, attrs)
            stack.append((name, cleaned_length, value, source))
        cursor = match.end()

    tail = marked_text[cursor:]
    cleaned_parts.append(tail)
    intermediate = "".join(cleaned_parts)
    if stack:
        raise ValueError(f"Unclosed narration-control tag: {stack[-1][0]}")
    residue = _CONTROL_TAG_RE.sub("", marked_text)
    if _CONTROL_FRAGMENT_RE.search(residue):
        raise ValueError("Malformed narration-control tag")

    controls.sort(key=lambda item: (item.start_char, item.end_char))
    previous_end = -1
    for control in controls:
        if control.start_char < previous_end:
            raise ValueError(
                "Nested or overlapping narration controls are not supported"
            )
        if _PAUSE_RE.search(intermediate[control.start_char : control.end_char]):
            raise ValueError("Pause markers cannot be nested inside narration controls")
        previous_end = control.end_char

    cleaned_text, pause_plan = parse_pause_markers(intermediate)
    remapped = []
    for control in controls:
        start_char, end_char = _remap_unchanged_span(
            intermediate,
            cleaned_text,
            control.start_char,
            control.end_char,
        )
        remapped.append(
            NarrationControl(
                control.kind,
                start_char,
                end_char,
                control.value,
                control.source,
            )
        )

    return NarrationPlan(
        text=cleaned_text,
        controls=tuple(remapped),
        pause_plan=pause_plan,
        source_text=text,
    )


def normalized_words(text: str) -> list[str]:
    return [match.group().lower() for match in _WORD_RE.finditer(text)]


def control_word_indices(text: str, control: NarrationControl) -> tuple[int, int]:
    word_matches = list(_WORD_RE.finditer(text))
    selected = [
        index
        for index, match in enumerate(word_matches)
        if match.end() > control.start_char and match.start() < control.end_char
    ]
    if not selected:
        raise ValueError(f"{control.kind} control does not cover any words")
    return selected[0], selected[-1]


def align_expected_words(expected: list[str], asr_words: list[dict]) -> dict[int, int]:
    actual = [word["normalized"] for word in asr_words]
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)
    mapping = {}
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            mapping[block.a + delta] = block.b + delta
    return mapping


def build_inpaint_template(
    tokens: torch.Tensor,
    frame_start: int,
    frame_end: int,
    new_core_frames: int,
    mask_id: int,
    transition_frames: int = 5,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Replace one token span with masks while preserving every outside token."""
    total_frames = tokens.shape[-1]
    if not 0 <= frame_start < frame_end <= total_frames:
        raise ValueError("Invalid controlled frame span")
    if new_core_frames < 1:
        raise ValueError("Controlled span must retain at least one frame")
    window_start = max(0, frame_start - transition_frames)
    window_end = min(total_frames, frame_end + transition_frames)
    left_transition = frame_start - window_start
    right_transition = window_end - frame_end
    new_window_frames = left_transition + new_core_frames + right_transition
    masks = torch.full(
        (tokens.shape[0], new_window_frames),
        mask_id,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    template = torch.cat(
        (tokens[:, :window_start], masks, tokens[:, window_end:]), dim=-1
    )
    new_core_start = window_start + left_transition
    return (
        template,
        (window_start, window_end),
        (
            new_core_start,
            new_core_start + new_core_frames,
        ),
    )


def _conditioned_text(
    text: str,
    control: NarrationControl,
    pronunciation: Optional[str | tuple[tuple[int, int, str], ...]] = None,
) -> str:
    before = text[: control.start_char]
    surface = text[control.start_char : control.end_char]
    after = text[control.end_char :]
    if isinstance(pronunciation, str):
        spoken = pronunciation
    elif pronunciation:
        pieces = []
        cursor = 0
        for start, end, replacement in pronunciation:
            pieces.extend((surface[cursor:start], replacement))
            cursor = end
        pieces.append(surface[cursor:])
        spoken = "".join(pieces)
    else:
        spoken = surface
    if control.kind == "emphasis":
        if control.value == "reduced":
            replacement = f"({spoken})"
        elif control.value == "strong":
            replacement = f"—{spoken if pronunciation else surface.upper()}—"
        else:
            replacement = f"—{spoken}—"
    elif control.kind == "aside":
        replacement = f"({spoken})"
    elif control.kind == "intonation":
        replacement = (spoken if pronunciation else surface.rstrip(".!?")) + (
            "?" if control.value == "rising" else "."
        )
    else:
        replacement = spoken
    return before + replacement + after


def _core_frame_target(control: NarrationControl, original_frames: int) -> int:
    if control.kind == "rate":
        factor = float(control.value)
    elif control.kind == "aside":
        factor = 1.12
    elif control.kind == "emphasis":
        factor = {"reduced": 1.1, "moderate": 0.87, "strong": 0.74}[str(control.value)]
    else:
        factor = 1.0
    return max(1, int(math.floor(original_frames / factor + 0.5)))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_tokens(model: OmniVoice, tokens: torch.Tensor) -> np.ndarray:
    return (
        model.audio_tokenizer.decode(
            tokens.to(model.audio_tokenizer.device).unsqueeze(0)
        )
        .audio_values[0]
        .detach()
        .cpu()
        .numpy()
        .squeeze(0)
    )


def _transcribe(aligner, audio: np.ndarray, sample_rate: int = 24000) -> dict[str, Any]:
    waveform = torch.from_numpy(audio.astype(np.float32, copy=False))
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=16000
        )
    segments, info = aligner.transcribe(
        waveform.numpy(),
        language="en",
        beam_size=5,
        word_timestamps=True,
    )
    words = []
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())
        for word in segment.words or []:
            normalized = normalized_words(word.word)
            if not normalized:
                continue
            words.append(
                {
                    "word": word.word.strip(),
                    "normalized": normalized[0],
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(word.probability),
                }
            )
    return {
        "text": " ".join(part for part in text_parts if part).strip(),
        "words": words,
        "normalized_words": [word["normalized"] for word in words],
        "language": info.language,
        "language_probability": float(info.language_probability),
    }


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


def _pitch_slope(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size < int(sample_rate * 0.12):
        return 0.0
    waveform = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate,
            frame_time=0.02,
            win_length=5,
            freq_low=70,
            freq_high=350,
        ).squeeze(0)
    except RuntimeError:
        return 0.0
    pitch = pitch[torch.isfinite(pitch) & (pitch > 0)]
    if pitch.numel() < 4:
        return 0.0
    split = max(1, pitch.numel() // 3)
    first = float(pitch[:split].median())
    last = float(pitch[-split:].median())
    return math.log2(max(last, 1.0) / max(first, 1.0)) * 12.0


def _aligned_control_audio(
    text: str,
    control: NarrationControl,
    transcript: dict[str, Any],
    audio: np.ndarray,
    sample_rate: int,
) -> Optional[dict[str, Any]]:
    expected = normalized_words(text)
    mapping = align_expected_words(expected, transcript["words"])
    first_word, last_word = control_word_indices(text, control)
    if control.kind == "intonation":
        # Sentence-ending pitch needs enough voiced context to measure and repaint.
        first_word = max(first_word, last_word - 2)
    required = range(first_word, last_word + 1)
    if any(index not in mapping for index in required):
        return None
    first = transcript["words"][mapping[first_word]]
    last = transcript["words"][mapping[last_word]]
    start_sample = max(0, int(first["start"] * sample_rate))
    end_sample = min(audio.size, int(last["end"] * sample_rate))
    segment = audio[start_sample:end_sample]
    return {
        "first_word_index": first_word,
        "last_word_index": last_word,
        "start_seconds": first["start"],
        "end_seconds": last["end"],
        "duration_seconds": max(0.0, last["end"] - first["start"]),
        "rms_db": _rms_db(segment),
        "pitch_slope_semitones": _pitch_slope(segment, sample_rate),
        "word_match_ratio": len(mapping) / max(1, len(expected)),
    }


def _score_candidate(
    text: str,
    control: NarrationControl,
    transcript: dict[str, Any],
    audio: np.ndarray,
    sample_rate: int,
    target_duration: float,
) -> tuple[float, dict[str, Any]]:
    metrics = _aligned_control_audio(text, control, transcript, audio, sample_rate)
    if metrics is None:
        return -1e9, {"alignment_failed": True}
    word_match = metrics["word_match_ratio"]
    duration_error = abs(metrics["duration_seconds"] - target_duration)
    score = word_match * 100.0 - duration_error * 12.0
    whole_rms = _rms_db(audio)
    prominence = metrics["rms_db"] - whole_rms
    if control.kind == "emphasis":
        weight = {"reduced": -2.0, "moderate": 1.0, "strong": 2.0}[str(control.value)]
        score += prominence * weight
    elif control.kind == "aside":
        score -= prominence * 1.5
    elif control.kind == "intonation":
        direction = 1.0 if control.value == "rising" else -1.0
        score += metrics["pitch_slope_semitones"] * direction * 3.0
    metrics.update(
        {
            "alignment_failed": False,
            "whole_rms_db": whole_rms,
            "prominence_db": prominence,
            "target_duration_seconds": target_duration,
            "duration_error_seconds": duration_error,
        }
    )
    return score, metrics


def _shift_fixed_spans(
    spans: tuple[tuple[int, int], ...],
    old_window: tuple[int, int],
    new_window_length: int,
) -> tuple[tuple[int, int], ...]:
    old_start, old_end = old_window
    delta = new_window_length - (old_end - old_start)
    shifted = []
    for start, end in spans:
        if end <= old_start:
            shifted.append((start, end))
        elif start >= old_end:
            shifted.append((start + delta, end + delta))
        else:
            raise ValueError(
                "Narration control is too close to a fixed pause-token span"
            )
    return tuple(shifted)


class NarrationController:
    """Apply local narration controls while reusing one voice-clone prompt."""

    def __init__(
        self,
        model: OmniVoice,
        aligner: Any = None,
        aligner_path: str | Path = "Systran/faster-whisper-small.en",
        aligner_device: str = "cpu",
        aligner_compute_type: str = "int8",
        local_files_only: bool = False,
    ):
        self.model = model
        self._aligner = aligner
        self.aligner_path = str(aligner_path)
        self.aligner_device = aligner_device
        self.aligner_compute_type = aligner_compute_type
        self.local_files_only = local_files_only

    @property
    def aligner(self):
        if self._aligner is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Narration controls require faster-whisper for word alignment"
                ) from exc
            self._aligner = WhisperModel(
                self.aligner_path,
                device=self.aligner_device,
                compute_type=self.aligner_compute_type,
                local_files_only=self.local_files_only,
            )
        return self._aligner

    def _resolve_prompt(
        self,
        voice_clone_prompt: Optional[VoiceClonePrompt],
        ref_audio,
        ref_text: Optional[str],
        preprocess_prompt: bool,
    ) -> VoiceClonePrompt:
        if voice_clone_prompt is not None:
            if ref_audio is not None or ref_text is not None:
                raise ValueError(
                    "Use voice_clone_prompt or ref_audio/ref_text, not both"
                )
            return voice_clone_prompt
        if ref_audio is None:
            raise ValueError("Narration controls require one clone reference")
        return self.model.create_voice_clone_prompt(
            ref_audio,
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )

    def _make_task(
        self,
        text: str,
        prompt: VoiceClonePrompt,
        language: Optional[str],
        config: OmniVoiceGenerationConfig,
        target_template: torch.Tensor,
        pause_spans: tuple[tuple[int, int], ...],
    ) -> GenerationTask:
        task = self.model._preprocess_all(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
            preprocess_prompt=config.preprocess_prompt,
            speed=1.0,
            duration=None,
        )
        task.target_lens = [target_template.shape[-1]]
        task.target_templates = [target_template]
        task.pause_spans = [pause_spans]
        task.controlled = [True]
        return task

    def generate(
        self,
        text: str,
        *,
        ref_audio=None,
        ref_text: Optional[str] = None,
        voice_clone_prompt: Optional[VoiceClonePrompt] = None,
        language: Optional[str] = "English",
        shorthand: bool = False,
        num_step: int = 32,
        candidates: int = 3,
        seed: int = 1234,
        preprocess_prompt: bool = True,
        pronunciations: Optional[dict[str, str]] = None,
        trace: bool = False,
    ) -> NarrationResult:
        plan = parse_narration_controls(text, shorthand=shorthand)
        return self.generate_plan(
            plan,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_clone_prompt=voice_clone_prompt,
            language=language,
            num_step=num_step,
            candidates=candidates,
            seed=seed,
            preprocess_prompt=preprocess_prompt,
            pronunciations=pronunciations,
            trace=trace,
        )

    @torch.inference_mode()
    def generate_plan(
        self,
        plan: NarrationPlan,
        *,
        ref_audio=None,
        ref_text: Optional[str] = None,
        voice_clone_prompt: Optional[VoiceClonePrompt] = None,
        language: Optional[str] = "English",
        num_step: int = 32,
        candidates: int = 3,
        seed: int = 1234,
        preprocess_prompt: bool = True,
        pronunciations: Optional[dict[str, str]] = None,
        trace: bool = False,
    ) -> NarrationResult:
        if not isinstance(plan, NarrationPlan):
            raise TypeError("plan must be a NarrationPlan")
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        resolved_pronunciations = _resolve_pronunciations(plan, pronunciations)
        prompt = self._resolve_prompt(
            voice_clone_prompt, ref_audio, ref_text, preprocess_prompt
        )
        config = OmniVoiceGenerationConfig(
            num_step=num_step, preprocess_prompt=preprocess_prompt
        )

        _set_seed(seed)
        baseline_text, baseline_pause_plan = _apply_pronunciations_to_pause_plan(
            plan.text, plan.pause_plan, pronunciations
        )
        baseline_task = self.model._preprocess_all(
            text=baseline_text,
            language=language,
            voice_clone_prompt=prompt,
            preprocess_prompt=preprocess_prompt,
            speed=1.0,
            duration=None,
            pause_plan=baseline_pause_plan if baseline_pause_plan.pauses else None,
        )
        short_idx, long_idx = baseline_task.get_indices(
            config, self.model.audio_tokenizer.config.frame_rate
        )
        if long_idx or short_idx != [0]:
            raise ValueError("Narration controls support one short-form item only")
        baseline_trace = [] if trace else None
        baseline_tokens = self.model._generate_iterative(
            baseline_task, config, trace=baseline_trace
        )[0]
        baseline_raw = _decode_tokens(self.model, baseline_tokens)
        baseline_audio = self.model._decode_and_post_process(
            baseline_tokens,
            baseline_task.ref_rms[0],
            config,
            preserve_internal_silence=bool(plan.pause_plan.pauses),
        )

        current_tokens = baseline_tokens
        current_raw = baseline_raw
        pause_spans = baseline_task.pause_spans[0]
        expected_words = normalized_words(plan.text)
        control_reports = []
        selected_traces = []
        frame_rate = self.model.audio_tokenizer.config.frame_rate

        for control_index, control in enumerate(plan.controls):
            transcript = _transcribe(
                self.aligner, current_raw, self.model.sampling_rate
            )
            mapping = align_expected_words(expected_words, transcript["words"])
            first_word, last_word = control_word_indices(plan.text, control)
            if any(index not in mapping for index in range(first_word, last_word + 1)):
                raise RuntimeError(f"Could not align words for {control.kind} control")
            first = transcript["words"][mapping[first_word]]
            last = transcript["words"][mapping[last_word]]
            source_metrics = _aligned_control_audio(
                plan.text,
                control,
                transcript,
                current_raw,
                self.model.sampling_rate,
            )
            assert source_metrics is not None
            source_metrics["whole_rms_db"] = _rms_db(current_raw)
            source_metrics["prominence_db"] = (
                source_metrics["rms_db"] - source_metrics["whole_rms_db"]
            )
            frame_start = max(0, int(math.floor(first["start"] * frame_rate)))
            frame_end = min(
                current_tokens.shape[-1],
                max(frame_start + 1, int(math.ceil(last["end"] * frame_rate))),
            )
            original_frames = frame_end - frame_start
            new_core_frames = _core_frame_target(control, original_frames)
            template, old_window, new_core_span = build_inpaint_template(
                current_tokens,
                frame_start,
                frame_end,
                new_core_frames,
                self.model.config.audio_mask_id,
            )
            new_window_length = (
                old_window[1]
                - old_window[0]
                + template.shape[-1]
                - current_tokens.shape[-1]
            )
            pause_spans = _shift_fixed_spans(pause_spans, old_window, new_window_length)
            pronunciation = resolved_pronunciations.get(control_index)
            conditioned_text = _conditioned_text(
                plan.text,
                control,
                pronunciation=pronunciation,
            )
            conditioned_text = _apply_pronunciations(
                conditioned_text, pronunciations
            )
            target_duration = (
                source_metrics["duration_seconds"]
                if control.kind == "intonation"
                else new_core_frames / frame_rate
            )
            candidate_reports = []
            best = None

            for candidate_index in range(candidates):
                candidate_seed = seed + (control_index + 1) * 100 + candidate_index
                _set_seed(candidate_seed)
                task = self._make_task(
                    conditioned_text,
                    prompt,
                    language,
                    config,
                    template,
                    pause_spans,
                )
                candidate_trace = [] if trace else None
                candidate_tokens = self.model._generate_iterative(
                    task, config, trace=candidate_trace
                )[0]
                fixed = template != self.model.config.audio_mask_id
                fixed_tokens_unchanged = bool(
                    torch.equal(candidate_tokens[fixed], template[fixed])
                )
                candidate_raw = _decode_tokens(self.model, candidate_tokens)
                candidate_transcript = _transcribe(
                    self.aligner, candidate_raw, self.model.sampling_rate
                )
                score, metrics = _score_candidate(
                    plan.text,
                    control,
                    candidate_transcript,
                    candidate_raw,
                    self.model.sampling_rate,
                    target_duration,
                )
                candidate_report = {
                    "candidate": candidate_index,
                    "seed": candidate_seed,
                    "score": score,
                    "transcript": candidate_transcript["text"],
                    "normalized_words": candidate_transcript["normalized_words"],
                    "fixed_tokens_unchanged": fixed_tokens_unchanged,
                    "fixed_token_fraction": float(fixed.float().mean().item()),
                    "metrics": metrics,
                }
                candidate_reports.append(candidate_report)
                if best is None or score > best[0]:
                    best = (
                        score,
                        candidate_tokens,
                        candidate_raw,
                        candidate_index,
                        candidate_trace[0] if candidate_trace else None,
                    )

            assert best is not None
            current_tokens = best[1]
            current_raw = best[2]
            control_reports.append(
                {
                    "control": asdict(control),
                    "source_frame_span": [frame_start, frame_end],
                    "source_frames": original_frames,
                    "target_frames": new_core_frames,
                    "pronunciation": pronunciation,
                    "conditioned_text": conditioned_text,
                    "source_metrics": source_metrics,
                    "inpaint_window": list(old_window),
                    "new_core_span": list(new_core_span),
                    "selected_candidate": best[3],
                    "candidates": candidate_reports,
                }
            )
            if trace:
                control_reports[-1]["selected_trace_index"] = control_index
                selected_traces.append(best[4])

        final_audio = self.model._decode_and_post_process(
            current_tokens,
            baseline_task.ref_rms[0],
            config,
            preserve_internal_silence=bool(plan.pause_plan.pauses),
        )
        final_transcript = _transcribe(
            self.aligner, current_raw, self.model.sampling_rate
        )
        pronunciation_report = {
            str(surface): _normalize_cmu_pronunciation(value)
            for surface, value in (pronunciations or {}).items()
        }
        for index, value in resolved_pronunciations.items():
            surface = plan.text[
                plan.controls[index].start_char : plan.controls[index].end_char
            ]
            if isinstance(value, str):
                pronunciation_report[surface] = value
            else:
                pronunciation_report[surface] = [
                    {
                        "surface": surface[start:end],
                        "pronunciation": pronunciation,
                    }
                    for start, end, pronunciation in value
                ]
        report = {
            "format": "omnivoice_single_reference_narration_v2",
            "single_reference": True,
            "reference_switching": False,
            "seed": seed,
            "num_step": num_step,
            "candidate_count": candidates,
            "pronunciations": pronunciation_report,
            "baseline_frames": baseline_tokens.shape[-1],
            "final_frames": current_tokens.shape[-1],
            "baseline_transcript": _transcribe(
                self.aligner, baseline_raw, self.model.sampling_rate
            )["text"],
            "final_transcript": final_transcript["text"],
            "final_normalized_words": final_transcript["normalized_words"],
            "expected_normalized_words": expected_words,
            "word_sequence_matches": final_transcript["normalized_words"]
            == expected_words,
            "controls": control_reports,
        }
        return NarrationResult(
            audio=final_audio,
            baseline_audio=baseline_audio,
            sample_rate=self.model.sampling_rate,
            plan=plan,
            tokens=current_tokens.detach().cpu(),
            baseline_tokens=baseline_tokens.detach().cpu(),
            report=report,
            traces=(
                {
                    "format": "omnivoice_narration_traces_v1",
                    "baseline": baseline_trace[0] if baseline_trace else None,
                    "controls": selected_traces,
                }
                if trace
                else None
            ),
        )
