from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import replace
from typing import Dict, Iterable, List, Optional

from omnivoice.narration.schema import DEFAULT_PRESETS, NarrationPlan, NarrationSegment

_PAUSE_RE = re.compile(r"\[pause\s*:\s*([0-9]+(?:\.[0-9]+)?)(ms|s)?\]", re.I)
_SPEED_RE = re.compile(r"\[speed\s*:\s*([0-9]+(?:\.[0-9]+)?)\]", re.I)
_SECTION_RE = re.compile(r"\[section(?:\s*:\s*([^\]]+))?\]", re.I)
_MARKER_RE = re.compile(r"\[(?:pause\s*:[^\]]+|speed\s*:[^\]]+|section(?::[^\]]+)?)\]", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", re.I)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BULLET_PREFIX_RE = re.compile(
    r"^\s*(?:[-*•·●▪▫◦‣►▶◆◇■□✓✔☑→]+|\d+[.)]|[A-Za-z][.)])\s+"
)
_ARTIFACT_ONLY_RE = re.compile(
    r"^\s*(?:[-*_=\u2500-\u257f•·●▪▫◦‣►▶◆◇■□✓✔☑|\\/]+|\d+\s*/\s*\d+|\d+)\s*$"
)
_NON_NARRATION_PREFIX_RE = re.compile(
    r"^\s*(?:fonte|source|refer[eê]ncia|references?|link|url|imagem|image|figura|"
    r"figure|gr[aá]fico|chart|tabela|table)\s*:",
    re.I,
)
_MEDIA_FILE_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|svg|pdf|pptx?|docx?)\b", re.I)
_SLIDE_LABEL_RE = re.compile(
    r"^\s*(?:[#>*_\-\s]*)(?:slide|sl\.|p[aá]gina|page|tela)\s*\d+"
    r"(?:\s*[:\-–—.)]\s*.*)?(?:[*_\s]*)$",
    re.I,
)
_SECTION_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s*)?([A-ZÁ-Ú0-9][^.!?]{0,80})\s*$")


def _stable_id(index: int, text: str, speed: float, pause_after_ms: int) -> str:
    raw = f"{index}|{text}|{speed:.4f}|{pause_after_ms}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _parse_pause_ms(value: str, unit: Optional[str]) -> int:
    number = float(value)
    if unit and unit.lower() == "s":
        number *= 1000
    return max(0, int(round(number)))


def _strip_slide_labels(lines: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        if _SLIDE_LABEL_RE.match(line):
            continue
        cleaned.append(line)
    return cleaned


def _drop_symbol_noise(text: str) -> str:
    kept: List[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category in {"So", "Sk"}:
            kept.append(" ")
            continue
        kept.append(char)
    return "".join(kept)


def _clean_slide_artifact_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).strip()
    if not line:
        return ""
    if _SPEED_RE.fullmatch(line) or _SECTION_RE.fullmatch(line):
        return line
    if _ARTIFACT_ONLY_RE.fullmatch(line):
        return ""
    if _NON_NARRATION_PREFIX_RE.match(line):
        return ""
    if _MEDIA_FILE_RE.search(line) and len(line.split()) <= 4:
        return ""

    line = _MARKDOWN_LINK_RE.sub(r"\1", line)
    line = _URL_RE.sub(" ", line)
    line = _HTML_TAG_RE.sub(" ", line)
    line = _BULLET_PREFIX_RE.sub("", line)
    line = re.sub(r"^[#*_`~>\s]+", "", line)
    line = re.sub(r"[*_`~]{1,3}", "", line)
    line = re.sub(r"\s*(?:->|=>|→|⇒|➜|➔|➡)\s*", ", ", line)
    line = re.sub(r"[|]{1,}", ", ", line)
    line = _drop_symbol_noise(line)
    line = re.sub(r"\s+", " ", line).strip(" -–—:;")
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", line):
        return ""
    alpha_num = sum(1 for char in line if char.isalnum())
    if len(line) >= 4 and alpha_num / max(len(line), 1) < 0.35:
        return ""
    return line


def _split_paragraph(paragraph: str) -> List[str]:
    paragraph = " ".join(paragraph.split())
    if not paragraph:
        return []

    pieces = re.findall(
        r".+?(?:[.!?。！？](?:\s*\[pause\s*:\s*[0-9]+(?:\.[0-9]+)?(?:ms|s)?\])?|$)(?=\s+|$)",
        paragraph,
        flags=re.I,
    )
    if not pieces:
        pieces = _SENTENCE_SPLIT_RE.split(paragraph)
    segments: List[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= 260:
            segments.append(piece)
            continue

        chunks = re.split(r"(?<=[,;:])\s+", piece)
        current = ""
        for chunk in chunks:
            next_text = f"{current} {chunk}".strip()
            if len(next_text) > 260 and current:
                segments.append(current)
                current = chunk
            else:
                current = next_text
        if current:
            segments.append(current)
    return segments


def _base_pause_for_text(text: str, paragraph_end: bool, section_break: bool, preset) -> int:
    if section_break:
        return preset.section_ms
    if paragraph_end:
        return preset.paragraph_ms
    if "..." in text or "…" in text:
        return preset.ellipsis_ms
    if text.rstrip().endswith((",", ";", ":")):
        return preset.comma_ms
    return preset.sentence_ms


def _remove_markers(text: str) -> str:
    return " ".join(_MARKER_RE.sub("", text).split())


def _clean_segment_text(text: str) -> str:
    text = _remove_markers(text)
    text = _clean_slide_artifact_line(text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:]){3,}", r"\1", text)
    return text.strip()


def parse_narration_text(
    text: str,
    preset_name: str = "Presentation",
    global_speed: float = 0.92,
    remove_slide_labels: bool = True,
    clean_slide_artifacts: bool = True,
    pause_overrides: Optional[Dict[str, int]] = None,
) -> NarrationPlan:
    """Parse a long narration script into editable TTS segments."""

    preset = replace(DEFAULT_PRESETS.get(preset_name, DEFAULT_PRESETS["Presentation"]))
    if pause_overrides:
        for key, value in pause_overrides.items():
            attr = f"{key}_ms" if not key.endswith("_ms") else key
            if hasattr(preset, attr) and value is not None:
                setattr(preset, attr, int(value))

    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = _strip_slide_labels(raw_lines) if remove_slide_labels else raw_lines

    segments: List[NarrationSegment] = []
    paragraph_lines: List[str] = []
    current_speed = float(global_speed or 1.0)
    current_section: Optional[str] = None
    pending_section_break = False

    def flush_paragraph(section_break: bool = False) -> None:
        nonlocal paragraph_lines, pending_section_break
        if not paragraph_lines:
            pending_section_break = pending_section_break or section_break
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
        paragraph_lines = []
        split_items = _split_paragraph(paragraph)
        for idx, item in enumerate(split_items):
            pause_match = _PAUSE_RE.search(item)
            speed_match = _SPEED_RE.search(item)
            segment_speed = current_speed
            if speed_match:
                segment_speed = float(speed_match.group(1))
            clean = _clean_segment_text(item) if clean_slide_artifacts else _remove_markers(item)
            if not clean:
                continue
            is_last = idx == len(split_items) - 1
            pause_ms = _base_pause_for_text(
                clean,
                paragraph_end=is_last,
                section_break=(section_break or pending_section_break) and is_last,
                preset=preset,
            )
            if pause_match:
                pause_ms = _parse_pause_ms(pause_match.group(1), pause_match.group(2))
            index = len(segments)
            segments.append(
                NarrationSegment(
                    id=_stable_id(index, clean, segment_speed, pause_ms),
                    index=index,
                    text=clean,
                    pause_after_ms=pause_ms,
                    speed=segment_speed,
                    section=current_section,
                )
            )
        pending_section_break = False

    for raw_line in lines:
        line = raw_line.strip()
        if clean_slide_artifacts:
            line = _clean_slide_artifact_line(line)
        if not line:
            flush_paragraph()
            continue

        speed_line = _SPEED_RE.fullmatch(line)
        if speed_line:
            current_speed = float(speed_line.group(1))
            continue

        section_line = _SECTION_RE.fullmatch(line)
        if section_line:
            flush_paragraph(section_break=True)
            current_section = (section_line.group(1) or current_section or "Section").strip()
            pending_section_break = True
            continue

        heading = _SECTION_HEADING_RE.match(line)
        if heading and len(line.split()) <= 8 and not re.search(r"[.!?]$", line):
            flush_paragraph(section_break=True)
            current_section = heading.group(1).strip(" *#")
            pending_section_break = True
            continue

        paragraph_lines.append(line)

    flush_paragraph()

    return NarrationPlan(
        preset=preset.name,
        settings={
            "global_speed": float(global_speed or 1.0),
            "remove_slide_labels": bool(remove_slide_labels),
            "clean_slide_artifacts": bool(clean_slide_artifacts),
            "pauses": {
                "comma": preset.comma_ms,
                "sentence": preset.sentence_ms,
                "ellipsis": preset.ellipsis_ms,
                "paragraph": preset.paragraph_ms,
                "section": preset.section_ms,
            },
        },
        segments=segments,
    )
