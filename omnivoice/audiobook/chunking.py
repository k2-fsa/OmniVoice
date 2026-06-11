from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from omnivoice.audiobook.docx import DocxDocument, DocxParagraph


@dataclass
class AudiobookChunk:
    id: str
    index: int
    title: str
    text: str
    word_count: int
    paragraph_start: int
    paragraph_end: int
    previous_summary: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ChunkingConfig:
    max_words: int = 2400
    target_words: int = 1800
    overlap_summary_words: int = 80


def _count_words(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def _chunk_id(index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{index:04d}_{digest}"


def _summary(text: str, max_words: int) -> str:
    words = [word for word in text.split() if word.strip()]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[-max_words:])


def _split_oversized_paragraph(paragraph: DocxParagraph, max_words: int) -> List[DocxParagraph]:
    words = paragraph.text.split()
    if len(words) <= max_words:
        return [paragraph]
    pieces: List[DocxParagraph] = []
    for offset in range(0, len(words), max_words):
        pieces.append(
            DocxParagraph(
                index=paragraph.index,
                text=" ".join(words[offset : offset + max_words]),
                style=paragraph.style,
            )
        )
    return pieces


def chunk_docx_document(
    document: DocxDocument,
    config: Optional[ChunkingConfig] = None,
) -> List[AudiobookChunk]:
    config = config or ChunkingConfig()
    if config.max_words < 200:
        raise ValueError("max_words must be at least 200 for audiobook planning chunks")
    if config.target_words > config.max_words:
        raise ValueError("target_words cannot exceed max_words")

    expanded: List[DocxParagraph] = []
    oversized_indices: set[int] = set()
    for paragraph in document.paragraphs:
        pieces = _split_oversized_paragraph(paragraph, config.max_words)
        if len(pieces) > 1:
            oversized_indices.add(paragraph.index)
        expanded.extend(pieces)

    chunks: List[AudiobookChunk] = []
    current: List[DocxParagraph] = []
    current_words = 0
    previous_summary: Optional[str] = None

    def flush() -> None:
        nonlocal current, current_words, previous_summary
        if not current:
            return
        text = "\n\n".join(paragraph.text for paragraph in current).strip()
        index = len(chunks)
        warnings: List[str] = []
        if any(paragraph.index in oversized_indices for paragraph in current):
            warnings.append("oversized_paragraph_split")
        chunk = AudiobookChunk(
            id=_chunk_id(index, text),
            index=index,
            title=f"Bloco {index + 1}",
            text=text,
            word_count=_count_words(text),
            paragraph_start=current[0].index,
            paragraph_end=current[-1].index,
            previous_summary=previous_summary,
            warnings=warnings,
        )
        chunks.append(chunk)
        previous_summary = _summary(text, config.overlap_summary_words)
        current = []
        current_words = 0

    for paragraph in expanded:
        words = _count_words(paragraph.text)
        if current and current_words + words > config.max_words:
            flush()
        current.append(paragraph)
        current_words += words
        if current_words >= config.target_words:
            flush()

    flush()
    return chunks
