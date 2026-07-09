from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass
class DocxParagraph:
    index: int
    text: str
    style: Optional[str] = None


@dataclass
class DocxDocument:
    path: str
    sha256: str
    paragraphs: List[DocxParagraph]

    @property
    def plain_text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs if paragraph.text)


def _read_docx_xml(path: Path) -> bytes:
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Unsupported manuscript type: {path.suffix}")
    with zipfile.ZipFile(path) as archive:
        try:
            return archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError(f"DOCX missing word/document.xml: {path}") from exc


def _paragraph_style(paragraph) -> Optional[str]:
    p_pr = paragraph.find(f"{WORD_NS}pPr")
    if p_pr is None:
        return None
    style = p_pr.find(f"{WORD_NS}pStyle")
    if style is None:
        return None
    return style.attrib.get(f"{WORD_NS}val")


def _paragraph_text(paragraph) -> str:
    parts: List[str] = []
    for node in paragraph.iter():
        tag = node.tag
        if tag == f"{WORD_NS}t" and node.text:
            parts.append(node.text)
        elif tag == f"{WORD_NS}tab":
            parts.append("\t")
        elif tag in {f"{WORD_NS}br", f"{WORD_NS}cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def extract_docx_structure(path: str | Path) -> DocxDocument:
    docx_path = Path(path)
    content = docx_path.read_bytes()
    xml = _read_docx_xml(docx_path)
    root = ElementTree.fromstring(xml)

    paragraphs: List[DocxParagraph] = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        text = _paragraph_text(paragraph)
        if not text:
            continue
        paragraphs.append(
            DocxParagraph(
                index=len(paragraphs),
                text=text,
                style=_paragraph_style(paragraph),
            )
        )

    if not paragraphs:
        raise ValueError(f"No readable paragraphs found in DOCX: {docx_path}")

    return DocxDocument(
        path=str(docx_path),
        sha256=hashlib.sha256(content).hexdigest(),
        paragraphs=paragraphs,
    )
