#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Book text extraction and segmentation for audiobooks.

Kept dependency-light on purpose: only ``pypdf`` is needed (``.pdf``), and only
when a PDF is actually read. ``.epub``, ``.docx`` and ``.html`` are parsed with
the Python standard library. Legacy binary ``.doc`` needs antiword or
LibreOffice, so it degrades with a clear message when neither is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

SUPPORTED_EXTS = (".txt", ".md", ".pdf", ".epub", ".doc", ".docx", ".html", ".htm")

_PAGE_BREAK_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…:;])\s+|(?<=[。！？])")
_SPACE_RE = re.compile(r"[ \t 　]+")


class ExtractionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# HTML (used directly and for EPUB chapters)
# ---------------------------------------------------------------------------


class _HTMLText(HTMLParser):
    _BLOCK = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "td",
        "section",
        "article",
        "blockquote",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "figcaption",
        "pre",
        "table",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "noscript"):
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "noscript"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_to_text(markup: str | bytes) -> str:
    if isinstance(markup, bytes):
        markup = markup.decode("utf-8", errors="replace")
    parser = _HTMLText()
    parser.feed(markup)
    parser.close()
    return "".join(parser.parts)


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _txt(path: Path) -> str:
    return _decode(path.read_bytes())


def _pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - install guard
        raise ExtractionError(
            "PDF support needs the 'pypdf' package. Install it with "
            "'pip install pypdf' (or 'pip install omnivoice[audiobook]')."
        ) from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ExtractionError("This PDF is password protected.") from exc

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def _docx(path: Path) -> str:
    """Read a .docx by pulling text runs out of word/document.xml."""
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ExtractionError("This file is not a readable .docx document.") from exc

    root = ET.fromstring(xml)
    out: list[str] = []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t" and node.text:
            out.append(node.text)
        elif tag in ("p", "br", "tab"):
            out.append("\n" if tag != "tab" else "\t")
    return "".join(out)


def _doc(path: Path) -> str:
    """Legacy binary .doc: antiword, LibreOffice, or a mislabelled .docx."""
    antiword = shutil.which("antiword")
    if antiword:
        result = subprocess.run(
            [antiword, str(path)], capture_output=True, text=True, errors="replace"
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout

    if zipfile.is_zipfile(path):
        return _docx(path)

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    tmp,
                    str(path),
                ],
                capture_output=True,
            )
            converted = next(Path(tmp).glob("*.docx"), None)
            if converted:
                return _docx(converted)

    raise ExtractionError(
        "Legacy .doc files can't be read without antiword or LibreOffice. "
        "Open the file in Word/LibreOffice and save it as .docx, then drop it again."
    )


def _epub(path: Path) -> str:
    """Read an EPUB in spine order using only the standard library."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

        opf_name = None
        try:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            for node in container.iter():
                if node.tag.rsplit("}", 1)[-1] == "rootfile":
                    opf_name = node.attrib.get("full-path")
                    break
        except (KeyError, ET.ParseError):
            pass
        if not opf_name:
            opf_name = next((n for n in names if n.lower().endswith(".opf")), None)

        ordered: list[str] = []
        if opf_name:
            try:
                opf = ET.fromstring(archive.read(opf_name))
                base = opf_name.rsplit("/", 1)[0] + "/" if "/" in opf_name else ""
                manifest: dict[str, str] = {}
                spine: list[str] = []
                for node in opf.iter():
                    tag = node.tag.rsplit("}", 1)[-1]
                    if tag == "item":
                        media = node.attrib.get("media-type", "")
                        if "html" in media or node.attrib.get(
                            "href", ""
                        ).lower().endswith((".xhtml", ".html", ".htm")):
                            manifest[node.attrib.get("id", "")] = node.attrib.get(
                                "href", ""
                            )
                    elif tag == "itemref":
                        spine.append(node.attrib.get("idref", ""))
                for idref in spine:
                    href = manifest.get(idref)
                    if href:
                        ordered.append(base + href)
            except (KeyError, ET.ParseError):
                ordered = []

        if not ordered:
            ordered = sorted(
                n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))
            )

        chapters = []
        for name in ordered:
            try:
                chapters.append(_html_to_text(archive.read(name)))
            except KeyError:
                continue
    return "\n\n".join(chapters)


_READERS = {
    ".txt": _txt,
    ".md": _txt,
    ".text": _txt,
    ".pdf": _pdf,
    ".epub": _epub,
    ".docx": _docx,
    ".doc": _doc,
    ".html": lambda p: _html_to_text(p.read_bytes()),
    ".htm": lambda p: _html_to_text(p.read_bytes()),
}


# ---------------------------------------------------------------------------
# Cleanup + segmentation
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\t", " ").replace(" ", " ").replace("　", " ")
    # Join hard-wrapped lines inside a paragraph, keep blank lines as breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = _SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _PAGE_BREAK_RE.sub("\n\n", text)
    return text.strip()


def extract_text(path: str | Path) -> str:
    file_path = Path(path)
    ext = file_path.suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ExtractionError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(set(_READERS)))}"
        )
    text = clean_text(reader(file_path))
    if not text.strip():
        raise ExtractionError(
            "No text could be read from this file. If it is a scanned PDF, it "
            "needs OCR first."
        )
    return text


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Hard-split a sentence that is longer than the budget, on soft breaks."""
    pieces: list[str] = []
    remainder = sentence
    while len(remainder) > max_chars:
        window = remainder[:max_chars]
        cut = max(
            window.rfind(", "),
            window.rfind("; "),
            window.rfind("，"),
            window.rfind("、"),
            window.rfind(" "),
        )
        if cut < max_chars // 3:
            cut = max_chars
        pieces.append(remainder[:cut].strip())
        remainder = remainder[cut:].strip()
    if remainder:
        pieces.append(remainder)
    return pieces


def split_segments(text: str, max_chars: int = 300) -> list[str]:
    """Split book text into TTS-sized segments, keeping sentences intact."""
    max_chars = max(80, int(max_chars))
    segments: list[str] = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [
            s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()
        ]
        if not sentences:
            continue

        buffer = ""
        for sentence in sentences:
            for part in _split_long(sentence, max_chars):
                if not buffer:
                    buffer = part
                elif len(buffer) + 1 + len(part) <= max_chars:
                    buffer = f"{buffer} {part}"
                else:
                    segments.append(buffer)
                    buffer = part
        if buffer:
            segments.append(buffer)

    return segments
