#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from html import escape
import logging
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

import inflect
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.text import add_punctuation, chunk_text_punctuation


LOG = logging.getLogger("omnivoice.openai_tts_server")
BACKEND_MODEL_ID = os.getenv("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
API_MODEL_ID = os.getenv("OMNIVOICE_API_MODEL_ID", "omnivoice")
DEFAULT_HOST = os.getenv("OMNIVOICE_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("OMNIVOICE_PORT", "6655"))
DEFAULT_VOICE = os.getenv("OMNIVOICE_DEFAULT_VOICE", "alloy")
DEFAULT_AUDIO_CHUNK_DURATION = float(
    os.getenv("OMNIVOICE_AUDIO_CHUNK_DURATION", "15.0")
)
DEFAULT_AUDIO_CHUNK_THRESHOLD = float(
    os.getenv("OMNIVOICE_AUDIO_CHUNK_THRESHOLD", "30.0")
)
DEFAULT_SENTENCE_CHUNKING_MIN_CHARS = int(
    os.getenv("OMNIVOICE_SENTENCE_CHUNKING_MIN_CHARS", "280")
)
MAX_SANITIZED_INPUT_CHARS = int(os.getenv("OMNIVOICE_MAX_INPUT_CHARS", "12000"))
MAX_RAW_INPUT_CHARS = int(
    os.getenv("OMNIVOICE_MAX_RAW_INPUT_CHARS", str(MAX_SANITIZED_INPUT_CHARS * 2))
)
DEBUG_PREVIEW_CHARS = int(os.getenv("OMNIVOICE_DEBUG_PREVIEW_CHARS", "160"))
LOCAL_VOICE_REFERENCE_ROOT = Path(
    os.getenv("OMNIVOICE_LOCAL_VOICE_ROOT", "/home/op/Libro-Gregoria-Variacion/audio")
)

VALID_TLDS = [
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "int",
    "biz",
    "info",
    "name",
    "pro",
    "coop",
    "museum",
    "travel",
    "jobs",
    "mobi",
    "tel",
    "asia",
    "cat",
    "xxx",
    "aero",
    "arpa",
    "bg",
    "br",
    "ca",
    "cn",
    "de",
    "es",
    "eu",
    "fr",
    "in",
    "it",
    "jp",
    "mx",
    "nl",
    "ru",
    "uk",
    "us",
    "io",
    "co",
]
VALID_UNITS = {
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "in": "inch",
    "ft": "foot",
    "yd": "yard",
    "mi": "mile",
    "g": "gram",
    "kg": "kilogram",
    "mg": "milligram",
    "s": "second",
    "ms": "millisecond",
    "min": "minute",
    "h": "hour",
    "l": "liter",
    "ml": "milliliter",
    "cl": "centiliter",
    "dl": "deciliter",
    "kph": "kilometer per hour",
    "mph": "mile per hour",
    "mi/h": "mile per hour",
    "m/s": "meter per second",
    "km/h": "kilometer per hour",
    "mm/s": "millimeter per second",
    "cm/s": "centimeter per second",
    "ft/s": "feet per second",
    "cm/h": "centimeter per hour",
    "degc": "degree celsius",
    "degf": "degree fahrenheit",
    "°c": "degree celsius",
    "°f": "degree fahrenheit",
    "pa": "pascal",
    "kpa": "kilopascal",
    "mpa": "megapascal",
    "atm": "atmosphere",
    "hz": "hertz",
    "khz": "kilohertz",
    "mhz": "megahertz",
    "ghz": "gigahertz",
    "v": "volt",
    "kv": "kilovolt",
    "mv": "megavolt",
    "a": "amp",
    "ma": "milliamp",
    "ka": "kiloamp",
    "w": "watt",
    "kw": "kilowatt",
    "mw": "megawatt",
    "j": "joule",
    "kj": "kilojoule",
    "mj": "megajoule",
    "ω": "ohm",
    "kω": "kiloohm",
    "mω": "megaohm",
    "f": "farad",
    "µf": "microfarad",
    "uf": "microfarad",
    "nf": "nanofarad",
    "pf": "picofarad",
    "b": "bit",
    "kb": "kilobit",
    "mb": "megabit",
    "gb": "gigabit",
    "tb": "terabit",
    "pb": "petabit",
    "kbps": "kilobit per second",
    "mbps": "megabit per second",
    "gbps": "gigabit per second",
    "tbps": "terabit per second",
    "px": "pixel",
}
MONEY_UNITS = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}
INFLECT_ENGINE = inflect.engine()

SMART_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)
CJK_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "、": ",",
        "。": ".",
        "，": ",",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
    }
)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MODEL_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^<>|\r\n]{1,128}\|>")
HTML_TAG_PATTERN = re.compile(r"</?[^>\r\n]{1,256}>")
URL_PATTERN = re.compile(
    r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+(\.(?:"
    + "|".join(VALID_TLDS)
    + r"))+|[0-9]{1,3}(?:\.[0-9]{1,3}){3})(:[0-9]+)?([/?][^\s]*)?",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)
PHONE_GROUP_PATTERN = re.compile(
    r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})"
)
GENERIC_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d(). \-]{7,}\d)")
UNIT_PATTERN = re.compile(
    r"((?<!\w)([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
    + "|".join(sorted(VALID_UNITS, key=len, reverse=True))
    + r")(?=[^\w\d]|\b)",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"([0-9]{1,2}\s*:\s*[0-9]{2}(?:\s*:\s*[0-9]{2})?)(\s*(?:pm|am)\b)?",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(
    r"(-?)(["
    + "".join(MONEY_UNITS.keys())
    + r"])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)
PROTECTED_TAG_PATTERN = re.compile(r"\[[^\[\]\r\n]{1,128}\]")
WHITESPACE_PATTERN = re.compile(r"\s+")
OPTIONAL_PLURALIZATION_PATTERN = re.compile(r"\(s\)")
THOUSANDS_SEPARATOR_PATTERN = re.compile(r"(?<=\d),(?=\d)")
RANGE_DASH_PATTERN = re.compile(r"(?<=\d)-(?=\d)")
TRAILING_URL_PUNCTUATION = ".,!?;:)"
SYMBOL_REPLACEMENTS = {
    "~": " ",
    "@": " at ",
    "#": " number ",
    "$": " dollar ",
    "&": " and ",
    "%": " percent ",
    "^": " ",
    "*": " ",
    "_": " ",
    "|": " ",
    "\\": " ",
    "/": " slash ",
    "=": " equals ",
    "+": " plus ",
    "<": " less than ",
    ">": " greater than ",
}

@dataclass(frozen=True, slots=True)
class VoiceOptionDefinition:
    id: str
    fallback_name: str
    fallback_instruct: Optional[str]
    sample_file: Optional[str] = None
    sample_ref_text: Optional[str] = None
    sample_label: Optional[str] = None
    sample_duration_seconds: Optional[float] = None
    default_language: Optional[str] = None

    @property
    def sample_path(self) -> Optional[Path]:
        if self.sample_file is None:
            return None
        return LOCAL_VOICE_REFERENCE_ROOT / self.sample_file

    def has_local_sample(self) -> bool:
        sample_path = self.sample_path
        return sample_path is not None and sample_path.is_file()

    def display_name(self) -> str:
        if self.has_local_sample() and self.sample_label is not None:
            duration = (
                f" ({int(round(self.sample_duration_seconds))}s local reference)"
                if self.sample_duration_seconds is not None
                else " (local reference)"
            )
            return f"{self.id} -> {self.sample_label}{duration}"
        return self.fallback_name


@dataclass(frozen=True, slots=True)
class ResolvedVoice:
    voice_id: str
    display_name: str
    instruct: Optional[str]
    ref_audio_path: Optional[Path]
    ref_text: Optional[str]
    default_language: Optional[str]


VOICE_OPTIONS: list[VoiceOptionDefinition] = [
    VoiceOptionDefinition(
        id="alloy",
        fallback_name="alloy (male, moderate pitch, american accent)",
        fallback_instruct="male, moderate pitch, american accent",
        sample_file="omnivoice_refs/britishMan_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Man",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="echo",
        fallback_name="echo (male, low pitch, british accent)",
        fallback_instruct="male, low pitch, british accent",
        sample_file="omnivoice_refs/CordobesMan_10s.wav",
        sample_ref_text=(
            "Che, te quiero presentar los ultimos cambios en la Feli que nace "
            "de adentro tuyo. Vos sos el que maneja tu propio destino, no hay otro."
        ),
        sample_label="Cordobes Man",
        sample_duration_seconds=10.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="fable",
        fallback_name="fable (male, high pitch, australian accent)",
        fallback_instruct="male, high pitch, australian accent",
        sample_file="omnivoice_refs/britishMan_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Man",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="onyx",
        fallback_name="onyx (male, very low pitch, american accent)",
        fallback_instruct="male, very low pitch, american accent",
        sample_file="omnivoice_refs/CordobesMan_10s.wav",
        sample_ref_text=(
            "Che, te quiero presentar los ultimos cambios en la Feli que nace "
            "de adentro tuyo. Vos sos el que maneja tu propio destino, no hay otro."
        ),
        sample_label="Cordobes Man",
        sample_duration_seconds=10.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="nova",
        fallback_name="nova (female, moderate pitch, american accent)",
        fallback_instruct="female, moderate pitch, american accent",
        sample_file="omnivoice_refs/britishWoman_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Woman",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="shimmer",
        fallback_name="shimmer (female, high pitch, british accent)",
        fallback_instruct="female, high pitch, british accent",
        sample_file="omnivoice_refs/testargentina_8s.wav",
        sample_ref_text=(
            "Definitivamente el departamento tiene que tener terraza o balcon. "
            "Hace mucho tiempo que tenia ganas de usar esta aplicacion para aprender a cocinar."
        ),
        sample_label="Argentina Voice",
        sample_duration_seconds=8.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="british_man",
        fallback_name="british_man (male, moderate pitch, british accent)",
        fallback_instruct="male, moderate pitch, british accent",
        sample_file="omnivoice_refs/britishMan_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Man",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="british_woman",
        fallback_name="british_woman (female, moderate pitch, british accent)",
        fallback_instruct="female, moderate pitch, british accent",
        sample_file="omnivoice_refs/britishWoman_10s.wav",
        sample_ref_text=(
            "I'd like to introduce you to the latest changes on happiness from "
            "within yourself. You are in charge of your own destiny."
        ),
        sample_label="British Woman",
        sample_duration_seconds=10.0,
        default_language="en",
    ),
    VoiceOptionDefinition(
        id="cordobes_man",
        fallback_name="cordobes_man (male, moderate pitch, argentinian accent)",
        fallback_instruct="male, moderate pitch, argentinian accent",
        sample_file="omnivoice_refs/CordobesMan_10s.wav",
        sample_ref_text=(
            "Che, te quiero presentar los ultimos cambios en la Feli que nace "
            "de adentro tuyo. Vos sos el que maneja tu propio destino, no hay otro."
        ),
        sample_label="Cordobes Man",
        sample_duration_seconds=10.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="argentina_voice",
        fallback_name="argentina_voice (female, warm pitch, argentinian accent)",
        fallback_instruct="female, warm pitch, argentinian accent",
        sample_file="omnivoice_refs/testargentina_8s.wav",
        sample_ref_text=(
            "Definitivamente el departamento tiene que tener terraza o balcon. "
            "Hace mucho tiempo que tenia ganas de usar esta aplicacion para aprender a cocinar."
        ),
        sample_label="Argentina Voice",
        sample_duration_seconds=8.0,
        default_language="es",
    ),
    VoiceOptionDefinition(
        id="auto",
        fallback_name="auto (voice design)",
        fallback_instruct=None,
    ),
]
VOICE_LOOKUP = {item.id: item for item in VOICE_OPTIONS}
SUPPORTED_MODEL_OPTIONS = [
    {"id": API_MODEL_ID, "name": "OmniVoice"},
    {"id": "tts-1", "name": "OmniVoice (OpenAI alias)"},
    {"id": "tts-1-hd", "name": "OmniVoice HD (OpenAI alias)"},
    {"id": "gpt-4o-mini-tts", "name": "OmniVoice mini TTS (OpenAI alias)"},
]
SUPPORTED_MODEL_ALIASES = {
    API_MODEL_ID,
    BACKEND_MODEL_ID,
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
}
SUPPORTED_RESPONSE_FORMATS = {
    "mp3": ("audio/mpeg", "mp3"),
    "wav": ("audio/wav", "wav"),
    "flac": ("audio/flac", "flac"),
    "ogg": ("audio/ogg", "ogg"),
    "opus": ("audio/opus", "opus"),
}


def _build_frontend_page() -> str:
    voice_cards = []
    local_voice_count = 0

    for option in VOICE_OPTIONS:
        has_local_sample = option.has_local_sample()
        if has_local_sample:
            local_voice_count += 1

        source_label = "Local reference clip" if has_local_sample else "Voice design prompt"
        duration_label = (
            f"{int(round(option.sample_duration_seconds))}s clip"
            if option.sample_duration_seconds is not None
            else "No reference clip"
        )
        language_label = option.default_language or "auto"
        detail_text = option.sample_ref_text or option.fallback_instruct or "Automatic voice selection"

        voice_cards.append(
            f"""
            <article class="voice-card">
              <div class="voice-card__top">
                <span class="badge">{escape(option.id)}</span>
                <span class="status">{escape(source_label)}</span>
              </div>
              <h3>{escape(option.display_name())}</h3>
              <p class="meta">Language: {escape(language_label)} · {escape(duration_label)}</p>
              <p class="body">{escape(detail_text)}</p>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OmniVoice | OpenAI-compatible TTS</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #07111f;
        --panel: rgba(10, 20, 36, 0.92);
        --panel-strong: rgba(13, 28, 49, 0.98);
        --text: #eaf2ff;
        --muted: #a8b8d8;
        --accent: #6ea8ff;
        --accent-strong: #8d6bff;
        --border: rgba(122, 154, 219, 0.18);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(110, 168, 255, 0.25), transparent 28%),
          radial-gradient(circle at top right, rgba(141, 107, 255, 0.2), transparent 24%),
          linear-gradient(180deg, #05101d 0%, #07111f 52%, #09131f 100%);
        color: var(--text);
      }}

      a {{
        color: inherit;
      }}

      .shell {{
        width: min(1180px, calc(100% - 32px));
        margin: 0 auto;
        padding: 32px 0 48px;
      }}

      .hero,
      .panel {{
        background: linear-gradient(180deg, var(--panel), var(--panel-strong));
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: 0 28px 64px rgba(0, 0, 0, 0.28);
      }}

      .hero {{
        padding: 36px;
        display: grid;
        gap: 18px;
      }}

      .eyebrow {{
        margin: 0;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.78rem;
        color: var(--accent);
      }}

      h1 {{
        margin: 0;
        font-size: clamp(2.3rem, 5vw, 4.2rem);
        line-height: 1.02;
      }}

      .lede {{
        margin: 0;
        max-width: 68ch;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
      }}

      .actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }}

      .button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 11px 16px;
        border-radius: 999px;
        border: 1px solid rgba(146, 175, 235, 0.28);
        background: rgba(255, 255, 255, 0.04);
        text-decoration: none;
        font-weight: 600;
      }}

      .button.primary {{
        background: linear-gradient(135deg, var(--accent), var(--accent-strong));
        color: #08101d;
        border-color: transparent;
      }}

      .stats {{
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
      }}

      .stat {{
        padding: 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(146, 175, 235, 0.14);
      }}

      .stat span {{
        display: block;
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 6px;
      }}

      .stat strong {{
        font-size: 1.05rem;
      }}

      .panel {{
        margin-top: 22px;
        padding: 24px;
      }}

      .panel h2 {{
        margin: 0 0 8px;
        font-size: 1.4rem;
      }}

      .panel p.subhead {{
        margin: 0 0 18px;
        color: var(--muted);
        line-height: 1.6;
      }}

      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 14px;
      }}

      .voice-card {{
        padding: 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(146, 175, 235, 0.14);
      }}

      .voice-card__top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 12px;
      }}

      .badge,
      .status {{
        display: inline-flex;
        align-items: center;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        line-height: 1;
      }}

      .badge {{
        background: rgba(110, 168, 255, 0.16);
        color: #d6e5ff;
      }}

      .status {{
        background: rgba(141, 107, 255, 0.16);
        color: #e6dcff;
      }}

      .voice-card h3 {{
        margin: 0 0 8px;
        font-size: 1.08rem;
      }}

      .meta {{
        margin: 0 0 10px;
        color: var(--muted);
        font-size: 0.92rem;
      }}

      .body {{
        margin: 0;
        color: #d6e1f6;
        line-height: 1.6;
      }}

      .credits {{
        display: grid;
        gap: 10px;
      }}

      .credits strong {{
        color: #f5f8ff;
      }}

      .footer {{
        margin-top: 18px;
        color: var(--muted);
        font-size: 0.92rem;
      }}

      @media (max-width: 640px) {{
        .shell {{
          width: min(100% - 20px, 1180px);
          padding-top: 10px;
        }}

        .hero,
        .panel {{
          border-radius: 20px;
        }}

        .hero {{
          padding: 24px;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">OpenAI-compatible TTS</p>
        <h1>OmniVoice</h1>
        <p class="lede">
          FastAPI text-to-speech on <strong>0.0.0.0:6655</strong> with request sanitization,
          sentence-aware chunking, and local voice presets that work cleanly with Open WebUI.
        </p>
        <div class="actions">
          <a class="button primary" href="/v1/audio/speech">Speech endpoint</a>
          <a class="button" href="/v1/audio/voices">Voice catalogue</a>
          <a class="button" href="/health">Health</a>
        </div>
        <div class="stats">
          <div class="stat"><span>Local voice presets</span><strong>{len(VOICE_OPTIONS)}</strong></div>
          <div class="stat"><span>Local reference clips</span><strong>{local_voice_count}</strong></div>
          <div class="stat"><span>API models</span><strong>{len(SUPPORTED_MODEL_OPTIONS)}</strong></div>
          <div class="stat"><span>Sentence chunking</span><strong>Enabled</strong></div>
        </div>
      </section>

      <section class="panel">
        <h2>Voice presets</h2>
        <p class="subhead">
          Each preset maps to either a compact local reference clip or a direct voice-design
          prompt, so Open WebUI can pick a stable voice without guessing.
        </p>
        <div class="grid">
          {"".join(voice_cards)}
        </div>
      </section>

      <section class="panel credits">
        <h2>Credits</h2>
        <p>
          Thanks to the original OmniVoice creators and contributors for the model, research,
          and open-source implementation this server is built on.
        </p>
        <p class="footer">
          Browser page: <a href="/ui">/ui</a> · JSON endpoints: <a href="/v1/audio/models">/v1/audio/models</a>,
          <a href="/v1/audio/voices">/v1/audio/voices</a>
        </p>
      </section>
    </main>
  </body>
</html>"""


def _render_frontend_page() -> HTMLResponse:
    return HTMLResponse(_build_frontend_page())


def _configure_logging() -> None:
    if getattr(_configure_logging, "_configured", False):
        return
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=False,
    )
    _configure_logging._configured = True


DEFAULT_DEVICE = get_best_device()


class TextSanitizationOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    normalize: bool = True
    unit_normalization: bool = False
    strip_html: bool = True
    strip_model_control_tokens: bool = True
    quote_normalization: bool = True
    url_normalization: bool = True
    email_normalization: bool = True
    optional_pluralization_normalization: bool = True
    phone_normalization: bool = True
    number_normalization: bool = True
    money_normalization: bool = True
    time_normalization: bool = True
    replace_remaining_symbols: bool = True


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: Optional[str] = Field(default=None, max_length=MAX_RAW_INPUT_CHARS)
    text: Optional[str] = Field(default=None, max_length=MAX_RAW_INPUT_CHARS)
    model: Optional[str] = None
    voice: Optional[str] = Field(default=None, max_length=512)
    response_format: Literal["mp3", "wav", "flac", "ogg", "opus"] = "mp3"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
    language: Optional[str] = Field(default=None, max_length=64)
    ref_text: Optional[str] = Field(default=None, max_length=MAX_RAW_INPUT_CHARS)
    instruct: Optional[str] = Field(default=None, max_length=1024)
    duration: Optional[float] = Field(default=None, gt=0.0, le=3600.0)
    num_step: Optional[int] = Field(default=None, ge=1, le=128)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    t_shift: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    denoise: Optional[bool] = None
    postprocess_output: Optional[bool] = None
    layer_penalty_factor: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    position_temperature: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    class_temperature: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    normalization_options: TextSanitizationOptions = Field(
        default_factory=TextSanitizationOptions
    )
    sentence_chunking: bool = True
    sentence_chunking_min_chars: int = Field(
        default=DEFAULT_SENTENCE_CHUNKING_MIN_CHARS,
        ge=64,
        le=MAX_SANITIZED_INPUT_CHARS,
    )
    audio_chunk_duration: Optional[float] = Field(default=None, gt=0.5, le=60.0)
    audio_chunk_threshold: Optional[float] = Field(default=None, ge=0.0, le=600.0)

    @model_validator(mode="after")
    def _validate_text_fields(self) -> SpeechRequest:
        values = [value for value in (self.input, self.text) if value is not None]
        if not values:
            raise ValueError("One of 'input' or 'text' must be provided")
        if len(values) == 2 and values[0].strip() != values[1].strip():
            raise ValueError(
                "When both 'input' and 'text' are provided they must contain the same text"
            )
        return self

    def raw_text(self) -> str:
        return self.input if self.input is not None else self.text or ""


@dataclass(slots=True)
class PreparedSpeechRequest:
    text: str
    instruct: Optional[str]
    ref_text: Optional[str]
    language: Optional[str]
    response_format: str
    voice_id: str
    voice_display_name: str
    voice_ref_audio_path: Optional[Path]
    voice_ref_text: Optional[str]
    chunk_plan: list[str]
    force_sentence_chunking: bool
    generation_config: OmniVoiceGenerationConfig


class OmniVoiceService:
    def __init__(self, model_id: str, device: str, idle_timeout: float = 300.0) -> None:
        self.model_id = model_id
        self.device = device
        self.model: OmniVoice | None = None
        self.load_error: str | None = None
        self.load_lock: asyncio.Lock | None = None
        self._idle_timeout = idle_timeout
        self._idle_task: asyncio.Task | None = None
        self._last_used: float = 0.0
        self._voice_prompt_cache: dict[str, object] = {}

    def set_lock(self, lock: asyncio.Lock) -> None:
        self.load_lock = lock

    def _touch(self) -> None:
        self._last_used = time.monotonic()
        self._schedule_idle_offload()

    def _schedule_idle_offload(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
        if self._idle_timeout > 0 and self.model is not None:
            self._idle_task = asyncio.get_event_loop().create_task(
                self._idle_offload_loop()
            )

    async def _idle_offload_loop(self) -> None:
        try:
            while True:
                elapsed = time.monotonic() - self._last_used
                remaining = self._idle_timeout - elapsed
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            return

        if self.model is not None:
            LOG.info(
                "Idle timeout (%.0fs) reached, offloading model from %s",
                self._idle_timeout,
                self.device,
            )
            try:
                if self.device.startswith("cuda"):
                    await asyncio.to_thread(self._offload_model_sync)
                else:
                    self.model = None
                LOG.info("Model offloaded successfully")
            except Exception:
                LOG.exception("Failed to offload model")

    def _offload_model_sync(self) -> None:
        if self.model is None:
            return
        import gc

        self.model.cpu()
        del self.model
        self.model = None
        self._voice_prompt_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model_sync(self) -> OmniVoice:
        LOG.info("Loading OmniVoice model %s on %s", self.model_id, self.device)
        if self.device.startswith("cuda"):
            from transformers import modeling_utils

            original_allocator_warmup = modeling_utils.caching_allocator_warmup

            # The default Transformers warmup can briefly allocate several extra
            # GiB on cold load. Skipping it keeps OmniVoice lazy-loading viable
            # on the RTX 3060 when other workloads already share that card.
            modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
            try:
                model = OmniVoice.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                )
            finally:
                modeling_utils.caching_allocator_warmup = original_allocator_warmup
        else:
            model = OmniVoice.from_pretrained(
                self.model_id,
                device_map=self.device,
            )
        model.eval()
        LOG.info(
            "Model loaded on %s with sampling rate %s",
            model.device,
            getattr(model, "sampling_rate", "unknown"),
        )
        return model

    async def get_model(self) -> OmniVoice:
        if self.model is not None:
            self._touch()
            return self.model
        if self.load_lock is None:
            raise RuntimeError("Load lock is not initialized")

        async with self.load_lock:
            if self.model is not None:
                self._touch()
                return self.model

            try:
                self.load_error = None
                self.model = await asyncio.to_thread(self._load_model_sync)
                self._touch()
            except Exception as exc:
                self.load_error = str(exc)
                LOG.exception("Failed to load OmniVoice")
                raise

            return self.model

    def health(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "api_model_id": API_MODEL_ID,
            "device": self.device,
            "model_loaded": self.model is not None,
            "load_error": self.load_error,
            "default_voice": DEFAULT_VOICE,
            "idle_timeout": self._idle_timeout,
        }

    def get_or_create_voice_clone_prompt(
        self,
        model: OmniVoice,
        *,
        cache_key: str,
        ref_audio_path: Path,
        ref_text: Optional[str] = None,
    ):
        cached = self._voice_prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
        )
        self._voice_prompt_cache[cache_key] = prompt
        return prompt


def _protect_bracket_tags(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        key = f"OMNIVOICETAG{uuid4().hex.upper()}X{len(protected)}TAG"
        protected[key] = match.group(0)
        return key

    return PROTECTED_TAG_PATTERN.sub(_replace, text), protected


def _restore_bracket_tags(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def _normalize_email(match: re.Match[str]) -> str:
    user, domain = match.group(0).split("@", 1)
    user = (
        user.replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
    )
    domain = domain.replace(".", " dot ").replace("-", " dash ")
    return f"{user} at {domain}"


def _normalize_url(match: re.Match[str]) -> str:
    original = match.group(0)
    url = original.rstrip(TRAILING_URL_PUNCTUATION)
    trailing = original[len(url) :]
    spoken = url
    spoken = re.sub(
        r"^https?://",
        lambda item: "https " if item.group(0).lower().startswith("https") else "http ",
        spoken,
        flags=re.IGNORECASE,
    )
    spoken = re.sub(r"^www\.", "www ", spoken, flags=re.IGNORECASE)
    spoken = spoken.replace(".", " dot ")
    spoken = spoken.replace("/", " slash ")
    spoken = spoken.replace("?", " question mark ")
    spoken = spoken.replace("=", " equals ")
    spoken = spoken.replace("&", " and ")
    spoken = spoken.replace("-", " dash ")
    spoken = spoken.replace("_", " underscore ")
    spoken = spoken.replace(":", " colon ")
    spoken = WHITESPACE_PATTERN.sub(" ", spoken).strip()
    return f"{spoken}{trailing}"


def _normalize_phone(match: re.Match[str]) -> str:
    value = match.group(0)
    digits = [char for char in value if char.isdigit()]
    if not digits:
        return value
    prefix = "plus " if value.strip().startswith("+") else ""
    return prefix + " ".join(digits)


def _should_apply_english_expansions(language: Optional[str]) -> bool:
    if language is None:
        return True
    language_key = language.lower().replace("_", "-").split("-", 1)[0]
    return language_key in {"en", "eng"}


def _conditional_int(number: float, threshold: float = 0.00001):
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def _translate_multiplier(multiplier: str) -> str:
    mapping = {
        "k": "thousand",
        "m": "million",
        "b": "billion",
        "t": "trillion",
    }
    key = multiplier.strip().lower()
    return mapping.get(key, multiplier.strip())


def _split_four_digit_year(number: float) -> str:
    number_str = str(_conditional_int(number))
    return (
        f"{INFLECT_ENGINE.number_to_words(number_str[:2])} "
        f"{INFLECT_ENGINE.number_to_words(number_str[2:])}"
    )


def _normalize_unit(match: re.Match[str]) -> str:
    unit_string = match.group(6).strip()
    unit_name = VALID_UNITS.get(unit_string.lower())
    if unit_name is None:
        return match.group(0)

    parts = unit_name.split(" ")
    number = match.group(1).strip().replace(",", "")
    if parts[0].endswith("bit") and unit_string[-1:] == "B":
        parts[0] = parts[0][:-3] + "byte"
    parts[0] = INFLECT_ENGINE.no(parts[0], number)
    return " ".join(parts)


def _normalize_grouped_phone(match: re.Match[str]) -> str:
    country_code, _, area_code, telephone_prefix, line_number = match.groups()
    parts: list[str] = []
    if country_code:
        parts.append(
            INFLECT_ENGINE.number_to_words(
                country_code.replace("+", ""),
                group=1,
                comma="",
            )
        )
    parts.append(
        INFLECT_ENGINE.number_to_words(
            area_code.replace("(", "").replace(")", ""),
            group=1,
            comma="",
        )
    )
    parts.append(
        INFLECT_ENGINE.number_to_words(telephone_prefix, group=1, comma="")
    )
    parts.append(INFLECT_ENGINE.number_to_words(line_number, group=1, comma=""))
    return ", ".join(filter(None, parts))


def _normalize_time(match: re.Match[str]) -> str:
    time_value = match.group(1)
    meridiem = (match.group(2) or "").strip().lower()
    time_parts = [part.strip() for part in time_value.split(":")]

    result = [INFLECT_ENGINE.number_to_words(int(time_parts[0]))]
    minutes = int(time_parts[1])
    if minutes == 0:
        result.append("o'clock")
    elif minutes < 10:
        result.append(f"oh {INFLECT_ENGINE.number_to_words(minutes)}")
    else:
        result.append(INFLECT_ENGINE.number_to_words(minutes))

    if len(time_parts) > 2:
        seconds = int(time_parts[2])
        result.append(
            f"and {INFLECT_ENGINE.number_to_words(seconds)} "
            f"{INFLECT_ENGINE.plural('second', seconds)}"
        )

    if meridiem:
        result.append(meridiem)
    return " ".join(result)


def _normalize_money(match: re.Match[str]) -> str:
    bill, coin = MONEY_UNITS[match.group(2)]
    amount_text = match.group(3)
    try:
        amount = float(amount_text)
    except ValueError:
        return match.group(0)

    if match.group(1) == "-":
        amount *= -1

    multiplier = _translate_multiplier(match.group(4))
    multiplier_suffix = f" {multiplier}" if multiplier else ""
    abs_amount = abs(amount)

    if abs_amount % 1 == 0 or multiplier:
        spoken_amount = INFLECT_ENGINE.number_to_words(_conditional_int(abs_amount))
        spoken_bill = INFLECT_ENGINE.plural(
            bill,
            count=max(int(abs_amount), 1),
        )
        prefix = "minus " if amount < 0 else ""
        return f"{prefix}{spoken_amount}{multiplier_suffix} {spoken_bill}"

    whole_amount = int(math.floor(abs_amount))
    cents = int(amount_text.split(".")[-1].ljust(2, "0"))
    spoken_bill = INFLECT_ENGINE.plural(bill, count=max(whole_amount, 1))
    prefix = "minus " if amount < 0 else ""
    return (
        f"{prefix}{INFLECT_ENGINE.number_to_words(whole_amount)} {spoken_bill} and "
        f"{INFLECT_ENGINE.number_to_words(cents)} "
        f"{INFLECT_ENGINE.plural(coin, count=cents)}"
    )


def _normalize_number(match: re.Match[str]) -> str:
    try:
        number = float(match.group(2))
    except ValueError:
        return match.group(0)

    if match.group(1) == "-":
        number *= -1

    multiplier = _translate_multiplier(match.group(3))
    if not multiplier:
        abs_number = abs(number)
        if (
            abs_number % 1 == 0
            and len(str(int(abs_number))) == 4
            and abs_number > 1500
            and int(abs_number) % 1000 > 9
        ):
            prefix = "minus " if number < 0 else ""
            return f"{prefix}{_split_four_digit_year(abs_number)}"

    spoken_number = INFLECT_ENGINE.number_to_words(_conditional_int(abs(number)))
    prefix = "minus " if number < 0 else ""
    multiplier_suffix = f" {multiplier}" if multiplier else ""
    return f"{prefix}{spoken_number}{multiplier_suffix}"


def _normalize_titles_and_abbreviations(text: str) -> str:
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
    return text


def _normalize_english_like_text(
    text: str, options: TextSanitizationOptions
) -> str:
    normalized = text
    if options.unit_normalization:
        normalized = UNIT_PATTERN.sub(_normalize_unit, normalized)
    if options.optional_pluralization_normalization:
        normalized = OPTIONAL_PLURALIZATION_PATTERN.sub("s", normalized)
        normalized = re.sub(r"(?<=\d)s\b", " s", normalized, flags=re.IGNORECASE)
    if options.phone_normalization:
        normalized = PHONE_GROUP_PATTERN.sub(_normalize_grouped_phone, normalized)
    if options.time_normalization:
        normalized = TIME_PATTERN.sub(_normalize_time, normalized)

    normalized = _normalize_titles_and_abbreviations(normalized)
    normalized = THOUSANDS_SEPARATOR_PATTERN.sub("", normalized)

    if options.money_normalization:
        normalized = MONEY_PATTERN.sub(_normalize_money, normalized)
    if options.number_normalization:
        normalized = NUMBER_PATTERN.sub(_normalize_number, normalized)

    return normalized


def _basic_cleanup(
    text: str,
    *,
    strip_html: bool,
    strip_model_control_tokens: bool,
    normalize_quotes: bool,
) -> str:
    if normalize_quotes:
        text = text.translate(SMART_PUNCTUATION_TRANSLATION)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(CJK_PUNCTUATION_TRANSLATION)
    if strip_model_control_tokens:
        text = MODEL_CONTROL_TOKEN_PATTERN.sub(" ", text)
    if strip_html:
        text = HTML_TAG_PATTERN.sub(" ", text)
    text = CONTROL_CHAR_PATTERN.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def sanitize_speech_text(
    text: str,
    *,
    language: Optional[str],
    options: TextSanitizationOptions,
) -> str:
    if not text:
        return ""
    if not options.normalize:
        return add_punctuation(
            _basic_cleanup(
                text,
                strip_html=options.strip_html,
                strip_model_control_tokens=options.strip_model_control_tokens,
                normalize_quotes=options.quote_normalization,
            )
        )

    protected_text, protected = _protect_bracket_tags(text)
    sanitized = _basic_cleanup(
        protected_text,
        strip_html=options.strip_html,
        strip_model_control_tokens=options.strip_model_control_tokens,
        normalize_quotes=options.quote_normalization,
    )

    if options.email_normalization:
        sanitized = EMAIL_PATTERN.sub(_normalize_email, sanitized)
    if options.url_normalization:
        sanitized = URL_PATTERN.sub(_normalize_url, sanitized)
    if _should_apply_english_expansions(language):
        sanitized = _normalize_english_like_text(sanitized, options)
    elif options.optional_pluralization_normalization:
        sanitized = OPTIONAL_PLURALIZATION_PATTERN.sub("s", sanitized)
    if options.phone_normalization:
        sanitized = GENERIC_PHONE_PATTERN.sub(_normalize_phone, sanitized)
    if options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            sanitized = sanitized.replace(symbol, replacement)

    sanitized = RANGE_DASH_PATTERN.sub(" to ", sanitized)
    sanitized = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]",
        lambda match: match.group(0).replace(".", "-"),
        sanitized,
    )
    sanitized = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", sanitized)
    sanitized = WHITESPACE_PATTERN.sub(" ", sanitized).strip()
    sanitized = _restore_bracket_tags(sanitized, protected)
    if language is not None:
        LOG.debug("Sanitized text for language=%s -> %s", language, sanitized[:120])
    return add_punctuation(sanitized)


def sanitize_prompt_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = _basic_cleanup(
        text,
        strip_html=True,
        strip_model_control_tokens=True,
        normalize_quotes=True,
    )
    return cleaned or None


def _plan_sentence_chunks(text: str, min_chars: int) -> list[str]:
    if len(text) <= min_chars:
        return [text]
    planned = chunk_text_punctuation(
        text=text,
        chunk_len=min_chars,
        min_chunk_len=max(24, min_chars // 4),
    )
    return planned or [text]


def _supported_models() -> list[dict[str, str]]:
    return [dict(item) for item in SUPPORTED_MODEL_OPTIONS]


def _supported_voices() -> list[dict[str, str]]:
    return [{"id": item.id, "name": item.display_name()} for item in VOICE_OPTIONS]


def _truncate_preview(text: Optional[str], limit: int = DEBUG_PREVIEW_CHARS) -> str:
    if not text:
        return ""
    preview = WHITESPACE_PATTERN.sub(" ", text).strip()
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit].rstrip()}..."


def _normalization_summary(options: TextSanitizationOptions) -> str:
    enabled = [key for key, value in options.model_dump().items() if value]
    return ",".join(enabled) if enabled else "none"


def _guess_request_source(request: Request) -> str:
    user_agent = (request.headers.get("user-agent") or "").strip()
    user_agent_lower = user_agent.lower()
    if "python-requests" in user_agent_lower:
        return "python-requests/open-webui-like"
    if "mozilla" in user_agent_lower:
        return "browser"
    if user_agent:
        return user_agent[:80]
    return "unknown"


def _resolve_model(requested: Optional[str]) -> str:
    if not requested:
        return API_MODEL_ID
    if requested in SUPPORTED_MODEL_ALIASES or requested == service.model_id:
        return API_MODEL_ID
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported model '{requested}'. Supported aliases: "
            f"{', '.join(sorted(SUPPORTED_MODEL_ALIASES))}"
        ),
    )


def _resolve_voice(requested: Optional[str]) -> ResolvedVoice:
    voice = (requested or DEFAULT_VOICE).strip()
    if not voice:
        raise HTTPException(status_code=400, detail="voice must not be empty")

    voice_key = voice.lower()
    preset = VOICE_LOOKUP.get(voice_key)
    if preset is not None:
        use_local_sample = preset.has_local_sample()
        return ResolvedVoice(
            voice_id=preset.id,
            display_name=preset.display_name(),
            instruct=None if use_local_sample else preset.fallback_instruct,
            ref_audio_path=preset.sample_path if use_local_sample else None,
            ref_text=preset.sample_ref_text if use_local_sample else None,
            default_language=preset.default_language,
        )

    return ResolvedVoice(
        voice_id=voice,
        display_name=voice,
        instruct=voice,
        ref_audio_path=None,
        ref_text=None,
        default_language=None,
    )


def _prepare_request(payload: SpeechRequest) -> PreparedSpeechRequest:
    _resolve_model(payload.model)
    response_format = payload.response_format
    resolved_voice = _resolve_voice(payload.voice)
    effective_language = payload.language or resolved_voice.default_language

    text = sanitize_speech_text(
        payload.raw_text(),
        language=effective_language,
        options=payload.normalization_options,
    )
    if not text:
        raise HTTPException(
            status_code=400, detail="Input text is empty after sanitization"
        )
    if len(text) > MAX_SANITIZED_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input text is too long after sanitization ({len(text)} chars). "
                f"Maximum allowed is {MAX_SANITIZED_INPUT_CHARS}."
            ),
        )

    instruct = sanitize_prompt_text(payload.instruct)
    if instruct is None and resolved_voice.ref_audio_path is None:
        instruct = sanitize_prompt_text(resolved_voice.instruct)

    resolved_ref_text = (
        sanitize_prompt_text(payload.ref_text)
        if payload.ref_text is not None
        else sanitize_prompt_text(resolved_voice.ref_text)
    )
    ref_text = resolved_ref_text if resolved_voice.ref_audio_path is None else None
    chunk_plan = _plan_sentence_chunks(text, payload.sentence_chunking_min_chars)
    force_sentence_chunking = payload.sentence_chunking and len(chunk_plan) > 1

    generation_config_kwargs: dict[str, object] = {}
    for field_name in (
        "num_step",
        "guidance_scale",
        "t_shift",
        "denoise",
        "postprocess_output",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
    ):
        value = getattr(payload, field_name)
        if value is not None:
            generation_config_kwargs[field_name] = value

    generation_config_kwargs["audio_chunk_duration"] = (
        payload.audio_chunk_duration
        if payload.audio_chunk_duration is not None
        else DEFAULT_AUDIO_CHUNK_DURATION
    )
    generation_config_kwargs["audio_chunk_threshold"] = (
        0.0
        if force_sentence_chunking
        else (
            payload.audio_chunk_threshold
            if payload.audio_chunk_threshold is not None
            else DEFAULT_AUDIO_CHUNK_THRESHOLD
        )
    )

    return PreparedSpeechRequest(
        text=text,
        instruct=instruct,
        ref_text=ref_text,
        language=effective_language,
        response_format=response_format,
        voice_id=resolved_voice.voice_id,
        voice_display_name=resolved_voice.display_name,
        voice_ref_audio_path=resolved_voice.ref_audio_path,
        voice_ref_text=resolved_ref_text if resolved_voice.ref_audio_path is not None else None,
        chunk_plan=chunk_plan,
        force_sentence_chunking=force_sentence_chunking,
        generation_config=OmniVoiceGenerationConfig.from_dict(generation_config_kwargs),
    )


def _waveform_to_bytes(
    waveform: torch.Tensor,
    sample_rate: int,
    response_format: str,
) -> tuple[bytes, str]:
    media_type, suffix = SUPPORTED_RESPONSE_FORMATS[response_format]

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.detach().cpu().float().clamp(-1.0, 1.0).contiguous()

    with tempfile.TemporaryDirectory(prefix="omnivoice-tts-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_path = tmp_path / "output.wav"
        torchaudio.save(str(wav_path), waveform, sample_rate, format="wav")

        if response_format == "wav":
            return wav_path.read_bytes(), media_type

        out_path = tmp_path / f"output.{suffix}"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
        ]

        if response_format == "mp3":
            ffmpeg_cmd += ["-codec:a", "libmp3lame", "-b:a", "192k"]
        elif response_format == "flac":
            ffmpeg_cmd += ["-codec:a", "flac"]
        elif response_format in {"ogg", "opus"}:
            ffmpeg_cmd += ["-codec:a", "libopus"]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format '{response_format}'",
            )

        ffmpeg_cmd.append(str(out_path))
        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg is required to encode the requested audio format",
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise HTTPException(
                status_code=500,
                detail=f"Audio encoding failed: {stderr or exc}",
            ) from exc
        return out_path.read_bytes(), media_type


async def _synthesize_prepared(
    prepared: PreparedSpeechRequest,
    payload: SpeechRequest,
    *,
    request_id: Optional[str] = None,
) -> tuple[PreparedSpeechRequest, bytes, str]:
    model = await service.get_model()

    generation_kwargs: dict[str, object] = {
        "text": prepared.text,
        "language": prepared.language,
        "speed": payload.speed,
        "generation_config": prepared.generation_config,
    }
    if prepared.ref_text is not None:
        generation_kwargs["ref_text"] = prepared.ref_text
    if prepared.instruct is not None:
        generation_kwargs["instruct"] = prepared.instruct
    if payload.duration is not None:
        generation_kwargs["duration"] = payload.duration

    LOG.info(
        "Synthesizing speech (voice=%s, response_format=%s, text_chars=%d, text_chunks=%d, forced_sentence_chunking=%s, voice_source=%s)",
        prepared.voice_id,
        prepared.response_format,
        len(prepared.text),
        len(prepared.chunk_plan),
        prepared.force_sentence_chunking,
        "local-reference" if prepared.voice_ref_audio_path is not None else "style-prompt",
    )
    LOG.debug("Sanitized input preview: %s", prepared.text[:200])

    def _run_generation() -> tuple[bytes, str]:
        generation_args = dict(generation_kwargs)
        if prepared.voice_ref_audio_path is not None:
            generation_args["voice_clone_prompt"] = service.get_or_create_voice_clone_prompt(
                model,
                cache_key=(
                    f"{prepared.voice_ref_audio_path.resolve()}::{prepared.voice_ref_text!r}"
                ),
                ref_audio_path=prepared.voice_ref_audio_path,
                ref_text=prepared.voice_ref_text,
            )
        with torch.inference_mode():
            audios = model.generate(**generation_args)
        if not audios:
            raise RuntimeError("OmniVoice returned no audio")
        waveform = audios[0]
        return _waveform_to_bytes(
            waveform,
            int(model.sampling_rate),
            prepared.response_format,
        )

    try:
        audio_bytes, media_type = await asyncio.to_thread(_run_generation)
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception(
            "Speech synthesis failed (request_id=%s, voice=%s, voice_source=%s, language=%s)",
            request_id or "n/a",
            prepared.voice_id,
            "local-reference" if prepared.voice_ref_audio_path is not None else "style-prompt",
            prepared.language,
        )
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from exc

    return prepared, audio_bytes, media_type


@asynccontextmanager
async def lifespan(_: FastAPI):
    _configure_logging()
    service.set_lock(asyncio.Lock())
    LOG.info(
        "Starting OmniVoice TTS server (api_model=%s, backend_model=%s, device=%s, idle_timeout=%.0fs)",
        API_MODEL_ID,
        service.model_id,
        service.device,
        service._idle_timeout,
    )
    yield
    if service._idle_task is not None:
        service._idle_task.cancel()
    if service.model is not None:
        LOG.info("Shutting down, offloading model...")
        try:
            if service.device.startswith("cuda"):
                await asyncio.to_thread(service._offload_model_sync)
            else:
                service.model = None
        except Exception:
            LOG.exception("Error offloading model during shutdown")


service = OmniVoiceService(BACKEND_MODEL_ID, DEFAULT_DEVICE)
app = FastAPI(title="OmniVoice OpenAI-Compatible TTS", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    request_id = uuid4().hex[:12]
    request.state.request_id = request_id
    request.state.request_source = _guess_request_source(request)
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        LOG.exception(
            "Request failed (request_id=%s, source=%s, client=%s, %s %s, %.1fms)",
            request_id,
            request.state.request_source,
            getattr(request.client, "host", "unknown"),
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-OmniVoice-Request-Id"] = request_id
    if request.url.path not in {"/", "/health"}:
        LOG.info(
            "%s %s -> %s in %.1fms (request_id=%s, source=%s, client=%s)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
            request.state.request_source,
            getattr(request.client, "host", "unknown"),
        )
    return response


@app.get("/")
@app.get("/health")
@app.get("/v1/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "supported_response_formats": sorted(SUPPORTED_RESPONSE_FORMATS),
        **service.health(),
    }


@app.get("/ui", response_class=HTMLResponse)
@app.get("/credits", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    return _render_frontend_page()


@app.get("/v1/models")
@app.get("/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": model["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
            for model in _supported_models()
        ],
    }


@app.get("/audio/models")
@app.get("/v1/audio/models")
async def list_audio_models() -> dict[str, list[dict[str, str]]]:
    return {"models": _supported_models()}


@app.get("/audio/voices")
@app.get("/v1/audio/voices")
async def list_audio_voices() -> dict[str, list[dict[str, str]]]:
    return {"voices": _supported_voices()}


@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def audio_speech(payload: SpeechRequest, request: Request) -> Response:
    _ = request.headers.get("authorization")
    request_id = getattr(request.state, "request_id", uuid4().hex[:12])
    request_source = getattr(request.state, "request_source", _guess_request_source(request))
    prepared = _prepare_request(payload)
    voice_mode = (
        "local-reference" if prepared.voice_ref_audio_path is not None else "style-prompt"
    )
    ref_text_source = (
        "request"
        if payload.ref_text is not None
        else "preset"
        if prepared.voice_ref_text is not None
        else "none"
    )
    LOG.info(
        "Prepared speech request (request_id=%s, source=%s, voice=%s, voice_mode=%s, language=%s, raw_chars=%d, sanitized_chars=%d, normalization=%s, ref_audio=%s, ref_text_source=%s, ref_text_chars=%d, preview_raw=%r, preview_sanitized=%r)",
        request_id,
        request_source,
        prepared.voice_id,
        voice_mode,
        prepared.language,
        len(payload.raw_text()),
        len(prepared.text),
        _normalization_summary(payload.normalization_options),
        prepared.voice_ref_audio_path.name if prepared.voice_ref_audio_path is not None else "-",
        ref_text_source,
        len(prepared.voice_ref_text or prepared.ref_text or ""),
        _truncate_preview(payload.raw_text()),
        _truncate_preview(prepared.text),
    )
    prepared, audio_bytes, media_type = await _synthesize_prepared(
        prepared,
        payload,
        request_id=request_id,
    )
    _, suffix = SUPPORTED_RESPONSE_FORMATS[prepared.response_format]
    headers = {
        "Content-Disposition": f'attachment; filename="speech.{suffix}"',
        "X-OmniVoice-Text-Chunks": str(len(prepared.chunk_plan)),
        "X-OmniVoice-Forced-Chunking": str(prepared.force_sentence_chunking).lower(),
        "X-OmniVoice-Voice-Mode": voice_mode,
        "X-OmniVoice-Sanitized-Chars": str(len(prepared.text)),
        "X-OmniVoice-Language": prepared.language or "",
        "X-OmniVoice-Voice-Source": (
            "local-reference" if prepared.voice_ref_audio_path is not None else "style-prompt"
        ),
    }
    return Response(content=audio_bytes, media_type=media_type, headers=headers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice OpenAI-compatible TTS server"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--model-id", default=BACKEND_MODEL_ID)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=float(os.getenv("OMNIVOICE_IDLE_TIMEOUT", "300")),
        help="Seconds of inactivity before offloading model from GPU (0 to disable)",
    )
    return parser


def main() -> None:
    _configure_logging()
    args = build_parser().parse_args()

    global service
    service = OmniVoiceService(
        args.model_id, args.device, idle_timeout=args.idle_timeout
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True,
        workers=1,
    )


if __name__ == "__main__":
    main()
