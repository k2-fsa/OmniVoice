from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from omnivoice.audiobook.chunking import AudiobookChunk


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class OpenRouterError(RuntimeError):
    pass


@dataclass
class OpenRouterConfig:
    model: str
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = OPENROUTER_CHAT_COMPLETIONS_URL
    models_url: str = OPENROUTER_MODELS_URL
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 90
    max_retries: int = 2
    data_collection: str = "deny"
    require_zero_data_retention: bool = True
    require_model_support: bool = True
    require_structured_outputs: bool = False
    site_url: Optional[str] = None
    app_name: str = "OmniVoice Local"
    response_healing: bool = False
    extra_models: List[str] = field(default_factory=list)


@dataclass
class OpenRouterChunkResult:
    chunk_id: str
    content: Dict[str, Any]
    model: Optional[str] = None


Transport = Callable[[urllib.request.Request, int], bytes]


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


def build_openrouter_messages(chunk: AudiobookChunk, *, language: str, genre: str) -> List[Dict[str, str]]:
    system = (
        "You structure manuscript chunks into audiobook narration JSON. "
        "Preserve author meaning. Do not rewrite prose except minimal cleanup for narration. "
        "Return only data matching the provided schema."
    )
    user = {
        "chunk_id": chunk.id,
        "language": language,
        "genre": genre,
        "previous_summary": chunk.previous_summary or "",
        "instructions": [
            "Split into audiobook-ready segments.",
            "Keep each segment suitable for TTS.",
            "Use pause_after_ms and speed to improve listening quality.",
            "For technical books, preserve commands, acronyms, and terms.",
            "For fiction, preserve dialogue and scene rhythm.",
        ],
        "text": chunk.text,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_openrouter_payload(
    chunk: AudiobookChunk,
    config: OpenRouterConfig,
    *,
    language: str,
    genre: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": build_openrouter_messages(chunk, language=language, genre=genre),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "audiobook_chunk_plan",
                "strict": True,
                "schema": audiobook_chunk_schema(),
            },
        },
        "provider": {
            "require_parameters": True,
            "data_collection": config.data_collection,
        },
    }
    if config.require_zero_data_retention:
        payload["provider"]["zdr"] = True
    if config.extra_models:
        payload["models"] = config.extra_models
    if config.response_healing:
        payload["plugins"] = [{"id": "response-healing"}]
    return payload


def validate_openrouter_chunk_content(content: Dict[str, Any]) -> None:
    allowed_top = {"chapters", "warnings"}
    extra_top = set(content) - allowed_top
    if extra_top:
        raise OpenRouterError(f"OpenRouter response has unexpected top-level fields: {sorted(extra_top)}")
    chapters = content.get("chapters")
    warnings = content.get("warnings")
    if not isinstance(chapters, list) or not isinstance(warnings, list):
        raise OpenRouterError("OpenRouter response must include list fields: chapters and warnings")
    for chapter in chapters:
        if not isinstance(chapter, dict):
            raise OpenRouterError("Each chapter must be an object")
        chapter_extra = set(chapter) - {"title", "segments"}
        if chapter_extra:
            raise OpenRouterError(f"Chapter has unexpected fields: {sorted(chapter_extra)}")
        if not isinstance(chapter.get("title"), str):
            raise OpenRouterError("Chapter title must be a string")
        segments = chapter.get("segments")
        if not isinstance(segments, list):
            raise OpenRouterError("Chapter segments must be a list")
        for segment in segments:
            if not isinstance(segment, dict):
                raise OpenRouterError("Each segment must be an object")
            segment_extra = set(segment) - {
                "text",
                "speaker",
                "pause_after_ms",
                "speed",
                "tone",
                "pronunciation_notes",
            }
            if segment_extra:
                raise OpenRouterError(f"Segment has unexpected fields: {sorted(segment_extra)}")
            for key in ["text", "speaker", "tone"]:
                if not isinstance(segment.get(key), str) or not segment.get(key):
                    raise OpenRouterError(f"Segment {key} must be a non-empty string")
            if not isinstance(segment.get("pause_after_ms"), int):
                raise OpenRouterError("Segment pause_after_ms must be an integer")
            if not isinstance(segment.get("speed"), (int, float)):
                raise OpenRouterError("Segment speed must be numeric")


def _default_transport(request: urllib.request.Request, timeout_seconds: int) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


class OpenRouterAudiobookClient:
    def __init__(self, config: OpenRouterConfig, transport: Optional[Transport] = None):
        self.config = config
        self.transport = transport or _default_transport

    def _api_key(self) -> str:
        value = os.environ.get(self.config.api_key_env)
        if not value:
            raise OpenRouterError(f"Missing required environment variable: {self.config.api_key_env}")
        return value

    def _send(self, request: urllib.request.Request) -> bytes:
        retry_statuses = {408, 429, 502, 503}
        last_error: Optional[BaseException] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self.transport(request, self.config.timeout_seconds)
            except urllib.error.HTTPError as exc:
                last_error = exc
                should_retry = exc.code in retry_statuses and attempt < self.config.max_retries
                if not should_retry:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.25 * (attempt + 1)
                time.sleep(min(delay, 2.0))
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
                time.sleep(0.25 * (attempt + 1))
        raise OpenRouterError(f"OpenRouter request failed after retries: {last_error}")

    def validate_model_support(self) -> None:
        api_key = self._api_key()
        request = urllib.request.Request(
            self.config.models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        try:
            raw = self._send(request)
            response = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise OpenRouterError(f"Could not validate OpenRouter model support: {exc}") from exc

        for model in response.get("data", []):
            if model.get("id") != self.config.model:
                continue
            supported = set(model.get("supported_parameters") or [])
            if not supported.intersection({"response_format", "structured_outputs"}):
                raise OpenRouterError(
                    f"OpenRouter model does not advertise structured output support: {self.config.model}"
                )
            if self.config.require_structured_outputs and "structured_outputs" not in supported:
                raise OpenRouterError(
                    f"OpenRouter model does not advertise json_schema structured outputs: {self.config.model}"
                )
            return
        raise OpenRouterError(f"OpenRouter model not found in models response: {self.config.model}")

    def structure_chunk(
        self,
        chunk: AudiobookChunk,
        *,
        language: str,
        genre: str,
        consent: bool = False,
    ) -> OpenRouterChunkResult:
        if not consent:
            raise OpenRouterError("Explicit online-provider consent is required before sending a chunk")
        if self.config.require_model_support:
            self.validate_model_support()
        payload = build_openrouter_payload(chunk, self.config, language=language, genre=genre)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url or "http://127.0.0.1",
            "X-Title": self.config.app_name,
            "X-OpenRouter-Title": self.config.app_name,
        }
        request = urllib.request.Request(
            self.config.base_url,
            data=body,
            headers=headers,
            method="POST",
        )
        raw = self._send(request)

        try:
            response = json.loads(raw.decode("utf-8"))
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if isinstance(content, str):
                content_data = json.loads(content)
            elif isinstance(content, dict):
                content_data = content
            else:
                raise TypeError("message.content must be JSON string or object")
            validate_openrouter_chunk_content(content_data)
        except Exception as exc:
            raise OpenRouterError(f"Invalid OpenRouter structured response: {exc}") from exc

        return OpenRouterChunkResult(
            chunk_id=chunk.id,
            content=content_data,
            model=response.get("model"),
        )
