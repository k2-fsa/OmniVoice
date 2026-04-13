import unittest
from unittest.mock import AsyncMock, patch

import torch
from fastapi.testclient import TestClient

from omnivoice.openai_tts_server import (
    DEFAULT_AUDIO_CHUNK_THRESHOLD,
    TextSanitizationOptions,
    VOICE_LOOKUP,
    _resolve_voice,
    _supported_models,
    _supported_voices,
    app,
    sanitize_prompt_text,
    sanitize_speech_text,
    service,
)


class _FakeModel:
    def __init__(self) -> None:
        self.sampling_rate = 24000
        self.calls: list[dict[str, object]] = []
        self.voice_prompt_calls: list[dict[str, object]] = []

    def create_voice_clone_prompt(self, *, ref_audio: str, ref_text=None):
        prompt = {"ref_audio": ref_audio, "ref_text": ref_text}
        self.voice_prompt_calls.append(prompt)
        return prompt

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [torch.zeros(1, 480)]


class OpenAITTSServerTests(unittest.TestCase):
    def setUp(self) -> None:
        service._voice_prompt_cache.clear()

    def test_sanitize_speech_text_preserves_bracket_tags_and_removes_control_tokens(
        self,
    ) -> None:
        text = " Hello\tworld <|text_end|>\n[laughter] me@example.com "
        sanitized = sanitize_speech_text(
            text,
            language="en",
            options=TextSanitizationOptions(),
        )

        self.assertNotIn("<|text_end|>", sanitized)
        self.assertIn("[laughter]", sanitized)
        self.assertIn("me at example dot com", sanitized)
        self.assertEqual(sanitized, sanitized.strip())
        self.assertTrue(sanitized.endswith("."))

    def test_sanitize_prompt_text_strips_model_tokens(self) -> None:
        sanitized = sanitize_prompt_text(" <|lang_start|>  male, low pitch \n")
        self.assertEqual(sanitized, "male, low pitch")

    def test_sanitize_speech_text_applies_kokoro_style_normalization(self) -> None:
        sanitized = sanitize_speech_text(
            "Dr. Smith meets me at 10:05 pm. It costs $50.30 for 5km in 1998(s).",
            language="en",
            options=TextSanitizationOptions(unit_normalization=True),
        )

        self.assertIn("Doctor Smith", sanitized)
        self.assertIn("ten oh five pm", sanitized)
        self.assertIn("fifty dollars and thirty cents", sanitized)
        self.assertIn("five kilometers", sanitized)
        self.assertIn("nineteen ninety-eight", sanitized)

    def test_supported_models_and_voices_include_openwebui_facing_entries(self) -> None:
        model_ids = {model["id"] for model in _supported_models()}
        voice_ids = {voice["id"] for voice in _supported_voices()}

        self.assertIn("tts-1", model_ids)
        self.assertIn("gpt-4o-mini-tts", model_ids)
        self.assertIn("alloy", voice_ids)
        self.assertIn("british_man", voice_ids)

    def test_resolve_voice_prefers_local_reference_when_available(self) -> None:
        resolved = _resolve_voice("alloy")
        option = VOICE_LOOKUP["alloy"]

        self.assertEqual(resolved.voice_id, "alloy")
        if option.has_local_sample():
            self.assertIsNotNone(resolved.ref_audio_path)
            self.assertIsNone(resolved.instruct)
            self.assertTrue((resolved.ref_text or "").strip())
        else:
            self.assertIsNone(resolved.ref_audio_path)
            self.assertIsNotNone(resolved.instruct)

    def test_audio_endpoint_forces_sentence_chunking_for_long_input(self) -> None:
        fake_model = _FakeModel()
        long_text = " ".join(f"Sentence {idx}." for idx in range(1, 60))

        with (
            patch.object(service, "get_model", new=AsyncMock(return_value=fake_model)),
            patch(
                "omnivoice.openai_tts_server._waveform_to_bytes",
                return_value=(b"audio-bytes", "audio/mpeg"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": long_text,
                    "voice": "alloy",
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"audio-bytes")
        self.assertGreater(int(response.headers["x-omnivoice-text-chunks"]), 1)
        self.assertEqual(response.headers["x-omnivoice-forced-chunking"], "true")

        call = fake_model.calls[0]
        generation_config = call["generation_config"]
        self.assertEqual(generation_config.audio_chunk_threshold, 0.0)
        self.assertTrue(call["text"].endswith("."))
        if VOICE_LOOKUP["alloy"].has_local_sample():
            self.assertIn("voice_clone_prompt", call)
            self.assertTrue(call["voice_clone_prompt"]["ref_text"].strip())
        else:
            self.assertIn("instruct", call)

    def test_audio_endpoint_keeps_default_chunk_threshold_for_short_input(self) -> None:
        fake_model = _FakeModel()

        with (
            patch.object(service, "get_model", new=AsyncMock(return_value=fake_model)),
            patch(
                "omnivoice.openai_tts_server._waveform_to_bytes",
                return_value=(b"audio-bytes", "audio/mpeg"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "alloy",
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-omnivoice-text-chunks"], "1")
        self.assertEqual(response.headers["x-omnivoice-forced-chunking"], "false")

        call = fake_model.calls[0]
        generation_config = call["generation_config"]
        self.assertEqual(generation_config.audio_chunk_threshold, DEFAULT_AUDIO_CHUNK_THRESHOLD)
        if VOICE_LOOKUP["alloy"].has_local_sample():
            self.assertIn("voice_clone_prompt", call)
            self.assertTrue(call["voice_clone_prompt"]["ref_text"].strip())

    def test_audio_endpoint_allows_ref_text_override_for_local_voice(self) -> None:
        fake_model = _FakeModel()

        with (
            patch.object(service, "get_model", new=AsyncMock(return_value=fake_model)),
            patch(
                "omnivoice.openai_tts_server._waveform_to_bytes",
                return_value=(b"audio-bytes", "audio/mpeg"),
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "alloy",
                    "ref_text": "Custom reference sentence.",
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200)
        if VOICE_LOOKUP["alloy"].has_local_sample():
            self.assertEqual(
                fake_model.calls[0]["voice_clone_prompt"]["ref_text"],
                "Custom reference sentence.",
            )

    def test_audio_endpoint_validates_speed_bounds(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "alloy",
                    "speed": 4.5,
                },
            )

        self.assertEqual(response.status_code, 422)

    def test_frontend_page_shows_credits_and_voice_summary(self) -> None:
        with TestClient(app) as client:
            response = client.get("/ui")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Thanks to the original OmniVoice creators", response.text)
        self.assertIn("OpenAI-compatible TTS", response.text)
        self.assertIn("british_man", response.text)
        self.assertIn("cordobes_man", response.text)


if __name__ == "__main__":
    unittest.main()
