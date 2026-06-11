import json
import os
import unittest
import urllib.error
import urllib.request

from omnivoice.audiobook.chunking import AudiobookChunk
from omnivoice.audiobook.openrouter import (
    OpenRouterAudiobookClient,
    OpenRouterConfig,
    OpenRouterError,
    build_openrouter_payload,
)


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: urllib.request.Request, timeout_seconds: int) -> bytes:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return json.dumps(response).encode("utf-8")


def _chunk():
    return AudiobookChunk(
        id="chunk_0001",
        index=0,
        title="Bloco 1",
        text="Texto do livro.",
        word_count=3,
        paragraph_start=0,
        paragraph_end=0,
    )


class OpenRouterAudiobookTest(unittest.TestCase):
    def setUp(self):
        self.old_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"

    def tearDown(self):
        if self.old_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self.old_key

    def test_payload_enforces_json_schema_and_privacy(self):
        payload = build_openrouter_payload(
            _chunk(),
            OpenRouterConfig(model="test/model"),
            language="pt-BR",
            genre="technical",
        )

        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertEqual(payload["provider"]["data_collection"], "deny")
        self.assertTrue(payload["provider"]["zdr"])

    def test_missing_consent_does_not_call_transport(self):
        transport = FakeTransport([])
        client = OpenRouterAudiobookClient(
            OpenRouterConfig(model="test/model", require_model_support=False),
            transport=transport,
        )

        with self.assertRaises(OpenRouterError):
            client.structure_chunk(_chunk(), language="pt-BR", genre="technical", consent=False)

        self.assertEqual(transport.requests, [])

    def test_missing_api_key_fails_before_request(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        transport = FakeTransport([])
        client = OpenRouterAudiobookClient(
            OpenRouterConfig(model="test/model", require_model_support=False),
            transport=transport,
        )

        with self.assertRaises(OpenRouterError):
            client.structure_chunk(_chunk(), language="pt-BR", genre="technical", consent=True)

        self.assertEqual(transport.requests, [])

    def test_model_support_and_structured_response(self):
        transport = FakeTransport(
            [
                {
                    "data": [
                        {
                            "id": "test/model",
                            "supported_parameters": ["response_format", "structured_outputs"],
                        }
                    ]
                },
                {
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "chapters": [
                                            {
                                                "title": "Capitulo",
                                                "segments": [
                                                    {
                                                        "text": "Texto narravel.",
                                                        "speaker": "narrator",
                                                        "pause_after_ms": 750,
                                                        "speed": 0.92,
                                                        "tone": "neutral",
                                                        "pronunciation_notes": [],
                                                    }
                                                ],
                                            }
                                        ],
                                        "warnings": [],
                                    }
                                )
                            }
                        }
                    ],
                },
            ]
        )
        client = OpenRouterAudiobookClient(OpenRouterConfig(model="test/model"), transport=transport)

        result = client.structure_chunk(_chunk(), language="pt-BR", genre="technical", consent=True)

        self.assertEqual(result.content["chapters"][0]["title"], "Capitulo")
        self.assertEqual(len(transport.requests), 2)

    def test_rejects_extra_fields(self):
        transport = FakeTransport(
            [
                {
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "chapters": [],
                                        "warnings": [],
                                        "unexpected": True,
                                    }
                                )
                            }
                        }
                    ],
                }
            ]
        )
        client = OpenRouterAudiobookClient(
            OpenRouterConfig(model="test/model", require_model_support=False),
            transport=transport,
        )

        with self.assertRaises(OpenRouterError):
            client.structure_chunk(_chunk(), language="pt-BR", genre="technical", consent=True)

    def test_retries_transient_url_error(self):
        transport = FakeTransport(
            [
                urllib.error.URLError("temporary"),
                {
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "chapters": [
                                            {
                                                "title": "Capitulo",
                                                "segments": [
                                                    {
                                                        "text": "Texto.",
                                                        "speaker": "narrator",
                                                        "pause_after_ms": 700,
                                                        "speed": 0.92,
                                                        "tone": "neutral",
                                                    }
                                                ],
                                            }
                                        ],
                                        "warnings": [],
                                    }
                                )
                            }
                        }
                    ],
                },
            ]
        )
        client = OpenRouterAudiobookClient(
            OpenRouterConfig(model="test/model", require_model_support=False, max_retries=1),
            transport=transport,
        )

        result = client.structure_chunk(_chunk(), language="pt-BR", genre="technical", consent=True)

        self.assertEqual(result.content["chapters"][0]["title"], "Capitulo")
        self.assertEqual(len(transport.requests), 2)


if __name__ == "__main__":
    unittest.main()
