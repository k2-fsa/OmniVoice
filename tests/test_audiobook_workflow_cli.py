import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from omnivoice.audiobook import workflow_cli


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _write_docx(path: Path):
    xml = (
        f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>'
        "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr><w:r><w:t>Capitulo</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Texto original.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)


def _structured_result(path: Path):
    path.write_text(
        json.dumps(
            {
                "chapters": [
                    {
                        "title": "Capitulo",
                        "segments": [
                            {
                                "text": "Texto narravel.",
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
        ),
        encoding="utf-8",
    )


class AudiobookWorkflowCliTest(unittest.TestCase):
    def test_merge_openrouter_result_then_status_and_marks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docx = tmp_path / "book.docx"
            result = tmp_path / "chunk.json"
            plan = tmp_path / "plan.json"
            generated = tmp_path / "generated.json"
            failed = tmp_path / "failed.json"
            _write_docx(docx)
            _structured_result(result)

            merge_argv = [
                "cmd",
                "merge-openrouter",
                "--docx",
                str(docx),
                "--result",
                str(result),
                "--output",
                str(plan),
                "--title",
                "Livro",
                "--model",
                "test/model",
            ]
            with mock.patch.object(sys, "argv", merge_argv):
                workflow_cli.main()

            data = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(data["settings"]["external_provider"], "openrouter")
            self.assertEqual(data["chapters"][0]["segments"][0]["status"], "pending")

            status_argv = ["cmd", "status", "--plan", str(plan)]
            captured = io.StringIO()
            with mock.patch.object(sys, "argv", status_argv), mock.patch("sys.stdout", captured):
                workflow_cli.main()
            self.assertEqual(json.loads(captured.getvalue()), {"total": 1, "pending": 1, "generated": 0, "failed": 0})

            mark_generated_argv = [
                "cmd",
                "mark-generated",
                "--plan",
                str(plan),
                "--segment-id",
                "seg_001_000001",
                "--audio-path",
                "audio/seg.wav",
                "--output",
                str(generated),
            ]
            with mock.patch.object(sys, "argv", mark_generated_argv):
                workflow_cli.main()
            generated_data = json.loads(generated.read_text(encoding="utf-8"))
            self.assertEqual(generated_data["chapters"][0]["segments"][0]["status"], "generated")
            self.assertEqual(generated_data["chapters"][0]["segments"][0]["audio_path"], "audio/seg.wav")

            mark_failed_argv = [
                "cmd",
                "mark-failed",
                "--plan",
                str(generated),
                "--segment-id",
                "seg_001_000001",
                "--error",
                "render failed",
                "--output",
                str(failed),
            ]
            with mock.patch.object(sys, "argv", mark_failed_argv):
                workflow_cli.main()
            failed_data = json.loads(failed.read_text(encoding="utf-8"))
            self.assertEqual(failed_data["chapters"][0]["segments"][0]["status"], "failed")
            self.assertEqual(failed_data["chapters"][0]["segments"][0]["error"], "render failed")

    def test_next_redacts_text_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docx = tmp_path / "book.docx"
            result = tmp_path / "chunk.json"
            plan = tmp_path / "plan.json"
            _write_docx(docx)
            _structured_result(result)

            with mock.patch.object(
                sys,
                "argv",
                [
                    "cmd",
                    "merge-openrouter",
                    "--docx",
                    str(docx),
                    "--result",
                    str(result),
                    "--output",
                    str(plan),
                    "--title",
                    "Livro",
                    "--model",
                    "test/model",
                ],
            ):
                workflow_cli.main()

            captured = io.StringIO()
            with mock.patch.object(sys, "argv", ["cmd", "next", "--plan", str(plan)]), mock.patch(
                "sys.stdout", captured
            ):
                workflow_cli.main()

            data = json.loads(captured.getvalue())
            self.assertEqual(data["segment_id"], "seg_001_000001")
            self.assertTrue(data["text_redacted"])
            self.assertNotIn("text", data)

    def test_merge_rejects_invalid_result_without_partial_output(self):
        bad_results = [
            {"chapters": [], "warnings": []},
            {"chapters": [{"title": "Capitulo", "segments": []}], "warnings": []},
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
                                "extra": True,
                            }
                        ],
                    }
                ],
                "warnings": [],
            },
            {
                "chapters": [
                    {
                        "title": "Capitulo",
                        "segments": [
                            {
                                "text": "Texto.",
                                "speaker": "narrator",
                                "pause_after_ms": "700",
                                "speed": 0.92,
                                "tone": "neutral",
                            }
                        ],
                    }
                ],
                "warnings": [],
            },
        ]
        for index, bad_result in enumerate(bad_results):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    docx = tmp_path / "book.docx"
                    result = tmp_path / "bad.json"
                    plan = tmp_path / "plan.json"
                    _write_docx(docx)
                    result.write_text(json.dumps(bad_result), encoding="utf-8")

                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "cmd",
                            "merge-openrouter",
                            "--docx",
                            str(docx),
                            "--result",
                            str(result),
                            "--output",
                            str(plan),
                            "--title",
                            "Livro",
                            "--model",
                            "test/model",
                        ],
                    ):
                        with self.assertRaises(ValueError):
                            workflow_cli.main()

                    self.assertFalse(plan.exists())


if __name__ == "__main__":
    unittest.main()
