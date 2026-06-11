import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from omnivoice.audiobook import openrouter_cli


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _write_docx(path: Path, paragraphs=3):
    body = "".join(
        f"<w:p><w:r><w:t>Paragrafo {i} com texto para chunk.</w:t></w:r></w:p>"
        for i in range(paragraphs)
    )
    document_xml = f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)


class OpenRouterCliTest(unittest.TestCase):
    def test_preview_only_writes_no_provider_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "book.docx"
            output = Path(tmp) / "preview.json"
            _write_docx(docx)
            argv = [
                "cmd",
                "--docx",
                str(docx),
                "--output",
                str(output),
                "--model",
                "test/model",
                "--preview-only",
            ]

            with mock.patch.object(sys, "argv", argv):
                openrouter_cli.main()

            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(data["provider_call"])
            self.assertIn("chunk", data)

    def test_chunk_index_out_of_range_fails_before_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx = Path(tmp) / "book.docx"
            _write_docx(docx)
            argv = [
                "cmd",
                "--docx",
                str(docx),
                "--output",
                str(Path(tmp) / "out.json"),
                "--model",
                "test/model",
                "--chunk-index",
                "99",
            ]

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    openrouter_cli.main()


if __name__ == "__main__":
    unittest.main()
