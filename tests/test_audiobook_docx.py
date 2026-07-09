import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from omnivoice.audiobook.docx import extract_docx_structure
from omnivoice.audiobook.planner import (
    AudiobookPlanConfig,
    create_audiobook_plan_from_docx,
    plan_to_json,
)


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _paragraph(text, style=None):
    style_xml = ""
    if style:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
    return (
        f"<w:p>{style_xml}<w:r><w:t>{text}</w:t></w:r></w:p>"
    )


def _write_docx(path: Path) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>'
        + _paragraph("Introducao", "Heading1")
        + _paragraph("Este capitulo explica o sistema local.")
        + _paragraph("Ele nao deve enviar texto para a internet.")
        + _paragraph("Capitulo Dois", "Heading1")
        + _paragraph("Agora temos outro bloco narravel.")
        + "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)


class AudiobookDocxTest(unittest.TestCase):
    def test_extract_docx_structure_reads_paragraphs_and_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "book.docx"
            _write_docx(docx_path)

            document = extract_docx_structure(docx_path)

            self.assertEqual(document.paragraphs[0].text, "Introducao")
            self.assertEqual(document.paragraphs[0].style, "Heading1")
            self.assertEqual(len(document.sha256), 64)
            self.assertIn("sistema local", document.plain_text)

    def test_create_audiobook_plan_from_docx(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "book.docx"
            _write_docx(docx_path)

            plan = create_audiobook_plan_from_docx(
                docx_path,
                AudiobookPlanConfig(
                    title="Livro Local",
                    author="Teste",
                    genre="technical",
                    speed=0.9,
                ),
            )
            data = json.loads(plan_to_json(plan))

            self.assertEqual(data["project"]["title"], "Livro Local")
            self.assertEqual(data["project"]["genre"], "technical")
            self.assertEqual(data["voice_profile"]["style"], "technical_clear")
            self.assertEqual(len(data["chapters"]), 2)
            self.assertEqual(data["chapters"][0]["title"], "Introducao")
            self.assertTrue(data["chapters"][0]["segments"])
            self.assertEqual(data["settings"]["offline_required"], True)
            self.assertIsNone(data["settings"]["external_provider"])

    def test_fiction_profile_sets_narrative_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "novel.docx"
            _write_docx(docx_path)

            plan = create_audiobook_plan_from_docx(
                docx_path,
                AudiobookPlanConfig(title="Romance", genre="fiction"),
            )

            self.assertEqual(plan.voice_profile.style, "fiction_narrative")
            self.assertEqual(plan.qc_targets.target_words_per_minute, 158)
            self.assertEqual(plan.chapters[0].segments[0].tone, "narrative")


if __name__ == "__main__":
    unittest.main()
