import unittest

from omnivoice.audiobook.docx import DocxDocument, DocxParagraph
from omnivoice.audiobook.planner import (
    AudiobookPlanConfig,
    create_audiobook_plan_from_openrouter_results,
)


class AudiobookOpenRouterPlanTest(unittest.TestCase):
    def _document(self):
        return DocxDocument(
            path="book.docx",
            sha256="abc123",
            paragraphs=[DocxParagraph(index=0, text="Texto.")],
        )

    def test_openrouter_results_become_local_plan(self):
        plan = create_audiobook_plan_from_openrouter_results(
            self._document(),
            AudiobookPlanConfig(title="Livro", genre="fiction", speed=0.94),
            [
                {
                    "chapters": [
                        {
                            "title": "Cena 1",
                            "segments": [
                                {
                                    "text": "Ela abriu a porta.",
                                    "speaker": "narrator",
                                    "pause_after_ms": 900,
                                    "speed": 0.94,
                                    "tone": "suspense",
                                }
                            ],
                        }
                    ],
                    "warnings": [],
                }
            ],
            model="test/model",
        )

        self.assertEqual(plan.project.source_docx_hash, "abc123")
        self.assertFalse(plan.settings["offline_required"])
        self.assertEqual(plan.settings["external_provider"], "openrouter")
        self.assertEqual(plan.settings["openrouter_model"], "test/model")
        self.assertEqual(plan.chapters[0].segments[0].id, "seg_001_000001")
        self.assertEqual(len(plan.chapters[0].segments[0].text_hash), 64)
        self.assertEqual(plan.chapters[0].segments[0].tone, "suspense")

    def test_malformed_items_are_ignored_but_empty_plan_fails(self):
        with self.assertRaises(ValueError):
            create_audiobook_plan_from_openrouter_results(
                self._document(),
                AudiobookPlanConfig(title="Livro"),
                [{"chapters": [{"title": "Vazio", "segments": [{"text": ""}]}]}],
                model="test/model",
            )

    def test_defaults_for_missing_segment_fields(self):
        plan = create_audiobook_plan_from_openrouter_results(
            self._document(),
            AudiobookPlanConfig(title="Livro", speed=0.91),
            [{"chapters": [{"title": "Capitulo", "segments": [{"text": "Texto limpo."}]}]}],
            model="test/model",
        )
        segment = plan.chapters[0].segments[0]

        self.assertEqual(segment.speaker, "narrator")
        self.assertEqual(segment.pause_after_ms, 750)
        self.assertAlmostEqual(segment.speed, 0.91)
        self.assertEqual(segment.tone, "neutral")


if __name__ == "__main__":
    unittest.main()
