import unittest

from omnivoice.narration.parser import parse_narration_text


class NarrationParserTest(unittest.TestCase):
    def test_strips_slide_labels_and_keeps_narration(self):
        plan = parse_narration_text(
            "Slide 1: Abertura\nHoje vamos falar sobre segurança.\n\nSlide 2\nNa dúvida, valide.",
            preset_name="Presentation",
            global_speed=0.9,
            remove_slide_labels=True,
        )

        texts = [segment.text for segment in plan.segments]
        self.assertEqual(texts, ["Hoje vamos falar sobre segurança.", "Na dúvida, valide."])
        self.assertNotIn("Slide", " ".join(texts))
        self.assertEqual(plan.segments[0].pause_after_ms, 1400)

    def test_manual_pause_and_speed_markers(self):
        plan = parse_narration_text(
            "[speed:0.8]\nPrimeira frase. [pause:1.2s]\nSegunda frase. [pause:500]",
            preset_name="Manual",
            global_speed=1.0,
        )

        self.assertEqual(len(plan.segments), 2)
        self.assertEqual(plan.segments[0].pause_after_ms, 1200)
        self.assertEqual(plan.segments[1].pause_after_ms, 500)
        self.assertAlmostEqual(plan.segments[0].speed, 0.8)
        self.assertNotIn("[pause", plan.segments[0].text)

    def test_cleans_slide_artifacts_before_tts(self):
        plan = parse_narration_text(
            "\n".join(
                [
                    "Slide 3: Diagnóstico",
                    "### O problema",
                    "• O sistema lê bullets, emojis 🔥 e setas → sem limpar. [pause:800]",
                    "---",
                    "Fonte: material local",
                    "2/10",
                    "A solução precisa separar o roteiro em frases humanas.",
                ]
            ),
            preset_name="Presentation",
            global_speed=0.92,
            remove_slide_labels=True,
        )

        texts = [segment.text for segment in plan.segments]
        joined = " ".join(texts)
        self.assertEqual(
            texts,
            [
                "O sistema lê bullets, emojis e setas, sem limpar.",
                "A solução precisa separar o roteiro em frases humanas.",
            ],
        )
        self.assertEqual(plan.segments[0].pause_after_ms, 800)
        self.assertNotIn("•", joined)
        self.assertNotIn("🔥", joined)
        self.assertNotIn("Fonte", joined)
        self.assertNotIn("http", joined)
        self.assertNotIn("2/10", joined)


if __name__ == "__main__":
    unittest.main()
