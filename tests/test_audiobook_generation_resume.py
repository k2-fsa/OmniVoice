import tempfile
import unittest
from pathlib import Path

from omnivoice.audiobook.generation import (
    AudiobookGenerationJob,
    load_generation_checkpoint,
    write_generation_checkpoint,
)
from omnivoice.audiobook.schema import (
    AudiobookChapter,
    AudiobookPlan,
    AudiobookProject,
    AudiobookQcTargets,
    AudiobookSegment,
    VoiceProfile,
)


def _plan():
    return AudiobookPlan(
        project=AudiobookProject(
            title="Livro",
            author="Autor",
            language="pt-BR",
            genre="technical",
            source_docx_hash="hash",
            created_at="2026-06-11T00:00:00+00:00",
        ),
        voice_profile=VoiceProfile(),
        chapters=[
            AudiobookChapter(
                id="ch_001",
                title="Capitulo",
                order=1,
                segments=[
                    AudiobookSegment(
                        id="seg_001",
                        text="A",
                        text_hash="ha",
                        speaker="narrator",
                        pause_after_ms=500,
                        speed=0.9,
                        tone="neutral",
                        chapter_id="ch_001",
                    ),
                    AudiobookSegment(
                        id="seg_002",
                        text="B",
                        text_hash="hb",
                        speaker="narrator",
                        pause_after_ms=700,
                        speed=0.9,
                        tone="neutral",
                        chapter_id="ch_001",
                        status="failed",
                        error="old",
                    ),
                ],
            )
        ],
        qc_targets=AudiobookQcTargets(),
    )


class AudiobookGenerationResumeTest(unittest.TestCase):
    def test_pending_next_progress_and_marks(self):
        job = AudiobookGenerationJob(_plan())

        self.assertEqual([item.segment_id for item in job.pending_segments()], ["seg_001", "seg_002"])
        self.assertEqual(job.next_segment().segment_id, "seg_001")

        job.mark_generated("seg_001", "audio/seg_001.wav")
        job.mark_failed("seg_002", "boom")

        self.assertEqual(job.progress(), {"total": 2, "pending": 0, "generated": 1, "failed": 1})
        self.assertEqual(job.plan.chapters[0].segments[0].audio_path, "audio/seg_001.wav")
        self.assertIsNone(job.plan.chapters[0].segments[0].error)
        self.assertEqual(job.plan.chapters[0].segments[1].error, "boom")

    def test_checkpoint_roundtrip(self):
        plan = _plan()
        AudiobookGenerationJob(plan).mark_generated("seg_001", "audio/seg_001.wav")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = write_generation_checkpoint(plan, Path(tmp) / "checkpoint.json")
            loaded = load_generation_checkpoint(checkpoint)

        self.assertEqual(loaded.chapters[0].segments[0].status, "generated")
        self.assertEqual(loaded.chapters[0].segments[0].audio_path, "audio/seg_001.wav")
        self.assertEqual(loaded.project.title, "Livro")


if __name__ == "__main__":
    unittest.main()
