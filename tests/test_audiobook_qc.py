import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omnivoice.audiobook import qc_cli
from omnivoice.audiobook.qc import inspect_audiobook_plan_audio
from omnivoice.audiobook.schema import (
    AudiobookChapter,
    AudiobookPlan,
    AudiobookProject,
    AudiobookQcTargets,
    AudiobookSegment,
    VoiceProfile,
)


class FakeProbeRunner:
    def __init__(self, duration="1.25", sample_rate="44100"):
        self.stdout = json.dumps(
            {
                "format": {
                    "duration": duration,
                    "tags": {"peak_dbfs": "-4.0", "loudness_lufs": "-20.0"},
                },
                "streams": [{"codec_type": "audio", "sample_rate": sample_rate, "channels": 1}],
            }
        )

    def __call__(self, command, capture_output, text, check):
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout, stderr="")


def _base_plan(audio_path=None, status="generated"):
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
                        text="Texto.",
                        text_hash="hash",
                        speaker="narrator",
                        pause_after_ms=500,
                        speed=0.92,
                        tone="neutral",
                        chapter_id="ch_001",
                        status=status,
                        audio_path=audio_path,
                    )
                ],
            )
        ],
        qc_targets=AudiobookQcTargets(),
    )


class AudiobookQcTest(unittest.TestCase):
    def test_qc_passes_with_generated_readable_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "seg.wav"
            audio.write_bytes(b"audio")
            report = inspect_audiobook_plan_audio(
                _base_plan(str(audio)),
                runner=FakeProbeRunner(),
                ffprobe_path="ffprobe",
            )

            self.assertEqual(report.gate_status, "pass")
            self.assertAlmostEqual(report.duration_seconds, 1.25)
            self.assertEqual(report.sample_rate_hz, 44100)
            self.assertEqual(report.channel_count, 1)
            self.assertEqual(report.peak_dbfs, -4.0)
            self.assertEqual(report.loudness_lufs, -20.0)
            self.assertEqual(report.required_fixes, [])

    def test_qc_fails_for_pending_missing_and_zero_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "zero.wav"
            audio.write_bytes(b"")
            plan = _base_plan(str(audio), status="pending")

            report = inspect_audiobook_plan_audio(plan, runner=FakeProbeRunner(), ffprobe_path="ffprobe")

            self.assertEqual(report.gate_status, "fail")
            self.assertEqual(report.pending_segments, ["seg_001"])
            self.assertEqual(report.zero_byte_segments, ["seg_001"])
            self.assertIn("Generate pending segments", report.required_fixes)

    def test_qc_fails_for_sample_rate_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "seg.wav"
            audio.write_bytes(b"audio")
            report = inspect_audiobook_plan_audio(
                _base_plan(str(audio)),
                runner=FakeProbeRunner(sample_rate="48000"),
                ffprobe_path="ffprobe",
            )

            self.assertEqual(report.gate_status, "fail")
            self.assertIn("Normalize sample rate to 44100 Hz", report.required_fixes)

    def test_qc_fails_for_peak_loudness_and_channels(self):
        class BadProbeRunner:
            def __call__(self, command, capture_output, text, check):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "format": {
                                "duration": "1.0",
                                "tags": {"peak_dbfs": "-1.0", "loudness_lufs": "-15.0"},
                            },
                            "streams": [{"codec_type": "audio", "sample_rate": "44100", "channels": 2}],
                        }
                    ),
                    stderr="",
                )

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "seg.wav"
            audio.write_bytes(b"audio")
            report = inspect_audiobook_plan_audio(
                _base_plan(str(audio)),
                runner=BadProbeRunner(),
                ffprobe_path="ffprobe",
            )

            self.assertEqual(report.gate_status, "fail")
            self.assertIn("Reduce peak level to -3.0 dBFS or lower", report.required_fixes)
            self.assertIn("Normalize loudness to -20.0 LUFS", report.required_fixes)
            self.assertIn("Normalize audio to mono", report.required_fixes)

    def test_qc_cli_exits_2_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = tmp_path / "checkpoint.json"
            report = tmp_path / "qc_report.json"
            plan.write_text(json.dumps(_base_plan(status="pending").to_dict()), encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                ["cmd", "--plan", str(plan), "--output", str(report)],
            ):
                with self.assertRaises(SystemExit) as ctx:
                    qc_cli.main()

            self.assertEqual(ctx.exception.code, 2)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["gate_status"], "fail")


if __name__ == "__main__":
    unittest.main()
