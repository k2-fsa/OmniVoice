import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _write_docx(path: Path) -> None:
    xml = (
        f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>'
        "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/></w:pPr><w:r><w:t>Capitulo</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Texto original para o fluxo completo.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)


def _write_structured_result(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "chapters": [
                    {
                        "title": "Capitulo",
                        "segments": [
                            {
                                "text": "Texto narravel para o fluxo completo.",
                                "speaker": "narrator",
                                "pause_after_ms": 500,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not available")
class AudiobookE2EFfmpegTest(unittest.TestCase):
    def test_docx_openrouter_merge_checkpoint_master_qc_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docx = tmp_path / "book.docx"
            result = tmp_path / "chunk-0001.json"
            plan = tmp_path / "audiobook_plan.json"
            checkpoint = tmp_path / "checkpoint.json"
            segment_wav = tmp_path / "seg.wav"
            master_wav = tmp_path / "master.wav"
            qc_report = tmp_path / "qc_report.json"
            _write_docx(docx)
            _write_structured_result(result)

            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "omnivoice.audiobook.workflow_cli",
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
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.25",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(segment_wav),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            before_hash = _sha256(segment_wav)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "omnivoice.audiobook.workflow_cli",
                    "mark-generated",
                    "--plan",
                    str(plan),
                    "--segment-id",
                    "seg_001_000001",
                    "--audio-path",
                    str(segment_wav),
                    "--output",
                    str(checkpoint),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "omnivoice.audiobook.mastering_cli",
                    "--input",
                    str(segment_wav),
                    "--output",
                    str(master_wav),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(_sha256(segment_wav), before_hash)

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    str(master_wav),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            probe_data = json.loads(probe.stdout)
            self.assertEqual(int(probe_data["streams"][0]["sample_rate"]), 44100)
            self.assertGreater(float(probe_data["format"]["duration"]), 0.0)

            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "omnivoice.audiobook.qc_cli",
                    "--plan",
                    str(checkpoint),
                    "--output",
                    str(qc_report),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(qc_report.read_text(encoding="utf-8"))["gate_status"], "pass")


if __name__ == "__main__":
    unittest.main()
