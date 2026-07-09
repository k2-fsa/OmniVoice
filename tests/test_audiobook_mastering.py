import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omnivoice.audiobook.mastering import (
    ConcatOptions,
    FFmpegError,
    MasteringOptions,
    build_audio_filter,
    concat_audio_files,
    ffprobe_media_info,
    remaster_audio,
)


class FakeRunner:
    def __init__(self, stdout=""):
        self.commands = []
        self.stdout = stdout

    def __call__(self, command, capture_output, text, check):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=self.stdout, stderr="")


class AudiobookMasteringTest(unittest.TestCase):
    def test_build_audio_filter_contains_requested_steps(self):
        filters = build_audio_filter(
            MasteringOptions(
                tempo=1.1,
                trim_silence=True,
                dynamic_normalize=True,
                compressor=True,
                limiter=True,
            )
        )

        self.assertIn("silenceremove", filters)
        self.assertIn("atempo=1.1", filters)
        self.assertIn("dynaudnorm", filters)
        self.assertIn("acompressor", filters)
        self.assertIn("loudnorm=I=-20.00:TP=-3.00:LRA=11.00", filters)
        self.assertIn("alimiter", filters)

    def test_invalid_tempo_is_rejected(self):
        with self.assertRaises(ValueError):
            build_audio_filter(MasteringOptions(tempo=2.5))

    def test_concat_normalizes_stream_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.wav"
            second = Path(tmp) / "b.wav"
            output = Path(tmp) / "out.wav"
            first.write_bytes(b"x")
            second.write_bytes(b"x")
            runner = FakeRunner()

            concat_audio_files([first, second], output, runner=runner, ffmpeg_path="ffmpeg")

        command = runner.commands[0]
        self.assertIn("-n", command)
        self.assertIn("-ar", command)
        self.assertIn("44100", command)
        self.assertIn("pcm_s16le", command)

    def test_concat_copy_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.wav"
            output = Path(tmp) / "out.wav"
            first.write_bytes(b"x")
            runner = FakeRunner()

            concat_audio_files(
                [first],
                output,
                options=ConcatOptions(normalize_stream=False),
                runner=runner,
                ffmpeg_path="ffmpeg",
            )

            self.assertIn("-c", runner.commands[0])
            self.assertIn("copy", runner.commands[0])

    def test_remaster_fails_cleanly_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            source.write_bytes(b"x")
            with mock.patch("shutil.which", return_value=None):
                with self.assertRaises(FFmpegError):
                    remaster_audio(source, Path(tmp) / "out.wav")

    def test_remaster_forces_output_sample_rate_and_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            source.write_bytes(b"x")
            runner = FakeRunner()

            remaster_audio(source, Path(tmp) / "out.wav", runner=runner, ffmpeg_path="ffmpeg")

            command = runner.commands[0]
            self.assertIn("-n", command)
            self.assertIn("-ar", command)
            self.assertIn("44100", command)
            self.assertIn("-ac", command)
            self.assertIn("1", command)

    def test_mastering_rejects_source_output_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "same.wav"
            source.write_bytes(b"x")

            with self.assertRaises(FFmpegError):
                remaster_audio(source, source, runner=FakeRunner(), ffmpeg_path="ffmpeg")

            with self.assertRaises(FFmpegError):
                concat_audio_files([source], source, runner=FakeRunner(), ffmpeg_path="ffmpeg")

    def test_ffprobe_json_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.wav"
            source.write_bytes(b"x")
            runner = FakeRunner(stdout=json.dumps({"format": {"duration": "1.0"}}))

            info = ffprobe_media_info(source, runner=runner, ffprobe_path="ffprobe")

            self.assertEqual(info["format"]["duration"], "1.0")


if __name__ == "__main__":
    unittest.main()
