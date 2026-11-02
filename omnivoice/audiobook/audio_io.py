#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio assembly and export for audiobooks.

A whole book does not fit in memory, so segments are streamed to disk as they
are generated instead of being concatenated in RAM at the end.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

FORMATS = ("mp3", "wav", "m4a", "ogg")

# Formats libsndfile writes on its own (MP3 support arrived in libsndfile 1.1).
# Only AAC/M4A still needs ffmpeg.
_SNDFILE_FORMATS = {"mp3": "MP3", "wav": "WAV", "ogg": "OGG"}

_CODECS = {
    "m4a": ["-codec:a", "aac", "-b:a", "192k"],
}


class ExportError(RuntimeError):
    pass


def find_ffmpeg() -> str | None:
    """Locate ffmpeg, if the machine has one. Only AAC/M4A export needs it."""
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")


def _sndfile_format(fmt: str) -> str | None:
    """The libsndfile format name for ``fmt``, or None if it needs ffmpeg."""
    name = _SNDFILE_FORMATS.get(fmt)
    if name and name in sf.available_formats():
        return name
    return None


def _offered_formats() -> str:
    """Human-readable list of the formats this install can write without ffmpeg."""
    names = [fmt.upper() for fmt in _SNDFILE_FORMATS if _sndfile_format(fmt)]
    if len(names) <= 1:
        return names[0] if names else "WAV"
    return ", ".join(names[:-1]) + f" or {names[-1]}"


def unique_path(directory: Path, stem: str, extension: str) -> Path:
    """``book.mp3``, then ``book (2).mp3``, … so nothing is overwritten."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}.{extension}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem} ({counter}).{extension}"
        counter += 1
    return candidate


class AudioSink:
    """Writes speech to the output file one segment at a time.

    An hour of 24 kHz mono float audio is ~350 MB, and joining the segments at
    the end briefly doubles that. So each segment is streamed straight to disk
    as it is generated, with the pause between segments written in between.

    Everything libsndfile can encode (WAV, MP3, OGG) is written incrementally.
    AAC/M4A is the one exception: it goes to a temporary WAV that ffmpeg turns
    into the final file on :meth:`finish`.
    """

    def __init__(
        self,
        out_path: str | Path,
        sample_rate: int,
        fmt: str,
        gap_seconds: float = 0.0,
    ) -> None:
        self.out_path = Path(out_path)
        self.sample_rate = int(sample_rate)
        self.fmt = fmt.lower().lstrip(".")
        self.segments = 0
        self.frames = 0

        direct = _sndfile_format(self.fmt)
        self._tmpdir: Path | None = None
        self._encoding = direct or "WAV"

        if direct:
            target = self.out_path
        else:
            # Fail before any audio is generated, not hours later.
            if not find_ffmpeg():
                raise ExportError(
                    f"{self.fmt.upper()} export needs ffmpeg, which wasn't found "
                    f"on PATH. Pick {_offered_formats()} instead."
                )
            self._tmpdir = Path(tempfile.mkdtemp(prefix="omnivoice-"))
            target = self._tmpdir / "audiobook.wav"

        self._gap = np.zeros(
            int(self.sample_rate * max(0.0, gap_seconds)), dtype=np.float32
        )
        try:
            self._handle = sf.SoundFile(
                str(target),
                mode="w",
                samplerate=self.sample_rate,
                channels=1,
                format=self._encoding,
            )
        except Exception:
            # Do not leave a stray temp directory behind if the file won't open.
            self._remove_tmpdir()
            raise

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    def write(self, chunk) -> None:
        """Append one segment, preceded by the inter-segment pause if any."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if not chunk.size or self._handle is None:
            return
        if self.segments and self._gap.size:
            self._handle.write(self._gap)
            self.frames += self._gap.size
        self._handle.write(chunk)
        self.frames += chunk.size
        self.segments += 1

    def finish(self) -> Path:
        """Close the file, and run the ffmpeg pass for M4A. Safe to repeat."""
        self._close()

        if self._tmpdir is None:
            return self.out_path

        try:
            cmd = [
                find_ffmpeg(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(self._tmpdir / "audiobook.wav"),
            ]
            cmd += _CODECS.get(self.fmt, [])
            cmd.append(str(self.out_path))
            result = subprocess.run(
                cmd, capture_output=True, text=True, errors="replace"
            )
            if result.returncode != 0 or not self.out_path.exists():
                raise ExportError(
                    f"ffmpeg failed to write {self.fmt.upper()}: "
                    f"{result.stderr.strip()[:400]}"
                )
        finally:
            self._remove_tmpdir()
        return self.out_path

    def discard(self) -> None:
        """Close the file and delete everything this sink created."""
        self._close()
        self._remove_tmpdir()
        try:
            self.out_path.unlink()
        except OSError:
            pass

    def _close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001 - best effort on failure paths
                pass
            self._handle = None

    def _remove_tmpdir(self) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


def export(waveform, sample_rate: int, out_path: str | Path, fmt: str) -> Path:
    """Write a single waveform to disk in the requested format."""
    sink = AudioSink(out_path, sample_rate, fmt)
    try:
        sink.write(waveform)
        return sink.finish()
    except BaseException:
        sink.discard()
        raise
