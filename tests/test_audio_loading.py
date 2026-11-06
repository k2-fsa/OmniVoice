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

"""Tests for loading reference audio from containers soundfile cannot read.

Covers MP3/M4A reference audio: with librosa 1.0 the old fallback re-raised
soundfile's error, so voice cloning could not use those files at all.
"""

import io
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from omnivoice.utils.audio import load_audio, load_audio_bytes, load_waveform
from omnivoice.utils import audio as audio_module

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None, reason="ffmpeg is required to create and decode these containers"
)

# (container, ffmpeg encoder flags) for the formats soundfile cannot read.
CONTAINERS = [
    ("m4a", ["-c:a", "aac", "-f", "ipod"]),
    ("mp3", ["-c:a", "libmp3lame", "-f", "mp3"]),
]

SAMPLE_RATE = 24000
NUM_SAMPLES = SAMPLE_RATE  # one second


def _write_source_wav(tmp_path):
    """Write a one-second mono WAV the encoders below can re-encode."""
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal(NUM_SAMPLES) * 0.01).astype("float32")
    wav_path = tmp_path / "reference.wav"
    sf.write(wav_path, signal, SAMPLE_RATE)
    return wav_path


def _encode(source, target, flags):
    """Re-encode *source* into *target* with ffmpeg, failing the test on error."""
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(source)] + flags + [str(target)],
        check=True,
        capture_output=True,
    )
    return target


@requires_ffmpeg
@pytest.mark.parametrize("container,flags", CONTAINERS)
def test_load_waveform_reads_a_container_soundfile_cannot(tmp_path, container, flags):
    """A container soundfile cannot read is still decoded, not rejected.

    libsndfile 1.2+ decodes MP3 on its own, so the fallback is only exercised
    for formats it does not understand (M4A/AAC on every version so far); the
    assertion below is that the file loads either way.
    """
    source = _write_source_wav(tmp_path)
    target = _encode(source, tmp_path / f"reference.{container}", flags)

    data, sample_rate = load_waveform(str(target))
    assert sample_rate == SAMPLE_RATE
    # (C, T), mono, within the encoder's tolerance for the source length.
    assert data.shape[0] == 1
    assert abs(data.shape[1] - NUM_SAMPLES) < SAMPLE_RATE * 0.1


@requires_ffmpeg
def test_m4a_reference_audio_is_decoded_by_the_fallback(tmp_path):
    """M4A specifically: soundfile cannot read it, so the fallback must decode it.

    This is the format the bug report is about (recordings from macOS/iOS).
    """
    source = _write_source_wav(tmp_path)
    target = _encode(source, tmp_path / "reference.m4a", ["-c:a", "aac", "-f", "ipod"])

    # The premise: soundfile alone cannot read this container.
    with pytest.raises(Exception):
        sf.read(str(target))

    data, sample_rate = load_waveform(str(target))
    assert sample_rate == SAMPLE_RATE
    assert data.shape[0] == 1
    assert abs(data.shape[1] - NUM_SAMPLES) < SAMPLE_RATE * 0.1


@requires_ffmpeg
@pytest.mark.parametrize("container,flags", CONTAINERS)
def test_load_audio_resamples_a_container_soundfile_cannot(tmp_path, container, flags):
    """load_audio returns mono audio at the requested rate for those files."""
    source = _write_source_wav(tmp_path)
    target = _encode(source, tmp_path / f"reference.{container}", flags)

    data = load_audio(str(target), 16000)
    assert data.ndim == 2 and data.shape[0] == 1
    assert abs(data.shape[1] - 16000) < 16000 * 0.1


@requires_ffmpeg
@pytest.mark.parametrize("container,flags", CONTAINERS)
def test_load_audio_bytes_reads_the_same_container(tmp_path, container, flags):
    """The in-memory path decodes the same containers as the file path."""
    source = _write_source_wav(tmp_path)
    target = _encode(source, tmp_path / f"reference.{container}", flags)

    data = load_audio_bytes(target.read_bytes(), 16000)
    assert data.ndim == 2 and data.shape[0] == 1
    assert abs(data.shape[1] - 16000) < 16000 * 0.1


def test_load_waveform_still_reads_wav_without_ffmpeg(tmp_path):
    """A WAV keeps working through soundfile even when ffmpeg is absent."""
    source = _write_source_wav(tmp_path)

    data, sample_rate = load_waveform(str(source))
    assert sample_rate == SAMPLE_RATE
    assert data.shape == (1, NUM_SAMPLES)


@pytest.mark.parametrize("as_bytes", [False, True])
def test_ffmpeg_fallback_is_held_to_local_protocols(tmp_path, monkeypatch, as_bytes):
    """A reference file cannot steer ffmpeg onto the network.

    An uploaded container can be a playlist whose segments are HTTP URLs, so
    the fallback has to whitelist the local protocols only and leave the
    remote ones out.
    """
    captured = {}

    class _Refused:
        returncode = 1
        stdout = b""
        stderr = b"ffmpeg refused the input"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Refused()

    monkeypatch.setattr(audio_module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        if as_bytes:
            audio_module._decode_with_ffmpeg(b"#EXTM3U\n", is_bytes=True)
        else:
            playlist = tmp_path / "playlist.m3u8"
            playlist.write_text("#EXTM3U\n", encoding="utf-8")
            audio_module._decode_with_ffmpeg(str(playlist))

    cmd = captured["cmd"]
    assert cmd[cmd.index("-protocol_whitelist") + 1] == "file,pipe"
    # The restriction is an input option, so it has to precede the -i it guards.
    assert cmd.index("-protocol_whitelist") < cmd.index("-i")


def _encode_long_m4a(tmp_path, seconds=30):
    """Encode an M4A long enough that its index lands after the audio data."""
    rng = np.random.default_rng(0)
    signal = (rng.standard_normal(SAMPLE_RATE * seconds) * 0.01).astype("float32")
    wav_path = tmp_path / "long.wav"
    sf.write(wav_path, signal, SAMPLE_RATE)
    return _encode(wav_path, tmp_path / "long.m4a", ["-c:a", "aac", "-f", "ipod"])


@requires_ffmpeg
def test_load_audio_bytes_reads_a_container_whose_index_is_at_the_end(tmp_path):
    """A long M4A keeps its moov atom after the audio, so the input must seek.

    Decoded through a pipe this file yields an empty waveform at exit status 0,
    which is why the bytes go through a temporary file.
    """
    target = _encode_long_m4a(tmp_path)
    raw = target.read_bytes()

    # The premise: the index follows the audio, and the file is bigger than the
    # buffer ffmpeg would otherwise keep in memory for a pipe.
    assert raw.find(b"moov") > raw.find(b"mdat")
    assert len(raw) > 100_000

    data = load_audio_bytes(raw, 16000)
    assert data.shape[0] == 1
    assert abs(data.shape[1] - 16000 * 30) < 16000 * 3


def test_an_empty_decode_is_an_error(monkeypatch):
    """ffmpeg can exit 0 having decoded nothing; that is not a waveform."""
    header = io.BytesIO()
    sf.write(header, np.zeros(0, dtype="float32"), SAMPLE_RATE, format="WAV")

    class _Empty:
        returncode = 0
        stdout = header.getvalue()
        stderr = b""

    monkeypatch.setattr(audio_module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio_module.subprocess, "run", lambda cmd, **kwargs: _Empty())

    with pytest.raises(RuntimeError, match="decoded no audio"):
        audio_module._decode_with_ffmpeg(b"RIFF....WAVE", is_bytes=True)
