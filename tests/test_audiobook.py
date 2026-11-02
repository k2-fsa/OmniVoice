#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../LICENSE for clarification regarding multiple authors
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

"""Tests for the audiobook helpers: text extraction/segmentation, audio
assembly and the saved-voice library. None of these load the model."""

import numpy as np
import pytest
import soundfile as sf

from omnivoice.audiobook import audio_io, text_io, voice_store


# ---------------------------------------------------------------------------
# text_io
# ---------------------------------------------------------------------------


def test_extract_text_txt(tmp_path):
    book = tmp_path / "book.txt"
    book.write_text("Hello world.\r\n\r\nSecond paragraph.", encoding="utf-8")
    text = text_io.extract_text(book)
    assert "Hello world." in text
    assert "\n\n" in text


def test_extract_text_html_skips_scripts(tmp_path):
    book = tmp_path / "book.html"
    book.write_text(
        "<html><body><p>One.</p><script>hidden()</script><p>Two.</p></body></html>",
        encoding="utf-8",
    )
    text = text_io.extract_text(book)
    assert "One." in text and "Two." in text
    assert "hidden" not in text


def test_extract_text_unsupported_type(tmp_path):
    book = tmp_path / "book.xyz"
    book.write_text("nope", encoding="utf-8")
    with pytest.raises(text_io.ExtractionError):
        text_io.extract_text(book)


def test_extract_text_empty_file(tmp_path):
    book = tmp_path / "empty.txt"
    book.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(text_io.ExtractionError):
        text_io.extract_text(book)


def test_split_segments_keeps_short_text_whole():
    text = "First sentence. Second sentence. Third sentence."
    assert text_io.split_segments(text, max_chars=80) == [text]


def test_split_segments_splits_paragraphs_and_respects_budget():
    paragraph_a = "A " * 200  # long, with no sentence punctuation
    segments = text_io.split_segments(f"{paragraph_a}\n\nShort one.", max_chars=100)
    assert len(segments) > 2
    assert all(len(segment) <= 100 for segment in segments)
    assert segments[-1] == "Short one."


def test_split_segments_raises_tiny_budget_to_floor():
    segments = text_io.split_segments("word " * 100, max_chars=10)
    assert all(len(segment) <= 80 for segment in segments)


# ---------------------------------------------------------------------------
# audio_io
# ---------------------------------------------------------------------------


def test_unique_path_does_not_overwrite(tmp_path):
    first = audio_io.unique_path(tmp_path, "book", "wav")
    first.write_bytes(b"x")
    second = audio_io.unique_path(tmp_path, "book", "wav")
    assert first.name == "book.wav"
    assert second.name == "book (2).wav"


def test_audio_sink_streams_segments(tmp_path):
    out = tmp_path / "book.wav"
    sink = audio_io.AudioSink(out, 24000, "wav", gap_seconds=0.1)
    tone = np.zeros(2400, dtype=np.float32)
    sink.write(tone)
    sink.write(tone)
    path = sink.finish()

    data, rate = sf.read(path, dtype="float32")
    assert rate == 24000
    assert sink.segments == 2
    # Gap goes between the segments, not before the first or after the last.
    assert len(data) == 2400 + 2400 + 2400


@pytest.mark.parametrize("fmt", ["mp3", "ogg"])
def test_audio_sink_lossy_formats(tmp_path, fmt):
    if audio_io._sndfile_format(fmt) is None:
        pytest.skip(f"libsndfile on this install cannot write {fmt}")
    out = tmp_path / f"book.{fmt}"
    sink = audio_io.AudioSink(out, 24000, fmt)
    sink.write(np.zeros(4800, dtype=np.float32))
    path = sink.finish()

    info = sf.info(path)
    assert info.samplerate == 24000
    assert info.frames > 0


def test_audio_sink_discard_removes_output(tmp_path):
    out = tmp_path / "book.wav"
    sink = audio_io.AudioSink(out, 24000, "wav")
    sink.write(np.zeros(10, dtype=np.float32))
    sink.discard()
    assert not out.exists()


def test_m4a_without_ffmpeg_raises_before_generating(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_io, "find_ffmpeg", lambda: None)
    with pytest.raises(audio_io.ExportError):
        audio_io.AudioSink(tmp_path / "book.m4a", 24000, "m4a")
    assert not (tmp_path / "book.m4a").exists()


# ---------------------------------------------------------------------------
# voice_store
# ---------------------------------------------------------------------------


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    target = tmp_path / "voices"
    monkeypatch.setattr(voice_store, "VOICES_DIR", target)
    return target


def test_save_design_voice_roundtrip(voices_dir):
    voice = voice_store.save_design_voice("Narrator", "male, british accent", "en")
    assert voice["kind"] == "design"
    assert voice_store.voice_names() == ["Narrator"]

    stored = voice_store.get_voice("Narrator")
    assert stored["instruct"] == "male, british accent"
    assert (voices_dir / "narrator" / "meta.json").exists()

    voice_store.delete_voice("Narrator")
    assert voice_store.voice_names() == []


def test_save_design_voice_requires_description(voices_dir):
    with pytest.raises(voice_store.VoiceError):
        voice_store.save_design_voice("Narrator", "  ", "en")


def test_save_clone_voice_copies_reference(voices_dir, tmp_path):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFFfake")
    voice = voice_store.save_clone_voice("My Voice", str(reference), " hello ", "en")

    assert voice["kind"] == "clone"
    copied = voice_store.reference_path(voice)
    assert copied is not None
    assert copied.read_bytes() == b"RIFFfake"


def test_voice_slug_collision_keeps_first_name(voices_dir):
    voice_store.save_design_voice("My Voice", "male", "en")
    with pytest.raises(voice_store.VoiceError):
        voice_store.save_design_voice("my-voice", "female", "en")


def test_delete_missing_voice_raises(voices_dir):
    with pytest.raises(voice_store.VoiceError):
        voice_store.delete_voice("nope")
