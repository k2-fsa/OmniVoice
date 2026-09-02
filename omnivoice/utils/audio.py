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

"""Audio I/O and processing utilities.

Provides functions for loading, resampling, silence removal,
chunking, cross-fading, and format conversion.

All public functions in this module operate on **numpy float32 arrays**
with shape ``(C, T)`` (channels-first).
"""

import io
import logging
import math
import os
import secrets
import shutil
import time
from numbers import Integral, Real
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent

logger = logging.getLogger(__name__)

_PCM16_EDGE_SCAN_CHUNK_SAMPLES = 1_048_576


def _create_atomic_wav_temp(parent: Path) -> Path:
    """Create a same-directory temporary file with normal umask semantics."""
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    for _ in range(128):
        temporary = parent / f".omnivoice-{secrets.token_hex(8)}.tmp.wav"
        try:
            descriptor = os.open(temporary, flags, 0o666)
        except FileExistsError:
            continue
        os.close(descriptor)
        return temporary
    raise FileExistsError("could not allocate a unique atomic WAV staging file")


def _replace_wav_staging_file(temporary: Path, destination: Path) -> None:
    """Commit a same-directory staged file atomically."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        def extended_path(path: Path) -> str:
            value = os.path.abspath(path)
            if value.startswith("\\\\?\\"):
                return value
            if value.startswith("\\\\"):
                return "\\\\?\\UNC\\" + value[2:]
            return "\\\\?\\" + value

        temporary_path = extended_path(temporary)
        destination_path = extended_path(destination)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move_file.restype = wintypes.BOOL
        if not move_file(
            temporary_path,
            destination_path,
            0x00000001 | 0x00000008,  # REPLACE_EXISTING | WRITE_THROUGH
        ):
            move_error = ctypes.get_last_error()
            raise ctypes.WinError(move_error)
        return

    if destination.exists():
        destination_stat = destination.stat()
        shutil.copystat(destination, temporary)
        if hasattr(os, "chown"):
            try:
                os.chown(temporary, destination_stat.st_uid, destination_stat.st_gid)
            except PermissionError:
                pass
        os.utime(
            temporary,
            ns=(destination_stat.st_atime_ns, time.time_ns()),
        )
    directory_descriptor = None
    try:
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        except OSError as exc:
            logger.warning(
                "Directory fsync is unavailable for %s: %s",
                destination.parent,
                exc,
            )
        os.replace(temporary, destination)
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError as exc:
                logger.warning(
                    "WAV commit succeeded, but directory fsync is unsupported for %s: %s",
                    destination.parent,
                    exc,
                )
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _write_wav_atomic(
    path: str | os.PathLike[str],
    audio: np.ndarray,
    sampling_rate: int,
    *,
    subtype: str | None = None,
    writer: Callable[..., object] = sf.write,
) -> None:
    """Stage a WAV beside its destination and atomically replace it."""
    destination = Path(path)
    temporary = _create_atomic_wav_temp(destination.parent)
    try:
        if subtype is None:
            writer(str(temporary), audio, sampling_rate)
        else:
            writer(str(temporary), audio, sampling_rate, subtype=subtype)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _replace_wav_staging_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_output_wav(
    path: str | os.PathLike[str],
    audio: np.ndarray,
    sampling_rate: int,
    output_mode: str,
) -> None:
    """Write one CLI waveform atomically with the mode-specific WAV subtype."""

    _write_wav_atomic(
        path,
        audio,
        sampling_rate,
        subtype="FLOAT" if output_mode == "raw_codec" else None,
        writer=sf.write,
    )


def _validate_output_waveform(audio, *, label: str) -> np.ndarray:
    """Validate one generated waveform without a long-form-sized allocation."""
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        raise RuntimeError(f"{label} must be a one-dimensional numpy array")
    if audio.size == 0:
        raise RuntimeError(f"{label} must contain at least one sample")
    if not np.issubdtype(audio.dtype, np.floating):
        raise RuntimeError(f"{label} must use a real floating-point dtype")
    chunk_size = 1_048_576
    for start in range(0, audio.size, chunk_size):
        if not np.isfinite(audio[start : start + chunk_size]).all():
            raise RuntimeError(f"{label} contains a non-finite sample")
    return audio


def _validate_nonnegative_integer(name: str, value: int) -> int:
    """Return *value* as an int or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return int(value)


def _validate_positive_integer(name: str, value: int) -> int:
    """Return *value* as a positive platform-sized integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if value > np.iinfo(np.intp).max:
        raise ValueError(f"{name} exceeds the platform sample-count limit")
    return value


def _validate_optional_nonnegative_integer(
    name: str,
    value: int | None,
) -> int | None:
    """Validate an optional non-negative integer without coercion."""
    if value is None:
        return None
    return _validate_nonnegative_integer(name, value)


def _validate_nonnegative_real(name: str, value: float) -> float:
    """Return *value* as a finite float or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _validate_optional_peak_limit(name: str, value: float | None) -> float | None:
    """Return a peak limit in ``(0, 1]`` or preserve ``None``."""
    if value is None:
        return None
    value = _validate_nonnegative_real(name, value)
    if value <= 0 or value > 1:
        raise ValueError(f"{name} must be greater than zero and at most one")
    return value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_waveform(audio_path: str):
    """Load audio from a file path, returning (data, sample_rate).

    Tries two backends in order:
    1. soundfile — covers WAV/FLAC/OGG etc., no ffmpeg needed.
    2. librosa — covers MP3/M4A etc. via audioread + ffmpeg.

    Returns:
        (data, sample_rate) where data is a numpy float32 array of
        shape (C, T).
    """
    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        return data.T, sr  # (T, C) → (C, T)
    except Exception:
        # soundfile cannot handle MP3/M4A etc., fall back to librosa.
        import librosa

        data, sr = librosa.load(audio_path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data, sr


def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    """Load a waveform from file and resample to the target rate.

    Parameters:
        audio_path: path of the audio.
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).
    """
    data, sr = load_waveform(audio_path)

    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


def load_audio_bytes(raw: bytes, sampling_rate: int) -> np.ndarray:
    """Load audio from in-memory bytes and resample.

    Parameters:
        raw: raw audio file bytes (e.g. from WebDataset).
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).
    """
    buf = io.BytesIO(raw)

    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T  # (T, C) → (C, T)
    except Exception:
        import librosa

        buf.seek(0)
        data, sr = librosa.load(buf, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]

    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


# ---------------------------------------------------------------------------
# Audio processing (all numpy in / numpy out)
# ---------------------------------------------------------------------------


def numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a numpy float32 array of shape (C, T) to a pydub AudioSegment."""
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()  # interleave channels
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def audiosegment_to_numpy(aseg: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a numpy float32 array of shape (C, T)."""
    data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
    if aseg.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, aseg.channels).T


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
    keep_mid_sil: int | None = None,
) -> np.ndarray:
    """Shorten long middle silences and trim edge silences.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        mid_sil: minimum middle-silence duration in ms (0 to skip).
        lead_sil: kept leading silence in ms.
        trail_sil: kept trailing silence in ms.
        keep_mid_sil: maximum total duration kept from each detected middle
            silence, in ms. ``None`` keeps at most ``mid_sil`` ms.

    Returns:
        Numpy array with shape (C, T').
    """
    mid_sil = _validate_nonnegative_integer("mid_sil", mid_sil)
    lead_sil = _validate_nonnegative_integer("lead_sil", lead_sil)
    trail_sil = _validate_nonnegative_integer("trail_sil", trail_sil)
    if keep_mid_sil is None:
        keep_mid_sil = mid_sil
    else:
        keep_mid_sil = _validate_nonnegative_integer("keep_mid_sil", keep_mid_sil)

    if audio.ndim != 2:
        raise ValueError("audio must have shape (channels, samples)")
    if audio.shape[-1] == 0:
        return audio

    # Pydub is used only to detect millisecond ranges. Reconstructing the
    # returned waveform from AudioSegment would quantize float input to PCM16
    # and hard-clip samples outside [-1, 1]. Slice the original float array
    # instead so silence post-processing never changes voiced samples.
    processed = audio
    detection_proxy = numpy_to_audiosegment(processed, sampling_rate)

    if mid_sil > 0:
        keep_per_side = keep_mid_sil // 2
        output_ranges = [
            [start - keep_per_side, end + keep_per_side]
            for start, end in detect_nonsilent(
                detection_proxy,
                min_silence_len=mid_sil,
                silence_thresh=-50,
                seek_step=10,
            )
        ]
        for current, following in zip(output_ranges, output_ranges[1:]):
            if following[0] < current[1]:
                midpoint = (current[1] + following[0]) // 2
                current[1] = midpoint
                following[0] = midpoint

        sample_ranges = [
            (
                max(0, int(start * sampling_rate / 1000.0)),
                min(processed.shape[-1], int(end * sampling_rate / 1000.0)),
            )
            for start, end in output_ranges
        ]
        chunks = [
            processed[..., start:end] for start, end in sample_ranges if end > start
        ]
        processed = np.concatenate(chunks, axis=-1) if chunks else processed[..., :0]

    if processed.shape[-1] == 0:
        return processed

    edge_proxy = numpy_to_audiosegment(processed, sampling_rate)
    leading_silence = detect_leading_silence(
        edge_proxy,
        silence_threshold=-50,
    )
    trailing_silence = detect_leading_silence(
        edge_proxy.reverse(),
        silence_threshold=-50,
    )
    start_ms = max(0, leading_silence - lead_sil)
    end_ms = min(len(edge_proxy), len(edge_proxy) - trailing_silence + trail_sil)
    start_sample = max(0, int(start_ms * sampling_rate / 1000.0))
    end_sample = min(processed.shape[-1], int(end_ms * sampling_rate / 1000.0))
    return processed[..., start_sample:end_sample]


def limit_audio_peak(
    audio: np.ndarray,
    peak_limit: float | None,
) -> np.ndarray:
    """Scale a waveform only when its absolute peak exceeds *peak_limit*.

    This operation preserves duration and relative dynamics. ``None`` keeps
    the waveform unchanged.
    """
    peak_limit = _validate_optional_peak_limit("peak_limit", peak_limit)
    if peak_limit is None or audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= peak_limit or peak <= 1e-12:
        return audio
    return audio * (peak_limit / peak)


def _find_pcm16_active_edge(
    audio: np.ndarray,
    *,
    from_end: bool,
) -> int | None:
    """Find one active PCM16-proxy edge with bounded temporary memory.

    Edge alignment only needs the first or last active frame. Scanning bounded
    chunks avoids materializing a full PCM16 proxy, a full boolean mask, and
    one platform-sized integer for every active frame in long-form audio.
    """
    sample_count = audio.shape[-1]
    if from_end:
        chunk_end = sample_count
        while chunk_end > 0:
            chunk_start = max(
                0,
                chunk_end - _PCM16_EDGE_SCAN_CHUNK_SAMPLES,
            )
            chunk = audio[..., chunk_start:chunk_end]
            detection_proxy = (chunk * 32768.0).clip(-32768, 32767).astype(np.int16)
            active_samples = np.any(detection_proxy != 0, axis=0)
            if bool(np.any(active_samples)):
                return chunk_end - 1 - int(np.argmax(active_samples[::-1]))
            chunk_end = chunk_start
        return None

    chunk_start = 0
    while chunk_start < sample_count:
        chunk_end = min(
            sample_count,
            chunk_start + _PCM16_EDGE_SCAN_CHUNK_SAMPLES,
        )
        chunk = audio[..., chunk_start:chunk_end]
        detection_proxy = (chunk * 32768.0).clip(-32768, 32767).astype(np.int16)
        active_samples = np.any(detection_proxy != 0, axis=0)
        if bool(np.any(active_samples)):
            return chunk_start + int(np.argmax(active_samples))
        chunk_start = chunk_end
    return None


def _find_exact_active_edge(
    audio: np.ndarray,
    *,
    from_end: bool,
) -> int | None:
    """Find an exactly non-zero edge frame with bounded temporary memory."""

    sample_count = audio.shape[-1]
    if from_end:
        chunk_end = sample_count
        while chunk_end > 0:
            chunk_start = max(0, chunk_end - _PCM16_EDGE_SCAN_CHUNK_SAMPLES)
            active_samples = np.any(
                audio[..., chunk_start:chunk_end] != 0,
                axis=0,
            )
            if bool(np.any(active_samples)):
                return chunk_end - 1 - int(np.argmax(active_samples[::-1]))
            chunk_end = chunk_start
        return None

    chunk_start = 0
    while chunk_start < sample_count:
        chunk_end = min(
            sample_count,
            chunk_start + _PCM16_EDGE_SCAN_CHUNK_SAMPLES,
        )
        active_samples = np.any(
            audio[..., chunk_start:chunk_end] != 0,
            axis=0,
        )
        if bool(np.any(active_samples)):
            return chunk_start + int(np.argmax(active_samples))
        chunk_start = chunk_end
    return None


def match_edge_silence(
    audio: np.ndarray,
    sampling_rate: int,
    target_lead_silence_ms: int | None = None,
    target_trail_silence_ms: int | None = None,
) -> np.ndarray:
    """Match either output edge to an exact amount of digital silence.

    ``None`` preserves the corresponding edge exactly as received. A numeric
    target removes zero-valued PCM16 proxy samples on that edge and replaces
    them with the requested number of zero-valued samples. This sample-accurate
    rule avoids millisecond detector rounding and preserves PCM-representable
    low-level attacks, including audio below a conventional dBFS silence
    threshold. Voiced samples are always sliced from the original
    floating-point waveform.

    The target values describe alignment supplied by the caller; this function
    does not infer timing from reference audio. Callers that must fit a fixed
    window should subtract intentional edge silence from the pre-synthesis
    audio-token budget.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        target_lead_silence_ms: exact leading-silence anchor in milliseconds
            before final container fitting, or ``None`` to keep the existing
            leading edge.
        target_trail_silence_ms: exact trailing-silence anchor in milliseconds
            before final container fitting, or ``None`` to keep the existing
            trailing edge.

    Returns:
        Numpy array with shape (C, T').
    """
    target_lead_silence_ms = _validate_optional_nonnegative_integer(
        "target_lead_silence_ms", target_lead_silence_ms
    )
    target_trail_silence_ms = _validate_optional_nonnegative_integer(
        "target_trail_silence_ms", target_trail_silence_ms
    )

    if audio.ndim != 2:
        raise ValueError("audio must have shape (channels, samples)")
    if audio.shape[-1] == 0:
        return audio
    if target_lead_silence_ms is None and target_trail_silence_ms is None:
        return audio

    start_sample = 0
    if target_lead_silence_ms is not None:
        first_active_sample = _find_pcm16_active_edge(audio, from_end=False)
        # A silent model output is not turned into an apparently valid timing pad.
        if first_active_sample is None:
            return audio[..., :0]
        start_sample = first_active_sample
    end_sample = audio.shape[-1]
    if target_trail_silence_ms is not None:
        last_active_sample = _find_pcm16_active_edge(audio, from_end=True)
        if last_active_sample is None:
            return audio[..., :0]
        end_sample = max(start_sample, last_active_sample + 1)

    parts: list[np.ndarray] = []
    if target_lead_silence_ms is not None:
        lead_samples = round(target_lead_silence_ms * sampling_rate / 1000.0)
        if lead_samples:
            parts.append(np.zeros((audio.shape[0], lead_samples), dtype=audio.dtype))

    parts.append(audio[..., start_sample:end_sample])

    if target_trail_silence_ms is not None:
        trail_samples = round(target_trail_silence_ms * sampling_rate / 1000.0)
        if trail_samples:
            parts.append(np.zeros((audio.shape[0], trail_samples), dtype=audio.dtype))

    return np.concatenate(parts, axis=-1)


def fit_audio_to_target_samples(
    audio: np.ndarray,
    target_samples: int,
    *,
    protect_leading_edge: bool = False,
    protect_trailing_edge: bool = False,
    target_name: str = "target_samples",
) -> tuple[np.ndarray, int, int, int, int]:
    """Fit a waveform to an exact sample-frame count without cutting activity.

    Underflow is always filled by appending exact floating-point zeros after
    the complete source waveform. This preserves the onset and every existing
    edge sample byte-for-byte, including a protected trailing anchor.
    Protected edges are never shortened.
    Overflow removes only contiguous frames that are exactly zero in every
    channel, preferring the trailing edge. Protected edges represent explicit
    pre-framing edge-silence anchors: their authenticated samples are never
    shortened or rewritten, although separately reported outer-container zeros
    may be appended after a protected trailing anchor to satisfy underflow.

    Returns:
        ``(audio, padded_lead, padded_trail, trimmed_lead, trimmed_trail)``.
    """
    target_samples = _validate_positive_integer(target_name, target_samples)
    if not isinstance(protect_leading_edge, bool):
        raise TypeError("protect_leading_edge must be a bool")
    if not isinstance(protect_trailing_edge, bool):
        raise TypeError("protect_trailing_edge must be a bool")
    if not isinstance(audio, np.ndarray) or audio.ndim != 2:
        raise ValueError("audio must be a numpy array with shape (channels, samples)")

    source_samples = audio.shape[-1]
    protect_any_edge = protect_leading_edge or protect_trailing_edge
    if source_samples == target_samples:
        if protect_any_edge and not bool(np.any(audio)):
            raise ValueError(
                f"{target_name} cannot preserve an exact edge target on a waveform "
                "without an active sample"
            )
        return audio, 0, 0, 0, 0

    if source_samples < target_samples:
        missing = target_samples - source_samples
        if protect_any_edge and not bool(np.any(audio)):
            raise ValueError(
                f"{target_name} cannot preserve an exact edge target on a waveform "
                "without an active sample"
            )
        fitted = np.zeros((audio.shape[0], target_samples), dtype=audio.dtype)
        fitted[..., :source_samples] = audio
        return fitted, 0, missing, 0, 0

    excess = source_samples - target_samples
    if not bool(np.any(audio)):
        if protect_any_edge:
            raise ValueError(
                f"{target_name} cannot preserve an exact edge target on a waveform "
                "without an active sample"
            )
        return audio[..., :target_samples], 0, 0, 0, excess

    # Scan bounded chunks rather than materializing one boolean per frame for
    # the complete long-form waveform.
    first_active = _find_exact_active_edge(audio, from_end=False)
    last_active = _find_exact_active_edge(audio, from_end=True)
    if first_active is None or last_active is None:
        raise RuntimeError("non-silent waveform lost its active edge during framing")
    leading_zeros = first_active
    trailing_zeros = source_samples - 1 - last_active
    removable_trailing = 0 if protect_trailing_edge else trailing_zeros
    removable_leading = 0 if protect_leading_edge else leading_zeros
    removable = removable_trailing + removable_leading
    if excess > removable:
        active_cut = excess - removable
        raise ValueError(
            f"{target_name} targets {target_samples} samples, but the post-pipeline "
            f"waveform has {source_samples} samples and only {removable} removable "
            f"digital-silence edge samples; trimming {active_cut} active samples "
            "is forbidden"
        )

    trimmed_trailing = min(excess, removable_trailing)
    trimmed_leading = excess - trimmed_trailing
    end = source_samples - trimmed_trailing if trimmed_trailing else source_samples
    fitted = audio[..., trimmed_leading:end]
    if fitted.shape[-1] != target_samples:
        raise RuntimeError("exact-duration framing produced an unexpected sample count")
    return fitted, 0, 0, trimmed_leading, trimmed_trailing


def remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int = 100,
    trail_sil: int = 300,
    silence_threshold: float = -50,
) -> AudioSegment:
    """Remove edge silences, keeping *lead_sil* / *trail_sil* ms."""
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def fade_and_pad_audio(
    audio: np.ndarray,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Apply fade-in/out and pad with silence to prevent clicks.

    Args:
        audio: numpy array of shape (C, T).
        pad_duration: silence padding duration per side (seconds).
        fade_duration: fade curve duration (seconds).
        sample_rate: audio sampling rate.

    Returns:
        Processed numpy array of shape (C, T_new).
    """
    pad_duration = _validate_nonnegative_real("pad_duration", pad_duration)
    fade_duration = _validate_nonnegative_real("fade_duration", fade_duration)

    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)

    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            fade_in = np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., :k] *= fade_in

            fade_out = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= fade_out

    if pad_samples > 0:
        silence = np.zeros(
            (processed.shape[0], pad_samples),
            dtype=processed.dtype,
        )
        processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


def trim_long_audio(
    audio: np.ndarray,
    sampling_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 3.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    """Trim audio to <= *max_duration* by splitting at the largest silence gap.

    Only trims when the audio exceeds *trim_threshold* seconds.

    Args:
        audio: numpy array of shape (C, T).
        sampling_rate: audio sampling rate.
        max_duration: maximum duration in seconds.
        min_duration: minimum duration in seconds.
        trim_threshold: only trim if audio is longer than this (seconds).

    Returns:
        Trimmed numpy array.
    """
    duration = audio.shape[-1] / sampling_rate
    if duration <= trim_threshold:
        return audio

    seg = numpy_to_audiosegment(audio, sampling_rate)
    nonsilent = detect_nonsilent(
        seg, min_silence_len=100, silence_thresh=-40, seek_step=10
    )
    if not nonsilent:
        return audio

    max_ms = int(max_duration * 1000)
    min_ms = int(min_duration * 1000)

    best_split = 0
    for start, end in nonsilent:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < min_ms:
        best_split = min(max_ms, len(seg))

    trimmed = seg[:best_split]
    return audiosegment_to_numpy(trimmed)


def cross_fade_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> np.ndarray:
    """Concatenate audio chunks with silence gaps and cross-fade at boundaries.

    Args:
        chunks: list of numpy arrays, each (C, T).
        sample_rate: audio sample rate.
        silence_duration: total silence gap duration in seconds.

    Returns:
        Merged numpy array (C, T_total).
    """
    if len(chunks) == 1:
        return chunks[0]

    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n
    merged = chunks[0].copy()

    for chunk in chunks[1:]:
        parts = [merged]

        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            w_out = np.linspace(1, 0, fout_n, dtype=np.float32)[np.newaxis, :]
            parts[-1][..., -fout_n:] *= w_out

        parts.append(np.zeros((chunks[0].shape[0], silence_n), dtype=np.float32))

        fade_in = chunk.copy()
        fin_n = min(fade_n, fade_in.shape[-1])
        if fin_n > 0:
            w_in = np.linspace(0, 1, fin_n, dtype=np.float32)[np.newaxis, :]
            fade_in[..., :fin_n] *= w_in

        parts.append(fade_in)
        merged = np.concatenate(parts, axis=-1)

    return merged
