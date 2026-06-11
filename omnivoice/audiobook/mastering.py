from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


Runner = Callable[..., subprocess.CompletedProcess]


class FFmpegError(RuntimeError):
    pass


@dataclass
class MasteringOptions:
    target_lufs: float = -20.0
    true_peak: float = -3.0
    loudness_range: float = 11.0
    tempo: float = 1.0
    trim_silence: bool = False
    dynamic_normalize: bool = False
    compressor: bool = False
    limiter: bool = True
    output_format: str = "wav"
    sample_rate_hz: int = 44100
    channels: int = 1
    overwrite: bool = False


@dataclass
class ConcatOptions:
    normalize_stream: bool = True
    sample_rate_hz: int = 44100
    channels: int = 1
    overwrite: bool = False


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"Required audio tool not found on PATH: {name}")
    return path


def _run(command: Sequence[str], runner: Optional[Runner] = None) -> subprocess.CompletedProcess:
    run = runner or subprocess.run
    completed = run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FFmpegError(completed.stderr.strip() or f"Command failed: {' '.join(command)}")
    return completed


def _concat_list(paths: Sequence[Path], list_path: Path) -> None:
    lines: List[str] = []
    for path in paths:
        if not path.exists():
            raise FFmpegError(f"Audio segment not found: {path}")
        escaped = str(path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_output_is_not_source(output: Path, sources: Sequence[Path]) -> None:
    output_resolved = output.resolve()
    for source in sources:
        if output_resolved == Path(source).resolve():
            raise FFmpegError(f"Output path must not overwrite source audio: {output}")


def concat_audio_files(
    inputs: Sequence[Path],
    output: Path,
    *,
    options: Optional[ConcatOptions] = None,
    runner: Optional[Runner] = None,
    ffmpeg_path: Optional[str] = None,
) -> Path:
    if not inputs:
        raise FFmpegError("No audio inputs supplied for concatenation")
    inputs = [Path(item) for item in inputs]
    ffmpeg = ffmpeg_path or require_binary("ffmpeg")
    options = options or ConcatOptions()
    _ensure_output_is_not_source(output, inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.with_suffix(".concat.txt")
    _concat_list(inputs, list_path)
    try:
        command = [
            ffmpeg,
            "-y" if options.overwrite else "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if options.normalize_stream:
            command.extend(
                [
                    "-ar",
                    str(options.sample_rate_hz),
                    "-ac",
                    str(options.channels),
                    "-c:a",
                    "pcm_s16le",
                ]
            )
        else:
            command.extend(["-c", "copy"])
        command.append(str(output))
        _run(command, runner=runner)
    finally:
        if list_path.exists():
            list_path.unlink()
    return output


def build_audio_filter(options: MasteringOptions) -> str:
    filters: List[str] = []
    if options.trim_silence:
        filters.append("silenceremove=start_periods=1:start_duration=0.2:start_threshold=-50dB")
    if abs(options.tempo - 1.0) > 0.001:
        if not 0.5 <= options.tempo <= 2.0:
            raise ValueError("tempo must be between 0.5 and 2.0 for ffmpeg atempo")
        filters.append(f"atempo={options.tempo:.4g}")
    if options.dynamic_normalize:
        filters.append("dynaudnorm=f=150:g=15")
    if options.compressor:
        filters.append("acompressor=threshold=-18dB:ratio=2:attack=20:release=250")
    filters.append(
        "loudnorm="
        f"I={options.target_lufs:.2f}:"
        f"TP={options.true_peak:.2f}:"
        f"LRA={options.loudness_range:.2f}"
    )
    if options.limiter:
        filters.append(f"alimiter=limit={10 ** (options.true_peak / 20):.6f}")
    return ",".join(filters)


def remaster_audio(
    input_path: Path,
    output_path: Path,
    options: Optional[MasteringOptions] = None,
    *,
    runner: Optional[Runner] = None,
    ffmpeg_path: Optional[str] = None,
) -> Path:
    if not input_path.exists():
        raise FFmpegError(f"Input audio not found: {input_path}")
    options = options or MasteringOptions()
    ffmpeg = ffmpeg_path or require_binary("ffmpeg")
    _ensure_output_is_not_source(output_path, [input_path])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y" if options.overwrite else "-n",
            "-i",
            str(input_path),
            "-af",
            build_audio_filter(options),
            "-ar",
            str(options.sample_rate_hz),
            "-ac",
            str(options.channels),
            str(output_path),
        ],
        runner=runner,
    )
    return output_path


def ffprobe_media_info(
    input_path: Path,
    *,
    runner: Optional[Runner] = None,
    ffprobe_path: Optional[str] = None,
) -> Dict[str, object]:
    if not input_path.exists():
        raise FFmpegError(f"Input audio not found: {input_path}")
    ffprobe = ffprobe_path or require_binary("ffprobe")
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(input_path),
        ],
        runner=runner,
    )
    return json.loads(completed.stdout or "{}")
