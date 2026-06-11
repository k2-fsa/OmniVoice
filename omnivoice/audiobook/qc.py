from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from omnivoice.audiobook.mastering import FFmpegError, Runner, ffprobe_media_info
from omnivoice.audiobook.schema import AudiobookPlan


@dataclass
class QcReport:
    gate_status: str
    duration_seconds: float = 0.0
    sample_rate_hz: Optional[int] = None
    channel_count: Optional[int] = None
    peak_dbfs: Optional[float] = None
    loudness_lufs: Optional[float] = None
    missing_segments: List[str] = field(default_factory=list)
    failed_segments: List[str] = field(default_factory=list)
    pending_segments: List[str] = field(default_factory=list)
    zero_byte_segments: List[str] = field(default_factory=list)
    unreadable_segments: List[str] = field(default_factory=list)
    required_fixes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _duration_from_info(info: Dict[str, object]) -> float:
    fmt = info.get("format")
    if isinstance(fmt, dict):
        try:
            return float(fmt.get("duration") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _sample_rate_from_info(info: Dict[str, object]) -> Optional[int]:
    streams = info.get("streams")
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") == "audio" or "sample_rate" in stream:
            try:
                return int(stream.get("sample_rate"))
            except (TypeError, ValueError):
                return None
    return None


def _channel_count_from_info(info: Dict[str, object]) -> Optional[int]:
    streams = info.get("streams")
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") == "audio" or "channels" in stream:
            try:
                return int(stream.get("channels"))
            except (TypeError, ValueError):
                return None
    return None


def _iter_metadata_tags(info: Dict[str, object]) -> List[Dict[str, object]]:
    tags: List[Dict[str, object]] = []
    fmt = info.get("format")
    if isinstance(fmt, dict) and isinstance(fmt.get("tags"), dict):
        tags.append(fmt["tags"])
    streams = info.get("streams")
    if isinstance(streams, list):
        for stream in streams:
            if isinstance(stream, dict) and isinstance(stream.get("tags"), dict):
                tags.append(stream["tags"])
    return tags


def _float_tag(info: Dict[str, object], names: set[str]) -> Optional[float]:
    lowered_names = {name.lower() for name in names}
    for tags in _iter_metadata_tags(info):
        for key, value in tags.items():
            if str(key).lower() not in lowered_names:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def inspect_audiobook_plan_audio(
    plan: AudiobookPlan,
    *,
    runner: Optional[Runner] = None,
    ffprobe_path: Optional[str] = None,
) -> QcReport:
    report = QcReport(gate_status="pass")
    sample_rates: set[int] = set()
    channel_counts: set[int] = set()
    peaks: List[float] = []
    loudness_values: List[float] = []

    for chapter in plan.chapters:
        for segment in chapter.segments:
            if segment.status == "failed":
                report.failed_segments.append(segment.id)
            elif segment.status != "generated":
                report.pending_segments.append(segment.id)

            if not segment.audio_path:
                report.missing_segments.append(segment.id)
                continue

            path = Path(segment.audio_path)
            if not path.exists():
                report.missing_segments.append(segment.id)
                continue
            if path.stat().st_size == 0:
                report.zero_byte_segments.append(segment.id)
                continue

            try:
                info = ffprobe_media_info(path, runner=runner, ffprobe_path=ffprobe_path)
            except (FFmpegError, OSError, ValueError) as exc:
                report.unreadable_segments.append(f"{segment.id}: {type(exc).__name__}: {exc}")
                continue

            report.duration_seconds += _duration_from_info(info)
            sample_rate = _sample_rate_from_info(info)
            if sample_rate:
                sample_rates.add(sample_rate)
            channel_count = _channel_count_from_info(info)
            if channel_count:
                channel_counts.add(channel_count)
            peak = _float_tag(info, {"peak_dbfs", "true_peak", "lavfi.r128.true_peak"})
            if peak is not None:
                peaks.append(peak)
            loudness = _float_tag(info, {"loudness_lufs", "integrated_loudness", "lavfi.r128.i"})
            if loudness is not None:
                loudness_values.append(loudness)

    if len(sample_rates) == 1:
        report.sample_rate_hz = next(iter(sample_rates))
        if report.sample_rate_hz != plan.qc_targets.sample_rate_hz:
            report.required_fixes.append(
                f"Normalize sample rate to {plan.qc_targets.sample_rate_hz} Hz"
            )
    elif len(sample_rates) > 1:
        report.required_fixes.append("Normalize segment sample rates before final assembly")

    if len(channel_counts) == 1:
        report.channel_count = next(iter(channel_counts))
        if report.channel_count != 1:
            report.required_fixes.append("Normalize audio to mono")
    elif len(channel_counts) > 1:
        report.required_fixes.append("Normalize segment channel counts before final assembly")

    if peaks:
        report.peak_dbfs = max(peaks)
        if report.peak_dbfs > plan.qc_targets.peak_dbfs_max:
            report.required_fixes.append(
                f"Reduce peak level to {plan.qc_targets.peak_dbfs_max:.1f} dBFS or lower"
            )
    if loudness_values:
        report.loudness_lufs = sum(loudness_values) / len(loudness_values)
        lower = plan.qc_targets.target_loudness_lufs - plan.qc_targets.allowed_loudness_tolerance
        upper = plan.qc_targets.target_loudness_lufs + plan.qc_targets.allowed_loudness_tolerance
        if not lower <= report.loudness_lufs <= upper:
            report.required_fixes.append(
                f"Normalize loudness to {plan.qc_targets.target_loudness_lufs:.1f} LUFS"
            )

    if report.failed_segments:
        report.required_fixes.append("Regenerate failed segments")
    if report.pending_segments:
        report.required_fixes.append("Generate pending segments")
    if report.missing_segments:
        report.required_fixes.append("Provide audio paths for all generated segments")
    if report.zero_byte_segments:
        report.required_fixes.append("Regenerate zero-byte audio files")
    if report.unreadable_segments:
        report.required_fixes.append("Fix unreadable audio files or ffprobe/tooling errors")
    if report.duration_seconds <= 0 and not report.required_fixes:
        report.required_fixes.append("QC could not measure positive audio duration")

    report.gate_status = "pass" if not report.required_fixes else "fail"
    return report
