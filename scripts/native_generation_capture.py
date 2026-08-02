#!/usr/bin/env python3
"""Capture OmniVoice generation traces without ASR or external alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig


MODEL_PATH = Path(
    "/home/mustafa/.cache/huggingface/hub/models--k2-fsa--OmniVoice/"
    "snapshots/c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)
REFERENCE_PATH = Path(
    "/home/mustafa/Desktop/pythonprojects/gentest/data/tts/clone_sources/"
    "12_male_american_accent_performance_references/explaining_24k.wav"
)
REFERENCE_TEXT = (
    "To the human eye, the light on the remote will just look blank. "
    "But through your camera, you'll see like a bluish light flickering. "
    "Because the camera receives the infrared radiation, and then it converts "
    "it into a color humans can see on the screen."
)
DEFAULT_TEXT = (
    "Davey's debt begins with a visible balance. Soon, though, Tony is no longer "
    "collecting cash. He is collecting the business behind it, and the account "
    "closes only after the store is exhausted."
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    weighted = np.where(valid, values * weights[:, None], 0.0)
    denominator = np.where(valid, weights[:, None], 0.0).sum(axis=0)
    return np.divide(
        weighted.sum(axis=0),
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float32),
        where=denominator > 0,
    )


def frame_audio(audio: np.ndarray, frame_count: int, samples_per_frame: int) -> np.ndarray:
    wanted = frame_count * samples_per_frame
    if audio.size < wanted:
        audio = np.pad(audio, (0, wanted - audio.size))
    return audio[:wanted].reshape(frame_count, samples_per_frame)


def pitch_track(audio: np.ndarray, sample_rate: int, frame_count: int) -> np.ndarray:
    waveform = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate,
            frame_time=1.0 / 25.0,
            win_length=5,
            freq_low=70,
            freq_high=350,
        ).squeeze(0).cpu().numpy()
    except RuntimeError:
        return np.zeros(frame_count, dtype=np.float32)
    if pitch.size == 0:
        return np.zeros(frame_count, dtype=np.float32)
    source = np.linspace(0.0, 1.0, pitch.size)
    target = np.linspace(0.0, 1.0, frame_count)
    return np.interp(target, source, pitch).astype(np.float32)


def estimated_words(text: str, frame_count: int, estimator) -> list[dict]:
    cumulative = [0.0]
    for char in text:
        cumulative.append(cumulative[-1] + estimator._get_char_weight(char))
    total = cumulative[-1] or 1.0
    words = []
    for index, match in enumerate(WORD_RE.finditer(text)):
        start_frame = cumulative[match.start()] / total * frame_count
        end_frame = cumulative[match.end()] / total * frame_count
        words.append(
            {
                "index": index,
                "text": match.group(),
                "start_frame": round(start_frame, 3),
                "end_frame": round(end_frame, 3),
                "start_seconds": round(start_frame / 25.0, 4),
                "end_seconds": round(end_frame / 25.0, 4),
                "source": "duration_estimator",
            }
        )
    return words


def punctuation_boundaries(text: str, frame_count: int, estimator) -> list[dict]:
    cumulative = [0.0]
    for char in text:
        cumulative.append(cumulative[-1] + estimator._get_char_weight(char))
    total = cumulative[-1] or 1.0
    boundaries = []
    for index, char in enumerate(text):
        if char not in ".,;:!?":
            continue
        frame = cumulative[index + 1] / total * frame_count
        boundaries.append(
            {
                "character": char,
                "after_char": index + 1,
                "frame": round(frame, 3),
                "seconds": round(frame / 25.0, 4),
            }
        )
    return boundaries


def contiguous_regions(mask: np.ndarray, min_frames: int = 2) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= min_frames
    ]


def pause_candidates(
    rms_db: np.ndarray,
    silence_similarity: np.ndarray,
    punctuation: list[dict],
) -> tuple[float, list[dict]]:
    median_rms = float(np.median(rms_db))
    threshold = min(-40.0, median_rms - 15.0)
    quiet = rms_db <= threshold
    silence_like = silence_similarity >= 0.375
    mask = quiet | (silence_like & (rms_db <= median_rms - 10.0))
    regions = []
    for start, end in contiguous_regions(mask, min_frames=2):
        center = (start + end) / 2.0
        max_similarity = float(silence_similarity[start:end].max())
        nearest = min(
            punctuation,
            key=lambda item: abs(float(item["frame"]) - center),
            default=None,
        )
        regions.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_seconds": round(start / 25.0, 4),
                "end_seconds": round(end / 25.0, 4),
                "duration_seconds": round((end - start) / 25.0, 4),
                "mean_rms_db": round(float(rms_db[start:end].mean()), 3),
                "max_silence_token_similarity": round(max_similarity, 4),
                "confidence": (
                    "codec_confirmed" if max_similarity >= 0.5 else "energy_only"
                ),
                "edge": start == 0 or end == rms_db.size,
                "nearest_estimated_punctuation": nearest,
                "evidence": "decoded_energy_or_codec_silence_similarity",
            }
        )
    return threshold, regions


def round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    return np.round(values.astype(np.float64), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-step", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    model = OmniVoice.from_pretrained(
        str(args.model_path), device_map="cuda:0", dtype=torch.float16
    )
    prompt = model.create_voice_clone_prompt(str(args.reference), REFERENCE_TEXT)
    config = OmniVoiceGenerationConfig(num_step=args.num_step)
    task = model._preprocess_all(
        text=args.text,
        language="English",
        voice_clone_prompt=prompt,
        preprocess_prompt=config.preprocess_prompt,
        speed=1.0,
        duration=None,
    )
    trace: list[dict] = []
    generation_started = time.perf_counter()
    with torch.inference_mode():
        tokens = model._generate_iterative(task, config, trace=trace)[0]
        raw = (
            model.audio_tokenizer.decode(
                tokens.to(model.audio_tokenizer.device).unsqueeze(0)
            )
            .audio_values[0]
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
        )
        final = model._decode_and_post_process(
            tokens,
            task.ref_rms[0],
            config,
            preserve_internal_silence=False,
        )
        silence_reference = model._encode_pause_tokens(25, None).detach().cpu()
    generation_seconds = time.perf_counter() - generation_started

    raw_path = args.output_dir / "raw.wav"
    final_path = args.output_dir / "final.wav"
    trace_path = args.output_dir / "trace.npz"
    sf.write(raw_path, raw, model.sampling_rate, subtype="PCM_16")
    sf.write(final_path, final, model.sampling_rate, subtype="PCM_16")

    item = trace[0]
    token_array = tokens.detach().cpu().numpy()
    frame_count = token_array.shape[-1]
    codebook_weights = np.asarray(
        model.config.audio_codebook_weights, dtype=np.float32
    )
    codebook_weights /= codebook_weights.sum()
    silence_array = silence_reference.numpy()
    matches = token_array[:, :, None] == silence_array[:, None, :]
    silence_similarity = (matches * codebook_weights[:, None, None]).sum(axis=0).max(axis=1)

    samples_per_frame = model.sampling_rate // model.audio_tokenizer.config.frame_rate
    frames = frame_audio(raw, frame_count, samples_per_frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = np.max(np.abs(frames), axis=1)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    peak_db = 20.0 * np.log10(np.maximum(peak, 1e-7))
    pitch_hz = pitch_track(raw, model.sampling_rate, frame_count)

    words = estimated_words(args.text, frame_count, model.duration_estimator)
    punctuation = punctuation_boundaries(args.text, frame_count, model.duration_estimator)
    threshold, pauses = pause_candidates(rms_db, silence_similarity, punctuation)

    trace_arrays = {
        key: item[key].numpy()
        for key in ("unmask_step", "token_logprob", "entropy", "cfg_delta", "fixed", "pause")
    }
    unmask = finite_weighted_mean(trace_arrays["unmask_step"].astype(np.float32), codebook_weights)
    logprob = finite_weighted_mean(trace_arrays["token_logprob"], codebook_weights)
    entropy = finite_weighted_mean(trace_arrays["entropy"], codebook_weights)
    cfg_delta = finite_weighted_mean(trace_arrays["cfg_delta"], codebook_weights)
    strong_internal = [
        region
        for region in pauses
        if region["confidence"] == "codec_confirmed" and not region["edge"]
    ]
    strong_mask = np.zeros(frame_count, dtype=bool)
    for region in strong_internal:
        strong_mask[region["start_frame"] : region["end_frame"]] = True

    def signal_comparison(values: np.ndarray) -> dict[str, float] | None:
        if not strong_mask.any() or strong_mask.all():
            return None
        return {
            "pause_mean": round(float(values[strong_mask].mean()), 4),
            "other_mean": round(float(values[~strong_mask].mean()), 4),
        }

    np.savez_compressed(
        trace_path,
        tokens=token_array.astype(np.uint16),
        silence_reference=silence_array.astype(np.uint16),
        **trace_arrays,
    )
    report = {
        "format": "omnivoice_native_generation_capture_v1",
        "text": args.text,
        "seed": args.seed,
        "num_step": args.num_step,
        "model_revision": MODEL_REVISION,
        "reference": {
            "basename": args.reference.name,
            "sha256": sha256(args.reference),
        },
        "provenance": {
            "whisper_or_asr_used": False,
            "external_word_aligner_used": False,
            "word_positions": "OmniVoice RuleDurationEstimator weight projection",
            "pause_metadata": "exact GenerationTask pause spans; natural pauses are not labeled",
            "pause_inference": "codec-token silence similarity plus decoded 40 ms RMS",
            "pitch_measurement": "torchaudio pitch estimate from decoded model audio",
        },
        "audio": {
            "sample_rate": model.sampling_rate,
            "frame_rate": model.audio_tokenizer.config.frame_rate,
            "samples_per_frame": samples_per_frame,
            "raw_file": raw_path.name,
            "final_file": final_path.name,
            "raw_sha256": sha256(raw_path),
            "final_sha256": sha256(final_path),
            "raw_duration_seconds": round(raw.size / model.sampling_rate, 4),
            "final_duration_seconds": round(final.size / model.sampling_rate, 4),
        },
        "generation": {
            "target_frames": int(task.target_lens[0]),
            "generated_frames": frame_count,
            "codebooks": int(token_array.shape[0]),
            "fixed_token_count": int(trace_arrays["fixed"].sum()),
            "explicit_pause_token_count": int(trace_arrays["pause"].sum()),
            "runtime_seconds": round(generation_seconds, 3),
            "total_runtime_seconds": round(time.perf_counter() - started, 3),
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        },
        "word_estimates": words,
        "punctuation_estimates": punctuation,
        "pause_detection": {
            "rms_threshold_db": round(threshold, 3),
            "minimum_frames": 2,
            "candidates": pauses,
            "accuracy_status": "time-local evidence only; word boundary accuracy unverified without alignment",
        },
        "native_findings": {
            "codec_confirmed_internal_pause_candidates": strong_internal,
            "energy_only_internal_candidate_count": sum(
                region["confidence"] == "energy_only" and not region["edge"]
                for region in pauses
            ),
            "edge_region_count": sum(region["edge"] for region in pauses),
            "pause_signal_comparison": {
                "rms_db": signal_comparison(rms_db),
                "silence_token_similarity": signal_comparison(silence_similarity),
                "entropy": signal_comparison(entropy),
                "cfg_delta": signal_comparison(cfg_delta),
                "token_logprob": signal_comparison(logprob),
            },
            "interpretation": (
                "Codec-confirmed regions are stronger native-only pause evidence. "
                "Energy-only regions can be consonant closures or brief articulation gaps."
            ),
        },
        "frames": {
            "rms_db": round_list(rms_db, 3),
            "peak_db": round_list(peak_db, 3),
            "pitch_hz": round_list(pitch_hz, 2),
            "silence_token_similarity": round_list(silence_similarity, 4),
            "unmask_step": round_list(unmask, 3),
            "token_logprob": round_list(logprob, 4),
            "entropy": round_list(entropy, 4),
            "cfg_delta": round_list(cfg_delta, 4),
        },
        "files": {
            "trace": trace_path.name,
        },
    }
    report_path = args.output_dir / "capture.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(report_path), "audio": str(final_path)}))


if __name__ == "__main__":
    main()
