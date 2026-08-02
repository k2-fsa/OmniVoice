#!/usr/bin/env python3
"""Generate and measure OmniVoice native-pause smoke matrices."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.controls import parse_pause_markers


SOURCE_REVISION = "3d2bd9d07bbe8d16c2439745b0ded450dc41e215"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
MODEL_PATH = Path(
    "/home/mustafa/.cache/huggingface/hub/"
    "models--k2-fsa--OmniVoice/snapshots/"
    "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)
ALIGNER_PATH = Path(
    "/home/mustafa/.cache/huggingface/hub/"
    "models--Systran--faster-whisper-small.en/snapshots/"
    "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
)
REFERENCE_DIR = Path(
    "/home/mustafa/Desktop/pythonprojects/gentest/data/tts/clone_sources/"
    "12_male_american_accent_performance_references"
)
SEED = 1234
SAMPLE_RATE = 24000
FRAME_RATE = 25
SAMPLES_PER_FRAME = 960


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    group: str
    original_text: str
    requested_pause_seconds: float
    reference_basename: str
    reference_text: str


PRIMARY_TEXT = (
    "The first result looked convincing. "
    "The second test revealed a hidden timing problem."
)
SECONDARY_TEXT = (
    "From a distance, the mechanism looks perfectly ordinary. "
    "But the hidden timing is what makes the whole system work."
)
PRIMARY_REF_TEXT = (
    "But through your camera you'll see like a bluish light flickering. "
    "Because the camera receives the infrared radiation and then it converts "
    "it into a color humans can see on the screen."
)
SECONDARY_REF_TEXT = (
    "To the human eye, the light on the remote will just look blank. "
    "But through your camera, you'll see like a bluish light flickering. "
    "Because the camera receives the infrared radiation, and then it converts "
    "it into a color humans can see on the screen."
)


def smoke_cases(include_secondary: bool) -> list[SmokeCase]:
    cases = [
        SmokeCase(
            "primary_baseline",
            "primary",
            PRIMARY_TEXT,
            0.0,
            "explaining_sub_11s_24k.wav",
            PRIMARY_REF_TEXT,
        )
    ]
    boundary = "The first result looked convincing."
    remainder = " The second test revealed a hidden timing problem."
    for label, seconds in (("040", 0.4), ("080", 0.8), ("120", 1.2)):
        cases.append(
            SmokeCase(
                f"primary_pause_{label}",
                "primary",
                f"{boundary}<pause:{seconds:.2f}>{remainder}",
                seconds,
                "explaining_sub_11s_24k.wav",
                PRIMARY_REF_TEXT,
            )
        )
    if include_secondary:
        cases.extend(
            [
                SmokeCase(
                    "secondary_baseline",
                    "secondary",
                    SECONDARY_TEXT,
                    0.0,
                    "explaining_24k.wav",
                    SECONDARY_REF_TEXT,
                ),
                SmokeCase(
                    "secondary_pause_080",
                    "secondary",
                    "From a distance, the mechanism looks perfectly ordinary."
                    "<pause:0.80> But the hidden timing is what makes the whole "
                    "system work.",
                    0.8,
                    "explaining_24k.wav",
                    SECONDARY_REF_TEXT,
                ),
            ]
        )
    return cases


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transcribe(aligner: WhisperModel, path: Path) -> dict:
    segments, info = aligner.transcribe(
        str(path),
        language="en",
        beam_size=5,
        word_timestamps=True,
    )
    words = []
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())
        for word in segment.words or []:
            words.append(
                {
                    "word": word.word.strip(),
                    "normalized": normalized_words(word.word)[0]
                    if normalized_words(word.word)
                    else "",
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(word.probability),
                }
            )
    text = " ".join(part for part in text_parts if part).strip()
    return {
        "text": text,
        "normalized_words": [
            word["normalized"] for word in words if word["normalized"]
        ],
        "words": words,
        "language": info.language,
        "language_probability": float(info.language_probability),
    }


def find_boundary_anchors(cleaned_text: str, offset: int, asr: dict) -> dict | None:
    expected = normalized_words(cleaned_text)
    preceding_count = len(normalized_words(cleaned_text[:offset]))
    if preceding_count < 1 or preceding_count >= len(expected):
        return None
    actual = [word["normalized"] for word in asr["words"]]
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)
    mapping = {}
    for block in matcher.get_matching_blocks():
        for delta in range(block.size):
            mapping[block.a + delta] = block.b + delta
    before_expected = preceding_count - 1
    after_expected = preceding_count
    if before_expected not in mapping or after_expected not in mapping:
        return None
    before = asr["words"][mapping[before_expected]]
    after = asr["words"][mapping[after_expected]]
    return {
        "preceding_word": before,
        "following_word": after,
        "expected_preceding": expected[before_expected],
        "expected_following": expected[after_expected],
    }


def low_energy_measurement(path: Path, anchors: dict | None) -> dict | None:
    if anchors is None:
        return None
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    start_seconds = anchors["preceding_word"]["end"]
    end_seconds = anchors["following_word"]["start"]
    start = max(0, int(start_seconds * sample_rate))
    end = min(len(audio), int(end_seconds * sample_rate))
    frame_samples = max(1, int(0.01 * sample_rate))
    longest = 0
    longest_start = 0
    current = 0
    current_start = 0
    dbfs_values = []
    for frame_start in range(start, end, frame_samples):
        frame = audio[frame_start : min(frame_start + frame_samples, end)]
        if len(frame) == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        dbfs = 20.0 * np.log10(max(rms, 1e-12))
        dbfs_values.append(float(dbfs))
        if dbfs < -50.0:
            if current == 0:
                current_start = frame_start
            current += len(frame)
            if current > longest:
                longest = current
                longest_start = current_start
        else:
            current = 0
    longest_end = longest_start + longest
    return {
        "anchor_window_seconds": [start_seconds, end_seconds],
        "asr_word_gap_seconds": max(0.0, end_seconds - start_seconds),
        "longest_low_energy_seconds": longest / sample_rate,
        "low_energy_start_seconds": longest_start / sample_rate,
        "low_energy_end_seconds": longest_end / sample_rate,
        "minimum_frame_dbfs": min(dbfs_values) if dbfs_values else None,
    }


def transient_measurement(path: Path, low_energy: dict | None) -> dict | None:
    if low_energy is None or low_energy["longest_low_energy_seconds"] <= 0:
        return None
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    delta = np.abs(np.diff(audio))
    if len(delta) == 0:
        return None
    radius = int(0.05 * sample_rate)
    values = []
    for seconds in (
        low_energy["low_energy_start_seconds"],
        low_energy["low_energy_end_seconds"],
    ):
        center = int(seconds * sample_rate)
        values.append(float(delta[max(0, center - radius) : center + radius].max()))
    global_p999 = float(np.quantile(delta, 0.999))
    boundary_max = max(values)
    limit = max(0.25, global_p999 * 5.0)
    return {
        "entry_max_abs_delta": values[0],
        "exit_max_abs_delta": values[1],
        "global_p999_abs_delta": global_p999,
        "abnormal_spike_limit": limit,
        "abnormal_spike": boundary_max > limit,
    }


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def generate_case(
    model: OmniVoice,
    prompt,
    case: SmokeCase,
    output_dir: Path,
    num_step: int,
) -> dict:
    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    config = OmniVoiceGenerationConfig(num_step=num_step)
    started = time.perf_counter()
    task = model._preprocess_all(
        text=case.original_text,
        language="English",
        voice_clone_prompt=prompt,
        preprocess_prompt=config.preprocess_prompt,
        speed=1.0,
        duration=None,
    )
    with torch.inference_mode():
        tokens = model._generate_iterative(task, config)[0]
        raw = (
            model.audio_tokenizer.decode(
                tokens.to(model.audio_tokenizer.device).unsqueeze(0)
            )
            .audio_values[0]
            .cpu()
            .numpy()
            .squeeze(0)
        )
        final = model._decode_and_post_process(
            tokens,
            task.ref_rms[0],
            config,
            preserve_internal_silence=bool(task.controlled[0]),
        )
    runtime = time.perf_counter() - started

    raw_path = output_dir / f"{case.case_id}_raw.wav"
    final_path = output_dir / f"{case.case_id}_final.wav"
    token_path = output_dir / f"{case.case_id}_tokens.pt"
    sf.write(raw_path, raw, model.sampling_rate, subtype="FLOAT")
    sf.write(final_path, final, model.sampling_rate, subtype="FLOAT")
    torch.save(tokens.cpu(), token_path)

    cleaned_text, canonical_plan = parse_pause_markers(case.original_text)
    spans = task.pause_spans[0]
    fixed_unchanged = True
    if task.target_templates[0] is not None:
        template = task.target_templates[0]
        fixed = template != model.config.audio_mask_id
        fixed_unchanged = bool(torch.equal(tokens[fixed], template[fixed]))
    return {
        "case_id": case.case_id,
        "group": case.group,
        "original_text": case.original_text,
        "cleaned_text": cleaned_text,
        "canonical_plan": asdict(canonical_plan),
        "requested_pause_seconds": case.requested_pause_seconds,
        "requested_pause_frames": sum(end - start for start, end in spans),
        "pause_token_spans": [list(span) for span in spans],
        "target_frames": task.target_lens[0],
        "raw_samples": len(raw),
        "final_samples": len(final),
        "exact_raw_codec_length": len(raw) == task.target_lens[0] * SAMPLES_PER_FRAME,
        "fixed_tokens_unchanged": fixed_unchanged,
        "seed": SEED,
        "speed": 1.0,
        "duration": None,
        "language": "English",
        "num_step": num_step,
        "runtime_seconds": runtime,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "files": {
            "raw": str(raw_path),
            "final": str(final_path),
            "tokens": str(token_path),
        },
        "hashes": {
            "raw_sha256": sha256(raw_path),
            "final_sha256": sha256(final_path),
            "tokens_sha256": sha256(token_path),
        },
    }


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(windows):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def generate_two_pass_case(
    model: OmniVoice,
    prompt,
    case: SmokeCase,
    failed_result: dict,
    baseline_result: dict,
    output_dir: Path,
    num_step: int,
) -> dict:
    baseline_tokens = torch.load(
        baseline_result["files"]["tokens"], map_location=model.device
    )
    anchors = baseline_result["boundary_anchors"]["raw"]
    if anchors is None:
        raise RuntimeError("Two-pass fallback requires baseline boundary anchors")
    midpoint_seconds = (
        anchors["preceding_word"]["end"] + anchors["following_word"]["start"]
    ) / 2.0
    insertion_frame = int(np.floor(midpoint_seconds * FRAME_RATE + 0.5))
    insertion_frame = max(1, min(baseline_tokens.shape[-1] - 1, insertion_frame))
    pause_frames = int(np.floor(case.requested_pause_seconds * FRAME_RATE + 0.5))
    pause_tokens = model._encode_pause_tokens(pause_frames, None)
    template = torch.cat(
        (
            baseline_tokens[:, :insertion_frame],
            pause_tokens,
            baseline_tokens[:, insertion_frame:],
        ),
        dim=-1,
    )

    transition_windows = merge_windows(
        [
            (max(0, insertion_frame - 5), insertion_frame),
            (
                insertion_frame + pause_frames,
                min(template.shape[-1], insertion_frame + pause_frames + 5),
            ),
        ]
    )
    for start, end in transition_windows:
        template[:, start:end] = model.config.audio_mask_id

    config = OmniVoiceGenerationConfig(num_step=num_step)
    task = model._preprocess_all(
        text=failed_result["cleaned_text"],
        language="English",
        voice_clone_prompt=prompt,
        preprocess_prompt=config.preprocess_prompt,
        speed=1.0,
        duration=None,
    )
    if task.target_lens[0] != baseline_tokens.shape[-1]:
        raise RuntimeError("Two-pass baseline token length no longer matches text plan")
    task.target_lens = [template.shape[-1]]
    task.target_templates = [template]
    task.pause_spans = [((insertion_frame, insertion_frame + pause_frames),)]
    task.controlled = [True]

    set_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        tokens = model._generate_iterative(task, config)[0]
        raw = (
            model.audio_tokenizer.decode(
                tokens.to(model.audio_tokenizer.device).unsqueeze(0)
            )
            .audio_values[0]
            .cpu()
            .numpy()
            .squeeze(0)
        )
        final = model._decode_and_post_process(
            tokens,
            task.ref_rms[0],
            config,
            preserve_internal_silence=True,
        )
    runtime = time.perf_counter() - started

    fallback_id = f"{case.case_id}_two_pass"
    raw_path = output_dir / f"{fallback_id}_raw.wav"
    final_path = output_dir / f"{fallback_id}_final.wav"
    token_path = output_dir / f"{fallback_id}_tokens.pt"
    sf.write(raw_path, raw, model.sampling_rate, subtype="FLOAT")
    sf.write(final_path, final, model.sampling_rate, subtype="FLOAT")
    torch.save(tokens.cpu(), token_path)

    fixed = template != model.config.audio_mask_id
    fixed_unchanged = bool(torch.equal(tokens[fixed], template[fixed]))
    left_preserved = bool(
        torch.equal(
            tokens[:, : max(0, insertion_frame - 5)],
            baseline_tokens[:, : max(0, insertion_frame - 5)],
        )
    )
    right_baseline_start = min(baseline_tokens.shape[-1], insertion_frame + 5)
    right_output_start = right_baseline_start + pause_frames
    right_preserved = bool(
        torch.equal(
            tokens[:, right_output_start:], baseline_tokens[:, right_baseline_start:]
        )
    )
    return {
        "case_id": fallback_id,
        "fallback_for_case_id": case.case_id,
        "group": case.group,
        "original_text": case.original_text,
        "cleaned_text": failed_result["cleaned_text"],
        "canonical_plan": failed_result["canonical_plan"],
        "requested_pause_seconds": case.requested_pause_seconds,
        "requested_pause_frames": pause_frames,
        "pause_token_spans": [[insertion_frame, insertion_frame + pause_frames]],
        "baseline_insertion_midpoint_seconds": midpoint_seconds,
        "baseline_insertion_frame": insertion_frame,
        "transition_windows": [list(window) for window in transition_windows],
        "baseline_tokens_preserved_outside_transitions": left_preserved
        and right_preserved,
        "target_frames": template.shape[-1],
        "raw_samples": len(raw),
        "final_samples": len(final),
        "exact_raw_codec_length": len(raw) == template.shape[-1] * SAMPLES_PER_FRAME,
        "fixed_tokens_unchanged": fixed_unchanged,
        "seed": SEED,
        "speed": 1.0,
        "duration": None,
        "language": "English",
        "num_step": num_step,
        "runtime_seconds": runtime,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "reference_basename": case.reference_basename,
        "reference_sha256": failed_result["reference_sha256"],
        "files": {
            "raw": str(raw_path),
            "final": str(final_path),
            "tokens": str(token_path),
        },
        "hashes": {
            "raw_sha256": sha256(raw_path),
            "final_sha256": sha256(final_path),
            "tokens_sha256": sha256(token_path),
        },
    }


def add_measurements(aligner: WhisperModel, result: dict) -> None:
    cleaned = result["cleaned_text"]
    pauses = result["canonical_plan"]["pauses"]
    offset = pauses[0]["after_char"] if pauses else cleaned.find(". ") + 1
    for kind in ("raw", "final"):
        path = Path(result["files"][kind])
        asr = transcribe(aligner, path)
        anchors = find_boundary_anchors(cleaned, offset, asr)
        low_energy = low_energy_measurement(path, anchors)
        result.setdefault("asr", {})[kind] = asr
        result.setdefault("boundary_anchors", {})[kind] = anchors
        result.setdefault("pause_measurements", {})[kind] = low_energy
        result.setdefault("transient_metrics", {})[kind] = transient_measurement(
            path, low_energy
        )


def evaluate_group(results: list[dict]) -> None:
    by_group = {}
    for result in results:
        by_group.setdefault(result["group"], []).append(result)
    for group_results in by_group.values():
        baseline = next(
            item for item in group_results if not item["canonical_plan"]["pauses"]
        )
        for result in group_results:
            for kind in ("raw", "final"):
                current_measure = result["pause_measurements"][kind]
                baseline_measure = baseline["pause_measurements"][kind]
                current_span = (
                    current_measure["longest_low_energy_seconds"]
                    if current_measure
                    else None
                )
                baseline_span = (
                    baseline_measure["longest_low_energy_seconds"]
                    if baseline_measure
                    else None
                )
                incremental = (
                    current_span - baseline_span
                    if current_span is not None and baseline_span is not None
                    else None
                )
                if result["pause_measurements"][kind] is None:
                    result["pause_measurements"][kind] = {}
                result["pause_measurements"][kind]["baseline_relative_seconds"] = (
                    incremental
                )
            baseline_words = baseline["asr"]["raw"]["normalized_words"]
            current_words = result["asr"]["raw"]["normalized_words"]
            result["word_order_matches_baseline"] = current_words == baseline_words
            result["wer_vs_baseline"] = jiwer.wer(
                " ".join(baseline_words), " ".join(current_words)
            )
            if result["canonical_plan"]["pauses"]:
                requested = result["requested_pause_seconds"]
                raw_incremental = result["pause_measurements"]["raw"][
                    "baseline_relative_seconds"
                ]
                final_incremental = result["pause_measurements"]["final"][
                    "baseline_relative_seconds"
                ]
                anchors_ok = all(
                    result["boundary_anchors"][kind] is not None
                    for kind in ("raw", "final")
                )
                transients_ok = all(
                    result["transient_metrics"][kind] is not None
                    and not result["transient_metrics"][kind]["abnormal_spike"]
                    for kind in ("raw", "final")
                )
                result["pass_checks"] = {
                    "exact_requested_codec_frames": result["requested_pause_frames"]
                    == int(np.floor(requested * FRAME_RATE + 0.5)),
                    "raw_pause_within_120ms": raw_incremental is not None
                    and abs(raw_incremental - requested) <= 0.12,
                    "final_pause_within_120ms": final_incremental is not None
                    and abs(final_incremental - requested) <= 0.12,
                    "boundary_words_present": anchors_ok,
                    "word_sequence_matches_baseline": result[
                        "word_order_matches_baseline"
                    ],
                    "no_boundary_word_omission_or_duplication": anchors_ok
                    and result["word_order_matches_baseline"],
                    "fixed_tokens_unchanged": result["fixed_tokens_unchanged"],
                    "no_abnormal_transient_spike": transients_ok,
                }
                result["automated_pass"] = all(result["pass_checks"].values())
            else:
                result["pass_checks"] = {}
                result["automated_pass"] = True


def relative_output_paths(results: list[dict], root: Path) -> None:
    for result in results:
        result["files"] = {
            key: str(Path(value).resolve().relative_to(root.resolve()))
            for key, value in result["files"].items()
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("quick", "acceptance"), default="quick")
    parser.add_argument("--num-step", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--aligner-path", type=Path, default=ALIGNER_PATH)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument(
        "--two-pass-fallback",
        action="store_true",
        help="Attempt boundary inpainting only for failed one-pass cases.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    num_step = args.num_step or (32 if args.preset == "quick" else 128)
    output_dir = args.output_dir or root / "outputs" / "pause_smoke" / args.preset
    report_path = args.report or output_dir / "pause_smoke_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    model = OmniVoice.from_pretrained(
        str(args.model_path), device_map="cuda:0", dtype=torch.float16
    )
    prompt_cache = {}
    cases = smoke_cases(include_secondary=args.preset == "acceptance")
    results = []
    for case in cases:
        if args.preset == "quick" and case.requested_pause_seconds not in (0.0, 0.8):
            continue
        if case.reference_basename not in prompt_cache:
            reference_path = args.reference_dir / case.reference_basename
            prompt_cache[case.reference_basename] = model.create_voice_clone_prompt(
                str(reference_path), case.reference_text
            )
        result = generate_case(
            model,
            prompt_cache[case.reference_basename],
            case,
            output_dir,
            num_step,
        )
        result["reference_basename"] = case.reference_basename
        result["reference_sha256"] = sha256(
            args.reference_dir / case.reference_basename
        )
        results.append(result)

    aligner = WhisperModel(
        str(args.aligner_path),
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )
    for result in results:
        add_measurements(aligner, result)
    evaluate_group(results)

    controlled = [item for item in results if item["canonical_plan"]["pauses"]]
    one_pass_automated_pass = bool(controlled) and all(
        item["automated_pass"] for item in controlled
    )
    fallback_results = []
    if args.two_pass_fallback and not one_pass_automated_pass:
        case_map = {case.case_id: case for case in cases}
        baselines = {
            item["group"]: item
            for item in results
            if not item["canonical_plan"]["pauses"]
        }
        for failed in controlled:
            if failed["automated_pass"]:
                continue
            case = case_map[failed["case_id"]]
            fallback = generate_two_pass_case(
                model,
                prompt_cache[case.reference_basename],
                case,
                failed,
                baselines[case.group],
                output_dir,
                num_step,
            )
            add_measurements(aligner, fallback)
            evaluate_group([baselines[case.group], fallback])
            fallback["pass_checks"]["baseline_tokens_preserved_outside_transitions"] = (
                fallback["baseline_tokens_preserved_outside_transitions"]
            )
            fallback["automated_pass"] = all(fallback["pass_checks"].values())
            fallback_results.append(fallback)

    fallback_by_case = {item["fallback_for_case_id"]: item for item in fallback_results}
    automated_pass = bool(controlled) and all(
        item["automated_pass"]
        or (
            item["case_id"] in fallback_by_case
            and fallback_by_case[item["case_id"]]["automated_pass"]
        )
        for item in controlled
    )
    if one_pass_automated_pass:
        decision = "one-pass accepted"
    elif automated_pass:
        decision = "two-pass required"
    elif args.two_pass_fallback:
        decision = "latent-pause approach failed"
    else:
        decision = "two-pass required"

    relative_output_paths(results, root)
    relative_output_paths(fallback_results, root)
    report = {
        "format": "omnivoice_native_pause_smoke_v1",
        "preset": args.preset,
        "source_revision": SOURCE_REVISION,
        "experiment_revision": git_revision(root),
        "model_revision": MODEL_REVISION,
        "aligner_revision": args.aligner_path.name,
        "python": subprocess.check_output(
            [str(root / ".venv/bin/python"), "--version"], text=True
        ).strip(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "num_step": num_step,
        "seed": SEED,
        "one_pass_automated_pass": one_pass_automated_pass,
        "automated_pass": automated_pass,
        "listening_status": "manual review pending",
        "decision": decision,
        "cases": results,
        "fallback_attempts": fallback_results,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "automated_pass": automated_pass,
                "decision": report["decision"],
                "cases": {
                    item["case_id"]: {
                        "automated_pass": item["automated_pass"],
                        "raw_incremental": item["pause_measurements"]["raw"].get(
                            "baseline_relative_seconds"
                        ),
                        "final_incremental": item["pause_measurements"]["final"].get(
                            "baseline_relative_seconds"
                        ),
                    }
                    for item in results
                },
                "fallback_attempts": {
                    item["case_id"]: {
                        "automated_pass": item["automated_pass"],
                        "raw_incremental": item["pause_measurements"]["raw"].get(
                            "baseline_relative_seconds"
                        ),
                        "final_incremental": item["pause_measurements"]["final"].get(
                            "baseline_relative_seconds"
                        ),
                    }
                    for item in fallback_results
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
