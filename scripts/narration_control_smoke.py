#!/usr/bin/env python3
"""Generate single-reference narration-control A/B samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

from omnivoice import NarrationController, OmniVoice


MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
MODEL_PATH = Path(
    "/home/mustafa/.cache/huggingface/hub/"
    "models--k2-fsa--OmniVoice/snapshots/"
    "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
)
ALIGNER_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
ALIGNER_PATH = Path(
    "/home/mustafa/.cache/huggingface/hub/"
    "models--Systran--faster-whisper-small.en/snapshots/"
    "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
)
REFERENCE_PATH = Path(
    "/home/mustafa/Desktop/pythonprojects/gentest/data/tts/clone_sources/"
    "12_male_american_accent_performance_references/"
    "explaining_sub_11s_24k.wav"
)
REFERENCE_TEXT = (
    "But through your camera you'll see like a bluish light flickering. "
    "Because the camera receives the infrared radiation and then it converts "
    "it into a color humans can see on the screen."
)
BASE_TEXT = "The mechanism looked ordinary. But the hidden timing changed everything."


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    text: str


CASES = (
    SmokeCase("baseline", BASE_TEXT),
    SmokeCase(
        "rate_slow",
        '<rate value="0.85">The mechanism looked ordinary.</rate> '
        "But the hidden timing changed everything.",
    ),
    SmokeCase(
        "rate_fast",
        '<rate value="1.20">The mechanism looked ordinary.</rate> '
        "But the hidden timing changed everything.",
    ),
    SmokeCase(
        "emphasis_moderate",
        "The mechanism looked ordinary. But the "
        '<emphasis strength="moderate">hidden timing</emphasis> '
        "changed everything.",
    ),
    SmokeCase(
        "emphasis_strong",
        "The mechanism looked ordinary. But the "
        '<emphasis strength="strong">hidden timing</emphasis> '
        "changed everything.",
    ),
    SmokeCase(
        "intonation_rising",
        "The mechanism looked ordinary. "
        '<intonation type="rising">But the hidden timing changed everything.</intonation>',
    ),
    SmokeCase(
        "intonation_falling",
        "The mechanism looked ordinary. "
        '<intonation type="falling">But the hidden timing changed everything.</intonation>',
    ),
    SmokeCase(
        "aside",
        "<aside>The mechanism looked ordinary.</aside> "
        "But the hidden timing changed everything.",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_control_metrics(result) -> dict | None:
    controls = result.report["controls"]
    if not controls:
        return None
    control = controls[0]
    selected = control["candidates"][control["selected_candidate"]]
    source = control["source_metrics"]
    metrics = selected["metrics"]
    return {
        "kind": control["control"]["kind"],
        "value": control["control"]["value"],
        "source_frames": control["source_frames"],
        "target_frames": control["target_frames"],
        "source_duration_seconds": source["duration_seconds"],
        "selected_duration_seconds": metrics.get("duration_seconds"),
        "target_duration_seconds": metrics.get("target_duration_seconds"),
        "duration_error_seconds": metrics.get("duration_error_seconds"),
        "source_prominence_db": source["prominence_db"],
        "selected_prominence_db": metrics.get("prominence_db"),
        "prominence_change_db": (
            metrics.get("prominence_db", 0.0) - source["prominence_db"]
        ),
        "source_pitch_slope_semitones": source["pitch_slope_semitones"],
        "selected_pitch_slope_semitones": metrics.get("pitch_slope_semitones"),
        "fixed_tokens_unchanged": selected["fixed_tokens_unchanged"],
        "fixed_token_fraction": selected["fixed_token_fraction"],
    }


def checks_for(metrics: dict | None, word_sequence_matches: bool) -> dict:
    checks = {"word_sequence_matches": word_sequence_matches}
    if metrics is None:
        return checks
    checks["fixed_tokens_unchanged"] = metrics["fixed_tokens_unchanged"]
    kind = metrics["kind"]
    if kind == "rate":
        checks["duration_within_150ms"] = metrics["duration_error_seconds"] <= 0.15
    elif kind == "emphasis":
        minimum_extension = 0.04 if metrics["value"] == "moderate" else 0.10
        duration_extended = metrics["selected_duration_seconds"] >= (
            metrics["source_duration_seconds"] + minimum_extension
        )
        minimum_prominence = 0.5 if metrics["value"] == "moderate" else 1.5
        prominence_increased = metrics["prominence_change_db"] >= minimum_prominence
        checks["codec_span_expanded"] = (
            metrics["target_frames"] > metrics["source_frames"]
        )
        checks["emphasis_proxy_detected"] = duration_extended or prominence_increased
    elif kind == "intonation":
        slope = metrics["selected_pitch_slope_semitones"]
        checks["pitch_direction_matches"] = (
            slope >= 0.0 if metrics["value"] == "rising" else slope <= -0.5
        )
    elif kind == "aside":
        checks["delivery_faster"] = metrics["selected_duration_seconds"] <= (
            metrics["source_duration_seconds"] - 0.05
        )
        checks["prominence_not_increased"] = metrics["prominence_change_db"] <= 0.0
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("quick", "acceptance"), default="quick")
    parser.add_argument("--num-step", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--aligner-path", type=Path, default=ALIGNER_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.case_id for case in CASES),
        help="Run only selected case; repeat for multiple cases.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    num_step = args.num_step or (32 if args.preset == "quick" else 128)
    candidates = args.candidates or (1 if args.preset == "quick" else 3)
    output_dir = args.output_dir or root / "outputs/narration_controls" / args.preset
    report_path = args.report or output_dir / "narration_control_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = OmniVoice.from_pretrained(
        str(args.model_path), device_map="cuda:0", dtype=torch.float16
    )
    prompt = model.create_voice_clone_prompt(str(args.reference), REFERENCE_TEXT)
    controller = NarrationController(
        model,
        aligner_path=args.aligner_path,
        local_files_only=True,
    )

    results = []
    baseline_written = False
    smoke_started = time.perf_counter()
    selected_cases = [
        case for case in CASES if not args.case or case.case_id in args.case
    ]
    for case in selected_cases:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        case_started = time.perf_counter()
        result = controller.generate(
            case.text,
            voice_clone_prompt=prompt,
            num_step=num_step,
            candidates=candidates,
            seed=1234,
        )
        audio_path = output_dir / f"{case.case_id}.wav"
        sf.write(audio_path, result.audio, result.sample_rate, subtype="FLOAT")
        if not baseline_written:
            baseline_path = output_dir / "shared_baseline.wav"
            sf.write(
                baseline_path,
                result.baseline_audio,
                result.sample_rate,
                subtype="FLOAT",
            )
            baseline_written = True
        metrics = selected_control_metrics(result)
        checks = checks_for(metrics, result.report["word_sequence_matches"])
        results.append(
            {
                "case_id": case.case_id,
                "source_text": case.text,
                "cleaned_text": result.plan.text,
                "controls": [control.__dict__ for control in result.plan.controls],
                "audio": str(audio_path.resolve().relative_to(root.resolve())),
                "audio_sha256": sha256(audio_path),
                "runtime_seconds": time.perf_counter() - case_started,
                "gpu_peak_memory_bytes": (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None
                ),
                "baseline_frames": result.report["baseline_frames"],
                "final_frames": result.report["final_frames"],
                "baseline_transcript": result.report["baseline_transcript"],
                "final_transcript": result.report["final_transcript"],
                "metrics": metrics,
                "checks": checks,
                "automated_pass": all(checks.values()),
                "controller_report": result.report,
            }
        )

    controlled = [item for item in results if item["controls"]]
    report = {
        "format": "omnivoice_single_reference_narration_smoke_v1",
        "experiment_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "model_revision": MODEL_REVISION,
        "aligner_revision": ALIGNER_REVISION,
        "single_reference": True,
        "reference_switching": False,
        "reference_basename": args.reference.name,
        "reference_sha256": sha256(args.reference),
        "num_step": num_step,
        "candidates": candidates,
        "runtime_seconds": time.perf_counter() - smoke_started,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "automated_pass": all(item["automated_pass"] for item in controlled),
        "listening_status": "manual review pending",
        "shared_baseline": str(
            (output_dir / "shared_baseline.wav").resolve().relative_to(root.resolve())
        ),
        "cases": results,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "automated_pass": report["automated_pass"],
                "cases": {
                    item["case_id"]: {
                        "automated_pass": item["automated_pass"],
                        "metrics": item["metrics"],
                    }
                    for item in results
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
