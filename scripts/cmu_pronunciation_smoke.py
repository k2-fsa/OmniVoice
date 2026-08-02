#!/usr/bin/env python3
"""Generate legacy narration controls with optional CMU conditioning."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from omnivoice import NarrationController, OmniVoice
from omnivoice.narration import _transcribe


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

UNEXPECTED = "AH2 N IH0 K S P EH1 K T IH0 D"
NOBODY_PRIMARY = "N OW1 B AA2 D IY2"
NOBODY_REDUCED = "N OW1 B AH0 D IY0"
GENUINE_PRIMARY = "JH EH1 N Y AH0 W AH0 N"
GENUINE_ALTERNATE = "JH EH1 N Y UW1 W AY2 N"


@dataclass(frozen=True)
class ControlledCase:
    case_id: str
    text: str
    variants: tuple[tuple[str, dict[str, str] | None], ...]


CONTROLLED_CASES = (
    ControlledCase(
        "unexpected",
        'This was <emphasis strength="strong">unexpected</emphasis>.',
        (("legacy", None), ("cmu", {"unexpected": UNEXPECTED})),
    ),
    ControlledCase(
        "nobody",
        'This was important, but <emphasis strength="strong">nobody</emphasis> '
        "understood why.",
        (
            ("legacy", None),
            ("cmu_primary", {"nobody": NOBODY_PRIMARY}),
            ("cmu_reduced", {"nobody": NOBODY_REDUCED}),
        ),
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_metrics(result) -> dict:
    control = result.report["controls"][0]
    selected = control["candidates"][control["selected_candidate"]]
    source = control["source_metrics"]
    metrics = selected["metrics"]
    return {
        "source_frames": control["source_frames"],
        "target_frames": control["target_frames"],
        "pronunciation": control["pronunciation"],
        "conditioned_text": control["conditioned_text"],
        "selected_candidate": control["selected_candidate"],
        "fixed_tokens_unchanged": selected["fixed_tokens_unchanged"],
        "source_duration_seconds": source["duration_seconds"],
        "selected_duration_seconds": metrics.get("duration_seconds"),
        "prominence_change_db": metrics.get("prominence_db", 0.0)
        - source["prominence_db"],
        "pitch_slope_change_semitones": metrics.get("pitch_slope_semitones", 0.0)
        - source["pitch_slope_semitones"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-step", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--aligner-path", type=Path, default=ALIGNER_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (
        root / "outputs/narration_controls/cmu_pronunciation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = OmniVoice.from_pretrained(
        str(args.model_path), device_map="cuda:0", dtype=torch.float16
    )
    prompt = model.create_voice_clone_prompt(str(args.reference), REFERENCE_TEXT)
    controller = NarrationController(
        model,
        aligner_path=args.aligner_path,
        local_files_only=True,
    )
    report = {
        "format": "omnivoice_cmu_pronunciation_smoke_v1",
        "experiment_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "reference_basename": args.reference.name,
        "reference_sha256": sha256(args.reference),
        "reference_switching": False,
        "num_step": args.num_step,
        "candidates": args.candidates,
        "dictionary_source": "CMUdict master and cmudict.0.7a",
        "listening_status": "manual review pending",
        "controlled_cases": [],
    }

    for case in CONTROLLED_CASES:
        case_dir = output_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_report = {"case_id": case.case_id, "variants": []}
        baseline_written = False
        for variant_id, pronunciations in case.variants:
            result = controller.generate(
                case.text,
                voice_clone_prompt=prompt,
                num_step=args.num_step,
                candidates=args.candidates,
                seed=1234,
                pronunciations=pronunciations,
            )
            if not baseline_written:
                baseline_path = case_dir / "00_baseline.wav"
                sf.write(
                    baseline_path,
                    result.baseline_audio,
                    result.sample_rate,
                    subtype="FLOAT",
                )
                case_report["baseline"] = baseline_path.name
                case_report["baseline_sha256"] = sha256(baseline_path)
                baseline_written = True
            audio_path = case_dir / f"{variant_id}.wav"
            sf.write(audio_path, result.audio, result.sample_rate, subtype="FLOAT")
            case_report["variants"].append(
                {
                    "id": variant_id,
                    "audio": audio_path.name,
                    "audio_sha256": sha256(audio_path),
                    "baseline_transcript": result.report["baseline_transcript"],
                    "final_transcript": result.report["final_transcript"],
                    "word_sequence_matches": result.report["word_sequence_matches"],
                    "metrics": selected_metrics(result),
                }
            )
            print(case.case_id, variant_id, "generated", flush=True)
        report["controlled_cases"].append(case_report)

    genuine_dir = output_dir / "genuine_native_brackets"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    genuine_variants = (
        ("plain", "genuine"),
        ("cmu_primary", f"[{GENUINE_PRIMARY}]"),
        ("cmu_alternate", f"[{GENUINE_ALTERNATE}]"),
    )
    genuine_report = {"case_id": "genuine_native_brackets", "variants": []}
    for variant_id, surface in genuine_variants:
        text = f"The final signature was {surface}, and everyone knew it."
        set_seed(1234)
        audio = model.generate(
            text=text,
            voice_clone_prompt=prompt,
            language="English",
            num_step=args.num_step,
        )[0]
        audio_path = genuine_dir / f"{variant_id}.wav"
        sf.write(audio_path, audio, model.sampling_rate, subtype="FLOAT")
        transcript = _transcribe(controller.aligner, audio, model.sampling_rate)
        genuine_report["variants"].append(
            {
                "id": variant_id,
                "generation_text": text,
                "audio": audio_path.name,
                "audio_sha256": sha256(audio_path),
                "transcript": transcript["text"],
                "normalized_words": transcript["normalized_words"],
            }
        )
        print("genuine_native_brackets", variant_id, "generated", flush=True)
    report["native_bracket_case"] = genuine_report

    report_path = output_dir / "cmu_pronunciation_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "report": str(report_path)}))


if __name__ == "__main__":
    main()
