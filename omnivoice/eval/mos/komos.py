#!/usr/bin/env python3
# Copyright 2026 Xiaomi Corp.
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

"""Evaluate Korean generated speech with the KOMOS MOS predictor."""

import argparse
import logging
from pathlib import Path
from statistics import fmean
from typing import Protocol

from omnivoice.utils.data_utils import read_test_list

MODEL_ID = "upskyy/komos"
MODEL_REVISION = "c558983b9777d4f01aa61003b151dea37db0cc2d"


class ScoringModel(Protocol):
    def score(self, audio_path: str | Path) -> float:
        """Return the calibrated MOS for an audio file."""
        ...


def get_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Korean KOMOS evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate KOMOS for the Korean subset of a TTS test list."
    )
    parser.add_argument(
        "--wav-path",
        type=Path,
        required=True,
        help="Directory containing generated speech files.",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        required=True,
        help="JSONL test list with id and language_id fields.",
    )
    parser.add_argument(
        "--extension",
        default="wav",
        help="Generated audio extension. Default: wav",
    )
    parser.add_argument(
        "--decode-path",
        type=Path,
        default=None,
        help="Optional output file for per-file and average scores.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device for KOMOS. Default: cuda",
    )
    return parser


def select_korean_wavs(
    samples: list[dict], wav_path: Path, extension: str
) -> list[Path]:
    """Map Korean test-list rows to generated audio paths."""
    suffix = extension.removeprefix(".")
    selected = [
        wav_path / f"{sample['id']}.{suffix}"
        for sample in samples
        if str(sample.get("language_id") or "").lower().split("-", 1)[0] == "ko"
    ]
    if not selected:
        raise ValueError("test list contains no Korean samples (language_id=ko)")
    return selected


def score_korean_wavs(
    wav_files: list[Path], model: ScoringModel
) -> tuple[list[tuple[Path, float]], list[tuple[Path, str]]]:
    """Score existing files while retaining actionable per-file failures."""
    results = []
    errors = []
    for wav_file in wav_files:
        if not wav_file.is_file():
            errors.append((wav_file, f"File not found: {wav_file}"))
            continue
        results.append((wav_file, float(model.score(wav_file))))
    return results, errors


def write_results(output_path: Path, results: list[tuple[Path, float]]) -> None:
    """Write stable tab-separated per-file scores and their utterance mean."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        output.write("Name\tKOMOS\n")
        for wav_file, score in results:
            output.write(f"{wav_file.name}\t{score:.3f}\n")
        if results:
            output.write(
                f"\nAverage KOMOS: {fmean(score for _, score in results):.3f} "
                f"({len(results)} samples)\n"
            )


def load_model(device: str) -> ScoringModel:
    """Load the pinned KOMOS checkpoint on the requested device."""
    from komos import KOMOS

    return KOMOS.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()


def main() -> int:
    """Score Korean test-list files and optionally save their MOS results."""
    args = get_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    samples = read_test_list(args.test_list)
    wav_files = select_korean_wavs(samples, args.wav_path, args.extension)
    logging.info("Scoring %d Korean files with KOMOS", len(wav_files))
    results, errors = score_korean_wavs(wav_files, load_model(args.device))

    for wav_file, message in errors:
        logging.error("Failed %s: %s", wav_file, message)
    if not results:
        raise RuntimeError("KOMOS produced no valid scores")

    average = fmean(score for _, score in results)
    logging.info(
        "KOMOS score: %.3f (%d/%d files)", average, len(results), len(wav_files)
    )
    if args.decode_path:
        write_results(args.decode_path, results)
        logging.info("Saved KOMOS results to %s", args.decode_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
