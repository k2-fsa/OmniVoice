"""Colab/GPU runner for the frozen 72-sample audio manifest; not run locally."""

import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch

from omnivoice import OmniVoice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--checkpoint", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    model = OmniVoice.from_pretrained(args.checkpoint, device_map=args.device, dtype=torch.float16)
    prompt = model.create_voice_clone_prompt(ref_audio=str(args.reference_audio), ref_text=args.reference_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        output = args.output_dir / row["audio_path"]
        if output.is_file() and output.stat().st_size > 0: continue
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        audio = model.generate(text=row["target_text"], voice_clone_prompt=prompt,
                               normalize_text=False, num_step=16)
        sf.write(output, audio[0], model.sampling_rate)


if __name__ == "__main__":
    main()
