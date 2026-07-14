#!/usr/bin/env python3
"""Run a smoke test against an OmniVoice MLX staging directory."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from omnivoice.mlx import OmniVoiceMLX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", default="Hello.")
    parser.add_argument("--instruct", default="female, british accent")
    parser.add_argument("--duration", type=float, default=0.8)
    parser.add_argument("--num-step", type=int, default=4)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    model = OmniVoiceMLX.from_pretrained(args.model, dtype=args.dtype)
    t_load = time.perf_counter()
    audio = model.generate(
        text=args.text,
        instruct=args.instruct,
        duration=args.duration,
        num_step=args.num_step,
        postprocess_output=False,
    )[0]
    t_gen = time.perf_counter()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, model.sampling_rate)
    print(
        {
            "model": args.model,
            "output": str(out),
            "sampling_rate": model.sampling_rate,
            "samples": int(audio.shape[0]),
            "duration": audio.shape[0] / model.sampling_rate,
            "load_seconds": round(t_load - t0, 3),
            "generate_seconds": round(t_gen - t_load, 3),
        }
    )


if __name__ == "__main__":
    main()
