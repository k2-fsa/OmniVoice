#!/usr/bin/env python3
"""Build the five local Hugging Face staging directories for OmniVoice MLX."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOS = {
    "OmniVoice-MLX": ("float32", None),
    "OmniVoice-MLX-fp32": ("float32", None),
    "OmniVoice-MLX-bf16": ("bfloat16", None),
    "OmniVoice-MLX-8bit": ("int8", 8),
    "OmniVoice-MLX-4bit": ("int4", 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="/Volumes/usb_main/home/index_mlx/models/OmniVoice-official",
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/usb_main/home/index_mlx/huggingface",
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    scripts = Path(__file__).resolve().parent
    source = Path(args.source)

    for name, (precision, bits) in REPOS.items():
        out = root / name
        if args.skip_existing and (out / "model.safetensors").exists():
            print(f"skip existing {out}")
            continue
        repo_id = f"mlx-community/{name}"
        if bits is None:
            run(
                [
                    sys.executable,
                    str(scripts / "convert_mlx.py"),
                    "--source",
                    str(source),
                    "--output",
                    str(out),
                    "--dtype",
                    precision,
                    "--repo-id",
                    repo_id,
                    "--variant",
                    precision,
                ]
            )
        else:
            dense_source = root / "OmniVoice-MLX-fp32"
            if not (dense_source / "model.safetensors").exists():
                run(
                    [
                        sys.executable,
                        str(scripts / "convert_mlx.py"),
                        "--source",
                        str(source),
                        "--output",
                        str(dense_source),
                        "--dtype",
                        "float32",
                        "--repo-id",
                        "mlx-community/OmniVoice-MLX-fp32",
                        "--variant",
                        "float32",
                    ]
                )
            run(
                [
                    sys.executable,
                    str(scripts / "quantize_mlx.py"),
                    "--source",
                    str(dense_source),
                    "--output",
                    str(out),
                    "--bits",
                    str(bits),
                    "--group-size",
                    str(args.group_size),
                    "--repo-id",
                    repo_id,
                    "--variant",
                    precision,
                ]
            )


if __name__ == "__main__":
    main()
