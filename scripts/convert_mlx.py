#!/usr/bin/env python3
"""Export an OmniVoice checkpoint into an MLX staging directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx


STATIC_FILES = [
    ".gitattributes",
    "chat_template.jinja",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Official OmniVoice checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output MLX staging directory.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Floating-point dtype for exported model weights.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target Hugging Face repo id to record in mlx_manifest.json.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Human-readable variant name to record in README and manifest.",
    )
    parser.add_argument(
        "--copy-mode",
        default="hardlink",
        choices=["hardlink", "copy"],
        help="How to copy static files and the audio tokenizer.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> mx.Dtype:
    return {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }[name]


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_static(source: Path, output: Path, mode: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for rel in STATIC_FILES:
        src = source / rel
        if src.exists():
            link_or_copy(src, output / rel, mode)

    audio_src = source / "audio_tokenizer"
    audio_dst = output / "audio_tokenizer"
    audio_dst.mkdir(parents=True, exist_ok=True)
    for src in audio_src.rglob("*"):
        if src.is_file() and ".cache" not in src.parts:
            link_or_copy(src, audio_dst / src.relative_to(audio_src), mode)


def load_source_weights(source: Path) -> dict[str, mx.array]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
    else:
        files = ["model.safetensors"]

    weights: dict[str, mx.array] = {}
    for name in files:
        weights.update(mx.load(str(source / name)))
    return weights


def save_index(output: Path, weights: dict[str, mx.array]) -> None:
    weight_map = {key: "model.safetensors" for key in sorted(weights)}
    total_size = sum(int(array.nbytes) for array in weights.values())
    total_parameters = sum(int(array.size) for array in weights.values())
    index = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": total_parameters,
        },
        "weight_map": weight_map,
    }
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def write_manifest(output: Path, args: argparse.Namespace, weights: dict[str, mx.array]) -> None:
    manifest = {
        "format": "mlx-compatible-safetensors",
        "model": "OmniVoice",
        "source_repo": "k2-fsa/OmniVoice",
        "source_model_dir": str(Path(args.source).resolve()),
        "target_repo": args.repo_id,
        "variant": args.variant or args.dtype,
        "precision": args.dtype,
        "weight_pattern": "model.safetensors",
        "index_file": "model.safetensors.index.json",
        "config_file": "config.json",
        "audio_tokenizer_dir": "audio_tokenizer",
        "audio_tokenizer_repo": "eustlb/higgs-audio-v2-tokenizer",
        "tensor_count": len(weights),
        "quantization": None,
    }
    with (output / "mlx_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def write_readme(output: Path, args: argparse.Namespace) -> None:
    variant = args.variant or args.dtype
    repo = args.repo_id or output.name
    source_readme = Path(args.source) / "README.md"
    official = (
        source_readme.read_text(encoding="utf-8")
        if source_readme.exists()
        else "# OmniVoice\n"
    )
    text = inject_mlx_note(official, variant=variant, repo=repo)
    with (output / "README.md").open("w", encoding="utf-8") as f:
        f.write(text)


def inject_mlx_note(readme: str, variant: str, repo: str) -> str:
    start = "<!-- OMNIVOICE_MLX_NOTE_START -->"
    end = "<!-- OMNIVOICE_MLX_NOTE_END -->"
    if start in readme and end in readme:
        before = readme.split(start, 1)[0]
        after = readme.split(end, 1)[1].lstrip("\n")
        readme = before + after

    note = f"""{start}
> [!IMPORTANT]
> This repository is a community MLX conversion of the official `k2-fsa/OmniVoice` Hugging Face release.
>
> The official model and original model card remain the authoritative source for model capability, license, training, and citation details. This repository only adds Apple Silicon / MLX packaging and runtime notes.
>
> MLX runtime and conversion scripts are maintained here:
>
> - GitHub: https://github.com/ailuntx/OmniVoice-MLX
> - Hugging Face target repo: https://huggingface.co/{repo}
>
> Current local variant: `{variant}`.

## MLX Usage

```bash
pip install omnivoice mlx

omnivoice-infer-mlx \\
  --model . \\
  --text "Hello." \\
  --instruct "female, british accent" \\
  --output out.wav
```

The `audio_tokenizer/` directory is included in this repository so the model can be loaded from a single local directory.

{end}
"""
    if readme.startswith("---\n"):
        close = readme.find("\n---\n", 4)
        if close != -1:
            insert_at = close + len("\n---\n")
            return readme[:insert_at] + "\n" + note + "\n" + readme[insert_at:].lstrip("\n")
    return note + "\n" + readme


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    dtype = resolve_dtype(args.dtype)

    copy_static(source, output, args.copy_mode)
    weights = load_source_weights(source)
    converted = {
        key: value.astype(dtype) if value.dtype in (mx.float16, mx.float32, mx.bfloat16) else value
        for key, value in weights.items()
    }
    mx.save_safetensors(str(output / "model.safetensors"), converted)
    save_index(output, converted)
    write_manifest(output, args, converted)
    write_readme(output, args)
    print(output)


if __name__ == "__main__":
    main()
