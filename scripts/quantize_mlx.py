#!/usr/bin/env python3
"""Create row-wise 8-bit or 4-bit OmniVoice MLX staging weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


STATIC_FILES = [
    ".gitattributes",
    "chat_template.jinja",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Dense OmniVoice checkpoint directory.")
    parser.add_argument("--output", required=True, help="Quantized output directory.")
    parser.add_argument("--bits", type=int, required=True, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--copy-mode", default="hardlink", choices=["hardlink", "copy"])
    return parser.parse_args()


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


def iter_weight_files(source: Path) -> list[Path]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            index = json.load(f)
        return [source / name for name in sorted(set(index["weight_map"].values()))]
    return [source / "model.safetensors"]


def should_quantize(key: str, tensor: np.ndarray) -> bool:
    if tensor.ndim != 2:
        return False
    if not np.issubdtype(tensor.dtype, np.floating):
        return False
    return key.endswith(".weight")


def quantize_tensor(x: np.ndarray, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    x = x.astype(np.float32, copy=False)
    shape = x.shape
    cols = shape[-1]
    flat = x.reshape(-1, cols)
    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size
    if padded_cols != cols:
        padded = np.zeros((flat.shape[0], padded_cols), dtype=np.float32)
        padded[:, :cols] = flat
        flat = padded
    grouped = flat.reshape(flat.shape[0], groups, group_size)

    qmax = 127 if bits == 8 else 7
    scales = np.max(np.abs(grouped), axis=-1, keepdims=True) / qmax
    scales = np.maximum(scales, 1e-8)
    quant = np.clip(np.round(grouped / scales), -qmax - (1 if bits == 4 else 0), qmax)

    if bits == 8:
        qweight = quant.astype(np.int8)
    else:
        q4 = quant.astype(np.int16) + 8
        qweight = (q4[:, :, 0::2] | (q4[:, :, 1::2] << 4)).astype(np.uint8)

    info = {
        "bits": bits,
        "shape": list(shape),
        "group_size": group_size,
    }
    return qweight, scales.squeeze(-1).astype(np.float16), info


def write_index(output: Path, tensors: dict[str, np.ndarray]) -> None:
    weight_map = {key: "model.safetensors" for key in sorted(tensors)}
    total_size = sum(int(value.nbytes) for value in tensors.values())
    total_parameters = sum(int(value.size) for value in tensors.values())
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "total_size": total_size,
                    "total_parameters": total_parameters,
                },
                "weight_map": weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")


def write_manifest(
    output: Path,
    args: argparse.Namespace,
    quantized: dict[str, dict],
    tensor_count: int,
) -> None:
    variant = args.variant or f"{args.bits}bit"
    manifest = {
        "format": "mlx-compatible-safetensors",
        "model": "OmniVoice",
        "source_repo": "k2-fsa/OmniVoice",
        "source_model_dir": str(Path(args.source).resolve()),
        "target_repo": args.repo_id,
        "variant": variant,
        "precision": f"int{args.bits}",
        "weight_pattern": "model.safetensors",
        "index_file": "model.safetensors.index.json",
        "config_file": "config.json",
        "audio_tokenizer_dir": "audio_tokenizer",
        "audio_tokenizer_repo": "eustlb/higgs-audio-v2-tokenizer",
        "tensor_count": tensor_count,
        "quantization": {
            "format": "omnivoice-rowwise",
            "bits": args.bits,
            "group_size": args.group_size,
            "tensors": quantized,
        },
    }
    with (output / "mlx_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def write_readme(output: Path, args: argparse.Namespace) -> None:
    variant = args.variant or f"{args.bits}bit"
    repo = args.repo_id or output.name
    source_readme = Path(args.source) / "README.md"
    official = (
        source_readme.read_text(encoding="utf-8")
        if source_readme.exists()
        else "# OmniVoice\n"
    )
    text = inject_mlx_note(official, variant=variant, repo=repo, bits=args.bits)
    with (output / "README.md").open("w", encoding="utf-8") as f:
        f.write(text)


def inject_mlx_note(readme: str, variant: str, repo: str, bits: int) -> str:
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
> Current local variant: `{variant}` row-wise {bits}-bit. The OmniVoice MLX loader reads `mlx_manifest.json` and dequantizes weights at load time.

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
    copy_static(source, output, args.copy_mode)

    tensors: dict[str, np.ndarray] = {}
    quantized: dict[str, dict] = {}
    for file in iter_weight_files(source):
        with safe_open(file, framework="np") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if should_quantize(key, tensor):
                    qweight, scales, info = quantize_tensor(tensor, args.bits, args.group_size)
                    tensors[key] = qweight
                    tensors[f"{key}.scales"] = scales
                    quantized[key] = info
                elif np.issubdtype(tensor.dtype, np.floating):
                    tensors[key] = tensor.astype(np.float16)
                else:
                    tensors[key] = tensor

    save_file(tensors, str(output / "model.safetensors"))
    write_index(output, tensors)
    write_manifest(output, args, quantized, len(tensors))
    write_readme(output, args)
    print(output)


if __name__ == "__main__":
    main()
