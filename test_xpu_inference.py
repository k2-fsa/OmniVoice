"""Smoke test: run OmniVoice inference on Intel Arc (XPU).

Usage:
    python test_xpu_inference.py
"""
import sys
import time

import torch
import torchaudio

from omnivoice import OmniVoice
from omnivoice.utils.device import get_best_device


def main() -> int:
    device = get_best_device()
    print(f"[xpu-test] Detected device: {device}")
    if device == "xpu":
        print(f"[xpu-test] XPU device name: {torch.xpu.get_device_name(0)}")
        props = torch.xpu.get_device_properties(0)
        print(f"[xpu-test] Total VRAM: {props.total_memory / (1024**3):.1f} GB")

    dtype = torch.float16 if device in ("cuda", "xpu") else torch.float32

    print("[xpu-test] Loading model from k2-fsa/OmniVoice (may download)...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=f"{device}:0" if device in ("cuda", "xpu") else device,
        dtype=dtype,
    )
    print(f"[xpu-test] Model loaded in {time.time() - t0:.1f}s")

    text = "Hello from Intel Arc. This is the first time OmniVoice has run on Intel's XPU backend."
    print(f"[xpu-test] Generating: {text!r}")

    t0 = time.time()
    audio = model.generate(
        text=text,
        instruct="male, moderate pitch, american accent",
    )
    gen_time = time.time() - t0
    duration_s = audio[0].shape[-1] / 24000
    rtf = gen_time / duration_s
    print(f"[xpu-test] Generated {duration_s:.2f}s audio in {gen_time:.2f}s (RTF={rtf:.3f})")

    out_path = "xpu_test_out.wav"
    try:
        torchaudio.save(out_path, audio[0], 24000)
    except (ImportError, RuntimeError):
        # torchaudio 2.11+ routes save() through torchcodec which is not
        # always available. Fall back to soundfile.
        import soundfile as sf
        sf.write(out_path, audio[0].squeeze().cpu().numpy(), 24000)
    print(f"[xpu-test] Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
