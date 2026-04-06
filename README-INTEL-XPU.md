# OmniVoice on Intel Arc GPUs (XPU)

Unofficial Intel Arc / Intel XPU fork of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice).

**Upstream:** k2-fsa/OmniVoice (Apache-2.0)
**Fork maintainer:** [@smashingtags](https://github.com/smashingtags) — [mjashley.com](https://mjashley.com)
**Status:** Inference working on Intel Arc Pro B50 (16 GB, Battlemage)
**Branch:** `intel-xpu`

OmniVoice is a massive multilingual zero-shot TTS model supporting 600+ languages, voice cloning, and voice design. This fork adds support for Intel's XPU backend (Intel Arc and Intel Arc Pro GPUs) so the model runs natively on Intel hardware instead of falling back to CPU.

## What Changed

This fork makes minimal, additive changes to get inference running on Intel XPU:

- **Device detection** — `omnivoice.utils.device` centralizes CUDA / XPU / MPS / CPU detection. `infer.py`, `infer_batch.py`, and `demo.py` now auto-detect Intel Arc GPUs.
- **flex_attention safety** — `torch.nn.attention.flex_attention` is imported lazily with a fallback. The packed-sequence training path raises a clear error on backends without flex_attention kernels.
- **fp16 dtype** — ASR pipeline now uses fp16 on XPU, matching the CUDA path.
- **pyproject.toml** — unpinned `torch==2.8.*` so you can install the XPU wheels (which are on the 2.11.x line) without a hard conflict.

**What did NOT need to change:**

- No C++/CUDA kernels were touched — the upstream codebase has none.
- No `flash_attn` replacement needed — OmniVoice relies on transformers' attention selection, which falls back to SDPA automatically when `flash_attn` is not installed.
- The model weights and architecture are unchanged.

## Hardware Tested

| GPU | VRAM | Arch | OS / Setup | Status |
|-----|------|------|------------|--------|
| Intel Arc Pro B50 | 16 GB | Battlemage | Windows 11 | Inference working |
| Intel Arc A310 LP | 3.76 GB usable | Alchemist | Ubuntu 24.04 in privileged Proxmox LXC | Inference working |
| Intel Arc Pro B70 | 32 GB | Big Battlemage | — | Planned |
| Intel Arc A380 | 6 GB | Alchemist | — | Planned |

**Cross-architecture confirmed**: the same 9-file, ~300-line diff runs on both Alchemist (first-gen Arc) and Battlemage (second-gen Arc Pro). OmniVoice fp16 occupies roughly 1.9 GB of VRAM once loaded, so it fits even on the 4 GB A310 with room for activations.

## Installation

### 1. Install Intel GPU drivers

Download from Intel: https://www.intel.com/content/www/us/en/download/785597/

On Windows you also want the latest Intel Arc driver. Verify your GPU is recognized:

```
> Get-CimInstance Win32_VideoController | Select-Object Name
Intel(R) Arc(TM) Pro B50 Graphics
```

### 2. Create a venv

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install PyTorch with XPU support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Verify:

```python
import torch
print(torch.__version__)             # 2.11.0+xpu or newer
print(torch.xpu.is_available())      # True
print(torch.xpu.get_device_name(0))  # Intel(R) Arc(TM) Pro B50 Graphics
```

### 4. Install this fork

```bash
git clone -b intel-xpu https://github.com/smashingtags/OmniVoice.git
cd OmniVoice
pip install --no-deps -e .
pip install transformers==5.3.0 accelerate pydub gradio tensorboardX webdataset numpy soundfile
```

The `--no-deps` flag is important — it prevents pip from pulling in a CPU-only torch that would overwrite the XPU wheel.

### 5. Smoke test

```bash
python test_xpu_inference.py
```

This downloads the model from HuggingFace (~5-15 GB depending on components), loads it on the B50, generates a short sample with voice design, and saves to `xpu_test_out.wav`.

## Usage

Inference works exactly like upstream OmniVoice. Set `device_map="xpu:0"` in place of `"cuda:0"`:

```python
from omnivoice import OmniVoice
import torch
import torchaudio

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="xpu:0",
    dtype=torch.float16,
)

audio = model.generate(
    text="Hello, this is running on an Intel Arc GPU.",
    instruct="female, medium pitch, american accent",
)
torchaudio.save("out.wav", audio[0], 24000)
```

The CLI entry points (`omnivoice-infer`, `omnivoice-infer-batch`, `omnivoice-demo`) auto-detect Intel XPU and will prefer it over CPU without any flag changes.

## Known Limitations

- **Training** — not tested. The packed-sequence training path uses `torch.nn.attention.flex_attention`, which has partial XPU support. Single-GPU SDPA training should work but has not been validated.
- **flash_attn** — cannot be installed on Intel XPU. The model loader will report this as a capability miss; transformers falls back to SDPA automatically. Performance is still good.
- **Multi-GPU** — not tested with multiple Intel Arc cards yet. Planned once multiple B50/B70 cards can be swapped through the test rig.

## Benchmarks

### Intel Arc Pro B50 (16 GB, Battlemage) — Windows 11

| Run | Audio duration | Gen time | RTF |
|-----|---------------|----------|-----|
| Cold start (first run) | 4.85s | 9.13s | 1.88x |
| Warm (second run) | 5.32s | 7.09s | 1.33x |

**Environment:**
- Windows 11
- Python 3.11.9
- PyTorch 2.11.0+xpu
- Intel Arc Pro B50 (driver 32.0.101.8314)
- 16 GB VRAM

**Notes:**
- First successful inference run of OmniVoice on Intel Arc hardware that we're aware of.
- Model loaded in ~4 seconds from cached weights.
- RTF is above 1 (slower than real-time) on Windows — expected. Intel's XPU PyTorch backend on Windows is known to be slower than Linux.
- No CUDA emulation, no CPU fallback — the model is running on the Intel GPU directly via `torch.xpu`.

### Intel Arc A310 LP (4 GB, Alchemist) — Ubuntu 24.04 in Proxmox LXC

| Run | Audio duration | Gen time | RTF |
|-----|---------------|----------|-----|
| Full run (voice design) | 5.30 s | 8.28 s | 1.56 |

**Environment:**
- Proxmox VE 9.1.5 host, kernel 6.17.9-1-pve
- Privileged LXC, Ubuntu 24.04 container
- `/dev/dri/card1` + `/dev/dri/renderD128` bind-mounted into the container
- Python 3.12, PyTorch 2.11.0+xpu
- Intel Arc A310 LP (Alchemist, 3.76 GB usable VRAM)
- Intel OpenCL ICD 25.18, Level Zero 1.21.9, libze-intel-gpu 25.18

**Compute sanity check (no model):**
- 1024×1024 fp16 matmul: 0.61 ms/op
- 4096×4096 fp16 matmul: 88.7 ms/op, **1.55 TFLOPS**

**Notes:**
- **OmniVoice in fp16 occupies ~1.89 GB of VRAM at steady state.** It fits on the 4 GB A310 with meaningful headroom for activations. I assumed it would OOM — it did not.
- Model load was faster than the B50 on Windows (2.3 s vs 3.9 s), likely because the Linux XPU stack is better tuned.
- Same 9-file diff runs on Alchemist and Battlemage with no arch-specific code.
- Privileged LXC with `lxc.cgroup2.devices.allow` entries for majors 226 (DRI) and 29 (framebuffer) — no VFIO passthrough or kernel command-line changes required.

### Cross-Architecture Summary

| GPU | Arch | VRAM (usable) | OS | Load | Warm RTF | Approx price |
|-----|------|---------------|-----|------|----------|--------------|
| Arc A310 LP | Alchemist (Gen 1) | 3.76 GB | Linux (LXC) | 2.3 s | 1.56 | ~$100 |
| Arc Pro B50 | Battlemage (Gen 2) | 15.5 GB | Windows 11 | 3.9 s | 1.33 | ~$349 |

The same code path works on a $100 low-profile card and a $349 workstation card. That's the headline: **OmniVoice on Intel Arc is not a one-card fluke**.

## Upstream

Upstream OmniVoice is maintained by [k2-fsa](https://github.com/k2-fsa) / Xiaomi. Please file bugs in the correct repo:

- **Intel XPU issues** → [this fork](https://github.com/smashingtags/OmniVoice/issues)
- **Model / feature issues** → [upstream](https://github.com/k2-fsa/OmniVoice/issues)

## License

Apache-2.0, same as upstream.
