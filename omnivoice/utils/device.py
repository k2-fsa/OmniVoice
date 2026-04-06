"""Device detection helpers with Intel XPU (Arc GPU) support.

Added for the intel-xpu fork (smashingtags/OmniVoice).
Supports CUDA, Intel XPU, Apple MPS, and CPU.
"""

import torch


def get_best_device() -> str:
    """Auto-detect the best available device: CUDA > XPU > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_and_count() -> tuple[str, int]:
    """Return (device_str, device_count)."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.xpu.device_count()
    if torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def device_str(device_type: str, index: int = 0) -> str:
    """Build a device string like 'cuda:0', 'xpu:0', 'mps', 'cpu'."""
    if device_type in ("cuda", "xpu"):
        return f"{device_type}:{index}"
    return device_type


def empty_cache(device_type: str) -> None:
    """Call the device-specific empty_cache if available."""
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif device_type == "mps" and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def supports_fp16(device_type: str) -> bool:
    """Whether fp16 should be used on this device."""
    return device_type in ("cuda", "xpu")
