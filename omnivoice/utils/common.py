#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
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

"""Shared utility functions."""

import argparse
import random
from typing import Any

import numpy as np
import torch


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fix_random_seed(random_seed: int):
    """
    Set the same random seed for the libraries and modules.
    Includes the ``random`` module, numpy, and torch.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    # Ensure deterministic ID creation
    rd = random.Random()
    rd.seed(random_seed)


def resolve_device_string(device: Any) -> str:
    """Return a representative device string for runtime selection."""
    if device is None:
        return "cpu"
    if isinstance(device, dict):
        values = [resolve_device_string(v) for v in device.values()]
        for value in values:
            if value.startswith("cuda"):
                return value
        for value in values:
            if value.startswith("mps"):
                return value
        return values[0] if values else "cpu"
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def get_best_device() -> str:
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_best_device_and_count() -> tuple[str, int]:
    """Return the best device string plus the number of usable devices."""
    device = get_best_device()
    if device.startswith("cuda"):
        return device, torch.cuda.device_count()
    return device, 1


def _get_cuda_device_index(device: Any) -> int:
    normalized = torch.device(resolve_device_string(device))
    return normalized.index if normalized.index is not None else 0


def _cuda_supports_bfloat16(device: Any) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(_get_cuda_device_index(device))
    return major >= 8


def resolve_inference_dtype(device: Any, dtype: Any = "auto") -> torch.dtype:
    """Pick a safe and fast inference dtype for the requested device."""
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype is None:
        dtype = "auto"

    if not isinstance(dtype, str):
        raise TypeError(f"Unsupported dtype specifier: {dtype!r}")

    normalized_dtype = dtype.lower()
    if normalized_dtype == "auto":
        device_str = resolve_device_string(device)
        if device_str.startswith("cuda"):
            return (
                torch.bfloat16
                if _cuda_supports_bfloat16(device_str)
                else torch.float16
            )
        return torch.float32

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized_dtype not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of: {valid}, auto")
    return aliases[normalized_dtype]


def configure_cuda_inference(device: Any) -> None:
    """Enable CUDA matmul fast paths that are safe for inference."""
    if not resolve_device_string(device).startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
