#!/usr/bin/env python3
"""Standard-library configuration helpers for the OmniVoice demo."""

import os
from pathlib import Path


def env_value(name: str, default: str) -> str:
    """Return a non-empty environment value, or its safe default."""
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


def env_port(name: str = "OMNIVOICE_PORT", default: int = 7860) -> int:
    """Read and validate a TCP port from the environment."""
    value = env_value(name, str(default))
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535, got {port}")
    return port


def ensure_output_dir(path: str | None = None) -> Path:
    """Create and return the directory used for Gradio-generated files."""
    output_dir = Path(path or env_value("OMNIVOICE_OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
