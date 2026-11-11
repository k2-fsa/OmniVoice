import subprocess
import sys
from pathlib import Path

import numpy as np

from omnivoice.utils.audio import resample_audio


def test_resample_audio_preserves_channels_and_float32():
    waveform = np.zeros((2, 1600), dtype=np.float32)

    resampled = resample_audio(waveform, orig_freq=16000, new_freq=8000)

    assert resampled.shape == (2, 800)
    assert resampled.dtype == np.float32


def test_import_omnivoice_without_torchaudio():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import builtins

original_import = builtins.__import__

def import_without_torchaudio(name, *args, **kwargs):
    if name == "torchaudio" or name.startswith("torchaudio."):
        raise ModuleNotFoundError("torchaudio is intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_torchaudio
import omnivoice
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
