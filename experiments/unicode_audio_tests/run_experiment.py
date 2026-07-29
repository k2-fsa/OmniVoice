"""Reproduce the Unicode inference experiment with an already-loaded model.

In a notebook or Python session where ``model`` is an initialized OmniVoice
instance:

    from experiments.unicode_audio_tests.run_experiment import run_experiment
    metadata_path = run_experiment(model)
"""

import json
import random
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoiceGenerationConfig
from omnivoice.utils.text import normalize_text_input

BASE_TEXT = "Khánh Huyền khuyên Quỳnh chuyển chuyến tàu đến Huế vào chiều thứ Năm."
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    model: Any,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    seed: int = 2026,
    generation_config: OmniVoiceGenerationConfig | None = None,
) -> Path:
    """Generate original and explicitly repaired Unicode variants.

    ``model`` must already be loaded. Auto-voice generation is used, so no
    reference audio or transcript is required.
    """
    config = generation_config or OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        position_temperature=5.0,
        class_temperature=0.0,
    )
    variants = {
        "nfc": unicodedata.normalize("NFC", BASE_TEXT),
        "nfd": unicodedata.normalize("NFD", BASE_TEXT),
        "nonbreaking_space": BASE_TEXT.replace(" ", "\u00a0"),
        "zero_width_space": BASE_TEXT.replace("Huyền", "Hu\u200byền"),
    }

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sampling_rate = int(getattr(model, "sampling_rate", 24000))
    records = []

    for name, original_text in variants.items():
        normalized_text = normalize_text_input(original_text)
        for input_kind, inference_text in (
            ("original", original_text),
            ("normalized", normalized_text),
        ):
            _set_seed(seed)
            audio = model.generate(
                text=inference_text,
                generation_config=config,
            )[0]
            output_path = results_dir / f"{name}_{input_kind}.wav"
            sf.write(output_path, audio, sampling_rate)
            records.append(
                {
                    "case": name,
                    "input_kind": input_kind,
                    "input_repr": repr(inference_text),
                    "normalized_text": normalized_text,
                    "duration_seconds": len(audio) / sampling_rate,
                    "sampling_rate": sampling_rate,
                    "seed": seed,
                    "generation_config": asdict(config),
                    "output_file": output_path.name,
                }
            )

    metadata_path = results_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return metadata_path
