"""Single-item inference CLI for OmniVoice.

Generates audio from a single text input using voice cloning,
voice design, or auto voice.

Usage:
    # Voice cloning
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --ref_audio ref.wav --ref_text "Reference transcript." --output out.wav

    # Voice design
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." \
        --instruct "male, British accent" --output out.wav

    # Auto voice
    omnivoice-infer --model k2-fsa/OmniVoice \
        --text "Hello, this is a text for text-to-speech." --output out.wav
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from omnivoice.models.omnivoice import (
    OmniVoice,
    _normalize_final_duration_targets,
    _validate_final_duration_inputs,
)
from omnivoice.utils.audio import _validate_output_waveform, write_output_wav
from omnivoice.utils.common import (
    get_best_device,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    positive_unit_float,
    str2bool,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        default=None,
        help="Path to a LoRA adapter directory (e.g. a LoRA training "
        "checkpoint) to apply on top of --model. Merged in-memory before "
        "generation.",
    )
    # Voice cloning
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio file path for voice cloning.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference text describing the reference audio.",
    )
    # Voice design
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name (e.g. 'English') or code (e.g. 'en').",
    )
    # Generation parameters
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=positive_float, default=1.0)
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=None,
        help="Pre-synthesis audio-token budget in seconds. If set, overrides "
        "the model's duration estimation. Output post-processing can change "
        "the physical WAV duration.",
    )
    final_duration_group = parser.add_mutually_exclusive_group()
    final_duration_group.add_argument(
        "--final_duration",
        type=positive_float,
        default=None,
        help="Exact physical output duration in seconds, converted to sample "
        "frames with decimal HALF_UP rounding. Independent of --duration.",
    )
    final_duration_group.add_argument(
        "--final_duration_samples",
        type=positive_int,
        default=None,
        help="Authoritative exact physical output length in integer sample "
        "frames per channel. Must be greater than zero.",
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument(
        "--postprocess_output",
        type=str2bool,
        default=True,
        help="Whether to shorten long internal silences and trim edge silence.",
    )
    parser.add_argument(
        "--output_mode",
        choices=("processed", "raw_codec"),
        default="processed",
        help="Return normal processed audio or the unmodified codec-decoder "
        "waveform. raw_codec bypasses all signal post-processing; an explicit "
        "physical duration still applies zero-only output framing.",
    )
    parser.add_argument(
        "--output_min_silence_ms",
        type=nonnegative_int,
        default=500,
        help="Minimum internal silence duration to shorten, in milliseconds.",
    )
    parser.add_argument(
        "--output_preserve_active_edges",
        action="store_true",
        help="Keep all nonzero outer-edge samples during silence removal, "
        "including quiet noise. Fades and final alignment remain independent.",
    )
    parser.add_argument(
        "--output_keep_silence_ms",
        type=nonnegative_int,
        default=None,
        help="Maximum total silence retained from each shortened gap. "
        "By default, preserves historical per-side behavior and retains up to "
        "twice --output_min_silence_ms in total.",
    )
    parser.add_argument(
        "--output_lead_silence_ms",
        type=nonnegative_int,
        default=100,
        help="Leading silence retained before optional padding, in milliseconds.",
    )
    parser.add_argument(
        "--output_trail_silence_ms",
        type=nonnegative_int,
        default=100,
        help="Trailing silence retained before optional padding, in milliseconds.",
    )
    parser.add_argument(
        "--output_peak_limit",
        type=positive_unit_float,
        default=None,
        help="Optional final absolute peak ceiling in the interval (0, 1].",
    )
    parser.add_argument(
        "--output_target_lead_silence_ms",
        type=nonnegative_int,
        default=None,
        help="Optional exact pre-framing leading-silence anchor in milliseconds. "
        "This overrides generic edge padding on the leading side; final-duration "
        "fitting may add separately reported outer-container zero fill.",
    )
    parser.add_argument(
        "--output_target_trail_silence_ms",
        type=nonnegative_int,
        default=None,
        help="Optional exact pre-framing trailing-silence anchor in milliseconds. "
        "This overrides generic edge padding on the trailing side; final-duration "
        "fitting may add separately reported outer-container zero fill.",
    )
    parser.add_argument("--pad_duration", type=nonnegative_float, default=0.1)
    parser.add_argument("--fade_duration", type=nonnegative_float, default=0.1)
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Auto-detected if not specified.",
    )
    return parser


def _validate_generated_output(audios, final_duration_target) -> np.ndarray:
    """Validate model cardinality, shape, and physical length before writing."""
    if not isinstance(audios, (list, tuple)) or len(audios) != 1:
        actual_count = len(audios) if isinstance(audios, (list, tuple)) else "non-list"
        raise RuntimeError(
            f"OmniVoice returned an unexpected number of outputs: expected 1, got {actual_count}"
        )
    audio = audios[0]
    audio = _validate_output_waveform(audio, label="OmniVoice output 0")
    if (
        final_duration_target is not None
        and audio.shape[0] != final_duration_target.samples
    ):
        raise RuntimeError(
            "OmniVoice output 0 violates the requested physical duration: "
            f"expected {final_duration_target.samples} samples, got {audio.shape[0]}"
        )
    return audio


def _validate_output_path(path: str) -> None:
    """Reject structurally invalid output paths before loading the model."""

    destination = Path(path)
    parent = destination.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError(
            f"output parent directory does not exist or is not a directory: {parent}"
        )
    if os.path.lexists(destination) and not destination.is_file():
        raise ValueError(f"output path exists and is not a regular file: {destination}")


def main():
    args = get_parser().parse_args()
    _validate_final_duration_inputs(
        args.final_duration,
        args.final_duration_samples,
        batch_size=1,
    )
    _validate_output_path(args.output)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    if args.lora_adapter:
        from omnivoice.utils.lora import load_lora_adapter

        logging.info(f"Applying LoRA adapter from {args.lora_adapter} ...")
        model = load_lora_adapter(model, args.lora_adapter)

    final_duration_targets = _normalize_final_duration_targets(
        args.final_duration,
        args.final_duration_samples,
        batch_size=1,
        sampling_rate=model.sampling_rate,
    )
    final_duration_target = (
        final_duration_targets[0] if final_duration_targets is not None else None
    )

    logging.info(f"Generating audio for: {args.text[:80]}...")
    physical_duration_kwargs = {}
    if args.final_duration is not None:
        physical_duration_kwargs["final_duration"] = args.final_duration
    if args.final_duration_samples is not None:
        physical_duration_kwargs["final_duration_samples"] = args.final_duration_samples
    audios = model.generate(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        duration=args.duration,
        **physical_duration_kwargs,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        output_mode=args.output_mode,
        output_min_silence_ms=args.output_min_silence_ms,
        output_keep_silence_ms=args.output_keep_silence_ms,
        output_preserve_active_edges=args.output_preserve_active_edges,
        output_lead_silence_ms=args.output_lead_silence_ms,
        output_trail_silence_ms=args.output_trail_silence_ms,
        output_peak_limit=args.output_peak_limit,
        output_target_lead_silence_ms=args.output_target_lead_silence_ms,
        output_target_trail_silence_ms=args.output_target_trail_silence_ms,
        pad_duration=args.pad_duration,
        fade_duration=args.fade_duration,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
    )
    audio = _validate_generated_output(audios, final_duration_target)

    write_output_wav(
        args.output,
        audio,
        model.sampling_rate,
        args.output_mode,
    )
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
