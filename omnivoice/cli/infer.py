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

import torch

import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import (
    get_best_device,
    nonnegative_float,
    nonnegative_int,
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
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Pre-synthesis audio-token budget in seconds. If set, overrides "
        "the model's duration estimation. Output post-processing can change "
        "the physical WAV duration.",
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
        "waveform. raw_codec bypasses all output post-processing.",
    )
    parser.add_argument(
        "--output_min_silence_ms",
        type=nonnegative_int,
        default=500,
        help="Minimum internal silence duration to shorten, in milliseconds.",
    )
    parser.add_argument(
        "--output_keep_silence_ms",
        type=nonnegative_int,
        default=None,
        help="Maximum total silence retained from each shortened gap. "
        "Defaults to --output_min_silence_ms.",
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
        help="Optional exact final leading silence in milliseconds. This "
        "overrides generic edge padding on the leading side.",
    )
    parser.add_argument(
        "--output_target_trail_silence_ms",
        type=nonnegative_int,
        default=None,
        help="Optional exact final trailing silence in milliseconds. This "
        "overrides generic edge padding on the trailing side.",
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


def _write_output_wav(path, audio, sampling_rate, output_mode):
    """Write raw codec samples losslessly while preserving processed defaults."""

    if output_mode == "raw_codec":
        sf.write(path, audio, sampling_rate, subtype="FLOAT")
    else:
        sf.write(path, audio, sampling_rate)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    if args.lora_adapter:
        from omnivoice.utils.lora import load_lora_adapter

        logging.info(f"Applying LoRA adapter from {args.lora_adapter} ...")
        model = load_lora_adapter(model, args.lora_adapter)

    logging.info(f"Generating audio for: {args.text[:80]}...")
    audios = model.generate(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        duration=args.duration,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        output_mode=args.output_mode,
        output_min_silence_ms=args.output_min_silence_ms,
        output_keep_silence_ms=args.output_keep_silence_ms,
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

    _write_output_wav(
        args.output,
        audios[0],
        model.sampling_rate,
        args.output_mode,
    )
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
