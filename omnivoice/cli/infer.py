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
import torchaudio

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.audio import save_audio
from omnivoice.utils.common import str2bool
from omnivoice.utils.i18n import init_i18n


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=_("OmniVoice single-item inference"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help=_("Model checkpoint path or HuggingFace repo id."),
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help=_("Text to synthesize."),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=_("Output WAV file path."),
    )
    # Voice cloning
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help=_("Reference audio file path for voice cloning."),
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help=_("Reference text describing the reference audio."),
    )
    # Voice design
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help=_("Style instruction for voice design mode."),
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=_("Language name (e.g. 'English') or code (e.g. 'en')."),
    )
    # Generation parameters
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=_("Fixed output duration in seconds. If set, overrides the "
        "model's duration estimation. The speed factor is automatically "
        "adjusted to match while preserving language-aware pacing."),
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument(
        "--postprocess_output",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=_("Device to use for inference. Auto-detected if not specified."),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help=_("Interface language (en, pt_BR, zh)."),
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    # Initialize i18n
    init_i18n(args.lang)

    device = args.device or get_best_device()
    logging.info(_("Loading model from {model} on {device} ...").format(
        model=args.model, device=device
    ))
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    logging.info(_("Generating audio for: {text}...").format(
        text=args.text[:80]
    ))
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
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
    )

    save_audio(audios[0], model.sampling_rate, args.output)
    logging.info(_("Saved to {output}").format(output=args.output))


if __name__ == "__main__":
    main()
