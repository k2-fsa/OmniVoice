"""Single-item MLX inference CLI for OmniVoice."""

import argparse
import logging

import soundfile as sf

from omnivoice.mlx import OmniVoiceMLX
from omnivoice.utils.common import str2bool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item inference with the MLX backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="k2-fsa/OmniVoice")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ref_audio", type=str, default=None)
    parser.add_argument("--ref_text", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument("--postprocess_output", type=str2bool, default=True)
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="MLX weight and compute dtype.",
    )
    parser.add_argument(
        "--audio_tokenizer_device",
        type=str,
        default="cpu",
        help="Device map for the Transformers Higgs audio tokenizer.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    args = get_parser().parse_args()

    logging.info("Loading MLX model from %s ...", args.model)
    model = OmniVoiceMLX.from_pretrained(
        args.model,
        dtype=args.dtype,
        audio_tokenizer_device=args.audio_tokenizer_device,
    )

    logging.info("Generating audio for: %s...", args.text[:80])
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

    sf.write(args.output, audios[0], model.sampling_rate)
    logging.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
