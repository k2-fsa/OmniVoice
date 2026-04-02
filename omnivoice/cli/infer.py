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
import time

import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import str2bool
from omnivoice.utils.text import chunk_text_punctuation


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        default=None,
        help="Text to synthesize (ignored if --interactive is used).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Path to output audio file (used as base name in interactive mode).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (load model once, enter text repeatedly).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable sentence-level streaming (process and save chunks independently).",
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
        help="Fixed output duration in seconds. If set, overrides the "
        "model's duration estimation. The speed factor is automatically "
        "adjusted to match while preserving language-aware pacing.",
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
        help="Device to use for inference. Auto-detected if not specified.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

    def run_gen(text, output_path, is_stream=False):
        if is_stream:
            chunks = chunk_text_punctuation(text, chunk_len=100)
            if len(chunks) > 1:
                logging.info(f"Streaming mode: split input into {len(chunks)} chunks.")
                for i, chunk in enumerate(chunks):
                    chunk_out = output_path
                    if chunk_out.endswith(".wav"):
                        chunk_out = chunk_out[:-4] + f"_chunk_{i}.wav"
                    else:
                        chunk_out = chunk_out + f"_chunk_{i}.wav"
                    run_gen(chunk, chunk_out, is_stream=False)
                return

        logging.info(f"Generating audio for: {text[:80]}...")
        start_time = time.perf_counter()
        audios = model.generate(
            text=text,
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
        gen_time = time.perf_counter() - start_time
        audio_duration = audios[0].shape[-1] / model.sampling_rate
        rtf = gen_time / audio_duration

        logging.info(
            f"Generation complete: {gen_time:.2f}s for {audio_duration:.2f}s audio (RTF: {rtf:.3f})"
        )
        sf.write(output_path, audios[0].squeeze(0).cpu().numpy(), model.sampling_rate)
        logging.info(f"Saved to {output_path}")

    if args.interactive:
        logging.info("Entering interactive mode. Type 'exit' to quit.")
        logging.info("Tip: You can use --stream at startup to enable sentence-level streaming.")
        logging.info("Warming up model...")
        model.generate(text="Warmup", num_step=1)

        idx = 0
        while True:
            try:
                user_text = input("\nEnter text to synthesize > ").strip()
                if not user_text:
                    continue
                if user_text.lower() in ("exit", "quit"):
                    break

                out_name = args.output
                if out_name.endswith(".wav"):
                    out_name = out_name[:-4] + f"_{idx}.wav"
                else:
                    out_name = out_name + f"_{idx}.wav"

                run_gen(user_text, out_name, is_stream=args.stream)
                idx += 1
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Error during generation: {e}")
    else:
        run_gen(args.text, args.output, is_stream=args.stream)


if __name__ == "__main__":
    main()
