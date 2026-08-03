"""Single-item and multi-sentence inference CLI for OmniVoice.

Generates audio from one or more text inputs using voice cloning,
voice design, or auto voice.  When multiple sentences are supplied
(via ``--text`` repeated or ``--sentences``), one WAV file is written
per sentence.

Usage:
    # Voice cloning — single sentence
    omnivoice-infer --model k2-fsa/OmniVoice \\
        --text "Hello, this is a test." \\
        --ref_audio ref.wav --ref_text "Reference transcript." \\
        --output out.wav

    # Voice cloning — ref_text omitted (Whisper auto-transcribes)
    omnivoice-infer --model k2-fsa/OmniVoice \\
        --text "Hello, this is a test." \\
        --ref_audio ref.wav --output out.wav

    # Voice cloning — multiple sentences → out_001.wav, out_002.wav, …
    omnivoice-infer --model k2-fsa/OmniVoice \\
        --text "First sentence." --text "Second sentence." \\
        --ref_audio ref.wav --output out.wav

    # Voice cloning — sentences from a text file (one per line)
    omnivoice-infer --model k2-fsa/OmniVoice \\
        --sentences sentences.txt \\
        --ref_audio ref.wav --output out.wav

    # Voice design
    omnivoice-infer --model k2-fsa/OmniVoice \\
        --text "Hello, this is a test." \\
        --instruct "male, British accent" --output out.wav

    # Auto voice
    omnivoice-infer --model k2-fsa/OmniVoice \\
        --text "Hello, this is a test." --output out.wav
"""

import argparse
import logging
import os

import torch
import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device, str2bool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item and multi-sentence inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )

    # ------------------------------------------------------------------ #
    # Text input — accepts one or more --text flags, or a sentences file  #
    # ------------------------------------------------------------------ #
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        type=str,
        action="append",
        dest="texts",
        metavar="TEXT",
        help="Text to synthesize. Repeat the flag to supply multiple sentences; "
             "each sentence produces its own output file.",
    )
    text_group.add_argument(
        "--sentences",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to a plain-text file with one sentence per line. "
             "Empty lines and lines starting with '#' are ignored. "
             "Each sentence produces its own output file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV file path. For a single sentence this is used as-is. "
             "For multiple sentences the stem is suffixed with a zero-padded "
             "index, e.g. 'out.wav' → 'out_001.wav', 'out_002.wav', …",
    )

    # ------------------------------------------------------------------ #
    # Voice cloning                                                        #
    # ------------------------------------------------------------------ #
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
        help="Transcript of the reference audio. "
             "If omitted, Whisper ASR will auto-transcribe it "
             "(requires the ASR model to be loaded; see --no_asr).",
    )

    # ------------------------------------------------------------------ #
    # ASR (auto-transcription) options                                     #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--no_asr",
        action="store_true",
        default=False,
        help="Do NOT load the Whisper ASR model. "
             "If --ref_audio is provided without --ref_text, inference will fail.",
    )
    parser.add_argument(
        "--asr_model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper ASR model path or HuggingFace repo id used for "
             "auto-transcribing reference audio when --ref_text is omitted.",
    )

    # ------------------------------------------------------------------ #
    # Voice design                                                         #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design mode "
             "(e.g. 'male, British accent'). "
             "Ignored when --ref_audio is provided.",
    )

    # ------------------------------------------------------------------ #
    # Language                                                             #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name (e.g. 'English') or code (e.g. 'en'). "
             "Auto-detected when not specified.",
    )

    # ------------------------------------------------------------------ #
    # Generation parameters                                                #
    # ------------------------------------------------------------------ #
    parser.add_argument("--num_step", type=int, default=32,
                        help="Number of diffusion decoding steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speaking speed factor (>1 faster, <1 slower).")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Fixed output duration in seconds. Overrides --speed when set.",
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument("--postprocess_output", type=str2bool, default=True)
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


def _output_path(base: str, index: int, total: int) -> str:
    """Return a zero-padded output path for multi-sentence mode.

    For a single sentence the original path is returned unchanged.
    For multiple sentences 'out.wav' becomes 'out_001.wav', etc.
    """
    if total == 1:
        return base
    root, ext = os.path.splitext(base)
    width = max(3, len(str(total)))
    return f"{root}_{str(index + 1).zfill(width)}{ext}"


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    # ------------------------------------------------------------------ #
    # Collect sentences                                                    #
    # ------------------------------------------------------------------ #
    if args.sentences is not None:
        with open(args.sentences, "r", encoding="utf-8") as fh:
            sentences = [
                line.rstrip("\n")
                for line in fh
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if not sentences:
            raise ValueError(f"No sentences found in {args.sentences!r}")
    else:
        sentences = args.texts  # list from repeated --text flags

    total = len(sentences)
    logging.info(f"Total sentences to synthesize: {total}")

    # ------------------------------------------------------------------ #
    # Determine whether ASR is needed                                      #
    # ------------------------------------------------------------------ #
    needs_asr = (
        args.ref_audio is not None
        and args.ref_text is None
        and not args.no_asr
    )

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    device = args.device or get_best_device()
    logging.info(f"Loading model from {args.model!r} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.float16,
        load_asr=needs_asr,
        asr_model_name=args.asr_model,
    )

    # ------------------------------------------------------------------ #
    # Build voice-clone prompt once (reused across all sentences)          #
    # ------------------------------------------------------------------ #
    voice_clone_prompt = None
    if args.ref_audio is not None:
        logging.info(
            "Building voice-clone prompt from %r%s",
            args.ref_audio,
            " (ref_text will be auto-transcribed)" if args.ref_text is None else "",
        )
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,  # None → auto-transcribe via Whisper
        )
        logging.info("Voice-clone prompt ready. ref_text: %r", voice_clone_prompt.ref_text)

    # ------------------------------------------------------------------ #
    # Generate — one call per sentence so each gets its own file          #
    # ------------------------------------------------------------------ #
    for idx, text in enumerate(sentences):
        out_path = _output_path(args.output, idx, total)
        logging.info(
            "[%d/%d] Generating: %s → %s",
            idx + 1,
            total,
            repr(text[:80] + ("…" if len(text) > 80 else "")),
            out_path,
        )

        generate_kwargs = dict(
            text=text,
            language=args.language,
            voice_clone_prompt=voice_clone_prompt,
            instruct=args.instruct if voice_clone_prompt is None else None,
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
        if args.duration is not None:
            generate_kwargs["duration"] = args.duration

        audios = model.generate(**generate_kwargs)

        # Ensure output directory exists
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        sf.write(out_path, audios[0], model.sampling_rate)
        logging.info("Saved → %s", out_path)

    logging.info("Done. %d file(s) written.", total)


if __name__ == "__main__":
    main()
