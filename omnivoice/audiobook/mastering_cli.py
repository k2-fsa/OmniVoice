from __future__ import annotations

import argparse
from pathlib import Path

from omnivoice.audiobook.mastering import ConcatOptions, MasteringOptions, concat_audio_files, remaster_audio


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concatenate and remaster audiobook audio with FFmpeg.")
    parser.add_argument("--input", action="append", required=True, help="Input audio path. Repeat in order.")
    parser.add_argument("--concat-output", help="Optional concatenated intermediate output.")
    parser.add_argument("--concat-copy", action="store_true", help="Use concat -c copy instead of WAV normalization.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tempo", type=float, default=1.0)
    parser.add_argument("--target-lufs", type=float, default=-20.0)
    parser.add_argument("--true-peak", type=float, default=-3.0)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--dynamic-normalize", action="store_true")
    parser.add_argument("--compressor", action="store_true")
    parser.add_argument("--no-limiter", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    inputs = [Path(item) for item in args.input]
    output = Path(args.output)
    if len(inputs) > 1:
        concat_output = Path(args.concat_output) if args.concat_output else output.with_suffix(".concat.wav")
        source = concat_audio_files(
            inputs,
            concat_output,
            options=ConcatOptions(normalize_stream=not args.concat_copy, overwrite=args.overwrite),
        )
    else:
        source = inputs[0]

    remaster_audio(
        source,
        output,
        MasteringOptions(
            tempo=args.tempo,
            target_lufs=args.target_lufs,
            true_peak=args.true_peak,
            trim_silence=args.trim_silence,
            dynamic_normalize=args.dynamic_normalize,
            compressor=args.compressor,
            limiter=not args.no_limiter,
            overwrite=args.overwrite,
        ),
    )
    print(f"Wrote remastered audiobook audio: {output}")


if __name__ == "__main__":
    main()
