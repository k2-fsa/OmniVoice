#!/usr/bin/env python3
"""Try the Vietnamese text normalizer with a lazily loaded BamiBERT detector."""

import argparse
import sys

from omnivoice.text_normalization import (
    DEFAULT_MODEL_PATH,
    get_bamibert_detector,
    normalize_candidates,
)
from omnivoice.text_normalization.types import Diagnostic

def build_detector(model_path=None, device="cpu"):
    """Obtain the shared lazy detector used by production inference."""
    return get_bamibert_detector(model_path, device)


def _format_candidate(candidate) -> str:
    return (
        f"label={candidate.label}, "
        f"span=({candidate.start}, {candidate.end}), "
        f"surface={candidate.text!r}, score={candidate.score}"
    )


def _format_diagnostic(diagnostic: Diagnostic) -> str:
    parts = [
        f"action={diagnostic.action}",
        f"reason={diagnostic.reason}",
    ]
    if diagnostic.effective_label is not None:
        parts.append(f"effective_label={diagnostic.effective_label}")
    if diagnostic.replacement is not None:
        parts.append(f"replacement={diagnostic.replacement!r}")
    if diagnostic.value is not None:
        parts.append(f"value={diagnostic.value!r}")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Try Vietnamese text normalization with BamiBERT candidates")
    parser.add_argument("text", help="Text to normalize")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the BamiBERT model directory",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Detector device understood by Transformers (default: cpu)",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Suppress printing structured diagnostics",
    )
    args = parser.parse_args(argv)

    text = args.text
    print(f"Original text: {text}")

    try:
        detector = build_detector(args.model_path, args.device)
        candidates = detector(text)
        print("Raw NER candidates:")
        if candidates:
            for candidate in candidates:
                print(f"  - {_format_candidate(candidate)}")
        else:
            print("  - <none>")

        result = normalize_candidates(text, candidates)
        print(f"Normalized text: {result.text}")
        if not args.no_diagnostics:
            print("Structured diagnostics:")
            if result.diagnostics:
                for diagnostic in result.diagnostics:
                    print(f"  - {_format_diagnostic(diagnostic)}")
            else:
                print("  - <none>")
    except Exception as error:  # outer CLI boundary preserves the complete input
        print("Raw NER candidates:")
        print("  - <none>")
        print(f"Normalized text: {text}")
        if not args.no_diagnostics:
            print("Structured diagnostics:")
            print(f"  - action=preserved, reason=script failure: {error}")

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
