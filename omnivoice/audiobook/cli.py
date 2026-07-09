from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnivoice.audiobook.offline_audit import audit_offline_runtime
from omnivoice.audiobook.planner import (
    AudiobookPlanConfig,
    create_audiobook_plan_from_docx,
    write_plan,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a local OmniVoice audiobook JSON plan from a DOCX manuscript.",
    )
    parser.add_argument("--docx", required=True, help="Path to the source DOCX manuscript.")
    parser.add_argument("--output", required=True, help="Path for the generated audiobook JSON plan.")
    parser.add_argument("--title", required=True, help="Audiobook title.")
    parser.add_argument("--author", default="", help="Author name.")
    parser.add_argument("--language", default="pt-BR", help="Language code, default pt-BR.")
    parser.add_argument("--genre", choices=["technical", "fiction"], default="technical")
    parser.add_argument("--speed", type=float, default=0.92)
    parser.add_argument("--preset", choices=["Natural", "Presentation", "Manual"], default="Presentation")
    parser.add_argument("--voice-mode", choices=["design", "clone"], default="design")
    parser.add_argument("--default-voice", default="narrator")
    parser.add_argument("--max-segment-chars", type=int, default=900)
    parser.add_argument("--target-words-per-minute", type=int)
    parser.add_argument(
        "--offline-audit",
        action="store_true",
        help="Print offline runtime audit before creating the plan.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.offline_audit:
        audit = audit_offline_runtime()
        print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
        if not audit.passed:
            raise SystemExit(2)

    config = AudiobookPlanConfig(
        title=args.title,
        author=args.author,
        language=args.language,
        genre=args.genre,
        voice_mode=args.voice_mode,
        default_voice=args.default_voice,
        speed=args.speed,
        preset=args.preset,
        max_segment_chars=args.max_segment_chars,
        target_words_per_minute=args.target_words_per_minute,
    )
    plan = create_audiobook_plan_from_docx(Path(args.docx), config)
    output = write_plan(plan, args.output)
    print(f"Wrote audiobook plan: {output}")


if __name__ == "__main__":
    main()
