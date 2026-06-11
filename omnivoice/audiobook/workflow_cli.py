from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnivoice.audiobook.docx import extract_docx_structure
from omnivoice.audiobook.generation import (
    AudiobookGenerationJob,
    load_generation_checkpoint,
    write_generation_checkpoint,
)
from omnivoice.audiobook.planner import (
    AudiobookPlanConfig,
    create_audiobook_plan_from_openrouter_results,
    write_plan,
)


def _add_plan_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", default="")
    parser.add_argument("--language", default="pt-BR")
    parser.add_argument("--genre", choices=["technical", "fiction"], default="technical")
    parser.add_argument("--speed", type=float, default=0.92)
    parser.add_argument("--preset", choices=["Natural", "Presentation", "Manual"], default="Presentation")
    parser.add_argument("--model", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate OmniVoice audiobook planning and generation checkpoints.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    merge = subcommands.add_parser("merge-openrouter", help="Merge structured OpenRouter chunk JSON files into a plan.")
    merge.add_argument("--docx", required=True)
    merge.add_argument("--result", action="append", required=True, help="Structured chunk JSON. Repeat in order.")
    merge.add_argument("--output", required=True)
    _add_plan_config_args(merge)

    status = subcommands.add_parser("status", help="Print generation progress for a plan/checkpoint.")
    status.add_argument("--plan", required=True)

    next_segment = subcommands.add_parser("next", help="Print the next pending/failed segment.")
    next_segment.add_argument("--plan", required=True)
    next_segment.add_argument("--include-text", action="store_true", help="Include manuscript text in local output.")

    mark_generated = subcommands.add_parser("mark-generated", help="Mark a segment generated and write checkpoint.")
    mark_generated.add_argument("--plan", required=True)
    mark_generated.add_argument("--segment-id", required=True)
    mark_generated.add_argument("--audio-path", required=True)
    mark_generated.add_argument("--output", required=True)

    mark_failed = subcommands.add_parser("mark-failed", help="Mark a segment failed and write checkpoint.")
    mark_failed.add_argument("--plan", required=True)
    mark_failed.add_argument("--segment-id", required=True)
    mark_failed.add_argument("--error", required=True)
    mark_failed.add_argument("--output", required=True)

    return parser


def _load_results(paths: list[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"OpenRouter result must be a JSON object: {path}")
        results.append(data)
    return results


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "merge-openrouter":
        document = extract_docx_structure(args.docx)
        plan = create_audiobook_plan_from_openrouter_results(
            document,
            AudiobookPlanConfig(
                title=args.title,
                author=args.author,
                language=args.language,
                genre=args.genre,
                speed=args.speed,
                preset=args.preset,
            ),
            _load_results(args.result),
            model=args.model,
        )
        output = write_plan(plan, args.output)
        print(f"Wrote merged OpenRouter audiobook plan: {output}")
        return

    plan = load_generation_checkpoint(Path(args.plan))
    job = AudiobookGenerationJob(plan)

    if args.command == "status":
        print(json.dumps(job.progress(), ensure_ascii=False, indent=2))
        return
    if args.command == "next":
        item = job.next_segment()
        if item and not args.include_text:
            data = {
                "chapter_id": item.chapter_id,
                "segment_id": item.segment_id,
                "status": item.status,
                "audio_path": item.audio_path,
                "text_redacted": True,
            }
        else:
            data = item.__dict__ if item else None
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if args.command == "mark-generated":
        job.mark_generated(args.segment_id, args.audio_path)
        write_generation_checkpoint(job.plan, Path(args.output))
        print(f"Wrote generation checkpoint: {args.output}")
        return
    if args.command == "mark-failed":
        job.mark_failed(args.segment_id, args.error)
        write_generation_checkpoint(job.plan, Path(args.output))
        print(f"Wrote generation checkpoint: {args.output}")
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
