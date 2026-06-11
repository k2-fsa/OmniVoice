from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omnivoice.audiobook.chunking import ChunkingConfig, chunk_docx_document
from omnivoice.audiobook.docx import extract_docx_structure
from omnivoice.audiobook.openrouter import OpenRouterAudiobookClient, OpenRouterConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Structure one DOCX audiobook chunk through OpenRouter.",
    )
    parser.add_argument("--docx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--language", default="pt-BR")
    parser.add_argument("--genre", choices=["technical", "fiction"], default="technical")
    parser.add_argument("--max-words", type=int, default=2400)
    parser.add_argument("--target-words", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--response-healing", action="store_true")
    parser.add_argument(
        "--confirm-online-provider",
        action="store_true",
        help="Required for real OpenRouter calls. Confirms this chunk may be sent to OpenRouter.",
    )
    parser.add_argument("--skip-model-support-check", action="store_true")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write the selected chunk metadata and text without calling OpenRouter.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include manuscript text in preview output. Preview metadata is redacted by default.",
    )
    return parser


def _chunk_preview(chunk, *, include_text: bool) -> dict[str, object]:
    data: dict[str, object] = {
        "id": chunk.id,
        "index": chunk.index,
        "title": chunk.title,
        "word_count": chunk.word_count,
        "paragraph_start": chunk.paragraph_start,
        "paragraph_end": chunk.paragraph_end,
        "warnings": chunk.warnings,
    }
    if include_text:
        data["text"] = chunk.text
        data["previous_summary"] = chunk.previous_summary
    else:
        data["text_redacted"] = True
        data["previous_summary_redacted"] = chunk.previous_summary is not None
    return data


def main() -> None:
    args = _build_parser().parse_args()
    if not args.model:
        raise SystemExit("--model or OPENROUTER_MODEL is required")
    document = extract_docx_structure(args.docx)
    chunks = chunk_docx_document(
        document,
        ChunkingConfig(max_words=args.max_words, target_words=args.target_words),
    )
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise SystemExit(f"chunk-index out of range. available=0..{len(chunks) - 1}")
    chunk = chunks[args.chunk_index]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.preview_only:
        output.write_text(
            json.dumps(
                {
                    "chunk": _chunk_preview(chunk, include_text=args.include_text),
                    "total_chunks": len(chunks),
                    "provider_call": False,
                    "text_redacted": not args.include_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote OpenRouter chunk preview: {output}")
        return

    client = OpenRouterAudiobookClient(
        OpenRouterConfig(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            response_healing=args.response_healing,
            require_model_support=not args.skip_model_support_check,
        )
    )
    result = client.structure_chunk(
        chunk,
        language=args.language,
        genre=args.genre,
        consent=args.confirm_online_provider,
    )
    output.write_text(
        json.dumps(result.content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote OpenRouter structured chunk: {output}")


if __name__ == "__main__":
    main()
