"""Evaluate candidate-span coverage on an explicitly dev-only JSONL file."""

import argparse
import json
from collections import Counter
from pathlib import Path

from omnivoice.text_normalization import get_bamibert_detector

from .schema import load_cases


def analyze(path: Path) -> dict:
    if "held" in path.name.lower() or "test" in path.name.lower():
        raise ValueError("refusing a held-out/test-looking input; pass dev-only JSONL")
    cases = load_cases(path)
    detector = get_bamibert_detector()
    counts = Counter()
    errors = []
    for case in cases:
        predicted = list(detector(case.text))
        exact = {(span.start, span.end) for span in predicted}
        for gold in case.spans:
            counts["gold_spans"] += 1
            if (gold.start, gold.end) in exact:
                counts["exact_boundary"] += 1
                continue
            covering = [
                span
                for span in predicted
                if span.start <= gold.start and span.end >= gold.end
            ]
            kind = "covered_wrong_boundary" if covering else "missed"
            counts[kind] += 1
            errors.append(
                {
                    "id": case.id,
                    "gold": as_span(gold),
                    "kind": kind,
                    "predicted": [as_span(span) for span in predicted],
                }
            )
    total = counts["gold_spans"]
    counts["cases"] = len(cases)
    return {
        "summary": dict(counts),
        "exact_boundary_recall": counts["exact_boundary"] / total if total else 1.0,
        "coverage_recall": (
            counts["exact_boundary"] + counts["covered_wrong_boundary"]
        )
        / total
        if total
        else 1.0,
        "errors": errors,
    }


def as_span(span) -> dict:
    return {"start": span.start, "end": span.end, "surface": span.text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.dev_jsonl)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
