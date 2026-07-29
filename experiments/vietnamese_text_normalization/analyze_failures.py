"""Group benchmark failures into coarse, explicitly heuristic categories."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

UNSUPPORTED_CLASSES = {
    "MEASURE",
    "MEASURE_RANGE",
    "PERCENT",
    "PERCENT_RANGE",
    "SIGNED_PERCENT",
    "POWER",
    "EQUATION",
    "SCIENTIFIC_NOTATION",
    "ORDINAL_ROMAN",
    "VERSION",
    "TEMPERATURE",
    "DIGITAL_STORAGE",
    "RANGE",
    "INTERVAL",
    "MEDICAL_RATIO",
}
CONTEXT_CLASSES = {
    "IDENTIFIER",
    "ORDER_ID",
    "ROOM",
    "PHONE",
    "EMERGENCY_PHONE",
    "SERVICE_PHONE",
    "ADDRESS",
    "APARTMENT",
    "LICENSE_PLATE",
    "SEAT",
    "BUS_ROUTE",
    "RATIO",
    "SCORE",
}


def category(row: dict) -> str:
    if not row["available"]:
        return "package/runtime failure"
    if not row["score_eligible"]:
        return "questionable gold label"
    role = row["semiotic_class"]
    if role in UNSUPPORTED_CLASSES:
        return "unsupported semiotic class"
    if role in CONTEXT_CLASSES:
        return "contextual ambiguity"
    if role in {"QUANTITY", "YEAR", "NEGATIVE_QUANTITY", "SIGNED_NUMBER"}:
        return "incorrect cardinal morphology"
    if role in {"DATE", "ISO_DATE", "TIME", "DURATION", "MONEY", "FRACTION"}:
        return "structural parsing failure"
    return "punctuation/span corruption or unsupported form"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
    ]
    failures = [row for row in rows if not row["acceptable_variant_match"]]
    counts = Counter((row["split"], row["method"], category(row)) for row in failures)
    examples = defaultdict(list)
    for row in failures:
        key = (row["split"], row["method"], category(row))
        if len(examples[key]) < 3:
            examples[key].append(
                {
                    "id": row["case_id"],
                    "input": row["original_text"],
                    "output": row["normalized_output"],
                    "expected": row["expected_spoken_text"],
                    "class": row["semiotic_class"],
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "failure_taxonomy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "method", "category", "count"])
        writer.writerows((*key, count) for key, count in sorted(counts.items()))
    report = {
        "warning": "categories are heuristic analysis labels, not audited dataset annotations",
        "counts": [
            {"split": key[0], "method": key[1], "category": key[2], "count": count}
            for key, count in sorted(counts.items())
        ],
        "representative_examples": [
            {"split": key[0], "method": key[1], "category": key[2], "examples": value}
            for key, value in sorted(examples.items())
        ],
    }
    (args.output_dir / "failure_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
