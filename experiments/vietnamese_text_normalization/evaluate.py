"""Run all available adapters and emit case-level and aggregate results."""

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .adapters import METHODS
from .schema import validate_record

SOE_UNAVAILABLE = (
    "Disabled: soe-vinorm 0.3.2 calls huggingface_hub.snapshot_download for "
    "vinhdq842/soe-vinorm CRF weights; model downloads are prohibited."
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_external(
    python: Path, runner: Path, method: str, records: list[dict]
) -> dict[str, dict]:
    payload = "".join(
        json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n"
        for row in records
    )
    completed = subprocess.run(
        [str(python), str(runner), method],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        error = f"exit {completed.returncode}: {completed.stderr.strip()}"
        return {
            row["id"]: {
                "output_text": "",
                "available": False,
                "changed": False,
                "uncertain": True,
                "error": error,
                "metadata": {},
            }
            for row in records
        }
    return {
        value["id"]: value for value in map(json.loads, completed.stdout.splitlines())
    }


def score(rows: list[dict]) -> dict:
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    report = {}
    for method, values in sorted(by_method.items()):
        eligible = [row for row in values if row["score_eligible"]]
        available = [row for row in eligible if row["available"]]
        exact = sum(row["exact_match"] for row in available)
        acceptable = sum(row["acceptable_variant_match"] for row in available)
        casefold = sum(row["casefold_variant_match"] for row in available)
        changed = sum(row["changed"] for row in available)
        abstained = sum(row["uncertain_or_unsupported"] for row in eligible)
        per_class = defaultdict(lambda: [0, 0])
        for row in available:
            per_class[row["semiotic_class"]][0] += row["acceptable_variant_match"]
            per_class[row["semiotic_class"]][1] += 1
        represented = [
            correct / total for correct, total in per_class.values() if total >= 2
        ]
        report[method] = {
            "total": len(values),
            "score_eligible": len(eligible),
            "excluded_questionable_gold": len(values) - len(eligible),
            "available": len(available),
            "unavailable_or_error": len(eligible) - len(available),
            "exact_match": [exact, len(available)],
            "acceptable_variant_match": [acceptable, len(available)],
            "casefold_variant_match_diagnostic": [casefold, len(available)],
            "changed": [changed, len(available)],
            "coverage": [len(eligible) - abstained, len(eligible)],
            "abstention_or_unsupported": [abstained, len(eligible)],
            "sentence_errors": [len(available) - acceptable, len(available)],
            "macro_accuracy_classes_n_ge_2": sum(represented) / len(represented)
            if represented
            else None,
            "per_class": {
                key: {"correct": value[0], "total": value[1]}
                for key, value in sorted(per_class.items())
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-python", type=Path)
    parser.add_argument("--old-variants", type=Path)
    args = parser.parse_args()
    records = read_jsonl(args.canonical)
    for record in records:
        validate_record(record)
    split = {row["id"]: row["split"] for row in read_jsonl(args.splits)}
    external = {}
    if args.external_python:
        runner = Path(__file__).parent / "external_runner.py"
        external["vietnormalizer"] = run_external(
            args.external_python, runner, "vietnormalizer", records
        )
    external["soe_vinorm"] = {
        row["id"]: {
            "output_text": "",
            "available": False,
            "changed": False,
            "uncertain": True,
            "error": SOE_UNAVAILABLE,
            "metadata": {"version": "0.3.2"},
        }
        for row in records
    }
    results = []
    for record in records:
        expected = record["expected_spoken_text"]
        variants = record["acceptable_variants"]
        method_results = {
            name: adapter(record["text"], "vi").to_dict()
            for name, adapter in METHODS.items()
        }
        method_results.update(
            {name: values[record["id"]] for name, values in external.items()}
        )
        for method, result in method_results.items():
            output = result["output_text"]
            results.append(
                {
                    "case_id": record["id"],
                    "split": split[record["id"]],
                    "original_text": record["text"],
                    "numeric_span": record["numeric_span"],
                    "semiotic_class": record["semiotic_class"],
                    "domain": record["domain"],
                    "expected_spoken_text": expected,
                    "acceptable_variants": variants,
                    "method": method,
                    "score_eligible": record["id"] != "time_duration_009",
                    "method_version": result.get("metadata", {}).get("version"),
                    "normalized_output": output,
                    "available": result["available"],
                    "changed": result["changed"],
                    "exact_match": result["available"] and output == expected,
                    "acceptable_variant_match": result["available"]
                    and output in [expected, *variants],
                    "casefold_variant_match": result["available"]
                    and output.casefold()
                    in [expected.casefold(), *(item.casefold() for item in variants)],
                    "uncertain_or_unsupported": result["uncertain"],
                    "runtime_error": result["error"],
                    "source_metadata": record["metadata"],
                    "method_metadata": result.get("metadata", {}),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "case_results.jsonl"
    with results_path.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        part: score([row for row in results if row["split"] == part])
        for part in ("development", "test")
    }
    summary["integrity_warning"] = (
        "test is mechanically split but contaminated by prior inspection/evaluation"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = Counter(
        (row["method"], row["semiotic_class"])
        for row in results
        if row["available"] and not row["acceptable_variant_match"]
    )
    with (args.output_dir / "failure_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "semiotic_class", "failures"])
        writer.writerows((*key, value) for key, value in sorted(failures.items()))
    old_audio = {}
    if args.old_variants:
        for value in read_jsonl(args.old_variants):
            if value.get("system") == "raw":
                old_audio[value["id"]] = value.get("audio_path")
    indexed = defaultdict(dict)
    for row in results:
        indexed[row["case_id"]][row["method"]] = row
    with (args.output_dir / "colab_changed_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for case_id, methods in sorted(indexed.items()):
            baseline, rule = methods["omnivoice_fallback"], methods["contextual_rule"]
            if baseline["normalized_output"] == rule["normalized_output"]:
                continue
            stream.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "original_text": rule["original_text"],
                        "existing_baseline_text": baseline["normalized_output"],
                        "rule_baseline_text": rule["normalized_output"],
                        "semantic_class": rule["semiotic_class"],
                        "expected_text": rule["expected_spoken_text"],
                        "split": rule["split"],
                        "uncertainty": rule["uncertain_or_unsupported"],
                        "old_audio_path": old_audio.get(case_id),
                        "destination_path": f"changed/{case_id}_rule.wav",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
