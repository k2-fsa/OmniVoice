"""Reproducible text benchmark for Vietnamese normalization systems.

Accepts this experiment's ``gold`` schema and the earlier pilot schema whose
references are in ``preferred_spoken``/``acceptable_spoken``. It never rewrites
its input and can emit the IDs whose normalized text changed for later audio
regeneration.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from omnivoice.utils.text import _num2words_segment, normalize_text

_AUDITED_CORRECTIONS = {
    "time_duration_009": {
        "old": "Cửa hàng đóng cửa lúc hai mươi ba giờ năm chín phút.",
        "corrected": "Cửa hàng đóng cửa lúc hai mươi ba giờ năm mươi chín phút.",
        "reason": "59 is a clock minute and must be read as the cardinal fifty-nine.",
    }
}


def load_records(path: Path) -> List[dict]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value["records"] if isinstance(value, dict) else value
    else:
        records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    for index, record in enumerate(records, 1):
        required = {"id", "raw_text", "role"}
        if not required <= record.keys() or not (
            "gold" in record or "preferred_spoken" in record
        ):
            raise ValueError(f"{path}:record {index}: unsupported benchmark schema")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark IDs must be unique")
    return records


def references(record: dict) -> List[str]:
    correction = _AUDITED_CORRECTIONS.get(record["id"])
    if correction and correction["old"] in record.get("preferred_spoken", ()):
        return [correction["corrected"]]
    if "gold" in record:
        return [record["gold"]]
    return list(record.get("preferred_spoken", ())) + list(
        record.get("acceptable_spoken", ())
    )


def _fallback(text: str) -> str:
    return _num2words_segment(text, "vi")


def evaluate(records: Iterable[dict]) -> dict:
    records = list(records)
    report = {
        "cases": len(records),
        "evaluation_status": "development cases; not a held-out generalization estimate",
        "systems": {
            name: {
                "correct": 0,
                "unchanged": 0,
                "per_class": defaultdict(lambda: [0, 0]),
            }
            for name in ("unchanged_text", "num2words_fallback", "contextual_prototype")
        },
        "comparisons": [],
        "label_audit_issues": [
            {"id": record["id"], **_AUDITED_CORRECTIONS[record["id"]]}
            for record in records
            if record["id"] in _AUDITED_CORRECTIONS
            and _AUDITED_CORRECTIONS[record["id"]]["old"]
            in record.get("preferred_spoken", ())
        ],
    }
    for record in records:
        raw = record["raw_text"]
        expected = references(record)
        candidate_output = normalize_text(raw, "vi")
        outputs = {
            "unchanged_text": raw,
            "num2words_fallback": _fallback(raw),
            "contextual_prototype": candidate_output,
        }
        for name, output in outputs.items():
            stats = report["systems"][name]
            ok = output in expected
            stats["correct"] += ok
            stats["unchanged"] += output == raw
            stats["per_class"][record["role"]][0] += ok
            stats["per_class"][record["role"]][1] += 1
        report["comparisons"].append(
            {
                "id": record["id"],
                "input_text": raw,
                "expected_text": expected,
                "unchanged_text": raw,
                "baseline_text": outputs["num2words_fallback"],
                "prototype_text": candidate_output,
                "dataset_class": record["role"],
                "prototype_classes": [],
                "prototype_changed": candidate_output != raw,
                "prototype_exact_match": candidate_output in expected,
                "uncertain_or_unsupported": candidate_output == raw,
                "decisions": [],
            }
        )
    for name, stats in report["systems"].items():
        per_class = stats.pop("per_class")
        stats["exact_match"] = {"correct": stats.pop("correct"), "total": len(records)}
        stats["unchanged"] = {"count": stats["unchanged"], "total": len(records)}
        stats["per_class"] = {
            role: {"correct": value[0], "total": value[1]}
            for role, value in sorted(per_class.items())
        }
    report["prototype_diagnostics"] = {
        "changed_cases": sum(
            item["prototype_changed"] for item in report["comparisons"]
        ),
        "uncertain_or_unsupported_cases": sum(
            item["uncertain_or_unsupported"] for item in report["comparisons"]
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write Colab JSONL for cases where baseline and prototype differ",
    )
    parser.add_argument(
        "--old-variants",
        type=Path,
        help="Optional earlier diagnostic JSONL containing raw audio paths",
    )
    args = parser.parse_args()
    records = load_records(args.dataset)
    report = evaluate(records)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    if args.manifest:
        old_audio = {}
        if args.old_variants:
            with args.old_variants.open(encoding="utf-8") as stream:
                for line in stream:
                    variant = json.loads(line)
                    if variant.get("system") == "raw":
                        old_audio[variant["id"]] = variant.get("audio_path")
        with args.manifest.open("w", encoding="utf-8") as stream:
            by_id = {record["id"]: record for record in records}
            for item in report["comparisons"]:
                if item["baseline_text"] == item["prototype_text"]:
                    continue
                source = by_id[item["id"]]
                row = {
                    "case_id": item["id"],
                    "original_text": item["input_text"],
                    "baseline_text": item["baseline_text"],
                    "prototype_text": item["prototype_text"],
                    "semantic_class": item["prototype_classes"]
                    or [item["dataset_class"]],
                    "expected_text": item["expected_text"],
                    "old_audio_path": source.get("audio_path")
                    or old_audio.get(item["id"]),
                    "destination_path": f"changed/{item['id']}_prototype.wav",
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
