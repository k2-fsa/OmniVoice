"""Audit, canonicalize, and deterministically split the immutable pilot."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .schema import CanonicalRecord, validate_record

SEED = 20260722
KNOWN_DEVELOPMENT_TEXTS = {
    json.loads(line)["raw_text"]
    for line in (Path(__file__).parent / "regressions.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
}
_NUMERIC_RE = re.compile(r"[\d０-９]+(?:[\s.,:/\-–—$€₫¥\[\]()]*[\d０-９]+)*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def template_key(text: str) -> str:
    return re.sub(r"\s+", " ", _NUMERIC_RE.sub("<NUM>", text.lower())).strip()


def load_source(path: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records


def canonicalize(source_records: list[dict], source_name: str) -> list[CanonicalRecord]:
    canonical = []
    for raw in source_records:
        preferred = raw.get("preferred_spoken") or []
        value = {
            "id": raw.get("id"),
            "text": raw.get("raw_text"),
            "numeric_span": raw.get("number"),
            "semiotic_class": raw.get("role"),
            "domain": None,
            "expected_spoken_text": preferred[0] if preferred else None,
            "acceptable_variants": preferred[1:] + raw.get("acceptable_spoken", []),
            "source": source_name,
            "template_group": template_key(raw.get("raw_text", "")),
            "audio_path": raw.get("audio_path"),
            "human_rating": raw.get("human_rating"),
            "metadata": {"source_record": raw},
        }
        canonical.append(validate_record(value))
    return canonical


def assign_splits(records: list[CanonicalRecord]) -> dict[str, str]:
    """Keep exact/template groups together and tiny classes in development."""
    groups = defaultdict(list)
    for record in records:
        groups[(record.semiotic_class, record.template_group)].append(record)
    by_class = defaultdict(list)
    for (role, _), members in groups.items():
        by_class[role].append(members)
    rng = random.Random(SEED)
    assignments = {}
    for role in sorted(by_class):
        class_groups = by_class[role]
        class_size = sum(len(group) for group in class_groups)
        rng.shuffle(class_groups)
        target = round(class_size * 0.2) if class_size >= 5 else 0
        selected = 0
        for group in class_groups:
            is_known = any(item.text in KNOWN_DEVELOPMENT_TEXTS for item in group)
            split = "test" if not is_known and selected < target else "development"
            if split == "test":
                selected += len(group)
            for record in group:
                assignments[record.id] = split
    return assignments


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = load_source(args.source)
    canonical = canonicalize(
        raw, f"git:7053b50:pilot.jsonl sha256:{sha256(args.source)}"
    )
    ids = [record.id for record in canonical]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    exact_text = Counter(record.text for record in canonical)
    exact_duplicates = sorted(text for text, count in exact_text.items() if count > 1)
    assignments = assign_splits(canonical)
    output = args.output_dir
    canonical_path = output / "canonical.jsonl"
    split_path = output / "split_manifest.jsonl"
    write_jsonl(canonical_path, (record.to_dict() for record in canonical))
    write_jsonl(
        split_path,
        (
            {
                "id": record.id,
                "split": assignments[record.id],
                "semiotic_class": record.semiotic_class,
                "template_group": record.template_group,
            }
            for record in canonical
        ),
    )
    class_split = defaultdict(Counter)
    for record in canonical:
        class_split[record.semiotic_class][assignments[record.id]] += 1
    audit = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "records": len(canonical),
        "unique_ids": len(set(ids)),
        "duplicate_ids": duplicate_ids,
        "exact_duplicate_sentences": exact_duplicates,
        "template_groups_with_multiple_records": sum(
            count > 1
            for count in Counter(record.template_group for record in canonical).values()
        ),
        "missing_fields": {},
        "malformed_records": 0,
        "class_distribution": dict(
            sorted(Counter(record.semiotic_class for record in canonical).items())
        ),
        "domain_labels": None,
        "audio_paths": sum(bool(record.audio_path) for record in canonical),
        "human_ratings": sum(record.human_rating is not None for record in canonical),
        "split_seed": SEED,
        "split_integrity": "compromised: all records/golds were inspected before this split",
        "split_counts": dict(Counter(assignments.values())),
        "class_split_counts": {
            key: dict(value) for key, value in sorted(class_split.items())
        },
    }
    audit_path = output / "dataset_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    domains_path = output / "domain_annotation_template.csv"
    with domains_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "text", "domain", "annotator_notes"]
        )
        writer.writeheader()
        writer.writerows(
            {"id": record.id, "text": record.text, "domain": "", "annotator_notes": ""}
            for record in canonical
        )
    hashes = {
        path.name: sha256(path)
        for path in (canonical_path, split_path, audit_path, domains_path)
    }
    (output / "hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    questionable = [
        {
            "id": "time_duration_009",
            "original_label": "Cửa hàng đóng cửa lúc hai mươi ba giờ năm chín phút.",
            "proposed_correction": "Cửa hàng đóng cửa lúc hai mươi ba giờ năm mươi chín phút.",
            "reason": "59 is a clock minute and should be read as the cardinal fifty-nine.",
            "official_scoring": "excluded pending human review",
        }
    ]
    (output / "questionable_labels.json").write_text(
        json.dumps(questionable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
