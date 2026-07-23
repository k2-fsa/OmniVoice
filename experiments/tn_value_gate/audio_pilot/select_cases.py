"""Select a deterministic, diverse 24-case audio value-gate pilot."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260722
HIGH_RISK = {
    "ROOM", "IDENTIFIER", "ORDER_ID", "NATIONAL_ID", "PHONE", "EMERGENCY_PHONE",
    "SERVICE_PHONE", "FRACTION", "MEDICAL_RATIO", "RATIO", "SCORE", "INTERVAL",
    "ADDRESS", "LICENSE_PLATE", "VERSION", "DECIMAL",
}
CONTROLS = {"QUANTITY", "DATE", "TIME", "MONEY", "YEAR", "MEASURE"}


def stable_shuffle(values: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    result = list(values); rng.shuffle(result)
    return result


def round_robin(values: list[dict], count: int) -> list[dict]:
    groups = defaultdict(list)
    for value in values: groups[value["role"]].append(value)
    selected = []
    while len(selected) < count and groups:
        for role in sorted(list(groups)):
            if groups[role]: selected.append(groups[role].pop(0))
            if not groups[role]: del groups[role]
            if len(selected) == count: break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--best-existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    best = json.loads(args.best_existing.read_text(encoding="utf-8"))["system"]
    values = [json.loads(line) for line in args.outputs.read_text(encoding="utf-8").splitlines()]
    by_id = defaultdict(dict)
    for value in values: by_id[value["id"]][value["system"]] = value
    candidates = []
    for case_id, methods in by_id.items():
        current, existing = methods["current_rule"], methods[best]
        if existing["output_text"] == current["preferred_gold"]:
            continue
        candidates.append({"id": case_id, "original_text": current["original_text"],
                           "preferred_gold": current["preferred_gold"], "role": current["role"],
                           "best_existing_system": best, "best_existing_text": existing["output_text"],
                           "current_rule_text": current["output_text"],
                           "cluster": "cluster_unknown",
                           "systems_disagree": existing["output_text"] != current["output_text"]})
    candidates = stable_shuffle(candidates, SEED)
    risk = [value for value in candidates if value["role"] in HIGH_RISK]
    controls = [value for value in candidates if value["role"] in CONTROLS]
    selected = round_robin(risk, 18) + round_robin(controls, 6)
    if len(selected) < 24:
        used = {value["id"] for value in selected}
        selected += [value for value in candidates if value["id"] not in used][:24 - len(selected)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        fields = ["selection_order", "id", "original_text", "preferred_gold", "role", "cluster",
                  "best_existing_system", "best_existing_text", "current_rule_text",
                  "systems_disagree", "selection_rationale"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for index, value in enumerate(selected, 1):
            rationale = "contextual/high-risk disagreement" if value["role"] in HIGH_RISK else "deterministic control"
            writer.writerow({"selection_order": index, **value, "selection_rationale": rationale})
    print(json.dumps({"selected": len(selected), "best_existing": best,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
