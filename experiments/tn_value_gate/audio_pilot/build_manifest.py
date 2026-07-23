"""Create reproducibly blinded RAW/BEST_EXISTING/GOLD audio manifests."""

import argparse
import csv
import random
from pathlib import Path

SEED = 20260722


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    with args.selection.open(encoding="utf-8", newline="") as stream:
        selected = list(csv.DictReader(stream))
    conditions = []
    for row in selected:
        for condition, field in (("RAW", "original_text"), ("BEST_EXISTING", "best_existing_text"),
                                 ("GOLD", "preferred_gold")):
            conditions.append({"source_case_id": row["id"], "condition": condition,
                               "target_text": row[field], "role": row["role"],
                               "best_existing_system": row["best_existing_system"]})
    random.Random(SEED).shuffle(conditions)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    blind_path = args.artifacts / "audio_blind_manifest.csv"
    key_path = args.artifacts / "audio_condition_key.csv"
    rating_path = args.artifacts / "audio_rating_sheet.csv"
    with blind_path.open("w", encoding="utf-8", newline="") as blind, key_path.open(
        "w", encoding="utf-8", newline=""
    ) as key, rating_path.open("w", encoding="utf-8", newline="") as rating:
        blind_writer = csv.DictWriter(blind, fieldnames=["blind_sample_id", "source_case_id", "role", "audio_path"])
        key_fields = ["blind_sample_id", "source_case_id", "condition", "target_text", "role",
                      "best_existing_system", "audio_path"]
        key_writer = csv.DictWriter(key, fieldnames=key_fields)
        rating_fields = ["blind_sample_id", "source_case_id", "number_reading_correctness",
                         "naturalness_1_to_5", "intelligibility_1_to_5", "perceived_spoken_form", "comment"]
        rating_writer = csv.DictWriter(rating, fieldnames=rating_fields)
        blind_writer.writeheader(); key_writer.writeheader(); rating_writer.writeheader()
        for index, row in enumerate(conditions, 1):
            blind_id = f"S{index:03d}"; audio_path = f"audio/{blind_id}.wav"
            blind_writer.writerow({"blind_sample_id": blind_id, "source_case_id": row["source_case_id"],
                                   "role": row["role"], "audio_path": audio_path})
            key_writer.writerow({"blind_sample_id": blind_id, **row, "audio_path": audio_path})
            rating_writer.writerow({"blind_sample_id": blind_id, "source_case_id": row["source_case_id"],
                                     "number_reading_correctness": "", "naturalness_1_to_5": "",
                                     "intelligibility_1_to_5": "", "perceived_spoken_form": "", "comment": ""})
    print(f"{len(selected)} cases, {len(conditions)} blinded samples")


if __name__ == "__main__":
    main()
