"""Run the complete frozen-dataset text value gate."""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .adapters import current_rule, num2words_vi
from .evaluation.canonicalize import canonicalize
from .evaluation.metrics import (
    latency_summary,
    mcnemar_exact,
    normalized_cer,
    paired_bootstrap,
    wilson,
)
from .evaluation.report import write_summary_csv, write_value_gate

SYSTEM_ORDER = ("current_rule", "num2words_vi", "vietnormalizer", "soe_vinorm")
CORRECTED_GOLD = {
    "time_duration_009": {
        "original": "Cửa hàng đóng cửa lúc hai mươi ba giờ năm chín phút.",
        "corrected": "Cửa hàng đóng cửa lúc hai mươi ba giờ năm mươi chín phút.",
        "reason": "59 is a clock minute; digit-by-digit 'năm chín' is not the intended reading.",
    }
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"id", "raw_text", "number", "role", "preferred_spoken"}
    seen = set()
    for index, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"record {index} missing {sorted(missing)}")
        if row["id"] in seen:
            raise ValueError(f"duplicate id: {row['id']}")
        seen.add(row["id"])
        if row["number"] not in row["raw_text"]:
            raise ValueError(f"numeric span absent for {row['id']}")
    return rows


def run_external(
    python: Path, worker: Path, method: str, rows: list[dict], env: dict
) -> dict[str, dict]:
    payload = "".join(
        json.dumps({"id": row["id"], "text": row["raw_text"]}, ensure_ascii=False)
        + "\n"
        for row in rows
    )
    module = ".".join(worker.with_suffix("").parts[-4:])
    completed = subprocess.run(
        [str(python), "-m", module, method],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    if completed.returncode:
        error = completed.stderr.strip() or f"exit {completed.returncode}"
        return {
            row["id"]: {
                "output_text": "",
                "supported": False,
                "status": "runtime_error",
                "error_type": "SubprocessError",
                "error_message": error,
                "latency_ms": 0.0,
                "metadata": {},
            }
            for row in rows
        }
    values = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    if len(values) != len(rows):
        raise RuntimeError(
            f"{method}: expected {len(rows)} outputs, received {len(values)}"
        )
    return {value["id"]: value for value in values}


def schema_audit(rows: list[dict]) -> dict:
    fields = set().union(*(row.keys() for row in rows))
    missing = {
        field: sum(field not in row or row[field] in (None, "", []) for row in rows)
        for field in sorted(fields)
    }
    return {
        "raw_text_field": "raw_text",
        "preferred_gold_field": "preferred_spoken[0]",
        "acceptable_gold_field": "preferred_spoken[1:] + acceptable_spoken",
        "semantic_type_field": "role",
        "reading_style_field": None,
        "semantic_group_field": None,
        "ambiguity_cluster_field": None,
        "split_field": None,
        "stored_rule_output_field": None,
        "missing_values": missing,
        "duplicate_ids": len(rows) - len({row["id"] for row in rows}),
        "record_count": len(rows),
        "role_count": len({row["role"] for row in rows}),
        "role_distribution": dict(sorted(Counter(row["role"] for row in rows).items())),
        "semantic_group_count": 0,
        "reading_style_count": 0,
        "dataset_designation": "pilot development set",
        "historical_115_of_160": "Historical 115/160 result could not be independently reproduced.",
        "historical_reason": "No historical split manifest and matching frozen executable/output were found.",
    }


def aggregate(outputs: list[dict]) -> dict:
    systems = {}
    for system in SYSTEM_ORDER:
        rows = [row for row in outputs if row["system"] == system]
        statuses = Counter(row["status"] for row in rows)
        attempted = sum(row["status"] in {"success", "partial"} for row in rows)
        strict = sum(row["strict_preferred_match"] for row in rows)
        canonical = sum(row["canonical_preferred_match"] for row in rows)
        acceptable = sum(row["acceptable_match"] for row in rows)
        latency = latency_summary([row["latency_ms"] for row in rows])
        systems[system] = {
            "records": len(rows),
            "attempted": attempted,
            "coverage": attempted / len(rows),
            "success": statuses["success"],
            "partial": statuses["partial"],
            "unsupported": statuses["unsupported"],
            "runtime_error": statuses["runtime_error"],
            "strict_preferred": strict,
            "canonical_preferred": canonical,
            "acceptable": acceptable,
            "strict_preferred_pair": [strict, len(rows)],
            "canonical_preferred_pair": [canonical, len(rows)],
            "acceptable_pair": [acceptable, len(rows)],
            "sentence_errors": len(rows) - acceptable,
            "conditional_canonical_preferred_pair": [
                sum(
                    row["canonical_preferred_match"]
                    for row in rows
                    if row["status"] in {"success", "partial"}
                ),
                attempted,
            ],
            **latency,
            "mean_normalized_cer": statistics.fmean(
                row["normalized_cer"] for row in rows
            ),
            "canonical_wilson_low": wilson(canonical, len(rows))[0],
            "canonical_wilson_high": wilson(canonical, len(rows))[1],
        }
    return systems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--external-python", type=Path, required=True)
    parser.add_argument(
        "--artifacts", type=Path, default=Path(__file__).parent / "artifacts"
    )
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.resolve()
    rows = load_dataset(dataset)
    manifest = {
        "absolute_path": str(dataset),
        "file_size_bytes": dataset.stat().st_size,
        "record_count": len(rows),
        "sha256": sha256(dataset),
        "source": "git 7053b50:experiments/vi_number_normalization/data/pilot.jsonl",
        "designation": "pilot development set",
    }
    (artifacts / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    audit = schema_audit(rows)
    (artifacts / "schema_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    worker = Path(__file__).parent / "adapters" / "external_worker.py"
    external_env = os.environ | {"HF_HOME": str(Path.home() / ".cache" / "huggingface")}
    external = {
        name: run_external(args.external_python, worker, name, rows, external_env)
        for name in ("vietnormalizer", "soe_vinorm")
    }
    outputs = []
    for row in rows:
        preferred_original = row["preferred_spoken"][0]
        correction = CORRECTED_GOLD.get(row["id"])
        preferred = correction["corrected"] if correction else preferred_original
        acceptable = row["preferred_spoken"][1:] + row.get("acceptable_spoken", [])
        results = {
            "current_rule": current_rule.normalize(row["raw_text"]).to_dict(),
            "num2words_vi": num2words_vi.normalize(row["raw_text"]).to_dict(),
            **{name: values[row["id"]] for name, values in external.items()},
        }
        for name in SYSTEM_ORDER:
            result = results[name]
            output = result["output_text"]
            canonical_output = canonicalize(output)
            canonical_gold = canonicalize(preferred)
            acceptable_canonical = [
                canonicalize(value) for value in [preferred, *acceptable]
            ]
            outputs.append(
                {
                    "id": row["id"],
                    "original_text": row["raw_text"],
                    "preferred_gold": preferred,
                    "source_preferred_gold": preferred_original,
                    "acceptable_gold": acceptable,
                    "role": row["role"],
                    "semantic_group": None,
                    "reading_style": None,
                    "ambiguity_cluster": "cluster_unknown",
                    "system": name,
                    "output_text": output,
                    **result,
                    "strict_preferred_match": output == preferred,
                    "canonical_preferred_match": canonical_output == canonical_gold,
                    "acceptable_match": canonical_output in acceptable_canonical,
                    "char_edit_distance": int(
                        round(
                            normalized_cer(canonical_output, canonical_gold)
                            * max(1, len(canonical_gold))
                        )
                    ),
                    "normalized_cer": normalized_cer(canonical_output, canonical_gold),
                    "gold_correction": correction,
                    "source_metadata": row,
                }
            )
    output_path = artifacts / "baseline_outputs.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for value in outputs:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    systems = aggregate(outputs)
    roles = {}
    for role in sorted({row["role"] for row in outputs}):
        roles[role] = aggregate([row for row in outputs if row["role"] == role])
    summary = {
        "dataset": manifest,
        "systems": systems,
        "limitations": [
            "pilot development set; rules were informed by data",
            "no audited ambiguity cluster/domain/reading_style metadata",
        ],
    }
    (artifacts / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (artifacts / "baseline_by_role.json").write_text(
        json.dumps(roles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (artifacts / "baseline_by_role.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "role",
            "system",
            "records",
            "attempted",
            "canonical_preferred",
            "acceptable",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role, role_systems in roles.items():
            for system, values in role_systems.items():
                writer.writerow(
                    {
                        "role": role,
                        "system": system,
                        **{field: values[field] for field in fields[2:]},
                    }
                )
    write_summary_csv(artifacts / "baseline_summary.csv", systems)
    comparisons = {}
    current_flags = [
        row["canonical_preferred_match"]
        for row in outputs
        if row["system"] == "current_rule"
    ]
    for name in ("num2words_vi", "vietnormalizer", "soe_vinorm"):
        flags = [
            row["canonical_preferred_match"] for row in outputs if row["system"] == name
        ]
        comparisons[name] = {
            "mcnemar_vs_current": mcnemar_exact(flags, current_flags),
            "bootstrap_current_minus_baseline": paired_bootstrap(flags, current_flags),
        }
    (artifacts / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    with (artifacts / "error_slices.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "id",
            "original_text",
            "gold",
            "system",
            "output",
            "semantic_group",
            "reading_style",
            "ambiguity_cluster",
            "automatic_mismatch_type",
            "manual_review_needed",
            "note",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in outputs:
            if value["strict_preferred_match"]:
                mismatch = "exact"
            elif value["canonical_preferred_match"]:
                mismatch = "formatting_only"
            elif value["status"] == "unsupported":
                mismatch = "unsupported"
            elif value["status"] == "runtime_error":
                mismatch = "runtime_error"
            else:
                mismatch = "lexical_or_semantic_mismatch"
            writer.writerow(
                {
                    "id": value["id"],
                    "original_text": value["original_text"],
                    "gold": value["preferred_gold"],
                    "system": value["system"],
                    "output": value["output_text"],
                    "semantic_group": "",
                    "reading_style": "",
                    "ambiguity_cluster": "cluster_unknown",
                    "automatic_mismatch_type": mismatch,
                    "manual_review_needed": mismatch
                    not in {"exact", "formatting_only"},
                    "note": "corrected gold documented"
                    if value["gold_correction"]
                    else "",
                }
            )
    best_existing = max(
        ("num2words_vi", "vietnormalizer", "soe_vinorm"),
        key=lambda name: systems[name]["canonical_preferred"],
    )
    (artifacts / "best_existing.json").write_text(
        json.dumps(
            {
                "system": best_existing,
                "selection_metric": "canonical preferred sentence accuracy on pilot development set",
                "score": systems[best_existing]["canonical_preferred_pair"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    by_id = defaultdict(dict)
    for value in outputs:
        by_id[value["id"]][value["system"]] = value
    informative = []
    for case_id in sorted(by_id):
        methods = by_id[case_id]
        current = methods["current_rule"]
        existing = methods[best_existing]
        all_wrong = not any(value["acceptable_match"] for value in methods.values())
        if current["acceptable_match"] and not existing["acceptable_match"]:
            category = "current correct; best existing wrong"
        elif existing["acceptable_match"] and not current["acceptable_match"]:
            category = "best existing correct; current wrong"
        elif all_wrong:
            category = "all automatic systems wrong"
        elif len({value["output_text"] for value in methods.values()}) > 1:
            category = "systems disagree"
        else:
            continue
        informative.append(
            {
                "id": case_id,
                "role": current["role"],
                "category": category,
                "original_text": current["original_text"],
                "current_rule": current["output_text"],
                "best_existing": existing["output_text"],
                "preferred_gold": current["preferred_gold"],
            }
        )
    category_order = {
        "current correct; best existing wrong": 0,
        "best existing correct; current wrong": 1,
        "all automatic systems wrong": 2,
        "systems disagree": 3,
    }
    informative.sort(
        key=lambda value: (
            category_order[value["category"]],
            value["role"],
            value["id"],
        )
    )
    diverse = []
    for category in category_order:
        category_values = [
            value for value in informative if value["category"] == category
        ]
        diverse.extend(category_values[:7])
    if len(diverse) < 25:
        chosen = {value["id"] for value in diverse}
        diverse.extend(value for value in informative if value["id"] not in chosen)[
            : 25 - len(diverse)
        ]
    informative = diverse[:25]
    with (artifacts / "informative_errors.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(informative[0]))
        writer.writeheader()
        writer.writerows(informative)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "dirty_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines(),
        "external_python": str(args.external_python),
        "hf_home": external_env["HF_HOME"],
        "cpu": platform.processor(),
        "cuda_available": _cuda_available(),
        "external_packages": {
            "num2words": "0.5.14",
            "vietnormalizer": "0.2.3",
            "soe-vinorm": "0.3.2",
            "onnxruntime": "1.19.2",
        },
        "soe_model": {
            "repository": "vinhdq842/soe-vinorm",
            "revision": "cb9705b",
            "device": "CPU",
        },
        "gpu_probe": "nvidia-smi unavailable; torch reports no CUDA device",
        "command": "python -m experiments.tn_value_gate.run_all --dataset ... --external-python ...",
    }
    (artifacts / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    freeze_env = os.environ | {"UV_CACHE_DIR": "/tmp/omnivoice-tn-value-gate-uv-cache"}
    freeze = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", str(args.external_python)],
        text=True,
        env=freeze_env,
    )
    (artifacts / "requirements_freeze.txt").write_text(freeze, encoding="utf-8")
    write_value_gate(
        artifacts / "value_gate_report.md",
        manifest,
        audit,
        summary,
        best_existing,
        "audio gate pending",
        informative,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
