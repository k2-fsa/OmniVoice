"""Benchmark the four authorized Vietnamese normalization conditions.

The upstream and improved VietNormalizer backends run in separate Python
interpreters so editable development installs cannot contaminate the upstream
0.2.3 baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

BACKENDS = ("identity", "custom", "vietnormalizer_upstream", "vietnormalizer_improved")
CORRUPTED_FRAGMENTS = ("đồngơn", "đ ơn")
ALLOWED_REWRITTEN_WORDS = {
    "đ",
    "₫",
    "vnd",
    "vnđ",
    "usd",
    "hkd",
    "kg",
    "g",
    "mg",
    "km",
    "m",
    "cm",
    "mm",
    "l",
    "ml",
    "h",
    "s",
    "t",
}
_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?])")


def canonicalize(text: str) -> str:
    value = unicodedata.normalize("NFC", text).strip()
    value = _SPACE.sub(" ", value)
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", value)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acceptable_options(row: dict) -> list[str]:
    return [*row["preferred_spoken"], *row.get("acceptable_spoken", [])]


def lexical_words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category[0] in {"L", "M"}:
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def corrupted_word(raw: str, output: str) -> bool:
    folded = output.casefold()
    if any(fragment in folded for fragment in CORRUPTED_FRAGMENTS):
        return True
    for word in lexical_words(raw):
        if word.casefold() in ALLOWED_REWRITTEN_WORDS:
            continue
        if len(word) >= 2 and set(word) <= set("IVXLC"):
            continue
        if word.casefold() not in folded:
            return True
    return False


def worker(backend: str) -> None:
    from vietnormalizer import VietnameseNormalizer

    normalizer = VietnameseNormalizer(enable_transliteration=False)
    method = normalizer.normalize if backend == "upstream" else normalizer.normalize_numeric
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        try:
            output = method(request["input"])
            error = None
        except Exception as exc:  # benchmark must preserve per-record failure details
            output = request["input"]
            error = f"{type(exc).__name__}: {exc}"
        print(
            json.dumps(
                {
                    "id": request["id"],
                    "output": output,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error": error,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def run_external(python: Path, backend: str, rows: list[dict]) -> dict[str, dict]:
    payload = "".join(
        json.dumps({"id": row["id"], "input": row["input"]}, ensure_ascii=False) + "\n"
        for row in rows
    )
    completed = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--worker", backend],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"worker exited {completed.returncode}")
    values = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(values) != len(rows):
        raise RuntimeError(f"{backend}: expected {len(rows)} records, received {len(values)}")
    return {value["id"]: value for value in values}


def backend_metadata(python: Path) -> dict:
    code = (
        "import inspect, json, platform; "
        "from importlib.metadata import version; "
        "from vietnormalizer import VietnameseNormalizer; "
        "print(json.dumps({'package_version': version('vietnormalizer'), "
        "'python': platform.python_version(), "
        "'constructor': str(inspect.signature(VietnameseNormalizer)), "
        "'normalize': str(inspect.signature(VietnameseNormalizer.normalize)), "
        "'has_normalize_numeric': hasattr(VietnameseNormalizer, 'normalize_numeric')}))"
    )
    return json.loads(
        subprocess.check_output([str(python), "-c", code], text=True).strip()
    )


def run_local(fn: Callable[[str], str], rows: list[dict]) -> dict[str, dict]:
    values = {}
    for row in rows:
        started = time.perf_counter()
        try:
            output = fn(row["input"])
            error = None
        except Exception as exc:
            output = row["input"]
            error = f"{type(exc).__name__}: {exc}"
        values[row["id"]] = {
            "id": row["id"],
            "output": output,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": error,
        }
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))]


def aggregate(rows: list[dict]) -> dict:
    latencies = [row["latency_ms"] for row in rows]
    elapsed_ms = sum(latencies)
    count = len(rows)
    return {
        "case_count": count,
        "preferred_exact": sum(row["preferred_exact"] for row in rows),
        "acceptable_exact": sum(row["acceptable_exact"] for row in rows),
        "casefold_acceptable": sum(row["casefold_acceptable"] for row in rows),
        "changed_inputs": sum(row["changed"] for row in rows),
        "false_positives": sum(row["false_positive"] for row in rows),
        "false_negatives": sum(row["false_negative"] for row in rows),
        "corrupted_words": sum(row["corrupted_word"] for row in rows),
        "boundary_errors": sum(row["boundary_error"] for row in rows),
        "unsupported": sum(row["unsupported"] for row in rows),
        "runtime_errors": sum(row["error"] is not None for row in rows),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 0.95),
        "throughput_items_per_second": count / (elapsed_ms / 1000) if elapsed_ms else None,
    }


def evaluate(
    source_rows: list[dict],
    results: dict[str, dict[str, dict]],
) -> tuple[list[dict], dict]:
    outputs: list[dict] = []
    for row in source_rows:
        options = row["expected_options"]
        canonical_options = {canonicalize(value) for value in options}
        folded_options = {value.casefold() for value in canonical_options}
        for backend in BACKENDS:
            result = results[backend][row["id"]]
            output = result["output"]
            acceptable = canonicalize(output) in canonical_options
            corrupt = corrupted_word(row["input"], output)
            changed = output != row["input"]
            outputs.append(
                {
                    **row,
                    "backend": backend,
                    **result,
                    "preferred_exact": output == options[0],
                    "acceptable_exact": acceptable,
                    "casefold_acceptable": canonicalize(output).casefold() in folded_options,
                    "changed": changed,
                    "corrupted_word": corrupt,
                    "boundary_error": corrupt,
                    "false_positive": corrupt,
                    "false_negative": not changed and not acceptable,
                    "unsupported": not changed and not acceptable,
                }
            )

    summary = {"systems": {}, "by_category": {}}
    for backend in BACKENDS:
        summary["systems"][backend] = aggregate(
            [row for row in outputs if row["backend"] == backend]
        )
    for category in sorted({row["category"] for row in source_rows}):
        summary["by_category"][category] = {
            backend: aggregate(
                [
                    row
                    for row in outputs
                    if row["backend"] == backend and row["category"] == category
                ]
            )
            for backend in BACKENDS
        }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("upstream", "improved"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--smoke-cases", type=Path)
    parser.add_argument("--upstream-python", type=Path)
    parser.add_argument("--improved-python", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--improved-repo", type=Path)
    parser.add_argument("--upstream-tests-pass", action="store_true")
    parser.add_argument("--new-numeric-tests-pass", action="store_true")
    parser.add_argument("--omnivoice-integration-tests-pass", action="store_true")
    parser.add_argument("--no-custom-fallback", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    required = (
        args.dataset,
        args.smoke_cases,
        args.upstream_python,
        args.improved_python,
        args.artifacts,
        args.improved_repo,
    )
    if any(value is None for value in required):
        parser.error("driver mode requires dataset, smoke cases, both Python paths, and artifacts")

    from omnivoice.utils.text import normalize_text

    def custom_normalize(text: str) -> str:
        return normalize_text(text, "vi")

    dataset_path = args.dataset.resolve()
    smoke_path = args.smoke_cases.resolve()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    corpus = load_jsonl(dataset_path)
    dataset_rows = [
        {
            "id": row["id"],
            "category": row["role"],
            "input": row["raw_text"],
            "expected_options": acceptable_options(row),
            "annotated_span": row["number"],
        }
        for row in corpus
    ]
    smoke_rows = load_jsonl(smoke_path)

    def run_all(rows: list[dict]) -> dict[str, dict[str, dict]]:
        return {
            "identity": run_local(lambda value: value, rows),
            "custom": run_local(custom_normalize, rows),
            "vietnormalizer_upstream": run_external(
                args.upstream_python.absolute(), "upstream", rows
            ),
            "vietnormalizer_improved": run_external(
                args.improved_python.absolute(), "improved", rows
            ),
        }

    corpus_outputs, corpus_summary = evaluate(dataset_rows, run_all(dataset_rows))
    smoke_outputs, smoke_summary = evaluate(smoke_rows, run_all(smoke_rows))

    by_id = defaultdict(dict)
    for row in corpus_outputs:
        by_id[row["id"]][row["backend"]] = row
    comparison = Counter()
    casefold_comparison = Counter()
    regressions = []
    casefold_regressions = []
    for case_id, methods in by_id.items():
        upstream = methods["vietnormalizer_upstream"]["acceptable_exact"]
        improved = methods["vietnormalizer_improved"]["acceptable_exact"]
        key = (
            "both_correct"
            if upstream and improved
            else "upstream_correct_improved_wrong"
            if upstream
            else "upstream_wrong_improved_correct"
            if improved
            else "both_wrong"
        )
        comparison[key] += 1
        if upstream and not improved:
            regressions.append(
                {
                    "id": case_id,
                    "input": methods["vietnormalizer_upstream"]["input"],
                    "upstream": methods["vietnormalizer_upstream"]["output"],
                    "improved": methods["vietnormalizer_improved"]["output"],
                    "expected_options": methods["vietnormalizer_improved"]["expected_options"],
                }
            )

        upstream_folded = methods["vietnormalizer_upstream"]["casefold_acceptable"]
        improved_folded = methods["vietnormalizer_improved"]["casefold_acceptable"]
        folded_key = (
            "both_correct"
            if upstream_folded and improved_folded
            else "upstream_correct_improved_wrong"
            if upstream_folded
            else "upstream_wrong_improved_correct"
            if improved_folded
            else "both_wrong"
        )
        casefold_comparison[folded_key] += 1
        if upstream_folded and not improved_folded:
            casefold_regressions.append(
                {
                    "id": case_id,
                    "input": methods["vietnormalizer_upstream"]["input"],
                    "upstream": methods["vietnormalizer_upstream"]["output"],
                    "improved": methods["vietnormalizer_improved"]["output"],
                    "expected_options": methods["vietnormalizer_improved"]["expected_options"],
                }
            )

    hard_negative = [
        row
        for row in smoke_outputs
        if row["backend"] == "vietnormalizer_improved" and row["category"] == "HARD_NEGATIVE"
    ]
    upstream_metrics = corpus_summary["systems"]["vietnormalizer_upstream"]
    improved_metrics = corpus_summary["systems"]["vietnormalizer_improved"]
    gate = {
        "corrupted_word_zero": improved_metrics["corrupted_words"] == 0,
        "regression_105_order_fixed": not next(
            row
            for row in smoke_outputs
            if row["backend"] == "vietnormalizer_improved"
            and row["id"] == "smoke_cardinal_boundary"
        )["corrupted_word"],
        "no_serious_corruption": improved_metrics["boundary_errors"] == 0,
        "acceptable_not_below_upstream": (
            improved_metrics["acceptable_exact"] >= upstream_metrics["acceptable_exact"]
        ),
        "casefold_acceptable_not_below_upstream": (
            improved_metrics["casefold_acceptable"] >= upstream_metrics["casefold_acceptable"]
        ),
        "hard_negative_false_positive_not_increased": not any(
            row["false_positive"] for row in hard_negative
        ),
        "upstream_tests_pass": args.upstream_tests_pass,
        "new_numeric_tests_pass": args.new_numeric_tests_pass,
        "omnivoice_integration_tests_pass": args.omnivoice_integration_tests_pass,
        "no_custom_fallback": args.no_custom_fallback,
    }
    gate["passed"] = all(gate.values())

    manifest = {
        "path": str(dataset_path),
        "case_count": len(corpus),
        "unique_id_count": len({row["id"] for row in corpus}),
        "duplicate_input_count": len(corpus)
        - len({row["raw_text"] for row in corpus}),
        "category_distribution": dict(sorted(Counter(row["role"] for row in corpus).items())),
        "source": "git 7053b50:experiments/vi_number_normalization/data/pilot.jsonl",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256(dataset_path),
        "intended_use": "development/regression only; not a final held-out set",
    }
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": 20260721,
        "dataset_sha256": manifest["sha256"],
        "upstream_python": str(args.upstream_python.absolute()),
        "improved_python": str(args.improved_python.absolute()),
        "upstream": {
            "repository": "https://github.com/nghimestudio/vietnormalizer",
            **backend_metadata(args.upstream_python.absolute()),
        },
        "improved": {
            "repository_path": str(args.improved_repo.absolute()),
            "base_commit": subprocess.check_output(
                ["git", "-C", str(args.improved_repo), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "branch": subprocess.check_output(
                ["git", "-C", str(args.improved_repo), "branch", "--show-current"],
                text=True,
            ).strip(),
            "dirty_status": subprocess.check_output(
                ["git", "-C", str(args.improved_repo), "status", "--short"],
                text=True,
            ).splitlines(),
            **backend_metadata(args.improved_python.absolute()),
        },
    }

    artifacts_to_write = {
        "dataset_manifest.json": manifest,
        "environment.json": environment,
        "benchmark_summary.json": corpus_summary,
        "smoke_summary.json": smoke_summary,
        "upstream_improved_comparison.json": {
            "strict_counts": dict(comparison),
            "strict_regressions": regressions,
            "casefold_counts": dict(casefold_comparison),
            "casefold_regressions": casefold_regressions,
        },
        "integration_gate.json": gate,
    }
    for name, value in artifacts_to_write.items():
        (artifacts / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for name, rows in (
        ("benchmark_outputs.jsonl", corpus_outputs),
        ("smoke_outputs.jsonl", smoke_outputs),
    ):
        with (artifacts / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
