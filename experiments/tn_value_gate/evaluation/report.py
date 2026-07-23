"""Write concise CSV and Markdown reports from aggregate values."""

import csv
from pathlib import Path


def write_summary_csv(path: Path, systems: dict) -> None:
    fields = ["system", "records", "attempted", "coverage", "success", "partial", "unsupported",
              "runtime_error", "strict_preferred", "canonical_preferred", "acceptable",
              "sentence_errors", "mean_latency_ms", "median_latency_ms", "mean_normalized_cer",
              "canonical_wilson_low", "canonical_wilson_high"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in systems.items():
            writer.writerow({"system": name, **{key: values.get(key) for key in fields if key != "system"}})


def fraction(value: list[int]) -> str:
    return f"{value[0]}/{value[1]} = {100 * value[0] / value[1]:.1f}%" if value[1] else "0/0"


def write_value_gate(path: Path, manifest: dict, audit: dict, summary: dict, best_existing: str,
                     audio_status: str, informative: list[dict]) -> None:
    systems = summary["systems"]
    lines = [
        "# Vietnamese TN value-gate report", "", "## Dataset", "",
        f"- Source: `{manifest['absolute_path']}`",
        f"- Records: {manifest['record_count']}", f"- SHA-256: `{manifest['sha256']}`",
        "- Designation: **pilot development set**; no unbiased test accuracy is claimed.",
        f"- Historical 115/160: {audit['historical_115_of_160']}", "", "## Text baselines", "",
        "| System | Attempted | Canonical preferred | Acceptable | Status |", "|---|---:|---:|---:|---|",
    ]
    for name in ("current_rule", "num2words_vi", "vietnormalizer", "soe_vinorm"):
        value = systems[name]
        lines.append(f"| {name} | {value['attempted']}/{value['records']} | "
                     f"{fraction(value['canonical_preferred_pair'])} | {fraction(value['acceptable_pair'])} | "
                     f"{value['runtime_error']} runtime errors |")
    lines += ["", f"Best existing automatic baseline by predeclared canonical preferred accuracy: **{best_existing}**.",
              "Current rule, VietNormalizer, and soe-vinorm tie for highest coverage at 200/200; soe-vinorm has "
              "the best preferred and acceptable accuracy.",
              "", "Strict matching preserves original casing and spacing; canonical formatting only normalizes NFC, "
              "whitespace, and spaces before punctuation. It never lowercases or rewrites words.",
              "VietNormalizer lowercases complete sentences, so it scores zero exact matches under this predeclared "
              "case-sensitive protocol; its lower CER shows that this is not equivalent to total verbalization failure.",
              "", "## Ambiguity and errors", "",
              "The source schema has no audited `semantic_group`, `reading_style`, ambiguity cluster, or domain. "
              "Accordingly, official ambiguous-subset and domain accuracy are unavailable (`cluster_unknown`).",
              "Surface-form slices are diagnostic only and are not called audited ambiguity labels.", "",
              "Per-role results are stored in `baseline_by_role.csv` and `baseline_by_role.json`. Accuracy for the "
              "ambiguous subset is unavailable because the dataset has no trustworthy ambiguity labels.", "",
              "The largest remaining gaps are contextual identifiers/rooms/phones, plus deterministic coverage for "
              "units, ratios, dates, decimals, versions, and punctuation. Package disagreement demonstrates that "
              "both parsing/post-processing and context matter. Text-level evidence therefore shows meaningful "
              "remaining headroom for a custom system, but the missing ambiguity labels prevent claiming that the "
              "headroom is specifically or predominantly contextual.", "",
              "### 25 informative disagreements", "",
              "The full machine-readable selection is in `informative_errors.csv`.", "",
              "| ID | Role | Slice | Raw | Current | Best existing | Gold |",
              "|---|---|---|---|---|---|---|",]
    for value in informative:
        cells = [value[key].replace("|", "\\|") for key in
                 ("id", "role", "category", "original_text", "current_rule", "best_existing", "preferred_gold")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons", "",
              "Paired McNemar and bootstrap results are stored in `paired_comparisons.json`. Because rules were "
              "edited after inspecting pilot data, p-values are descriptive and not confirmatory.", "",
              "## Audio value gate", "", f"Status: **{audio_status}**.",
              "No product conclusion about raw versus normalized OmniVoice audio is made without listening ratings.",
              "", "## Current decision", "", "**Decision D — Evidence insufficient.**", "",
              "Text-level results can justify a 24-case listening pilot, but cannot answer whether normalization "
              "improves OmniVoice correctness or naturalness. The smallest valuable next step is to run the frozen "
              "72-sample RAW/BEST_EXISTING/GOLD Colab manifest with one exact reference transcript, then collect "
              "blinded human ratings.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
