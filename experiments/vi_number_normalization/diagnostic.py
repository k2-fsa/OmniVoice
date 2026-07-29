#!/usr/bin/env python3
"""Vietnamese numerical normalization diagnostic pipeline."""

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EXPERIMENT_DIR / "data" / "pilot.jsonl"
DEFAULT_ROLE_MAP = EXPERIMENT_DIR / "role_groups.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/omnivoice_vi_number_diagnostic")
SYSTEMS = ("raw", "current", "oracle")
HUMAN_COLUMNS = (
    "number_correct",
    "pronunciation_clear",
    "naturalness_1_to_5",
    "error_type",
    "evaluator_notes",
)
EVALUATION_COLUMNS = (
    "id",
    "system",
    "role",
    "broad_group",
    "number",
    "text_sent",
    "audio_path",
    *HUMAN_COLUMNS,
)


@dataclass(frozen=True)
class GenerationSettings:
    seed: int = 2026
    speed: float = 1.0
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    denoise: bool = True
    postprocess_output: bool = True
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0

    def model_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("seed")
        values["normalize_text"] = False
        return values


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records = []
    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"Cannot read {path}: {exc}"]
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            errors.append(f"line {line_number}: blank lines are not valid records")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"line {line_number}: record must be a JSON object")
            continue
        value["_line"] = line_number
        records.append(value)
    return records, errors


def load_role_map(path: Path) -> tuple[dict[str, str], list[str]]:
    errors = []
    try:
        groups = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Cannot load role mapping {path}: {exc}"]
    if not isinstance(groups, dict):
        return {}, ["Role mapping must be a JSON object"]
    role_map = {}
    for group, roles in groups.items():
        if not isinstance(group, str) or not group or not isinstance(roles, list):
            errors.append(f"Invalid broad-group entry: {group!r}")
            continue
        for role in roles:
            if not isinstance(role, str) or not role:
                errors.append(f"Invalid role in broad group {group!r}: {role!r}")
            elif role in role_map:
                errors.append(
                    f"Role {role!r} is mapped to both {role_map[role]!r} and {group!r}"
                )
            else:
                role_map[role] = group
    return role_map, errors


def _validate_string_list(
    record: dict[str, Any], field: str, required: bool
) -> list[str]:
    errors = []
    if field not in record and not required:
        return errors
    value = record.get(field)
    if not isinstance(value, list) or (required and not value):
        return [f"{field} must be {'a nonempty' if required else 'a'} list"]
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{field} must contain only nonempty strings")
    return errors


def validate_records(
    records: list[dict[str, Any]], role_map: dict[str, str]
) -> tuple[list[str], list[str], Counter]:
    errors = []
    warnings = []
    seen_ids = set()
    roles = Counter()
    required_types = {
        "id": str,
        "raw_text": str,
        "number": str,
        "role": str,
        "preferred_spoken": list,
    }
    for index, record in enumerate(records, 1):
        location = f"line {record.get('_line', index)}"
        for field, expected_type in required_types.items():
            if field not in record:
                errors.append(f"{location}: missing required field {field!r}")
            elif not isinstance(record[field], expected_type):
                errors.append(
                    f"{location}: {field} must be {expected_type.__name__}"
                )
        record_id = record.get("id")
        if isinstance(record_id, str):
            if not record_id.strip():
                errors.append(f"{location}: id must be nonempty")
            elif record_id in seen_ids:
                errors.append(f"{location}: duplicate id {record_id!r}")
            seen_ids.add(record_id)
        for message in _validate_string_list(record, "preferred_spoken", True):
            errors.append(f"{location}: {message}")
        for message in _validate_string_list(record, "acceptable_spoken", False):
            errors.append(f"{location}: {message}")

        preferred = record.get("preferred_spoken", [])
        acceptable = record.get("acceptable_spoken", [])
        if isinstance(preferred, list) and all(isinstance(x, str) for x in preferred):
            if len(preferred) != len(set(preferred)):
                errors.append(f"{location}: preferred_spoken contains duplicates")
            if isinstance(acceptable, list) and all(
                isinstance(x, str) for x in acceptable
            ):
                overlap = set(preferred) & set(acceptable)
                if overlap:
                    errors.append(
                        f"{location}: preferred_spoken overlaps acceptable_spoken: "
                        f"{sorted(overlap)!r}"
                    )
        raw_text = record.get("raw_text")
        number = record.get("number")
        if isinstance(raw_text, str) and isinstance(number, str):
            if not raw_text.strip():
                errors.append(f"{location}: raw_text must be nonempty")
            if not number.strip():
                errors.append(f"{location}: number must be nonempty")
            elif number not in raw_text:
                errors.append(
                    f"{location}: number {number!r} does not occur in raw_text"
                )
        role = record.get("role")
        if isinstance(role, str):
            roles[role] += 1
            if role not in role_map:
                errors.append(f"{location}: unknown role {role!r}")

    for role, count in sorted(roles.items()):
        if count == 1:
            warnings.append(f"role {role!r} has only one record")
    return errors, warnings, roles


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "_line"}


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_subset(
    records: list[dict[str, Any]],
    role_map: dict[str, str],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError("sample_size cannot exceed the dataset size")
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda item: item["id"])
    tie_breakers = {record["id"]: rng.random() for record in ordered}
    selected = []
    selected_ids = set()

    def add(record: dict[str, Any]) -> bool:
        if record["id"] in selected_ids or len(selected) >= sample_size:
            return False
        selected.append(record)
        selected_ids.add(record["id"])
        return True

    # Prioritize complete contextual families: the same written number appearing
    # with different roles or preferred readings is more diagnostic as a pair.
    by_number = defaultdict(list)
    for record in ordered:
        by_number[record["number"]].append(record)
    contrastive = []
    for number, family in by_number.items():
        readings = {record["preferred_spoken"][0] for record in family}
        roles = {record["role"] for record in family}
        if len(family) > 1 and (len(readings) > 1 or len(roles) > 1):
            contrastive.append((number, family))
    contrastive.sort(
        key=lambda item: (
            -len({record["role"] for record in item[1]}),
            -len(item[1]),
            min(tie_breakers[record["id"]] for record in item[1]),
            item[0],
        )
    )
    for _, family in contrastive:
        if len(selected) + len(family) > sample_size:
            continue
        for record in sorted(family, key=lambda item: tie_breakers[item["id"]]):
            add(record)

    by_group = defaultdict(list)
    for record in ordered:
        by_group[role_map[record["role"]]].append(record)
    for group in sorted(by_group):
        candidates = sorted(
            by_group[group], key=lambda item: (tie_breakers[item["id"]], item["id"])
        )
        if not any(role_map[item["role"]] == group for item in selected):
            for record in candidates:
                if add(record):
                    break

    group_order = sorted(by_group)
    rng.shuffle(group_order)
    while len(selected) < sample_size:
        progress = False
        group_counts = Counter(role_map[item["role"]] for item in selected)
        for group in sorted(group_order, key=lambda item: (group_counts[item], item)):
            candidates = sorted(
                by_group[group],
                key=lambda item: (tie_breakers[item["id"]], item["id"]),
            )
            for record in candidates:
                if add(record):
                    progress = True
                    break
            if len(selected) >= sample_size:
                break
        if not progress:
            break
    return selected


def write_selection(
    output: Path,
    selected: list[dict[str, Any]],
    role_map: dict[str, str],
    dataset: Path,
    role_map_path: Path,
    seed: int,
    sample_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "dataset": str(dataset),
            "dataset_sha256": dataset_sha256(dataset),
            "role_mapping": str(role_map_path),
            "role_mapping_sha256": dataset_sha256(role_map_path),
            "seed": seed,
            "sample_size": sample_size,
            "selection_policy": "contrastive_families_then_balanced_broad_groups",
            "broad_group_distribution": dict(
                sorted(Counter(role_map[item["role"]] for item in selected).items())
            ),
        },
        "records": [
            {**clean_record(record), "broad_group": role_map[record["role"]]}
            for record in selected
        ],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Invalid selection manifest: {path}")
    return payload


def stable_audio_filename(record_id: str, system: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", record_id).strip("._") or "record"
    digest = hashlib.sha256(f"{record_id}\0{system}".encode()).hexdigest()[:10]
    return f"{safe_id}__{system}__{digest}.wav"


def load_current_normalizer() -> Callable[[str, str | None], str]:
    """Load the repository's real normalizer without importing the TTS model."""
    try:
        from omnivoice.utils.text import normalize_text

        return normalize_text
    except ModuleNotFoundError as exc:
        # A source checkout may not have heavy TTS dependencies installed.
        # Load the same utility file directly so validation/dry-run stays light.
        text_path = EXPERIMENT_DIR.parents[1] / "omnivoice" / "utils" / "text.py"
        spec = importlib.util.spec_from_file_location(
            "_omnivoice_diagnostic_text", text_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Cannot load OmniVoice text normalizer: {text_path}"
            ) from exc
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logging.warning(
            "Loaded omnivoice/utils/text.py directly because optional runtime "
            "dependencies are unavailable: %s",
            exc,
        )
        return module.normalize_text


def build_variants(
    selection: dict[str, Any],
    output_dir: Path,
    settings: GenerationSettings,
    normalizer: Callable[[str, str | None], str] | None = None,
    generation_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if normalizer is None:
        normalizer = load_current_normalizer()
    variants = []
    generation_config = {**asdict(settings), **(generation_context or {})}
    for record in selection["records"]:
        current_text = normalizer(record["raw_text"], "vi")
        texts = {
            "raw": record["raw_text"],
            "current": current_text,
            "oracle": record["preferred_spoken"][0],
        }
        for system in SYSTEMS:
            audio_path = output_dir / "audio" / stable_audio_filename(
                record["id"], system
            )
            variants.append(
                {
                    "id": record["id"],
                    "role": record["role"],
                    "broad_group": record["broad_group"],
                    "number": record["number"],
                    "system": system,
                    "original_text": record["raw_text"],
                    "text_sent": texts[system],
                    "preferred_spoken": record["preferred_spoken"],
                    "acceptable_spoken": record.get("acceptable_spoken", []),
                    "generation_config": generation_config,
                    "normalize_text": False,
                    "audio_path": str(audio_path),
                }
            )
    return variants


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_evaluation_csv(path: Path, variants: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS)
        writer.writeheader()
        for variant in variants:
            row = {column: "" for column in EVALUATION_COLUMNS}
            for column in EVALUATION_COLUMNS:
                if column in variant and column not in HUMAN_COLUMNS:
                    row[column] = variant[column]
            writer.writerow(row)


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_audio_generation(
    variants: list[dict[str, Any]],
    output_dir: Path,
    checkpoint: str,
    reference_audio: str,
    reference_text: str,
    device: str | None,
    settings: GenerationSettings,
    model_factory: Callable[..., Any] | None = None,
    audio_writer: Callable[[str, Any, int], None] | None = None,
    seed_setter: Callable[[int], None] | None = None,
) -> dict[str, int]:
    if device is None:
        from omnivoice.utils.common import get_best_device

        device = get_best_device()
    if model_factory is None:
        import torch

        from omnivoice import OmniVoice

        model_factory = OmniVoice.from_pretrained
        dtype = torch.float32 if str(device).startswith("cpu") else torch.float16
    else:
        dtype = None
    if audio_writer is None:
        import soundfile as sf

        audio_writer = sf.write
    model = model_factory(checkpoint, device_map=device, dtype=dtype)
    prompt = model.create_voice_clone_prompt(
        ref_audio=reference_audio,
        ref_text=reference_text,
    )
    progress_path = output_dir / "generation_results.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    with progress_path.open("a", encoding="utf-8") as progress:
        for variant in variants:
            audio_path = Path(variant["audio_path"])
            if audio_path.is_file() and audio_path.stat().st_size > 0:
                summary["skipped"] += 1
                continue
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            (seed_setter or _set_seed)(settings.seed)
            result = {
                "id": variant["id"],
                "system": variant["system"],
                "audio_path": str(audio_path),
            }
            try:
                audio = model.generate(
                    text=variant["text_sent"],
                    voice_clone_prompt=prompt,
                    **settings.model_kwargs(),
                )[0]
                audio_writer(str(audio_path), audio, model.sampling_rate)
                result["status"] = "success"
                summary["generated"] += 1
            except Exception as exc:  # keep completed clips and continue
                result.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                summary["errors"] += 1
                logging.exception(
                    "Generation failed for %s/%s", variant["id"], variant["system"]
                )
            progress.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.flush()
    return dict(summary)


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _parse_naturalness(value: str) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def _system_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    number_values = [_parse_bool(row.get("number_correct", "")) for row in rows]
    pronunciation_values = [
        _parse_bool(row.get("pronunciation_clear", "")) for row in rows
    ]
    naturalness_values = [
        _parse_naturalness(row.get("naturalness_1_to_5", "")) for row in rows
    ]
    number_done = [value for value in number_values if value is not None]
    pronunciation_done = [value for value in pronunciation_values if value is not None]
    naturalness_done = [value for value in naturalness_values if value is not None]
    error_counts = Counter(
        row.get("error_type", "").strip()
        for row in rows
        if row.get("error_type", "").strip()
    )
    return {
        "rows": len(rows),
        "numeric_correctness_rate": (
            sum(number_done) / len(number_done) if number_done else None
        ),
        "numeric_rows_evaluated": len(number_done),
        "pronunciation_correctness_rate": (
            sum(pronunciation_done) / len(pronunciation_done)
            if pronunciation_done
            else None
        ),
        "pronunciation_rows_evaluated": len(pronunciation_done),
        "mean_naturalness": (
            sum(naturalness_done) / len(naturalness_done) if naturalness_done else None
        ),
        "naturalness_rows_evaluated": len(naturalness_done),
        "number_error_counts": {
            error: error_counts.get(error, 0)
            for error in (
                "missing-number",
                "repeated-number",
                "substituted-number",
            )
        },
        "other_error_counts": {
            key: count
            for key, count in sorted(error_counts.items())
            if key
            not in {"missing-number", "repeated-number", "substituted-number"}
        },
    }


def summarize_evaluations(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_system = {
        system: _system_summary([row for row in rows if row.get("system") == system])
        for system in SYSTEMS
    }
    groups = sorted(
        {row.get("broad_group", "") for row in rows if row.get("broad_group")}
    )
    by_group = {
        group: {
            system: _system_summary(
                [
                    row
                    for row in rows
                    if row.get("broad_group") == group and row.get("system") == system
                ]
            )
            for system in SYSTEMS
        }
        for group in groups
    }
    by_id = defaultdict(dict)
    for row in rows:
        if row.get("id") and row.get("system") in SYSTEMS:
            by_id[row["id"]][row["system"]] = row
    oracle_succeeds = []
    oracle_fails = []
    complete_pairs = 0
    paired_values = {
        "numeric_correctness": {system: [] for system in SYSTEMS},
        "pronunciation_correctness": {system: [] for system in SYSTEMS},
        "naturalness": {system: [] for system in SYSTEMS},
    }
    for record_id, system_rows in sorted(by_id.items()):
        if set(system_rows) != set(SYSTEMS):
            continue
        number_values = {
            system: _parse_bool(system_rows[system].get("number_correct", ""))
            for system in SYSTEMS
        }
        pronunciation_values = {
            system: _parse_bool(
                system_rows[system].get("pronunciation_clear", "")
            )
            for system in SYSTEMS
        }
        naturalness_values = {
            system: _parse_naturalness(
                system_rows[system].get("naturalness_1_to_5", "")
            )
            for system in SYSTEMS
        }
        for metric, values in (
            ("numeric_correctness", number_values),
            ("pronunciation_correctness", pronunciation_values),
            ("naturalness", naturalness_values),
        ):
            if all(value is not None for value in values.values()):
                for system in SYSTEMS:
                    paired_values[metric][system].append(values[system])
        if all(value is not None for value in number_values.values()):
            complete_pairs += 1
            if number_values["oracle"] and (
                not number_values["raw"] or not number_values["current"]
            ):
                oracle_succeeds.append(record_id)
            if not number_values["oracle"]:
                oracle_fails.append(record_id)
    paired = {}
    for metric, systems in paired_values.items():
        count = len(systems["raw"])
        paired[metric] = {
            "records": count,
            "raw": sum(systems["raw"]) / count if count else None,
            "current": sum(systems["current"]) / count if count else None,
            "oracle": sum(systems["oracle"]) / count if count else None,
        }
    return {
        "systems": by_system,
        "broad_groups": by_group,
        "paired_results": paired,
        "paired_records_evaluated": complete_pairs,
        "oracle_succeeds_where_raw_or_current_fails": oracle_succeeds,
        "oracle_also_fails": oracle_fails,
    }


def _validated_dataset(
    dataset: Path, role_map_path: Path
) -> tuple[list[dict[str, Any]], dict[str, str], list[str], Counter]:
    records, errors = read_jsonl(dataset)
    role_map, mapping_errors = load_role_map(role_map_path)
    validation_errors, warnings, roles = validate_records(records, role_map)
    all_errors = errors + mapping_errors + validation_errors
    if all_errors:
        raise ValueError("\n".join(all_errors))
    return records, role_map, warnings, roles


def command_validate(args: argparse.Namespace) -> int:
    records, parse_errors = read_jsonl(args.dataset)
    role_map, mapping_errors = load_role_map(args.role_map)
    errors, warnings, roles = validate_records(records, role_map)
    errors = parse_errors + mapping_errors + errors
    print(f"records: {len(records)}")
    print("role distribution:")
    for role, count in sorted(roles.items()):
        print(f"  {role}: {count}")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


def command_select(args: argparse.Namespace) -> int:
    records, role_map, warnings, _ = _validated_dataset(args.dataset, args.role_map)
    for warning in warnings:
        logging.warning(warning)
    selected = select_subset(records, role_map, args.sample_size, args.seed)
    write_selection(
        args.output,
        selected,
        role_map,
        args.dataset,
        args.role_map,
        args.seed,
        args.sample_size,
    )
    print(f"selected {len(selected)} records: {args.output}")
    return 0


def _settings_from_args(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        seed=args.seed,
        speed=args.speed,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
    )


def command_generate(args: argparse.Namespace) -> int:
    selection = load_selection(args.selection)
    settings = _settings_from_args(args)
    variants = build_variants(
        selection,
        args.output_dir,
        settings,
        generation_context={
            "checkpoint": args.checkpoint,
            "reference_audio": args.reference_audio,
            "reference_text": args.reference_text,
            "device": args.device,
        },
    )
    manifest_path = args.output_dir / "variants.jsonl"
    evaluation_path = args.output_dir / "evaluation.csv"
    write_jsonl(manifest_path, variants)
    write_evaluation_csv(evaluation_path, variants)
    print(f"variant manifest: {manifest_path}")
    print(f"evaluation sheet: {evaluation_path}")
    if args.dry_run:
        print("dry run: model was not loaded")
        return 0
    missing = [
        name
        for name, value in (
            ("--checkpoint", args.checkpoint),
            ("--reference-audio", args.reference_audio),
            ("--reference-text", args.reference_text),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Audio generation requires: {', '.join(missing)}")
    summary = run_audio_generation(
        variants,
        args.output_dir,
        args.checkpoint,
        args.reference_audio,
        args.reference_text,
        args.device,
        settings,
    )
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary.get("errors") else 0


def command_summarize(args: argparse.Namespace) -> int:
    with args.evaluation.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = summarize_evaluations(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate pilot JSONL")
    validate_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    validate_parser.add_argument("--role-map", type=Path, default=DEFAULT_ROLE_MAP)
    validate_parser.set_defaults(func=command_validate)

    select_parser = subparsers.add_parser("select", help="Select diagnostic subset")
    select_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    select_parser.add_argument("--role-map", type=Path, default=DEFAULT_ROLE_MAP)
    select_parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "selection.json"
    )
    select_parser.add_argument("--seed", type=int, default=2026)
    select_parser.add_argument("--sample-size", type=int, default=30)
    select_parser.set_defaults(func=command_select)

    generate_parser = subparsers.add_parser(
        "generate", help="Create manifests and optionally generate audio"
    )
    generate_parser.add_argument(
        "--selection", type=Path, default=DEFAULT_OUTPUT_DIR / "selection.json"
    )
    generate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.add_argument("--checkpoint")
    generate_parser.add_argument("--reference-audio")
    generate_parser.add_argument("--reference-text")
    generate_parser.add_argument("--device")
    generate_parser.add_argument("--seed", type=int, default=2026)
    generate_parser.add_argument("--speed", type=float, default=1.0)
    generate_parser.add_argument("--num-step", type=int, default=32)
    generate_parser.add_argument("--guidance-scale", type=float, default=2.0)
    generate_parser.add_argument("--t-shift", type=float, default=0.1)
    generate_parser.add_argument(
        "--denoise", action=argparse.BooleanOptionalAction, default=True
    )
    generate_parser.add_argument(
        "--postprocess-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate_parser.add_argument("--layer-penalty-factor", type=float, default=5.0)
    generate_parser.add_argument("--position-temperature", type=float, default=5.0)
    generate_parser.add_argument("--class-temperature", type=float, default=0.0)
    generate_parser.set_defaults(func=command_generate)

    summary_parser = subparsers.add_parser(
        "summarize", help="Summarize completed human evaluations"
    )
    summary_parser.add_argument("--evaluation", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path)
    summary_parser.set_defaults(func=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
