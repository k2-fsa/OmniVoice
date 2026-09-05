#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data utilities for batch inference and evaluation.

Provides ``read_test_list()`` to parse JSONL test list files used by
``omnivoice.cli.infer_batch`` and evaluation scripts.
"""

import json
import math
from pathlib import Path


class JsonlTestListError(ValueError):
    """Report a fail-closed JSONL test-list contract violation."""


def _strict_object(pairs):
    """Build a JSON object while rejecting duplicate keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise JsonlTestListError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value):
    """Reject Python's non-standard NaN/Infinity JSON extensions."""
    raise JsonlTestListError(f"non-standard JSON numeric constant {value!r}")


TEST_LIST_FIELDS = frozenset(
    {
        "id",
        "text",
        "ref_audio",
        "ref_text",
        "language_id",
        "language_name",
        "duration",
        "speed",
        "final_duration",
        "final_duration_samples",
        "instruct",
    }
)
_OPTIONAL_STRING_FIELDS = (
    "ref_audio",
    "ref_text",
    "language_id",
    "language_name",
    "instruct",
)
_OPTIONAL_POSITIVE_REAL_FIELDS = (
    "duration",
    "speed",
    "final_duration",
)


def _validate_known_field_types(obj, *, path, line_no):
    """Validate every documented JSONL field before runtime startup."""
    for field in _OPTIONAL_STRING_FIELDS:
        value = obj.get(field)
        if value is not None and not isinstance(value, str):
            raise JsonlTestListError(
                f"{path}: line {line_no}: {field!r} must be a string or null"
            )

    for field in _OPTIONAL_POSITIVE_REAL_FIELDS:
        value = obj.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JsonlTestListError(
                f"{path}: line {line_no}: {field!r} must be a positive finite number or null"
            )
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite or value <= 0:
            raise JsonlTestListError(
                f"{path}: line {line_no}: {field!r} must be a positive finite number or null"
            )

    sample_target = obj.get("final_duration_samples")
    if sample_target is not None and (
        isinstance(sample_target, bool)
        or not isinstance(sample_target, int)
        or sample_target <= 0
    ):
        raise JsonlTestListError(
            f"{path}: line {line_no}: 'final_duration_samples' must be a positive integer or null"
        )


def read_test_list(path, *, reject_unknown_fields=False):
    """Read a JSONL test list file.

    Each line should be a JSON object.  Only ``id`` and ``text`` are required;
    all other fields are optional (default to ``None``):
        id, text, ref_audio, ref_text, instruct,
        language_id, language_name, duration, speed,
        final_duration, final_duration_samples

    Note: ``language_name`` is only used by evaluation scripts (under
    ``omnivoice/eval/``) for grouping and reporting results.  The model
    itself only consumes ``language_id``.

    Blank lines are ignored. Malformed JSON, duplicate JSON keys, invalid
    required fields, and duplicate sample IDs abort the entire read so a batch
    can never report success after silently dropping an item. Set
    ``reject_unknown_fields`` for production CLI input, where silently ignoring
    a misspelled generation authority would be unsafe.

    Returns a list of dicts.
    """
    path = Path(path)
    samples = []
    id_lines = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_nonfinite_json_constant,
                )
            except json.JSONDecodeError as exc:
                raise JsonlTestListError(
                    f"{path}: line {line_no}: invalid JSON "
                    f"({exc.msg} at column {exc.colno})"
                ) from exc
            except JsonlTestListError as exc:
                raise JsonlTestListError(f"{path}: line {line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                raise JsonlTestListError(
                    f"{path}: line {line_no}: each JSONL entry must be an object"
                )
            if reject_unknown_fields:
                unknown_fields = sorted(set(obj) - TEST_LIST_FIELDS)
                if unknown_fields:
                    raise JsonlTestListError(
                        f"{path}: line {line_no}: unknown field(s): "
                        + ", ".join(repr(field) for field in unknown_fields)
                    )
                _validate_known_field_types(obj, path=path, line_no=line_no)
            sample_id = obj.get("id")
            text = obj.get("text")
            if not isinstance(sample_id, str) or not sample_id:
                raise JsonlTestListError(
                    f"{path}: line {line_no}: 'id' must be a non-empty string"
                )
            if not isinstance(text, str) or not text:
                raise JsonlTestListError(
                    f"{path}: line {line_no}: 'text' must be a non-empty string"
                )
            if sample_id in id_lines:
                raise JsonlTestListError(
                    f"{path}: line {line_no}: duplicate sample id {sample_id!r}; "
                    f"first declared at line {id_lines[sample_id]}"
                )
            id_lines[sample_id] = line_no

            sample = {
                "id": sample_id,
                "text": text,
                "ref_audio": obj.get("ref_audio"),
                "ref_text": obj.get("ref_text"),
                "language_id": obj.get("language_id"),
                "language_name": obj.get("language_name"),
                "duration": obj.get("duration"),
                "speed": obj.get("speed"),
                "final_duration": obj.get("final_duration"),
                "final_duration_samples": obj.get("final_duration_samples"),
                "instruct": obj.get("instruct"),
            }
            samples.append(sample)
    return samples
