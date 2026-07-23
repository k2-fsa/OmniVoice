"""String, interval, and paired-comparison metrics."""

import math
import random
import statistics


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def normalized_cer(output: str, gold: str) -> float:
    return edit_distance(output, gold) / max(1, len(gold))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def mcnemar_exact(left: list[bool], right: list[bool]) -> dict:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    return {"left_only": left_only, "right_only": right_only,
            "discordant": discordant, "two_sided_exact_p": p_value}


def paired_bootstrap(left: list[bool], right: list[bool], seed: int = 20260722,
                     samples: int = 10000) -> dict:
    if not left:
        return {"difference": None, "ci95": None, "samples": samples}
    rng = random.Random(seed)
    diffs = []
    size = len(left)
    for _ in range(samples):
        indices = [rng.randrange(size) for _ in range(size)]
        diffs.append(sum(right[i] - left[i] for i in indices) / size)
    diffs.sort()
    return {"difference": sum(right) / size - sum(left) / size,
            "ci95": [diffs[int(samples * 0.025)], diffs[int(samples * 0.975)]],
            "samples": samples}


def latency_summary(values: list[float]) -> dict:
    return {"mean_ms": statistics.fmean(values) if values else None,
            "median_ms": statistics.median(values) if values else None}
