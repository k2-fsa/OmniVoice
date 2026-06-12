from __future__ import annotations

import math
import re
from dataclasses import dataclass


TOKEN_WORD_FACTOR = 1.35


@dataclass(frozen=True)
class TokenEstimate:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class CostEstimate:
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    is_actual: bool = False


def estimate_text_tokens(text: str) -> int:
    words = re.findall(r"\S+", text or "")
    if not words:
        return 0
    return max(1, math.ceil(len(words) * TOKEN_WORD_FACTOR))


def estimate_chunk_usage(text: str, *, expected_output_tokens: int = 512) -> TokenEstimate:
    input_tokens = estimate_text_tokens(text)
    output_tokens = max(0, int(expected_output_tokens))
    return TokenEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def estimate_cost(
    usage: TokenEstimate,
    *,
    input_per_million: float,
    output_per_million: float,
    currency: str = "USD",
    is_actual: bool = False,
) -> CostEstimate:
    input_cost = usage.input_tokens * input_per_million / 1_000_000
    output_cost = usage.output_tokens * output_per_million / 1_000_000
    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        currency=currency,
        is_actual=is_actual,
    )
