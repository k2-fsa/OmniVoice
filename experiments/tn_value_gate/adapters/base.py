"""Shared adapter result contract."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["success", "partial", "unsupported", "runtime_error"]


@dataclass(frozen=True)
class NormalizationResult:
    output_text: str
    supported: bool
    status: Status
    error_type: str | None
    error_message: str | None
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
