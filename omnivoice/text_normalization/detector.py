"""Lazy, process-local BamiBERT detector lifecycle and prediction conversion."""

from __future__ import annotations

import importlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .types import CandidateSpan

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "models" / "bamibert_augmented_best"
MODEL_PATH_ENV = "OMNIVOICE_BAMIBERT_MODEL"
DEVICE_ENV = "OMNIVOICE_BAMIBERT_DEVICE"
_MONEY_SUFFIX_RE = re.compile(
    r"[\s\u00a0\u202f]*(?:đồng|VND|VNĐ|₫|USD|\$|EUR|€|đ)"
    r"(?=$|[\s.,;:!?])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration fields that materially identify a detector instance."""

    model_path: str
    device: str

    @classmethod
    def resolve(
        cls, model_path: str | Path | None = None, device: str | None = None
    ) -> "DetectorConfig":
        configured_path = Path(
            model_path or os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
        ).expanduser()
        if not configured_path.is_absolute():
            configured_path = _PROJECT_ROOT / configured_path
        return cls(
            model_path=str(configured_path.resolve()),
            device=device or os.environ.get(DEVICE_ENV, "cpu"),
        )


class DetectorLoadError(RuntimeError):
    """A detector configuration failed to load and is cached as failed."""


def configure_bamibert(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> None:
    """Set process configuration inherited by CLI batch worker processes."""
    if model_path is not None:
        os.environ[MODEL_PATH_ENV] = str(model_path)
    if device is not None:
        os.environ[DEVICE_ENV] = device


def convert_predictions(
    text: str, predictions: Sequence[Mapping[str, Any]]
) -> list[CandidateSpan]:
    """Validate transformer predictions and bind spans to the original text."""
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("detector output must be a sequence of mappings")

    candidates: list[CandidateSpan] = []
    for index, item in enumerate(predictions):
        if not isinstance(item, Mapping):
            raise TypeError(f"prediction {index} is not a mapping")
        try:
            start = int(item["start"])
            end = int(item["end"])
            label = item.get("entity_group", item.get("entity", item.get("label")))
            score = float(item["score"]) if item.get("score") is not None else None
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed prediction {index}: {error}") from error
        if not isinstance(label, str) or not label:
            raise ValueError(f"malformed prediction {index}: label is missing")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"malformed prediction {index}: invalid span bounds")
        # The trained model may label only the numeric amount as MONEY while
        # leaving an adjacent currency token outside the entity. Bind the
        # smallest complete currency surface so deterministic validation has
        # the marker it requires. No other label is context-expanded here.
        if label.upper() == "MONEY":
            suffix = _MONEY_SUFFIX_RE.match(text, end)
            if suffix and suffix.end() > end:
                end = suffix.end()
        normalized_label = label.upper()
        if normalized_label == "ID":
            normalized_label = "IDENTIFIER"
        elif normalized_label == "UNIT":
            normalized_label = "MEASUREMENT"
        candidates.append(
            CandidateSpan(
                start=start,
                end=end,
                label=normalized_label,
                text=text[start:end],
                score=score,
            )
        )
    return candidates


class BamiBertDetector:
    """Thread-safe adapter around a Transformers token-classification pipeline."""

    def __init__(self, pipeline: Callable[[str], Any], inference_mode):
        self._pipeline = pipeline
        self._inference_mode = inference_mode
        self._prediction_lock = threading.Lock()
        model = getattr(pipeline, "model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    def __call__(self, text: str) -> list[CandidateSpan]:
        # Transformers pipelines and their tokenizers are not documented as
        # safe for concurrent mutation. Serialize prediction in Gradio workers.
        with self._prediction_lock, self._inference_mode():
            predictions = self._pipeline(text)
        return convert_predictions(text, predictions)


_CACHE_LOCK = threading.Lock()
_CACHED_CONFIG: DetectorConfig | None = None
_CACHED_DETECTOR: BamiBertDetector | None = None
_CACHED_FAILURE: DetectorLoadError | None = None


def _build_detector(config: DetectorConfig) -> BamiBertDetector:
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    pipeline = transformers.pipeline(
        "token-classification",
        model=config.model_path,
        tokenizer=config.model_path,
        aggregation_strategy="simple",
        device=config.device,
    )
    return BamiBertDetector(pipeline, torch.inference_mode)


def get_bamibert_detector(
    model_path: str | Path | None = None,
    device: str | None = None,
    *,
    factory: Callable[[DetectorConfig], BamiBertDetector] | None = None,
) -> BamiBertDetector:
    """Return the one cached detector, or the cached failure for its config.

    The cache is deliberately bounded to one configuration. A failed load is
    remembered so repeated requests do not continually retry a large model.
    Changing model path/device, restarting the process, or calling the private
    test reset helper permits a new attempt.
    """
    global _CACHED_CONFIG, _CACHED_DETECTOR, _CACHED_FAILURE
    config = DetectorConfig.resolve(model_path, device)
    with _CACHE_LOCK:
        if config == _CACHED_CONFIG:
            if _CACHED_DETECTOR is not None:
                return _CACHED_DETECTOR
            if _CACHED_FAILURE is not None:
                raise _CACHED_FAILURE
        try:
            detector = (factory or _build_detector)(config)
        except Exception as error:
            failure = DetectorLoadError(
                f"failed to load BamiBERT from {config.model_path!r} "
                f"on {config.device!r}: {error}"
            )
            _CACHED_CONFIG = config
            _CACHED_DETECTOR = None
            _CACHED_FAILURE = failure
            raise failure from error
        _CACHED_CONFIG = config
        _CACHED_DETECTOR = detector
        _CACHED_FAILURE = None
        return detector


def _reset_detector_cache() -> None:
    """Reset process-local state for deterministic lifecycle tests."""
    global _CACHED_CONFIG, _CACHED_DETECTOR, _CACHED_FAILURE
    with _CACHE_LOCK:
        _CACHED_CONFIG = None
        _CACHED_DETECTOR = None
        _CACHED_FAILURE = None
