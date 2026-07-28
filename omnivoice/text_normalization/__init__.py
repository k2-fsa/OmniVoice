"""Production-oriented, candidate-driven Vietnamese text normalization."""

from .detector import (
    DEFAULT_MODEL_PATH,
    BamiBertDetector,
    DetectorConfig,
    DetectorLoadError,
    configure_bamibert,
    convert_predictions,
    get_bamibert_detector,
)
from .pipeline import normalize_candidates, normalize_from_ner
from .types import CandidateSpan, Diagnostic, EntityLabel, NormalizationResult

__all__ = [
    "DEFAULT_MODEL_PATH",
    "BamiBertDetector",
    "CandidateSpan",
    "DetectorConfig",
    "DetectorLoadError",
    "Diagnostic",
    "EntityLabel",
    "NormalizationResult",
    "convert_predictions",
    "configure_bamibert",
    "get_bamibert_detector",
    "normalize_candidates",
    "normalize_from_ner",
]
