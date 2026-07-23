"""Adapter for the production Vietnamese numeric-normalization backend."""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class _Normalizer(Protocol):
    def normalize_numeric(self, text: str) -> str: ...


class VietNormalizerAdapter:
    """Data-safe adapter for the numeric-only VietNormalizer fork."""

    role = "production-numeric-backend"

    def __init__(self, implementation: _Normalizer | None = None):
        if implementation is None:
            try:
                from vietnormalizer import VietnameseNormalizer
            except ImportError as error:
                raise ImportError(
                    "Vietnamese normalization requires the VietNormalizer fork "
                    "with VietnameseNormalizer.normalize_numeric(). Install the "
                    "reviewed immutable fork revision before enabling "
                    "normalize_text."
                ) from error

            implementation = VietnameseNormalizer(enable_transliteration=False)
        if not callable(getattr(implementation, "normalize_numeric", None)):
            raise RuntimeError(
                "Installed VietNormalizer does not provide normalize_numeric(); "
                "the upstream 0.2.3 package is not the production numeric backend."
            )
        self._implementation = implementation

    def normalize(self, text: str) -> str:
        try:
            output = self._implementation.normalize_numeric(text)
        except (TypeError, ValueError, RuntimeError) as error:
            logger.warning(
                "VietNormalizer numeric normalization failed (%s); keeping the "
                "complete raw target.",
                type(error).__name__,
            )
            return text
        if not isinstance(output, str):
            logger.warning("VietNormalizer returned a non-string; keeping raw text.")
            return text
        return output
