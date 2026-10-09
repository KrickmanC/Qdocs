from dataclasses import dataclass
from typing import Any

import numpy


@dataclass(frozen=True)
class InitializeRequest:
    """Initialize the isolated PaddleOCR-VL pipeline."""

    pipeline_kwargs: dict[str, Any]


@dataclass(frozen=True)
class PredictRequest:
    """Run PaddleOCR-VL for a batch of BGR images."""

    images: list[numpy.ndarray]


@dataclass(frozen=True)
class ShutdownRequest:
    """Stop the isolated PaddleOCR-VL worker."""


@dataclass(frozen=True)
class ReadyResponse:
    """Confirm that the PaddleOCR-VL pipeline is ready."""


@dataclass(frozen=True)
class PredictionResponse:
    """Return JSON-compatible PaddleOCR-VL results."""

    results: list[dict[str, Any]]


@dataclass(frozen=True)
class StoppedResponse:
    """Confirm a graceful worker shutdown."""


@dataclass(frozen=True)
class ErrorResponse:
    """Return a worker failure with its remote traceback."""

    operation: str
    message: str
    traceback: str
