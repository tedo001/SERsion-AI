"""Detection backends and the factory that selects between them."""

from __future__ import annotations

from typing import Optional

from sentinel.detection.base import (
    BaseDetector, DetectorError, NullDetector, cuda_device_name,
    resolve_torch_device,
)
from sentinel.detection.rfdetr import RFDETRDetector
from sentinel.detection.rtdetr import RTDETRDetector
from sentinel.models import AppSettings, BackendKind

__all__ = [
    "BaseDetector", "DetectorError", "NullDetector", "RTDETRDetector",
    "RFDETRDetector", "create_detector", "resolve_torch_device",
    "cuda_device_name",
]


def create_detector(
    backend: BackendKind, settings: AppSettings, variant: Optional[str] = None
) -> BaseDetector:
    """Factory used by the inference worker; the only place backends are named.

    ``variant`` is the backend-specific checkpoint (an RT-DETR hub id) or model
    class name (an RF-DETR size class); None selects that backend's default.
    """
    if backend is BackendKind.RT_DETR:
        return (RTDETRDetector(settings, variant) if variant
                else RTDETRDetector(settings))
    if backend is BackendKind.RF_DETR:
        return RFDETRDetector(settings, variant or "auto")
    return NullDetector(settings)
