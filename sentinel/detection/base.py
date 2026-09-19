"""Detector interface and device resolution.

Every backend implements :class:`BaseDetector`; nothing outside this package
may know which backend is running.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sentinel.config import LOGGER
from sentinel.deps import DEP_TORCH
from sentinel.models import AppSettings, Detection, DevicePref, ModelState

__all__ = [
    "DetectorError", "BaseDetector", "NullDetector",
    "resolve_torch_device", "cuda_device_name",
]


class DetectorError(RuntimeError):
    """Raised when a backend cannot be loaded or a prediction fails fatally."""

def resolve_torch_device(pref: DevicePref) -> str:
    """Resolve a device preference into a concrete torch device string.

    CUDA is only selected when torch reports it as actually available, so the
    application degrades to CPU instead of crashing on machines without a GPU.
    """
    torch = DEP_TORCH.module
    if torch is None:
        return "cpu"
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - broken driver installs raise all sorts
        cuda_ok = False
    if pref is DevicePref.CPU:
        return "cpu"
    if pref is DevicePref.CUDA:
        return "cuda" if cuda_ok else "cpu"
    return "cuda" if cuda_ok else "cpu"


def cuda_device_name() -> Optional[str]:
    torch = DEP_TORCH.module
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        return None
    return None

class BaseDetector(ABC):
    """Common interface every detection backend must implement.

    Contract:
        * ``load_model()``   - blocking, may raise DetectorError.
        * ``predict(frame)`` - takes a BGR uint8 ndarray, returns List[Detection]
                               in ORIGINAL frame pixel coordinates.
        * ``unload_model()`` - releases weights / GPU memory, idempotent.

    Nothing outside the detector subclasses may know which backend is running.
    """

    #: Human readable backend name shown in the UI.
    display_name: str = "Detector"

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._state: ModelState = ModelState.UNLOADED
        self._status_message: str = "Not loaded"
        self._device: str = "cpu"
        self._lock = threading.RLock()

    # -- state ---------------------------------------------------------------
    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._state is ModelState.READY

    def _set_state(self, state: ModelState, message: str) -> None:
        self._state = state
        self._status_message = message
        LOGGER.info("[%s] %s -> %s", self.display_name, state.value, message)

    # -- capability probing ---------------------------------------------------
    @staticmethod
    @abstractmethod
    def dependency_status() -> Tuple[bool, str]:
        """Return (available, human readable reason / install hint)."""

    # -- lifecycle -----------------------------------------------------------
    @abstractmethod
    def load_model(self) -> None:
        """Load weights.  Must set state to READY or raise DetectorError."""

    @abstractmethod
    def predict(self, frame: "np.ndarray") -> List[Detection]:
        """Run inference on one BGR frame."""

    @abstractmethod
    def unload_model(self) -> None:
        """Free all model resources.  Must be safe to call repeatedly."""

    # -- helpers shared by concrete backends ----------------------------------
    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    def _letterbox_scale(self, frame: "np.ndarray") -> Tuple["np.ndarray", float]:
        """Resize the long edge to the configured inference size.

        Returns the resized frame and the scale factor needed to map boxes back
        to the original resolution.  Aspect ratio is preserved (no letterbox
        padding is required because both backends accept arbitrary sizes).
        """
        target = int(self._settings.inference_size)
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= 0:
            return frame, 1.0
        scale = target / float(longest)
        if abs(scale - 1.0) < 0.02:
            return frame, 1.0
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        # Recompute the exact scale from the rounded size for correct rescaling.
        return resized, w / float(new_w)

    @staticmethod
    def _clip_box(
        box: Sequence[float], width: int, height: int
    ) -> Optional[Tuple[float, float, float, float]]:
        x1 = max(0.0, min(float(box[0]), width - 1.0))
        y1 = max(0.0, min(float(box[1]), height - 1.0))
        x2 = max(0.0, min(float(box[2]), width - 1.0))
        y2 = max(0.0, min(float(box[3]), height - 1.0))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            return None
        return (x1, y1, x2, y2)

    def _filter(self, detections: List[Detection]) -> List[Detection]:
        """Apply the person-only UI filter, if enabled."""
        if not self._settings.person_classes_only:
            return detections
        return [d for d in detections if d.is_person]

class NullDetector(BaseDetector):
    """Placeholder used when no backend is loaded.

    Keeps the pipeline running (and the simulation fully functional) instead of
    forcing the application into an error state.
    """

    display_name = "No model"

    @staticmethod
    def dependency_status() -> Tuple[bool, str]:
        return True, "Always available"

    def load_model(self) -> None:
        self._set_state(ModelState.READY, "No detection backend active")

    def predict(self, frame: "np.ndarray") -> List[Detection]:
        return []

    def unload_model(self) -> None:
        self._set_state(ModelState.UNLOADED, "Not loaded")
