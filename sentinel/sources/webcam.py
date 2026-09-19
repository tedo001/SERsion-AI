"""Webcam / capture-device source."""

from __future__ import annotations

import time
from typing import Any, List, Optional

import cv2

from sentinel.config import LOGGER
from sentinel.models import FramePacket, SourceKind
from sentinel.sources.base import BaseVideoSource, SourceError, SourceInfo

__all__ = ["WebcamSource"]


class WebcamSource(BaseVideoSource):
    """Webcam / capture-device source."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720) -> None:
        super().__init__()
        self.device_index = int(device_index)
        self.requested_size = (int(width), int(height))
        self._cap: Optional[Any] = None
        self._fail_streak = 0

    @staticmethod
    def enumerate_devices(max_devices: int = 6) -> List[int]:
        """Probe device indices.  Cheap but not free - call sparingly."""
        found: List[int] = []
        for index in range(max_devices):
            cap = None
            try:
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ok, _frame = cap.read()
                    if ok:
                        found.append(index)
            except Exception:  # noqa: BLE001 - probing must never raise
                continue
            finally:
                if cap is not None:
                    cap.release()
        return found

    def open(self) -> None:
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"Camera {self.device_index} is unavailable.\n"
                "It may be disconnected, in use by another application, or "
                "blocked by the operating system's camera privacy settings."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_size[1])
        # A small internal buffer keeps webcam latency low.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - unsupported on some backends
            pass

        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise SourceError(
                f"Camera {self.device_index} opened but returned no frames."
            )
        height, width = probe.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (1.0 <= fps <= 240.0):
            fps = 30.0

        self._cap = cap
        self._opened = True
        self._index = 0
        self._fail_streak = 0
        self._info = SourceInfo(
            kind=SourceKind.WEBCAM,
            name=f"Camera {self.device_index}",
            width=width, height=height, fps=fps,
            detail=f"device index {self.device_index}",
        )
        LOGGER.info("Opened camera %d (%dx%d @ %.1f fps)",
                    self.device_index, width, height, fps)

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            self._fail_streak += 1
            if self._fail_streak > 60:
                raise SourceError("Camera stopped delivering frames (disconnected?).")
            return None
        self._fail_streak = 0
        self._index += 1
        return FramePacket(
            frame=frame, index=self._index, timestamp=time.time(),
            source_name=self._info.name,
        )

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Camera release error: %s", exc)
            self._cap = None
        self._opened = False
