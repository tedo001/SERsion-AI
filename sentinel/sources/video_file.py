"""Local video-file source."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import cv2

from sentinel.config import LOGGER
from sentinel.models import FramePacket, SourceKind
from sentinel.sources.base import BaseVideoSource, SourceError, SourceInfo

__all__ = ["LocalVideoSource"]


class LocalVideoSource(BaseVideoSource):
    """Video file source built on cv2.VideoCapture.

    Handles missing files, unsupported containers and corrupt frames without
    ever propagating an exception into the capture thread's main loop.
    """

    SUPPORTED_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")

    def __init__(self, path: str, loop: bool = True) -> None:
        super().__init__()
        self.path = str(path)
        self.loop = loop
        self._cap: Optional[Any] = None
        self._consecutive_failures = 0

    def open(self) -> None:
        file_path = Path(self.path)
        if not file_path.exists():
            raise SourceError(f"Video file not found:\n{self.path}")
        if not file_path.is_file():
            raise SourceError(f"Not a file:\n{self.path}")
        if file_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            LOGGER.warning("Unusual video suffix '%s' - attempting anyway",
                           file_path.suffix)

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"OpenCV could not open '{file_path.name}'.\n"
                "The container or codec may be unsupported by this OpenCV build.\n"
                "Try re-encoding to H.264 MP4."
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Probe one frame: some corrupt files open successfully but decode nothing.
        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise SourceError(
                f"'{file_path.name}' opened but no frame could be decoded.\n"
                "The file may be corrupt or truncated."
            )
        if width <= 0 or height <= 0:
            height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not (1.0 <= fps <= 240.0):
            LOGGER.warning("Implausible FPS %.2f reported; assuming 25", fps)
            fps = 25.0

        self._cap = cap
        self._opened = True
        self._index = 0
        self._consecutive_failures = 0
        self._info = SourceInfo(
            kind=SourceKind.FILE,
            name=file_path.name,
            width=width, height=height, fps=fps, frame_count=max(0, count),
            seekable=count > 0,
            detail=str(file_path),
        )
        LOGGER.info("Opened video %s (%dx%d @ %.1f fps, %d frames)",
                    file_path.name, width, height, fps, count)

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.loop and self._index > 0:
                # End of file: rewind rather than tearing the pipeline down.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._index = 0
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    return None
            else:
                return None
        if frame.size == 0 or frame.ndim != 3:
            # Corrupt frame: skip it, but give up if it keeps happening.
            self._consecutive_failures += 1
            if self._consecutive_failures > 30:
                raise SourceError("Too many undecodable frames - stream aborted.")
            return None
        self._consecutive_failures = 0
        self._index += 1
        return FramePacket(
            frame=frame, index=self._index, timestamp=time.time(),
            source_name=self._info.name,
        )

    def seek(self, frame_index: int) -> bool:
        if self._cap is None or not self._info.seekable:
            return False
        target = int(max(0, min(frame_index, max(0, self._info.frame_count - 1))))
        ok = bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, target))
        if ok:
            self._index = target
        return ok

    def restart(self) -> bool:
        return self.seek(0)

    @property
    def supports_seek(self) -> bool:
        return self._info.seekable

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("VideoCapture release error: %s", exc)
            self._cap = None
        self._opened = False
