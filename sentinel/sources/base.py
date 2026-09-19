"""Video source interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from sentinel.models import FramePacket, SourceKind

__all__ = ["SourceError", "SourceInfo", "BaseVideoSource"]


class SourceError(RuntimeError):
    """Raised when a video source cannot be opened or has failed terminally."""

@dataclass(slots=True)
class SourceInfo:
    """Metadata describing an open source, shown in the Video Sources page."""

    kind: SourceKind
    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    seekable: bool = False
    detail: str = ""

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "-"

    @property
    def duration_s(self) -> float:
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return 0.0


class BaseVideoSource(ABC):
    """Common interface for every frame producer."""

    def __init__(self) -> None:
        self._info = SourceInfo(kind=SourceKind.SIMULATION, name="unknown")
        self._index = 0
        self._opened = False

    @property
    def info(self) -> SourceInfo:
        return self._info

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def position(self) -> int:
        return self._index

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource.  Raises SourceError on failure."""

    @abstractmethod
    def read(self) -> Optional[FramePacket]:
        """Return the next frame, or None when the stream has ended."""

    @abstractmethod
    def release(self) -> None:
        """Release the resource.  Must be idempotent."""

    # Optional capabilities - the defaults keep non-seekable sources honest.
    def seek(self, frame_index: int) -> bool:
        return False

    def restart(self) -> bool:
        return False

    @property
    def supports_seek(self) -> bool:
        return False
