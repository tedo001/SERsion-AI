"""Bounded frame queue with a stale-frame drop policy.

When inference is slower than capture, the newest frame is what matters; older
frames are discarded rather than queued.  This bounds both memory AND latency,
which is the behaviour an edge appliance needs.
"""

from __future__ import annotations

import queue
from typing import Optional

from sentinel.models import FramePacket

__all__ = ["FrameQueue"]


class FrameQueue:
    """Bounded frame queue with a stale-frame drop policy.

    When inference is slower than capture, the newest frame is what matters; old
    frames are discarded rather than queued.  This bounds memory AND latency,
    which is the behaviour an edge appliance needs.
    """

    def __init__(self, maxsize: int = 2) -> None:
        self._queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=max(1, maxsize))
        self.dropped = 0
        self.pushed = 0

    def resize(self, maxsize: int) -> None:
        maxsize = max(1, int(maxsize))
        if maxsize == self._queue.maxsize:
            return
        replacement: "queue.Queue[FramePacket]" = queue.Queue(maxsize=maxsize)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                replacement.put_nowait(item)
            except queue.Full:
                self.dropped += 1
        self._queue = replacement

    def put(self, packet: FramePacket) -> None:
        """Enqueue, evicting the oldest frame when full."""
        self.pushed += 1
        try:
            self._queue.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()          # evict oldest
            self.dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            self.dropped += 1

    def get(self, timeout: float = 0.25) -> Optional[FramePacket]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self, timeout: float = 0.25) -> Optional[FramePacket]:
        """Block for one frame, then drain to the newest available packet."""
        packet = self.get(timeout=timeout)
        if packet is None:
            return None
        while True:
            try:
                newer = self._queue.get_nowait()
            except queue.Empty:
                return packet
            self.dropped += 1
            packet = newer

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize
