"""Frame producers: video file, webcam and synthetic simulation."""

from __future__ import annotations

from sentinel.sources.base import BaseVideoSource, SourceError, SourceInfo
from sentinel.sources.simulation import SimActor, SimulationSource
from sentinel.sources.video_file import LocalVideoSource
from sentinel.sources.webcam import WebcamSource

__all__ = [
    "BaseVideoSource", "SourceError", "SourceInfo",
    "LocalVideoSource", "WebcamSource", "SimulationSource", "SimActor",
]



