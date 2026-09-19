"""Vision pipeline stages that sit between tracking and the UI."""

from __future__ import annotations

from sentinel.vision.analytics import AnalyticsEngine, AnalyticsSnapshot
from sentinel.vision.events import EventEngine
from sentinel.vision.renderer import FrameRenderer
from sentinel.vision.supervision_adapter import SupervisionAdapter
from sentinel.vision.zones import default_lines, default_zones

__all__ = [
    "AnalyticsEngine", "AnalyticsSnapshot", "EventEngine", "FrameRenderer",
    "SupervisionAdapter", "default_zones", "default_lines",
]



