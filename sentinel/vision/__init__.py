"""Vision pipeline stages that sit between tracking and the UI."""

from __future__ import annotations

from sentinel.vision.analytics import AnalyticsEngine, AnalyticsSnapshot
from sentinel.vision.events import EventEngine
from sentinel.vision.night_vision import NightVisionProcessor
from sentinel.vision.thermal import ThermalProcessor, ThermalStatus
from sentinel.vision.renderer import FrameRenderer, ThermalOverlay
from sentinel.vision.supervision_adapter import SupervisionAdapter
from sentinel.vision.zones import default_lines, default_zones

__all__ = [
    "AnalyticsEngine", "AnalyticsSnapshot", "EventEngine", "FrameRenderer",
    "NightVisionProcessor", "SupervisionAdapter", "ThermalOverlay",
    "ThermalProcessor", "ThermalStatus", "default_zones", "default_lines",
]



