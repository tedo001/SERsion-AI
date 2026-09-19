"""Execution machinery: telemetry, the frame queue, workers and the controller.

Importing this package pulls in PyQt6 (the workers and controller are QObjects).
The pure-Python pieces are importable individually without Qt:

    from sentinel.runtime.frame_queue import FrameQueue
    from sentinel.runtime.telemetry import PerformanceMonitor
"""

from __future__ import annotations

from sentinel.runtime.controller import PipelineController
from sentinel.runtime.frame_queue import FrameQueue
from sentinel.runtime.telemetry import PerformanceMonitor, SystemTelemetry
from sentinel.runtime.workers import CaptureWorker, InferenceWorker

__all__ = [
    "FrameQueue", "PerformanceMonitor", "SystemTelemetry",
    "CaptureWorker", "InferenceWorker", "PipelineController",
]
