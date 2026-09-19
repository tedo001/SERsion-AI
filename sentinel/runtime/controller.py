"""Pipeline orchestration, independent of any particular UI.

:class:`PipelineController` owns everything with a lifecycle - the video
source, the two worker threads, the bounded frame queue, the tracker/event
configuration and the database session - and exposes that state as Qt signals.

Keeping this out of the main window means the window is only responsible for
composing widgets and translating user intent into controller calls.  It also
means the pipeline can be driven headlessly (tests, a future service mode)
without constructing a single widget.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from sentinel.config import LOGGER
from sentinel.detection import resolve_torch_device
from sentinel.models import (
    AppMode, AppSettings, BackendKind, FrameResult, LineConfig, ModelState,
    PipelineState, SecurityEvent, SourceKind, ZoneConfig,
)
from sentinel.runtime.frame_queue import FrameQueue
from sentinel.runtime.telemetry import PerformanceMonitor, SystemTelemetry
from sentinel.runtime.workers import CaptureWorker, InferenceWorker
from sentinel.sources import (
    BaseVideoSource, LocalVideoSource, SimulationSource, SourceInfo,
    WebcamSource,
)
from sentinel.storage import EventDatabase
from sentinel.vision import AnalyticsEngine

__all__ = ["PipelineController"]


class PipelineController(QObject):
    """Owns the capture/inference threads and the active video source."""

    # -- outward signals (all emitted on the GUI thread) ----------------------
    stateChanged = pyqtSignal(object)        # PipelineState
    resultReady = pyqtSignal(object)         # FrameResult
    eventsRaised = pyqtSignal(object)        # List[SecurityEvent]
    sourceOpened = pyqtSignal(object)        # SourceInfo
    sourceFailed = pyqtSignal(str)
    sourceEnded = pyqtSignal()
    captureStats = pyqtSignal(float, int)    # capture fps, frame position
    modelStateChanged = pyqtSignal(object, str)   # ModelState, message
    errorRaised = pyqtSignal(str)

    def __init__(
        self, settings: AppSettings, database: EventDatabase,
        analytics: AnalyticsEngine, parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.database = database
        self.analytics = analytics

        self.frame_queue = FrameQueue(settings.queue_size)
        self.state: PipelineState = PipelineState.IDLE
        self.zones: List[ZoneConfig] = []
        self.lines: List[LineConfig] = []

        self._capture: Optional[CaptureWorker] = None
        self._inference: Optional[InferenceWorker] = None
        self._source: Optional[BaseVideoSource] = None
        self._source_info: Optional[SourceInfo] = None
        self._source_kind: Optional[SourceKind] = None

        self._latest_result: Optional[FrameResult] = None
        self._result_lock = threading.Lock()
        self._idle_monitor = PerformanceMonitor()

        self._model_state = ModelState.UNLOADED
        self._model_message = "No detection backend active"
        self._session_frames = 0
        self._session_events = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def source_info(self) -> Optional[SourceInfo]:
        return self._source_info

    @property
    def source_kind(self) -> Optional[SourceKind]:
        return self._source_kind

    @property
    def model_state(self) -> ModelState:
        return self._model_state

    @property
    def model_message(self) -> str:
        return self._model_message

    @property
    def session_frames(self) -> int:
        return self._session_frames

    @property
    def is_running(self) -> bool:
        return self.state in (PipelineState.RUNNING, PipelineState.PAUSED)

    @property
    def is_paused(self) -> bool:
        return self._capture is not None and self._capture.is_paused

    @property
    def pipeline_fps(self) -> float:
        return self._inference.pipeline_fps if self._inference is not None else 0.0

    @property
    def queue_fill(self) -> Tuple[int, int]:
        return (self.frame_queue.size, self.frame_queue.capacity)

    @property
    def dropped_frames(self) -> int:
        return self.frame_queue.dropped

    @property
    def telemetry(self) -> SystemTelemetry:
        if self._inference is not None:
            return self._inference.telemetry
        return self._idle_monitor.sample(resolve_torch_device(self.settings.device))

    def latest_result(self) -> Optional[FrameResult]:
        with self._result_lock:
            return self._latest_result

    # ------------------------------------------------------------------
    # Source selection
    # ------------------------------------------------------------------
    def activate_simulation(self, people: int, vehicles: int) -> None:
        source = SimulationSource(self.settings.mode, people=people, vehicles=vehicles)
        self._activate(source, SourceKind.SIMULATION)

    def activate_file(self, path: str, loop: bool = True) -> None:
        self._activate(LocalVideoSource(path, loop=loop), SourceKind.FILE)

    def activate_webcam(self, device_index: int) -> None:
        self._activate(WebcamSource(device_index), SourceKind.WEBCAM)

    def _activate(self, source: BaseVideoSource, kind: SourceKind) -> None:
        """Swap the active source, restarting the pipeline around it."""
        if self.is_running:
            self.stop(emit_state=False)
        self._source = source
        self._source_kind = kind
        self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.state is PipelineState.RUNNING:
            return
        if self._source is None:
            # Simulation is always available, so an un-configured start still
            # produces a working console rather than an error.
            self._source = SimulationSource(self.settings.mode)
            self._source_kind = SourceKind.SIMULATION

        self._set_state(PipelineState.STARTING)
        self.frame_queue = FrameQueue(self.settings.queue_size)
        self.analytics.reset()

        self._capture = CaptureWorker(self.frame_queue, self)
        self._capture.set_source(self._source)
        self._capture.set_target_fps(self.settings.target_fps)
        self._capture.set_frame_skip(self.settings.frame_skip)
        self._capture.sourceOpened.connect(self._on_source_opened)
        self._capture.sourceFailed.connect(self._on_source_failed)
        self._capture.sourceEnded.connect(self._on_source_ended)
        self._capture.statsUpdated.connect(self.captureStats)

        self._inference = InferenceWorker(
            self.frame_queue, self.settings, self.analytics, self.database, self
        )
        self._inference.configure_zones(self.zones, self.lines)
        self._inference.resultReady.connect(self._on_result)
        self._inference.eventsRaised.connect(self._on_events)
        self._inference.modelStateChanged.connect(self._on_model_state)
        self._inference.errorRaised.connect(self.errorRaised)

        self._inference.start()
        self._capture.start()

        # Re-apply a backend that was loaded before this restart.
        if self._model_state is ModelState.READY:
            self._inference.request_backend(self.settings.backend)

        self._start_session()
        self._set_state(PipelineState.RUNNING)

    def stop(self, emit_state: bool = True) -> None:
        """Tear down both threads deterministically and release the device."""
        capture, self._capture = self._capture, None
        inference, self._inference = self._inference, None

        if capture is not None:
            capture.stop()
            if not capture.wait(3000):
                LOGGER.warning("Capture thread did not stop in time; terminating")
                capture.terminate()
                capture.wait(1000)
            capture.deleteLater()

        if inference is not None:
            inference.stop()
            if not inference.wait(4000):
                LOGGER.warning("Inference thread did not stop in time; terminating")
                inference.terminate()
                inference.wait(1000)
            inference.shutdown()
            inference.deleteLater()

        self.frame_queue.clear()
        if self._source is not None:
            try:
                self._source.release()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                LOGGER.debug("Source release error: %s", exc)

        self.database.end_session(self._session_frames, self._session_events)
        if emit_state:
            self._set_state(PipelineState.STOPPED)
        else:
            self.state = PipelineState.STOPPED

    def toggle_pause(self) -> bool:
        """Pause or resume capture; returns the new paused state."""
        if self._capture is None:
            return False
        paused = not self._capture.is_paused
        self._capture.pause(paused)
        self._set_state(PipelineState.PAUSED if paused else PipelineState.RUNNING)
        return paused

    def restart_source(self) -> None:
        if self._capture is not None:
            self._capture.request_restart()
        if self._inference is not None:
            self._inference.request_reset()
        self.analytics.reset()

    def seek(self, frame_index: int) -> None:
        if self._capture is not None:
            self._capture.request_seek(frame_index)

    def shutdown(self) -> None:
        if self.is_running or self.state is PipelineState.STARTING:
            self.stop(emit_state=False)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def apply_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.frame_queue.resize(settings.queue_size)
        if self._capture is not None:
            self._capture.set_target_fps(settings.target_fps)
            self._capture.set_frame_skip(settings.frame_skip)
        if self._inference is not None:
            self._inference.apply_settings(settings)

    def configure_zones(
        self, zones: Sequence[ZoneConfig], lines: Sequence[LineConfig]
    ) -> None:
        self.zones = list(zones)
        self.lines = list(lines)
        if self._inference is not None:
            self._inference.configure_zones(
                self.zones, self.lines,
                self._source_info.name if self._source_info else "",
            )

    def set_mode(self, mode: AppMode) -> None:
        """Apply a scenario switch, re-seeding a live simulation in place."""
        self.settings.mode = mode
        if isinstance(self._source, SimulationSource):
            self._source.set_mode(mode)

    def set_simulation_population(self, people: int, vehicles: int) -> None:
        if isinstance(self._source, SimulationSource):
            self._source.set_population(people, vehicles)

    def request_backend(
        self, backend: Optional[BackendKind], variant: Optional[str] = None
    ) -> bool:
        """Queue a model load (or unload with None) on the inference thread."""
        if self._inference is None:
            return False
        self._inference.request_backend(backend, variant)
        return True

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------
    def _set_state(self, state: PipelineState) -> None:
        self.state = state
        self.stateChanged.emit(state)

    def _start_session(self) -> None:
        self._session_frames = 0
        self._session_events = 0
        self.database.start_session(
            self.settings.mode.value,
            self._source_info.name if self._source_info else "none",
            self.settings.backend.value
            if self._model_state is ModelState.READY else "none",
            resolve_torch_device(self.settings.device),
        )

    @pyqtSlot(object)
    def _on_source_opened(self, info: SourceInfo) -> None:
        self._source_info = info
        if self._inference is not None:
            self._inference.configure_zones(self.zones, self.lines, info.name)
        self.sourceOpened.emit(info)

    @pyqtSlot(str)
    def _on_source_failed(self, message: str) -> None:
        self._set_state(PipelineState.ERROR)
        self.sourceFailed.emit(message)
        self.stop()

    @pyqtSlot()
    def _on_source_ended(self) -> None:
        self.sourceEnded.emit()
        self.stop()

    @pyqtSlot(object)
    def _on_result(self, result: FrameResult) -> None:
        # Keep only the newest result: a burst of worker output must never
        # queue up repaints and stall the GUI.
        with self._result_lock:
            self._latest_result = result
        self._session_frames = result.index
        self.resultReady.emit(result)

    @pyqtSlot(object)
    def _on_events(self, events: List[SecurityEvent]) -> None:
        self._session_events += len(events)
        self.eventsRaised.emit(events)

    @pyqtSlot(object, str)
    def _on_model_state(self, state: ModelState, message: str) -> None:
        self._model_state = state
        self._model_message = message
        self.modelStateChanged.emit(state, message)
