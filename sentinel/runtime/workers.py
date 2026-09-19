"""Producer / consumer worker threads.

    CaptureWorker (QThread)   -> bounded FrameQueue (drops stale frames)
                              -> InferenceWorker (QThread)
                                 detect -> track -> zones/events -> annotate
                              -> resultReady signal -> GUI repaint timer

Every expensive operation lives in these two threads.  The GUI thread only ever
receives finished :class:`FrameResult` objects through queued signals.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from sentinel.config import LOGGER
from sentinel.detection import (
    BaseDetector, DetectorError, NullDetector, create_detector,
)
from sentinel.models import (
    AppSettings, BackendKind, Detection, FramePacket, FrameResult, LineConfig,
    ModelState, Severity, SourceKind, Timings, TrackedObject, TrackPhase,
    ZoneConfig,
)
from sentinel.runtime.frame_queue import FrameQueue
from sentinel.runtime.telemetry import PerformanceMonitor, SystemTelemetry
from sentinel.sources import BaseVideoSource, SourceError, SourceInfo
from sentinel.storage import EventDatabase
from sentinel.tracking import KalmanTracker
from sentinel.vision import (
    AnalyticsEngine, EventEngine, FrameRenderer, SupervisionAdapter,
)

__all__ = ["CaptureWorker", "InferenceWorker"]


class CaptureWorker(QThread):
    """Producer thread: pulls frames from the active source into the queue.

    Owns the source exclusively.  All control operations (pause, seek,
    restart) are applied inside the thread loop via simple flags, so the GUI
    thread never touches a cv2.VideoCapture.
    """

    sourceOpened = pyqtSignal(object)     # SourceInfo
    sourceFailed = pyqtSignal(str)
    sourceEnded = pyqtSignal()
    statsUpdated = pyqtSignal(float, int)  # capture fps, frame index

    def __init__(self, frame_queue: FrameQueue, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CaptureWorker")
        self._queue = frame_queue
        self._source: Optional[BaseVideoSource] = None
        self._running = False
        self._paused = False
        self._target_fps = 25.0
        self._frame_skip = 0
        self._seek_request: Optional[int] = None
        self._restart_request = False
        self._mutex = threading.Lock()
        self._capture_fps = 0.0
        self._fps_window: Deque[float] = deque(maxlen=30)

    # -- control (called from the GUI thread) -----------------------------
    def set_source(self, source: BaseVideoSource) -> None:
        with self._mutex:
            self._source = source

    def set_target_fps(self, fps: float) -> None:
        self._target_fps = float(max(1.0, min(120.0, fps)))

    def set_frame_skip(self, skip: int) -> None:
        self._frame_skip = max(0, int(skip))

    def pause(self, paused: bool) -> None:
        self._paused = bool(paused)

    @property
    def is_paused(self) -> bool:
        return self._paused

    def request_seek(self, frame_index: int) -> None:
        self._seek_request = int(frame_index)

    def request_restart(self) -> None:
        self._restart_request = True

    def stop(self) -> None:
        self._running = False

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    @property
    def source_info(self) -> Optional[SourceInfo]:
        source = self._source
        return source.info if source is not None else None

    # -- thread body ------------------------------------------------------
    def run(self) -> None:  # noqa: C901 - a capture loop is inherently branchy
        with self._mutex:
            source = self._source
        if source is None:
            self.sourceFailed.emit("No video source configured.")
            return

        try:
            source.open()
        except SourceError as exc:
            self.sourceFailed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - unexpected backend errors
            LOGGER.exception("Unexpected source error")
            self.sourceFailed.emit(f"Unexpected source error: {exc}")
            return

        self.sourceOpened.emit(source.info)
        self._running = True
        frame_counter = 0
        last_emit = 0.0
        failure_streak = 0

        try:
            while self._running:
                loop_start = time.perf_counter()

                if self._paused:
                    self.msleep(40)
                    continue

                if self._restart_request:
                    self._restart_request = False
                    source.restart()
                    self._queue.clear()

                if self._seek_request is not None:
                    target, self._seek_request = self._seek_request, None
                    source.seek(target)
                    self._queue.clear()

                try:
                    packet = source.read()
                except SourceError as exc:
                    self.sourceFailed.emit(str(exc))
                    break
                except Exception as exc:  # noqa: BLE001 - one bad frame must
                    # never take the application down.
                    LOGGER.warning("Frame read error: %s", exc)
                    failure_streak += 1
                    if failure_streak > 50:
                        self.sourceFailed.emit(f"Capture aborted: {exc}")
                        break
                    self.msleep(20)
                    continue

                if packet is None:
                    failure_streak += 1
                    if source.info.kind is SourceKind.FILE and failure_streak > 3:
                        self.sourceEnded.emit()
                        break
                    if failure_streak > 150:
                        self.sourceFailed.emit("Source stopped delivering frames.")
                        break
                    self.msleep(10)
                    continue

                failure_streak = 0
                frame_counter += 1

                # Frame skipping: a cheap throughput lever for slow hardware.
                if self._frame_skip > 0 and (frame_counter % (self._frame_skip + 1)) != 0:
                    continue

                self._queue.put(packet)

                # Capture FPS over a rolling window.
                now = time.perf_counter()
                self._fps_window.append(now)
                if len(self._fps_window) >= 2:
                    span = self._fps_window[-1] - self._fps_window[0]
                    if span > 0:
                        self._capture_fps = (len(self._fps_window) - 1) / span
                if now - last_emit > 0.5:
                    last_emit = now
                    self.statsUpdated.emit(self._capture_fps, source.position)

                # FPS limiter keeps a fast file source from saturating the CPU.
                budget = 1.0 / self._target_fps
                elapsed = time.perf_counter() - loop_start
                remaining = budget - elapsed
                if remaining > 0:
                    self.msleep(int(remaining * 1000))
        finally:
            # Releasing the capture here guarantees the device is freed even
            # if the loop exits through an exception path.
            try:
                source.release()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Source release error: %s", exc)
            LOGGER.info("Capture worker finished")


class InferenceWorker(QThread):
    """Consumer thread: detect -> track -> zones/events -> annotate.

    This is where every expensive operation lives.  The GUI thread only ever
    receives finished FrameResult objects through a queued signal.
    """

    resultReady = pyqtSignal(object)          # FrameResult
    eventsRaised = pyqtSignal(object)         # List[SecurityEvent]
    modelStateChanged = pyqtSignal(object, str)   # ModelState, message
    errorRaised = pyqtSignal(str)

    def __init__(
        self, frame_queue: FrameQueue, settings: AppSettings,
        analytics: AnalyticsEngine, database: EventDatabase,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("InferenceWorker")
        self._queue = frame_queue
        self._settings = settings
        self._analytics = analytics
        self._database = database

        self._detector: BaseDetector = NullDetector(settings)
        self._tracker = KalmanTracker(
            settings.max_age, settings.min_hits, settings.iou_threshold
        )
        self._events = EventEngine(settings)
        self._adapter = SupervisionAdapter()
        self._renderer = FrameRenderer(self._adapter)
        self._monitor = PerformanceMonitor()

        self._zones: List[ZoneConfig] = []
        self._lines: List[LineConfig] = []
        self._running = False
        self._mutex = threading.RLock()
        self._backend_request: Optional[Tuple[BackendKind, Optional[str]]] = None
        self._unload_request = False
        self._reset_request = False

        self._fps_window: Deque[float] = deque(maxlen=30)
        self._pipeline_fps = 0.0
        self._frames_processed = 0
        self._events_emitted = 0
        self._inference_failures = 0
        self._last_stat_write = 0.0

    # -- introspection ----------------------------------------------------
    @property
    def pipeline_fps(self) -> float:
        return self._pipeline_fps

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def events_emitted(self) -> int:
        return self._events_emitted

    @property
    def detector(self) -> BaseDetector:
        return self._detector

    @property
    def telemetry(self) -> SystemTelemetry:
        return self._monitor.sample(self._detector.device)

    # -- control ----------------------------------------------------------
    def configure_zones(
        self, zones: Sequence[ZoneConfig], lines: Sequence[LineConfig],
        source_name: str = "",
    ) -> None:
        with self._mutex:
            self._zones = list(zones)
            self._lines = list(lines)
            self._events.configure(self._zones, self._lines, source_name)

    def apply_settings(self, settings: AppSettings) -> None:
        """Hot-apply settings.  Threshold / overlay changes take effect on
        the next frame; a backend change is queued for the worker loop."""
        with self._mutex:
            self._settings = settings
            self._detector.update_settings(settings)
            self._events.update_settings(settings)
            self._tracker.configure(
                settings.max_age, settings.min_hits, settings.iou_threshold
            )

    def request_backend(self, backend: Optional[BackendKind],
                        variant: Optional[str] = None) -> None:
        """Ask the worker to (re)load a detection backend, or unload with None.

        ``variant`` names the checkpoint / size class to load; the worker
        applies it on its own thread so the GUI never blocks on a download.
        """
        with self._mutex:
            if backend is None:
                self._unload_request = True
                self._backend_request = None
            else:
                self._backend_request = (backend, variant)
                self._unload_request = False

    def request_reset(self) -> None:
        self._reset_request = True

    def stop(self) -> None:
        self._running = False

    # -- backend management ----------------------------------------------
    def _swap_backend(self, backend: BackendKind,
                      variant: Optional[str] = None) -> None:
        """Load a new detector, keeping the old one until the new one is up.

        Model switching happens without restarting the application; if the
        new backend fails to load, the pipeline falls back to NullDetector
        and keeps running (simulation stays fully functional).
        """
        old = self._detector
        self.modelStateChanged.emit(ModelState.LOADING, f"Loading {backend.value} ...")
        detector = create_detector(backend, self._settings, variant)
        try:
            detector.load_model()
        except DetectorError as exc:
            LOGGER.error("Backend load failed: %s", exc)
            self.modelStateChanged.emit(detector.state, str(exc))
            self.errorRaised.emit(str(exc))
            with self._mutex:
                self._detector = NullDetector(self._settings)
                self._detector.load_model()
            try:
                old.unload_model()
            except Exception:  # noqa: BLE001
                pass
            return
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.exception("Unexpected backend error")
            self.modelStateChanged.emit(ModelState.ERROR, str(exc))
            self.errorRaised.emit(f"Unexpected model error: {exc}")
            return

        with self._mutex:
            self._detector = detector
        try:
            old.unload_model()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Old detector unload error: %s", exc)
        self._tracker.reset()
        self.modelStateChanged.emit(detector.state, detector.status_message)

    def _unload_backend(self) -> None:
        with self._mutex:
            old = self._detector
            self._detector = NullDetector(self._settings)
            self._detector.load_model()
        try:
            old.unload_model()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Unload error: %s", exc)
        self.modelStateChanged.emit(ModelState.UNLOADED, "No detection backend active")

    # -- thread body ------------------------------------------------------
    def run(self) -> None:
        self._running = True
        LOGGER.info("Inference worker started")
        while self._running:
            # Service pending control requests before touching a frame.
            if self._backend_request is not None:
                backend, variant = self._backend_request
                self._backend_request = None
                self._swap_backend(backend, variant)
            if self._unload_request:
                self._unload_request = False
                self._unload_backend()
            if self._reset_request:
                self._reset_request = False
                self._tracker.reset()
                self._events.reset()

            packet = self._queue.get_latest(timeout=0.2)
            if packet is None:
                continue
            try:
                result = self._process_packet(packet)
            except Exception as exc:  # noqa: BLE001 - a single bad frame must
                # never terminate the pipeline or the GUI.
                LOGGER.exception("Frame processing error")
                self._inference_failures += 1
                if self._inference_failures in (1, 10, 50):
                    self.errorRaised.emit(
                        f"Frame processing error ({self._inference_failures}): {exc}"
                    )
                continue
            if result is not None:
                self.resultReady.emit(result)
        LOGGER.info("Inference worker finished")

    def _process_packet(self, packet: FramePacket) -> Optional[FrameResult]:
        frame = packet.frame
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return None   # invalid frame: skip silently

        with self._mutex:
            settings = self._settings
            detector = self._detector
            zones = list(self._zones)
            lines = list(self._lines)

        timings = Timings()
        height, width = frame.shape[:2]

        # ---- Detection ---------------------------------------------------
        t0 = time.perf_counter()
        detections: List[Detection] = []
        synthetic_used = False
        use_model = detector.is_ready and not isinstance(detector, NullDetector)
        if packet.synthetic_detections is not None and not (
            use_model and settings.detect_on_simulation
        ):
            # Simulation ground truth: the path that makes the product
            # demonstrable with no model weights installed.
            detections = [
                d for d in packet.synthetic_detections
                if d.confidence >= settings.confidence
            ]
            if settings.person_classes_only:
                detections = [d for d in detections if d.is_person]
            synthetic_used = True
        elif use_model:
            try:
                detections = detector.predict(frame)
            except DetectorError as exc:
                self._inference_failures += 1
                if self._inference_failures in (1, 5, 25):
                    self.errorRaised.emit(str(exc))
                detections = []
        timings.detect_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Tracking ----------------------------------------------------
        t0 = time.perf_counter()
        if settings.tracking_enabled:
            tracks = self._tracker.update(detections)
            lost = self._tracker.pop_lost()
        else:
            # Tracking disabled: surface detections as ephemeral pseudo-tracks
            # so the renderer and counters keep working.
            tracks = [
                TrackedObject(
                    track_id=-1 - i, bbox=d.bbox, class_id=d.class_id,
                    class_name=d.class_name, confidence=d.confidence,
                    position=d.center, velocity=(0.0, 0.0), age=1, hits=1,
                    time_since_update=0, phase=TrackPhase.CONFIRMED,
                    last_seen=packet.timestamp,
                )
                for i, d in enumerate(detections)
            ]
            lost = []
        timings.track_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Zones / events ----------------------------------------------
        t0 = time.perf_counter()
        if settings.tracking_enabled:
            events, zone_status, alert_ids = self._events.process(
                tracks, lost, width, height
            )
        else:
            events, zone_status, alert_ids = [], [], set()
        timings.event_ms = (time.perf_counter() - t0) * 1000.0

        people = sum(1 for t in tracks if t.is_person)
        vehicles = sum(1 for t in tracks if t.is_vehicle)
        others = len(tracks) - people - vehicles

        # ---- Annotation ---------------------------------------------------
        t0 = time.perf_counter()
        canvas = frame.copy()   # never mutate the captured frame in place
        if settings.show_zones and zones:
            status_map = {z.zone_id: z for z in zone_status}
            self._renderer.draw_zones(canvas, zones, status_map)
            self._renderer.draw_lines(canvas, lines, self._events.line_counts)
        self._renderer.draw_tracks(canvas, tracks, settings, alert_ids)
        self._renderer.draw_hud(
            canvas, fps=self._pipeline_fps, mode=settings.mode,
            source=packet.source_name, detector=detector.display_name,
            people=people, vehicles=vehicles, tracks=len(tracks),
            synthetic=packet.synthetic, settings=settings,
            critical=sum(1 for e in events if e.severity is Severity.CRITICAL),
        )
        timings.annotate_ms = (time.perf_counter() - t0) * 1000.0
        # Queue latency: wall-clock age of the frame minus the time this
        # thread spent on it.  packet.timestamp is time.time(), so the
        # comparison must use the same clock (not perf_counter).
        timings.capture_ms = max(
            0.0, (time.time() - packet.timestamp) * 1000.0 - timings.total_ms
        )

        # ---- Bookkeeping --------------------------------------------------
        now = time.perf_counter()
        self._fps_window.append(now)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self._pipeline_fps = (len(self._fps_window) - 1) / span
        self._frames_processed += 1

        if events:
            self._events_emitted += len(events)
            self._database.insert_events(events)
            self._analytics.record_events(events)
            self.eventsRaised.emit(events)

        self._analytics.record_frame(
            people=people, vehicles=vehicles, others=others, tracks=len(tracks),
            fps=self._pipeline_fps, timings=timings,
            class_names=[t.class_name for t in tracks],
        )

        # Periodic statistics row (every 5s) for the sessions history.
        if time.time() - self._last_stat_write > 5.0:
            self._last_stat_write = time.time()
            telemetry = self._monitor.sample(detector.device)
            self._database.record_statistics(
                fps=self._pipeline_fps, people=people, vehicles=vehicles,
                tracks=len(tracks), latency_ms=timings.total_ms,
                cpu_percent=telemetry.cpu_percent, ram_mb=telemetry.process_mb,
            )

        total_in, total_out = self._events.total_in_out
        return FrameResult(
            frame=canvas, index=packet.index, timestamp=packet.timestamp,
            detections=detections, tracks=tracks, events=events,
            zone_status=zone_status, people=people, vehicles=vehicles,
            others=others, timings=timings, source_name=packet.source_name,
            detector_label=("Synthetic GT" if synthetic_used
                            else detector.display_name),
            synthetic_detections=synthetic_used,
            line_in=total_in, line_out=total_out,
            occupancy=people,
        )

    def shutdown(self) -> None:
        """Release model resources from the worker's own context."""
        try:
            self._detector.unload_model()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Detector shutdown error: %s", exc)
