"""Rolling time-series and session counters.

Thread-safe: written by the inference worker, read by the GUI thread.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Sequence, Tuple

from sentinel.models import EventType, SecurityEvent, Severity, Timings

__all__ = ["AnalyticsSnapshot", "AnalyticsEngine"]


@dataclass(slots=True)
class AnalyticsSnapshot:
    """Aggregated counters read by the dashboard on its refresh timer."""

    people: int = 0
    vehicles: int = 0
    others: int = 0
    tracks: int = 0
    total_objects: int = 0
    peak_people: int = 0
    events_today: int = 0
    warnings: int = 0
    criticals: int = 0
    infos: int = 0
    entered: int = 0
    exited: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    detect_ms: float = 0.0
    track_ms: float = 0.0
    uptime_s: float = 0.0
    class_histogram: Tuple[Tuple[str, int], ...] = ()


class AnalyticsEngine:
    """Rolling time-series + counters.  Thread-safe (written by the worker,
    read by the GUI)."""

    HISTORY = 180   # samples retained per series (~1-3 minutes)

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started = time.time()
        self.people_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.vehicle_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.fps_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.latency_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.track_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.event_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)

        self._events_bucket: Deque[float] = deque(maxlen=2000)
        self._counts = Counter()
        self._class_histogram = Counter()
        self._peak_people = 0
        self._entered = 0
        self._exited = 0
        self._last_event_flush = time.time()
        self._events_since_flush = 0

    def reset(self) -> None:
        with self._lock:
            for series in (self.people_series, self.vehicle_series, self.fps_series,
                           self.latency_series, self.track_series, self.event_series):
                series.clear()
            self._events_bucket.clear()
            self._counts.clear()
            self._class_histogram.clear()
            self._peak_people = 0
            self._entered = 0
            self._exited = 0
            self._started = time.time()

    def record_frame(
        self, *, people: int, vehicles: int, others: int, tracks: int,
        fps: float, timings: Timings, class_names: Sequence[str],
    ) -> None:
        with self._lock:
            now = time.time()
            self.people_series.append((now, float(people)))
            self.vehicle_series.append((now, float(vehicles)))
            self.fps_series.append((now, float(fps)))
            self.latency_series.append((now, float(timings.total_ms)))
            self.track_series.append((now, float(tracks)))
            self._peak_people = max(self._peak_people, people)
            for name in class_names:
                self._class_histogram[name] += 1

            # Events-per-interval series, flushed once a second.
            if now - self._last_event_flush >= 1.0:
                self.event_series.append((now, float(self._events_since_flush)))
                self._events_since_flush = 0
                self._last_event_flush = now

    def record_events(self, events: Sequence[SecurityEvent]) -> None:
        with self._lock:
            for event in events:
                self._events_bucket.append(event.timestamp)
                self._counts[event.severity.value] += 1
                self._counts[event.event_type.value] += 1
                self._events_since_flush += 1
                if event.event_type is EventType.PERSON_ENTERED:
                    self._entered += 1
                elif event.event_type is EventType.PERSON_EXITED:
                    self._exited += 1

    def snapshot(
        self, *, people: int, vehicles: int, others: int, tracks: int,
        fps: float, timings: Timings,
    ) -> AnalyticsSnapshot:
        with self._lock:
            midnight = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
            events_today = sum(1 for ts in self._events_bucket if ts >= midnight)
            return AnalyticsSnapshot(
                people=people, vehicles=vehicles, others=others, tracks=tracks,
                total_objects=people + vehicles + others,
                peak_people=self._peak_people,
                events_today=events_today,
                warnings=self._counts.get(Severity.WARNING.value, 0),
                criticals=self._counts.get(Severity.CRITICAL.value, 0),
                infos=self._counts.get(Severity.INFO.value, 0),
                entered=self._entered, exited=self._exited,
                fps=fps, latency_ms=timings.total_ms,
                detect_ms=timings.detect_ms, track_ms=timings.track_ms,
                uptime_s=time.time() - self._started,
                class_histogram=tuple(self._class_histogram.most_common(8)),
            )

    def series_values(self, name: str) -> List[float]:
        """Copy a series' values for painting (never hand the deque to the GUI)."""
        with self._lock:
            series = getattr(self, f"{name}_series", None)
            if series is None:
                return []
            return [v for _t, v in series]
