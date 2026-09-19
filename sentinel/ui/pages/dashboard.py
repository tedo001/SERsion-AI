"""Operations dashboard: KPI tiles, trend charts and the edge system panel."""

from __future__ import annotations

import platform
from datetime import timedelta
from typing import Dict, Optional, Tuple

from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from sentinel.models import AppMode, PipelineState
from sentinel.runtime.telemetry import SystemTelemetry
from sentinel.theme import Palette
from sentinel.ui.widgets import (
    BarMeter, PipelineDiagram, Sparkline, StatCard, StatusPill, card,
    make_scroll,
)
from sentinel.vision import AnalyticsEngine
from sentinel.vision.analytics import AnalyticsSnapshot

__all__ = ["DashboardPage"]


class DashboardPage(QWidget):
    """Operations overview: KPI tiles, trend charts and the edge panel."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)

        # -- header ------------------------------------------------------
        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("Operations Dashboard")
        title.setObjectName("H1")
        self._subtitle = QLabel("Live edge analytics")
        self._subtitle.setObjectName("Dim")
        title_box.addWidget(title)
        title_box.addWidget(self._subtitle)
        head.addLayout(title_box)
        head.addStretch(1)
        self._mode_pill = StatusPill("CLASSROOM", Palette.ACCENT)
        self._state_pill = StatusPill("IDLE", Palette.TEXT_DIM)
        head.addWidget(self._mode_pill)
        head.addWidget(self._state_pill)
        layout.addLayout(head)

        # -- KPI tiles ----------------------------------------------------
        self.cards: Dict[str, StatCard] = {}
        grid = QGridLayout()
        grid.setSpacing(10)
        definitions = [
            ("fps", "Pipeline FPS", Palette.ACCENT),
            ("latency", "Total Latency", Palette.CYAN),
            ("people", "People", Palette.OK),
            ("vehicles", "Vehicles", Palette.WARN),
            ("tracks", "Active Tracks", Palette.VIOLET),
            ("objects", "Total Objects", Palette.ACCENT),
            ("events", "Events Today", Palette.CYAN),
            ("warnings", "Warnings", Palette.WARN),
            ("criticals", "Critical Alerts", Palette.CRIT),
            ("detect", "Detect Latency", Palette.MAGENTA),
            ("track_ms", "Track Latency", Palette.VIOLET),
            ("device", "Inference Device", Palette.OK),
        ]
        for index, (key, caption, accent) in enumerate(definitions):
            widget = StatCard(caption, "-", "", accent)
            self.cards[key] = widget
            grid.addWidget(widget, index // 6, index % 6)
        for column in range(6):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)

        # -- trend charts --------------------------------------------------
        charts_frame, charts_layout = card("Trends")
        charts_grid = QGridLayout()
        charts_grid.setSpacing(10)
        self.chart_objects = Sparkline("Object count over time", Palette.ACCENT)
        self.chart_fps = Sparkline("FPS over time", Palette.OK, unit=" fps")
        self.chart_events = Sparkline("Events over time", Palette.WARN, unit="/s")
        self.chart_latency = Sparkline("Latency over time", Palette.CYAN, unit=" ms")
        charts_grid.addWidget(self.chart_objects, 0, 0)
        charts_grid.addWidget(self.chart_fps, 0, 1)
        charts_grid.addWidget(self.chart_events, 1, 0)
        charts_grid.addWidget(self.chart_latency, 1, 1)
        charts_grid.setColumnStretch(0, 1)
        charts_grid.setColumnStretch(1, 1)
        charts_layout.addLayout(charts_grid)
        layout.addWidget(charts_frame)

        # -- edge system panel ---------------------------------------------
        edge_frame, edge_layout = card("Edge System")
        self.pipeline = PipelineDiagram()
        edge_layout.addWidget(self.pipeline)

        meters = QGridLayout()
        meters.setSpacing(12)
        self.meter_cpu = BarMeter("CPU", Palette.ACCENT)
        self.meter_ram = BarMeter("System RAM", Palette.VIOLET)
        self.meter_gpu = BarMeter("GPU Memory", Palette.OK)
        self.meter_queue = BarMeter("Frame Queue", Palette.CYAN)
        meters.addWidget(self.meter_cpu, 0, 0)
        meters.addWidget(self.meter_ram, 0, 1)
        meters.addWidget(self.meter_gpu, 1, 0)
        meters.addWidget(self.meter_queue, 1, 1)
        meters.setColumnStretch(0, 1)
        meters.setColumnStretch(1, 1)
        edge_layout.addLayout(meters)

        self._edge_detail = QLabel("-")
        self._edge_detail.setObjectName("Mono")
        self._edge_detail.setWordWrap(True)
        self._edge_detail.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
        edge_layout.addWidget(self._edge_detail)
        layout.addWidget(edge_frame)
        layout.addStretch(1)

        root.addWidget(make_scroll(content))

    # -- updates -----------------------------------------------------------
    def update_metrics(
        self, snapshot: AnalyticsSnapshot, telemetry: SystemTelemetry,
        mode: AppMode, state: PipelineState, queue_fill: Tuple[int, int],
        dropped: int, detector_label: str,
    ) -> None:
        self.cards["fps"].set_value(f"{snapshot.fps:.1f}", "frames / second")
        self.cards["latency"].set_value(f"{snapshot.latency_ms:.0f} ms", "end-to-end")
        self.cards["people"].set_value(snapshot.people, f"peak {snapshot.peak_people}")
        self.cards["vehicles"].set_value(snapshot.vehicles, "in view")
        self.cards["tracks"].set_value(snapshot.tracks, "kalman tracks")
        self.cards["objects"].set_value(snapshot.total_objects, "all classes")
        self.cards["events"].set_value(snapshot.events_today, "since midnight")
        self.cards["warnings"].set_value(snapshot.warnings, "this session")
        self.cards["criticals"].set_value(snapshot.criticals, "this session")
        self.cards["detect"].set_value(f"{snapshot.detect_ms:.0f} ms", detector_label)
        self.cards["track_ms"].set_value(f"{snapshot.track_ms:.1f} ms", "association")
        device_text = "CUDA" if telemetry.device == "cuda" else "CPU"
        self.cards["device"].set_value(
            device_text, telemetry.gpu_name or platform.processor() or platform.machine()
        )

        self._mode_pill.set_status(mode.value.upper(), Palette.ACCENT)
        state_colors = {
            PipelineState.RUNNING: Palette.OK,
            PipelineState.PAUSED: Palette.WARN,
            PipelineState.ERROR: Palette.CRIT,
        }
        self._state_pill.set_status(
            state.value.upper(), state_colors.get(state, Palette.TEXT_DIM)
        )

        # Edge meters.
        if telemetry.available:
            self.meter_cpu.set_value(telemetry.cpu_percent / 100.0,
                                     f"{telemetry.cpu_percent:.0f}%")
            self.meter_ram.set_value(
                telemetry.ram_percent / 100.0,
                f"{telemetry.ram_used_mb / 1024:.1f} / "
                f"{telemetry.ram_total_mb / 1024:.1f} GB",
            )
        else:
            self.meter_cpu.set_value(0.0, "psutil not installed")
            self.meter_ram.set_value(0.0, "psutil not installed")

        if telemetry.gpu_mem_total_mb > 0:
            self.meter_gpu.set_value(
                telemetry.gpu_mem_used_mb / telemetry.gpu_mem_total_mb,
                f"{telemetry.gpu_mem_used_mb:.0f} / {telemetry.gpu_mem_total_mb:.0f} MB",
            )
        else:
            self.meter_gpu.set_value(0.0, "no CUDA device")

        used, capacity = queue_fill
        self.meter_queue.set_value(
            used / max(1, capacity), f"{used} / {capacity}  ({dropped} dropped)"
        )

        active = -1 if state is not PipelineState.RUNNING else 6
        self.pipeline.set_state(
            active,
            {
                "Capture": f"{snapshot.fps:.0f} fps",
                "Detector": detector_label,
                "Kalman Tracker": f"{snapshot.tracks} trk",
                "Event Engine": f"{snapshot.warnings + snapshot.criticals} alerts",
                "Dashboard": f"{snapshot.latency_ms:.0f} ms",
            },
            healthy=state is not PipelineState.ERROR,
        )

        gpu_util = ("n/a" if telemetry.gpu_util_percent is None
                    else f"{telemetry.gpu_util_percent:.0f}%")
        self._edge_detail.setText(
            f"device={telemetry.device}   gpu_util={gpu_util}   "
            f"process_rss={telemetry.process_mb:.0f} MB   "
            f"uptime={timedelta(seconds=int(snapshot.uptime_s))}   "
            f"in={snapshot.entered} out={snapshot.exited}"
        )

    def update_charts(self, analytics: AnalyticsEngine) -> None:
        self.chart_objects.set_values(analytics.series_values("people"))
        self.chart_fps.set_values(analytics.series_values("fps"))
        self.chart_events.set_values(analytics.series_values("event"))
        self.chart_latency.set_values(analytics.series_values("latency"))

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)
