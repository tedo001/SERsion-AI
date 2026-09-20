"""Application shell.

The window composes the chrome (top bar, sidebar, page stack, dock, status bar)
and translates user intent into :class:`PipelineController` calls.  It owns no
pipeline state of its own: everything with a lifecycle lives in the controller,
and everything with widgets lives in a page.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QStackedWidget, QStatusBar,
    QVBoxLayout, QWidget,
)

from sentinel.config import (
    APP_NAME, APP_SUBTITLE, APP_VERSION, DB_PATH, LOGGER,
)
from sentinel.models import (
    AppMode, AppSettings, BackendKind, EventType, ModelState, PipelineState,
    SecurityEvent, Severity, SourceKind,
)
from sentinel.runtime import PipelineController
from sentinel.sources import SourceInfo, WebcamSource
from sentinel.storage import EventDatabase
from sentinel.storage.settings_store import SettingsManager
from sentinel.theme import Palette
from sentinel.ui.live_panel import LivePanel
from sentinel.ui.zone_editor import LineEditorDialog, ZoneEditorDialog
from sentinel.ui.pages import (
    AnalyticsPage, DashboardPage, EventsPage, LiveMonitorPage, ModelsPage,
    SettingsPage, SourcesPage, ZonesPage,
)
from sentinel.ui.widgets import StatusPill, divider, make_scroll
from sentinel.vision import AnalyticsEngine

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Top-level console window."""

    PAGES = ("Dashboard", "Live Monitor", "Video Sources", "Analytics",
             "Events", "Zones", "Models", "Settings")
    PAGE_ICONS = ("▦", "▶", "⛁", "◔", "⚑", "⬡", "◈", "⚙")

    #: Stack indices used when a handler needs to jump to a page.
    PAGE_LIVE = 1
    PAGE_SOURCES = 2
    PAGE_ZONES = 5

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - {APP_SUBTITLE}")
        self.resize(1500, 900)
        self.setMinimumSize(1040, 640)

        # -- services -----------------------------------------------------
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load_settings()
        self.database = EventDatabase()
        self.analytics = AnalyticsEngine()
        self.controller = PipelineController(
            self.settings, self.database, self.analytics, self
        )
        self.zones, self.lines = SettingsManager.load_zones(self.settings.mode)
        self.controller.configure_zones(self.zones, self.lines)

        self._recent_events: Deque[SecurityEvent] = deque(maxlen=400)
        self._shutting_down = False

        # -- UI -----------------------------------------------------------
        self._build_ui()
        self._build_status_bar()
        self._connect_pages()
        self._connect_controller()
        self.settings_manager.restore_geometry(self)
        self._apply_settings_to_pages()

        # -- timers ---------------------------------------------------------
        # One repaint timer drives all video updates; the workers never touch
        # widgets directly.  This is the GUI-throttling mechanism.
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_tick)
        self._render_timer.start(max(16, int(1000 / max(1, self.settings.gui_fps))))

        self._slow_timer = QTimer(self)
        self._slow_timer.timeout.connect(self._slow_tick)
        self._slow_timer.start(1000)

        self._log_system_event(
            f"{APP_NAME} v{APP_VERSION} ready. Simulation source available offline."
        )

    # ==================================================================
    # Construction
    # ==================================================================
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_top_bar())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.page_dashboard = DashboardPage()
        self.page_live = LiveMonitorPage(self.settings)
        self.page_sources = SourcesPage()
        self.page_analytics = AnalyticsPage()
        self.page_events = EventsPage()
        self.page_zones = ZonesPage()
        self.page_models = ModelsPage()
        self.page_settings = SettingsPage()
        for page in (self.page_dashboard, self.page_live, self.page_sources,
                     self.page_analytics, self.page_events, self.page_zones,
                     self.page_models, self.page_settings):
            self.stack.addWidget(page)
        body_layout.addWidget(self.stack, 1)
        root.addWidget(body, 1)
        self.setCentralWidget(central)

        self.page_zones.set_zones(self.zones)
        self.page_events.set_database_status(
            f"SQLite: {DB_PATH}" if self.database.available
            else f"SQLite unavailable ({self.database.error}) - events are not persisted"
        )
        self._build_right_dock()
        # The panel owns the line rows, so seed it once the dock exists.
        self.live_panel.set_lines(self.lines)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(56)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(14)

        logo = QLabel("L")
        logo.setFixedSize(30, 30)
        logo.setStyleSheet(
            f"background-color: {Palette.ACCENT}; border-radius: 8px;"
            f" color: #04101C; font-weight: 900; font-size: 15px;"
        )
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        name = QLabel(APP_NAME.upper())
        name.setObjectName("Brand")
        sub = QLabel(APP_SUBTITLE)
        sub.setObjectName("BrandSub")
        brand.addWidget(name)
        brand.addWidget(sub)
        layout.addLayout(brand)
        layout.addSpacing(10)
        layout.addWidget(divider(vertical=True))
        layout.addStretch(1)

        self.pill_system = StatusPill("SYSTEM IDLE", Palette.TEXT_DIM)
        self.pill_device = StatusPill("CPU", Palette.VIOLET)
        self.pill_model = StatusPill("NO MODEL", Palette.WARN)
        self.pill_fps = StatusPill("0.0 FPS", Palette.ACCENT)
        self.pill_source = StatusPill("NO SOURCE", Palette.TEXT_DIM)
        for pill in (self.pill_system, self.pill_device, self.pill_model,
                     self.pill_fps, self.pill_source):
            layout.addWidget(pill)

        layout.addWidget(divider(vertical=True))
        self.label_clock = QLabel("--:--:--")
        self.label_clock.setObjectName("Mono")
        self.label_clock.setStyleSheet(
            f"color: {Palette.TEXT}; font-size: 13px; font-weight: 600;"
        )
        layout.addWidget(self.label_clock)
        return bar

    def _build_sidebar(self) -> QWidget:
        container = QFrame()
        container.setObjectName("Sidebar")
        container.setFixedWidth(186)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(11, 14, 11, 14)
        layout.setSpacing(4)

        caption = QLabel("NAVIGATION")
        caption.setObjectName("Caption")
        layout.addWidget(caption)
        layout.addSpacing(4)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for index, (title, icon) in enumerate(zip(self.PAGES, self.PAGE_ICONS)):
            button = QPushButton(f"  {icon}   {title}")
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.nav_group.addButton(button, index)
            layout.addWidget(button)
        self.nav_group.button(0).setChecked(True)

        layout.addSpacing(12)
        layout.addWidget(divider())
        layout.addSpacing(10)

        caption2 = QLabel("SESSION CONTROL")
        caption2.setObjectName("Caption")
        layout.addWidget(caption2)

        self.btn_start = QPushButton("Start Pipeline")
        self.btn_start.setObjectName("Primary")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("Danger")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        for button in (self.btn_start, self.btn_pause, self.btn_stop):
            layout.addWidget(button)

        layout.addStretch(1)
        note = QLabel("Detection & tracking only.\nNo facial recognition.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 9px;")
        layout.addWidget(note)
        version = QLabel(f"v{APP_VERSION}")
        version.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 9px;")
        layout.addWidget(version)

        outer.addWidget(make_scroll(inner))
        return container

    def _build_right_dock(self) -> None:
        dock = QDockWidget("LIVE ANALYTICS", self)
        dock.setObjectName("RightDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.live_panel = LivePanel()
        dock.setWidget(make_scroll(self.live_panel, horizontal=True))
        dock.setMinimumWidth(250)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.dock_right = dock

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        bar.setSizeGripEnabled(True)
        self.setStatusBar(bar)
        self.status_labels: Dict[str, QLabel] = {}
        fields = ("Source", "Resolution", "FPS", "Device", "Model",
                  "Tracking", "Light", "Pipeline")
        for index, field in enumerate(fields):
            label = QLabel(f"{field}: -")
            label.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")
            self.status_labels[field] = label
            bar.addPermanentWidget(label)
            if index < len(fields) - 1:
                bar.addPermanentWidget(divider(vertical=True))
        self.status_message = QLabel("Ready")
        self.status_message.setStyleSheet(
            f"color: {Palette.TEXT_DIM}; font-size: 11px;"
        )
        bar.addWidget(self.status_message)

    # ==================================================================
    # Wiring
    # ==================================================================
    def _connect_pages(self) -> None:
        self.nav_group.idClicked.connect(self.stack.setCurrentIndex)
        # Sync the other way too, so programmatic page changes keep the
        # sidebar highlight correct.
        self.stack.currentChanged.connect(self._sync_nav_selection)

        self.btn_start.clicked.connect(self.controller.start)
        self.btn_stop.clicked.connect(self.controller.stop)
        self.btn_pause.clicked.connect(self._toggle_pause)

        live = self.page_live
        live.playPauseRequested.connect(self._toggle_pause)
        live.restartRequested.connect(self._on_restart)
        live.seekRequested.connect(self.controller.seek)
        live.snapshotRequested.connect(self._on_snapshot)
        live.confidenceChanged.connect(self._on_live_confidence)
        live.sourceKindChosen.connect(self._on_source_kind_chosen)
        live.modeChosen.connect(self._on_mode_chosen)

        sources = self.page_sources
        sources.simulationRequested.connect(self.controller.activate_simulation)
        sources.fileRequested.connect(self.controller.activate_file)
        sources.webcamRequested.connect(self.controller.activate_webcam)
        sources.browseRequested.connect(self._on_browse)
        sources.scanRequested.connect(self._on_scan_cameras)
        sources.populationChanged.connect(self.controller.set_simulation_population)

        self.page_analytics.refreshRequested.connect(self._reload_sessions)

        events = self.page_events
        events.filtersChanged.connect(self._reload_events)
        events.exportRequested.connect(self._on_export_csv)
        events.clearRequested.connect(self._on_clear_events)

        zones = self.page_zones
        zones.zonesChanged.connect(self._on_zones_changed)
        zones.saveRequested.connect(self._on_zones_save)
        zones.resetRequested.connect(self._on_zones_reset)
        zones.statusMessage.connect(self._set_status)

        # Editing straight from the live Zone Status panel.
        self.live_panel.editZoneRequested.connect(self._edit_zone)
        self.live_panel.editLineRequested.connect(self._edit_line)

        models = self.page_models
        models.loadRequested.connect(self._on_load_model)
        models.unloadRequested.connect(self._on_unload_model)
        models.backendChanged.connect(self._on_backend_changed)

        settings = self.page_settings
        settings.settingsApplied.connect(self._on_settings_applied)
        settings.resetRequested.connect(self._on_settings_reset)

    def _connect_controller(self) -> None:
        controller = self.controller
        controller.stateChanged.connect(self._on_state_changed)
        controller.eventsRaised.connect(self._on_events)
        controller.sourceOpened.connect(self._on_source_opened)
        controller.sourceFailed.connect(self._on_source_failed)
        controller.sourceEnded.connect(self._on_source_ended)
        controller.captureStats.connect(self._on_capture_stats)
        controller.modelStateChanged.connect(self._on_model_state)
        controller.errorRaised.connect(self._on_worker_error)

    def _sync_nav_selection(self, index: int) -> None:
        button = self.nav_group.button(index)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def _goto_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    # ==================================================================
    # Settings
    # ==================================================================
    def _apply_settings_to_pages(self) -> None:
        self.page_settings.apply_settings(self.settings)
        self.page_live.set_confidence(self.settings.confidence)
        self.page_live.set_mode(self.settings.mode)
        self.page_models.set_backend(self.settings.backend)
        self.page_dashboard.set_subtitle(
            f"{self.settings.mode.value} deployment  -  live edge analytics"
        )
        self._reload_events()
        self._reload_sessions()

    @pyqtSlot(object)
    def _on_settings_applied(self, settings: AppSettings) -> None:
        previous_mode = self.settings.mode
        self.settings = settings
        self.settings_manager.save_settings(settings)
        self.controller.apply_settings(settings)
        self._render_timer.setInterval(
            max(16, int(1000 / max(1, settings.gui_fps)))
        )
        if previous_mode is not settings.mode:
            self.switch_mode(settings.mode)
        self._apply_settings_to_pages()
        self._set_status("Settings applied")
        self._log_system_event(
            f"Settings applied (conf={settings.confidence:.2f}, "
            f"res={settings.inference_size}, device={settings.device.value})"
        )

    def _on_settings_reset(self) -> None:
        self.settings = AppSettings()
        self.settings_manager.save_settings(self.settings)
        self.controller.apply_settings(self.settings)
        self._apply_settings_to_pages()
        self._set_status("Settings restored to defaults")

    def _on_live_confidence(self, confidence: float) -> None:
        """Confidence tweaks from the Live Monitor apply immediately."""
        self.settings.confidence = confidence
        self.page_settings.set_confidence(confidence)
        self.controller.apply_settings(self.settings)

    def switch_mode(self, mode: AppMode) -> None:
        """Swap scenario: reload the zone layout and re-seed the simulation."""
        self.settings.mode = mode
        self.zones, self.lines = SettingsManager.load_zones(mode)
        self.controller.set_mode(mode)
        self.controller.configure_zones(self.zones, self.lines)
        self.page_zones.set_zones(self.zones)
        self.live_panel.set_lines(self.lines)
        self.page_live.set_mode(mode)
        self.page_settings.set_mode(mode)
        self.page_dashboard.set_subtitle(
            f"{mode.value} deployment  -  live edge analytics"
        )
        self._log_system_event(f"Deployment mode switched to {mode.value}")

    @pyqtSlot(object)
    def _on_mode_chosen(self, mode: AppMode) -> None:
        if mode is self.settings.mode:
            return
        self.switch_mode(mode)
        self.settings_manager.save_settings(self.settings)

    @pyqtSlot(object)
    def _on_source_kind_chosen(self, kind: SourceKind) -> None:
        if kind is SourceKind.SIMULATION:
            people, vehicles = self.page_sources.population()
            self.controller.activate_simulation(people, vehicles)
        elif kind is SourceKind.FILE:
            path = self.page_sources.file_path
            if not path:
                self._set_status(
                    "Choose a video file on the Video Sources page first"
                )
                self._goto_page(self.PAGE_SOURCES)
                return
            self.controller.activate_file(path, self.page_sources.loop_enabled)
        else:
            self.controller.activate_webcam(self.page_sources.selected_camera())

    # ==================================================================
    # Transport
    # ==================================================================
    def _toggle_pause(self) -> None:
        paused = self.controller.toggle_pause()
        text = "Resume" if paused else "Pause"
        self.btn_pause.setText(text)
        self.page_live.set_play_text(text)
        self.page_live.set_badge("PAUSED" if paused else None, Palette.WARN)
        self._set_status("Pipeline paused" if paused else "Pipeline resumed")

    def _on_restart(self) -> None:
        self.controller.restart_source()
        self._set_status("Source restarted")

    def _on_snapshot(self) -> None:
        result = self.controller.latest_result()
        if result is None:
            self._set_status("No frame available to save")
            return
        default = str(Path.home() / f"sentinel_{int(time.time())}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", default,
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )
        if not path:
            return
        try:
            if not cv2.imwrite(path, result.frame):
                raise OSError("OpenCV could not write the file")
            self._set_status(f"Snapshot saved to {path}")
        except Exception as exc:  # noqa: BLE001
            self._error_box("Snapshot failed", str(exc))

    # ==================================================================
    # Controller callbacks
    # ==================================================================
    @pyqtSlot(object)
    def _on_state_changed(self, state: PipelineState) -> None:
        running = state in (PipelineState.RUNNING, PipelineState.PAUSED)
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)
        self.page_live.set_transport_enabled(running)

        if state is PipelineState.RUNNING:
            self.btn_pause.setText("Pause")
            self.page_live.set_play_text("Pause")
        elif state is PipelineState.STARTING:
            self.page_live.show_placeholder(
                "STARTING", "Opening video source ...", Palette.ACCENT
            )
        elif state is PipelineState.STOPPED:
            self.page_live.show_placeholder("NO SIGNAL", "Pipeline stopped")
            self.page_zones.show_placeholder("PREVIEW", "Pipeline stopped")
            self._set_status("Pipeline stopped")
            self._log_system_event("Pipeline stopped")

    @pyqtSlot(object)
    def _on_source_opened(self, info: SourceInfo) -> None:
        duration = (f"  |  {info.frame_count} frames "
                    f"({timedelta(seconds=int(info.duration_s))})"
                    if info.frame_count else "")
        self.page_sources.set_active_source(
            f"{info.kind.value}  |  {info.name}\n"
            f"{info.resolution} @ {info.fps:.1f} fps{duration}"
            + (f"\n{info.detail}" if info.detail else "")
        )
        self.page_live.set_seekable(info.seekable, info.frame_count)
        self.page_live.set_source_kind(info.kind)
        # The SIMULATION marker is burned into the frame by the renderer so it
        # survives snapshots; the Qt overlay badge is reserved for UI state.
        self.page_live.set_badge(None)
        self._set_status(f"Source open: {info.name} ({info.resolution})")
        self._log_system_event(
            f"Source opened: {info.name} {info.resolution} @ {info.fps:.1f} fps"
        )

    @pyqtSlot(str)
    def _on_source_failed(self, message: str) -> None:
        self.page_live.show_placeholder(
            "SOURCE ERROR", message.splitlines()[0], Palette.CRIT
        )
        self._error_box("Video source error", message)
        self._log_system_event(
            f"Source error: {message.splitlines()[0]}", Severity.CRITICAL
        )

    @pyqtSlot()
    def _on_source_ended(self) -> None:
        self._set_status("End of video reached")
        self._log_system_event("End of video stream reached")

    @pyqtSlot(float, int)
    def _on_capture_stats(self, fps: float, position: int) -> None:
        info = self.controller.source_info
        self.page_live.set_position(position, info.frame_count if info else 0)

    @pyqtSlot(object)
    def _on_events(self, events: List[SecurityEvent]) -> None:
        for event in events:
            self._recent_events.append(event)
        self.live_panel.add_events(events)
        self.page_events.add_events(events)

    @pyqtSlot(object, str)
    def _on_model_state(self, state: ModelState, message: str) -> None:
        self.page_models.set_model_state(state, message, self.settings.backend)
        colors = {
            ModelState.READY: Palette.OK,
            ModelState.LOADING: Palette.ACCENT,
            ModelState.ERROR: Palette.CRIT,
            ModelState.UNAVAILABLE: Palette.WARN,
        }
        self.pill_model.set_status(
            f"{self.settings.backend.value} {state.value}".upper()
            if state is not ModelState.UNLOADED else "NO MODEL",
            colors.get(state, Palette.TEXT_DIM),
        )
        if state is ModelState.LOADING:
            self.page_live.show_placeholder("MODEL LOADING", message, Palette.ACCENT)
        self._log_system_event(f"Model {state.value}: {message}")

    @pyqtSlot(str)
    def _on_worker_error(self, message: str) -> None:
        self._set_status(message)
        self._log_system_event(message, Severity.WARNING)

    # ==================================================================
    # Render loop
    # ==================================================================
    def _render_tick(self) -> None:
        """GUI-thread repaint, throttled to the configured refresh rate."""
        result = self.controller.latest_result()
        if result is None:
            return

        # Only paint the widget that is actually visible - repainting a hidden
        # page is pure waste on an edge device.
        page = self.stack.currentIndex()
        if page == self.PAGE_LIVE:
            self.page_live.show_frame(result.frame)
        elif page == self.PAGE_ZONES:
            self.page_zones.show_frame(result.frame)

        self.live_panel.update_counts(result)

    def _slow_tick(self) -> None:
        """One-second cadence: clock, dashboard, telemetry, status bar."""
        self.label_clock.setText(datetime.now().strftime("%H:%M:%S"))

        result = self.controller.latest_result()
        fps = self.controller.pipeline_fps
        telemetry = self.controller.telemetry
        timings = result.timings if result is not None else None
        detector_label = result.detector_label if result is not None else "none"

        from sentinel.models import Timings  # local: avoids a cycle at import time

        snapshot = self.analytics.snapshot(
            people=result.people if result else 0,
            vehicles=result.vehicles if result else 0,
            others=result.others if result else 0,
            tracks=len(result.tracks) if result else 0,
            fps=fps, timings=timings or Timings(),
        )

        self.page_dashboard.update_metrics(
            snapshot, telemetry, self.settings.mode, self.controller.state,
            self.controller.queue_fill, self.controller.dropped_frames,
            detector_label,
        )
        self.page_dashboard.update_charts(self.analytics)
        self.page_analytics.update_series(self.analytics)
        self.page_analytics.update_classes(snapshot)
        self.page_analytics.set_summary(self._summary_text(snapshot, telemetry))

        self._update_pills(fps, telemetry)
        self._update_status_bar(fps, telemetry, detector_label)

        # Clear a stale alert banner after a quiet period.
        if self._recent_events:
            age = time.time() - self._recent_events[-1].timestamp
            if age > self.live_panel.ALERT_HOLD_S:
                self.live_panel.clear_alert()

    def _update_pills(self, fps: float, telemetry) -> None:
        state_colors = {
            PipelineState.RUNNING: Palette.OK,
            PipelineState.PAUSED: Palette.WARN,
            PipelineState.ERROR: Palette.CRIT,
        }
        self.pill_system.set_status(
            f"SYSTEM {self.controller.state.value.upper()}",
            state_colors.get(self.controller.state, Palette.TEXT_DIM),
        )
        device = telemetry.device.upper()
        suffix = (f" {telemetry.gpu_util_percent:.0f}%"
                  if telemetry.gpu_util_percent is not None else "")
        self.pill_device.set_status(
            f"{device}{suffix}",
            Palette.OK if device == "CUDA" else Palette.VIOLET,
        )
        self.pill_fps.set_status(
            f"{fps:.1f} FPS", Palette.OK if fps >= 12 else Palette.WARN
        )
        info = self.controller.source_info
        if info is not None:
            self.pill_source.set_status(info.kind.value.upper(), Palette.ACCENT)

    def _update_status_bar(self, fps: float, telemetry, detector_label: str) -> None:
        info = self.controller.source_info
        self.status_labels["Source"].setText(
            f"Source: {info.name if info else '-'}")
        self.status_labels["Resolution"].setText(
            f"Resolution: {info.resolution if info else '-'}")
        self.status_labels["FPS"].setText(f"FPS: {fps:.1f}")
        self.status_labels["Device"].setText(f"Device: {telemetry.device.upper()}")
        self.status_labels["Model"].setText(f"Model: {detector_label}")
        self.status_labels["Tracking"].setText(
            f"Tracking: {'on' if self.settings.tracking_enabled else 'off'}")
        self.status_labels["Light"].setText(f"Light: {self._light_label()}")
        self.status_labels["Pipeline"].setText(
            f"Pipeline: {self.controller.state.value}")

    def _light_label(self) -> str:
        """Scene-light summary for the status bar."""
        result = self.controller.latest_result()
        if result is None:
            return "-"
        night = result.night
        thermal = result.thermal
        parts = [f"{night.label} ({night.luminance:.0f})"]
        if thermal is not None and getattr(thermal, "active", False):
            parts.append(
                "thermal" if thermal.radiometric else "false-colour"
            )
        return "  ".join(parts)

    def _summary_text(self, snapshot, telemetry) -> str:
        info = self.controller.source_info
        return (
            f"Mode              {self.settings.mode.value}\n"
            f"Source            {info.name if info else '-'}\n"
            f"Resolution        {info.resolution if info else '-'}\n"
            f"Detector          {self.settings.backend.value} "
            f"({self.controller.model_state.value})\n"
            f"Device            {telemetry.device.upper()}\n"
            f"Uptime            {timedelta(seconds=int(snapshot.uptime_s))}\n"
            f"Frames processed  {self.controller.session_frames}\n"
            f"Peak people       {snapshot.peak_people}\n"
            f"Entries / exits   {snapshot.entered} / {snapshot.exited}\n"
            f"Events (session)  "
            f"{snapshot.infos + snapshot.warnings + snapshot.criticals}\n"
            f"  info            {snapshot.infos}\n"
            f"  warning         {snapshot.warnings}\n"
            f"  critical        {snapshot.criticals}\n"
            f"Dropped frames    {self.controller.dropped_frames}\n"
            f"Detect latency    {snapshot.detect_ms:.1f} ms\n"
            f"Track latency     {snapshot.track_ms:.2f} ms\n"
            f"Total latency     {snapshot.latency_ms:.1f} ms"
        )

    # ==================================================================
    # Sources page handlers
    # ==================================================================
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", str(Path.home()),
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)",
        )
        if not path:
            return
        self.page_sources.set_file(path, self._probe_video(path))

    @staticmethod
    def _probe_video(path: str) -> str:
        """Read a file's metadata for display without starting the pipeline."""
        try:
            probe = cv2.VideoCapture(path)
            if not probe.isOpened():
                probe.release()
                return (f"{Path(path).name} - OpenCV cannot open this file "
                        "(unsupported codec or container).")
            width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
            count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            probe.release()
            duration = timedelta(seconds=int(count / fps)) if fps > 0 else "-"
            return (f"{Path(path).name}   |   {width}x{height}   |   "
                    f"{fps:.2f} fps   |   {count} frames   |   {duration}")
        except Exception as exc:  # noqa: BLE001
            return f"Could not probe file: {exc}"

    def _on_scan_cameras(self) -> None:
        self._set_status("Scanning camera devices ...")
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()
        devices = WebcamSource.enumerate_devices()
        if devices:
            message = (f"Found {len(devices)} device(s): "
                       + ", ".join(str(d) for d in devices))
        else:
            message = ("No camera detected. The device may be disconnected, in "
                       "use, or blocked by OS privacy settings.")
        self.page_sources.set_cameras(devices, message)
        self._set_status("Camera scan complete")

    # ==================================================================
    # Events page handlers
    # ==================================================================
    def _reload_events(self) -> None:
        severity, event_type, search = self.page_events.filters()
        rows = self.database.query_events(
            limit=1000, severity=severity, event_type=event_type, search=search
        )
        self.page_events.load_rows(rows)
        self.page_events.set_counts(self.database.event_counts_by_severity())

    def _on_export_csv(self) -> None:
        if not self.database.available:
            self._error_box(
                "Export unavailable",
                "The local database is not available, so there is no stored "
                "history to export.",
            )
            return
        default = str(Path.home() / f"sentinel_events_{int(time.time())}.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Event History", default, "CSV files (*.csv)"
        )
        if not path:
            return
        severity, event_type, search = self.page_events.filters()
        rows = self.database.query_events(
            limit=100000, severity=severity, event_type=event_type, search=search
        )
        try:
            count = self.database.export_csv(Path(path), rows)
            self._set_status(f"Exported {count} events to {path}")
            QMessageBox.information(
                self, "Export complete", f"Exported {count} event(s) to:\n{path}"
            )
        except RuntimeError as exc:
            self._error_box("Export failed", str(exc))

    def _on_clear_events(self) -> None:
        answer = QMessageBox.question(
            self, "Clear event history",
            "Permanently delete all stored events from the local database?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer is not QMessageBox.StandardButton.Yes:
            return
        removed = self.database.clear_events()
        self.page_events.clear()
        self.live_panel.clear_events()
        self._recent_events.clear()
        self._reload_events()
        self._set_status(f"Cleared {removed} stored events")

    def _reload_sessions(self) -> None:
        self.page_analytics.load_sessions(self.database.recent_sessions())

    # ==================================================================
    # Zones page handlers
    # ==================================================================
    def _on_zones_changed(self) -> None:
        self.controller.configure_zones(self.zones, self.lines)
        self.live_panel.set_lines(self.lines)

    @pyqtSlot(str)
    def _edit_zone(self, zone_id: str) -> None:
        """Open the zone editor for a row clicked in the live status panel."""
        zone = next((z for z in self.zones if z.zone_id == zone_id), None)
        if zone is None:
            return
        dialog = ZoneEditorDialog(zone, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply()
            self.page_zones.refresh_table()
            self._on_zones_changed()
            SettingsManager.save_zones(self.settings.mode, self.zones, self.lines)
            self._set_status(f"Zone '{zone.name}' updated")
            self._log_system_event(
                f"Zone '{zone.name}' reconfigured "
                f"({zone.kind.value}, {zone.severity.value}, "
                f"max occupancy {zone.max_occupancy or 'disabled'})"
            )

    @pyqtSlot(str)
    def _edit_line(self, line_id: str) -> None:
        """Open the counting-line editor for a row in the live status panel."""
        line = next((l for l in self.lines if l.line_id == line_id), None)
        if line is None:
            return
        dialog = LineEditorDialog(line, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply()
            self._on_zones_changed()
            SettingsManager.save_zones(self.settings.mode, self.zones, self.lines)
            self._set_status(f"Line '{line.name}' updated")
            self._log_system_event(f"Counting line '{line.name}' reconfigured")

    def _on_zones_save(self) -> None:
        SettingsManager.save_zones(self.settings.mode, self.zones, self.lines)
        self._set_status("Zone layout saved")

    def _on_zones_reset(self) -> None:
        from sentinel.vision import default_lines, default_zones

        self.zones = default_zones(self.settings.mode)
        self.lines = default_lines(self.settings.mode)
        self.page_zones.set_zones(self.zones)
        self.live_panel.set_lines(self.lines)
        self.controller.configure_zones(self.zones, self.lines)
        self._set_status("Default zones restored")

    # ==================================================================
    # Models page handlers
    # ==================================================================
    @pyqtSlot(object)
    def _on_backend_changed(self, backend: BackendKind) -> None:
        self.settings.backend = backend
        self.page_settings.set_backend(backend)

    @pyqtSlot(object, object)
    def _on_load_model(self, backend: BackendKind, checkpoint: Optional[str]) -> None:
        from sentinel.detection import RFDETRDetector, RTDETRDetector

        ok, reason = (RTDETRDetector.dependency_status()
                      if backend is BackendKind.RT_DETR
                      else RFDETRDetector.dependency_status())
        if not ok:
            self._error_box(
                f"{backend.value} unavailable",
                f"{reason}\n\nSimulation mode remains fully functional without "
                "a detection backend.",
            )
            self._on_model_state(ModelState.UNAVAILABLE, reason)
            return

        self.settings.backend = backend
        self.settings_manager.save_settings(self.settings)
        if not self.controller.request_backend(backend, checkpoint):
            self._error_box(
                "Pipeline not running",
                "Start the pipeline first; the model is loaded inside the "
                "inference worker so the interface stays responsive.",
            )
            return
        self._set_status(f"Loading {backend.value} ...")

    def _on_unload_model(self) -> None:
        self.controller.request_backend(None)
        self._on_model_state(ModelState.UNLOADED, "No detection backend active")
        self._set_status("Model unloaded")

    # ==================================================================
    # Helpers
    # ==================================================================
    def _set_status(self, message: str) -> None:
        self.status_message.setText(message)
        LOGGER.debug("status: %s", message)

    def _error_box(self, title: str, message: str) -> None:
        LOGGER.error("%s: %s", title, message)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(title)
        box.setInformativeText(message)
        box.exec()

    def _log_system_event(
        self, message: str, severity: Severity = Severity.INFO
    ) -> None:
        """Record an application-level event in the same store as CV events."""
        info = self.controller.source_info
        event = SecurityEvent(
            timestamp=time.time(), event_type=EventType.SYSTEM_WARNING,
            severity=severity, description=message,
            mode=self.settings.mode.value,
            source=info.name if info else "system",
        )
        self.database.insert_events([event])
        self._recent_events.append(event)
        self.live_panel.add_events([event])

    # ==================================================================
    # Shutdown
    # ==================================================================
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Deterministic teardown: timers, threads, models, devices, database."""
        if self._shutting_down:
            event.accept()
            return
        self._shutting_down = True
        LOGGER.info("Shutting down ...")
        try:
            self._render_timer.stop()
            self._slow_timer.stop()
            self.controller.shutdown()
            self.settings_manager.save_settings(self.settings)
            self.settings_manager.save_geometry(self)
            SettingsManager.save_zones(self.settings.mode, self.zones, self.lines)
        except Exception as exc:  # noqa: BLE001 - shutdown must not hang
            LOGGER.exception("Error during shutdown: %s", exc)
        finally:
            self.database.close()
        LOGGER.info("Shutdown complete")
        event.accept()
