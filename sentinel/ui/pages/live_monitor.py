"""Live Monitor page: the video surface and its transport controls."""

from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from sentinel.models import AppMode, AppSettings, NightMode, SourceKind
from sentinel.theme import Palette
from sentinel.ui.video_view import VideoView
from sentinel.ui.widgets import divider

__all__ = ["LiveMonitorPage"]


class LiveMonitorPage(QWidget):
    """Owns the live video surface; emits intent, never drives the pipeline."""

    sourceKindChosen = pyqtSignal(object)    # SourceKind
    modeChosen = pyqtSignal(object)          # AppMode
    snapshotRequested = pyqtSignal()
    playPauseRequested = pyqtSignal()
    restartRequested = pyqtSignal()
    seekRequested = pyqtSignal(int)
    confidenceChanged = pyqtSignal(float)
    nightModeChanged = pyqtSignal(object)    # NightMode

    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # -- header ----------------------------------------------------------
        header = QHBoxLayout()
        title = QLabel("Live Monitor")
        title.setObjectName("H1")
        header.addWidget(title)
        header.addStretch(1)

        self.combo_source = QComboBox()
        self.combo_source.addItems([k.value for k in SourceKind])
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([m.value for m in AppMode])
        for label, widget in (("Source", self.combo_source), ("Mode", self.combo_mode)):
            caption = QLabel(label)
            caption.setObjectName("Caption")
            header.addWidget(caption)
            header.addWidget(widget)

        # Night-vision toggle, right where the operator is already looking.
        self.btn_night = QPushButton("NIGHT OFF")
        self.btn_night.setCheckable(True)
        self.btn_night.setMinimumWidth(116)
        self.btn_night.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_night.setShortcut("N")
        self.btn_night.setToolTip(
            "Toggle low-light enhancement (shortcut: N).\n"
            "Right-click for Off / Auto / On."
        )
        self.btn_night.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        header.addWidget(self.btn_night)

        self.btn_snapshot = QPushButton("Snapshot")
        self.btn_snapshot.setObjectName("Ghost")
        header.addWidget(self.btn_snapshot)
        layout.addLayout(header)

        self._night_mode = settings.night_mode
        self._night_live = False
        self._restyle_night()

        # -- video surface ----------------------------------------------------
        self.video_view = VideoView()
        self.video_view.show_placeholder(
            "NO SIGNAL", "Press Start Pipeline to begin monitoring"
        )
        video_frame = QFrame()
        video_frame.setObjectName("Panel")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(8, 8, 8, 8)
        video_layout.addWidget(self.video_view)
        layout.addWidget(video_frame, 1)

        # -- transport --------------------------------------------------------
        controls = QFrame()
        controls.setObjectName("Card")
        row = QHBoxLayout(controls)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)

        self.btn_play = QPushButton("Pause")
        self.btn_restart = QPushButton("Restart")
        self.btn_play.setEnabled(False)
        self.btn_restart.setEnabled(False)
        row.addWidget(self.btn_play)
        row.addWidget(self.btn_restart)

        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setEnabled(False)
        self.slider_seek.setRange(0, 0)
        row.addWidget(self.slider_seek, 1)
        self.label_position = QLabel("-- / --")
        self.label_position.setObjectName("Mono")
        row.addWidget(self.label_position)

        row.addWidget(divider(vertical=True))
        zoom_caption = QLabel("ZOOM")
        zoom_caption.setObjectName("Caption")
        row.addWidget(zoom_caption)
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(100, 500)
        self.slider_zoom.setValue(100)
        self.slider_zoom.setFixedWidth(110)
        row.addWidget(self.slider_zoom)

        row.addWidget(divider(vertical=True))
        conf_caption = QLabel("CONF")
        conf_caption.setObjectName("Caption")
        row.addWidget(conf_caption)
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(10, 100)
        self.slider_conf.setValue(int(settings.confidence * 100))
        self.slider_conf.setFixedWidth(110)
        row.addWidget(self.slider_conf)
        self.label_conf = QLabel(f"{settings.confidence:.2f}")
        self.label_conf.setObjectName("Mono")
        row.addWidget(self.label_conf)

        layout.addWidget(controls)
        self._connect()

    def _connect(self) -> None:
        self.btn_snapshot.clicked.connect(self.snapshotRequested)
        self.btn_play.clicked.connect(self.playPauseRequested)
        self.btn_restart.clicked.connect(self.restartRequested)
        self.slider_seek.sliderReleased.connect(
            lambda: self.seekRequested.emit(self.slider_seek.value())
        )
        self.slider_zoom.valueChanged.connect(
            lambda v: self.video_view.set_zoom(v / 100.0)
        )
        self.slider_conf.valueChanged.connect(self._on_confidence)
        self.combo_source.currentTextChanged.connect(self._on_source_text)
        self.combo_mode.currentTextChanged.connect(self._on_mode_text)
        self.btn_night.clicked.connect(self._on_night_clicked)
        self.btn_night.customContextMenuRequested.connect(self._night_menu)

    # -- internal signal adapters -------------------------------------------
    def _on_confidence(self, value: int) -> None:
        self.label_conf.setText(f"{value / 100:.2f}")
        self.confidenceChanged.emit(value / 100.0)

    def _on_source_text(self, text: str) -> None:
        try:
            self.sourceKindChosen.emit(SourceKind(text))
        except ValueError:
            pass

    def _on_mode_text(self, text: str) -> None:
        try:
            self.modeChosen.emit(AppMode(text))
        except ValueError:
            pass

    # -- night vision --------------------------------------------------------
    def _on_night_clicked(self) -> None:
        """Plain on/off toggle.

        AUTO counts as "on" for the first click, so pressing the button while
        AUTO is armed turns enhancement off rather than forcing it on - which is
        what someone reaching for an off switch expects.
        """
        if self._night_mode is NightMode.OFF:
            self.set_night_mode(NightMode.ON)
        else:
            self.set_night_mode(NightMode.OFF)
        self.nightModeChanged.emit(self._night_mode)

    def _night_menu(self, position) -> None:
        """Right-click menu - the only place AUTO is reachable from the toolbar."""
        menu = QMenu(self)
        for mode in NightMode:
            action = QAction(f"Night vision: {mode.value}", menu)
            action.setCheckable(True)
            action.setChecked(mode is self._night_mode)
            action.triggered.connect(
                lambda _checked=False, m=mode: self._choose_night(m)
            )
            menu.addAction(action)
        menu.exec(self.btn_night.mapToGlobal(position))

    def _choose_night(self, mode: NightMode) -> None:
        self.set_night_mode(mode)
        self.nightModeChanged.emit(mode)

    def set_night_mode(self, mode: NightMode) -> None:
        """Reflect the stored mode without emitting (used to sync from Settings)."""
        self._night_mode = mode
        self._restyle_night()

    def set_night_active(self, active: bool) -> None:
        """Show whether enhancement is actually running right now.

        In AUTO the mode alone does not say whether the scene is dark enough, so
        the button reports the live state rather than just the setting.
        """
        if active == self._night_live:
            return
        self._night_live = active
        self._restyle_night()

    def _restyle_night(self) -> None:
        mode = self._night_mode
        dot = "  \u25cf"
        if mode is NightMode.OFF:
            text, colour, checked = "NIGHT OFF", Palette.TEXT_DIM, False
        elif mode is NightMode.ON:
            checked, colour = True, Palette.CYAN
            text = "NIGHT ON" + (dot if self._night_live else "")
        else:
            checked = True
            colour = Palette.CYAN if self._night_live else Palette.TEXT_DIM
            text = "NIGHT AUTO" + (dot if self._night_live else "")

        self.btn_night.blockSignals(True)
        self.btn_night.setChecked(checked)
        self.btn_night.blockSignals(False)
        self.btn_night.setText(text)

        if checked and self._night_live:
            self.btn_night.setStyleSheet(
                f"QPushButton {{ background-color: {colour}; color: #04101C;"
                f" border: 1px solid {colour}; border-radius: 6px;"
                f" padding: 7px 12px; font-weight: 700; font-size: 11px; }}"
            )
        else:
            self.btn_night.setStyleSheet(
                f"QPushButton {{ background-color: transparent;"
                f" color: {colour}; border: 1px solid {Palette.BORDER_STRONG};"
                f" border-radius: 6px; padding: 7px 12px; font-weight: 700;"
                f" font-size: 11px; }}"
                f"QPushButton:hover {{ border-color: {Palette.CYAN};"
                f" color: {Palette.TEXT}; }}"
            )

    # -- view updates (called by MainWindow) ---------------------------------
    def show_frame(self, frame: "np.ndarray") -> None:
        self.video_view.set_frame(frame)

    def show_placeholder(self, title: str, subtitle: str = "",
                         accent: str = Palette.TEXT_FAINT) -> None:
        self.video_view.show_placeholder(title, subtitle, accent)

    def set_badge(self, text: Optional[str], color: str = Palette.WARN) -> None:
        self.video_view.set_badge(text, color)

    def set_transport_enabled(self, enabled: bool) -> None:
        self.btn_play.setEnabled(enabled)
        self.btn_restart.setEnabled(enabled)
        if not enabled:
            self.slider_seek.setEnabled(False)

    def set_play_text(self, text: str) -> None:
        self.btn_play.setText(text)

    def set_seekable(self, seekable: bool, frame_count: int = 0) -> None:
        self.slider_seek.setEnabled(seekable)
        if seekable:
            self.slider_seek.setRange(0, max(0, frame_count - 1))

    def set_position(self, position: int, total: int = 0) -> None:
        if self.slider_seek.isEnabled() and not self.slider_seek.isSliderDown():
            self.slider_seek.blockSignals(True)
            self.slider_seek.setValue(position)
            self.slider_seek.blockSignals(False)
        self.label_position.setText(
            f"{position} / {total}" if total else f"frame {position}"
        )

    def set_confidence(self, confidence: float) -> None:
        self.slider_conf.blockSignals(True)
        self.slider_conf.setValue(int(confidence * 100))
        self.slider_conf.blockSignals(False)
        self.label_conf.setText(f"{confidence:.2f}")

    def set_mode(self, mode: AppMode) -> None:
        self.combo_mode.blockSignals(True)
        self.combo_mode.setCurrentText(mode.value)
        self.combo_mode.blockSignals(False)

    def set_source_kind(self, kind: SourceKind) -> None:
        self.combo_source.blockSignals(True)
        self.combo_source.setCurrentText(kind.value)
        self.combo_source.blockSignals(False)
