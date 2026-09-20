"""Settings page: the single place AppSettings is edited.

The page never mutates the live settings object; it collects a fresh
:class:`AppSettings` and emits it, so the controller decides what to apply.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from sentinel.config import DB_PATH, LOG_PATH, ZONES_PATH
from sentinel.deps import DEP_SUPERVISION
from sentinel.models import (
    AppMode, AppSettings, BackendKind, DevicePref, NightMode, ThermalMode,
    ThermalPalette,
)
from sentinel.theme import Palette
from sentinel.ui.widgets import card, make_scroll

__all__ = ["SettingsPage"]


class SettingsPage(QWidget):
    """Grouped configuration form."""

    settingsApplied = pyqtSignal(object)   # AppSettings
    resetRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: Dict[int, int] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Settings")
        title.setObjectName("H1")
        layout.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        right = QVBoxLayout()
        right.setSpacing(14)

        left.addWidget(self._build_inference_card())
        left.addWidget(self._build_tracking_card())
        right.addWidget(self._build_performance_card())
        right.addWidget(self._build_overlay_card())
        right.addWidget(self._build_alert_card())
        left.addWidget(self._build_night_card())
        left.addWidget(self._build_thermal_card())

        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        layout.addLayout(columns)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_apply = QPushButton("Apply Settings")
        self.btn_apply.setObjectName("Primary")
        buttons.addWidget(self.btn_reset)
        buttons.addWidget(self.btn_apply)
        layout.addLayout(buttons)

        note = QLabel(
            f"Settings are stored per user. Database: {DB_PATH}    "
            f"Zones: {ZONES_PATH}    Log: {LOG_PATH}"
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
        layout.addWidget(note)
        layout.addStretch(1)

        root.addWidget(make_scroll(content))

        self.btn_apply.clicked.connect(
            lambda: self.settingsApplied.emit(self.collect())
        )
        self.btn_reset.clicked.connect(self.resetRequested)
        self.slider_conf.valueChanged.connect(
            lambda v: self.label_conf.setText(f"{v / 100:.2f}")
        )

    # ------------------------------------------------------------------
    def _add_row(self, grid: QGridLayout, label: str, widget: QWidget) -> None:
        """Append a labelled row, tracking each grid's own row counter."""
        row = self._rows.get(id(grid), 0)
        caption = QLabel(label)
        caption.setObjectName("Caption")
        grid.addWidget(caption, row, 0)
        grid.addWidget(widget, row, 1)
        grid.setColumnStretch(1, 1)
        self._rows[id(grid)] = row + 1

    def _build_inference_card(self) -> QWidget:
        frame, layout = card("Inference")
        grid = QGridLayout()
        grid.setSpacing(9)

        self.combo_backend = QComboBox()
        self.combo_backend.addItems([b.value for b in BackendKind])
        self._add_row(grid, "Detection model", self.combo_backend)

        conf_widget = QWidget()
        conf_layout = QHBoxLayout(conf_widget)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(10, 100)
        self.label_conf = QLabel("0.35")
        self.label_conf.setObjectName("Mono")
        self.label_conf.setFixedWidth(38)
        conf_layout.addWidget(self.slider_conf, 1)
        conf_layout.addWidget(self.label_conf)
        self._add_row(grid, "Confidence", conf_widget)

        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["320", "480", "640", "800"])
        self._add_row(grid, "Inference resolution", self.combo_resolution)

        self.combo_device = QComboBox()
        self.combo_device.addItems([d.value for d in DevicePref])
        self._add_row(grid, "Device", self.combo_device)

        self.check_half = QCheckBox("Use half precision when CUDA supports it")
        self._add_row(grid, "Precision", self.check_half)

        self.check_detect_sim = QCheckBox(
            "Run the model on simulated frames instead of synthetic ground truth"
        )
        self._add_row(grid, "Simulation", self.check_detect_sim)

        self.check_person_only = QCheckBox("Person class only")
        self._add_row(grid, "Class filter", self.check_person_only)
        layout.addLayout(grid)
        return frame

    def _build_tracking_card(self) -> QWidget:
        frame, layout = card("Tracking")
        grid = QGridLayout()
        grid.setSpacing(9)

        self.check_tracking = QCheckBox("Enable Kalman tracking")
        self._add_row(grid, "Tracking", self.check_tracking)
        self.spin_max_age = QSpinBox()
        self.spin_max_age.setRange(1, 200)
        self._add_row(grid, "Max coast frames", self.spin_max_age)
        self.spin_min_hits = QSpinBox()
        self.spin_min_hits.setRange(1, 30)
        self._add_row(grid, "Confirm after hits", self.spin_min_hits)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.05, 0.9)
        self.spin_iou.setSingleStep(0.05)
        self._add_row(grid, "Association IoU", self.spin_iou)
        layout.addLayout(grid)
        return frame

    def _build_performance_card(self) -> QWidget:
        frame, layout = card("Performance")
        grid = QGridLayout()
        grid.setSpacing(9)

        self.spin_target_fps = QSpinBox()
        self.spin_target_fps.setRange(1, 120)
        self._add_row(grid, "Capture FPS limit", self.spin_target_fps)
        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(0, 10)
        self._add_row(grid, "Frame skip", self.spin_frame_skip)
        self.spin_queue = QSpinBox()
        self.spin_queue.setRange(1, 10)
        self._add_row(grid, "Frame queue size", self.spin_queue)
        self.spin_gui_fps = QSpinBox()
        self.spin_gui_fps.setRange(5, 60)
        self._add_row(grid, "GUI refresh rate", self.spin_gui_fps)
        layout.addLayout(grid)
        return frame

    def _build_night_card(self) -> QWidget:
        """Low-light enhancement controls."""
        frame, layout = card("Night Vision / Low Light")
        grid = QGridLayout()
        grid.setSpacing(9)

        self.combo_night = QComboBox()
        self.combo_night.addItems([m.value for m in NightMode])
        self.combo_night.setToolTip(
            "Auto engages enhancement only when the scene is actually dark."
        )
        self._add_row(grid, "Mode", self.combo_night)

        threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(threshold_widget)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_night_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_night_threshold.setRange(10, 160)
        self.label_night_threshold = QLabel("70")
        self.label_night_threshold.setObjectName("Mono")
        self.label_night_threshold.setFixedWidth(38)
        threshold_layout.addWidget(self.slider_night_threshold, 1)
        threshold_layout.addWidget(self.label_night_threshold)
        self._add_row(grid, "Night threshold", threshold_widget)

        strength_widget = QWidget()
        strength_layout = QHBoxLayout(strength_widget)
        strength_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_night_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_night_strength.setRange(0, 100)
        self.label_night_strength = QLabel("0.60")
        self.label_night_strength.setObjectName("Mono")
        self.label_night_strength.setFixedWidth(38)
        strength_layout.addWidget(self.slider_night_strength, 1)
        strength_layout.addWidget(self.label_night_strength)
        self._add_row(grid, "Strength", strength_widget)

        self.check_night_denoise = QCheckBox("Denoise (gain amplifies sensor noise)")
        self._add_row(grid, "Denoise", self.check_night_denoise)
        self.check_night_detect = QCheckBox("Run detection on the enhanced frame")
        self._add_row(grid, "Detection", self.check_night_detect)
        self.check_night_tint = QCheckBox("Green display tint (cosmetic)")
        self._add_row(grid, "Tint", self.check_night_tint)
        layout.addLayout(grid)

        note = QLabel(
            "Contrast enhancement (CLAHE + gamma), not a thermal or infrared "
            "sensor. It amplifies light the camera did capture; in total "
            "darkness there is nothing to amplify. It adds no detection "
            "capability of its own - a model backend is still required."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
        layout.addWidget(note)

        self.slider_night_threshold.valueChanged.connect(
            lambda v: self.label_night_threshold.setText(str(v))
        )
        self.slider_night_strength.valueChanged.connect(
            lambda v: self.label_night_strength.setText(f"{v / 100:.2f}")
        )
        return frame

    def _build_thermal_card(self) -> QWidget:
        """Thermal palette + radiometric calibration."""
        frame, layout = card("Thermal Imaging")
        grid = QGridLayout()
        grid.setSpacing(9)

        self.combo_thermal = QComboBox()
        self.combo_thermal.addItems([m.value for m in ThermalMode])
        self.combo_thermal.currentTextChanged.connect(self._on_thermal_mode)
        self._add_row(grid, "Mode", self.combo_thermal)

        self.combo_palette = QComboBox()
        self.combo_palette.addItems([p.value for p in ThermalPalette])
        self._add_row(grid, "Palette", self.combo_palette)

        self.spin_hotspot = QSpinBox()
        self.spin_hotspot.setRange(1, 99)
        self.spin_hotspot.setSuffix(" %")
        self.spin_hotspot.setToolTip(
            "Intensity above which a region counts as a hotspot."
        )
        self._add_row(grid, "Hotspot threshold", self.spin_hotspot)

        self.check_thermal_overlay = QCheckBox("Colour scale and hotspot boxes")
        self._add_row(grid, "Overlay", self.check_thermal_overlay)

        self.spin_thermal_min = QDoubleSpinBox()
        self.spin_thermal_min.setRange(-40.0, 1000.0)
        self.spin_thermal_min.setSuffix(" \u00b0C")
        self.spin_thermal_max = QDoubleSpinBox()
        self.spin_thermal_max.setRange(-40.0, 2000.0)
        self.spin_thermal_max.setSuffix(" \u00b0C")
        self._add_row(grid, "Calibrated min", self.spin_thermal_min)
        self._add_row(grid, "Calibrated max", self.spin_thermal_max)
        layout.addLayout(grid)

        self.label_thermal_warning = QLabel()
        self.label_thermal_warning.setWordWrap(True)
        self.label_thermal_warning.setStyleSheet(
            f"color: {Palette.WARN}; font-size: 10px;"
        )
        layout.addWidget(self.label_thermal_warning)
        self._on_thermal_mode(self.combo_thermal.currentText())
        return frame

    def _on_thermal_mode(self, text: str) -> None:
        """Calibration only applies to a genuine thermal source."""
        radiometric = text == ThermalMode.RADIOMETRIC.value
        for widget in (self.spin_thermal_min, self.spin_thermal_max):
            widget.setEnabled(radiometric)
        if radiometric:
            self.label_thermal_warning.setText(
                "RADIOMETRIC: only valid if the active source really is a "
                "calibrated thermal camera. Enter the temperature span from "
                "that camera's datasheet or its own UI - the app maps pixel "
                "values onto it linearly. Pointed at an ordinary webcam this "
                "produces meaningless numbers."
            )
        elif text == ThermalMode.OFF.value:
            self.label_thermal_warning.setText(
                "Thermal display disabled."
            )
        else:
            self.label_thermal_warning.setText(
                "FALSE COLOUR: maps image brightness through a thermal palette. "
                "It is a display filter - no temperature is measured, and "
                "readings are reported as intensity percent."
            )

    def _build_overlay_card(self) -> QWidget:
        frame, layout = card("Overlays")
        self.check_boxes = QCheckBox("Bounding boxes")
        self.check_ids = QCheckBox("Track IDs")
        self.check_zones = QCheckBox("Zones and lines")
        self.check_confidence = QCheckBox("Confidence values")
        self.check_fps = QCheckBox("FPS overlay")
        self.check_trails = QCheckBox("Motion trails")
        self.check_supervision = QCheckBox(
            "Use supervision annotators when available"
        )
        for widget in (self.check_boxes, self.check_ids, self.check_zones,
                       self.check_confidence, self.check_fps, self.check_trails,
                       self.check_supervision):
            layout.addWidget(widget)
        if not DEP_SUPERVISION.available:
            self.check_supervision.setEnabled(False)
            self.check_supervision.setText(
                "Use supervision annotators (package not installed)"
            )
        return frame

    def _build_alert_card(self) -> QWidget:
        frame, layout = card("Scenario & Alert Thresholds")
        grid = QGridLayout()
        grid.setSpacing(9)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([m.value for m in AppMode])
        self._add_row(grid, "Deployment mode", self.combo_mode)
        self.spin_crowding = QSpinBox()
        self.spin_crowding.setRange(1, 200)
        self._add_row(grid, "Crowding threshold", self.spin_crowding)
        self.spin_high_occupancy = QSpinBox()
        self.spin_high_occupancy.setRange(1, 500)
        self._add_row(grid, "High occupancy threshold", self.spin_high_occupancy)
        self.spin_proximity = QSpinBox()
        self.spin_proximity.setRange(5, 400)
        self.spin_proximity.setSuffix(" px")
        self._add_row(grid, "Machine proximity radius", self.spin_proximity)
        layout.addLayout(grid)
        return frame

    # ------------------------------------------------------------------
    def apply_settings(self, settings: AppSettings) -> None:
        """Push a settings object into the widgets without re-emitting."""
        blocked = (self.combo_backend, self.slider_conf, self.combo_resolution,
                   self.combo_device, self.combo_mode)
        for widget in blocked:
            widget.blockSignals(True)

        self.combo_backend.setCurrentText(settings.backend.value)
        self.slider_conf.setValue(int(settings.confidence * 100))
        self.label_conf.setText(f"{settings.confidence:.2f}")
        self.combo_resolution.setCurrentText(str(settings.inference_size))
        self.combo_device.setCurrentText(settings.device.value)
        self.combo_mode.setCurrentText(settings.mode.value)

        for widget in blocked:
            widget.blockSignals(False)

        self.check_half.setChecked(settings.half_precision)
        self.check_detect_sim.setChecked(settings.detect_on_simulation)
        self.check_person_only.setChecked(settings.person_classes_only)
        self.check_tracking.setChecked(settings.tracking_enabled)
        self.spin_max_age.setValue(settings.max_age)
        self.spin_min_hits.setValue(settings.min_hits)
        self.spin_iou.setValue(settings.iou_threshold)
        self.spin_target_fps.setValue(settings.target_fps)
        self.spin_frame_skip.setValue(settings.frame_skip)
        self.spin_queue.setValue(settings.queue_size)
        self.spin_gui_fps.setValue(settings.gui_fps)
        self.check_boxes.setChecked(settings.show_boxes)
        self.check_ids.setChecked(settings.show_track_ids)
        self.check_zones.setChecked(settings.show_zones)
        self.check_confidence.setChecked(settings.show_confidence)
        self.check_fps.setChecked(settings.show_fps_overlay)
        self.check_trails.setChecked(settings.show_trails)
        self.check_supervision.setChecked(
            settings.use_supervision_annotators and DEP_SUPERVISION.available
        )
        self.combo_night.blockSignals(True)
        self.combo_night.setCurrentText(settings.night_mode.value)
        self.combo_night.blockSignals(False)
        self.slider_night_threshold.setValue(settings.night_threshold)
        self.label_night_threshold.setText(str(settings.night_threshold))
        self.slider_night_strength.setValue(int(settings.night_strength * 100))
        self.label_night_strength.setText(f"{settings.night_strength:.2f}")
        self.check_night_denoise.setChecked(settings.night_denoise)
        self.check_night_detect.setChecked(settings.night_enhance_detection)
        self.check_night_tint.setChecked(settings.night_tint)

        self.combo_thermal.setCurrentText(settings.thermal_mode.value)
        self.combo_palette.setCurrentText(settings.thermal_palette.value)
        self.spin_hotspot.setValue(settings.thermal_hotspot_threshold)
        self.check_thermal_overlay.setChecked(settings.thermal_overlay)
        self.spin_thermal_min.setValue(settings.thermal_min_c)
        self.spin_thermal_max.setValue(settings.thermal_max_c)

        self.spin_crowding.setValue(settings.crowding_threshold)
        self.spin_high_occupancy.setValue(settings.high_occupancy_threshold)
        self.spin_proximity.setValue(settings.proximity_px)

    def collect(self) -> AppSettings:
        """Read every widget back into a fresh AppSettings instance."""
        return AppSettings(
            mode=AppMode(self.combo_mode.currentText()),
            backend=BackendKind(self.combo_backend.currentText()),
            device=DevicePref(self.combo_device.currentText()),
            confidence=self.slider_conf.value() / 100.0,
            inference_size=int(self.combo_resolution.currentText()),
            half_precision=self.check_half.isChecked(),
            tracking_enabled=self.check_tracking.isChecked(),
            max_age=self.spin_max_age.value(),
            min_hits=self.spin_min_hits.value(),
            iou_threshold=float(self.spin_iou.value()),
            target_fps=self.spin_target_fps.value(),
            frame_skip=self.spin_frame_skip.value(),
            queue_size=self.spin_queue.value(),
            gui_fps=self.spin_gui_fps.value(),
            show_boxes=self.check_boxes.isChecked(),
            show_track_ids=self.check_ids.isChecked(),
            show_zones=self.check_zones.isChecked(),
            show_confidence=self.check_confidence.isChecked(),
            show_fps_overlay=self.check_fps.isChecked(),
            show_trails=self.check_trails.isChecked(),
            use_supervision_annotators=self.check_supervision.isChecked(),
            crowding_threshold=self.spin_crowding.value(),
            high_occupancy_threshold=self.spin_high_occupancy.value(),
            proximity_px=self.spin_proximity.value(),
            person_classes_only=self.check_person_only.isChecked(),
            detect_on_simulation=self.check_detect_sim.isChecked(),
            night_mode=NightMode(self.combo_night.currentText()),
            night_threshold=self.slider_night_threshold.value(),
            night_strength=self.slider_night_strength.value() / 100.0,
            night_denoise=self.check_night_denoise.isChecked(),
            night_tint=self.check_night_tint.isChecked(),
            night_enhance_detection=self.check_night_detect.isChecked(),
            thermal_mode=ThermalMode(self.combo_thermal.currentText()),
            thermal_palette=ThermalPalette(self.combo_palette.currentText()),
            thermal_hotspot_threshold=self.spin_hotspot.value(),
            thermal_overlay=self.check_thermal_overlay.isChecked(),
            thermal_min_c=float(self.spin_thermal_min.value()),
            thermal_max_c=float(self.spin_thermal_max.value()),
        )

    def set_confidence(self, confidence: float) -> None:
        """Mirror a confidence change made from the Live Monitor page."""
        self.slider_conf.blockSignals(True)
        self.slider_conf.setValue(int(confidence * 100))
        self.slider_conf.blockSignals(False)
        self.label_conf.setText(f"{confidence:.2f}")

    def set_night_mode(self, mode: NightMode) -> None:
        """Mirror a night-vision change made from the Live Monitor toolbar."""
        self.combo_night.blockSignals(True)
        self.combo_night.setCurrentText(mode.value)
        self.combo_night.blockSignals(False)

    def set_mode(self, mode: AppMode) -> None:
        self.combo_mode.blockSignals(True)
        self.combo_mode.setCurrentText(mode.value)
        self.combo_mode.blockSignals(False)

    def set_backend(self, backend: BackendKind) -> None:
        self.combo_backend.blockSignals(True)
        self.combo_backend.setCurrentText(backend.value)
        self.combo_backend.blockSignals(False)
