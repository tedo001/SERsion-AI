"""Video Sources page: choose between simulation, a local file and a webcam."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from sentinel.theme import Palette
from sentinel.ui.widgets import card, make_scroll

__all__ = ["SourcesPage"]


class SourcesPage(QWidget):
    """Source selection.  Emits requests; the controller does the work."""

    simulationRequested = pyqtSignal(int, int)   # people, vehicles
    fileRequested = pyqtSignal(str, bool)        # path, loop
    webcamRequested = pyqtSignal(int)            # device index
    browseRequested = pyqtSignal()
    scanRequested = pyqtSignal()
    populationChanged = pyqtSignal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Video Sources")
        title.setObjectName("H1")
        layout.addWidget(title)

        layout.addWidget(self._build_simulation_card())
        layout.addWidget(self._build_file_card())
        layout.addWidget(self._build_camera_card())

        active_frame, active_layout = card("Active Source")
        self.label_active = QLabel("None")
        self.label_active.setObjectName("Mono")
        self.label_active.setWordWrap(True)
        active_layout.addWidget(self.label_active)
        layout.addWidget(active_frame)

        layout.addStretch(1)
        root.addWidget(make_scroll(content))

    # ------------------------------------------------------------------
    def _build_simulation_card(self) -> QWidget:
        frame, layout = card("Simulation")
        description = QLabel(
            "Built-in synthetic CCTV scene with ground-truth annotations. "
            "Requires no model weights and no external files - the fastest way "
            "to validate zones, tracking and alerting."
        )
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {Palette.TEXT_DIM};")
        layout.addWidget(description)

        row = QHBoxLayout()
        row.setSpacing(10)
        people_caption = QLabel("People")
        people_caption.setObjectName("Caption")
        row.addWidget(people_caption)
        self.spin_people = QSpinBox()
        self.spin_people.setRange(1, 20)
        self.spin_people.setValue(10)
        row.addWidget(self.spin_people)

        vehicle_caption = QLabel("Vehicles")
        vehicle_caption.setObjectName("Caption")
        row.addWidget(vehicle_caption)
        self.spin_vehicles = QSpinBox()
        self.spin_vehicles.setRange(0, 6)
        self.spin_vehicles.setValue(2)
        row.addWidget(self.spin_vehicles)

        self.btn_use_sim = QPushButton("Use Simulation")
        self.btn_use_sim.setObjectName("Primary")
        self.btn_use_sim.clicked.connect(
            lambda: self.simulationRequested.emit(
                self.spin_people.value(), self.spin_vehicles.value()
            )
        )
        row.addWidget(self.btn_use_sim)
        row.addStretch(1)
        layout.addLayout(row)

        for spin in (self.spin_people, self.spin_vehicles):
            spin.valueChanged.connect(
                lambda _v: self.populationChanged.emit(
                    self.spin_people.value(), self.spin_vehicles.value()
                )
            )
        return frame

    def _build_file_card(self) -> QWidget:
        frame, layout = card("Local Video File")
        row = QHBoxLayout()
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText(
            "Select an MP4, AVI, MOV or MKV file ..."
        )
        self.edit_path.setReadOnly(True)
        row.addWidget(self.edit_path, 1)

        self.btn_browse = QPushButton("Browse ...")
        self.btn_browse.clicked.connect(self.browseRequested)
        row.addWidget(self.btn_browse)

        self.btn_use_file = QPushButton("Use File")
        self.btn_use_file.setObjectName("Primary")
        self.btn_use_file.setEnabled(False)
        self.btn_use_file.clicked.connect(self._emit_file)
        row.addWidget(self.btn_use_file)
        layout.addLayout(row)

        self.label_file_info = QLabel("No file selected")
        self.label_file_info.setObjectName("Mono")
        self.label_file_info.setStyleSheet(f"color: {Palette.TEXT_FAINT};")
        layout.addWidget(self.label_file_info)

        self.check_loop = QCheckBox("Loop playback when the file ends")
        self.check_loop.setChecked(True)
        layout.addWidget(self.check_loop)
        return frame

    def _build_camera_card(self) -> QWidget:
        frame, layout = card("Webcam / Capture Device")
        row = QHBoxLayout()
        caption = QLabel("Device")
        caption.setObjectName("Caption")
        row.addWidget(caption)

        self.combo_camera = QComboBox()
        self.combo_camera.addItems([f"Camera {i}" for i in range(4)])
        row.addWidget(self.combo_camera)

        self.btn_scan = QPushButton("Scan Devices")
        self.btn_scan.clicked.connect(self.scanRequested)
        row.addWidget(self.btn_scan)

        self.btn_use_camera = QPushButton("Use Camera")
        self.btn_use_camera.setObjectName("Primary")
        self.btn_use_camera.clicked.connect(
            lambda: self.webcamRequested.emit(self.selected_camera())
        )
        row.addWidget(self.btn_use_camera)
        row.addStretch(1)
        layout.addLayout(row)

        self.label_camera_info = QLabel(
            "Scanning probes camera indices 0-5 and may take a few seconds."
        )
        self.label_camera_info.setStyleSheet(
            f"color: {Palette.TEXT_FAINT}; font-size: 10px;"
        )
        layout.addWidget(self.label_camera_info)
        return frame

    # ------------------------------------------------------------------
    def _emit_file(self) -> None:
        path = self.edit_path.text().strip()
        if path:
            self.fileRequested.emit(path, self.check_loop.isChecked())

    def selected_camera(self) -> int:
        digits = "".join(ch for ch in self.combo_camera.currentText() if ch.isdigit())
        return int(digits) if digits else 0

    # -- view updates --------------------------------------------------------
    @property
    def file_path(self) -> str:
        return self.edit_path.text().strip()

    @property
    def loop_enabled(self) -> bool:
        return self.check_loop.isChecked()

    def population(self) -> Tuple[int, int]:
        return (self.spin_people.value(), self.spin_vehicles.value())

    def set_file(self, path: str, info: str) -> None:
        self.edit_path.setText(path)
        self.btn_use_file.setEnabled(bool(path))
        self.label_file_info.setText(info)

    def set_cameras(self, devices: Sequence[int], message: str) -> None:
        self.combo_camera.clear()
        if devices:
            self.combo_camera.addItems([f"Camera {i}" for i in devices])
        else:
            self.combo_camera.addItems([f"Camera {i}" for i in range(4)])
        self.label_camera_info.setText(message)

    def set_active_source(self, text: str) -> None:
        self.label_active.setText(text)
