"""Zones page: zone table plus an interactive polygon-drawing canvas."""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from sentinel.models import ZoneConfig, ZoneKind
from sentinel.theme import Palette
from sentinel.ui.video_view import VideoView
from sentinel.ui.widgets import card, make_scroll, qcolor

__all__ = ["ZonesPage"]


class ZonesPage(QWidget):
    """Edits a zone list in place and announces changes.

    The page owns the drawing interaction; it does not know about the pipeline.
    """

    zonesChanged = pyqtSignal()
    saveRequested = pyqtSignal()
    resetRequested = pyqtSignal()
    statusMessage = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zones: List[ZoneConfig] = []
        self._pending: List[Tuple[float, float]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Zone Configuration")
        title.setObjectName("H1")
        header.addWidget(title)
        header.addStretch(1)
        self.btn_reset = QPushButton("Restore Defaults")
        self.btn_save = QPushButton("Save Layout")
        self.btn_save.setObjectName("Primary")
        header.addWidget(self.btn_reset)
        header.addWidget(self.btn_save)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(make_scroll(self._build_editor()))
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 760])
        layout.addWidget(splitter, 1)

        self.btn_save.clicked.connect(self.saveRequested)
        self.btn_reset.clicked.connect(self.resetRequested)

    # ------------------------------------------------------------------
    def _build_editor(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 7, 0)
        layout.setSpacing(10)

        table_frame, table_layout = card("Zones")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["ENABLED", "NAME", "TYPE", "SEVERITY", "MAX OCC.", "POINTS"]
        )
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(True)
        self.table.setMinimumHeight(240)
        self.table.itemChanged.connect(self._on_item_changed)
        table_layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setObjectName("Danger")
        self.btn_delete.clicked.connect(self._on_delete)
        buttons.addWidget(self.btn_delete)
        buttons.addStretch(1)
        table_layout.addLayout(buttons)
        layout.addWidget(table_frame)

        draw_frame, draw_layout = card("Draw New Zone")
        description = QLabel(
            "Enable drawing, then click on the preview to place polygon "
            "vertices. Three or more points are required."
        )
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")
        draw_layout.addWidget(description)

        form = QGridLayout()
        form.setSpacing(8)
        name_caption = QLabel("Name")
        name_caption.setObjectName("Caption")
        form.addWidget(name_caption, 0, 0)
        self.edit_name = QLineEdit("New Zone")
        form.addWidget(self.edit_name, 0, 1)

        type_caption = QLabel("Type")
        type_caption.setObjectName("Caption")
        form.addWidget(type_caption, 1, 0)
        self.combo_kind = QComboBox()
        self.combo_kind.addItems([k.value for k in ZoneKind])
        form.addWidget(self.combo_kind, 1, 1)

        occ_caption = QLabel("Max occupancy")
        occ_caption.setObjectName("Caption")
        form.addWidget(occ_caption, 2, 0)
        self.spin_occupancy = QSpinBox()
        self.spin_occupancy.setRange(0, 200)
        self.spin_occupancy.setSpecialValueText("disabled")
        form.addWidget(self.spin_occupancy, 2, 1)
        form.setColumnStretch(1, 1)
        draw_layout.addLayout(form)

        buttons = QHBoxLayout()
        self.btn_draw = QPushButton("Start Drawing")
        self.btn_draw.setCheckable(True)
        self.btn_undo = QPushButton("Undo Point")
        self.btn_commit = QPushButton("Create Zone")
        self.btn_commit.setObjectName("Primary")
        self.btn_commit.setEnabled(False)
        for button in (self.btn_draw, self.btn_undo, self.btn_commit):
            buttons.addWidget(button)
        draw_layout.addLayout(buttons)
        layout.addWidget(draw_frame)
        layout.addStretch(1)

        self.btn_draw.toggled.connect(self._on_draw_toggled)
        self.btn_undo.clicked.connect(self._on_undo)
        self.btn_commit.clicked.connect(self._on_commit)
        return panel

    def _build_preview(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(7, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("Panel")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        caption = QLabel("PREVIEW / DRAWING CANVAS")
        caption.setObjectName("Caption")
        frame_layout.addWidget(caption)

        self.view = VideoView()
        self.view.show_placeholder(
            "PREVIEW", "Start the pipeline to draw zones over the live feed"
        )
        self.view.pointClicked.connect(self._on_point)
        frame_layout.addWidget(self.view, 1)
        layout.addWidget(frame, 1)
        return panel

    # ------------------------------------------------------------------
    # Zone list
    # ------------------------------------------------------------------
    def set_zones(self, zones: List[ZoneConfig]) -> None:
        """Adopt the live zone list (by reference - edits apply in place)."""
        self._zones = zones
        self.refresh_table()

    def refresh_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._zones))
        for row, zone in enumerate(self._zones):
            enabled = QTableWidgetItem()
            enabled.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            enabled.setCheckState(
                Qt.CheckState.Checked if zone.enabled else Qt.CheckState.Unchecked
            )
            self.table.setItem(row, 0, enabled)

            name_item = QTableWidgetItem(zone.name)
            name_item.setForeground(QBrush(qcolor(zone.color)))
            self.table.setItem(row, 1, name_item)

            kind_item = QTableWidgetItem(zone.kind.value)
            kind_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 2, kind_item)

            severity_item = QTableWidgetItem(zone.severity.value)
            severity_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            severity_item.setForeground(QBrush(qcolor(zone.severity.color)))
            self.table.setItem(row, 3, severity_item)

            occupancy_item = QTableWidgetItem(
                str(zone.max_occupancy) if zone.max_occupancy else "-"
            )
            self.table.setItem(row, 4, occupancy_item)

            points_item = QTableWidgetItem(str(len(zone.polygon)))
            points_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 5, points_item)
        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not (0 <= row < len(self._zones)):
            return
        zone = self._zones[row]
        column = item.column()
        if column == 0:
            zone.enabled = item.checkState() is Qt.CheckState.Checked
        elif column == 1:
            zone.name = item.text().strip() or zone.name
        elif column == 4:
            text = item.text().strip()
            try:
                zone.max_occupancy = max(0, int(text)) if text not in ("", "-") else 0
            except ValueError:
                self.table.blockSignals(True)
                item.setText(str(zone.max_occupancy) if zone.max_occupancy else "-")
                self.table.blockSignals(False)
                return
        self.zonesChanged.emit()

    def _on_delete(self) -> None:
        row = self.table.currentRow()
        if not (0 <= row < len(self._zones)):
            self.statusMessage.emit("Select a zone row to delete")
            return
        zone = self._zones.pop(row)
        self.refresh_table()
        self.zonesChanged.emit()
        self.statusMessage.emit(f"Zone '{zone.name}' deleted")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _on_draw_toggled(self, enabled: bool) -> None:
        self._pending = []
        self.view.set_edit_mode(enabled)
        self.view.set_edit_points([])
        self.btn_draw.setText("Stop Drawing" if enabled else "Start Drawing")
        self.btn_commit.setEnabled(False)
        if enabled:
            self.statusMessage.emit("Click on the preview to place zone vertices")

    def _on_point(self, x: float, y: float) -> None:
        self._pending.append((float(x), float(y)))
        self.view.set_edit_points(self._pending)
        self.btn_commit.setEnabled(len(self._pending) >= 3)
        self.statusMessage.emit(f"{len(self._pending)} point(s) placed")

    def _on_undo(self) -> None:
        if self._pending:
            self._pending.pop()
            self.view.set_edit_points(self._pending)
            self.btn_commit.setEnabled(len(self._pending) >= 3)

    def _on_commit(self) -> None:
        if len(self._pending) < 3:
            return
        kind = ZoneKind(self.combo_kind.currentText())
        zone = ZoneConfig(
            zone_id=uuid.uuid4().hex[:8],
            name=self.edit_name.text().strip() or f"{kind.value} Zone",
            kind=kind,
            polygon=list(self._pending),
            color=kind.default_color,
            severity=kind.default_severity,
            max_occupancy=self.spin_occupancy.value(),
        )
        self._zones.append(zone)
        self._pending = []
        self.view.set_edit_points([])
        self.btn_draw.setChecked(False)
        self.btn_commit.setEnabled(False)
        self.refresh_table()
        self.zonesChanged.emit()
        self.statusMessage.emit(f"Zone '{zone.name}' created")

    # -- preview -------------------------------------------------------------
    def show_frame(self, frame: "np.ndarray") -> None:
        self.view.set_frame(frame)

    def show_placeholder(self, title: str, subtitle: str = "") -> None:
        self.view.show_placeholder(title, subtitle)
