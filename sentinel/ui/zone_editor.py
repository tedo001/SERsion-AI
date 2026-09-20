"""Inline editors for zones and counting lines.

Reachable from the Zone Status panel, so an operator can retune a zone while
watching the live feed instead of navigating to the Zones page first.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QGridLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from sentinel.models import LineConfig, Severity, ZoneConfig, ZoneKind
from sentinel.theme import Palette

__all__ = ["ZoneEditorDialog", "LineEditorDialog"]


class _ColorButton(QPushButton):
    """Swatch button that opens the system colour picker."""

    def __init__(self, color: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._pick)
        self._restyle()

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self._restyle()

    def _restyle(self) -> None:
        self.setText(self._color.upper())
        # Choose readable text for the chosen swatch.
        chosen = QColor(self._color)
        text = "#0A0E13" if chosen.lightness() > 140 else "#E4EDF6"
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._color}; color: {text};"
            f" border: 1px solid {Palette.BORDER_STRONG}; border-radius: 6px;"
            f" font-weight: 700; font-size: 11px; }}"
        )

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, "Zone colour")
        if chosen.isValid():
            self.set_color(chosen.name())


class ZoneEditorDialog(QDialog):
    """Edit one :class:`ZoneConfig`.

    Applies to the zone object only when accepted, so Cancel is a true undo.
    """

    def __init__(self, zone: ZoneConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zone = zone
        self.setWindowTitle(f"Edit Zone - {zone.name}")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        heading = QLabel("Zone Configuration")
        heading.setObjectName("H2")
        layout.addWidget(heading)

        subtitle = QLabel(
            f"{len(zone.polygon)} polygon points  -  redraw the shape on the "
            "Zones page"
        )
        subtitle.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setSpacing(9)
        grid.setColumnStretch(1, 1)
        row = 0

        def add(label: str, widget: QWidget) -> None:
            nonlocal row
            caption = QLabel(label)
            caption.setObjectName("Caption")
            grid.addWidget(caption, row, 0)
            grid.addWidget(widget, row, 1)
            row += 1

        self.edit_name = QLineEdit(zone.name)
        add("Name", self.edit_name)

        self.combo_kind = QComboBox()
        self.combo_kind.addItems([k.value for k in ZoneKind])
        self.combo_kind.setCurrentText(zone.kind.value)
        self.combo_kind.currentTextChanged.connect(self._on_kind_changed)
        add("Type", self.combo_kind)

        self.combo_severity = QComboBox()
        self.combo_severity.addItems([s.value for s in Severity])
        self.combo_severity.setCurrentText(zone.severity.value)
        add("Severity", self.combo_severity)

        self.btn_color = _ColorButton(zone.color)
        add("Colour", self.btn_color)

        self.spin_occupancy = QSpinBox()
        self.spin_occupancy.setRange(0, 200)
        self.spin_occupancy.setSpecialValueText("disabled")
        self.spin_occupancy.setValue(zone.max_occupancy)
        self.spin_occupancy.setToolTip(
            "Raise a CROWDING event when more people than this are inside."
        )
        add("Max occupancy", self.spin_occupancy)

        self.spin_unattended = QDoubleSpinBox()
        self.spin_unattended.setRange(0.0, 3600.0)
        self.spin_unattended.setDecimals(0)
        self.spin_unattended.setSuffix(" s")
        self.spin_unattended.setSpecialValueText("disabled")
        self.spin_unattended.setValue(zone.unattended_after_s)
        self.spin_unattended.setToolTip(
            "Raise an UNATTENDED_AREA event when the zone stays empty this long."
        )
        add("Unattended after", self.spin_unattended)

        self.check_enabled = QCheckBox("Zone is active")
        self.check_enabled.setChecked(zone.enabled)
        add("Enabled", self.check_enabled)
        layout.addLayout(grid)

        self.label_hint = QLabel()
        self.label_hint.setWordWrap(True)
        self.label_hint.setStyleSheet(
            f"color: {Palette.TEXT_DIM}; font-size: 10px;"
            f" background-color: {Palette.BG_ELEV}; border-radius: 6px;"
            f" padding: 8px;"
        )
        layout.addWidget(self.label_hint)
        self._update_hint(zone.kind)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_kind_changed(self, text: str) -> None:
        """Follow the type's defaults unless the user has overridden them."""
        try:
            kind = ZoneKind(text)
        except ValueError:
            return
        self.combo_severity.setCurrentText(kind.default_severity.value)
        self.btn_color.set_color(kind.default_color)
        self._update_hint(kind)

    def _update_hint(self, kind: ZoneKind) -> None:
        if kind in (ZoneKind.RESTRICTED, ZoneKind.DANGER):
            text = ("Entry by any tracked object raises RESTRICTED_ZONE_ENTRY "
                    "at this zone's severity.")
        elif kind is ZoneKind.MACHINE:
            text = ("People coming within the machine proximity radius (set on "
                    "the Settings page) raise PROXIMITY_WARNING. This is a "
                    "geometric distance check, not a machine-state signal.")
        elif kind is ZoneKind.ENTRANCE:
            text = "Monitored area; combine with the entry/exit line for counts."
        else:
            text = ("Occupancy is counted and compared against the limits "
                    "above.")
        self.label_hint.setText(text)

    def apply(self) -> None:
        """Write the edited values back onto the zone."""
        zone = self._zone
        zone.name = self.edit_name.text().strip() or zone.name
        zone.kind = ZoneKind(self.combo_kind.currentText())
        zone.severity = Severity(self.combo_severity.currentText())
        zone.color = self.btn_color.color
        zone.max_occupancy = self.spin_occupancy.value()
        zone.unattended_after_s = float(self.spin_unattended.value())
        zone.enabled = self.check_enabled.isChecked()


class LineEditorDialog(QDialog):
    """Edit one :class:`LineConfig`, including its endpoints.

    Endpoints are normalised (0-1), so a line keeps its meaning when the source
    resolution changes.
    """

    def __init__(self, line: LineConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._line = line
        self.setWindowTitle(f"Edit Line - {line.name}")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        heading = QLabel("Counting Line")
        heading.setObjectName("H2")
        layout.addWidget(heading)

        grid = QGridLayout()
        grid.setSpacing(9)
        grid.setColumnStretch(1, 1)
        row = 0

        def add(label: str, widget: QWidget) -> None:
            nonlocal row
            caption = QLabel(label)
            caption.setObjectName("Caption")
            grid.addWidget(caption, row, 0)
            grid.addWidget(widget, row, 1)
            row += 1

        self.edit_name = QLineEdit(line.name)
        add("Name", self.edit_name)

        self.btn_color = _ColorButton(line.color)
        add("Colour", self.btn_color)

        self.spins = {}
        for key, label, value in (
            ("x1", "Start X", line.p1[0]), ("y1", "Start Y", line.p1[1]),
            ("x2", "End X", line.p2[0]), ("y2", "End Y", line.p2[1]),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.02)
            spin.setDecimals(3)
            spin.setValue(float(value))
            self.spins[key] = spin
            add(label, spin)

        self.check_enabled = QCheckBox("Line is active")
        self.check_enabled.setChecked(line.enabled)
        add("Enabled", self.check_enabled)
        layout.addLayout(grid)

        hint = QLabel(
            "Coordinates are fractions of the frame (0 = left/top, 1 = "
            "right/bottom), so the line survives a resolution change. Crossings "
            "raise LINE_CROSSED plus PERSON_ENTERED or PERSON_EXITED depending "
            "on direction."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {Palette.TEXT_DIM}; font-size: 10px;"
            f" background-color: {Palette.BG_ELEV}; border-radius: 6px;"
            f" padding: 8px;"
        )
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self) -> None:
        line = self._line
        line.name = self.edit_name.text().strip() or line.name
        line.color = self.btn_color.color
        line.p1 = (self.spins["x1"].value(), self.spins["y1"].value())
        line.p2 = (self.spins["x2"].value(), self.spins["y2"].value())
        line.enabled = self.check_enabled.isChecked()
