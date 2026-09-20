"""Right-hand live analytics dock: alerts, counts, zone status, recent events."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from sentinel.models import (
    FrameResult, LineConfig, SecurityEvent, Severity, ZoneStatus,
)
from sentinel.theme import Palette
from sentinel.ui.event_table import EventTable
from sentinel.ui.widgets import StatCard, card, qcolor

__all__ = ["LivePanel"]


class LivePanel(QWidget):
    """Live operational readout shown beside the video.

    Zone and line rows are editable in place: double-click (or select and press
    Edit) to open the corresponding editor, so an operator can retune a zone
    without leaving the live view.
    """

    #: Seconds after which the alert banner reverts to "no active alerts".
    ALERT_HOLD_S = 20.0

    #: Qt item-data role carrying ("zone"|"line", id) for each status row.
    ROW_REF = int(Qt.ItemDataRole.UserRole) + 1

    editZoneRequested = pyqtSignal(str)   # zone_id
    editLineRequested = pyqtSignal(str)   # line_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lines: List[LineConfig] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 11, 11, 11)
        layout.setSpacing(11)

        # -- active alerts ---------------------------------------------------
        alerts_frame, alerts_layout = card("Active Alerts")
        self.label_alert = QLabel("No active alerts")
        self.label_alert.setWordWrap(True)
        self.label_alert.setStyleSheet(
            f"color: {Palette.OK}; font-weight: 600; font-size: 12px;"
        )
        alerts_layout.addWidget(self.label_alert)
        self.label_alert_detail = QLabel("System nominal")
        self.label_alert_detail.setWordWrap(True)
        self.label_alert_detail.setStyleSheet(
            f"color: {Palette.TEXT_FAINT}; font-size: 10px;"
        )
        alerts_layout.addWidget(self.label_alert_detail)
        layout.addWidget(alerts_frame)

        # -- live counts ------------------------------------------------------
        counts_frame, counts_layout = card("Live Counts")
        grid = QGridLayout()
        grid.setSpacing(8)
        self.cards: Dict[str, StatCard] = {
            "people": StatCard("People", "0", "", Palette.OK),
            "vehicles": StatCard("Vehicles", "0", "", Palette.WARN),
            "tracks": StatCard("Tracks", "0", "", Palette.VIOLET),
            "flow": StatCard("In / Out", "0 / 0", "", Palette.CYAN),
        }
        for index, widget in enumerate(self.cards.values()):
            grid.addWidget(widget, index // 2, index % 2)
        counts_layout.addLayout(grid)
        layout.addWidget(counts_frame)

        # -- zone status (editable) ---------------------------------------------
        zone_frame, zone_layout = card("Zone Status")
        self.list_zones = QListWidget()
        self.list_zones.setMinimumHeight(120)
        self.list_zones.setMaximumHeight(190)
        self.list_zones.setToolTip(
            "Double-click a zone or line to edit its settings"
        )
        self.list_zones.itemDoubleClicked.connect(self._on_row_activated)
        zone_layout.addWidget(self.list_zones)

        zone_actions = QHBoxLayout()
        zone_actions.setSpacing(6)
        self.btn_edit_zone = QPushButton("Edit Selected")
        self.btn_edit_zone.setObjectName("Ghost")
        self.btn_edit_zone.clicked.connect(self._on_edit_clicked)
        zone_actions.addWidget(self.btn_edit_zone)
        zone_actions.addStretch(1)
        hint = QLabel("double-click to edit")
        hint.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 9px;")
        zone_actions.addWidget(hint)
        zone_layout.addLayout(zone_actions)
        layout.addWidget(zone_frame)

        layout.addWidget(self._build_thermal_card())

        # -- recent events ------------------------------------------------------
        events_frame, events_layout = card("Recent Events")
        self.table_recent = EventTable(compact=True)
        self.table_recent.setMinimumHeight(170)
        events_layout.addWidget(self.table_recent)
        layout.addWidget(events_frame, 1)

    # ------------------------------------------------------------------
    def update_counts(self, result: FrameResult) -> None:
        self.cards["people"].set_value(result.people, "in view")
        self.cards["vehicles"].set_value(result.vehicles, "in view")
        self.cards["tracks"].set_value(len(result.tracks), "active")
        self.cards["flow"].set_value(
            f"{result.line_in} / {result.line_out}", "entries / exits"
        )
        self.update_zones(result.zone_status, result.line_counts)
        self.update_thermal(result)

    def _build_thermal_card(self) -> QWidget:
        """Thermal readout row: max, average, hotspot share and scope note."""
        frame, layout = card("Thermal Analytics")
        grid = QGridLayout()
        grid.setSpacing(8)
        self.thermal_cards: Dict[str, StatCard] = {
            "max": StatCard("Max", "-", "", Palette.CRIT),
            "avg": StatCard("Average", "-", "", Palette.WARN),
            "hotspots": StatCard("Hotspots", "0", "", Palette.MAGENTA),
            "coverage": StatCard("Hot Area", "0%", "", Palette.CYAN),
        }
        for index, widget in enumerate(self.thermal_cards.values()):
            grid.addWidget(widget, index // 2, index % 2)
        layout.addLayout(grid)

        self.label_thermal_note = QLabel("Thermal mode off")
        self.label_thermal_note.setWordWrap(True)
        self.label_thermal_note.setStyleSheet(
            f"color: {Palette.TEXT_FAINT}; font-size: 9px;"
        )
        layout.addWidget(self.label_thermal_note)
        self.thermal_frame = frame
        return frame

    def update_thermal(self, result: FrameResult) -> None:
        """Render thermal readings, or an explicit n/a when none are valid."""
        status = result.thermal
        active = status is not None and getattr(status, "active", False)
        self.thermal_frame.setVisible(active)
        if not active:
            return

        self.thermal_cards["max"].set_value(status.format_max(), "peak")
        self.thermal_cards["avg"].set_value(status.format_avg(), "scene")
        self.thermal_cards["hotspots"].set_value(
            len(status.hotspots), "regions"
        )
        self.thermal_cards["coverage"].set_value(
            f"{status.hotspot_fraction * 100:.1f}%", "of frame"
        )
        if status.radiometric:
            self.label_thermal_note.setText(
                "Radiometric: values mapped from the calibrated range you set "
                "for this thermal source."
            )
            self.label_thermal_note.setStyleSheet(
                f"color: {Palette.OK}; font-size: 9px;"
            )
        else:
            # This is the important one: no pretending a webcam measures heat.
            self.label_thermal_note.setText(
                "FALSE COLOUR - values are image intensity, NOT temperature. "
                "Connect a calibrated thermal camera and switch to Radiometric "
                "for real degrees."
            )
            self.label_thermal_note.setStyleSheet(
                f"color: {Palette.WARN}; font-size: 9px;"
            )

    def set_lines(self, lines: Sequence[LineConfig]) -> None:
        """Adopt the counting lines so they can be listed and edited."""
        self._lines = list(lines)

    def update_zones(
        self, statuses: Sequence[ZoneStatus],
        line_counts: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        """Rebuild the status rows for every zone, then every counting line."""
        counts = line_counts or {}
        rows: List[Tuple[str, str, str]] = []   # (ref_kind, ref_id, text)
        colours: List[str] = []

        for status in statuses:
            limit = f" / {status.max_occupancy}" if status.max_occupancy else ""
            flag = "  BREACH" if status.breached else ""
            state = "" if status.enabled else "  (disabled)"
            # Show the object total when non-person objects are also inside, so
            # a vehicle-triggered breach is not displayed as "0 BREACH".
            extra = (f"  ({status.objects} obj)"
                     if status.objects != status.occupancy else "")
            rows.append((
                "zone", status.zone_id,
                f"{status.name}:  {status.occupancy}{limit}{extra}{flag}{state}",
            ))
            colours.append(
                Palette.CRIT if status.breached
                else (status.color if status.enabled else Palette.TEXT_FAINT)
            )

        for line in self._lines:
            inbound, outbound = counts.get(line.line_id, (0, 0))
            state = "" if line.enabled else "  (disabled)"
            rows.append((
                "line", line.line_id,
                f"{line.name}:  in {inbound} / out {outbound}{state}",
            ))
            colours.append(line.color if line.enabled else Palette.TEXT_FAINT)

        if self.list_zones.count() != len(rows):
            self.list_zones.clear()
            for _ in rows:
                self.list_zones.addItem(QListWidgetItem(""))
        for index, ((kind, ref_id, text), colour) in enumerate(zip(rows, colours)):
            item = self.list_zones.item(index)
            if item is None:
                continue
            item.setText(text)
            item.setForeground(QBrush(qcolor(colour)))
            item.setData(self.ROW_REF, f"{kind}:{ref_id}")

    # -- row editing ---------------------------------------------------------
    def _on_row_activated(self, item: QListWidgetItem) -> None:
        self._emit_edit(item)

    def _on_edit_clicked(self) -> None:
        self._emit_edit(self.list_zones.currentItem())

    def _emit_edit(self, item: Optional[QListWidgetItem]) -> None:
        if item is None:
            return
        ref = item.data(self.ROW_REF)
        if not isinstance(ref, str) or ":" not in ref:
            return
        kind, ref_id = ref.split(":", 1)
        if kind == "zone":
            self.editZoneRequested.emit(ref_id)
        elif kind == "line":
            self.editLineRequested.emit(ref_id)

    def add_events(self, events: Sequence[SecurityEvent]) -> None:
        self.table_recent.add_events(events, limit=120)
        critical = [e for e in events if e.severity is Severity.CRITICAL]
        warnings = [e for e in events if e.severity is Severity.WARNING]
        if critical:
            self.label_alert.setText(f"CRITICAL: {critical[-1].description}")
            self.label_alert.setStyleSheet(
                f"color: {Palette.CRIT}; font-weight: 700; font-size: 12px;"
            )
            self.label_alert_detail.setText(
                f"{len(critical)} critical event(s) in this batch - "
                f"zone: {critical[-1].zone or 'n/a'}"
            )
        elif warnings:
            self.label_alert.setText(f"WARNING: {warnings[-1].description}")
            self.label_alert.setStyleSheet(
                f"color: {Palette.WARN}; font-weight: 700; font-size: 12px;"
            )
            self.label_alert_detail.setText(
                f"{len(warnings)} warning(s) in this batch"
            )

    def clear_alert(self) -> None:
        self.label_alert.setText("No active alerts")
        self.label_alert.setStyleSheet(
            f"color: {Palette.OK}; font-weight: 600; font-size: 12px;"
        )
        self.label_alert_detail.setText("System nominal")

    def clear_events(self) -> None:
        self.table_recent.setRowCount(0)
        self.clear_alert()
