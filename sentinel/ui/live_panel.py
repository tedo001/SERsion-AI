"""Right-hand live analytics dock: alerts, counts, zone status, recent events."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QGridLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from sentinel.models import FrameResult, SecurityEvent, Severity, ZoneStatus
from sentinel.theme import Palette
from sentinel.ui.event_table import EventTable
from sentinel.ui.widgets import StatCard, card, qcolor

__all__ = ["LivePanel"]


class LivePanel(QWidget):
    """Live operational readout shown beside the video."""

    #: Seconds after which the alert banner reverts to "no active alerts".
    ALERT_HOLD_S = 20.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
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

        # -- zone status -------------------------------------------------------
        zone_frame, zone_layout = card("Zone Status")
        self.list_zones = QListWidget()
        self.list_zones.setMinimumHeight(120)
        self.list_zones.setMaximumHeight(190)
        zone_layout.addWidget(self.list_zones)
        layout.addWidget(zone_frame)

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
        self.update_zones(result.zone_status)

    def update_zones(self, statuses: Sequence[ZoneStatus]) -> None:
        if self.list_zones.count() != len(statuses):
            self.list_zones.clear()
            for _ in statuses:
                self.list_zones.addItem(QListWidgetItem(""))
        for index, status in enumerate(statuses):
            item = self.list_zones.item(index)
            if item is None:
                continue
            limit = f" / {status.max_occupancy}" if status.max_occupancy else ""
            flag = "  BREACH" if status.breached else ""
            state = "" if status.enabled else "  (disabled)"
            # Show the object total when non-person objects are also inside, so
            # a vehicle-triggered breach is not displayed as "0 BREACH".
            extra = (f"  ({status.objects} obj)"
                     if status.objects != status.occupancy else "")
            item.setText(
                f"{status.name}:  {status.occupancy}{limit}{extra}{flag}{state}"
            )
            colour = (Palette.CRIT if status.breached
                      else (status.color if status.enabled else Palette.TEXT_FAINT))
            item.setForeground(QBrush(qcolor(colour)))

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
