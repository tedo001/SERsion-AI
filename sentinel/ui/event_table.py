"""Severity-coloured event table used by the right panel and the Events page."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional, Sequence

from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QHeaderView, QTableWidget, QTableWidgetItem, QWidget,
)

from sentinel.models import SecurityEvent, Severity
from sentinel.theme import Palette
from sentinel.ui.widgets import qcolor

__all__ = ["EventTable"]


class EventTable(QTableWidget):
    """Severity-coloured event table used by the right panel and Events page."""

    COLUMNS = ("Time", "Severity", "Type", "Track", "Zone", "Description")

    def __init__(self, compact: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, len(self.COLUMNS), parent)
        self._compact = compact
        self.setHorizontalHeaderLabels([c.upper() for c in self.COLUMNS])
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setWordWrap(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            len(self.COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
        )
        self.verticalHeader().setDefaultSectionSize(24 if compact else 28)
        if compact:
            # The compact variant lives in a narrow dock: hide detail columns.
            for column in (2, 3, 4):
                self.setColumnHidden(column, True)

    def add_events(self, events: Sequence[SecurityEvent], limit: int = 300) -> None:
        for event in events:
            self.insertRow(0)
            values = (
                event.clock, event.severity.value, event.event_type.value,
                "" if event.track_id is None else f"#{event.track_id}",
                event.zone or "", event.description,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 1:
                    item.setForeground(QBrush(qcolor(event.severity.color)))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                elif column == 0:
                    item.setForeground(QBrush(qcolor(Palette.TEXT_FAINT)))
                elif event.severity is Severity.CRITICAL:
                    item.setForeground(QBrush(qcolor(Palette.CRIT)))
                self.setItem(0, column, item)
        while self.rowCount() > limit:
            self.removeRow(self.rowCount() - 1)

    def load_rows(self, rows: Sequence[sqlite3.Row]) -> None:
        """Replace contents with rows read back from SQLite."""
        self.setRowCount(0)
        for row in rows:
            index = self.rowCount()
            self.insertRow(index)
            try:
                clock = datetime.fromtimestamp(float(row["ts"])).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError, KeyError, IndexError):
                clock = str(row["iso_ts"])
            severity = str(row["severity"])
            values = (
                clock, severity, str(row["event_type"]),
                "" if row["track_id"] is None else f"#{row['track_id']}",
                row["zone"] or "", str(row["description"]),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 1:
                    try:
                        item.setForeground(QBrush(qcolor(Severity(severity).color)))
                    except ValueError:
                        pass
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.setItem(index, column, item)
