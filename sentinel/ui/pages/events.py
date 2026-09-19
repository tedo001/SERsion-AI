"""Events page: filterable history over the local SQLite store."""

from __future__ import annotations

import sqlite3
from typing import Dict, Optional, Sequence, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QWidget,
)

from sentinel.models import EventType, SecurityEvent, Severity
from sentinel.theme import Palette
from sentinel.ui.event_table import EventTable
from sentinel.ui.widgets import StatCard

__all__ = ["EventsPage"]


class EventsPage(QWidget):
    """Event history browser with severity/type/search filters and CSV export."""

    filtersChanged = pyqtSignal()
    exportRequested = pyqtSignal()
    clearRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)

        # -- header / filters --------------------------------------------------
        header = QHBoxLayout()
        title = QLabel("Event History")
        title.setObjectName("H1")
        header.addWidget(title)
        header.addStretch(1)

        self.combo_severity = QComboBox()
        self.combo_severity.addItem("All severities", None)
        for severity in Severity:
            self.combo_severity.addItem(severity.value, severity.value)

        self.combo_type = QComboBox()
        self.combo_type.addItem("All types", None)
        for event_type in EventType:
            self.combo_type.addItem(event_type.value, event_type.value)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("Search description or zone ...")
        self.edit_search.setFixedWidth(230)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setObjectName("Primary")
        self.btn_clear = QPushButton("Clear History")
        self.btn_clear.setObjectName("Danger")

        for widget in (self.combo_severity, self.combo_type, self.edit_search,
                       self.btn_refresh, self.btn_export, self.btn_clear):
            header.addWidget(widget)
        layout.addLayout(header)

        # -- counters ------------------------------------------------------------
        counts = QHBoxLayout()
        counts.setSpacing(10)
        self.cards: Dict[str, StatCard] = {
            "total": StatCard("Total Events", "0", "stored", Palette.ACCENT),
            "info": StatCard("Info", "0", "", Palette.ACCENT),
            "warning": StatCard("Warnings", "0", "", Palette.WARN),
            "critical": StatCard("Critical", "0", "", Palette.CRIT),
        }
        for widget in self.cards.values():
            counts.addWidget(widget)
        counts.addStretch(1)
        layout.addLayout(counts)

        self.table = EventTable(compact=False)
        layout.addWidget(self.table, 1)

        self.label_db = QLabel()
        self.label_db.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
        layout.addWidget(self.label_db)

        self.btn_refresh.clicked.connect(self.filtersChanged)
        self.btn_export.clicked.connect(self.exportRequested)
        self.btn_clear.clicked.connect(self.clearRequested)
        self.combo_severity.currentIndexChanged.connect(self.filtersChanged)
        self.combo_type.currentIndexChanged.connect(self.filtersChanged)
        self.edit_search.returnPressed.connect(self.filtersChanged)

    # ------------------------------------------------------------------
    def filters(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (severity, event_type, search) as the database expects them."""
        return (
            self.combo_severity.currentData(),
            self.combo_type.currentData(),
            self.edit_search.text().strip() or None,
        )

    def load_rows(self, rows: Sequence[sqlite3.Row]) -> None:
        self.table.load_rows(rows)

    def add_events(self, events: Sequence[SecurityEvent]) -> None:
        self.table.add_events(events, limit=500)

    def set_counts(self, counts: Dict[str, int]) -> None:
        self.cards["total"].set_value(sum(counts.values()), "stored")
        self.cards["info"].set_value(counts.get("INFO", 0))
        self.cards["warning"].set_value(counts.get("WARNING", 0))
        self.cards["critical"].set_value(counts.get("CRITICAL", 0))

    def set_database_status(self, text: str) -> None:
        self.label_db.setText(text)

    def clear(self) -> None:
        self.table.setRowCount(0)
