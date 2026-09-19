"""Analytics page: time series, class histogram and session history."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from sentinel.theme import Palette
from sentinel.ui.widgets import Sparkline, card, make_scroll
from sentinel.vision import AnalyticsEngine
from sentinel.vision.analytics import AnalyticsSnapshot

__all__ = ["AnalyticsPage"]


class AnalyticsPage(QWidget):
    """Read-only analytics view, refreshed on the slow timer."""

    refreshRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Analytics")
        title.setObjectName("H1")
        layout.addWidget(title)

        # -- time series -------------------------------------------------------
        charts_frame, charts_layout = card("Time Series")
        grid = QGridLayout()
        grid.setSpacing(10)
        self.chart_people = Sparkline("People in view", Palette.OK)
        self.chart_tracks = Sparkline("Active tracks", Palette.VIOLET)
        self.chart_vehicles = Sparkline("Vehicles", Palette.WARN)
        self.chart_latency = Sparkline("Pipeline latency", Palette.CYAN, unit=" ms")
        for index, chart in enumerate((self.chart_people, self.chart_tracks,
                                       self.chart_vehicles, self.chart_latency)):
            chart.setMinimumHeight(110)
            grid.addWidget(chart, index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        charts_layout.addLayout(grid)
        layout.addWidget(charts_frame)

        # -- classes + summary --------------------------------------------------
        split = QHBoxLayout()
        split.setSpacing(14)

        class_frame, class_layout = card("Detected Classes (session)")
        self.table_classes = QTableWidget(0, 2)
        self.table_classes.setHorizontalHeaderLabels(["CLASS", "OBSERVATIONS"])
        self.table_classes.verticalHeader().setVisible(False)
        self.table_classes.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table_classes.setMinimumHeight(220)
        class_layout.addWidget(self.table_classes)
        split.addWidget(class_frame, 1)

        summary_frame, summary_layout = card("Session Summary")
        self.label_summary = QLabel("No data yet")
        self.label_summary.setObjectName("Mono")
        self.label_summary.setWordWrap(True)
        self.label_summary.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.label_summary.setStyleSheet(
            f"color: {Palette.TEXT_DIM}; font-size: 11px; line-height: 150%;"
        )
        summary_layout.addWidget(self.label_summary, 1)
        split.addWidget(summary_frame, 1)
        layout.addLayout(split)

        # -- session history -----------------------------------------------------
        history_frame, history_layout = card("Recent Sessions")
        self.table_sessions = QTableWidget(0, 6)
        self.table_sessions.setHorizontalHeaderLabels(
            ["STARTED", "MODE", "SOURCE", "BACKEND", "FRAMES", "EVENTS"]
        )
        self.table_sessions.verticalHeader().setVisible(False)
        self.table_sessions.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_sessions.setMinimumHeight(180)
        history_layout.addWidget(self.table_sessions)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("Ghost")
        self.btn_refresh.clicked.connect(self.refreshRequested)
        history_layout.addWidget(self.btn_refresh, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(history_frame)
        layout.addStretch(1)

        root.addWidget(make_scroll(content))

    # ------------------------------------------------------------------
    def update_series(self, analytics: AnalyticsEngine) -> None:
        self.chart_people.set_values(analytics.series_values("people"))
        self.chart_tracks.set_values(analytics.series_values("track"))
        self.chart_vehicles.set_values(analytics.series_values("vehicle"))
        self.chart_latency.set_values(analytics.series_values("latency"))

    def update_classes(self, snapshot: AnalyticsSnapshot) -> None:
        histogram = snapshot.class_histogram
        self.table_classes.setRowCount(len(histogram))
        for row, (name, count) in enumerate(histogram):
            self.table_classes.setItem(row, 0, QTableWidgetItem(str(name)))
            item = QTableWidgetItem(str(count))
            item.setTextAlignment(
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            )
            self.table_classes.setItem(row, 1, item)

    def set_summary(self, text: str) -> None:
        self.label_summary.setText(text)

    def load_sessions(self, rows: Sequence[sqlite3.Row]) -> None:
        self.table_sessions.setRowCount(len(rows))
        for index, row in enumerate(rows):
            try:
                started = datetime.fromtimestamp(
                    float(row["started_at"])).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError):
                started = "-"
            values = (started, row["mode"] or "-", row["source"] or "-",
                      row["backend"] or "-", str(row["frames"] or 0),
                      str(row["events"] or 0))
            for column, value in enumerate(values):
                self.table_sessions.setItem(
                    index, column, QTableWidgetItem(str(value))
                )
