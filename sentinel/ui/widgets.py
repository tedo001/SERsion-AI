"""Reusable presentation widgets and layout helpers.

These are hand-painted rather than pulled from a charting dependency: they
repaint many times a second next to a live video feed, so they stay cheap and
allocate nothing per frame.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF,
)
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from sentinel.theme import Palette

__all__ = [
    "qcolor", "make_scroll", "card", "divider",
    "StatusPill", "StatCard", "Sparkline", "BarMeter", "PipelineDiagram",
]


def qcolor(value: str, alpha: int = 255) -> QColor:
    color = QColor(value)
    color.setAlpha(alpha)
    return color

def make_scroll(widget: QWidget, horizontal: bool = False) -> QScrollArea:
    """Wrap a widget in a themed scroll area.

    Every side/config panel goes through this, which is what keeps the UI
    usable at 1366x768 - nothing can become unreachable when the window is
    small.
    """
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded if horizontal
        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return area

def card(title: str = "", spacing: int = 10) -> Tuple[QFrame, QVBoxLayout]:
    """Create a titled card frame and return (frame, content layout)."""
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 14)
    layout.setSpacing(spacing)
    if title:
        label = QLabel(title.upper())
        label.setObjectName("Caption")
        layout.addWidget(label)
    return frame, layout

def divider(vertical: bool = False) -> QFrame:
    line = QFrame()
    line.setObjectName("VDivider" if vertical else "Divider")
    line.setFrameShape(QFrame.Shape.VLine if vertical else QFrame.Shape.HLine)
    return line

class StatusPill(QLabel):
    """Compact coloured status chip used across the top bar and panels."""

    def __init__(self, text: str = "", color: str = Palette.TEXT_DIM,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._color = color
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(22)
        self._restyle()

    def set_status(self, text: str, color: str) -> None:
        if text == self.text() and color == self._color:
            return
        self._color = color
        self.setText(text)
        self._restyle()

    def _restyle(self) -> None:
        self.setStyleSheet(
            f"QLabel {{ color: {self._color};"
            f" background-color: {qcolor(self._color, 28).name(QColor.NameFormat.HexArgb)};"
            f" border: 1px solid {qcolor(self._color, 90).name(QColor.NameFormat.HexArgb)};"
            f" border-radius: 11px; padding: 3px 10px;"
            f" font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }}"
        )

class StatCard(QFrame):
    """Metric tile: caption, big value, optional sub-text and accent bar."""

    def __init__(self, caption: str, value: str = "-", sub: str = "",
                 accent: str = Palette.ACCENT, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self._accent = accent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 11, 13, 11)
        layout.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(6)
        self._dot = QLabel()
        self._dot.setFixedSize(7, 7)
        self._dot.setStyleSheet(
            f"background-color: {accent}; border-radius: 3px;"
        )
        self._caption = QLabel(caption.upper())
        self._caption.setObjectName("Caption")
        head.addWidget(self._dot)
        head.addWidget(self._caption)
        head.addStretch(1)
        layout.addLayout(head)

        self._value = QLabel(value)
        self._value.setObjectName("Metric")
        layout.addWidget(self._value)

        self._sub = QLabel(sub)
        self._sub.setObjectName("Dim")
        self._sub.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
        self._sub.setVisible(bool(sub))
        layout.addWidget(self._sub)
        self.setMinimumWidth(96)

    def set_value(self, value: Any, sub: str = "", accent: Optional[str] = None) -> None:
        text = str(value)
        if self._value.text() != text:
            self._value.setText(text)
        if sub and self._sub.text() != sub:
            self._sub.setText(sub)
            self._sub.setVisible(True)
        if accent and accent != self._accent:
            self._accent = accent
            self._dot.setStyleSheet(f"background-color: {accent}; border-radius: 3px;")
            self._value.setStyleSheet(f"color: {accent};")

class Sparkline(QWidget):
    """Lightweight time-series chart.

    Hand-painted rather than pulled from a charting dependency: it needs to
    repaint many times a second next to a live video feed, so it stays cheap
    and never allocates per-frame.
    """

    def __init__(self, title: str = "", color: str = Palette.ACCENT,
                 unit: str = "", fill: bool = True, max_points: int = 120,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._color = QColor(color)
        self._unit = unit
        self._fill = fill
        self._max_points = max_points
        self._values: List[float] = []
        self._minimum = 0.0
        self._maximum = 1.0
        self.setMinimumHeight(74)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_values(self, values: Sequence[float]) -> None:
        self._values = list(values)[-self._max_points:]
        if self._values:
            self._minimum = min(self._values)
            self._maximum = max(self._values)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(Palette.BG_CARD))
        painter.drawRoundedRect(QRectF(rect), 8, 8)
        painter.setPen(QPen(qcolor(Palette.BORDER), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(rect), 8, 8)

        pad_left, pad_top, pad_right, pad_bottom = 10, 22, 10, 10
        plot = QRectF(
            rect.left() + pad_left, rect.top() + pad_top,
            max(1.0, rect.width() - pad_left - pad_right),
            max(1.0, rect.height() - pad_top - pad_bottom),
        )

        # Header: title on the left, latest value on the right.
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(qcolor(Palette.TEXT_FAINT))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 5, rect.width() - 20, 16),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._title.upper(),
        )
        if self._values:
            painter.setPen(self._color)
            latest = self._values[-1]
            text = f"{latest:.1f}{self._unit}" if latest < 1000 else f"{latest:.0f}{self._unit}"
            painter.drawText(
                QRectF(rect.left() + 10, rect.top() + 5, rect.width() - 20, 16),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                text,
            )

        if len(self._values) < 2:
            painter.setPen(qcolor(Palette.TEXT_FAINT))
            painter.drawText(plot, int(Qt.AlignmentFlag.AlignCenter), "collecting data ...")
            painter.end()
            return

        span = max(1e-6, self._maximum - self._minimum)
        if span < 1e-3:                       # flat series: centre the line
            span = 1.0
        count = len(self._values)
        step = plot.width() / max(1, count - 1)

        points: List[QPointF] = []
        for i, value in enumerate(self._values):
            normalised = (value - self._minimum) / span
            x = plot.left() + i * step
            y = plot.bottom() - normalised * plot.height() * 0.88 - plot.height() * 0.06
            points.append(QPointF(x, y))

        # Baseline grid.
        painter.setPen(QPen(qcolor(Palette.BORDER), 1, Qt.PenStyle.DotLine))
        for frac in (0.0, 0.5, 1.0):
            y = plot.bottom() - frac * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)

        if self._fill:
            area = QPainterPath(path)
            area.lineTo(QPointF(points[-1].x(), plot.bottom()))
            area.lineTo(QPointF(points[0].x(), plot.bottom()))
            area.closeSubpath()
            gradient = QLinearGradient(0.0, plot.top(), 0.0, plot.bottom())
            gradient.setColorAt(0.0, qcolor(self._color.name(), 90))
            gradient.setColorAt(1.0, qcolor(self._color.name(), 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawPath(area)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self._color, 1.6))
        painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(points[-1], 2.6, 2.6)
        painter.end()

class BarMeter(QWidget):
    """Horizontal labelled meter (CPU, RAM, GPU, queue fill)."""

    def __init__(self, label: str, color: str = Palette.ACCENT,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label = label
        self._color = color
        self._value = 0.0
        self._text = "-"
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, fraction: float, text: str) -> None:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if abs(fraction - self._value) < 0.005 and text == self._text:
            return
        self._value = fraction
        self._text = text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(qcolor(Palette.TEXT_DIM))
        painter.drawText(
            QRectF(rect.left(), rect.top(), rect.width(), 14),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._label,
        )
        painter.setPen(qcolor(Palette.TEXT))
        painter.drawText(
            QRectF(rect.left(), rect.top(), rect.width(), 14),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            self._text,
        )

        track = QRectF(rect.left(), rect.top() + 20.0, rect.width(), 6.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(Palette.BG_ELEV))
        painter.drawRoundedRect(track, 3, 3)

        if self._value > 0.0:
            colour = self._color
            if self._value > 0.9:
                colour = Palette.CRIT
            elif self._value > 0.75:
                colour = Palette.WARN
            filled = QRectF(track)
            filled.setWidth(max(4.0, track.width() * self._value))
            painter.setBrush(qcolor(colour))
            painter.drawRoundedRect(filled, 3, 3)
        painter.end()

class PipelineDiagram(QWidget):
    """Live block diagram of the edge pipeline stages."""

    STAGES = ("Camera", "Capture", "Preprocess", "Detector",
              "Kalman Tracker", "Event Engine", "Dashboard")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._active_index = -1
        self._labels: Dict[str, str] = {}
        self._healthy = True
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_state(self, active_index: int, labels: Dict[str, str], healthy: bool = True) -> None:
        self._active_index = active_index
        self._labels = labels
        self._healthy = healthy
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(2, 2, -2, -2)
        count = len(self.STAGES)
        gap = 8.0
        box_w = max(56.0, (rect.width() - gap * (count - 1)) / count)
        box_h = 40.0
        top = rect.top() + 14.0

        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)

        for i, stage in enumerate(self.STAGES):
            x = rect.left() + i * (box_w + gap)
            box = QRectF(x, top, box_w, box_h)
            active = self._healthy and i <= self._active_index
            border = (Palette.ACCENT if active else Palette.BORDER)
            if not self._healthy:
                border = Palette.CRIT

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(qcolor(border, 26) if active else qcolor(Palette.BG_ELEV))
            painter.drawRoundedRect(box, 6, 6)
            painter.setPen(QPen(qcolor(border), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box, 6, 6)

            painter.setPen(qcolor(Palette.TEXT if active else Palette.TEXT_FAINT))
            painter.drawText(
                box.adjusted(2, 3, -2, -box_h / 2),
                int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                stage,
            )
            detail = self._labels.get(stage, "")
            if detail:
                painter.setPen(qcolor(Palette.ACCENT if active else Palette.TEXT_FAINT))
                painter.drawText(
                    QRectF(box.left(), box.bottom() - 16, box.width(), 14),
                    int(Qt.AlignmentFlag.AlignCenter), detail,
                )

            if i < count - 1:
                arrow_y = top + box_h / 2
                ax1 = box.right() + 1.5
                ax2 = box.right() + gap - 1.5
                painter.setPen(QPen(qcolor(Palette.BORDER_STRONG), 1))
                painter.drawLine(QPointF(ax1, arrow_y), QPointF(ax2, arrow_y))
                head = QPolygonF([
                    QPointF(ax2, arrow_y), QPointF(ax2 - 3, arrow_y - 2.5),
                    QPointF(ax2 - 3, arrow_y + 2.5),
                ])
                painter.setBrush(qcolor(Palette.BORDER_STRONG))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(head)
        painter.end()
