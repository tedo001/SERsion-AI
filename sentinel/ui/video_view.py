"""Aspect-correct video surface with zoom, pan and zone editing."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from sentinel.config import LOGGER
from sentinel.theme import Palette
from sentinel.ui.widgets import qcolor

__all__ = ["VideoView"]


class VideoView(QWidget):
    """Aspect-correct video surface with zoom, pan and zone editing.

    Memory discipline (important on a long-running surveillance client):
        * exactly one QImage is kept alive at a time, and it OWNS its pixel
          buffer (``.copy()`` of the RGB ndarray) - handing QImage a NumPy
          buffer that Python may free is the classic crash in Qt CV apps;
        * the QPixmap is rebuilt only when the widget actually repaints, so
          incoming frames never accumulate pixmaps;
        * frames arriving faster than the repaint timer simply overwrite the
          pending image.
    """

    pointClicked = pyqtSignal(float, float)   # normalised frame coords
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)

        self._image: Optional[QImage] = None
        self._frame_size: Tuple[int, int] = (0, 0)
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_origin = QPointF(0.0, 0.0)

        self._placeholder_title = "NO SIGNAL"
        self._placeholder_sub = "Select a video source to begin"
        self._placeholder_accent = Palette.TEXT_FAINT
        self._badge: Optional[Tuple[str, str]] = None
        self._edit_mode = False
        self._edit_points: List[Tuple[float, float]] = []
        self._last_paint_rect = QRectF()

    # -- content ----------------------------------------------------------
    def set_frame(self, frame: "np.ndarray") -> None:
        """Accept a BGR ndarray and keep a single owned QImage."""
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            height, width, _ = rgb.shape
            # .copy() detaches the QImage from the NumPy buffer.
            image = QImage(rgb.data, width, height, width * 3,
                           QImage.Format.Format_RGB888).copy()
        except Exception as exc:  # noqa: BLE001 - malformed frame
            LOGGER.debug("Frame conversion failed: %s", exc)
            return
        self._image = image
        self._frame_size = (width, height)
        self._placeholder_title = ""
        self.update()

    def show_placeholder(self, title: str, subtitle: str = "",
                         accent: str = Palette.TEXT_FAINT) -> None:
        self._image = None
        self._placeholder_title = title
        self._placeholder_sub = subtitle
        self._placeholder_accent = accent
        self.update()

    def set_badge(self, text: Optional[str], color: str = Palette.VIOLET) -> None:
        self._badge = (text, color) if text else None
        self.update()

    def clear(self) -> None:
        self._image = None
        self._frame_size = (0, 0)
        self.update()

    # -- zoom / pan --------------------------------------------------------
    def set_zoom(self, zoom: float) -> None:
        zoom = float(np.clip(zoom, 1.0, 5.0))
        if abs(zoom - self._zoom) < 1e-3:
            return
        self._zoom = zoom
        if zoom <= 1.0:
            self._pan = QPointF(0.0, 0.0)
        self.zoomChanged.emit(self._zoom)
        self.update()

    @property
    def zoom(self) -> float:
        return self._zoom

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.zoomChanged.emit(1.0)
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.set_zoom(self._zoom * (1.12 if delta > 0 else 1 / 1.12))
        event.accept()

    # -- zone editing ------------------------------------------------------
    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = bool(enabled)
        self._edit_points = []
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if enabled
                               else Qt.CursorShape.ArrowCursor))
        self.update()

    def set_edit_points(self, points: Sequence[Tuple[float, float]]) -> None:
        self._edit_points = list(points)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._edit_mode:
                normalised = self._widget_to_frame(event.position())
                if normalised is not None:
                    self.pointClicked.emit(normalised[0], normalised[1])
            elif self._zoom > 1.0:
                self._panning = True
                self._pan_origin = event.position()
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self._pan += delta
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()
        event.accept()

    # -- geometry ----------------------------------------------------------
    def _target_rect(self) -> QRectF:
        """Aspect-preserving destination rect for the current frame."""
        width, height = self._frame_size
        area = QRectF(self.rect())
        if width <= 0 or height <= 0:
            return area
        frame_ratio = width / height
        area_ratio = area.width() / max(1.0, area.height())
        if frame_ratio > area_ratio:
            draw_w = area.width()
            draw_h = draw_w / frame_ratio
        else:
            draw_h = area.height()
            draw_w = draw_h * frame_ratio
        draw_w *= self._zoom
        draw_h *= self._zoom
        x = area.center().x() - draw_w / 2 + self._pan.x()
        y = area.center().y() - draw_h / 2 + self._pan.y()
        return QRectF(x, y, draw_w, draw_h)

    def _widget_to_frame(self, position: QPointF) -> Optional[Tuple[float, float]]:
        rect = self._last_paint_rect
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        nx = (position.x() - rect.left()) / rect.width()
        ny = (position.y() - rect.top()) / rect.height()
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return None
        return (float(nx), float(ny))

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), qcolor("#05080B"))

        if self._image is None:
            self._paint_placeholder(painter)
            painter.end()
            return

        target = self._target_rect()
        self._last_paint_rect = target
        painter.save()
        if self._zoom > 1.0:
            painter.setClipRect(self.rect())
        painter.drawImage(target, self._image)
        painter.restore()

        # Thin frame border around the video area.
        painter.setPen(QPen(qcolor(Palette.BORDER_STRONG), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(target.adjusted(-0.5, -0.5, 0.5, 0.5))

        if self._edit_mode:
            self._paint_edit_overlay(painter, target)
        if self._badge is not None:
            self._paint_badge(painter)
        if self._zoom > 1.0:
            self._paint_zoom_hint(painter)
        painter.end()

    def _paint_placeholder(self, painter: QPainter) -> None:
        rect = QRectF(self.rect())
        painter.setPen(QPen(qcolor(Palette.BORDER), 1, Qt.PenStyle.DashLine))
        painter.drawRect(rect.adjusted(12, 12, -12, -12))

        font = painter.font()
        font.setPointSize(17)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(qcolor(self._placeholder_accent))
        painter.drawText(rect.adjusted(0, -14, 0, -14),
                         int(Qt.AlignmentFlag.AlignCenter),
                         self._placeholder_title or "NO SIGNAL")
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(qcolor(Palette.TEXT_FAINT))
        painter.drawText(rect.adjusted(0, 30, 0, 30),
                         int(Qt.AlignmentFlag.AlignCenter), self._placeholder_sub)

    def _paint_edit_overlay(self, painter: QPainter, target: QRectF) -> None:
        painter.setPen(QPen(qcolor(Palette.CYAN, 120), 1, Qt.PenStyle.DashLine))
        painter.drawRect(target)
        if not self._edit_points:
            return
        points = [
            QPointF(target.left() + nx * target.width(),
                    target.top() + ny * target.height())
            for nx, ny in self._edit_points
        ]
        painter.setPen(QPen(qcolor(Palette.CYAN), 2))
        painter.setBrush(qcolor(Palette.CYAN, 40))
        if len(points) >= 3:
            painter.drawPolygon(QPolygonF(points))
        elif len(points) == 2:
            painter.drawLine(points[0], points[1])
        painter.setBrush(qcolor(Palette.CYAN))
        painter.setPen(Qt.PenStyle.NoPen)
        for i, point in enumerate(points):
            painter.drawEllipse(point, 4, 4)
            painter.setPen(qcolor(Palette.TEXT))
            painter.drawText(point + QPointF(7, -6), str(i + 1))
            painter.setPen(Qt.PenStyle.NoPen)

    def _paint_badge(self, painter: QPainter) -> None:
        assert self._badge is not None
        text, color = self._badge
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 18
        box = QRectF(self.width() - width - 12, 12, width, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(color, 230))
        painter.drawRoundedRect(box, 11, 11)
        painter.setPen(qcolor("#0A0E13"))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    def _paint_zoom_hint(self, painter: QPainter) -> None:
        text = f"ZOOM {self._zoom:.1f}x  -  drag to pan, double-click to reset"
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 16
        box = QRectF(12, self.height() - 34, width, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor("#0A0E13", 190))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(qcolor(Palette.TEXT_DIM))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)
