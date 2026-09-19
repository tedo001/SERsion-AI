"""Startup system-check dialog.

Reports only what was actually probed - nothing here is assumed, and every
missing item carries its real install command.
"""

from __future__ import annotations

import os
import platform
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from sentinel.config import APP_NAME, APP_SUBTITLE, APP_VERSION
from sentinel.deps import (
    DEP_PSUTIL, DEP_RFDETR, DEP_SCIPY, DEP_SUPERVISION, DEP_TORCH,
    DEP_TRANSFORMERS, QT_AVAILABLE,
)
from sentinel.detection import RFDETRDetector, RTDETRDetector, cuda_device_name
from sentinel.theme import Palette
from sentinel.ui.widgets import card, divider, make_scroll

__all__ = ["DependencyDialog"]


class DependencyDialog(QDialog):
    """Startup status screen: core stack, model backends, device.

    Reports only what was actually probed - nothing here is assumed or
    fabricated, and every missing item comes with its real install command.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - System Check")
        self.setMinimumSize(620, 560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        header = QVBoxLayout()
        header.setSpacing(2)
        title = QLabel(APP_NAME)
        title.setObjectName("H1")
        subtitle = QLabel(f"{APP_SUBTITLE}  -  v{APP_VERSION}")
        subtitle.setObjectName("Dim")
        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)
        layout.addWidget(divider())

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 6, 0)
        body_layout.setSpacing(14)

        body_layout.addWidget(self._section("Core Runtime", self._core_rows()))
        body_layout.addWidget(self._section("Detection Backends", self._model_rows()))
        body_layout.addWidget(self._section("Compute Device", self._device_rows()))

        note = QLabel(
            "Simulation Mode runs without any model backend, so the full UI, "
            "tracking, zones, events and analytics are always demonstrable. "
            "Install a backend above to run inference on real video.\n\n"
            "This system performs object detection and tracking only. It does "
            "not perform facial recognition or identity inference, and stores "
            "no biometric data."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {Palette.TEXT_DIM}; background-color: {Palette.BG_CARD};"
            f" border: 1px solid {Palette.BORDER}; border-radius: 8px; padding: 11px;"
        )
        body_layout.addWidget(note)
        body_layout.addStretch(1)
        layout.addWidget(make_scroll(body), 1)

        buttons = QHBoxLayout()
        self._skip = QCheckBox("Don't show this again")
        buttons.addWidget(self._skip)
        buttons.addStretch(1)
        launch = QPushButton("Launch Console")
        launch.setObjectName("Primary")
        launch.clicked.connect(self.accept)
        launch.setDefault(True)
        buttons.addWidget(launch)
        layout.addLayout(buttons)

    @property
    def skip_next_time(self) -> bool:
        return self._skip.isChecked()

    @staticmethod
    def _section(title: str, rows: Sequence[Tuple[bool, str, str, str]]) -> QFrame:
        frame, layout = card(title)
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(2, 1)
        for row, (ok, name, version, detail) in enumerate(rows):
            mark = QLabel("OK" if ok else "X")
            mark.setFixedWidth(20)
            mark.setStyleSheet(
                f"color: {Palette.OK if ok else Palette.CRIT};"
                " font-weight: 800; font-size: 10px;"
            )
            label = QLabel(name)
            label.setStyleSheet("font-weight: 600;")
            version_label = QLabel(version)
            version_label.setStyleSheet(f"color: {Palette.TEXT_FAINT};")
            grid.addWidget(mark, row, 0)
            grid.addWidget(label, row, 1)
            grid.addWidget(version_label, row, 2)
            if detail:
                hint = QLabel(detail)
                hint.setWordWrap(True)
                hint.setStyleSheet(
                    f"color: {Palette.WARN if not ok else Palette.TEXT_FAINT};"
                    " font-size: 10px;"
                )
                grid.addWidget(hint, row, 3)
        layout.addLayout(grid)
        return frame

    @staticmethod
    def _core_rows() -> List[Tuple[bool, str, str, str]]:
        rows = [
            (True, "Python", platform.python_version(), ""),
            (QT_AVAILABLE, "PyQt6", "installed" if QT_AVAILABLE else "-",
             "" if QT_AVAILABLE else "pip install PyQt6"),
            (True, "OpenCV", cv2.__version__, ""),
            (True, "NumPy", np.__version__, ""),
        ]
        for probe in (DEP_TORCH, DEP_SUPERVISION, DEP_PSUTIL, DEP_SCIPY):
            rows.append((
                probe.available, probe.name, probe.version,
                "" if probe.available else probe.install_hint,
            ))
        return rows

    @staticmethod
    def _model_rows() -> List[Tuple[bool, str, str, str]]:
        rt_ok, rt_reason = RTDETRDetector.dependency_status()
        rf_ok, rf_reason = RFDETRDetector.dependency_status()
        return [
            (rt_ok, "RT-DETR", DEP_TRANSFORMERS.version if rt_ok else "-",
             "" if rt_ok else rt_reason),
            (rf_ok, "RF-DETR", DEP_RFDETR.version if rf_ok else "-",
             "" if rf_ok else rf_reason),
            (True, "Simulation (synthetic GT)", "built-in", ""),
        ]

    @staticmethod
    def _device_rows() -> List[Tuple[bool, str, str, str]]:
        gpu = cuda_device_name()
        rows = [(True, "CPU", f"{os.cpu_count() or '?'} cores  |  {platform.machine()}", "")]
        if gpu:
            rows.append((True, "CUDA GPU", gpu, ""))
        elif DEP_TORCH.available:
            rows.append((False, "CUDA GPU", "not available",
                         "Inference will run on CPU."))
        else:
            rows.append((False, "CUDA GPU", "unknown",
                         "Install PyTorch to detect GPU support."))
        return rows
