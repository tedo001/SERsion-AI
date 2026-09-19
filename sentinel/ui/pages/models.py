"""Models page: backend selection, load/unload, and honest availability reporting."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from sentinel.deps import DEP_RFDETR, DEP_TRANSFORMERS
from sentinel.detection import RFDETRDetector, RTDETRDetector
from sentinel.models import BackendKind, ModelState
from sentinel.theme import Palette
from sentinel.ui.widgets import card, make_scroll, qcolor

__all__ = ["ModelsPage"]

#: Modules that are deliberately NOT implemented.  Listed so the roadmap is
#: visible without ever implying the running system can do any of it.
FUTURE_MODULES = (
    "   -  PPE / helmet / vest compliance  (requires a PPE-trained model)",
    "   -  Fall and slip detection         (requires a pose / action model)",
    "   -  Weapon detection                (requires a dedicated model)",
    "   -  Fatigue or drowsiness           (requires a pose / face model)",
)


class ModelsPage(QWidget):
    """Backend control surface.  Loading happens on the inference thread."""

    loadRequested = pyqtSignal(object, object)   # BackendKind, checkpoint
    unloadRequested = pyqtSignal()
    backendChanged = pyqtSignal(object)          # BackendKind

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Detection Models")
        title.setObjectName("H1")
        layout.addWidget(title)

        # -- active backend ----------------------------------------------------
        status_frame, status_layout = card("Active Backend")
        self.label_state = QLabel("No detection backend active")
        self.label_state.setObjectName("H2")
        status_layout.addWidget(self.label_state)

        self.label_detail = QLabel(
            "Simulation mode supplies synthetic ground-truth detections, so "
            "the pipeline runs without a model."
        )
        self.label_detail.setWordWrap(True)
        self.label_detail.setStyleSheet(f"color: {Palette.TEXT_DIM};")
        status_layout.addWidget(self.label_detail)

        actions = QHBoxLayout()
        backend_caption = QLabel("Backend")
        backend_caption.setObjectName("Caption")
        actions.addWidget(backend_caption)
        self.combo_backend = QComboBox()
        self.combo_backend.addItems([b.value for b in BackendKind])
        actions.addWidget(self.combo_backend)

        checkpoint_caption = QLabel("Checkpoint")
        checkpoint_caption.setObjectName("Caption")
        actions.addWidget(checkpoint_caption)
        self.combo_checkpoint = QComboBox()
        actions.addWidget(self.combo_checkpoint, 1)

        self.btn_load = QPushButton("Load Model")
        self.btn_load.setObjectName("Primary")
        self.btn_unload = QPushButton("Unload")
        actions.addWidget(self.btn_load)
        actions.addWidget(self.btn_unload)
        status_layout.addLayout(actions)
        layout.addWidget(status_frame)

        # -- availability -------------------------------------------------------
        avail_frame, avail_layout = card("Backend Availability")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["BACKEND", "STATUS", "VERSION", "DETAIL"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.table.setMinimumHeight(140)
        avail_layout.addWidget(self.table)

        self.btn_recheck = QPushButton("Re-check")
        self.btn_recheck.setObjectName("Ghost")
        avail_layout.addWidget(self.btn_recheck, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(avail_frame)

        # -- scope honesty -------------------------------------------------------
        future_frame, future_layout = card("Future Modules (not implemented)")
        note = QLabel(
            "The modules below are NOT part of this build and are listed only "
            "to document the roadmap. This application ships object detection "
            "and tracking only; it cannot infer any of the following, and no "
            "part of the UI reports them:\n\n"
            + "\n".join(FUTURE_MODULES)
            + "\n\nFacial recognition and identity inference are deliberately "
            "out of scope and will not be added."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 11px;")
        future_layout.addWidget(note)
        layout.addWidget(future_frame)
        layout.addStretch(1)

        root.addWidget(make_scroll(content))

        self.combo_backend.currentTextChanged.connect(self._on_backend_text)
        self.btn_load.clicked.connect(self._on_load)
        self.btn_unload.clicked.connect(self.unloadRequested)
        self.btn_recheck.clicked.connect(self.refresh_availability)

        self.refresh_checkpoints()
        self.refresh_availability()

    # ------------------------------------------------------------------
    def _on_backend_text(self, text: str) -> None:
        try:
            backend = BackendKind(text)
        except ValueError:
            return
        self.refresh_checkpoints()
        self.backendChanged.emit(backend)

    def _on_load(self) -> None:
        try:
            backend = BackendKind(self.combo_backend.currentText())
        except ValueError:
            return
        self.loadRequested.emit(backend, self.combo_checkpoint.currentData())

    # ------------------------------------------------------------------
    def selected_backend(self) -> BackendKind:
        return BackendKind(self.combo_backend.currentText())

    def set_backend(self, backend: BackendKind) -> None:
        self.combo_backend.blockSignals(True)
        self.combo_backend.setCurrentText(backend.value)
        self.combo_backend.blockSignals(False)
        self.refresh_checkpoints()

    def refresh_checkpoints(self) -> None:
        """Populate checkpoints for the selected backend, from what is installed."""
        self.combo_checkpoint.blockSignals(True)
        self.combo_checkpoint.clear()
        try:
            backend = BackendKind(self.combo_backend.currentText())
        except ValueError:
            backend = BackendKind.RT_DETR
        if backend is BackendKind.RT_DETR:
            for checkpoint, label in RTDETRDetector.CHECKPOINTS:
                self.combo_checkpoint.addItem(label, checkpoint)
        else:
            variants = RFDETRDetector.available_variants()
            if variants:
                for variant in variants:
                    self.combo_checkpoint.addItem(variant, variant)
            else:
                self.combo_checkpoint.addItem("rfdetr not installed", None)
        self.combo_checkpoint.blockSignals(False)

    def refresh_availability(self) -> None:
        rt_ok, rt_reason = RTDETRDetector.dependency_status()
        rf_ok, rf_reason = RFDETRDetector.dependency_status()
        rows: List[Tuple[str, bool, str, str]] = [
            ("RT-DETR", rt_ok, DEP_TRANSFORMERS.version if rt_ok else "-",
             rt_reason if not rt_ok else "transformers RTDetrForObjectDetection"),
            ("RF-DETR", rf_ok, DEP_RFDETR.version if rf_ok else "-",
             rf_reason if not rf_ok else
             "variants: " + ", ".join(RFDETRDetector.available_variants())),
            ("Simulation ground truth", True, "built-in",
             "Always available - no weights required"),
        ]
        self.table.setRowCount(len(rows))
        for row, (name, ok, version, detail) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            status = QTableWidgetItem("AVAILABLE" if ok else "MISSING")
            status.setForeground(QBrush(qcolor(Palette.OK if ok else Palette.CRIT)))
            self.table.setItem(row, 1, status)
            self.table.setItem(row, 2, QTableWidgetItem(version))
            self.table.setItem(row, 3, QTableWidgetItem(detail))

    def set_model_state(
        self, state: ModelState, message: str, backend: BackendKind
    ) -> None:
        self.label_state.setText(f"{backend.value}: {state.value}")
        self.label_detail.setText(message)
        self.refresh_availability()
