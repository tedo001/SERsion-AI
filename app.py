#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOFOP Sentinel - Edge Vision Intelligence Platform
==================================================

A single-file, production-grade PyQt6 desktop application for edge computer-vision
surveillance in two deployment scenarios:

    * CLASSROOM  - school / classroom safety monitoring
    * FACTORY    - industrial / workplace safety monitoring

The application performs OBJECT DETECTION and OBJECT TRACKING only.  It does NOT
perform facial recognition, biometric identification, or any form of identity
inference, and it never stores biometric data.

Detection backends
------------------
    * RT-DETR  via HuggingFace ``transformers``  (RTDetrForObjectDetection)
    * RF-DETR  via Roboflow ``rfdetr``           (RFDETRNano/Small/Medium/Large)

Both backends are OPTIONAL.  The application is fully usable without them through
the built-in synthetic CCTV Simulation source, which ships ground-truth synthetic
detections so that tracking, zones, events, analytics and the dashboard can all be
demonstrated with zero model downloads.

Architecture (single file, modular sections - see the SECTION banners below)
---------------------------------------------------------------------------
     1. Imports & optional dependency probing
     2. Configuration / theme
     3. Data models
     4. Detector interfaces
     5. RT-DETR detector
     6. RF-DETR detector
     7. Kalman tracker
     8. Video sources
     9. Vision pipeline
    10. Event engine
    11. Analytics
    12. Database
    13. Worker threads & performance monitor
    14. Custom Qt widgets
    15. Dashboard
    16. Main window
    17. Application startup

Threading model (strict producer / consumer, GUI thread never blocks)
---------------------------------------------------------------------
    CaptureWorker (QThread)  ->  bounded FrameQueue (drops stale frames)
                             ->  InferenceWorker (QThread): detect -> track ->
                                 zones -> events -> analytics
                             ->  LatestSlot  ->  GUI repaint timer (throttled)

Author : LOFOP Engineering
License: MIT for this application source.  Model weights and model packages remain
         under their own upstream licenses (RT-DETR: Apache-2.0 via transformers /
         PekingU checkpoints;  RF-DETR: Apache-2.0 via Roboflow).  Users are
         responsible for complying with those licenses and with all applicable
         privacy / surveillance regulations in their jurisdiction.
"""

from __future__ import annotations

# =============================================================================
# SECTION 1 - IMPORTS & OPTIONAL DEPENDENCY PROBING
# =============================================================================

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import platform
import queue
import random
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

# -- Mandatory third-party dependencies ---------------------------------------
# NumPy and OpenCV are hard requirements; without them nothing (not even the
# simulation renderer) can run, so fail loudly but readably.
try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - fatal environment error
    sys.stderr.write("FATAL: NumPy is required.  Install with: pip install numpy\n")
    raise SystemExit(2) from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover - fatal environment error
    sys.stderr.write(
        "FATAL: OpenCV is required.  Install with: pip install opencv-python\n"
    )
    raise SystemExit(2) from exc


# -- Optional dependency probe -------------------------------------------------
class DependencyProbe:
    """Import a module lazily and record availability + version + error.

    Every optional dependency in the application goes through this class so the
    startup dependency screen can report an accurate, non-fabricated status.
    """

    __slots__ = ("name", "module_name", "install_hint", "purpose", "_module",
                 "_checked", "_error")

    def __init__(self, name: str, module_name: str, install_hint: str, purpose: str) -> None:
        self.name = name
        self.module_name = module_name
        self.install_hint = install_hint
        self.purpose = purpose
        self._module: Optional[Any] = None
        self._checked: bool = False
        self._error: Optional[str] = None

    def _probe(self) -> None:
        if self._checked:
            return
        self._checked = True
        try:
            import importlib

            self._module = importlib.import_module(self.module_name)
        except BaseException as exc:  # noqa: BLE001 - some libs raise non-Exception
            self._module = None
            self._error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        self._probe()
        return self._module is not None

    @property
    def module(self) -> Optional[Any]:
        self._probe()
        return self._module

    @property
    def error(self) -> Optional[str]:
        self._probe()
        return self._error

    @property
    def version(self) -> str:
        self._probe()
        if self._module is None:
            return "-"
        for attr in ("__version__", "VERSION", "version"):
            value = getattr(self._module, attr, None)
            if isinstance(value, str):
                return value
        return "installed"


DEP_TORCH = DependencyProbe(
    "PyTorch", "torch", "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    "Neural network runtime for both detection backends.")
DEP_SUPERVISION = DependencyProbe(
    "Supervision", "supervision", "pip install supervision",
    "Detection container, annotators, zone / line-crossing utilities.")
DEP_TRANSFORMERS = DependencyProbe(
    "Transformers (RT-DETR)", "transformers", "pip install transformers",
    "Provides RTDetrForObjectDetection + RTDetrImageProcessor.")
DEP_RFDETR = DependencyProbe(
    "RF-DETR", "rfdetr", "pip install rfdetr",
    "Roboflow RF-DETR real-time detection transformer.")
DEP_PILLOW = DependencyProbe(
    "Pillow", "PIL", "pip install pillow",
    "Image container accepted by both model backends.")
DEP_PSUTIL = DependencyProbe(
    "psutil", "psutil", "pip install psutil",
    "CPU / RAM telemetry for the edge system panel.")
DEP_SCIPY = DependencyProbe(
    "SciPy", "scipy", "pip install scipy",
    "Optimal (Hungarian) track association; greedy fallback used when absent.")

OPTIONAL_DEPENDENCIES: Tuple[DependencyProbe, ...] = (
    DEP_TORCH, DEP_SUPERVISION, DEP_TRANSFORMERS, DEP_RFDETR,
    DEP_PILLOW, DEP_PSUTIL, DEP_SCIPY,
)

# -- PyQt6 (mandatory for the GUI, optional for --self-test) -------------------
try:
    from PyQt6.QtCore import (
        QObject, QPointF, QRectF, QSettings, Qt, QThread, QTimer,
        pyqtSignal, pyqtSlot,
    )
    from PyQt6.QtGui import (
        QBrush, QColor, QCursor, QImage, QLinearGradient, QPainter,
        QPainterPath, QPalette, QPen, QPolygonF,
    )
    from PyQt6.QtWidgets import (
        QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
        QDockWidget, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
        QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
        QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea,
        QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget, QStatusBar,
        QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    )

    QT_AVAILABLE = True
    QT_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - handled by main()
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = str(exc)


# =============================================================================
# SECTION 2 - CONFIGURATION / THEME
# =============================================================================

APP_NAME = "LOFOP Sentinel"
APP_SUBTITLE = "Edge Vision Intelligence Platform"
APP_VERSION = "1.0.0"
ORG_NAME = "LOFOP"

APP_DIR = Path(os.environ.get("LOFOP_HOME", Path.home() / ".lofop_sentinel"))
DB_PATH = APP_DIR / "sentinel.db"
LOG_PATH = APP_DIR / "sentinel.log"
ZONES_PATH = APP_DIR / "zones.json"

LOGGER = logging.getLogger("lofop")


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging to stderr plus a rotating-ish file handler."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-16s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    LOGGER.addHandler(stream)
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 4 * 1024 * 1024:
            LOG_PATH.rename(LOG_PATH.with_suffix(".log.1"))
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(fmt)
        LOGGER.addHandler(file_handler)
    except OSError as exc:  # non-fatal: logging to file is best effort
        LOGGER.warning("File logging disabled: %s", exc)


class Palette:
    """Central dark-theme colour tokens.  All UI colours come from here."""

    BG_APP = "#0A0E13"
    BG_PANEL = "#0F151D"
    BG_CARD = "#141C26"
    BG_ELEV = "#1A2430"
    BG_INPUT = "#101821"

    BORDER = "#1F2C3A"
    BORDER_STRONG = "#2B3B4D"

    TEXT = "#E4EDF6"
    TEXT_DIM = "#7E90A3"
    TEXT_FAINT = "#55687C"

    ACCENT = "#2E9BFF"
    ACCENT_DIM = "#1B5C99"
    OK = "#2ED47A"
    WARN = "#F5A524"
    CRIT = "#FF4D5E"
    VIOLET = "#A78BFA"
    CYAN = "#22D3EE"
    MAGENTA = "#F472B6"

    TRACK_COLORS: Tuple[str, ...] = (
        "#2E9BFF", "#2ED47A", "#F5A524", "#A78BFA", "#22D3EE",
        "#F472B6", "#FB923C", "#4ADE80", "#60A5FA", "#E879F9",
    )

    @staticmethod
    def track_color_bgr(track_id: int) -> Tuple[int, int, int]:
        """Deterministic per-track BGR colour so IDs keep a stable colour."""
        hex_color = Palette.TRACK_COLORS[track_id % len(Palette.TRACK_COLORS)]
        return hex_to_bgr(hex_color)


def hex_to_bgr(value: str) -> Tuple[int, int, int]:
    """'#RRGGBB' -> (B, G, R) tuple for OpenCV drawing."""
    value = value.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


# Qt style sheet.  string.Template is used because QSS braces collide with
# f-string / str.format syntax.
QSS_TEMPLATE = Template(
    """
* { outline: 0; }

QWidget {
    background-color: $BG_APP;
    color: $TEXT;
    font-family: "Inter", "SF Pro Text", "Segoe UI", "Ubuntu", "DejaVu Sans", sans-serif;
    font-size: 12px;
}

QMainWindow, QDialog { background-color: $BG_APP; }

QToolTip {
    background-color: $BG_ELEV;
    color: $TEXT;
    border: 1px solid $BORDER_STRONG;
    padding: 5px 7px;
    border-radius: 4px;
}

/* ---------- Structural frames ---------- */
QFrame#Card {
    background-color: $BG_CARD;
    border: 1px solid $BORDER;
    border-radius: 10px;
}
QFrame#Panel {
    background-color: $BG_PANEL;
    border: 1px solid $BORDER;
    border-radius: 10px;
}
QFrame#TopBar {
    background-color: $BG_PANEL;
    border: 0px;
    border-bottom: 1px solid $BORDER;
    border-radius: 0px;
}
QFrame#Sidebar {
    background-color: $BG_PANEL;
    border: 0px;
    border-right: 1px solid $BORDER;
    border-radius: 0px;
}
QFrame#Divider {
    background-color: $BORDER;
    max-height: 1px;
    border: 0px;
}
QFrame#VDivider {
    background-color: $BORDER;
    max-width: 1px;
    border: 0px;
}

/* ---------- Typography helpers ---------- */
QLabel#H1 { font-size: 20px; font-weight: 700; color: $TEXT; }
QLabel#H2 { font-size: 15px; font-weight: 600; color: $TEXT; }
QLabel#H3 { font-size: 12px; font-weight: 600; color: $TEXT; }
QLabel#Caption {
    font-size: 10px; font-weight: 600; color: $TEXT_FAINT;
    letter-spacing: 1px;
}
QLabel#Dim { color: $TEXT_DIM; }
QLabel#Brand { font-size: 16px; font-weight: 800; letter-spacing: 1px; color: $TEXT; }
QLabel#BrandSub { font-size: 10px; color: $TEXT_DIM; letter-spacing: 1px; }
QLabel#Metric { font-size: 22px; font-weight: 700; color: $TEXT; }
QLabel#MetricSmall { font-size: 16px; font-weight: 700; color: $TEXT; }
QLabel#Mono {
    font-family: "JetBrains Mono", "Cascadia Mono", "Menlo", "Consolas", monospace;
    color: $TEXT_DIM;
}

/* ---------- Buttons ---------- */
QPushButton {
    background-color: $BG_ELEV;
    color: $TEXT;
    border: 1px solid $BORDER_STRONG;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover { background-color: #22303F; border-color: #38506A; }
QPushButton:pressed { background-color: #18222E; }
QPushButton:disabled { color: $TEXT_FAINT; background-color: #121A23; border-color: $BORDER; }
QPushButton#Primary {
    background-color: $ACCENT; color: #04101C; border: 1px solid $ACCENT;
}
QPushButton#Primary:hover { background-color: #4FAEFF; }
QPushButton#Primary:pressed { background-color: #2385DE; }
QPushButton#Danger { background-color: #3A1720; color: $CRIT; border-color: #5E2530; }
QPushButton#Danger:hover { background-color: #4A1C28; }
QPushButton#Ghost { background-color: transparent; border-color: $BORDER; color: $TEXT_DIM; }
QPushButton#Ghost:hover { color: $TEXT; border-color: $BORDER_STRONG; }

/* ---------- Sidebar navigation ---------- */
QPushButton#NavButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 9px 12px;
    text-align: left;
    font-weight: 600;
    color: $TEXT_DIM;
}
QPushButton#NavButton:hover { background-color: #16202B; color: $TEXT; }
QPushButton#NavButton:checked {
    background-color: #12283E;
    color: #8CC8FF;
    border: 1px solid #1D3E5C;
}

/* ---------- Inputs ---------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit {
    background-color: $BG_INPUT;
    border: 1px solid $BORDER_STRONG;
    border-radius: 6px;
    padding: 6px 8px;
    color: $TEXT;
    selection-background-color: $ACCENT_DIM;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #3A526C; }
QComboBox:focus, QSpinBox:focus, QLineEdit:focus { border-color: $ACCENT; }
QComboBox::drop-down { border: 0px; width: 18px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid $TEXT_DIM;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: $BG_ELEV;
    border: 1px solid $BORDER_STRONG;
    selection-background-color: $ACCENT_DIM;
    color: $TEXT;
    padding: 4px;
}

QCheckBox { spacing: 8px; color: $TEXT; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border-radius: 4px;
    border: 1px solid $BORDER_STRONG;
    background-color: $BG_INPUT;
}
QCheckBox::indicator:checked { background-color: $ACCENT; border-color: $ACCENT; }
QCheckBox::indicator:hover { border-color: $ACCENT; }

/* ---------- Slider ---------- */
QSlider::groove:horizontal {
    height: 4px; background: #1C2836; border-radius: 2px;
}
QSlider::sub-page:horizontal { background: $ACCENT; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #D6E6F5; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px; border: 2px solid $BG_APP;
}
QSlider::handle:horizontal:hover { background: #FFFFFF; }

/* ---------- Tables ---------- */
QTableWidget, QTreeWidget, QListWidget {
    background-color: $BG_CARD;
    border: 1px solid $BORDER;
    border-radius: 8px;
    gridline-color: $BORDER;
    alternate-background-color: #121A24;
}
QTableWidget::item, QListWidget::item { padding: 5px 6px; border: 0px; }
QTableWidget::item:selected, QListWidget::item:selected {
    background-color: #17334D; color: $TEXT;
}
QHeaderView::section {
    background-color: $BG_PANEL;
    color: $TEXT_FAINT;
    padding: 7px 6px;
    border: 0px;
    border-bottom: 1px solid $BORDER;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
}
QTableCornerButton::section { background-color: $BG_PANEL; border: 0px; }

/* ---------- Scroll areas & bars ---------- */
QScrollArea { border: 0px; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical {
    background: transparent; width: 10px; margin: 2px 2px 2px 0px;
}
QScrollBar::handle:vertical {
    background: #263544; border-radius: 5px; min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #34465A; }
QScrollBar:horizontal {
    background: transparent; height: 10px; margin: 0px 2px 2px 2px;
}
QScrollBar::handle:horizontal {
    background: #263544; border-radius: 5px; min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: #34465A; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0px; width: 0px; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---------- Docks, splitters, status bar, tabs ---------- */
QDockWidget {
    titlebar-close-icon: none; titlebar-normal-icon: none;
    color: $TEXT_DIM; font-size: 10px; font-weight: 700;
}
QDockWidget::title {
    background: $BG_PANEL; padding: 7px 10px;
    border-bottom: 1px solid $BORDER;
}
QSplitter::handle { background-color: $BG_APP; }
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }
QStatusBar {
    background-color: $BG_PANEL;
    border-top: 1px solid $BORDER;
    color: $TEXT_DIM;
}
QStatusBar::item { border: 0px; }
QTabWidget::pane {
    border: 1px solid $BORDER; border-radius: 8px; top: -1px;
    background: $BG_CARD;
}
QTabBar::tab {
    background: transparent; color: $TEXT_DIM;
    padding: 7px 14px; border: 1px solid transparent;
    border-top-left-radius: 7px; border-top-right-radius: 7px;
    font-weight: 600;
}
QTabBar::tab:selected { color: $TEXT; background: $BG_CARD; border-color: $BORDER; }
QTabBar::tab:hover { color: $TEXT; }

QGroupBox {
    border: 1px solid $BORDER; border-radius: 9px;
    margin-top: 16px; padding-top: 10px;
    background-color: $BG_CARD;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; top: 2px;
    color: $TEXT_FAINT; font-size: 10px; font-weight: 700;
    letter-spacing: 1px;
}
"""
)


def build_stylesheet() -> str:
    """Render the QSS template with the palette tokens."""
    tokens = {
        key: getattr(Palette, key)
        for key in dir(Palette)
        if key.isupper() and isinstance(getattr(Palette, key), str)
    }
    return QSS_TEMPLATE.substitute(**tokens)


# =============================================================================
# SECTION 3 - DATA MODELS
# =============================================================================

class Severity(Enum):
    """Event severity ladder used across the event engine and the UI."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}[self.value]

    @property
    def color(self) -> str:
        return {
            "INFO": Palette.ACCENT,
            "WARNING": Palette.WARN,
            "CRITICAL": Palette.CRIT,
        }[self.value]


class EventType(Enum):
    """Catalogue of events the EventEngine can raise."""

    PERSON_ENTERED = "PERSON_ENTERED"
    PERSON_EXITED = "PERSON_EXITED"
    RESTRICTED_ZONE_ENTRY = "RESTRICTED_ZONE_ENTRY"
    LINE_CROSSED = "LINE_CROSSED"
    CROWDING = "CROWDING"
    HIGH_OCCUPANCY = "HIGH_OCCUPANCY"
    VEHICLE_DETECTED = "VEHICLE_DETECTED"
    OBJECT_LOST = "OBJECT_LOST"
    UNATTENDED_AREA = "UNATTENDED_AREA"
    PROXIMITY_WARNING = "PROXIMITY_WARNING"
    SYSTEM_WARNING = "SYSTEM_WARNING"


class AppMode(Enum):
    """Deployment scenario."""

    CLASSROOM = "Classroom"
    FACTORY = "Factory"


class SourceKind(Enum):
    SIMULATION = "Simulation"
    FILE = "Video File"
    WEBCAM = "Webcam"


class BackendKind(Enum):
    RT_DETR = "RT-DETR"
    RF_DETR = "RF-DETR"


class DevicePref(Enum):
    AUTO = "Auto"
    CPU = "CPU"
    CUDA = "CUDA"


class ModelState(Enum):
    """Lifecycle state of a detection backend, surfaced directly in the UI."""

    UNLOADED = "Unloaded"
    LOADING = "Loading"
    READY = "Ready"
    UNAVAILABLE = "Unavailable"
    ERROR = "Error"


class PipelineState(Enum):
    IDLE = "Idle"
    STARTING = "Starting"
    RUNNING = "Running"
    PAUSED = "Paused"
    STOPPED = "Stopped"
    ERROR = "Error"


class TrackPhase(Enum):
    """Kalman track lifecycle phase."""

    TENTATIVE = "Tentative"   # seen, not yet confirmed by enough hits
    CONFIRMED = "Confirmed"   # actively matched to detections
    COASTING = "Coasting"     # predicted only, detection temporarily lost
    LOST = "Lost"             # scheduled for deletion


class ZoneKind(Enum):
    """Semantic zone category; drives default colour and default severity."""

    RESTRICTED = "Restricted"
    DANGER = "Danger"
    MACHINE = "Machine"
    WORK = "Work"
    TEACHER = "Teacher"
    STUDENT = "Student"
    ENTRANCE = "Entrance"
    MONITOR = "Monitor"

    @property
    def default_color(self) -> str:
        return {
            "Restricted": Palette.CRIT,
            "Danger": "#FF7A45",
            "Machine": Palette.WARN,
            "Work": Palette.OK,
            "Teacher": Palette.VIOLET,
            "Student": Palette.ACCENT,
            "Entrance": Palette.CYAN,
            "Monitor": Palette.TEXT_DIM,
        }[self.value]

    @property
    def default_severity(self) -> Severity:
        return {
            "Restricted": Severity.CRITICAL,
            "Danger": Severity.CRITICAL,
            "Machine": Severity.WARNING,
            "Work": Severity.INFO,
            "Teacher": Severity.INFO,
            "Student": Severity.INFO,
            "Entrance": Severity.INFO,
            "Monitor": Severity.INFO,
        }[self.value]


# COCO class groupings.  Both supported backends are COCO-pretrained, so the
# application reasons about semantic groups rather than raw class ids.
PERSON_CLASS_NAMES = frozenset({"person"})
VEHICLE_CLASS_NAMES = frozenset({
    "car", "truck", "bus", "motorcycle", "bicycle", "train", "forklift",
})


@dataclass(slots=True)
class Detection:
    """Backend-agnostic detection.  Every detector normalises to this type."""

    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in frame pixels
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def anchor(self) -> Tuple[float, float]:
        """Bottom-centre point - the ground contact used for zone tests."""
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y2)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def is_person(self) -> bool:
        return self.class_name in PERSON_CLASS_NAMES

    @property
    def is_vehicle(self) -> bool:
        return self.class_name in VEHICLE_CLASS_NAMES


@dataclass(slots=True)
class TrackedObject:
    """Immutable snapshot of a Kalman track, safe to pass between threads."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    position: Tuple[float, float]       # centre in pixels
    velocity: Tuple[float, float]       # pixels / frame
    age: int                            # frames since birth
    hits: int                           # total matched detections
    time_since_update: int              # frames since last matched detection
    phase: TrackPhase
    last_seen: float                    # unix timestamp
    trail: Tuple[Tuple[float, float], ...] = ()

    @property
    def anchor(self) -> Tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y2)

    @property
    def speed(self) -> float:
        return float(math.hypot(self.velocity[0], self.velocity[1]))

    @property
    def is_person(self) -> bool:
        return self.class_name in PERSON_CLASS_NAMES

    @property
    def is_vehicle(self) -> bool:
        return self.class_name in VEHICLE_CLASS_NAMES


@dataclass
class ZoneConfig:
    """Polygon zone stored in NORMALISED coordinates (0..1).

    Normalised storage keeps zones valid across source resolutions, which matters
    because the user can switch between a 640x480 webcam and a 1080p file.
    """

    zone_id: str
    name: str
    kind: ZoneKind
    polygon: List[Tuple[float, float]]
    color: str = Palette.ACCENT
    severity: Severity = Severity.WARNING
    enabled: bool = True
    max_occupancy: int = 0            # 0 disables the crowding rule
    unattended_after_s: float = 0.0   # 0 disables the unattended-area rule

    def pixel_polygon(self, width: int, height: int) -> "np.ndarray":
        pts = [(x * width, y * height) for x, y in self.polygon]
        return np.array(pts, dtype=np.int32)

    def contains(self, point: Tuple[float, float], width: int, height: int) -> bool:
        """Point-in-polygon test in pixel space (cv2 is robust and fast here)."""
        if len(self.polygon) < 3:
            return False
        poly = self.pixel_polygon(width, height)
        return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0

    def distance_to(self, point: Tuple[float, float], width: int, height: int) -> float:
        """Signed distance in pixels (positive inside) to the polygon border."""
        if len(self.polygon) < 3:
            return -1e9
        poly = self.pixel_polygon(width, height)
        return float(cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), True))

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        data["kind"] = self.kind.value
        data["severity"] = self.severity.value
        data["polygon"] = [[float(x), float(y)] for x, y in self.polygon]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZoneConfig":
        return cls(
            zone_id=str(data.get("zone_id") or uuid.uuid4().hex[:8]),
            name=str(data.get("name", "Zone")),
            kind=ZoneKind(data.get("kind", ZoneKind.MONITOR.value)),
            polygon=[(float(p[0]), float(p[1])) for p in data.get("polygon", [])],
            color=str(data.get("color", Palette.ACCENT)),
            severity=Severity(data.get("severity", Severity.WARNING.value)),
            enabled=bool(data.get("enabled", True)),
            max_occupancy=int(data.get("max_occupancy", 0)),
            unattended_after_s=float(data.get("unattended_after_s", 0.0)),
        )


@dataclass
class LineConfig:
    """Directed counting line in normalised coordinates."""

    line_id: str
    name: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    enabled: bool = True
    color: str = Palette.CYAN

    def pixel_points(self, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (int(self.p1[0] * width), int(self.p1[1] * height)),
            (int(self.p2[0] * width), int(self.p2[1] * height)),
        )

    def side(self, point: Tuple[float, float]) -> float:
        """Signed side of the (normalised) point relative to the line."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id, "name": self.name,
            "p1": list(self.p1), "p2": list(self.p2),
            "enabled": self.enabled, "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineConfig":
        return cls(
            line_id=str(data.get("line_id") or uuid.uuid4().hex[:8]),
            name=str(data.get("name", "Line")),
            p1=(float(data["p1"][0]), float(data["p1"][1])),
            p2=(float(data["p2"][0]), float(data["p2"][1])),
            enabled=bool(data.get("enabled", True)),
            color=str(data.get("color", Palette.CYAN)),
        )


@dataclass(slots=True)
class SecurityEvent:
    """A single event record.  Mirrors the `events` SQLite table."""

    timestamp: float
    event_type: EventType
    severity: Severity
    description: str
    track_id: Optional[int] = None
    zone: Optional[str] = None
    mode: str = AppMode.CLASSROOM.value
    source: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp).isoformat(timespec="seconds")

    @property
    def clock(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

    def as_row(self) -> Tuple[Any, ...]:
        return (
            self.event_id, self.timestamp, self.iso, self.event_type.value,
            self.severity.value, self.track_id, self.zone, self.description,
            self.mode, self.source,
        )


@dataclass(slots=True)
class Timings:
    """Per-frame stage latencies in milliseconds."""

    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    detect_ms: float = 0.0
    track_ms: float = 0.0
    event_ms: float = 0.0
    annotate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (self.capture_ms + self.preprocess_ms + self.detect_ms
                + self.track_ms + self.event_ms + self.annotate_ms)


@dataclass(slots=True)
class FramePacket:
    """Unit of work produced by the capture thread."""

    frame: "np.ndarray"                 # BGR uint8
    index: int
    timestamp: float
    source_name: str
    synthetic: bool = False
    # Ground-truth detections supplied by the simulation source.  Only used when
    # no model backend is active, and always labelled as synthetic in the UI.
    synthetic_detections: Optional[List[Detection]] = None


@dataclass(slots=True)
class ZoneStatus:
    """Live per-zone state pushed to the right-hand panel."""

    zone_id: str
    name: str
    kind: ZoneKind
    color: str
    occupancy: int          # people only - what the crowding rules use
    objects: int            # every tracked object inside, vehicles included
    max_occupancy: int
    breached: bool
    enabled: bool


@dataclass(slots=True)
class FrameResult:
    """Everything the GUI needs to render one processed frame."""

    frame: "np.ndarray"                 # annotated BGR frame
    index: int
    timestamp: float
    detections: List[Detection]
    tracks: List[TrackedObject]
    events: List[SecurityEvent]
    zone_status: List[ZoneStatus]
    people: int
    vehicles: int
    others: int
    timings: Timings
    source_name: str
    detector_label: str
    synthetic_detections: bool
    line_in: int = 0
    line_out: int = 0
    occupancy: int = 0


@dataclass
class AppSettings:
    """User-tunable settings, persisted through QSettings."""

    mode: AppMode = AppMode.CLASSROOM
    backend: BackendKind = BackendKind.RT_DETR
    device: DevicePref = DevicePref.AUTO
    confidence: float = 0.35
    inference_size: int = 640
    half_precision: bool = True
    tracking_enabled: bool = True
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    target_fps: int = 25
    frame_skip: int = 0
    queue_size: int = 2
    gui_fps: int = 30

    show_boxes: bool = True
    show_track_ids: bool = True
    show_zones: bool = True
    show_confidence: bool = True
    show_fps_overlay: bool = True
    show_trails: bool = True
    use_supervision_annotators: bool = True

    crowding_threshold: int = 8
    high_occupancy_threshold: int = 15
    proximity_px: int = 60
    person_classes_only: bool = False
    detect_on_simulation: bool = False   # run a real model over simulated frames

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            data[f.name] = value.value if isinstance(value, Enum) else value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        out = cls()
        enum_map = {
            "mode": AppMode, "backend": BackendKind, "device": DevicePref,
        }
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            raw = data[f.name]
            try:
                if f.name in enum_map:
                    setattr(out, f.name, enum_map[f.name](raw))
                elif isinstance(getattr(out, f.name), bool):
                    setattr(out, f.name, bool(raw))
                elif isinstance(getattr(out, f.name), int):
                    setattr(out, f.name, int(raw))
                elif isinstance(getattr(out, f.name), float):
                    setattr(out, f.name, float(raw))
                else:
                    setattr(out, f.name, raw)
            except (ValueError, TypeError):
                LOGGER.debug("Ignoring invalid setting %s=%r", f.name, raw)
        return out


# =============================================================================
# SECTION 4 - DETECTOR INTERFACES
# =============================================================================

class DetectorError(RuntimeError):
    """Raised when a backend cannot be loaded or a prediction fails fatally."""


def resolve_torch_device(pref: DevicePref) -> str:
    """Resolve a device preference into a concrete torch device string.

    CUDA is only selected when torch reports it as actually available, so the
    application degrades to CPU instead of crashing on machines without a GPU.
    """
    torch = DEP_TORCH.module
    if torch is None:
        return "cpu"
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - broken driver installs raise all sorts
        cuda_ok = False
    if pref is DevicePref.CPU:
        return "cpu"
    if pref is DevicePref.CUDA:
        return "cuda" if cuda_ok else "cpu"
    return "cuda" if cuda_ok else "cpu"


def cuda_device_name() -> Optional[str]:
    torch = DEP_TORCH.module
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        return None
    return None


class BaseDetector(ABC):
    """Common interface every detection backend must implement.

    Contract:
        * ``load_model()``   - blocking, may raise DetectorError.
        * ``predict(frame)`` - takes a BGR uint8 ndarray, returns List[Detection]
                               in ORIGINAL frame pixel coordinates.
        * ``unload_model()`` - releases weights / GPU memory, idempotent.

    Nothing outside the detector subclasses may know which backend is running.
    """

    #: Human readable backend name shown in the UI.
    display_name: str = "Detector"

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._state: ModelState = ModelState.UNLOADED
        self._status_message: str = "Not loaded"
        self._device: str = "cpu"
        self._lock = threading.RLock()

    # -- state ---------------------------------------------------------------
    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._state is ModelState.READY

    def _set_state(self, state: ModelState, message: str) -> None:
        self._state = state
        self._status_message = message
        LOGGER.info("[%s] %s -> %s", self.display_name, state.value, message)

    # -- capability probing ---------------------------------------------------
    @staticmethod
    @abstractmethod
    def dependency_status() -> Tuple[bool, str]:
        """Return (available, human readable reason / install hint)."""

    # -- lifecycle -----------------------------------------------------------
    @abstractmethod
    def load_model(self) -> None:
        """Load weights.  Must set state to READY or raise DetectorError."""

    @abstractmethod
    def predict(self, frame: "np.ndarray") -> List[Detection]:
        """Run inference on one BGR frame."""

    @abstractmethod
    def unload_model(self) -> None:
        """Free all model resources.  Must be safe to call repeatedly."""

    # -- helpers shared by concrete backends ----------------------------------
    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    def _letterbox_scale(self, frame: "np.ndarray") -> Tuple["np.ndarray", float]:
        """Resize the long edge to the configured inference size.

        Returns the resized frame and the scale factor needed to map boxes back
        to the original resolution.  Aspect ratio is preserved (no letterbox
        padding is required because both backends accept arbitrary sizes).
        """
        target = int(self._settings.inference_size)
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= 0:
            return frame, 1.0
        scale = target / float(longest)
        if abs(scale - 1.0) < 0.02:
            return frame, 1.0
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        # Recompute the exact scale from the rounded size for correct rescaling.
        return resized, w / float(new_w)

    @staticmethod
    def _clip_box(
        box: Sequence[float], width: int, height: int
    ) -> Optional[Tuple[float, float, float, float]]:
        x1 = max(0.0, min(float(box[0]), width - 1.0))
        y1 = max(0.0, min(float(box[1]), height - 1.0))
        x2 = max(0.0, min(float(box[2]), width - 1.0))
        y2 = max(0.0, min(float(box[3]), height - 1.0))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            return None
        return (x1, y1, x2, y2)

    def _filter(self, detections: List[Detection]) -> List[Detection]:
        """Apply the person-only UI filter, if enabled."""
        if not self._settings.person_classes_only:
            return detections
        return [d for d in detections if d.is_person]


class NullDetector(BaseDetector):
    """Placeholder used when no backend is loaded.

    Keeps the pipeline running (and the simulation fully functional) instead of
    forcing the application into an error state.
    """

    display_name = "No model"

    @staticmethod
    def dependency_status() -> Tuple[bool, str]:
        return True, "Always available"

    def load_model(self) -> None:
        self._set_state(ModelState.READY, "No detection backend active")

    def predict(self, frame: "np.ndarray") -> List[Detection]:
        return []

    def unload_model(self) -> None:
        self._set_state(ModelState.UNLOADED, "Not loaded")


# =============================================================================
# SECTION 5 - RT-DETR DETECTOR
# =============================================================================

class RTDETRDetector(BaseDetector):
    """RT-DETR backend backed by HuggingFace ``transformers``.

    Uses the public, documented API:

        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        processor = RTDetrImageProcessor.from_pretrained(checkpoint)
        model     = RTDetrForObjectDetection.from_pretrained(checkpoint)
        inputs    = processor(images=image, return_tensors="pt")
        outputs   = model(**inputs)
        results   = processor.post_process_object_detection(
                        outputs, target_sizes=..., threshold=...)

    RT-DETRv2 checkpoints are supported through the same processor/head API and
    are selected automatically when the installed transformers build exposes
    ``RTDetrV2ForObjectDetection``.
    """

    display_name = "RT-DETR"

    #: (checkpoint, label) pairs, smallest / fastest first.
    CHECKPOINTS: Tuple[Tuple[str, str], ...] = (
        ("PekingU/rtdetr_r18vd", "RT-DETR R18 (fast)"),
        ("PekingU/rtdetr_r50vd", "RT-DETR R50 (accurate)"),
        ("PekingU/rtdetr_v2_r18vd", "RT-DETRv2 R18 (fast)"),
        ("PekingU/rtdetr_v2_r50vd", "RT-DETRv2 R50 (accurate)"),
    )

    def __init__(self, settings: AppSettings, checkpoint: str = "PekingU/rtdetr_r18vd") -> None:
        super().__init__(settings)
        self.checkpoint = checkpoint
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._id2label: Dict[int, str] = {}
        self._torch: Optional[Any] = None
        self._use_half: bool = False

    @staticmethod
    def dependency_status() -> Tuple[bool, str]:
        if not DEP_TORCH.available:
            return False, f"PyTorch missing. {DEP_TORCH.install_hint}"
        if not DEP_TRANSFORMERS.available:
            return False, f"transformers missing. {DEP_TRANSFORMERS.install_hint}"
        transformers = DEP_TRANSFORMERS.module
        assert transformers is not None
        if not hasattr(transformers, "RTDetrForObjectDetection"):
            return False, (
                "Installed transformers build has no RTDetrForObjectDetection. "
                "Upgrade with: pip install -U transformers"
            )
        return True, f"transformers {DEP_TRANSFORMERS.version}"

    def load_model(self) -> None:
        with self._lock:
            ok, reason = self.dependency_status()
            if not ok:
                self._set_state(ModelState.UNAVAILABLE, reason)
                raise DetectorError(reason)

            self._set_state(ModelState.LOADING, f"Loading {self.checkpoint} ...")
            transformers = DEP_TRANSFORMERS.module
            torch = DEP_TORCH.module
            assert transformers is not None and torch is not None
            self._torch = torch

            try:
                # v2 checkpoints need the v2 head class when it exists.
                is_v2 = "_v2_" in self.checkpoint
                model_cls = getattr(transformers, "RTDetrV2ForObjectDetection", None) \
                    if is_v2 else None
                if model_cls is None:
                    model_cls = transformers.RTDetrForObjectDetection
                processor_cls = getattr(transformers, "RTDetrImageProcessor", None) \
                    or transformers.AutoImageProcessor

                self._processor = processor_cls.from_pretrained(self.checkpoint)
                model = model_cls.from_pretrained(self.checkpoint)
            except Exception as exc:  # noqa: BLE001 - network / hub / disk errors
                message = self._explain_load_failure(exc)
                self._set_state(ModelState.ERROR, message)
                raise DetectorError(message) from exc

            self._device = resolve_torch_device(self._settings.device)
            try:
                model = model.to(self._device)
                model.eval()
                # Half precision only makes sense (and is only numerically safe
                # in this codebase) on CUDA.
                self._use_half = bool(
                    self._settings.half_precision and self._device == "cuda"
                )
                if self._use_half:
                    model = model.half()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Falling back to CPU/full precision: %s", exc)
                self._device = "cpu"
                self._use_half = False
                model = model.float().to("cpu")

            self._model = model
            raw_labels = getattr(getattr(model, "config", None), "id2label", None) or {}
            self._id2label = {int(k): str(v) for k, v in raw_labels.items()}
            self._set_state(
                ModelState.READY,
                f"{self.checkpoint} on {self._device.upper()}"
                + (" (fp16)" if self._use_half else ""),
            )

    @staticmethod
    def _explain_load_failure(exc: BaseException) -> str:
        text = str(exc).lower()
        # The hub reports an unreachable network as a plain OSError about not
        # finding the repo, so match those phrasings too rather than only the
        # obvious connection errors.
        if any(k in text for k in (
            "connection", "offline", "resolve", "timed out", "network", "proxy",
            "max retries", "huggingface.co", "can't load", "couldn't connect",
            "is not a local folder", "not a valid model identifier",
        )):
            return ("Could not download RT-DETR weights - the HuggingFace hub is "
                    "unreachable (no network, a proxy, or offline mode). "
                    "Pre-download once on a connected machine with: "
                    "huggingface-cli download PekingU/rtdetr_r18vd")
        if "no space" in text or "disk" in text:
            return "Not enough disk space to cache the RT-DETR weights."
        return f"RT-DETR load failed: {type(exc).__name__}: {exc}"

    def predict(self, frame: "np.ndarray") -> List[Detection]:
        with self._lock:
            if self._model is None or self._processor is None or self._torch is None:
                return []
            torch = self._torch
            height, width = frame.shape[:2]

            small, _scale = self._letterbox_scale(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            try:
                inputs = self._processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                if self._use_half:
                    inputs = {
                        k: (v.half() if v.dtype == torch.float32 else v)
                        for k, v in inputs.items()
                    }
                with torch.no_grad():
                    outputs = self._model(**inputs)
                # Post-process straight to ORIGINAL frame size: target_sizes is
                # (height, width), so no manual rescaling is needed.
                target_sizes = torch.tensor([(height, width)], device=self._device)
                results = self._processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=float(self._settings.confidence),
                )
            except Exception as exc:  # noqa: BLE001 - never kill the pipeline
                LOGGER.error("RT-DETR inference error: %s", exc)
                raise DetectorError(f"RT-DETR inference failed: {exc}") from exc

            if not results:
                return []
            result = results[0]
            detections: List[Detection] = []
            scores = result.get("scores")
            labels = result.get("labels")
            boxes = result.get("boxes")
            if scores is None or labels is None or boxes is None:
                return []
            for score, label, box in zip(
                scores.detach().cpu().tolist(),
                labels.detach().cpu().tolist(),
                boxes.detach().cpu().tolist(),
            ):
                clipped = self._clip_box(box, width, height)
                if clipped is None:
                    continue
                class_id = int(label)
                detections.append(
                    Detection(
                        bbox=clipped,
                        confidence=float(score),
                        class_id=class_id,
                        class_name=self._id2label.get(class_id, f"class_{class_id}"),
                    )
                )
            return self._filter(detections)

    def unload_model(self) -> None:
        with self._lock:
            model, self._model = self._model, None
            self._processor = None
            self._id2label = {}
            torch = self._torch
            del model
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            self._set_state(ModelState.UNLOADED, "Not loaded")


# =============================================================================
# SECTION 6 - RF-DETR DETECTOR
# =============================================================================

class RFDETRDetector(BaseDetector):
    """RF-DETR backend backed by the Roboflow ``rfdetr`` package.

    Uses the documented API:

        from rfdetr import RFDETRMedium
        from rfdetr.assets.coco_classes import COCO_CLASSES
        model = RFDETRMedium()
        detections = model.predict(image, threshold=0.5)   # -> sv.Detections

    The package has renamed its size classes across releases (``RFDETRBase`` in
    early versions, ``RFDETRNano/Small/Medium/Large`` later) and moved
    ``COCO_CLASSES`` between ``rfdetr.util`` and ``rfdetr.assets``.  All of that
    variability is absorbed here so the rest of the application never sees it.
    """

    display_name = "RF-DETR"

    #: Preferred variant order, smallest first.  Resolved against what the
    #: installed package actually exposes - never assumed.
    VARIANT_PREFERENCE: Tuple[str, ...] = (
        "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge",
    )

    def __init__(self, settings: AppSettings, variant: str = "auto") -> None:
        super().__init__(settings)
        self.variant = variant
        self._model: Optional[Any] = None
        self._class_names: Dict[int, str] = {}
        self._pil: Optional[Any] = None

    # -- introspection helpers -------------------------------------------------
    @staticmethod
    def available_variants() -> List[str]:
        """Variant class names actually exposed by the installed package."""
        module = DEP_RFDETR.module
        if module is None:
            return []
        found = [
            name for name in RFDETRDetector.VARIANT_PREFERENCE
            if hasattr(module, name)
        ]
        if found:
            return found
        # Fall back to discovering any RFDETR* detection class.
        return [
            name for name in dir(module)
            if name.startswith("RFDETR") and "Seg" not in name
            and "Keypoint" not in name and isinstance(getattr(module, name), type)
        ]

    @staticmethod
    def dependency_status() -> Tuple[bool, str]:
        if not DEP_TORCH.available:
            return False, f"PyTorch missing. {DEP_TORCH.install_hint}"
        if not DEP_RFDETR.available:
            err = DEP_RFDETR.error or "not installed"
            return False, (
                f"rfdetr not importable ({err}). Install with: pip install rfdetr"
            )
        if not RFDETRDetector.available_variants():
            return False, (
                "The installed rfdetr package exposes no RFDETR* model class. "
                "Upgrade with: pip install -U rfdetr"
            )
        return True, f"rfdetr {DEP_RFDETR.version}"

    @staticmethod
    def _load_coco_classes() -> Dict[int, str]:
        """Locate COCO_CLASSES across the package layouts rfdetr has shipped."""
        import importlib

        for path in ("rfdetr.assets.coco_classes", "rfdetr.util.coco_classes"):
            try:
                module = importlib.import_module(path)
            except Exception:  # noqa: BLE001
                continue
            table = getattr(module, "COCO_CLASSES", None)
            if isinstance(table, dict):
                return {int(k): str(v) for k, v in table.items()}
            if isinstance(table, (list, tuple)):
                return {i: str(v) for i, v in enumerate(table)}
        return {}

    def load_model(self) -> None:
        with self._lock:
            ok, reason = self.dependency_status()
            if not ok:
                self._set_state(ModelState.UNAVAILABLE, reason)
                raise DetectorError(reason)

            module = DEP_RFDETR.module
            assert module is not None
            variants = self.available_variants()
            name = self.variant if self.variant in variants else variants[0]
            self._set_state(ModelState.LOADING, f"Loading {name} ...")

            self._device = resolve_torch_device(self._settings.device)
            model_cls = getattr(module, name)

            # Constructor kwargs differ between releases; try the richest form
            # first and degrade gracefully rather than guessing a single API.
            attempts: List[Dict[str, Any]] = [
                {"device": self._device, "resolution": self._rf_resolution()},
                {"device": self._device},
                {},
            ]
            last_error: Optional[BaseException] = None
            model = None
            for kwargs in attempts:
                try:
                    model = model_cls(**kwargs)
                    break
                except TypeError as exc:
                    last_error = exc           # unsupported kwarg -> try simpler
                    continue
                except Exception as exc:  # noqa: BLE001 - weights download etc.
                    last_error = exc
                    break

            if model is None:
                message = self._explain_load_failure(last_error)
                self._set_state(ModelState.ERROR, message)
                raise DetectorError(message)

            # Optional inference optimisation offered by some releases.
            optimize = getattr(model, "optimize_for_inference", None)
            if callable(optimize):
                try:
                    optimize()
                    LOGGER.info("RF-DETR: optimize_for_inference() applied")
                except Exception as exc:  # noqa: BLE001 - purely opportunistic
                    LOGGER.debug("RF-DETR optimize_for_inference skipped: %s", exc)

            self._model = model
            self._class_names = self._load_coco_classes()
            self._pil = DEP_PILLOW.module
            self._set_state(
                ModelState.READY, f"{name} on {self._device.upper()}"
            )

    #: RF-DETR resolutions must satisfy two separate divisibility constraints:
    #: the DINOv2 patch grid (56) and, on the windowed Nano/Small variants, the
    #: window block size (patch_size 16 * num_windows 2 = 32).  Only multiples of
    #: their LCM (224) satisfy both, and a value that violates the second one is
    #: accepted by the constructor but fails later inside predict().
    RESOLUTION_STEP: int = 224

    def _rf_resolution(self) -> int:
        """Snap the configured inference size to a resolution RF-DETR accepts."""
        size = int(self._settings.inference_size)
        steps = max(1, int(round(size / float(self.RESOLUTION_STEP))))
        return steps * self.RESOLUTION_STEP

    @staticmethod
    def _explain_load_failure(exc: Optional[BaseException]) -> str:
        if exc is None:
            return "RF-DETR model could not be constructed."
        text = str(exc).lower()
        if any(k in text for k in ("connection", "offline", "resolve", "timed out",
                                   "network", "proxy", "max retries", "download")):
            return ("Could not download RF-DETR weights (no network access). "
                    "Run once with connectivity to populate the local cache.")
        return f"RF-DETR load failed: {type(exc).__name__}: {exc}"

    def predict(self, frame: "np.ndarray") -> List[Detection]:
        with self._lock:
            if self._model is None:
                return []
            height, width = frame.shape[:2]
            small, scale = self._letterbox_scale(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # A PIL image is the input form accepted by every rfdetr release;
            # fall back to the raw ndarray when Pillow is unavailable.
            image: Any = rgb
            if self._pil is not None:
                try:
                    image = self._pil.Image.fromarray(rgb)
                except Exception:  # noqa: BLE001
                    image = rgb

            try:
                raw = self._model.predict(
                    image, threshold=float(self._settings.confidence)
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("RF-DETR inference error: %s", exc)
                raise DetectorError(f"RF-DETR inference failed: {exc}") from exc

            # predict() may return a list when given a batch; take the first.
            if isinstance(raw, (list, tuple)):
                if not raw:
                    return []
                raw = raw[0]
            return self._filter(self._from_sv(raw, scale, width, height))

    def _from_sv(
        self, sv_detections: Any, scale: float, width: int, height: int
    ) -> List[Detection]:
        """Normalise a ``supervision.Detections`` object into our Detection list."""
        xyxy = getattr(sv_detections, "xyxy", None)
        if xyxy is None or len(xyxy) == 0:
            return []
        confidence = getattr(sv_detections, "confidence", None)
        class_id = getattr(sv_detections, "class_id", None)
        data = getattr(sv_detections, "data", None) or {}
        names = data.get("class_name") if isinstance(data, dict) else None

        out: List[Detection] = []
        for i in range(len(xyxy)):
            box = [float(v) * scale for v in xyxy[i]]
            clipped = self._clip_box(box, width, height)
            if clipped is None:
                continue
            cid = int(class_id[i]) if class_id is not None else -1
            if names is not None and i < len(names):
                cname = str(names[i])
            else:
                cname = self._class_names.get(cid, f"class_{cid}")
            conf = float(confidence[i]) if confidence is not None else 1.0
            out.append(
                Detection(bbox=clipped, confidence=conf, class_id=cid, class_name=cname)
            )
        return out

    def unload_model(self) -> None:
        with self._lock:
            model, self._model = self._model, None
            del model
            torch = DEP_TORCH.module
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            self._set_state(ModelState.UNLOADED, "Not loaded")


def create_detector(
    backend: BackendKind, settings: AppSettings, variant: Optional[str] = None
) -> BaseDetector:
    """Factory used by the inference worker; the only place backends are named.

    ``variant`` is the backend-specific checkpoint (an RT-DETR hub id) or model
    class name (an RF-DETR size class); None selects that backend's default.
    """
    if backend is BackendKind.RT_DETR:
        return (RTDETRDetector(settings, variant) if variant
                else RTDETRDetector(settings))
    if backend is BackendKind.RF_DETR:
        return RFDETRDetector(settings, variant or "auto")
    return NullDetector(settings)


# =============================================================================
# SECTION 7 - KALMAN TRACKER
# =============================================================================

def iou_matrix(boxes_a: "np.ndarray", boxes_b: "np.ndarray") -> "np.ndarray":
    """Vectorised IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]   # (N, 1, 4)
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]   # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area_a = np.clip(a[..., 2] - a[..., 0], 0.0, None) * np.clip(a[..., 3] - a[..., 1], 0.0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0.0, None) * np.clip(b[..., 3] - b[..., 1], 0.0, None)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-6), 0.0).astype(np.float32)


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over the bounding-box state

        x = [cx, cy, s, r, vx, vy, vs]^T

    where ``s`` is box area and ``r`` is the aspect ratio (width / height).
    This is the classic SORT parameterisation, implemented directly on NumPy so
    the application carries no extra filtering dependency.
    """

    __slots__ = ("x", "P", "F", "H", "Q", "R")

    def __init__(self, bbox: Tuple[float, float, float, float]) -> None:
        # State transition: position integrates velocity once per frame.
        self.F = np.eye(7, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement model observes (cx, cy, s, r) directly.
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Measurement noise: area and aspect are noisier than centre position.
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        # Process noise + initial covariance: velocities start highly uncertain.
        self.P = np.eye(7, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.Q = np.eye(7, dtype=np.float64)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4, 0] = self._to_z(bbox)

    # -- conversions ---------------------------------------------------------
    @staticmethod
    def _to_z(bbox: Sequence[float]) -> "np.ndarray":
        x1, y1, x2, y2 = (float(v) for v in bbox)
        w = max(1e-3, x2 - x1)
        h = max(1e-3, y2 - y1)
        return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h], dtype=np.float64)

    @staticmethod
    def _to_bbox(state: "np.ndarray") -> Tuple[float, float, float, float]:
        cx, cy, s, r = (float(state[i, 0]) for i in range(4))
        s = max(s, 1e-3)
        r = max(r, 1e-3)
        w = math.sqrt(s * r)
        h = s / w if w > 1e-6 else 1e-3
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    # -- filter steps --------------------------------------------------------
    def predict(self) -> Tuple[float, float, float, float]:
        # Guard against the area going negative under a strong negative vs.
        if self.x[6, 0] + self.x[2, 0] <= 0.0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox(self.x)

    def update(self, bbox: Sequence[float]) -> None:
        z = self._to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:          # singular innovation covariance
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        identity = np.eye(7, dtype=np.float64)
        self.P = (identity - K @ self.H) @ self.P

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._to_bbox(self.x)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.x[4, 0]), float(self.x[5, 0]))


class Track:
    """A single tracked object: Kalman state + identity + lifecycle bookkeeping."""

    __slots__ = ("track_id", "kf", "class_id", "class_name", "confidence",
                 "age", "hits", "hit_streak", "time_since_update", "phase",
                 "last_seen", "trail", "created_at", "_class_votes")

    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.kf = KalmanBoxFilter(detection.bbox)
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.phase = TrackPhase.TENTATIVE
        self.last_seen = time.time()
        self.created_at = self.last_seen
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=48)
        # Majority vote over recent class labels keeps the label stable when the
        # detector flickers between e.g. "person" and "backpack".
        self._class_votes: Counter = Counter({detection.class_name: 1})
        self.trail.append(detection.center)

    def predict(self) -> Tuple[float, float, float, float]:
        bbox = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return bbox

    def update(self, detection: Detection) -> None:
        self.kf.update(detection.bbox)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = detection.confidence
        self.last_seen = time.time()
        self._class_votes[detection.class_name] += 1
        self.class_name = self._class_votes.most_common(1)[0][0]
        if detection.class_name == self.class_name:
            self.class_id = detection.class_id
        bbox = self.kf.bbox
        self.trail.append(((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5))

    def refresh_phase(self, min_hits: int) -> None:
        if self.time_since_update == 0:
            self.phase = (TrackPhase.CONFIRMED if self.hits >= min_hits
                          else TrackPhase.TENTATIVE)
        elif self.hits >= min_hits:
            self.phase = TrackPhase.COASTING
        else:
            self.phase = TrackPhase.TENTATIVE

    def snapshot(self) -> TrackedObject:
        bbox = self.kf.bbox
        vx, vy = self.kf.velocity
        return TrackedObject(
            track_id=self.track_id,
            bbox=bbox,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            position=((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5),
            velocity=(vx, vy),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            phase=self.phase,
            last_seen=self.last_seen,
            trail=tuple(self.trail),
        )


class KalmanTracker:
    """Multi-object tracker: Kalman prediction + IoU / distance association.

    Association strategy:
        1. Predict every existing track forward one frame.
        2. Build an IoU cost matrix against the new detections.
        3. Solve it optimally with SciPy's Hungarian algorithm when available,
           otherwise greedily by descending IoU (both give stable IDs).
        4. Reject matches below ``iou_threshold``.
        5. Unmatched detections spawn tentative tracks; unmatched tracks coast on
           Kalman prediction until ``max_age`` frames pass.

    IDs are assigned once at birth and never reshuffled, so a track keeps its
    identity for as long as it survives.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[Track] = []
        self._next_id: int = 1
        self._lost: List[TrackedObject] = []

    # -- configuration --------------------------------------------------------
    def configure(self, max_age: int, min_hits: int, iou_threshold: float) -> None:
        self.max_age = max(1, int(max_age))
        self.min_hits = max(1, int(min_hits))
        self.iou_threshold = float(np.clip(iou_threshold, 0.01, 0.95))

    def reset(self) -> None:
        self._tracks.clear()
        self._lost.clear()
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks if t.phase is not TrackPhase.TENTATIVE)

    # -- association ----------------------------------------------------------
    def _associate(
        self, detections: List[Detection], predicted: "np.ndarray"
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Return (matches as (track_idx, det_idx), unmatched_tracks, unmatched_dets)."""
        n_tracks = len(predicted)
        n_dets = len(detections)
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        ious = iou_matrix(predicted, det_boxes)

        # Class-aware gating: matching a person track to a truck detection is
        # almost always an association error, so penalise cross-class pairs.
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                if track.class_name != det.class_name:
                    ious[ti, di] *= 0.5

        matches: List[Tuple[int, int]] = []
        if DEP_SCIPY.available:
            try:
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(-ious)
                matches = [(int(r), int(c)) for r, c in zip(rows, cols)]
            except Exception as exc:  # noqa: BLE001 - fall back to greedy
                LOGGER.debug("SciPy assignment failed (%s); using greedy", exc)
                matches = []
        if not matches:
            matches = self._greedy_match(ious)

        final: List[Tuple[int, int]] = []
        matched_tracks = set()
        matched_dets = set()
        for ti, di in matches:
            if ious[ti, di] < self.iou_threshold:
                continue
            final.append((ti, di))
            matched_tracks.add(ti)
            matched_dets.add(di)

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        return final, unmatched_tracks, unmatched_dets

    @staticmethod
    def _greedy_match(ious: "np.ndarray") -> List[Tuple[int, int]]:
        """Descending-IoU greedy matching - the dependency-free fallback."""
        pairs: List[Tuple[int, int]] = []
        used_rows: set = set()
        used_cols: set = set()
        flat = np.argsort(ious, axis=None)[::-1]
        n_cols = ious.shape[1]
        for index in flat:
            r, c = divmod(int(index), n_cols)
            if ious[r, c] <= 0.0:
                break
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
        return pairs

    # -- main step ------------------------------------------------------------
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Advance the tracker by one frame and return the current snapshots."""
        self._lost = []

        # 1. Predict.  Tracks whose filter diverges into NaN are dropped.
        predicted: List[Tuple[float, float, float, float]] = []
        alive: List[Track] = []
        for track in self._tracks:
            bbox = track.predict()
            if not all(math.isfinite(v) for v in bbox):
                LOGGER.debug("Dropping diverged track %s", track.track_id)
                continue
            predicted.append(bbox)
            alive.append(track)
        self._tracks = alive
        predicted_array = (np.array(predicted, dtype=np.float32)
                           if predicted else np.zeros((0, 4), dtype=np.float32))

        # 2/3. Associate and update.
        matches, _unmatched_tracks, unmatched_dets = self._associate(
            detections, predicted_array
        )
        for track_idx, det_idx in matches:
            self._tracks[track_idx].update(detections[det_idx])

        # 4. Spawn tracks for unmatched detections.
        for det_idx in unmatched_dets:
            track = Track(self._next_id, detections[det_idx])
            self._next_id += 1
            self._tracks.append(track)

        # 5. Retire stale tracks.
        survivors: List[Track] = []
        for track in self._tracks:
            track.refresh_phase(self.min_hits)
            if track.time_since_update > self.max_age:
                track.phase = TrackPhase.LOST
                if track.hits >= self.min_hits:
                    self._lost.append(track.snapshot())
                continue
            survivors.append(track)
        self._tracks = survivors

        # Publish only tracks that are confirmed (or coasting after confirmation),
        # which prevents one-frame detector noise from flashing boxes on screen.
        return [
            t.snapshot() for t in self._tracks
            if t.hits >= self.min_hits or t.phase is TrackPhase.CONFIRMED
        ]

    def pop_lost(self) -> List[TrackedObject]:
        """Tracks that died during the last update (drives OBJECT_LOST events)."""
        lost, self._lost = self._lost, []
        return lost


# =============================================================================
# SECTION 8 - VIDEO SOURCES
# =============================================================================

class SourceError(RuntimeError):
    """Raised when a video source cannot be opened or has failed terminally."""


@dataclass(slots=True)
class SourceInfo:
    """Metadata describing an open source, shown in the Video Sources page."""

    kind: SourceKind
    name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    seekable: bool = False
    detail: str = ""

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "-"

    @property
    def duration_s(self) -> float:
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return 0.0


class BaseVideoSource(ABC):
    """Common interface for every frame producer."""

    def __init__(self) -> None:
        self._info = SourceInfo(kind=SourceKind.SIMULATION, name="unknown")
        self._index = 0
        self._opened = False

    @property
    def info(self) -> SourceInfo:
        return self._info

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def position(self) -> int:
        return self._index

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource.  Raises SourceError on failure."""

    @abstractmethod
    def read(self) -> Optional[FramePacket]:
        """Return the next frame, or None when the stream has ended."""

    @abstractmethod
    def release(self) -> None:
        """Release the resource.  Must be idempotent."""

    # Optional capabilities - the defaults keep non-seekable sources honest.
    def seek(self, frame_index: int) -> bool:
        return False

    def restart(self) -> bool:
        return False

    @property
    def supports_seek(self) -> bool:
        return False


class LocalVideoSource(BaseVideoSource):
    """Video file source built on cv2.VideoCapture.

    Handles missing files, unsupported containers and corrupt frames without
    ever propagating an exception into the capture thread's main loop.
    """

    SUPPORTED_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")

    def __init__(self, path: str, loop: bool = True) -> None:
        super().__init__()
        self.path = str(path)
        self.loop = loop
        self._cap: Optional[Any] = None
        self._consecutive_failures = 0

    def open(self) -> None:
        file_path = Path(self.path)
        if not file_path.exists():
            raise SourceError(f"Video file not found:\n{self.path}")
        if not file_path.is_file():
            raise SourceError(f"Not a file:\n{self.path}")
        if file_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            LOGGER.warning("Unusual video suffix '%s' - attempting anyway",
                           file_path.suffix)

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"OpenCV could not open '{file_path.name}'.\n"
                "The container or codec may be unsupported by this OpenCV build.\n"
                "Try re-encoding to H.264 MP4."
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Probe one frame: some corrupt files open successfully but decode nothing.
        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise SourceError(
                f"'{file_path.name}' opened but no frame could be decoded.\n"
                "The file may be corrupt or truncated."
            )
        if width <= 0 or height <= 0:
            height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not (1.0 <= fps <= 240.0):
            LOGGER.warning("Implausible FPS %.2f reported; assuming 25", fps)
            fps = 25.0

        self._cap = cap
        self._opened = True
        self._index = 0
        self._consecutive_failures = 0
        self._info = SourceInfo(
            kind=SourceKind.FILE,
            name=file_path.name,
            width=width, height=height, fps=fps, frame_count=max(0, count),
            seekable=count > 0,
            detail=str(file_path),
        )
        LOGGER.info("Opened video %s (%dx%d @ %.1f fps, %d frames)",
                    file_path.name, width, height, fps, count)

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.loop and self._index > 0:
                # End of file: rewind rather than tearing the pipeline down.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._index = 0
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    return None
            else:
                return None
        if frame.size == 0 or frame.ndim != 3:
            # Corrupt frame: skip it, but give up if it keeps happening.
            self._consecutive_failures += 1
            if self._consecutive_failures > 30:
                raise SourceError("Too many undecodable frames - stream aborted.")
            return None
        self._consecutive_failures = 0
        self._index += 1
        return FramePacket(
            frame=frame, index=self._index, timestamp=time.time(),
            source_name=self._info.name,
        )

    def seek(self, frame_index: int) -> bool:
        if self._cap is None or not self._info.seekable:
            return False
        target = int(max(0, min(frame_index, max(0, self._info.frame_count - 1))))
        ok = bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, target))
        if ok:
            self._index = target
        return ok

    def restart(self) -> bool:
        return self.seek(0)

    @property
    def supports_seek(self) -> bool:
        return self._info.seekable

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("VideoCapture release error: %s", exc)
            self._cap = None
        self._opened = False


class WebcamSource(BaseVideoSource):
    """Webcam / capture-device source."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720) -> None:
        super().__init__()
        self.device_index = int(device_index)
        self.requested_size = (int(width), int(height))
        self._cap: Optional[Any] = None
        self._fail_streak = 0

    @staticmethod
    def enumerate_devices(max_devices: int = 6) -> List[int]:
        """Probe device indices.  Cheap but not free - call sparingly."""
        found: List[int] = []
        for index in range(max_devices):
            cap = None
            try:
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ok, _frame = cap.read()
                    if ok:
                        found.append(index)
            except Exception:  # noqa: BLE001 - probing must never raise
                continue
            finally:
                if cap is not None:
                    cap.release()
        return found

    def open(self) -> None:
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"Camera {self.device_index} is unavailable.\n"
                "It may be disconnected, in use by another application, or "
                "blocked by the operating system's camera privacy settings."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_size[1])
        # A small internal buffer keeps webcam latency low.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - unsupported on some backends
            pass

        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise SourceError(
                f"Camera {self.device_index} opened but returned no frames."
            )
        height, width = probe.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (1.0 <= fps <= 240.0):
            fps = 30.0

        self._cap = cap
        self._opened = True
        self._index = 0
        self._fail_streak = 0
        self._info = SourceInfo(
            kind=SourceKind.WEBCAM,
            name=f"Camera {self.device_index}",
            width=width, height=height, fps=fps,
            detail=f"device index {self.device_index}",
        )
        LOGGER.info("Opened camera %d (%dx%d @ %.1f fps)",
                    self.device_index, width, height, fps)

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            self._fail_streak += 1
            if self._fail_streak > 60:
                raise SourceError("Camera stopped delivering frames (disconnected?).")
            return None
        self._fail_streak = 0
        self._index += 1
        return FramePacket(
            frame=frame, index=self._index, timestamp=time.time(),
            source_name=self._info.name,
        )

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Camera release error: %s", exc)
            self._cap = None
        self._opened = False


@dataclass
class SimActor:
    """One synthetic person or vehicle moving on the simulated camera floor.

    Positions live in normalised floor coordinates (0..1).  Apparent size scales
    with ``y`` so actors further "up" the frame look further away, which gives
    the rendered scene a believable CCTV perspective.
    """

    actor_id: int
    x: float
    y: float
    target_x: float
    target_y: float
    speed: float
    kind: str                       # "person" | "vehicle"
    color: Tuple[int, int, int]
    phase: float = 0.0              # gait animation phase
    dwell: float = 0.0              # seconds to stand still at the target
    loiter_zone: Optional[str] = None
    visible: bool = True

    def step(self, dt: float, rng: random.Random) -> None:
        """Advance toward the current waypoint, picking a new one on arrival."""
        if self.dwell > 0.0:
            self.dwell = max(0.0, self.dwell - dt)
            self.phase += dt * 1.5
            return
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.hypot(dx, dy)
        if distance < 0.015:
            self.pick_target(rng)
            self.dwell = rng.uniform(0.0, 2.5) if self.kind == "person" else 0.0
            return
        step = min(self.speed * dt, distance)
        self.x += dx / distance * step
        self.y += dy / distance * step
        self.phase += dt * (6.0 if self.kind == "person" else 2.0)
        # Keep actors inside the visible floor area.
        self.x = float(np.clip(self.x, 0.03, 0.97))
        self.y = float(np.clip(self.y, 0.30, 0.97))

    def pick_target(self, rng: random.Random) -> None:
        if self.kind == "vehicle":
            # Vehicles patrol a horizontal lane.
            self.target_x = rng.choice([0.06, 0.94])
            self.target_y = float(np.clip(self.y + rng.uniform(-0.03, 0.03), 0.55, 0.9))
        else:
            self.target_x = rng.uniform(0.05, 0.95)
            self.target_y = rng.uniform(0.32, 0.95)

    def scale(self) -> float:
        """Perspective scale factor: 0.45 at the horizon, 1.0 in the foreground."""
        return 0.45 + 0.55 * float(np.clip((self.y - 0.28) / 0.7, 0.0, 1.0))

    def bbox(self, width: int, height: int) -> Tuple[float, float, float, float]:
        scale = self.scale()
        if self.kind == "vehicle":
            bw = 0.20 * scale * width
            bh = 0.11 * scale * height
        else:
            bw = 0.052 * scale * width
            bh = 0.155 * scale * height
        cx = self.x * width
        cy = self.y * height
        return (cx - bw / 2.0, cy - bh, cx + bw / 2.0, cy)


class SimulationSource(BaseVideoSource):
    """Built-in synthetic CCTV feed.

    Renders a stylised surveillance scene (perspective floor, props, grain,
    scanlines, timestamp burn-in) populated with autonomous actors, and emits
    GROUND-TRUTH bounding boxes alongside each frame.  Those boxes are what
    makes the whole product demonstrable with no model weights present, and they
    are always flagged as synthetic so the UI can label them honestly.

    Realism touches that matter for the rest of the pipeline:
        * per-frame box jitter          -> exercises the Kalman smoothing
        * random detection dropout      -> exercises coasting / re-association
        * confidence sampled per actor  -> exercises the confidence threshold
    """

    def __init__(
        self,
        mode: AppMode,
        width: int = 960,
        height: int = 540,
        fps: float = 25.0,
        people: int = 10,
        vehicles: int = 2,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.people = int(np.clip(people, 1, 20))
        self.vehicles = int(np.clip(vehicles, 0, 6))
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._actors: List[SimActor] = []
        self._background: Optional[np.ndarray] = None
        self._last_step = time.time()
        self._noise_bank: List[np.ndarray] = []
        self._noise_index = 0

    # -- lifecycle ------------------------------------------------------------
    def open(self) -> None:
        self._actors = self._spawn_actors()
        self._background = self._render_background()
        # Pre-render a small bank of noise fields; generating fresh noise every
        # frame is surprisingly expensive and would distort the FPS readings.
        self._noise_bank = [
            self._np_rng.normal(0.0, 5.0, (self.height, self.width, 1)).astype(np.float32)
            for _ in range(8)
        ]
        self._opened = True
        self._index = 0
        self._last_step = time.time()
        self._info = SourceInfo(
            kind=SourceKind.SIMULATION,
            name=f"SIM-CAM-01 ({self.mode.value})",
            width=self.width, height=self.height, fps=self.fps,
            detail="Built-in synthetic scene with ground-truth annotations",
        )
        LOGGER.info("Simulation started: %d people, %d vehicles, %s mode",
                    self.people, self.vehicles, self.mode.value)

    def set_population(self, people: int, vehicles: int) -> None:
        """Re-populate the scene without restarting the pipeline."""
        self.people = int(np.clip(people, 1, 20))
        self.vehicles = int(np.clip(vehicles, 0, 6))
        self._actors = self._spawn_actors()

    def set_mode(self, mode: AppMode) -> None:
        self.mode = mode
        self._background = self._render_background()
        self._actors = self._spawn_actors()
        self._info.name = f"SIM-CAM-01 ({mode.value})"

    def release(self) -> None:
        self._actors.clear()
        self._background = None
        self._noise_bank.clear()
        self._opened = False

    def restart(self) -> bool:
        self._index = 0
        self._actors = self._spawn_actors()
        return True

    # -- scene construction ---------------------------------------------------
    def _spawn_actors(self) -> List[SimActor]:
        actors: List[SimActor] = []
        palette = [hex_to_bgr(c) for c in (
            "#D6DEE8", "#B7C4D2", "#9FB0C2", "#C8B8A6", "#A8B6A0", "#C2AEBE",
        )]
        for i in range(self.people):
            x = self._rng.uniform(0.08, 0.92)
            y = self._rng.uniform(0.35, 0.92)
            actor = SimActor(
                actor_id=i + 1, x=x, y=y, target_x=x, target_y=y,
                speed=self._rng.uniform(0.02, 0.075),
                kind="person",
                color=palette[i % len(palette)],
                phase=self._rng.uniform(0.0, 6.28),
            )
            actor.pick_target(self._rng)
            actors.append(actor)

        if self.mode is AppMode.FACTORY:
            for j in range(self.vehicles):
                y = 0.62 + 0.12 * j
                actor = SimActor(
                    actor_id=1000 + j,
                    x=self._rng.uniform(0.15, 0.85), y=min(0.9, y),
                    target_x=0.9, target_y=min(0.9, y),
                    speed=self._rng.uniform(0.05, 0.11),
                    kind="vehicle",
                    color=hex_to_bgr("#E8A33D"),
                )
                actors.append(actor)
        return actors

    def _render_background(self) -> "np.ndarray":
        """Draw the static scene: floor gradient, perspective grid and props."""
        w, h = self.width, self.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.28)

        # Back wall with a vertical gradient.
        wall_top = np.array(hex_to_bgr("#2A3038"), dtype=np.float32)
        wall_bottom = np.array(hex_to_bgr("#3A424C"), dtype=np.float32)
        for y in range(horizon):
            t = y / max(1, horizon)
            frame[y, :] = (wall_top * (1 - t) + wall_bottom * t).astype(np.uint8)

        # Floor with a depth gradient (darker far away).
        floor_far = np.array(hex_to_bgr("#4A5058"), dtype=np.float32)
        floor_near = np.array(hex_to_bgr("#6A727C"), dtype=np.float32)
        for y in range(horizon, h):
            t = (y - horizon) / max(1, h - horizon)
            frame[y, :] = (floor_far * (1 - t) + floor_near * t).astype(np.uint8)

        # Perspective grid converging on a vanishing point.  It is drawn on a
        # scratch layer and composited only BELOW the horizon, so the lines stay
        # on the floor instead of streaking across the back wall.
        grid_layer = frame.copy()
        vp = (int(w * 0.5), int(h * 0.12))
        grid_color = (96, 104, 112)
        for i in range(-12, 13):
            x_bottom = int(w * 0.5 + i * w * 0.14)
            cv2.line(grid_layer, vp, (x_bottom, h), grid_color, 1, cv2.LINE_AA)
        depth = 0.0
        for _ in range(14):
            depth += 0.055 + depth * 0.36
            y = int(horizon + (h - horizon) * min(1.0, depth))
            if y >= h:
                break
            cv2.line(grid_layer, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)
        frame[horizon:, :] = grid_layer[horizon:, :]

        # Scene props differ per deployment scenario.
        if self.mode is AppMode.FACTORY:
            self._draw_factory_props(frame)
        else:
            self._draw_classroom_props(frame)

        # Wall / floor junction line.
        cv2.line(frame, (0, horizon), (w, horizon), (70, 78, 88), 2, cv2.LINE_AA)
        return frame

    def _draw_factory_props(self, frame: "np.ndarray") -> None:
        w, h = self.width, self.height
        # Machines along the left and right.
        for cx, cy, bw, bh in (
            (0.14, 0.52, 0.16, 0.13), (0.14, 0.78, 0.18, 0.15),
            (0.86, 0.55, 0.16, 0.13), (0.86, 0.82, 0.18, 0.15),
        ):
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            depth = int((y2 - y1) * 0.45)
            # Extruded body: a lit top face plus a darker front face reads as a
            # solid machine rather than a marking painted on the floor.
            cv2.rectangle(frame, (x1, y1 - depth), (x2, y2 - depth),
                          hex_to_bgr("#6E7A8A"), -1)
            cv2.rectangle(frame, (x1, y2 - depth), (x2, y2), hex_to_bgr("#3E4652"), -1)
            cv2.rectangle(frame, (x1, y1 - depth), (x2, y2),
                          hex_to_bgr("#8C99AB"), 2)
            cv2.line(frame, (x1, y2 - depth), (x2, y2 - depth),
                     hex_to_bgr("#8C99AB"), 1, cv2.LINE_AA)
            # Status lamp on the machine housing.
            cv2.rectangle(frame, (x1 + 8, y1 - depth + 8), (x1 + 28, y1 - depth + 20),
                          hex_to_bgr("#E8B84B"), -1)
        # Hazard floor striping in front of the danger area.
        for i in range(0, w, 34):
            cv2.line(frame, (i, int(h * 0.40)), (i + 16, int(h * 0.40)),
                     hex_to_bgr("#C8A13A"), 3, cv2.LINE_AA)
        # Conveyor band.
        cv2.rectangle(frame, (int(w * 0.30), int(h * 0.33)),
                      (int(w * 0.70), int(h * 0.38)), hex_to_bgr("#4E5661"), -1)

    def _draw_classroom_props(self, frame: "np.ndarray") -> None:
        w, h = self.width, self.height
        # Whiteboard on the back wall.
        cv2.rectangle(frame, (int(w * 0.30), int(h * 0.08)),
                      (int(w * 0.70), int(h * 0.24)), hex_to_bgr("#D8DEE6"), -1)
        cv2.rectangle(frame, (int(w * 0.30), int(h * 0.08)),
                      (int(w * 0.70), int(h * 0.24)), hex_to_bgr("#8A94A2"), 2)
        # Teacher desk.
        cv2.rectangle(frame, (int(w * 0.40), int(h * 0.30)),
                      (int(w * 0.60), int(h * 0.37)), hex_to_bgr("#6B5844"), -1)
        # Student desk grid, scaled by row for perspective.
        for row in range(4):
            y = 0.50 + row * 0.115
            scale = 0.45 + 0.55 * ((y - 0.28) / 0.7)
            dw = 0.085 * scale
            dh = 0.035 * scale
            for col in range(5):
                x = 0.18 + col * 0.16
                x1 = int((x - dw / 2) * w); y1 = int((y - dh / 2) * h)
                x2 = int((x + dw / 2) * w); y2 = int((y + dh / 2) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), hex_to_bgr("#7A6A52"), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), hex_to_bgr("#95856A"), 1)

    # -- actor rendering ------------------------------------------------------
    def _draw_person(self, frame: "np.ndarray", actor: SimActor) -> None:
        w, h = self.width, self.height
        scale = actor.scale()
        cx = actor.x * w
        base_y = actor.y * h
        body_h = 0.155 * scale * h
        body_w = 0.052 * scale * w
        head_r = max(2, int(body_w * 0.36))

        # Contact shadow grounds the figure on the floor.
        cv2.ellipse(frame, (int(cx), int(base_y)),
                    (int(body_w * 0.62), max(2, int(body_w * 0.22))),
                    0, 0, 360, (40, 44, 50), -1, cv2.LINE_AA)

        torso_top = base_y - body_h * 0.62
        gait = math.sin(actor.phase) * body_w * 0.28

        # Legs.
        leg_color = tuple(int(c * 0.6) for c in actor.color)
        cv2.line(frame, (int(cx), int(torso_top + body_h * 0.30)),
                 (int(cx - gait), int(base_y)), leg_color,
                 max(2, int(body_w * 0.20)), cv2.LINE_AA)
        cv2.line(frame, (int(cx), int(torso_top + body_h * 0.30)),
                 (int(cx + gait), int(base_y)), leg_color,
                 max(2, int(body_w * 0.20)), cv2.LINE_AA)
        # Torso.
        cv2.ellipse(frame, (int(cx), int(torso_top + body_h * 0.16)),
                    (int(body_w * 0.44), int(body_h * 0.28)),
                    0, 0, 360, actor.color, -1, cv2.LINE_AA)
        # Head (a plain filled circle - deliberately featureless; this system
        # performs no facial analysis of any kind).
        head_y = torso_top - head_r * 0.7
        cv2.circle(frame, (int(cx), int(head_y)), head_r,
                   tuple(int(min(255, c * 1.08)) for c in actor.color), -1, cv2.LINE_AA)

    def _draw_vehicle(self, frame: "np.ndarray", actor: SimActor) -> None:
        w, h = self.width, self.height
        scale = actor.scale()
        cx = actor.x * w
        base_y = actor.y * h
        bw = 0.20 * scale * w
        bh = 0.11 * scale * h

        cv2.ellipse(frame, (int(cx), int(base_y)),
                    (int(bw * 0.5), max(2, int(bh * 0.12))),
                    0, 0, 360, (40, 44, 50), -1, cv2.LINE_AA)
        x1, y1 = int(cx - bw / 2), int(base_y - bh)
        x2, y2 = int(cx + bw / 2), int(base_y - bh * 0.22)
        cv2.rectangle(frame, (x1, y1), (x2, y2), actor.color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      tuple(int(c * 0.65) for c in actor.color), 2)
        # Mast and wheels give the silhouette a forklift read.
        cv2.rectangle(frame, (x2 - int(bw * 0.10), int(base_y - bh * 1.6)),
                      (x2, y2), (90, 96, 104), -1)
        radius = max(2, int(bh * 0.20))
        cv2.circle(frame, (x1 + radius * 2, int(base_y - radius)), radius, (30, 32, 36), -1)
        cv2.circle(frame, (x2 - radius * 2, int(base_y - radius)), radius, (30, 32, 36), -1)

    def _apply_camera_effects(self, frame: "np.ndarray") -> "np.ndarray":
        """Grain, scanlines, vignette and burn-in overlay -> CCTV look."""
        noise = self._noise_bank[self._noise_index % len(self._noise_bank)]
        self._noise_index += 1
        out = np.clip(frame.astype(np.float32) + noise, 0, 255)

        # Subtle horizontal scanlines.
        out[::3, :, :] *= 0.965

        # Vignette, computed once and cached on the instance.
        vignette = getattr(self, "_vignette", None)
        if vignette is None:
            h, w = frame.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = h / 2.0, w / 2.0
            radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            vignette = np.clip(1.10 - 0.34 * radius ** 2, 0.55, 1.0).astype(np.float32)[..., None]
            self._vignette = vignette  # type: ignore[attr-defined]
        out *= vignette

        result = out.astype(np.uint8)

        # Timestamp and camera-ID burn-in, like a real DVR overlay.
        stamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(result, stamp, (12, self.height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, stamp, (12, self.height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 228, 236), 1, cv2.LINE_AA)
        # Camera-ID burn-in sits below y=30 so it never collides with the
        # pipeline HUD strip the renderer draws across the top of the frame.
        label = f"SIM-CAM-01  {self.mode.value.upper()}"
        cv2.putText(result, label, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 228, 236), 1, cv2.LINE_AA)
        # Recording dot, also clear of the HUD strip.
        if (self._index // 12) % 2 == 0:
            cv2.circle(result, (self.width - 22, 46), 5, hex_to_bgr("#FF4D5E"), -1, cv2.LINE_AA)
        return result

    # -- frame production -----------------------------------------------------
    def read(self) -> Optional[FramePacket]:
        if not self._opened or self._background is None:
            return None

        now = time.time()
        dt = min(0.2, max(0.0, now - self._last_step))
        self._last_step = now

        frame = self._background.copy()

        # Painter's algorithm: far actors first so nearer ones occlude them.
        actors = sorted(self._actors, key=lambda a: a.y)
        detections: List[Detection] = []
        for actor in actors:
            actor.step(dt, self._rng)
            if actor.kind == "vehicle":
                self._draw_vehicle(frame, actor)
            else:
                self._draw_person(frame, actor)

            # Ground-truth box, jittered so tracking has real work to do.
            x1, y1, x2, y2 = actor.bbox(self.width, self.height)
            jitter = 0.012 * (x2 - x1)
            x1 += self._rng.uniform(-jitter, jitter)
            x2 += self._rng.uniform(-jitter, jitter)
            y1 += self._rng.uniform(-jitter, jitter)
            y2 += self._rng.uniform(-jitter, jitter)

            # Occasional dropout emulates a real detector missing a frame and
            # forces the Kalman filter to coast.
            if self._rng.random() < 0.04:
                continue

            detections.append(
                Detection(
                    bbox=(max(0.0, x1), max(0.0, y1),
                          min(self.width - 1.0, x2), min(self.height - 1.0, y2)),
                    confidence=round(self._rng.uniform(0.62, 0.97), 3),
                    class_id=0 if actor.kind == "person" else 2,
                    class_name="person" if actor.kind == "person" else "truck",
                )
            )

        frame = self._apply_camera_effects(frame)
        self._index += 1
        return FramePacket(
            frame=frame, index=self._index, timestamp=now,
            source_name=self._info.name,
            synthetic=True,
            synthetic_detections=detections,
        )


# =============================================================================
# SECTION 9 - VISION PIPELINE (zone presets, supervision adapter, annotation)
# =============================================================================

def default_zones(mode: AppMode) -> List[ZoneConfig]:
    """Scenario-appropriate starter zones in normalised coordinates."""
    def rect(x1: float, y1: float, x2: float, y2: float) -> List[Tuple[float, float]]:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if mode is AppMode.CLASSROOM:
        return [
            ZoneConfig("teacher", "Teacher Area", ZoneKind.TEACHER,
                       rect(0.31, 0.29, 0.69, 0.45),
                       ZoneKind.TEACHER.default_color, Severity.INFO,
                       unattended_after_s=25.0),
            ZoneConfig("student", "Student Area", ZoneKind.STUDENT,
                       rect(0.10, 0.47, 0.90, 0.97),
                       ZoneKind.STUDENT.default_color, Severity.INFO,
                       max_occupancy=12),
            ZoneConfig("restricted", "Restricted Area", ZoneKind.RESTRICTED,
                       rect(0.02, 0.52, 0.19, 0.97),
                       ZoneKind.RESTRICTED.default_color, Severity.CRITICAL),
            ZoneConfig("entrance", "Entrance", ZoneKind.ENTRANCE,
                       rect(0.80, 0.60, 0.98, 0.97),
                       ZoneKind.ENTRANCE.default_color, Severity.INFO),
        ]
    return [
        ZoneConfig("machine", "Machine Zone", ZoneKind.MACHINE,
                   rect(0.03, 0.42, 0.26, 0.96),
                   ZoneKind.MACHINE.default_color, Severity.WARNING),
        ZoneConfig("danger", "Danger Zone", ZoneKind.DANGER,
                   rect(0.29, 0.29, 0.71, 0.42),
                   ZoneKind.DANGER.default_color, Severity.CRITICAL),
        ZoneConfig("worker", "Worker Zone", ZoneKind.WORK,
                   rect(0.28, 0.45, 0.73, 0.96),
                   ZoneKind.WORK.default_color, Severity.INFO,
                   max_occupancy=10),
        ZoneConfig("restricted", "Restricted Zone", ZoneKind.RESTRICTED,
                   rect(0.76, 0.45, 0.97, 0.96),
                   ZoneKind.RESTRICTED.default_color, Severity.CRITICAL),
        ZoneConfig("entrance", "Entrance", ZoneKind.ENTRANCE,
                   rect(0.40, 0.90, 0.62, 0.99),
                   ZoneKind.ENTRANCE.default_color, Severity.INFO),
    ]


def default_lines(mode: AppMode) -> List[LineConfig]:
    """A single entry/exit counting line, placed mid-floor for both scenarios."""
    y = 0.60 if mode is AppMode.CLASSROOM else 0.58
    return [LineConfig("entryline", "Entry / Exit Line", (0.04, y), (0.96, y))]


class SupervisionAdapter:
    """Thin, entirely optional bridge to the ``supervision`` library.

    The application's own data model is authoritative; this adapter only
    borrows supervision's detection container and annotators when the package is
    installed.  Every method degrades to a no-op / ``None`` when it is not, and
    the OpenCV renderer below covers the same ground, so nothing in the product
    depends on supervision being present.
    """

    def __init__(self) -> None:
        self.sv: Optional[Any] = DEP_SUPERVISION.module if DEP_SUPERVISION.available else None
        self._box_annotator: Optional[Any] = None
        self._label_annotator: Optional[Any] = None
        self._trace_annotator: Optional[Any] = None
        self._ready = False
        if self.sv is not None:
            self._build_annotators()

    @property
    def available(self) -> bool:
        return self._ready

    def _build_annotators(self) -> None:
        """Construct annotators defensively - their kwargs vary across releases."""
        sv = self.sv
        assert sv is not None
        try:
            palette = sv.ColorPalette.from_hex(list(Palette.TRACK_COLORS))
        except Exception:  # noqa: BLE001 - older/newer palette APIs
            palette = None
        try:
            kwargs: Dict[str, Any] = {"thickness": 2}
            if palette is not None:
                kwargs["color"] = palette
            self._box_annotator = sv.BoxAnnotator(**kwargs)
            label_kwargs: Dict[str, Any] = {"text_scale": 0.4, "text_padding": 4}
            if palette is not None:
                label_kwargs["color"] = palette
            self._label_annotator = sv.LabelAnnotator(**label_kwargs)
            try:
                self._trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)
            except Exception:  # noqa: BLE001 - trace annotator is a bonus only
                self._trace_annotator = None
            self._ready = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("supervision annotators unavailable: %s", exc)
            self._ready = False

    def to_sv_detections(self, tracks: Sequence[TrackedObject]) -> Optional[Any]:
        """Convert our tracks into a ``sv.Detections`` instance."""
        if self.sv is None or not tracks:
            return None
        try:
            xyxy = np.array([t.bbox for t in tracks], dtype=np.float32)
            confidence = np.array([t.confidence for t in tracks], dtype=np.float32)
            class_id = np.array([max(0, t.class_id) for t in tracks], dtype=int)
            tracker_id = np.array([t.track_id for t in tracks], dtype=int)
            return self.sv.Detections(
                xyxy=xyxy, confidence=confidence, class_id=class_id,
                tracker_id=tracker_id,
                data={"class_name": np.array([t.class_name for t in tracks])},
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("sv.Detections conversion failed: %s", exc)
            return None

    def annotate(
        self, frame: "np.ndarray", tracks: Sequence[TrackedObject],
        labels: Sequence[str], draw_traces: bool,
    ) -> Optional["np.ndarray"]:
        """Annotate with supervision; returns None to signal 'use the fallback'."""
        if not self._ready or self._box_annotator is None:
            return None
        detections = self.to_sv_detections(tracks)
        if detections is None:
            return None
        try:
            out = frame
            if draw_traces and self._trace_annotator is not None:
                out = self._trace_annotator.annotate(out, detections)
            out = self._box_annotator.annotate(out, detections)
            if self._label_annotator is not None and labels:
                out = self._label_annotator.annotate(out, detections, list(labels))
            return out
        except Exception as exc:  # noqa: BLE001 - never break rendering
            LOGGER.debug("supervision annotation failed, using fallback: %s", exc)
            self._ready = False
            return None


class FrameRenderer:
    """Draws zones, lines, boxes, IDs and HUD overlays onto the output frame.

    Uses supervision's annotators when available and falls back to a complete
    OpenCV implementation otherwise, so the visual output is equivalent either
    way.
    """

    def __init__(self, adapter: SupervisionAdapter) -> None:
        self.adapter = adapter

    # -- zones / lines --------------------------------------------------------
    @staticmethod
    def draw_zones(
        frame: "np.ndarray", zones: Sequence[ZoneConfig],
        status: Dict[str, ZoneStatus],
    ) -> None:
        if not zones:
            return
        overlay = frame.copy()
        height, width = frame.shape[:2]
        for zone in zones:
            if not zone.enabled or len(zone.polygon) < 3:
                continue
            poly = zone.pixel_polygon(width, height)
            state = status.get(zone.zone_id)
            breached = bool(state and state.breached)
            color = hex_to_bgr(Palette.CRIT if breached else zone.color)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2 if not breached else 3, cv2.LINE_AA)

            label = zone.name
            if state is not None:
                label = f"{zone.name}  [{state.occupancy}]"
            x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
            y = max(14, y)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x, y - th - 7), (x + tw + 8, y), color, -1)
            cv2.putText(frame, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (16, 18, 22), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, dst=frame)

    @staticmethod
    def draw_lines(
        frame: "np.ndarray", lines: Sequence[LineConfig],
        counts: Dict[str, Tuple[int, int]],
    ) -> None:
        height, width = frame.shape[:2]
        for line in lines:
            if not line.enabled:
                continue
            p1, p2 = line.pixel_points(width, height)
            color = hex_to_bgr(line.color)
            cv2.line(frame, p1, p2, (18, 20, 24), 5, cv2.LINE_AA)
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
            cv2.circle(frame, p1, 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p2, 4, color, -1, cv2.LINE_AA)
            inbound, outbound = counts.get(line.line_id, (0, 0))
            text = f"{line.name}  IN {inbound}  OUT {outbound}"
            tx = min(p1[0], p2[0]) + 6
            ty = max(16, min(p1[1], p2[1]) - 8)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, color, 1, cv2.LINE_AA)

    # -- detections / tracks --------------------------------------------------
    def draw_tracks(
        self, frame: "np.ndarray", tracks: Sequence[TrackedObject],
        settings: AppSettings, alert_ids: Iterable[int] = (),
    ) -> None:
        if not tracks or not settings.show_boxes:
            return
        alerts = set(alert_ids)
        labels = [self._label_for(t, settings) for t in tracks]

        # Supervision path (skipped for alerting tracks, which need the custom
        # red treatment that communicates severity at a glance).
        if (settings.use_supervision_annotators and self.adapter.available
                and not alerts):
            annotated = self.adapter.annotate(
                frame, tracks, labels, settings.show_trails
            )
            if annotated is not None:
                if annotated is not frame:
                    np.copyto(frame, annotated)
                return

        # OpenCV fallback / alert path.
        for track, label in zip(tracks, labels):
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            alerting = track.track_id in alerts
            color = hex_to_bgr(Palette.CRIT) if alerting \
                else Palette.track_color_bgr(track.track_id)
            thickness = 3 if alerting else 2

            if settings.show_trails and len(track.trail) > 2:
                pts = np.array(track.trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

            if track.phase is TrackPhase.COASTING:
                # Dashed box communicates "predicted, not currently detected".
                self._dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                # Corner accents give the boxes a product-grade look.
                arm = max(6, int((x2 - x1) * 0.22))
                for cx, cy, dx, dy in (
                    (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
                ):
                    cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, thickness + 1, cv2.LINE_AA)
                    cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, thickness + 1, cv2.LINE_AA)

            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                ly = max(th + 6, y1)
                cv2.rectangle(frame, (x1, ly - th - 7), (x1 + tw + 8, ly), color, -1)
                cv2.putText(frame, label, (x1 + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (16, 18, 22), 1, cv2.LINE_AA)

            if alerting:
                cv2.putText(frame, "ALERT", (x1, min(frame.shape[0] - 4, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, hex_to_bgr(Palette.CRIT),
                            2, cv2.LINE_AA)

    @staticmethod
    def _label_for(track: TrackedObject, settings: AppSettings) -> str:
        parts: List[str] = []
        if settings.show_track_ids:
            parts.append(f"#{track.track_id}")
        parts.append(track.class_name)
        if settings.show_confidence:
            parts.append(f"{track.confidence:.2f}")
        return " ".join(parts)

    @staticmethod
    def _dashed_rect(
        frame: "np.ndarray", p1: Tuple[int, int], p2: Tuple[int, int],
        color: Tuple[int, int, int], thickness: int, dash: int = 8,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)

    # -- HUD ------------------------------------------------------------------
    @staticmethod
    def draw_hud(
        frame: "np.ndarray", *, fps: float, mode: AppMode, source: str,
        detector: str, people: int, vehicles: int, tracks: int,
        synthetic: bool, settings: AppSettings, critical: int,
    ) -> None:
        height, width = frame.shape[:2]
        pad = 10
        bar_h = 30

        # Translucent top HUD strip.
        strip = frame[0:bar_h, 0:width]
        cv2.addWeighted(np.full_like(strip, (14, 16, 20)), 0.62, strip, 0.38, 0, dst=strip)

        left_text = f"{mode.value.upper()}  |  {source}  |  {detector}"
        cv2.putText(frame, left_text, (pad, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (226, 234, 244), 1, cv2.LINE_AA)

        right_bits: List[str] = []
        if settings.show_fps_overlay:
            right_bits.append(f"{fps:5.1f} FPS")
        right_bits.append(f"P {people}")
        if vehicles:
            right_bits.append(f"V {vehicles}")
        right_bits.append(f"T {tracks}")
        right_text = "   ".join(right_bits)
        (tw, _th), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, right_text, (width - tw - pad, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 234, 244), 1, cv2.LINE_AA)

        # SIMULATION badge - always visible when the feed is synthetic so the
        # operator can never mistake it for a real camera.
        if synthetic:
            badge = "SIMULATION"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            bx, by = width - bw - 16 - pad, bar_h + 6
            cv2.rectangle(frame, (bx, by), (bx + bw + 16, by + bh + 12),
                          hex_to_bgr(Palette.VIOLET), -1)
            cv2.putText(frame, badge, (bx + 8, by + bh + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 14, 30), 2, cv2.LINE_AA)

        # Critical-alert border pulse.
        if critical > 0:
            cv2.rectangle(frame, (2, 2), (width - 3, height - 3),
                          hex_to_bgr(Palette.CRIT), 3)

    @staticmethod
    def placeholder(
        width: int, height: int, title: str, subtitle: str = "",
        accent: str = Palette.TEXT_DIM,
    ) -> "np.ndarray":
        """Render a NO SIGNAL / MODEL LOADING style placeholder frame."""
        frame = np.full((height, width, 3), 14, dtype=np.uint8)
        frame[:, :, 0] = 18
        frame[:, :, 1] = 16
        # Faint diagonal hatching.
        for x in range(-height, width, 28):
            cv2.line(frame, (x, 0), (x + height, height), (26, 30, 36), 1, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, title, ((width - tw) // 2, (height + th) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, hex_to_bgr(accent), 2, cv2.LINE_AA)
        if subtitle:
            (sw, _sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, subtitle, ((width - sw) // 2, (height + th) // 2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 144, 160), 1, cv2.LINE_AA)
        return frame


# =============================================================================
# SECTION 10 - EVENT ENGINE
# =============================================================================

class EventEngine:
    """Turns per-frame track state into de-duplicated, severity-graded events.

    Rules implemented (all purely geometric / count based - see the note on
    capability honesty below):

        * PERSON_ENTERED / PERSON_EXITED  - directed crossing of a counting line
        * LINE_CROSSED                    - any track crossing a counting line
        * RESTRICTED_ZONE_ENTRY           - track anchor enters a restricted or
                                            danger zone
        * CROWDING                        - zone occupancy exceeds its limit
        * HIGH_OCCUPANCY                  - total people exceed the global limit
        * VEHICLE_DETECTED                - a vehicle-class track is confirmed
        * OBJECT_LOST                     - a confirmed track dies
        * UNATTENDED_AREA                 - a watched zone stays empty too long
        * PROXIMITY_WARNING               - a person track comes within N pixels
                                            of a machine-zone boundary

    CAPABILITY HONESTY: this engine reasons ONLY about object class, position,
    and geometry.  It does not and cannot infer PPE compliance, falls, fatigue,
    weapons, intent or identity.  Any such feature would require a dedicated
    model that this application does not ship.
    """

    #: Per-rule minimum seconds between two identical events.
    COOLDOWNS: Dict[EventType, float] = {
        EventType.RESTRICTED_ZONE_ENTRY: 4.0,
        EventType.CROWDING: 12.0,
        EventType.HIGH_OCCUPANCY: 15.0,
        EventType.VEHICLE_DETECTED: 20.0,
        EventType.PROXIMITY_WARNING: 8.0,
        EventType.UNATTENDED_AREA: 30.0,
        EventType.OBJECT_LOST: 0.0,
        EventType.LINE_CROSSED: 0.0,
        EventType.PERSON_ENTERED: 0.0,
        EventType.PERSON_EXITED: 0.0,
        EventType.SYSTEM_WARNING: 10.0,
    }

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._zones: List[ZoneConfig] = []
        self._lines: List[LineConfig] = []
        self._source_name: str = ""

        self._cooldown: Dict[Tuple[str, ...], float] = {}
        self._zone_membership: Dict[str, set] = {}      # zone_id -> track ids
        self._line_side: Dict[Tuple[str, int], float] = {}
        self._line_counts: Dict[str, List[int]] = {}    # line_id -> [in, out]
        self._zone_empty_since: Dict[str, float] = {}
        self._known_vehicles: set = set()
        self._lock = threading.RLock()

    # -- configuration --------------------------------------------------------
    def configure(
        self, zones: Sequence[ZoneConfig], lines: Sequence[LineConfig],
        source_name: str = "",
    ) -> None:
        with self._lock:
            self._zones = list(zones)
            self._lines = list(lines)
            self._source_name = source_name
            known_lines = {line.line_id for line in self._lines}
            for line_id in known_lines:
                self._line_counts.setdefault(line_id, [0, 0])
            self._zone_membership = {
                z.zone_id: self._zone_membership.get(z.zone_id, set()) for z in self._zones
            }

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    def reset(self) -> None:
        with self._lock:
            self._cooldown.clear()
            self._zone_membership = {z.zone_id: set() for z in self._zones}
            self._line_side.clear()
            self._line_counts = {line.line_id: [0, 0] for line in self._lines}
            self._zone_empty_since.clear()
            self._known_vehicles.clear()

    @property
    def line_counts(self) -> Dict[str, Tuple[int, int]]:
        with self._lock:
            return {k: (v[0], v[1]) for k, v in self._line_counts.items()}

    @property
    def total_in_out(self) -> Tuple[int, int]:
        with self._lock:
            total_in = sum(v[0] for v in self._line_counts.values())
            total_out = sum(v[1] for v in self._line_counts.values())
        return total_in, total_out

    # -- cooldown helper ------------------------------------------------------
    def _allowed(self, event_type: EventType, *key_parts: Any) -> bool:
        cooldown = self.COOLDOWNS.get(event_type, 5.0)
        if cooldown <= 0.0:
            return True
        key = (event_type.value,) + tuple(str(p) for p in key_parts)
        now = time.time()
        last = self._cooldown.get(key, 0.0)
        if now - last < cooldown:
            return False
        self._cooldown[key] = now
        return True

    def _make(
        self, event_type: EventType, severity: Severity, description: str,
        track_id: Optional[int] = None, zone: Optional[str] = None,
    ) -> SecurityEvent:
        return SecurityEvent(
            timestamp=time.time(), event_type=event_type, severity=severity,
            description=description, track_id=track_id, zone=zone,
            mode=self._settings.mode.value, source=self._source_name,
        )

    # -- main evaluation ------------------------------------------------------
    def process(
        self, tracks: Sequence[TrackedObject], lost: Sequence[TrackedObject],
        width: int, height: int,
    ) -> Tuple[List[SecurityEvent], List[ZoneStatus], set]:
        """Evaluate every rule for one frame.

        Returns (new events, per-zone status, ids of tracks currently alerting).
        """
        with self._lock:
            events: List[SecurityEvent] = []
            alert_ids: set = set()

            people = [t for t in tracks if t.is_person]
            vehicles = [t for t in tracks if t.is_vehicle]

            zone_status = self._evaluate_zones(tracks, people, width, height,
                                               events, alert_ids)
            self._evaluate_lines(tracks, width, height, events)
            self._evaluate_occupancy(people, events)
            self._evaluate_vehicles(vehicles, events)
            self._evaluate_proximity(people, width, height, events, alert_ids)
            self._evaluate_lost(lost, events)
            return events, zone_status, alert_ids

    # -- individual rules -----------------------------------------------------
    def _evaluate_zones(
        self, tracks: Sequence[TrackedObject], people: Sequence[TrackedObject],
        width: int, height: int, events: List[SecurityEvent], alert_ids: set,
    ) -> List[ZoneStatus]:
        statuses: List[ZoneStatus] = []
        now = time.time()

        for zone in self._zones:
            inside_ids: set = set()
            occupancy = 0
            for track in tracks:
                if not zone.enabled:
                    continue
                if zone.contains(track.anchor, width, height):
                    inside_ids.add(track.track_id)
                    if track.is_person:
                        occupancy += 1

            previous = self._zone_membership.get(zone.zone_id, set())
            entered = inside_ids - previous
            self._zone_membership[zone.zone_id] = inside_ids

            breached = False
            restricted = zone.kind in (ZoneKind.RESTRICTED, ZoneKind.DANGER)

            if restricted and zone.enabled and inside_ids:
                breached = True
                alert_ids.update(inside_ids)
                for track_id in entered:
                    if self._allowed(EventType.RESTRICTED_ZONE_ENTRY, zone.zone_id, track_id):
                        events.append(self._make(
                            EventType.RESTRICTED_ZONE_ENTRY, zone.severity,
                            f"Track #{track_id} entered restricted area '{zone.name}'",
                            track_id=track_id, zone=zone.name,
                        ))

            # Crowding inside a zone with a configured limit.
            if zone.enabled and zone.max_occupancy > 0 and occupancy > zone.max_occupancy:
                breached = True
                if self._allowed(EventType.CROWDING, zone.zone_id):
                    events.append(self._make(
                        EventType.CROWDING, Severity.WARNING,
                        f"'{zone.name}' occupancy {occupancy} exceeds limit "
                        f"{zone.max_occupancy}",
                        zone=zone.name,
                    ))

            # Unattended-area rule: a watched zone that stays empty too long.
            if zone.enabled and zone.unattended_after_s > 0:
                if occupancy > 0:
                    self._zone_empty_since.pop(zone.zone_id, None)
                else:
                    since = self._zone_empty_since.setdefault(zone.zone_id, now)
                    elapsed = now - since
                    if elapsed >= zone.unattended_after_s and \
                            self._allowed(EventType.UNATTENDED_AREA, zone.zone_id):
                        events.append(self._make(
                            EventType.UNATTENDED_AREA, Severity.WARNING,
                            f"'{zone.name}' has been unattended for "
                            f"{int(elapsed)}s",
                            zone=zone.name,
                        ))

            statuses.append(ZoneStatus(
                zone_id=zone.zone_id, name=zone.name, kind=zone.kind,
                color=zone.color, occupancy=occupancy, objects=len(inside_ids),
                max_occupancy=zone.max_occupancy, breached=breached,
                enabled=zone.enabled,
            ))
        return statuses

    def _evaluate_lines(
        self, tracks: Sequence[TrackedObject], width: int, height: int,
        events: List[SecurityEvent],
    ) -> None:
        for line in self._lines:
            if not line.enabled:
                continue
            counts = self._line_counts.setdefault(line.line_id, [0, 0])
            live_ids = set()
            for track in tracks:
                live_ids.add(track.track_id)
                # Normalised anchor keeps the maths resolution-independent.
                anchor = (track.anchor[0] / max(1, width), track.anchor[1] / max(1, height))
                side = line.side(anchor)
                key = (line.line_id, track.track_id)
                previous = self._line_side.get(key)
                self._line_side[key] = side
                if previous is None or abs(side) < 1e-6:
                    continue
                if (previous < 0) == (side < 0):
                    continue  # no sign change -> no crossing

                inbound = side < 0      # crossing toward the "upper" side = IN
                if inbound:
                    counts[0] += 1
                else:
                    counts[1] += 1

                events.append(self._make(
                    EventType.LINE_CROSSED, Severity.INFO,
                    f"Track #{track.track_id} crossed '{line.name}' "
                    f"({'in' if inbound else 'out'})",
                    track_id=track.track_id, zone=line.name,
                ))
                if track.is_person:
                    events.append(self._make(
                        EventType.PERSON_ENTERED if inbound else EventType.PERSON_EXITED,
                        Severity.INFO,
                        f"Person #{track.track_id} "
                        f"{'entered' if inbound else 'exited'} via '{line.name}'",
                        track_id=track.track_id, zone=line.name,
                    ))

            # Forget side history for tracks that no longer exist.
            stale = [k for k in self._line_side
                     if k[0] == line.line_id and k[1] not in live_ids]
            for key in stale:
                self._line_side.pop(key, None)

    def _evaluate_occupancy(
        self, people: Sequence[TrackedObject], events: List[SecurityEvent]
    ) -> None:
        count = len(people)
        if count > self._settings.high_occupancy_threshold and \
                self._allowed(EventType.HIGH_OCCUPANCY):
            events.append(self._make(
                EventType.HIGH_OCCUPANCY, Severity.WARNING,
                f"Total occupancy {count} exceeds threshold "
                f"{self._settings.high_occupancy_threshold}",
            ))
        elif count > self._settings.crowding_threshold and \
                self._allowed(EventType.CROWDING, "global"):
            events.append(self._make(
                EventType.CROWDING, Severity.WARNING,
                f"Crowding detected: {count} people in view "
                f"(threshold {self._settings.crowding_threshold})",
            ))

    def _evaluate_vehicles(
        self, vehicles: Sequence[TrackedObject], events: List[SecurityEvent]
    ) -> None:
        for vehicle in vehicles:
            if vehicle.track_id in self._known_vehicles:
                continue
            self._known_vehicles.add(vehicle.track_id)
            if self._allowed(EventType.VEHICLE_DETECTED, vehicle.track_id):
                events.append(self._make(
                    EventType.VEHICLE_DETECTED, Severity.INFO,
                    f"Vehicle detected ({vehicle.class_name}) as track "
                    f"#{vehicle.track_id}",
                    track_id=vehicle.track_id,
                ))

    def _evaluate_proximity(
        self, people: Sequence[TrackedObject], width: int, height: int,
        events: List[SecurityEvent], alert_ids: set,
    ) -> None:
        """Worker-machine proximity: distance from a person anchor to a machine
        zone boundary.  This is a geometric proximity heuristic only."""
        machine_zones = [z for z in self._zones
                         if z.enabled and z.kind is ZoneKind.MACHINE]
        if not machine_zones:
            return
        limit = float(self._settings.proximity_px)
        for zone in machine_zones:
            for person in people:
                distance = zone.distance_to(person.anchor, width, height)
                # Positive distance means inside the polygon; a small negative
                # value means just outside the machine footprint.
                if distance > 0 or abs(distance) > limit:
                    continue
                alert_ids.add(person.track_id)
                if self._allowed(EventType.PROXIMITY_WARNING, zone.zone_id, person.track_id):
                    events.append(self._make(
                        EventType.PROXIMITY_WARNING, Severity.WARNING,
                        f"Worker #{person.track_id} within {abs(distance):.0f}px "
                        f"of '{zone.name}'",
                        track_id=person.track_id, zone=zone.name,
                    ))

    def _evaluate_lost(
        self, lost: Sequence[TrackedObject], events: List[SecurityEvent]
    ) -> None:
        for track in lost:
            self._known_vehicles.discard(track.track_id)
            events.append(self._make(
                EventType.OBJECT_LOST, Severity.INFO,
                f"Track #{track.track_id} ({track.class_name}) lost after "
                f"{track.age} frames",
                track_id=track.track_id,
            ))


# =============================================================================
# SECTION 11 - ANALYTICS
# =============================================================================

@dataclass(slots=True)
class AnalyticsSnapshot:
    """Aggregated counters read by the dashboard on its refresh timer."""

    people: int = 0
    vehicles: int = 0
    others: int = 0
    tracks: int = 0
    total_objects: int = 0
    peak_people: int = 0
    events_today: int = 0
    warnings: int = 0
    criticals: int = 0
    infos: int = 0
    entered: int = 0
    exited: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    detect_ms: float = 0.0
    track_ms: float = 0.0
    uptime_s: float = 0.0
    class_histogram: Tuple[Tuple[str, int], ...] = ()


class AnalyticsEngine:
    """Rolling time-series + counters.  Thread-safe (written by the worker,
    read by the GUI)."""

    HISTORY = 180   # samples retained per series (~1-3 minutes)

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started = time.time()
        self.people_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.vehicle_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.fps_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.latency_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.track_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)
        self.event_series: Deque[Tuple[float, float]] = deque(maxlen=self.HISTORY)

        self._events_bucket: Deque[float] = deque(maxlen=2000)
        self._counts = Counter()
        self._class_histogram = Counter()
        self._peak_people = 0
        self._entered = 0
        self._exited = 0
        self._last_event_flush = time.time()
        self._events_since_flush = 0

    def reset(self) -> None:
        with self._lock:
            for series in (self.people_series, self.vehicle_series, self.fps_series,
                           self.latency_series, self.track_series, self.event_series):
                series.clear()
            self._events_bucket.clear()
            self._counts.clear()
            self._class_histogram.clear()
            self._peak_people = 0
            self._entered = 0
            self._exited = 0
            self._started = time.time()

    def record_frame(
        self, *, people: int, vehicles: int, others: int, tracks: int,
        fps: float, timings: Timings, class_names: Sequence[str],
    ) -> None:
        with self._lock:
            now = time.time()
            self.people_series.append((now, float(people)))
            self.vehicle_series.append((now, float(vehicles)))
            self.fps_series.append((now, float(fps)))
            self.latency_series.append((now, float(timings.total_ms)))
            self.track_series.append((now, float(tracks)))
            self._peak_people = max(self._peak_people, people)
            for name in class_names:
                self._class_histogram[name] += 1

            # Events-per-interval series, flushed once a second.
            if now - self._last_event_flush >= 1.0:
                self.event_series.append((now, float(self._events_since_flush)))
                self._events_since_flush = 0
                self._last_event_flush = now

    def record_events(self, events: Sequence[SecurityEvent]) -> None:
        with self._lock:
            for event in events:
                self._events_bucket.append(event.timestamp)
                self._counts[event.severity.value] += 1
                self._counts[event.event_type.value] += 1
                self._events_since_flush += 1
                if event.event_type is EventType.PERSON_ENTERED:
                    self._entered += 1
                elif event.event_type is EventType.PERSON_EXITED:
                    self._exited += 1

    def snapshot(
        self, *, people: int, vehicles: int, others: int, tracks: int,
        fps: float, timings: Timings,
    ) -> AnalyticsSnapshot:
        with self._lock:
            midnight = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
            events_today = sum(1 for ts in self._events_bucket if ts >= midnight)
            return AnalyticsSnapshot(
                people=people, vehicles=vehicles, others=others, tracks=tracks,
                total_objects=people + vehicles + others,
                peak_people=self._peak_people,
                events_today=events_today,
                warnings=self._counts.get(Severity.WARNING.value, 0),
                criticals=self._counts.get(Severity.CRITICAL.value, 0),
                infos=self._counts.get(Severity.INFO.value, 0),
                entered=self._entered, exited=self._exited,
                fps=fps, latency_ms=timings.total_ms,
                detect_ms=timings.detect_ms, track_ms=timings.track_ms,
                uptime_s=time.time() - self._started,
                class_histogram=tuple(self._class_histogram.most_common(8)),
            )

    def series_values(self, name: str) -> List[float]:
        """Copy a series' values for painting (never hand the deque to the GUI)."""
        with self._lock:
            series = getattr(self, f"{name}_series", None)
            if series is None:
                return []
            return [v for _t, v in series]


# =============================================================================
# SECTION 12 - DATABASE
# =============================================================================

class EventDatabase:
    """Local SQLite store for events, sessions and periodic statistics.

    Entirely local - the product requires no cloud backend.  Writes arrive from
    the inference thread while reads come from the GUI thread, so the single
    connection is opened with ``check_same_thread=False`` and every statement is
    serialised through one lock.
    """

    SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id    TEXT PRIMARY KEY,
            ts          REAL NOT NULL,
            iso_ts      TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            severity    TEXT NOT NULL,
            track_id    INTEGER,
            zone        TEXT,
            description TEXT NOT NULL,
            mode        TEXT,
            source      TEXT,
            session_id  TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            started_at  REAL NOT NULL,
            ended_at    REAL,
            mode        TEXT,
            source      TEXT,
            backend     TEXT,
            device      TEXT,
            frames      INTEGER DEFAULT 0,
            events      INTEGER DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS statistics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            ts          REAL NOT NULL,
            fps         REAL,
            people      INTEGER,
            vehicles    INTEGER,
            tracks      INTEGER,
            latency_ms  REAL,
            cpu_percent REAL,
            ram_mb      REAL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)",
        "CREATE INDEX IF NOT EXISTS idx_events_sev ON events(severity)",
        "CREATE INDEX IF NOT EXISTS idx_stats_session ON statistics(session_id)",
    )

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self.session_id: str = uuid.uuid4().hex[:12]
        self.available = False
        self.error: Optional[str] = None
        self._open()

    def _open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.path), check_same_thread=False, timeout=5.0
            )
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                for statement in self.SCHEMA:
                    self._conn.execute(statement)
                self._conn.commit()
            self.available = True
            LOGGER.info("Event database ready at %s", self.path)
        except sqlite3.Error as exc:
            self.error = str(exc)
            self.available = False
            self._conn = None
            LOGGER.error("SQLite unavailable (%s) - running without persistence", exc)

    # -- sessions -------------------------------------------------------------
    def start_session(self, mode: str, source: str, backend: str, device: str) -> str:
        self.session_id = uuid.uuid4().hex[:12]
        if not self.available or self._conn is None:
            return self.session_id
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO sessions (session_id, started_at, mode, source, "
                    "backend, device) VALUES (?, ?, ?, ?, ?, ?)",
                    (self.session_id, time.time(), mode, source, backend, device),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.error("start_session failed: %s", exc)
        return self.session_id

    def end_session(self, frames: int, events: int) -> None:
        if not self.available or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE sessions SET ended_at=?, frames=?, events=? "
                    "WHERE session_id=?",
                    (time.time(), int(frames), int(events), self.session_id),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.error("end_session failed: %s", exc)

    # -- events ---------------------------------------------------------------
    def insert_events(self, events: Sequence[SecurityEvent]) -> None:
        if not events or not self.available or self._conn is None:
            return
        rows = [event.as_row() + (self.session_id,) for event in events]
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO events (event_id, ts, iso_ts, event_type, "
                    "severity, track_id, zone, description, mode, source, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.error("insert_events failed: %s", exc)

    def record_statistics(
        self, *, fps: float, people: int, vehicles: int, tracks: int,
        latency_ms: float, cpu_percent: float, ram_mb: float,
    ) -> None:
        if not self.available or self._conn is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO statistics (session_id, ts, fps, people, vehicles, "
                    "tracks, latency_ms, cpu_percent, ram_mb) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.session_id, time.time(), float(fps), int(people),
                     int(vehicles), int(tracks), float(latency_ms),
                     float(cpu_percent), float(ram_mb)),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            LOGGER.debug("record_statistics failed: %s", exc)

    # -- queries --------------------------------------------------------------
    def query_events(
        self, limit: int = 500, severity: Optional[str] = None,
        event_type: Optional[str] = None, since: Optional[float] = None,
        search: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        if not self.available or self._conn is None:
            return []
        sql = "SELECT * FROM events WHERE 1=1"
        params: List[Any] = []
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        if since is not None:
            sql += " AND ts >= ?"
            params.append(float(since))
        if search:
            sql += " AND (description LIKE ? OR zone LIKE ?)"
            like = f"%{search}%"
            params.extend([like, like])
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        try:
            with self._lock:
                return list(self._conn.execute(sql, params).fetchall())
        except sqlite3.Error as exc:
            LOGGER.error("query_events failed: %s", exc)
            return []

    def event_counts_by_severity(self, since: Optional[float] = None) -> Dict[str, int]:
        if not self.available or self._conn is None:
            return {}
        sql = "SELECT severity, COUNT(*) AS n FROM events"
        params: List[Any] = []
        if since is not None:
            sql += " WHERE ts >= ?"
            params.append(float(since))
        sql += " GROUP BY severity"
        try:
            with self._lock:
                return {row["severity"]: int(row["n"])
                        for row in self._conn.execute(sql, params).fetchall()}
        except sqlite3.Error as exc:
            LOGGER.error("event_counts_by_severity failed: %s", exc)
            return {}

    def recent_sessions(self, limit: int = 25) -> List[sqlite3.Row]:
        if not self.available or self._conn is None:
            return []
        try:
            with self._lock:
                return list(self._conn.execute(
                    "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall())
        except sqlite3.Error as exc:
            LOGGER.error("recent_sessions failed: %s", exc)
            return []

    def clear_events(self) -> int:
        if not self.available or self._conn is None:
            return 0
        try:
            with self._lock:
                cursor = self._conn.execute("DELETE FROM events")
                self._conn.commit()
                return int(cursor.rowcount or 0)
        except sqlite3.Error as exc:
            LOGGER.error("clear_events failed: %s", exc)
            return 0

    # -- export ---------------------------------------------------------------
    def export_csv(self, destination: Path, rows: Optional[Sequence[sqlite3.Row]] = None) -> int:
        """Write event history to CSV.  Returns the number of exported rows."""
        if rows is None:
            rows = self.query_events(limit=100000)
        destination = Path(destination)
        columns = ["iso_ts", "event_type", "severity", "track_id", "zone",
                   "description", "mode", "source", "session_id"]
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([c.upper() for c in columns])
                for row in rows:
                    writer.writerow([row[c] if c in row.keys() else "" for c in columns])
            return len(rows)
        except OSError as exc:
            raise RuntimeError(f"CSV export failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                    LOGGER.info("Event database closed")
                except sqlite3.Error as exc:
                    LOGGER.debug("DB close error: %s", exc)
                finally:
                    self._conn = None
                    self.available = False


# =============================================================================
# SECTION 13 - WORKER THREADS & PERFORMANCE MONITOR
# =============================================================================

@dataclass(slots=True)
class SystemTelemetry:
    """Host telemetry for the Edge System panel."""

    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_percent: float = 0.0
    process_mb: float = 0.0
    gpu_name: Optional[str] = None
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_util_percent: Optional[float] = None
    device: str = "cpu"
    available: bool = False


class PerformanceMonitor:
    """Samples host CPU / RAM / GPU telemetry on a cheap cadence."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = float(interval_s)
        self._last_sample = 0.0
        self._cached = SystemTelemetry()
        self._psutil = DEP_PSUTIL.module if DEP_PSUTIL.available else None
        self._process: Optional[Any] = None
        if self._psutil is not None:
            try:
                self._process = self._psutil.Process(os.getpid())
                self._psutil.cpu_percent(interval=None)   # prime the counter
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("psutil init failed: %s", exc)
                self._psutil = None

    def sample(self, device: str = "cpu", force: bool = False) -> SystemTelemetry:
        now = time.time()
        if not force and (now - self._last_sample) < self.interval_s:
            return self._cached
        self._last_sample = now

        telemetry = SystemTelemetry(device=device)
        if self._psutil is not None:
            try:
                telemetry.cpu_percent = float(self._psutil.cpu_percent(interval=None))
                memory = self._psutil.virtual_memory()
                telemetry.ram_total_mb = memory.total / (1024 ** 2)
                telemetry.ram_used_mb = (memory.total - memory.available) / (1024 ** 2)
                telemetry.ram_percent = float(memory.percent)
                if self._process is not None:
                    telemetry.process_mb = self._process.memory_info().rss / (1024 ** 2)
                telemetry.available = True
            except Exception as exc:  # noqa: BLE001 - telemetry is best effort
                LOGGER.debug("psutil sample failed: %s", exc)

        torch = DEP_TORCH.module
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    telemetry.gpu_name = torch.cuda.get_device_name(0)
                    free_b, total_b = torch.cuda.mem_get_info()
                    telemetry.gpu_mem_total_mb = total_b / (1024 ** 2)
                    telemetry.gpu_mem_used_mb = (total_b - free_b) / (1024 ** 2)
                    try:
                        telemetry.gpu_util_percent = float(torch.cuda.utilization())
                    except Exception:  # noqa: BLE001 - needs pynvml on some builds
                        telemetry.gpu_util_percent = None
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("CUDA telemetry failed: %s", exc)

        self._cached = telemetry
        return telemetry


class FrameQueue:
    """Bounded frame queue with a stale-frame drop policy.

    When inference is slower than capture, the newest frame is what matters; old
    frames are discarded rather than queued.  This bounds memory AND latency,
    which is the behaviour an edge appliance needs.
    """

    def __init__(self, maxsize: int = 2) -> None:
        self._queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=max(1, maxsize))
        self.dropped = 0
        self.pushed = 0

    def resize(self, maxsize: int) -> None:
        maxsize = max(1, int(maxsize))
        if maxsize == self._queue.maxsize:
            return
        replacement: "queue.Queue[FramePacket]" = queue.Queue(maxsize=maxsize)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                replacement.put_nowait(item)
            except queue.Full:
                self.dropped += 1
        self._queue = replacement

    def put(self, packet: FramePacket) -> None:
        """Enqueue, evicting the oldest frame when full."""
        self.pushed += 1
        try:
            self._queue.put_nowait(packet)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()          # evict oldest
            self.dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            self.dropped += 1

    def get(self, timeout: float = 0.25) -> Optional[FramePacket]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self, timeout: float = 0.25) -> Optional[FramePacket]:
        """Block for one frame, then drain to the newest available packet."""
        packet = self.get(timeout=timeout)
        if packet is None:
            return None
        while True:
            try:
                newer = self._queue.get_nowait()
            except queue.Empty:
                return packet
            self.dropped += 1
            packet = newer

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize


if QT_AVAILABLE:

    class CaptureWorker(QThread):
        """Producer thread: pulls frames from the active source into the queue.

        Owns the source exclusively.  All control operations (pause, seek,
        restart) are applied inside the thread loop via simple flags, so the GUI
        thread never touches a cv2.VideoCapture.
        """

        sourceOpened = pyqtSignal(object)     # SourceInfo
        sourceFailed = pyqtSignal(str)
        sourceEnded = pyqtSignal()
        statsUpdated = pyqtSignal(float, int)  # capture fps, frame index

        def __init__(self, frame_queue: FrameQueue, parent: Optional[QObject] = None) -> None:
            super().__init__(parent)
            self.setObjectName("CaptureWorker")
            self._queue = frame_queue
            self._source: Optional[BaseVideoSource] = None
            self._running = False
            self._paused = False
            self._target_fps = 25.0
            self._frame_skip = 0
            self._seek_request: Optional[int] = None
            self._restart_request = False
            self._mutex = threading.Lock()
            self._capture_fps = 0.0
            self._fps_window: Deque[float] = deque(maxlen=30)

        # -- control (called from the GUI thread) -----------------------------
        def set_source(self, source: BaseVideoSource) -> None:
            with self._mutex:
                self._source = source

        def set_target_fps(self, fps: float) -> None:
            self._target_fps = float(max(1.0, min(120.0, fps)))

        def set_frame_skip(self, skip: int) -> None:
            self._frame_skip = max(0, int(skip))

        def pause(self, paused: bool) -> None:
            self._paused = bool(paused)

        @property
        def is_paused(self) -> bool:
            return self._paused

        def request_seek(self, frame_index: int) -> None:
            self._seek_request = int(frame_index)

        def request_restart(self) -> None:
            self._restart_request = True

        def stop(self) -> None:
            self._running = False

        @property
        def capture_fps(self) -> float:
            return self._capture_fps

        @property
        def source_info(self) -> Optional[SourceInfo]:
            source = self._source
            return source.info if source is not None else None

        # -- thread body ------------------------------------------------------
        def run(self) -> None:  # noqa: C901 - a capture loop is inherently branchy
            with self._mutex:
                source = self._source
            if source is None:
                self.sourceFailed.emit("No video source configured.")
                return

            try:
                source.open()
            except SourceError as exc:
                self.sourceFailed.emit(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - unexpected backend errors
                LOGGER.exception("Unexpected source error")
                self.sourceFailed.emit(f"Unexpected source error: {exc}")
                return

            self.sourceOpened.emit(source.info)
            self._running = True
            frame_counter = 0
            last_emit = 0.0
            failure_streak = 0

            try:
                while self._running:
                    loop_start = time.perf_counter()

                    if self._paused:
                        self.msleep(40)
                        continue

                    if self._restart_request:
                        self._restart_request = False
                        source.restart()
                        self._queue.clear()

                    if self._seek_request is not None:
                        target, self._seek_request = self._seek_request, None
                        source.seek(target)
                        self._queue.clear()

                    try:
                        packet = source.read()
                    except SourceError as exc:
                        self.sourceFailed.emit(str(exc))
                        break
                    except Exception as exc:  # noqa: BLE001 - one bad frame must
                        # never take the application down.
                        LOGGER.warning("Frame read error: %s", exc)
                        failure_streak += 1
                        if failure_streak > 50:
                            self.sourceFailed.emit(f"Capture aborted: {exc}")
                            break
                        self.msleep(20)
                        continue

                    if packet is None:
                        failure_streak += 1
                        if source.info.kind is SourceKind.FILE and failure_streak > 3:
                            self.sourceEnded.emit()
                            break
                        if failure_streak > 150:
                            self.sourceFailed.emit("Source stopped delivering frames.")
                            break
                        self.msleep(10)
                        continue

                    failure_streak = 0
                    frame_counter += 1

                    # Frame skipping: a cheap throughput lever for slow hardware.
                    if self._frame_skip > 0 and (frame_counter % (self._frame_skip + 1)) != 0:
                        continue

                    self._queue.put(packet)

                    # Capture FPS over a rolling window.
                    now = time.perf_counter()
                    self._fps_window.append(now)
                    if len(self._fps_window) >= 2:
                        span = self._fps_window[-1] - self._fps_window[0]
                        if span > 0:
                            self._capture_fps = (len(self._fps_window) - 1) / span
                    if now - last_emit > 0.5:
                        last_emit = now
                        self.statsUpdated.emit(self._capture_fps, source.position)

                    # FPS limiter keeps a fast file source from saturating the CPU.
                    budget = 1.0 / self._target_fps
                    elapsed = time.perf_counter() - loop_start
                    remaining = budget - elapsed
                    if remaining > 0:
                        self.msleep(int(remaining * 1000))
            finally:
                # Releasing the capture here guarantees the device is freed even
                # if the loop exits through an exception path.
                try:
                    source.release()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Source release error: %s", exc)
                LOGGER.info("Capture worker finished")


    class InferenceWorker(QThread):
        """Consumer thread: detect -> track -> zones/events -> annotate.

        This is where every expensive operation lives.  The GUI thread only ever
        receives finished FrameResult objects through a queued signal.
        """

        resultReady = pyqtSignal(object)          # FrameResult
        eventsRaised = pyqtSignal(object)         # List[SecurityEvent]
        modelStateChanged = pyqtSignal(object, str)   # ModelState, message
        errorRaised = pyqtSignal(str)

        def __init__(
            self, frame_queue: FrameQueue, settings: AppSettings,
            analytics: AnalyticsEngine, database: EventDatabase,
            parent: Optional[QObject] = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("InferenceWorker")
            self._queue = frame_queue
            self._settings = settings
            self._analytics = analytics
            self._database = database

            self._detector: BaseDetector = NullDetector(settings)
            self._tracker = KalmanTracker(
                settings.max_age, settings.min_hits, settings.iou_threshold
            )
            self._events = EventEngine(settings)
            self._adapter = SupervisionAdapter()
            self._renderer = FrameRenderer(self._adapter)
            self._monitor = PerformanceMonitor()

            self._zones: List[ZoneConfig] = []
            self._lines: List[LineConfig] = []
            self._running = False
            self._mutex = threading.RLock()
            self._backend_request: Optional[Tuple[BackendKind, Optional[str]]] = None
            self._unload_request = False
            self._reset_request = False

            self._fps_window: Deque[float] = deque(maxlen=30)
            self._pipeline_fps = 0.0
            self._frames_processed = 0
            self._events_emitted = 0
            self._inference_failures = 0
            self._last_stat_write = 0.0

        # -- introspection ----------------------------------------------------
        @property
        def pipeline_fps(self) -> float:
            return self._pipeline_fps

        @property
        def frames_processed(self) -> int:
            return self._frames_processed

        @property
        def events_emitted(self) -> int:
            return self._events_emitted

        @property
        def detector(self) -> BaseDetector:
            return self._detector

        @property
        def telemetry(self) -> SystemTelemetry:
            return self._monitor.sample(self._detector.device)

        # -- control ----------------------------------------------------------
        def configure_zones(
            self, zones: Sequence[ZoneConfig], lines: Sequence[LineConfig],
            source_name: str = "",
        ) -> None:
            with self._mutex:
                self._zones = list(zones)
                self._lines = list(lines)
                self._events.configure(self._zones, self._lines, source_name)

        def apply_settings(self, settings: AppSettings) -> None:
            """Hot-apply settings.  Threshold / overlay changes take effect on
            the next frame; a backend change is queued for the worker loop."""
            with self._mutex:
                self._settings = settings
                self._detector.update_settings(settings)
                self._events.update_settings(settings)
                self._tracker.configure(
                    settings.max_age, settings.min_hits, settings.iou_threshold
                )

        def request_backend(self, backend: Optional[BackendKind],
                            variant: Optional[str] = None) -> None:
            """Ask the worker to (re)load a detection backend, or unload with None.

            ``variant`` names the checkpoint / size class to load; the worker
            applies it on its own thread so the GUI never blocks on a download.
            """
            with self._mutex:
                if backend is None:
                    self._unload_request = True
                    self._backend_request = None
                else:
                    self._backend_request = (backend, variant)
                    self._unload_request = False

        def request_reset(self) -> None:
            self._reset_request = True

        def stop(self) -> None:
            self._running = False

        # -- backend management ----------------------------------------------
        def _swap_backend(self, backend: BackendKind,
                          variant: Optional[str] = None) -> None:
            """Load a new detector, keeping the old one until the new one is up.

            Model switching happens without restarting the application; if the
            new backend fails to load, the pipeline falls back to NullDetector
            and keeps running (simulation stays fully functional).
            """
            old = self._detector
            self.modelStateChanged.emit(ModelState.LOADING, f"Loading {backend.value} ...")
            detector = create_detector(backend, self._settings, variant)
            try:
                detector.load_model()
            except DetectorError as exc:
                LOGGER.error("Backend load failed: %s", exc)
                self.modelStateChanged.emit(detector.state, str(exc))
                self.errorRaised.emit(str(exc))
                with self._mutex:
                    self._detector = NullDetector(self._settings)
                    self._detector.load_model()
                try:
                    old.unload_model()
                except Exception:  # noqa: BLE001
                    pass
                return
            except Exception as exc:  # noqa: BLE001 - defensive
                LOGGER.exception("Unexpected backend error")
                self.modelStateChanged.emit(ModelState.ERROR, str(exc))
                self.errorRaised.emit(f"Unexpected model error: {exc}")
                return

            with self._mutex:
                self._detector = detector
            try:
                old.unload_model()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Old detector unload error: %s", exc)
            self._tracker.reset()
            self.modelStateChanged.emit(detector.state, detector.status_message)

        def _unload_backend(self) -> None:
            with self._mutex:
                old = self._detector
                self._detector = NullDetector(self._settings)
                self._detector.load_model()
            try:
                old.unload_model()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Unload error: %s", exc)
            self.modelStateChanged.emit(ModelState.UNLOADED, "No detection backend active")

        # -- thread body ------------------------------------------------------
        def run(self) -> None:
            self._running = True
            LOGGER.info("Inference worker started")
            while self._running:
                # Service pending control requests before touching a frame.
                if self._backend_request is not None:
                    backend, variant = self._backend_request
                    self._backend_request = None
                    self._swap_backend(backend, variant)
                if self._unload_request:
                    self._unload_request = False
                    self._unload_backend()
                if self._reset_request:
                    self._reset_request = False
                    self._tracker.reset()
                    self._events.reset()

                packet = self._queue.get_latest(timeout=0.2)
                if packet is None:
                    continue
                try:
                    result = self._process_packet(packet)
                except Exception as exc:  # noqa: BLE001 - a single bad frame must
                    # never terminate the pipeline or the GUI.
                    LOGGER.exception("Frame processing error")
                    self._inference_failures += 1
                    if self._inference_failures in (1, 10, 50):
                        self.errorRaised.emit(
                            f"Frame processing error ({self._inference_failures}): {exc}"
                        )
                    continue
                if result is not None:
                    self.resultReady.emit(result)
            LOGGER.info("Inference worker finished")

        def _process_packet(self, packet: FramePacket) -> Optional[FrameResult]:
            frame = packet.frame
            if frame is None or frame.size == 0 or frame.ndim != 3:
                return None   # invalid frame: skip silently

            with self._mutex:
                settings = self._settings
                detector = self._detector
                zones = list(self._zones)
                lines = list(self._lines)

            timings = Timings()
            height, width = frame.shape[:2]

            # ---- Detection ---------------------------------------------------
            t0 = time.perf_counter()
            detections: List[Detection] = []
            synthetic_used = False
            use_model = detector.is_ready and not isinstance(detector, NullDetector)
            if packet.synthetic_detections is not None and not (
                use_model and settings.detect_on_simulation
            ):
                # Simulation ground truth: the path that makes the product
                # demonstrable with no model weights installed.
                detections = [
                    d for d in packet.synthetic_detections
                    if d.confidence >= settings.confidence
                ]
                if settings.person_classes_only:
                    detections = [d for d in detections if d.is_person]
                synthetic_used = True
            elif use_model:
                try:
                    detections = detector.predict(frame)
                except DetectorError as exc:
                    self._inference_failures += 1
                    if self._inference_failures in (1, 5, 25):
                        self.errorRaised.emit(str(exc))
                    detections = []
            timings.detect_ms = (time.perf_counter() - t0) * 1000.0

            # ---- Tracking ----------------------------------------------------
            t0 = time.perf_counter()
            if settings.tracking_enabled:
                tracks = self._tracker.update(detections)
                lost = self._tracker.pop_lost()
            else:
                # Tracking disabled: surface detections as ephemeral pseudo-tracks
                # so the renderer and counters keep working.
                tracks = [
                    TrackedObject(
                        track_id=-1 - i, bbox=d.bbox, class_id=d.class_id,
                        class_name=d.class_name, confidence=d.confidence,
                        position=d.center, velocity=(0.0, 0.0), age=1, hits=1,
                        time_since_update=0, phase=TrackPhase.CONFIRMED,
                        last_seen=packet.timestamp,
                    )
                    for i, d in enumerate(detections)
                ]
                lost = []
            timings.track_ms = (time.perf_counter() - t0) * 1000.0

            # ---- Zones / events ----------------------------------------------
            t0 = time.perf_counter()
            if settings.tracking_enabled:
                events, zone_status, alert_ids = self._events.process(
                    tracks, lost, width, height
                )
            else:
                events, zone_status, alert_ids = [], [], set()
            timings.event_ms = (time.perf_counter() - t0) * 1000.0

            people = sum(1 for t in tracks if t.is_person)
            vehicles = sum(1 for t in tracks if t.is_vehicle)
            others = len(tracks) - people - vehicles

            # ---- Annotation ---------------------------------------------------
            t0 = time.perf_counter()
            canvas = frame.copy()   # never mutate the captured frame in place
            if settings.show_zones and zones:
                status_map = {z.zone_id: z for z in zone_status}
                self._renderer.draw_zones(canvas, zones, status_map)
                self._renderer.draw_lines(canvas, lines, self._events.line_counts)
            self._renderer.draw_tracks(canvas, tracks, settings, alert_ids)
            self._renderer.draw_hud(
                canvas, fps=self._pipeline_fps, mode=settings.mode,
                source=packet.source_name, detector=detector.display_name,
                people=people, vehicles=vehicles, tracks=len(tracks),
                synthetic=packet.synthetic, settings=settings,
                critical=sum(1 for e in events if e.severity is Severity.CRITICAL),
            )
            timings.annotate_ms = (time.perf_counter() - t0) * 1000.0
            # Queue latency: wall-clock age of the frame minus the time this
            # thread spent on it.  packet.timestamp is time.time(), so the
            # comparison must use the same clock (not perf_counter).
            timings.capture_ms = max(
                0.0, (time.time() - packet.timestamp) * 1000.0 - timings.total_ms
            )

            # ---- Bookkeeping --------------------------------------------------
            now = time.perf_counter()
            self._fps_window.append(now)
            if len(self._fps_window) >= 2:
                span = self._fps_window[-1] - self._fps_window[0]
                if span > 0:
                    self._pipeline_fps = (len(self._fps_window) - 1) / span
            self._frames_processed += 1

            if events:
                self._events_emitted += len(events)
                self._database.insert_events(events)
                self._analytics.record_events(events)
                self.eventsRaised.emit(events)

            self._analytics.record_frame(
                people=people, vehicles=vehicles, others=others, tracks=len(tracks),
                fps=self._pipeline_fps, timings=timings,
                class_names=[t.class_name for t in tracks],
            )

            # Periodic statistics row (every 5s) for the sessions history.
            if time.time() - self._last_stat_write > 5.0:
                self._last_stat_write = time.time()
                telemetry = self._monitor.sample(detector.device)
                self._database.record_statistics(
                    fps=self._pipeline_fps, people=people, vehicles=vehicles,
                    tracks=len(tracks), latency_ms=timings.total_ms,
                    cpu_percent=telemetry.cpu_percent, ram_mb=telemetry.process_mb,
                )

            total_in, total_out = self._events.total_in_out
            return FrameResult(
                frame=canvas, index=packet.index, timestamp=packet.timestamp,
                detections=detections, tracks=tracks, events=events,
                zone_status=zone_status, people=people, vehicles=vehicles,
                others=others, timings=timings, source_name=packet.source_name,
                detector_label=("Synthetic GT" if synthetic_used
                                else detector.display_name),
                synthetic_detections=synthetic_used,
                line_in=total_in, line_out=total_out,
                occupancy=people,
            )

        def shutdown(self) -> None:
            """Release model resources from the worker's own context."""
            try:
                self._detector.unload_model()
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("Detector shutdown error: %s", exc)


# =============================================================================
# SECTION 14 - CUSTOM QT WIDGETS
# =============================================================================

if QT_AVAILABLE:

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


# =============================================================================
# SECTION 15 - DASHBOARD
# =============================================================================

if QT_AVAILABLE:

    class DashboardPage(QWidget):
        """Operations overview: KPI tiles, trend charts and the edge panel."""

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)

            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(14)

            # -- header ------------------------------------------------------
            head = QHBoxLayout()
            title_box = QVBoxLayout()
            title_box.setSpacing(1)
            title = QLabel("Operations Dashboard")
            title.setObjectName("H1")
            self._subtitle = QLabel("Live edge analytics")
            self._subtitle.setObjectName("Dim")
            title_box.addWidget(title)
            title_box.addWidget(self._subtitle)
            head.addLayout(title_box)
            head.addStretch(1)
            self._mode_pill = StatusPill("CLASSROOM", Palette.ACCENT)
            self._state_pill = StatusPill("IDLE", Palette.TEXT_DIM)
            head.addWidget(self._mode_pill)
            head.addWidget(self._state_pill)
            layout.addLayout(head)

            # -- KPI tiles ----------------------------------------------------
            self.cards: Dict[str, StatCard] = {}
            grid = QGridLayout()
            grid.setSpacing(10)
            definitions = [
                ("fps", "Pipeline FPS", Palette.ACCENT),
                ("latency", "Total Latency", Palette.CYAN),
                ("people", "People", Palette.OK),
                ("vehicles", "Vehicles", Palette.WARN),
                ("tracks", "Active Tracks", Palette.VIOLET),
                ("objects", "Total Objects", Palette.ACCENT),
                ("events", "Events Today", Palette.CYAN),
                ("warnings", "Warnings", Palette.WARN),
                ("criticals", "Critical Alerts", Palette.CRIT),
                ("detect", "Detect Latency", Palette.MAGENTA),
                ("track_ms", "Track Latency", Palette.VIOLET),
                ("device", "Inference Device", Palette.OK),
            ]
            for index, (key, caption, accent) in enumerate(definitions):
                widget = StatCard(caption, "-", "", accent)
                self.cards[key] = widget
                grid.addWidget(widget, index // 6, index % 6)
            for column in range(6):
                grid.setColumnStretch(column, 1)
            layout.addLayout(grid)

            # -- trend charts --------------------------------------------------
            charts_frame, charts_layout = card("Trends")
            charts_grid = QGridLayout()
            charts_grid.setSpacing(10)
            self.chart_objects = Sparkline("Object count over time", Palette.ACCENT)
            self.chart_fps = Sparkline("FPS over time", Palette.OK, unit=" fps")
            self.chart_events = Sparkline("Events over time", Palette.WARN, unit="/s")
            self.chart_latency = Sparkline("Latency over time", Palette.CYAN, unit=" ms")
            charts_grid.addWidget(self.chart_objects, 0, 0)
            charts_grid.addWidget(self.chart_fps, 0, 1)
            charts_grid.addWidget(self.chart_events, 1, 0)
            charts_grid.addWidget(self.chart_latency, 1, 1)
            charts_grid.setColumnStretch(0, 1)
            charts_grid.setColumnStretch(1, 1)
            charts_layout.addLayout(charts_grid)
            layout.addWidget(charts_frame)

            # -- edge system panel ---------------------------------------------
            edge_frame, edge_layout = card("Edge System")
            self.pipeline = PipelineDiagram()
            edge_layout.addWidget(self.pipeline)

            meters = QGridLayout()
            meters.setSpacing(12)
            self.meter_cpu = BarMeter("CPU", Palette.ACCENT)
            self.meter_ram = BarMeter("System RAM", Palette.VIOLET)
            self.meter_gpu = BarMeter("GPU Memory", Palette.OK)
            self.meter_queue = BarMeter("Frame Queue", Palette.CYAN)
            meters.addWidget(self.meter_cpu, 0, 0)
            meters.addWidget(self.meter_ram, 0, 1)
            meters.addWidget(self.meter_gpu, 1, 0)
            meters.addWidget(self.meter_queue, 1, 1)
            meters.setColumnStretch(0, 1)
            meters.setColumnStretch(1, 1)
            edge_layout.addLayout(meters)

            self._edge_detail = QLabel("-")
            self._edge_detail.setObjectName("Mono")
            self._edge_detail.setWordWrap(True)
            self._edge_detail.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
            edge_layout.addWidget(self._edge_detail)
            layout.addWidget(edge_frame)
            layout.addStretch(1)

            root.addWidget(make_scroll(content))

        # -- updates -----------------------------------------------------------
        def update_metrics(
            self, snapshot: AnalyticsSnapshot, telemetry: SystemTelemetry,
            mode: AppMode, state: PipelineState, queue_fill: Tuple[int, int],
            dropped: int, detector_label: str,
        ) -> None:
            self.cards["fps"].set_value(f"{snapshot.fps:.1f}", "frames / second")
            self.cards["latency"].set_value(f"{snapshot.latency_ms:.0f} ms", "end-to-end")
            self.cards["people"].set_value(snapshot.people, f"peak {snapshot.peak_people}")
            self.cards["vehicles"].set_value(snapshot.vehicles, "in view")
            self.cards["tracks"].set_value(snapshot.tracks, "kalman tracks")
            self.cards["objects"].set_value(snapshot.total_objects, "all classes")
            self.cards["events"].set_value(snapshot.events_today, "since midnight")
            self.cards["warnings"].set_value(snapshot.warnings, "this session")
            self.cards["criticals"].set_value(snapshot.criticals, "this session")
            self.cards["detect"].set_value(f"{snapshot.detect_ms:.0f} ms", detector_label)
            self.cards["track_ms"].set_value(f"{snapshot.track_ms:.1f} ms", "association")
            device_text = "CUDA" if telemetry.device == "cuda" else "CPU"
            self.cards["device"].set_value(
                device_text, telemetry.gpu_name or platform.processor() or platform.machine()
            )

            self._mode_pill.set_status(mode.value.upper(), Palette.ACCENT)
            state_colors = {
                PipelineState.RUNNING: Palette.OK,
                PipelineState.PAUSED: Palette.WARN,
                PipelineState.ERROR: Palette.CRIT,
            }
            self._state_pill.set_status(
                state.value.upper(), state_colors.get(state, Palette.TEXT_DIM)
            )

            # Edge meters.
            if telemetry.available:
                self.meter_cpu.set_value(telemetry.cpu_percent / 100.0,
                                         f"{telemetry.cpu_percent:.0f}%")
                self.meter_ram.set_value(
                    telemetry.ram_percent / 100.0,
                    f"{telemetry.ram_used_mb / 1024:.1f} / "
                    f"{telemetry.ram_total_mb / 1024:.1f} GB",
                )
            else:
                self.meter_cpu.set_value(0.0, "psutil not installed")
                self.meter_ram.set_value(0.0, "psutil not installed")

            if telemetry.gpu_mem_total_mb > 0:
                self.meter_gpu.set_value(
                    telemetry.gpu_mem_used_mb / telemetry.gpu_mem_total_mb,
                    f"{telemetry.gpu_mem_used_mb:.0f} / {telemetry.gpu_mem_total_mb:.0f} MB",
                )
            else:
                self.meter_gpu.set_value(0.0, "no CUDA device")

            used, capacity = queue_fill
            self.meter_queue.set_value(
                used / max(1, capacity), f"{used} / {capacity}  ({dropped} dropped)"
            )

            active = -1 if state is not PipelineState.RUNNING else 6
            self.pipeline.set_state(
                active,
                {
                    "Capture": f"{snapshot.fps:.0f} fps",
                    "Detector": detector_label,
                    "Kalman Tracker": f"{snapshot.tracks} trk",
                    "Event Engine": f"{snapshot.warnings + snapshot.criticals} alerts",
                    "Dashboard": f"{snapshot.latency_ms:.0f} ms",
                },
                healthy=state is not PipelineState.ERROR,
            )

            gpu_util = ("n/a" if telemetry.gpu_util_percent is None
                        else f"{telemetry.gpu_util_percent:.0f}%")
            self._edge_detail.setText(
                f"device={telemetry.device}   gpu_util={gpu_util}   "
                f"process_rss={telemetry.process_mb:.0f} MB   "
                f"uptime={timedelta(seconds=int(snapshot.uptime_s))}   "
                f"in={snapshot.entered} out={snapshot.exited}"
            )

        def update_charts(self, analytics: AnalyticsEngine) -> None:
            self.chart_objects.set_values(analytics.series_values("people"))
            self.chart_fps.set_values(analytics.series_values("fps"))
            self.chart_events.set_values(analytics.series_values("event"))
            self.chart_latency.set_values(analytics.series_values("latency"))

        def set_subtitle(self, text: str) -> None:
            self._subtitle.setText(text)


# =============================================================================
# SECTION 16 - MAIN WINDOW
# =============================================================================

if QT_AVAILABLE:

    class SettingsManager:
        """Persists AppSettings and zone/line layouts between runs."""

        def __init__(self) -> None:
            self._qsettings = QSettings(ORG_NAME, APP_NAME)

        def load_settings(self) -> AppSettings:
            raw = self._qsettings.value("settings")
            if isinstance(raw, str) and raw:
                try:
                    return AppSettings.from_dict(json.loads(raw))
                except (json.JSONDecodeError, TypeError) as exc:
                    LOGGER.warning("Corrupt settings ignored: %s", exc)
            return AppSettings()

        def save_settings(self, settings: AppSettings) -> None:
            try:
                self._qsettings.setValue("settings", json.dumps(settings.to_dict()))
                self._qsettings.sync()
            except Exception as exc:  # noqa: BLE001 - never block shutdown
                LOGGER.warning("Could not persist settings: %s", exc)

        @property
        def skip_dependency_screen(self) -> bool:
            return str(self._qsettings.value("skip_dependency_screen", "false")).lower() == "true"

        @skip_dependency_screen.setter
        def skip_dependency_screen(self, value: bool) -> None:
            self._qsettings.setValue("skip_dependency_screen", "true" if value else "false")

        def save_geometry(self, window: QMainWindow) -> None:
            self._qsettings.setValue("geometry", window.saveGeometry())
            self._qsettings.setValue("windowState", window.saveState())

        def restore_geometry(self, window: QMainWindow) -> None:
            geometry = self._qsettings.value("geometry")
            state = self._qsettings.value("windowState")
            try:
                if geometry:
                    window.restoreGeometry(geometry)
                if state:
                    window.restoreState(state)
            except Exception as exc:  # noqa: BLE001 - corrupt blob
                LOGGER.debug("Could not restore window state: %s", exc)

        # -- zones -------------------------------------------------------------
        @staticmethod
        def load_zones(mode: AppMode) -> Tuple[List[ZoneConfig], List[LineConfig]]:
            try:
                if ZONES_PATH.exists():
                    data = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
                    bundle = data.get(mode.value)
                    if bundle:
                        zones = [ZoneConfig.from_dict(z) for z in bundle.get("zones", [])]
                        lines = [LineConfig.from_dict(l) for l in bundle.get("lines", [])]
                        if zones:
                            return zones, (lines or default_lines(mode))
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                LOGGER.warning("Could not load zones (%s); using defaults", exc)
            return default_zones(mode), default_lines(mode)

        @staticmethod
        def save_zones(mode: AppMode, zones: Sequence[ZoneConfig],
                       lines: Sequence[LineConfig]) -> None:
            try:
                data: Dict[str, Any] = {}
                if ZONES_PATH.exists():
                    try:
                        data = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        data = {}
                data[mode.value] = {
                    "zones": [z.to_dict() for z in zones],
                    "lines": [l.to_dict() for l in lines],
                }
                ZONES_PATH.parent.mkdir(parents=True, exist_ok=True)
                ZONES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except OSError as exc:
                LOGGER.error("Could not save zones: %s", exc)

    class MainWindow(QMainWindow):
        """Application shell: top bar, sidebar navigation, pages, docks, status bar."""

        PAGES = ("Dashboard", "Live Monitor", "Video Sources", "Analytics",
                 "Events", "Zones", "Models", "Settings")

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"{APP_NAME} - {APP_SUBTITLE}")
            self.resize(1500, 900)
            self.setMinimumSize(1040, 640)

            # -- core services ------------------------------------------------
            self.settings_manager = SettingsManager()
            self.settings = self.settings_manager.load_settings()
            self.database = EventDatabase()
            self.analytics = AnalyticsEngine()
            self.frame_queue = FrameQueue(self.settings.queue_size)

            self.zones, self.lines = SettingsManager.load_zones(self.settings.mode)
            self.pipeline_state = PipelineState.IDLE
            self._capture: Optional[CaptureWorker] = None
            self._inference: Optional[InferenceWorker] = None
            self._latest_result: Optional[FrameResult] = None
            self._result_lock = threading.Lock()
            self._pending_zone_points: List[Tuple[float, float]] = []
            self._active_source: Optional[BaseVideoSource] = None
            self._source_info: Optional[SourceInfo] = None
            self._model_state = ModelState.UNLOADED
            self._model_message = "No detection backend active"
            # Used for telemetry while no inference worker exists; keeping one
            # instance avoids re-opening a psutil handle on every tick.
            self._idle_monitor = PerformanceMonitor()
            self._shutting_down = False
            self._last_events: Deque[SecurityEvent] = deque(maxlen=400)
            self._session_frames = 0

            # -- UI -------------------------------------------------------------
            self._build_ui()
            self._build_status_bar()
            self._connect_signals()
            self.settings_manager.restore_geometry(self)
            self._apply_settings_to_ui()

            # -- timers ---------------------------------------------------------
            # One repaint timer drives all GUI updates; the workers never touch
            # widgets directly.  This is the GUI-throttling mechanism.
            self._render_timer = QTimer(self)
            self._render_timer.timeout.connect(self._render_tick)
            self._render_timer.start(max(16, int(1000 / max(1, self.settings.gui_fps))))

            self._slow_timer = QTimer(self)
            self._slow_timer.timeout.connect(self._slow_tick)
            self._slow_timer.start(1000)

            self._start_session()
            self._log_system_event(
                f"{APP_NAME} v{APP_VERSION} ready. Simulation source available offline."
            )

        # ------------------------------------------------------------------
        # UI construction
        # ------------------------------------------------------------------
        def _build_ui(self) -> None:
            central = QWidget()
            root = QVBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            root.addWidget(self._build_top_bar())

            body = QWidget()
            body_layout = QHBoxLayout(body)
            body_layout.setContentsMargins(0, 0, 0, 0)
            body_layout.setSpacing(0)
            body_layout.addWidget(self._build_sidebar())

            self.stack = QStackedWidget()
            self.page_dashboard = DashboardPage()
            self.page_live = self._build_live_page()
            self.page_sources = self._build_sources_page()
            self.page_analytics = self._build_analytics_page()
            self.page_events = self._build_events_page()
            self.page_zones = self._build_zones_page()
            self.page_models = self._build_models_page()
            self.page_settings = self._build_settings_page()
            for page in (self.page_dashboard, self.page_live, self.page_sources,
                         self.page_analytics, self.page_events, self.page_zones,
                         self.page_models, self.page_settings):
                self.stack.addWidget(page)
            body_layout.addWidget(self.stack, 1)
            root.addWidget(body, 1)
            self.setCentralWidget(central)

            self._build_right_dock()

        def _build_top_bar(self) -> QFrame:
            bar = QFrame()
            bar.setObjectName("TopBar")
            bar.setFixedHeight(56)
            layout = QHBoxLayout(bar)
            layout.setContentsMargins(16, 6, 16, 6)
            layout.setSpacing(14)

            logo = QLabel()
            logo.setFixedSize(30, 30)
            logo.setStyleSheet(
                f"background-color: {Palette.ACCENT}; border-radius: 8px;"
                f" color: #04101C; font-weight: 900; font-size: 15px;"
            )
            logo.setText("L")
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo)

            brand = QVBoxLayout()
            brand.setSpacing(0)
            name = QLabel(APP_NAME.upper())
            name.setObjectName("Brand")
            sub = QLabel(APP_SUBTITLE)
            sub.setObjectName("BrandSub")
            brand.addWidget(name)
            brand.addWidget(sub)
            layout.addLayout(brand)
            layout.addSpacing(10)
            layout.addWidget(divider(vertical=True))
            layout.addStretch(1)

            self.pill_system = StatusPill("SYSTEM IDLE", Palette.TEXT_DIM)
            self.pill_device = StatusPill("CPU", Palette.VIOLET)
            self.pill_model = StatusPill("NO MODEL", Palette.WARN)
            self.pill_fps = StatusPill("0.0 FPS", Palette.ACCENT)
            self.pill_source = StatusPill("NO SOURCE", Palette.TEXT_DIM)
            for pill in (self.pill_system, self.pill_device, self.pill_model,
                         self.pill_fps, self.pill_source):
                layout.addWidget(pill)

            layout.addWidget(divider(vertical=True))
            self.label_clock = QLabel("--:--:--")
            self.label_clock.setObjectName("Mono")
            self.label_clock.setStyleSheet(
                f"color: {Palette.TEXT}; font-size: 13px; font-weight: 600;"
            )
            layout.addWidget(self.label_clock)
            return bar

        def _build_sidebar(self) -> QWidget:
            container = QFrame()
            container.setObjectName("Sidebar")
            container.setFixedWidth(186)
            outer = QVBoxLayout(container)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            inner = QWidget()
            layout = QVBoxLayout(inner)
            layout.setContentsMargins(11, 14, 11, 14)
            layout.setSpacing(4)

            caption = QLabel("NAVIGATION")
            caption.setObjectName("Caption")
            layout.addWidget(caption)
            layout.addSpacing(4)

            self.nav_group = QButtonGroup(self)
            self.nav_group.setExclusive(True)
            icons = ("▦", "▶", "⛁", "◔", "⚑", "⬡", "◈", "⚙")
            for index, (title, icon) in enumerate(zip(self.PAGES, icons)):
                button = QPushButton(f"  {icon}   {title}")
                button.setObjectName("NavButton")
                button.setCheckable(True)
                button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                self.nav_group.addButton(button, index)
                layout.addWidget(button)
            self.nav_group.button(0).setChecked(True)

            layout.addSpacing(12)
            layout.addWidget(divider())
            layout.addSpacing(10)

            caption2 = QLabel("SESSION CONTROL")
            caption2.setObjectName("Caption")
            layout.addWidget(caption2)

            self.btn_start = QPushButton("Start Pipeline")
            self.btn_start.setObjectName("Primary")
            self.btn_pause = QPushButton("Pause")
            self.btn_stop = QPushButton("Stop")
            self.btn_stop.setObjectName("Danger")
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            for button in (self.btn_start, self.btn_pause, self.btn_stop):
                layout.addWidget(button)

            layout.addStretch(1)
            self.label_sidebar_note = QLabel(
                "Detection & tracking only.\nNo facial recognition."
            )
            self.label_sidebar_note.setWordWrap(True)
            self.label_sidebar_note.setStyleSheet(
                f"color: {Palette.TEXT_FAINT}; font-size: 9px;"
            )
            layout.addWidget(self.label_sidebar_note)

            version = QLabel(f"v{APP_VERSION}")
            version.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 9px;")
            layout.addWidget(version)

            outer.addWidget(make_scroll(inner))
            return container

        # -- live monitor page --------------------------------------------------
        def _build_live_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(10)

            header = QHBoxLayout()
            title = QLabel("Live Monitor")
            title.setObjectName("H1")
            header.addWidget(title)
            header.addStretch(1)

            self.combo_quick_source = QComboBox()
            self.combo_quick_source.addItems([k.value for k in SourceKind])
            self.combo_quick_mode = QComboBox()
            self.combo_quick_mode.addItems([m.value for m in AppMode])
            for label, widget in (("Source", self.combo_quick_source),
                                  ("Mode", self.combo_quick_mode)):
                caption = QLabel(label)
                caption.setObjectName("Caption")
                header.addWidget(caption)
                header.addWidget(widget)

            self.btn_snapshot = QPushButton("Snapshot")
            self.btn_snapshot.setObjectName("Ghost")
            header.addWidget(self.btn_snapshot)
            layout.addLayout(header)

            self.video_view = VideoView()
            self.video_view.show_placeholder(
                "NO SIGNAL", "Press Start Pipeline to begin monitoring"
            )

            video_frame = QFrame()
            video_frame.setObjectName("Panel")
            video_layout = QVBoxLayout(video_frame)
            video_layout.setContentsMargins(8, 8, 8, 8)
            video_layout.addWidget(self.video_view)
            layout.addWidget(video_frame, 1)

            # Transport controls.
            controls = QFrame()
            controls.setObjectName("Card")
            controls_layout = QHBoxLayout(controls)
            controls_layout.setContentsMargins(12, 8, 12, 8)
            controls_layout.setSpacing(10)

            self.btn_play = QPushButton("Pause")
            self.btn_restart = QPushButton("Restart")
            self.btn_play.setEnabled(False)
            self.btn_restart.setEnabled(False)
            controls_layout.addWidget(self.btn_play)
            controls_layout.addWidget(self.btn_restart)

            self.slider_seek = QSlider(Qt.Orientation.Horizontal)
            self.slider_seek.setEnabled(False)
            self.slider_seek.setRange(0, 0)
            controls_layout.addWidget(self.slider_seek, 1)
            self.label_position = QLabel("-- / --")
            self.label_position.setObjectName("Mono")
            controls_layout.addWidget(self.label_position)

            controls_layout.addWidget(divider(vertical=True))
            zoom_caption = QLabel("ZOOM")
            zoom_caption.setObjectName("Caption")
            controls_layout.addWidget(zoom_caption)
            self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
            self.slider_zoom.setRange(100, 500)
            self.slider_zoom.setValue(100)
            self.slider_zoom.setFixedWidth(110)
            controls_layout.addWidget(self.slider_zoom)

            controls_layout.addWidget(divider(vertical=True))
            conf_caption = QLabel("CONF")
            conf_caption.setObjectName("Caption")
            controls_layout.addWidget(conf_caption)
            self.slider_conf_live = QSlider(Qt.Orientation.Horizontal)
            self.slider_conf_live.setRange(10, 100)
            self.slider_conf_live.setValue(int(self.settings.confidence * 100))
            self.slider_conf_live.setFixedWidth(110)
            controls_layout.addWidget(self.slider_conf_live)
            self.label_conf_live = QLabel(f"{self.settings.confidence:.2f}")
            self.label_conf_live.setObjectName("Mono")
            controls_layout.addWidget(self.label_conf_live)

            layout.addWidget(controls)
            return page

        # -- right dock ---------------------------------------------------------
        def _build_right_dock(self) -> None:
            dock = QDockWidget("LIVE ANALYTICS", self)
            dock.setObjectName("RightDock")
            dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                                 | Qt.DockWidgetArea.LeftDockWidgetArea)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )

            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(11, 11, 11, 11)
            layout.setSpacing(11)

            # Active alerts.
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

            # Live counts.
            counts_frame, counts_layout = card("Live Counts")
            counts_grid = QGridLayout()
            counts_grid.setSpacing(8)
            self.panel_cards = {
                "people": StatCard("People", "0", "", Palette.OK),
                "vehicles": StatCard("Vehicles", "0", "", Palette.WARN),
                "tracks": StatCard("Tracks", "0", "", Palette.VIOLET),
                "occupancy": StatCard("In / Out", "0 / 0", "", Palette.CYAN),
            }
            for index, widget in enumerate(self.panel_cards.values()):
                counts_grid.addWidget(widget, index // 2, index % 2)
            counts_layout.addLayout(counts_grid)
            layout.addWidget(counts_frame)

            # Zone status.
            zone_frame, zone_layout = card("Zone Status")
            self.list_zone_status = QListWidget()
            self.list_zone_status.setMinimumHeight(120)
            self.list_zone_status.setMaximumHeight(190)
            zone_layout.addWidget(self.list_zone_status)
            layout.addWidget(zone_frame)

            # Recent events.
            events_frame, events_layout = card("Recent Events")
            self.table_recent = EventTable(compact=True)
            self.table_recent.setMinimumHeight(170)
            events_layout.addWidget(self.table_recent)
            layout.addWidget(events_frame, 1)

            dock.setWidget(make_scroll(panel, horizontal=True))
            dock.setMinimumWidth(250)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self.dock_right = dock

        # -- video sources page --------------------------------------------------
        def _build_sources_page(self) -> QWidget:
            page = QWidget()
            root = QVBoxLayout(page)
            root.setContentsMargins(0, 0, 0, 0)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(14)

            title = QLabel("Video Sources")
            title.setObjectName("H1")
            layout.addWidget(title)

            # Simulation.
            sim_frame, sim_layout = card("Simulation")
            sim_desc = QLabel(
                "Built-in synthetic CCTV scene with ground-truth annotations. "
                "Requires no model weights and no external files - the fastest way "
                "to validate zones, tracking and alerting."
            )
            sim_desc.setWordWrap(True)
            sim_desc.setStyleSheet(f"color: {Palette.TEXT_DIM};")
            sim_layout.addWidget(sim_desc)

            sim_controls = QHBoxLayout()
            sim_controls.setSpacing(10)
            people_caption = QLabel("People")
            people_caption.setObjectName("Caption")
            sim_controls.addWidget(people_caption)
            self.spin_sim_people = QSpinBox()
            self.spin_sim_people.setRange(1, 20)
            self.spin_sim_people.setValue(10)
            sim_controls.addWidget(self.spin_sim_people)

            vehicle_caption = QLabel("Vehicles")
            vehicle_caption.setObjectName("Caption")
            sim_controls.addWidget(vehicle_caption)
            self.spin_sim_vehicles = QSpinBox()
            self.spin_sim_vehicles.setRange(0, 6)
            self.spin_sim_vehicles.setValue(2)
            sim_controls.addWidget(self.spin_sim_vehicles)

            self.btn_use_sim = QPushButton("Use Simulation")
            self.btn_use_sim.setObjectName("Primary")
            sim_controls.addWidget(self.btn_use_sim)
            sim_controls.addStretch(1)
            sim_layout.addLayout(sim_controls)
            layout.addWidget(sim_frame)

            # Local file.
            file_frame, file_layout = card("Local Video File")
            file_row = QHBoxLayout()
            self.edit_file_path = QLineEdit()
            self.edit_file_path.setPlaceholderText(
                "Select an MP4, AVI, MOV or MKV file ..."
            )
            self.edit_file_path.setReadOnly(True)
            file_row.addWidget(self.edit_file_path, 1)
            self.btn_browse = QPushButton("Browse ...")
            file_row.addWidget(self.btn_browse)
            self.btn_use_file = QPushButton("Use File")
            self.btn_use_file.setObjectName("Primary")
            self.btn_use_file.setEnabled(False)
            file_row.addWidget(self.btn_use_file)
            file_layout.addLayout(file_row)

            self.label_file_info = QLabel("No file selected")
            self.label_file_info.setObjectName("Mono")
            self.label_file_info.setStyleSheet(f"color: {Palette.TEXT_FAINT};")
            file_layout.addWidget(self.label_file_info)

            self.check_loop = QCheckBox("Loop playback when the file ends")
            self.check_loop.setChecked(True)
            file_layout.addWidget(self.check_loop)
            layout.addWidget(file_frame)

            # Webcam.
            cam_frame, cam_layout = card("Webcam / Capture Device")
            cam_row = QHBoxLayout()
            device_caption = QLabel("Device")
            device_caption.setObjectName("Caption")
            cam_row.addWidget(device_caption)
            self.combo_camera = QComboBox()
            self.combo_camera.addItems([f"Camera {i}" for i in range(4)])
            cam_row.addWidget(self.combo_camera)
            self.btn_scan_cameras = QPushButton("Scan Devices")
            cam_row.addWidget(self.btn_scan_cameras)
            self.btn_use_camera = QPushButton("Use Camera")
            self.btn_use_camera.setObjectName("Primary")
            cam_row.addWidget(self.btn_use_camera)
            cam_row.addStretch(1)
            cam_layout.addLayout(cam_row)
            self.label_camera_info = QLabel(
                "Scanning probes camera indices 0-5 and may take a few seconds."
            )
            self.label_camera_info.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
            cam_layout.addWidget(self.label_camera_info)
            layout.addWidget(cam_frame)

            # Active source readout.
            active_frame, active_layout = card("Active Source")
            self.label_active_source = QLabel("None")
            self.label_active_source.setObjectName("Mono")
            self.label_active_source.setWordWrap(True)
            active_layout.addWidget(self.label_active_source)
            layout.addWidget(active_frame)

            layout.addStretch(1)
            root.addWidget(make_scroll(content))
            return page

        # -- analytics page ------------------------------------------------------
        def _build_analytics_page(self) -> QWidget:
            page = QWidget()
            root = QVBoxLayout(page)
            root.setContentsMargins(0, 0, 0, 0)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(14)

            title = QLabel("Analytics")
            title.setObjectName("H1")
            layout.addWidget(title)

            charts_frame, charts_layout = card("Time Series")
            grid = QGridLayout()
            grid.setSpacing(10)
            self.an_chart_people = Sparkline("People in view", Palette.OK)
            self.an_chart_tracks = Sparkline("Active tracks", Palette.VIOLET)
            self.an_chart_vehicles = Sparkline("Vehicles", Palette.WARN)
            self.an_chart_latency = Sparkline("Pipeline latency", Palette.CYAN, unit=" ms")
            for index, chart in enumerate((self.an_chart_people, self.an_chart_tracks,
                                           self.an_chart_vehicles, self.an_chart_latency)):
                chart.setMinimumHeight(110)
                grid.addWidget(chart, index // 2, index % 2)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            charts_layout.addLayout(grid)
            layout.addWidget(charts_frame)

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
            self.btn_refresh_sessions = QPushButton("Refresh")
            self.btn_refresh_sessions.setObjectName("Ghost")
            history_layout.addWidget(self.btn_refresh_sessions,
                                     alignment=Qt.AlignmentFlag.AlignRight)
            layout.addWidget(history_frame)
            layout.addStretch(1)

            root.addWidget(make_scroll(content))
            return page

        # -- events page ----------------------------------------------------------
        def _build_events_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(12)

            header = QHBoxLayout()
            title = QLabel("Event History")
            title.setObjectName("H1")
            header.addWidget(title)
            header.addStretch(1)

            self.combo_filter_severity = QComboBox()
            self.combo_filter_severity.addItem("All severities", None)
            for severity in Severity:
                self.combo_filter_severity.addItem(severity.value, severity.value)
            self.combo_filter_type = QComboBox()
            self.combo_filter_type.addItem("All types", None)
            for event_type in EventType:
                self.combo_filter_type.addItem(event_type.value, event_type.value)
            self.edit_search = QLineEdit()
            self.edit_search.setPlaceholderText("Search description or zone ...")
            self.edit_search.setFixedWidth(230)

            self.btn_refresh_events = QPushButton("Refresh")
            self.btn_export_csv = QPushButton("Export CSV")
            self.btn_export_csv.setObjectName("Primary")
            self.btn_clear_events = QPushButton("Clear History")
            self.btn_clear_events.setObjectName("Danger")

            for widget in (self.combo_filter_severity, self.combo_filter_type,
                           self.edit_search, self.btn_refresh_events,
                           self.btn_export_csv, self.btn_clear_events):
                header.addWidget(widget)
            layout.addLayout(header)

            counts = QHBoxLayout()
            counts.setSpacing(10)
            self.event_cards = {
                "total": StatCard("Total Events", "0", "stored", Palette.ACCENT),
                "info": StatCard("Info", "0", "", Palette.ACCENT),
                "warning": StatCard("Warnings", "0", "", Palette.WARN),
                "critical": StatCard("Critical", "0", "", Palette.CRIT),
            }
            for widget in self.event_cards.values():
                counts.addWidget(widget)
            counts.addStretch(1)
            layout.addLayout(counts)

            self.table_events = EventTable(compact=False)
            layout.addWidget(self.table_events, 1)

            self.label_db_status = QLabel()
            self.label_db_status.setStyleSheet(
                f"color: {Palette.TEXT_FAINT}; font-size: 10px;"
            )
            self.label_db_status.setText(
                f"SQLite: {DB_PATH}" if self.database.available
                else f"SQLite unavailable ({self.database.error}) - events are not persisted"
            )
            layout.addWidget(self.label_db_status)
            return page

        # -- zones page ------------------------------------------------------------
        def _build_zones_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(12)

            header = QHBoxLayout()
            title = QLabel("Zone Configuration")
            title.setObjectName("H1")
            header.addWidget(title)
            header.addStretch(1)
            self.btn_zone_reset = QPushButton("Restore Defaults")
            self.btn_zone_save = QPushButton("Save Layout")
            self.btn_zone_save.setObjectName("Primary")
            header.addWidget(self.btn_zone_reset)
            header.addWidget(self.btn_zone_save)
            layout.addLayout(header)

            splitter = QSplitter(Qt.Orientation.Horizontal)

            # Left: editable zone table.
            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 7, 0)
            left_layout.setSpacing(10)

            table_frame, table_layout = card("Zones")
            self.table_zones = QTableWidget(0, 6)
            self.table_zones.setHorizontalHeaderLabels(
                ["ENABLED", "NAME", "TYPE", "SEVERITY", "MAX OCC.", "POINTS"]
            )
            self.table_zones.verticalHeader().setVisible(False)
            zone_header = self.table_zones.horizontalHeader()
            zone_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            zone_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            zone_header.setStretchLastSection(True)
            self.table_zones.setMinimumHeight(240)
            table_layout.addWidget(self.table_zones)

            zone_buttons = QHBoxLayout()
            self.btn_zone_delete = QPushButton("Delete Selected")
            self.btn_zone_delete.setObjectName("Danger")
            zone_buttons.addWidget(self.btn_zone_delete)
            zone_buttons.addStretch(1)
            table_layout.addLayout(zone_buttons)
            left_layout.addWidget(table_frame)

            draw_frame, draw_layout = card("Draw New Zone")
            draw_desc = QLabel(
                "Enable drawing, then click on the preview to place polygon "
                "vertices. Three or more points are required."
            )
            draw_desc.setWordWrap(True)
            draw_desc.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")
            draw_layout.addWidget(draw_desc)

            form = QGridLayout()
            form.setSpacing(8)
            name_caption = QLabel("Name")
            name_caption.setObjectName("Caption")
            form.addWidget(name_caption, 0, 0)
            self.edit_zone_name = QLineEdit("New Zone")
            form.addWidget(self.edit_zone_name, 0, 1)
            type_caption = QLabel("Type")
            type_caption.setObjectName("Caption")
            form.addWidget(type_caption, 1, 0)
            self.combo_zone_kind = QComboBox()
            self.combo_zone_kind.addItems([k.value for k in ZoneKind])
            form.addWidget(self.combo_zone_kind, 1, 1)
            occ_caption = QLabel("Max occupancy")
            occ_caption.setObjectName("Caption")
            form.addWidget(occ_caption, 2, 0)
            self.spin_zone_occupancy = QSpinBox()
            self.spin_zone_occupancy.setRange(0, 200)
            self.spin_zone_occupancy.setSpecialValueText("disabled")
            form.addWidget(self.spin_zone_occupancy, 2, 1)
            form.setColumnStretch(1, 1)
            draw_layout.addLayout(form)

            draw_buttons = QHBoxLayout()
            self.btn_zone_draw = QPushButton("Start Drawing")
            self.btn_zone_draw.setCheckable(True)
            self.btn_zone_undo = QPushButton("Undo Point")
            self.btn_zone_commit = QPushButton("Create Zone")
            self.btn_zone_commit.setObjectName("Primary")
            self.btn_zone_commit.setEnabled(False)
            for button in (self.btn_zone_draw, self.btn_zone_undo, self.btn_zone_commit):
                draw_buttons.addWidget(button)
            draw_layout.addLayout(draw_buttons)
            left_layout.addWidget(draw_frame)
            left_layout.addStretch(1)
            splitter.addWidget(make_scroll(left))

            # Right: live preview that doubles as the drawing canvas.
            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(7, 0, 0, 0)
            preview_frame = QFrame()
            preview_frame.setObjectName("Panel")
            preview_layout = QVBoxLayout(preview_frame)
            preview_layout.setContentsMargins(8, 8, 8, 8)
            caption = QLabel("PREVIEW / DRAWING CANVAS")
            caption.setObjectName("Caption")
            preview_layout.addWidget(caption)
            self.zone_view = VideoView()
            self.zone_view.show_placeholder(
                "PREVIEW", "Start the pipeline to draw zones over the live feed"
            )
            preview_layout.addWidget(self.zone_view, 1)
            right_layout.addWidget(preview_frame, 1)
            splitter.addWidget(right)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([520, 760])
            layout.addWidget(splitter, 1)
            return page

        # -- models page ------------------------------------------------------------
        def _build_models_page(self) -> QWidget:
            page = QWidget()
            root = QVBoxLayout(page)
            root.setContentsMargins(0, 0, 0, 0)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(14)

            title = QLabel("Detection Models")
            title.setObjectName("H1")
            layout.addWidget(title)

            status_frame, status_layout = card("Active Backend")
            self.label_model_state = QLabel("No detection backend active")
            self.label_model_state.setObjectName("H2")
            status_layout.addWidget(self.label_model_state)
            self.label_model_detail = QLabel(
                "Simulation mode supplies synthetic ground-truth detections, so "
                "the pipeline runs without a model."
            )
            self.label_model_detail.setWordWrap(True)
            self.label_model_detail.setStyleSheet(f"color: {Palette.TEXT_DIM};")
            status_layout.addWidget(self.label_model_detail)

            actions = QHBoxLayout()
            backend_caption = QLabel("Backend")
            backend_caption.setObjectName("Caption")
            actions.addWidget(backend_caption)
            self.combo_backend = QComboBox()
            self.combo_backend.addItems([b.value for b in BackendKind])
            actions.addWidget(self.combo_backend)

            variant_caption = QLabel("Checkpoint")
            variant_caption.setObjectName("Caption")
            actions.addWidget(variant_caption)
            self.combo_checkpoint = QComboBox()
            actions.addWidget(self.combo_checkpoint, 1)

            self.btn_load_model = QPushButton("Load Model")
            self.btn_load_model.setObjectName("Primary")
            self.btn_unload_model = QPushButton("Unload")
            actions.addWidget(self.btn_load_model)
            actions.addWidget(self.btn_unload_model)
            status_layout.addLayout(actions)
            layout.addWidget(status_frame)

            avail_frame, avail_layout = card("Backend Availability")
            self.table_models = QTableWidget(0, 4)
            self.table_models.setHorizontalHeaderLabels(
                ["BACKEND", "STATUS", "VERSION", "DETAIL"]
            )
            self.table_models.verticalHeader().setVisible(False)
            self.table_models.horizontalHeader().setSectionResizeMode(
                3, QHeaderView.ResizeMode.Stretch
            )
            self.table_models.setMinimumHeight(140)
            avail_layout.addWidget(self.table_models)
            self.btn_recheck_models = QPushButton("Re-check")
            self.btn_recheck_models.setObjectName("Ghost")
            avail_layout.addWidget(self.btn_recheck_models,
                                   alignment=Qt.AlignmentFlag.AlignRight)
            layout.addWidget(avail_frame)

            future_frame, future_layout = card("Future Modules (not implemented)")
            future_note = QLabel(
                "The modules below are NOT part of this build and are listed only "
                "to document the roadmap. This application ships object detection "
                "and tracking only; it cannot infer any of the following, and no "
                "part of the UI reports them:\n\n"
                "   -  PPE / helmet / vest compliance  (requires a PPE-trained model)\n"
                "   -  Fall and slip detection         (requires a pose / action model)\n"
                "   -  Weapon detection                (requires a dedicated model)\n"
                "   -  Fatigue or drowsiness           (requires a pose / face model)\n\n"
                "Facial recognition and identity inference are deliberately out of "
                "scope and will not be added."
            )
            future_note.setWordWrap(True)
            future_note.setStyleSheet(
                f"color: {Palette.TEXT_FAINT}; font-size: 11px;"
            )
            future_layout.addWidget(future_note)
            layout.addWidget(future_frame)
            layout.addStretch(1)

            root.addWidget(make_scroll(content))
            return page

        # -- settings page -----------------------------------------------------------
        def _build_settings_page(self) -> QWidget:
            page = QWidget()
            root = QVBoxLayout(page)
            root.setContentsMargins(0, 0, 0, 0)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(18, 16, 18, 18)
            layout.setSpacing(14)

            title = QLabel("Settings")
            title.setObjectName("H1")
            layout.addWidget(title)

            columns = QHBoxLayout()
            columns.setSpacing(14)
            left = QVBoxLayout()
            left.setSpacing(14)
            right = QVBoxLayout()
            right.setSpacing(14)

            # Inference.
            inf_frame, inf_layout = card("Inference")
            # Each settings group gets its own two-column grid; the helper keeps
            # a per-grid row counter so groups can be built independently.
            rows: Dict[int, int] = {}

            def add_row(grid: QGridLayout, label: str, widget: QWidget) -> None:
                row = rows.get(id(grid), 0)
                caption = QLabel(label)
                caption.setObjectName("Caption")
                grid.addWidget(caption, row, 0)
                grid.addWidget(widget, row, 1)
                grid.setColumnStretch(1, 1)
                rows[id(grid)] = row + 1

            form = QGridLayout()
            form.setSpacing(9)

            self.combo_set_backend = QComboBox()
            self.combo_set_backend.addItems([b.value for b in BackendKind])
            add_row(form, "Detection model", self.combo_set_backend)

            conf_widget = QWidget()
            conf_layout = QHBoxLayout(conf_widget)
            conf_layout.setContentsMargins(0, 0, 0, 0)
            self.slider_conf = QSlider(Qt.Orientation.Horizontal)
            self.slider_conf.setRange(10, 100)
            self.label_conf = QLabel("0.35")
            self.label_conf.setObjectName("Mono")
            self.label_conf.setFixedWidth(38)
            conf_layout.addWidget(self.slider_conf, 1)
            conf_layout.addWidget(self.label_conf)
            add_row(form, "Confidence", conf_widget)

            self.combo_resolution = QComboBox()
            self.combo_resolution.addItems(["320", "480", "640", "800"])
            add_row(form, "Inference resolution", self.combo_resolution)

            self.combo_device = QComboBox()
            self.combo_device.addItems([d.value for d in DevicePref])
            add_row(form, "Device", self.combo_device)

            self.check_half = QCheckBox("Use half precision when CUDA supports it")
            add_row(form, "Precision", self.check_half)

            self.check_detect_sim = QCheckBox(
                "Run the model on simulated frames instead of synthetic ground truth"
            )
            add_row(form, "Simulation", self.check_detect_sim)

            self.check_person_only = QCheckBox("Person class only")
            add_row(form, "Class filter", self.check_person_only)
            inf_layout.addLayout(form)
            left.addWidget(inf_frame)

            # Tracking.
            track_frame, track_layout = card("Tracking")
            track_form = QGridLayout()
            track_form.setSpacing(9)

            self.check_tracking = QCheckBox("Enable Kalman tracking")
            add_row(track_form, "Tracking", self.check_tracking)
            self.spin_max_age = QSpinBox()
            self.spin_max_age.setRange(1, 200)
            add_row(track_form, "Max coast frames", self.spin_max_age)
            self.spin_min_hits = QSpinBox()
            self.spin_min_hits.setRange(1, 30)
            add_row(track_form, "Confirm after hits", self.spin_min_hits)
            self.spin_iou = QDoubleSpinBox()
            self.spin_iou.setRange(0.05, 0.9)
            self.spin_iou.setSingleStep(0.05)
            add_row(track_form, "Association IoU", self.spin_iou)
            track_layout.addLayout(track_form)
            left.addWidget(track_frame)

            # Performance.
            perf_frame, perf_layout = card("Performance")
            perf_form = QGridLayout()
            perf_form.setSpacing(9)

            self.spin_target_fps = QSpinBox()
            self.spin_target_fps.setRange(1, 120)
            add_row(perf_form, "Capture FPS limit", self.spin_target_fps)
            self.spin_frame_skip = QSpinBox()
            self.spin_frame_skip.setRange(0, 10)
            add_row(perf_form, "Frame skip", self.spin_frame_skip)
            self.spin_queue = QSpinBox()
            self.spin_queue.setRange(1, 10)
            add_row(perf_form, "Frame queue size", self.spin_queue)
            self.spin_gui_fps = QSpinBox()
            self.spin_gui_fps.setRange(5, 60)
            add_row(perf_form, "GUI refresh rate", self.spin_gui_fps)
            perf_layout.addLayout(perf_form)
            right.addWidget(perf_frame)

            # Overlays.
            overlay_frame, overlay_layout = card("Overlays")
            self.check_boxes = QCheckBox("Bounding boxes")
            self.check_ids = QCheckBox("Track IDs")
            self.check_zones = QCheckBox("Zones and lines")
            self.check_confidence = QCheckBox("Confidence values")
            self.check_fps_overlay = QCheckBox("FPS overlay")
            self.check_trails = QCheckBox("Motion trails")
            self.check_supervision = QCheckBox("Use supervision annotators when available")
            for widget in (self.check_boxes, self.check_ids, self.check_zones,
                           self.check_confidence, self.check_fps_overlay,
                           self.check_trails, self.check_supervision):
                overlay_layout.addWidget(widget)
            if not DEP_SUPERVISION.available:
                self.check_supervision.setEnabled(False)
                self.check_supervision.setText(
                    "Use supervision annotators (package not installed)"
                )
            right.addWidget(overlay_frame)

            # Scenario and alerts.
            alert_frame, alert_layout = card("Scenario & Alert Thresholds")
            alert_form = QGridLayout()
            alert_form.setSpacing(9)

            self.combo_mode = QComboBox()
            self.combo_mode.addItems([m.value for m in AppMode])
            add_row(alert_form, "Deployment mode", self.combo_mode)
            self.spin_crowding = QSpinBox()
            self.spin_crowding.setRange(1, 200)
            add_row(alert_form, "Crowding threshold", self.spin_crowding)
            self.spin_high_occupancy = QSpinBox()
            self.spin_high_occupancy.setRange(1, 500)
            add_row(alert_form, "High occupancy threshold", self.spin_high_occupancy)
            self.spin_proximity = QSpinBox()
            self.spin_proximity.setRange(5, 400)
            self.spin_proximity.setSuffix(" px")
            add_row(alert_form, "Machine proximity radius", self.spin_proximity)
            alert_layout.addLayout(alert_form)
            right.addWidget(alert_frame)

            columns.addLayout(left, 1)
            columns.addLayout(right, 1)
            layout.addLayout(columns)

            buttons = QHBoxLayout()
            buttons.addStretch(1)
            self.btn_settings_reset = QPushButton("Reset to Defaults")
            self.btn_settings_apply = QPushButton("Apply Settings")
            self.btn_settings_apply.setObjectName("Primary")
            buttons.addWidget(self.btn_settings_reset)
            buttons.addWidget(self.btn_settings_apply)
            layout.addLayout(buttons)

            note = QLabel(
                f"Settings are stored per user. Database: {DB_PATH}    "
                f"Zones: {ZONES_PATH}    Log: {LOG_PATH}"
            )
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {Palette.TEXT_FAINT}; font-size: 10px;")
            layout.addWidget(note)
            layout.addStretch(1)

            root.addWidget(make_scroll(content))
            return page

        def _build_status_bar(self) -> None:
            bar = QStatusBar()
            bar.setSizeGripEnabled(True)
            self.setStatusBar(bar)
            self.status_labels: Dict[str, QLabel] = {}
            fields = ("Source", "Resolution", "FPS", "Device", "Model",
                      "Tracking", "Pipeline")
            for index, field_name in enumerate(fields):
                label = QLabel(f"{field_name}: -")
                label.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")
                self.status_labels[field_name] = label
                bar.addPermanentWidget(label)
                if index < len(fields) - 1:
                    bar.addPermanentWidget(divider(vertical=True))
            self.status_message = QLabel("Ready")
            self.status_message.setStyleSheet(f"color: {Palette.TEXT_DIM}; font-size: 11px;")
            bar.addWidget(self.status_message)

        # ------------------------------------------------------------------
        # Signal wiring
        # ------------------------------------------------------------------
        def _connect_signals(self) -> None:
            self.nav_group.idClicked.connect(self.stack.setCurrentIndex)
            # Sync the other way too, so programmatic page changes (deep links
            # from a status message, command-line start-up) keep the sidebar
            # highlight correct.
            self.stack.currentChanged.connect(self._sync_nav_selection)

            self.btn_start.clicked.connect(self._on_start)
            self.btn_pause.clicked.connect(self._on_toggle_pause)
            self.btn_stop.clicked.connect(self._on_stop)
            self.btn_play.clicked.connect(self._on_toggle_pause)
            self.btn_restart.clicked.connect(self._on_restart)
            self.slider_seek.sliderReleased.connect(self._on_seek)
            self.slider_zoom.valueChanged.connect(
                lambda v: self.video_view.set_zoom(v / 100.0)
            )
            self.btn_snapshot.clicked.connect(self._on_snapshot)
            self.slider_conf_live.valueChanged.connect(self._on_live_confidence)

            self.combo_quick_source.currentTextChanged.connect(self._on_quick_source)
            self.combo_quick_mode.currentTextChanged.connect(self._on_quick_mode)

            # Sources page.
            self.btn_use_sim.clicked.connect(lambda: self._activate_source(SourceKind.SIMULATION))
            self.btn_browse.clicked.connect(self._on_browse)
            self.btn_use_file.clicked.connect(lambda: self._activate_source(SourceKind.FILE))
            self.btn_scan_cameras.clicked.connect(self._on_scan_cameras)
            self.btn_use_camera.clicked.connect(lambda: self._activate_source(SourceKind.WEBCAM))

            # Events page.
            self.btn_refresh_events.clicked.connect(self._reload_events)
            self.btn_export_csv.clicked.connect(self._on_export_csv)
            self.btn_clear_events.clicked.connect(self._on_clear_events)
            self.combo_filter_severity.currentIndexChanged.connect(self._reload_events)
            self.combo_filter_type.currentIndexChanged.connect(self._reload_events)
            self.edit_search.returnPressed.connect(self._reload_events)

            # Analytics page.
            self.btn_refresh_sessions.clicked.connect(self._reload_sessions)

            # Zones page.
            self.btn_zone_draw.toggled.connect(self._on_zone_draw_toggled)
            self.btn_zone_undo.clicked.connect(self._on_zone_undo)
            self.btn_zone_commit.clicked.connect(self._on_zone_commit)
            self.btn_zone_delete.clicked.connect(self._on_zone_delete)
            self.btn_zone_save.clicked.connect(self._on_zone_save)
            self.btn_zone_reset.clicked.connect(self._on_zone_reset)
            self.zone_view.pointClicked.connect(self._on_zone_point)
            self.table_zones.itemChanged.connect(self._on_zone_item_changed)

            # Models page.
            self.combo_backend.currentTextChanged.connect(self._on_backend_combo)
            self.btn_load_model.clicked.connect(self._on_load_model)
            self.btn_unload_model.clicked.connect(self._on_unload_model)
            self.btn_recheck_models.clicked.connect(self._refresh_model_table)

            # Settings page.
            self.slider_conf.valueChanged.connect(
                lambda v: self.label_conf.setText(f"{v / 100:.2f}")
            )
            self.btn_settings_apply.clicked.connect(self._on_apply_settings)
            self.btn_settings_reset.clicked.connect(self._on_reset_settings)

        def _sync_nav_selection(self, index: int) -> None:
            button = self.nav_group.button(index)
            if button is not None and not button.isChecked():
                button.setChecked(True)

        # ------------------------------------------------------------------
        # Settings <-> UI
        # ------------------------------------------------------------------
        def _apply_settings_to_ui(self) -> None:
            """Push the settings model into every widget (without re-triggering)."""
            blockers = [
                self.combo_set_backend, self.slider_conf, self.combo_resolution,
                self.combo_device, self.combo_mode, self.combo_backend,
                self.slider_conf_live, self.combo_quick_mode,
            ]
            for widget in blockers:
                widget.blockSignals(True)

            self.combo_set_backend.setCurrentText(self.settings.backend.value)
            self.combo_backend.setCurrentText(self.settings.backend.value)
            self.slider_conf.setValue(int(self.settings.confidence * 100))
            self.label_conf.setText(f"{self.settings.confidence:.2f}")
            self.slider_conf_live.setValue(int(self.settings.confidence * 100))
            self.label_conf_live.setText(f"{self.settings.confidence:.2f}")
            self.combo_resolution.setCurrentText(str(self.settings.inference_size))
            self.combo_device.setCurrentText(self.settings.device.value)
            self.combo_mode.setCurrentText(self.settings.mode.value)
            self.combo_quick_mode.setCurrentText(self.settings.mode.value)

            for widget in blockers:
                widget.blockSignals(False)

            self.check_half.setChecked(self.settings.half_precision)
            self.check_detect_sim.setChecked(self.settings.detect_on_simulation)
            self.check_person_only.setChecked(self.settings.person_classes_only)
            self.check_tracking.setChecked(self.settings.tracking_enabled)
            self.spin_max_age.setValue(self.settings.max_age)
            self.spin_min_hits.setValue(self.settings.min_hits)
            self.spin_iou.setValue(self.settings.iou_threshold)
            self.spin_target_fps.setValue(self.settings.target_fps)
            self.spin_frame_skip.setValue(self.settings.frame_skip)
            self.spin_queue.setValue(self.settings.queue_size)
            self.spin_gui_fps.setValue(self.settings.gui_fps)
            self.check_boxes.setChecked(self.settings.show_boxes)
            self.check_ids.setChecked(self.settings.show_track_ids)
            self.check_zones.setChecked(self.settings.show_zones)
            self.check_confidence.setChecked(self.settings.show_confidence)
            self.check_fps_overlay.setChecked(self.settings.show_fps_overlay)
            self.check_trails.setChecked(self.settings.show_trails)
            self.check_supervision.setChecked(
                self.settings.use_supervision_annotators and DEP_SUPERVISION.available
            )
            self.spin_crowding.setValue(self.settings.crowding_threshold)
            self.spin_high_occupancy.setValue(self.settings.high_occupancy_threshold)
            self.spin_proximity.setValue(self.settings.proximity_px)

            self._refresh_checkpoint_combo()
            self._refresh_model_table()
            self._refresh_zone_table()
            self._reload_events()
            self._reload_sessions()

        def _collect_settings(self) -> AppSettings:
            """Read every widget back into a fresh AppSettings instance."""
            settings = AppSettings(
                mode=AppMode(self.combo_mode.currentText()),
                backend=BackendKind(self.combo_set_backend.currentText()),
                device=DevicePref(self.combo_device.currentText()),
                confidence=self.slider_conf.value() / 100.0,
                inference_size=int(self.combo_resolution.currentText()),
                half_precision=self.check_half.isChecked(),
                tracking_enabled=self.check_tracking.isChecked(),
                max_age=self.spin_max_age.value(),
                min_hits=self.spin_min_hits.value(),
                iou_threshold=float(self.spin_iou.value()),
                target_fps=self.spin_target_fps.value(),
                frame_skip=self.spin_frame_skip.value(),
                queue_size=self.spin_queue.value(),
                gui_fps=self.spin_gui_fps.value(),
                show_boxes=self.check_boxes.isChecked(),
                show_track_ids=self.check_ids.isChecked(),
                show_zones=self.check_zones.isChecked(),
                show_confidence=self.check_confidence.isChecked(),
                show_fps_overlay=self.check_fps_overlay.isChecked(),
                show_trails=self.check_trails.isChecked(),
                use_supervision_annotators=self.check_supervision.isChecked(),
                crowding_threshold=self.spin_crowding.value(),
                high_occupancy_threshold=self.spin_high_occupancy.value(),
                proximity_px=self.spin_proximity.value(),
                person_classes_only=self.check_person_only.isChecked(),
                detect_on_simulation=self.check_detect_sim.isChecked(),
            )
            return settings

        def _on_apply_settings(self) -> None:
            previous_mode = self.settings.mode
            self.settings = self._collect_settings()
            self.settings_manager.save_settings(self.settings)

            self.frame_queue.resize(self.settings.queue_size)
            self._render_timer.setInterval(max(16, int(1000 / max(1, self.settings.gui_fps))))

            if self._capture is not None:
                self._capture.set_target_fps(self.settings.target_fps)
                self._capture.set_frame_skip(self.settings.frame_skip)
            if self._inference is not None:
                self._inference.apply_settings(self.settings)

            if previous_mode is not self.settings.mode:
                self._switch_mode(self.settings.mode)

            self._apply_settings_to_ui()
            self._set_status("Settings applied")
            self._log_system_event(
                f"Settings applied (conf={self.settings.confidence:.2f}, "
                f"res={self.settings.inference_size}, device={self.settings.device.value})"
            )

        def _on_reset_settings(self) -> None:
            self.settings = AppSettings()
            self.settings_manager.save_settings(self.settings)
            self._apply_settings_to_ui()
            if self._inference is not None:
                self._inference.apply_settings(self.settings)
            self._set_status("Settings restored to defaults")

        def _on_live_confidence(self, value: int) -> None:
            """Confidence tweaks from the Live Monitor apply immediately."""
            self.settings.confidence = value / 100.0
            self.label_conf_live.setText(f"{self.settings.confidence:.2f}")
            self.slider_conf.blockSignals(True)
            self.slider_conf.setValue(value)
            self.slider_conf.blockSignals(False)
            self.label_conf.setText(f"{self.settings.confidence:.2f}")
            if self._inference is not None:
                self._inference.apply_settings(self.settings)

        def _switch_mode(self, mode: AppMode) -> None:
            """Swap scenario: reload zones and re-seed the simulation scene."""
            self.settings.mode = mode
            self.zones, self.lines = SettingsManager.load_zones(mode)
            self._refresh_zone_table()
            if self._inference is not None:
                self._inference.configure_zones(
                    self.zones, self.lines,
                    self._source_info.name if self._source_info else "",
                )
            if isinstance(self._active_source, SimulationSource):
                self._active_source.set_mode(mode)
            self.combo_quick_mode.blockSignals(True)
            self.combo_quick_mode.setCurrentText(mode.value)
            self.combo_quick_mode.blockSignals(False)
            self.page_dashboard.set_subtitle(
                f"{mode.value} deployment  -  live edge analytics"
            )
            self._log_system_event(f"Deployment mode switched to {mode.value}")

        def _on_quick_mode(self, text: str) -> None:
            try:
                mode = AppMode(text)
            except ValueError:
                return
            if mode is self.settings.mode:
                return
            self.combo_mode.blockSignals(True)
            self.combo_mode.setCurrentText(text)
            self.combo_mode.blockSignals(False)
            self._switch_mode(mode)
            self.settings_manager.save_settings(self.settings)

        def _on_quick_source(self, text: str) -> None:
            try:
                kind = SourceKind(text)
            except ValueError:
                return
            if kind is SourceKind.FILE and not self.edit_file_path.text():
                self._set_status("Choose a video file on the Video Sources page first")
                self.stack.setCurrentIndex(2)
                self.nav_group.button(2).setChecked(True)
                return
            self._activate_source(kind)

        # ------------------------------------------------------------------
        # Pipeline lifecycle
        # ------------------------------------------------------------------
        def _build_source(self, kind: SourceKind) -> Optional[BaseVideoSource]:
            """Instantiate the requested source, reporting problems in the UI."""
            if kind is SourceKind.SIMULATION:
                return SimulationSource(
                    self.settings.mode,
                    people=self.spin_sim_people.value(),
                    vehicles=self.spin_sim_vehicles.value(),
                )
            if kind is SourceKind.FILE:
                path = self.edit_file_path.text().strip()
                if not path:
                    self._error_box("No video file", "Choose a video file first.")
                    return None
                return LocalVideoSource(path, loop=self.check_loop.isChecked())
            if kind is SourceKind.WEBCAM:
                text = self.combo_camera.currentText()
                digits = "".join(ch for ch in text if ch.isdigit())
                return WebcamSource(int(digits) if digits else 0)
            return None

        def _activate_source(self, kind: SourceKind) -> None:
            """Switch the pipeline to a new source (restarting it if running)."""
            was_running = self.pipeline_state in (PipelineState.RUNNING, PipelineState.PAUSED)
            if was_running:
                self._stop_pipeline(update_ui=False)
            source = self._build_source(kind)
            if source is None:
                return
            self._active_source = source
            self.combo_quick_source.blockSignals(True)
            self.combo_quick_source.setCurrentText(kind.value)
            self.combo_quick_source.blockSignals(False)
            self._set_status(f"Source set to {kind.value}")
            self._start_pipeline()

        def _start_pipeline(self) -> None:
            if self.pipeline_state is PipelineState.RUNNING:
                return
            if self._active_source is None:
                self._active_source = self._build_source(SourceKind.SIMULATION)
            if self._active_source is None:
                return

            self.pipeline_state = PipelineState.STARTING
            self.frame_queue = FrameQueue(self.settings.queue_size)
            self.analytics.reset()

            self._capture = CaptureWorker(self.frame_queue, self)
            self._capture.set_source(self._active_source)
            self._capture.set_target_fps(self.settings.target_fps)
            self._capture.set_frame_skip(self.settings.frame_skip)
            self._capture.sourceOpened.connect(self._on_source_opened)
            self._capture.sourceFailed.connect(self._on_source_failed)
            self._capture.sourceEnded.connect(self._on_source_ended)
            self._capture.statsUpdated.connect(self._on_capture_stats)

            self._inference = InferenceWorker(
                self.frame_queue, self.settings, self.analytics, self.database, self
            )
            self._inference.configure_zones(self.zones, self.lines)
            self._inference.resultReady.connect(self._on_result)
            self._inference.eventsRaised.connect(self._on_events)
            self._inference.modelStateChanged.connect(self._on_model_state)
            self._inference.errorRaised.connect(self._on_worker_error)

            self._inference.start()
            self._capture.start()

            # Re-apply the configured backend if one was loaded previously.
            if self._model_state is ModelState.READY:
                self._inference.request_backend(self.settings.backend)

            self.pipeline_state = PipelineState.RUNNING
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_restart.setEnabled(True)
            self.btn_pause.setText("Pause")
            self.btn_play.setText("Pause")
            self.video_view.show_placeholder("STARTING", "Opening video source ...",
                                             Palette.ACCENT)
            self._start_session()
            self._log_system_event("Pipeline started")

        def _stop_pipeline(self, update_ui: bool = True) -> None:
            """Tear the pipeline down, joining both threads deterministically."""
            capture, self._capture = self._capture, None
            inference, self._inference = self._inference, None

            if capture is not None:
                capture.stop()
                if not capture.wait(3000):
                    LOGGER.warning("Capture thread did not stop in time; terminating")
                    capture.terminate()
                    capture.wait(1000)
                capture.deleteLater()

            if inference is not None:
                inference.stop()
                if not inference.wait(4000):
                    LOGGER.warning("Inference thread did not stop in time; terminating")
                    inference.terminate()
                    inference.wait(1000)
                inference.shutdown()
                inference.deleteLater()

            self.frame_queue.clear()
            if self._active_source is not None:
                try:
                    self._active_source.release()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Source release error: %s", exc)

            self.database.end_session(self._session_frames, len(self._last_events))
            self.pipeline_state = PipelineState.STOPPED

            if update_ui:
                self.btn_start.setEnabled(True)
                self.btn_pause.setEnabled(False)
                self.btn_stop.setEnabled(False)
                self.btn_play.setEnabled(False)
                self.btn_restart.setEnabled(False)
                self.slider_seek.setEnabled(False)
                self.video_view.show_placeholder("NO SIGNAL", "Pipeline stopped")
                self.zone_view.show_placeholder("PREVIEW", "Pipeline stopped")
                self._set_status("Pipeline stopped")
                self._log_system_event("Pipeline stopped")

        def _start_session(self) -> None:
            self._session_frames = 0
            source_name = self._source_info.name if self._source_info else "none"
            self.database.start_session(
                self.settings.mode.value, source_name,
                self.settings.backend.value if self._model_state is ModelState.READY else "none",
                resolve_torch_device(self.settings.device),
            )

        # -- control handlers ---------------------------------------------------
        def _on_start(self) -> None:
            self._start_pipeline()

        def _on_stop(self) -> None:
            self._stop_pipeline()

        def _on_toggle_pause(self) -> None:
            if self._capture is None:
                return
            paused = not self._capture.is_paused
            self._capture.pause(paused)
            self.pipeline_state = PipelineState.PAUSED if paused else PipelineState.RUNNING
            text = "Resume" if paused else "Pause"
            self.btn_pause.setText(text)
            self.btn_play.setText(text)
            self.video_view.set_badge("PAUSED" if paused else None, Palette.WARN)
            self._set_status("Pipeline paused" if paused else "Pipeline resumed")

        def _on_restart(self) -> None:
            if self._capture is not None:
                self._capture.request_restart()
            if self._inference is not None:
                self._inference.request_reset()
            self.analytics.reset()
            self._set_status("Source restarted")

        def _on_seek(self) -> None:
            if self._capture is not None and self.slider_seek.isEnabled():
                self._capture.request_seek(self.slider_seek.value())

        def _on_snapshot(self) -> None:
            with self._result_lock:
                result = self._latest_result
            if result is None:
                self._set_status("No frame available to save")
                return
            default = str(Path.home() / f"sentinel_{int(time.time())}.png")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Snapshot", default, "PNG Image (*.png);;JPEG Image (*.jpg)"
            )
            if not path:
                return
            try:
                if not cv2.imwrite(path, result.frame):
                    raise OSError("OpenCV could not write the file")
                self._set_status(f"Snapshot saved to {path}")
            except Exception as exc:  # noqa: BLE001
                self._error_box("Snapshot failed", str(exc))

        # -- worker callbacks ---------------------------------------------------
        @pyqtSlot(object)
        def _on_source_opened(self, info: SourceInfo) -> None:
            self._source_info = info
            if self._inference is not None:
                self._inference.configure_zones(self.zones, self.lines, info.name)
            self.label_active_source.setText(
                f"{info.kind.value}  |  {info.name}\n"
                f"{info.resolution} @ {info.fps:.1f} fps"
                + (f"  |  {info.frame_count} frames "
                   f"({timedelta(seconds=int(info.duration_s))})" if info.frame_count else "")
                + (f"\n{info.detail}" if info.detail else "")
            )
            self.slider_seek.setEnabled(info.seekable)
            if info.seekable:
                self.slider_seek.setRange(0, max(0, info.frame_count - 1))
            # The SIMULATION marker is burned into the frame by the renderer so
            # that it survives snapshots and exports; the Qt overlay badge is
            # reserved for transient UI state such as PAUSED.
            self.video_view.set_badge(None)
            self._set_status(f"Source open: {info.name} ({info.resolution})")
            self._log_system_event(
                f"Source opened: {info.name} {info.resolution} @ {info.fps:.1f} fps"
            )

        @pyqtSlot(str)
        def _on_source_failed(self, message: str) -> None:
            self.pipeline_state = PipelineState.ERROR
            self.video_view.show_placeholder("SOURCE ERROR", message.splitlines()[0],
                                             Palette.CRIT)
            self._error_box("Video source error", message)
            self._log_system_event(f"Source error: {message.splitlines()[0]}",
                                   Severity.CRITICAL)
            self._stop_pipeline()

        @pyqtSlot()
        def _on_source_ended(self) -> None:
            self._set_status("End of video reached")
            self._log_system_event("End of video stream reached")
            self._stop_pipeline()

        @pyqtSlot(float, int)
        def _on_capture_stats(self, fps: float, position: int) -> None:
            if self.slider_seek.isEnabled() and not self.slider_seek.isSliderDown():
                self.slider_seek.blockSignals(True)
                self.slider_seek.setValue(position)
                self.slider_seek.blockSignals(False)
            if self._source_info is not None and self._source_info.frame_count:
                total = self._source_info.frame_count
                self.label_position.setText(f"{position} / {total}")
            else:
                self.label_position.setText(f"frame {position}")

        @pyqtSlot(object)
        def _on_result(self, result: FrameResult) -> None:
            """Store the newest result; painting happens on the render timer.

            Keeping only the latest result means a burst of worker output can
            never queue up repaints and stall the GUI.
            """
            with self._result_lock:
                self._latest_result = result
            self._session_frames = result.index

        @pyqtSlot(object)
        def _on_events(self, events: List[SecurityEvent]) -> None:
            for event in events:
                self._last_events.append(event)
            self.table_recent.add_events(events, limit=120)
            self.table_events.add_events(events, limit=500)

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
                self.label_alert_detail.setText(f"{len(warnings)} warning(s) in this batch")

        @pyqtSlot(object, str)
        def _on_model_state(self, state: ModelState, message: str) -> None:
            self._model_state = state
            self._model_message = message
            self.label_model_state.setText(f"{self.settings.backend.value}: {state.value}")
            self.label_model_detail.setText(message)
            colors = {
                ModelState.READY: Palette.OK,
                ModelState.LOADING: Palette.ACCENT,
                ModelState.ERROR: Palette.CRIT,
                ModelState.UNAVAILABLE: Palette.WARN,
            }
            self.pill_model.set_status(
                f"{self.settings.backend.value} {state.value}".upper()
                if state is not ModelState.UNLOADED else "NO MODEL",
                colors.get(state, Palette.TEXT_DIM),
            )
            if state is ModelState.LOADING:
                self.video_view.show_placeholder(
                    "MODEL LOADING", message, Palette.ACCENT
                )
            self._refresh_model_table()
            self._log_system_event(f"Model {state.value}: {message}")

        @pyqtSlot(str)
        def _on_worker_error(self, message: str) -> None:
            self._set_status(message)
            self._log_system_event(message, Severity.WARNING)

        # ------------------------------------------------------------------
        # Render loop
        # ------------------------------------------------------------------
        def _render_tick(self) -> None:
            """GUI-thread repaint, throttled to the configured refresh rate."""
            with self._result_lock:
                result = self._latest_result
            if result is None:
                return

            page = self.stack.currentIndex()
            # Only paint the widget that is actually visible - repainting a
            # hidden page is pure waste on an edge device.
            if page == 1:
                self.video_view.set_frame(result.frame)
            elif page == 5:
                self.zone_view.set_frame(result.frame)

            self.panel_cards["people"].set_value(result.people, "in view")
            self.panel_cards["vehicles"].set_value(result.vehicles, "in view")
            self.panel_cards["tracks"].set_value(len(result.tracks), "active")
            self.panel_cards["occupancy"].set_value(
                f"{result.line_in} / {result.line_out}", "entries / exits"
            )
            self._refresh_zone_status(result.zone_status)

        def _slow_tick(self) -> None:
            """One-second cadence: clock, dashboard, telemetry, status bar."""
            self.label_clock.setText(datetime.now().strftime("%H:%M:%S"))

            with self._result_lock:
                result = self._latest_result

            fps = self._inference.pipeline_fps if self._inference is not None else 0.0
            timings = result.timings if result is not None else Timings()
            people = result.people if result is not None else 0
            vehicles = result.vehicles if result is not None else 0
            others = result.others if result is not None else 0
            tracks = len(result.tracks) if result is not None else 0

            snapshot = self.analytics.snapshot(
                people=people, vehicles=vehicles, others=others, tracks=tracks,
                fps=fps, timings=timings,
            )
            telemetry = (self._inference.telemetry if self._inference is not None
                         else self._idle_monitor.sample(
                             resolve_torch_device(self.settings.device)))
            detector_label = (result.detector_label if result is not None
                              else "none")

            self.page_dashboard.update_metrics(
                snapshot, telemetry, self.settings.mode, self.pipeline_state,
                (self.frame_queue.size, self.frame_queue.capacity),
                self.frame_queue.dropped, detector_label,
            )
            self.page_dashboard.update_charts(self.analytics)

            # Analytics page charts + summary.
            self.an_chart_people.set_values(self.analytics.series_values("people"))
            self.an_chart_tracks.set_values(self.analytics.series_values("track"))
            self.an_chart_vehicles.set_values(self.analytics.series_values("vehicle"))
            self.an_chart_latency.set_values(self.analytics.series_values("latency"))
            self._refresh_class_table(snapshot)
            self._refresh_summary(snapshot, telemetry)

            # Top bar pills.
            state_colors = {
                PipelineState.RUNNING: Palette.OK,
                PipelineState.PAUSED: Palette.WARN,
                PipelineState.ERROR: Palette.CRIT,
            }
            self.pill_system.set_status(
                f"SYSTEM {self.pipeline_state.value.upper()}",
                state_colors.get(self.pipeline_state, Palette.TEXT_DIM),
            )
            device = telemetry.device.upper()
            self.pill_device.set_status(
                f"{device}" + (f" {telemetry.gpu_util_percent:.0f}%"
                               if telemetry.gpu_util_percent is not None else ""),
                Palette.OK if device == "CUDA" else Palette.VIOLET,
            )
            self.pill_fps.set_status(f"{fps:.1f} FPS",
                                     Palette.OK if fps >= 12 else Palette.WARN)
            if self._source_info is not None:
                self.pill_source.set_status(
                    self._source_info.kind.value.upper(), Palette.ACCENT
                )

            # Status bar.
            info = self._source_info
            self.status_labels["Source"].setText(
                f"Source: {info.name if info else '-'}")
            self.status_labels["Resolution"].setText(
                f"Resolution: {info.resolution if info else '-'}")
            self.status_labels["FPS"].setText(f"FPS: {fps:.1f}")
            self.status_labels["Device"].setText(f"Device: {telemetry.device.upper()}")
            self.status_labels["Model"].setText(f"Model: {detector_label}")
            self.status_labels["Tracking"].setText(
                f"Tracking: {'on' if self.settings.tracking_enabled else 'off'}")
            self.status_labels["Pipeline"].setText(
                f"Pipeline: {self.pipeline_state.value}")

            # Clear stale alert banner after a quiet period.
            if self._last_events:
                age = time.time() - self._last_events[-1].timestamp
                if age > 20:
                    self.label_alert.setText("No active alerts")
                    self.label_alert.setStyleSheet(
                        f"color: {Palette.OK}; font-weight: 600; font-size: 12px;"
                    )
                    self.label_alert_detail.setText("System nominal")

        # ------------------------------------------------------------------
        # Panel refreshers
        # ------------------------------------------------------------------
        def _refresh_zone_status(self, statuses: Sequence[ZoneStatus]) -> None:
            if self.list_zone_status.count() != len(statuses):
                self.list_zone_status.clear()
                for _ in statuses:
                    self.list_zone_status.addItem(QListWidgetItem(""))
            for index, status in enumerate(statuses):
                item = self.list_zone_status.item(index)
                if item is None:
                    continue
                limit = f" / {status.max_occupancy}" if status.max_occupancy else ""
                flag = "  BREACH" if status.breached else ""
                state = "" if status.enabled else "  (disabled)"
                # Show the object total when non-person objects are also inside,
                # so a vehicle-triggered breach is not displayed as "0 BREACH".
                extra = (f"  ({status.objects} obj)"
                         if status.objects != status.occupancy else "")
                item.setText(
                    f"{status.name}:  {status.occupancy}{limit}{extra}{flag}{state}"
                )
                colour = (Palette.CRIT if status.breached
                          else (status.color if status.enabled else Palette.TEXT_FAINT))
                item.setForeground(QBrush(qcolor(colour)))

        def _refresh_class_table(self, snapshot: AnalyticsSnapshot) -> None:
            histogram = snapshot.class_histogram
            self.table_classes.setRowCount(len(histogram))
            for row, (name, count) in enumerate(histogram):
                self.table_classes.setItem(row, 0, QTableWidgetItem(str(name)))
                item = QTableWidgetItem(str(count))
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter))
                self.table_classes.setItem(row, 1, item)

        def _refresh_summary(self, snapshot: AnalyticsSnapshot,
                             telemetry: SystemTelemetry) -> None:
            info = self._source_info
            self.label_summary.setText(
                f"Mode              {self.settings.mode.value}\n"
                f"Source            {info.name if info else '-'}\n"
                f"Resolution        {info.resolution if info else '-'}\n"
                f"Detector          {self.settings.backend.value} "
                f"({self._model_state.value})\n"
                f"Device            {telemetry.device.upper()}\n"
                f"Uptime            {timedelta(seconds=int(snapshot.uptime_s))}\n"
                f"Frames processed  {self._session_frames}\n"
                f"Peak people       {snapshot.peak_people}\n"
                f"Entries / exits   {snapshot.entered} / {snapshot.exited}\n"
                f"Events (session)  {snapshot.infos + snapshot.warnings + snapshot.criticals}\n"
                f"  info            {snapshot.infos}\n"
                f"  warning         {snapshot.warnings}\n"
                f"  critical        {snapshot.criticals}\n"
                f"Dropped frames    {self.frame_queue.dropped}\n"
                f"Detect latency    {snapshot.detect_ms:.1f} ms\n"
                f"Track latency     {snapshot.track_ms:.2f} ms\n"
                f"Total latency     {snapshot.latency_ms:.1f} ms"
            )

        def _refresh_model_table(self) -> None:
            rt_ok, rt_reason = RTDETRDetector.dependency_status()
            rf_ok, rf_reason = RFDETRDetector.dependency_status()
            rows = [
                ("RT-DETR", rt_ok, DEP_TRANSFORMERS.version if rt_ok else "-",
                 rt_reason if not rt_ok else "transformers RTDetrForObjectDetection"),
                ("RF-DETR", rf_ok, DEP_RFDETR.version if rf_ok else "-",
                 rf_reason if not rf_ok else
                 "variants: " + ", ".join(RFDETRDetector.available_variants())),
                ("Simulation ground truth", True, "built-in",
                 "Always available - no weights required"),
            ]
            self.table_models.setRowCount(len(rows))
            for row, (name, ok, version, detail) in enumerate(rows):
                self.table_models.setItem(row, 0, QTableWidgetItem(name))
                status_item = QTableWidgetItem("AVAILABLE" if ok else "MISSING")
                status_item.setForeground(QBrush(qcolor(Palette.OK if ok else Palette.CRIT)))
                self.table_models.setItem(row, 1, status_item)
                self.table_models.setItem(row, 2, QTableWidgetItem(version))
                self.table_models.setItem(row, 3, QTableWidgetItem(detail))

        def _refresh_checkpoint_combo(self) -> None:
            """Populate the checkpoint list for the selected backend."""
            self.combo_checkpoint.blockSignals(True)
            self.combo_checkpoint.clear()
            backend = BackendKind(self.combo_backend.currentText())
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

        def _on_backend_combo(self, text: str) -> None:
            try:
                self.settings.backend = BackendKind(text)
            except ValueError:
                return
            self.combo_set_backend.blockSignals(True)
            self.combo_set_backend.setCurrentText(text)
            self.combo_set_backend.blockSignals(False)
            self._refresh_checkpoint_combo()

        def _on_load_model(self) -> None:
            backend = BackendKind(self.combo_backend.currentText())
            checkpoint = self.combo_checkpoint.currentData()
            ok, reason = (RTDETRDetector.dependency_status()
                          if backend is BackendKind.RT_DETR
                          else RFDETRDetector.dependency_status())
            if not ok:
                self._error_box(
                    f"{backend.value} unavailable",
                    f"{reason}\n\nSimulation mode remains fully functional without "
                    "a detection backend.",
                )
                self._on_model_state(ModelState.UNAVAILABLE, reason)
                return

            self.settings.backend = backend
            self.settings_manager.save_settings(self.settings)
            if self._inference is None:
                self._error_box(
                    "Pipeline not running",
                    "Start the pipeline first; the model is loaded inside the "
                    "inference worker so the interface stays responsive.",
                )
                return
            self._inference.request_backend(backend, checkpoint)
            self._set_status(f"Loading {backend.value} ...")

        def _on_unload_model(self) -> None:
            if self._inference is not None:
                self._inference.request_backend(None)
            self._on_model_state(ModelState.UNLOADED, "No detection backend active")
            self._set_status("Model unloaded")

        # ------------------------------------------------------------------
        # Source page handlers
        # ------------------------------------------------------------------
        def _on_browse(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", str(Path.home()),
                "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)",
            )
            if not path:
                return
            self.edit_file_path.setText(path)
            self.btn_use_file.setEnabled(True)
            # Probe metadata for display without starting the pipeline.
            try:
                probe = cv2.VideoCapture(path)
                if probe.isOpened():
                    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
                    count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    duration = timedelta(seconds=int(count / fps)) if fps > 0 else "-"
                    self.label_file_info.setText(
                        f"{Path(path).name}   |   {width}x{height}   |   "
                        f"{fps:.2f} fps   |   {count} frames   |   {duration}"
                    )
                else:
                    self.label_file_info.setText(
                        f"{Path(path).name} - OpenCV cannot open this file "
                        "(unsupported codec or container)."
                    )
                probe.release()
            except Exception as exc:  # noqa: BLE001
                self.label_file_info.setText(f"Could not probe file: {exc}")

        def _on_scan_cameras(self) -> None:
            self._set_status("Scanning camera devices ...")
            QApplication.processEvents()
            devices = WebcamSource.enumerate_devices()
            self.combo_camera.clear()
            if devices:
                self.combo_camera.addItems([f"Camera {i}" for i in devices])
                self.label_camera_info.setText(
                    f"Found {len(devices)} device(s): "
                    + ", ".join(str(d) for d in devices)
                )
            else:
                self.combo_camera.addItems([f"Camera {i}" for i in range(4)])
                self.label_camera_info.setText(
                    "No camera detected. The device may be disconnected, in use, "
                    "or blocked by OS privacy settings."
                )
            self._set_status("Camera scan complete")

        # ------------------------------------------------------------------
        # Events page handlers
        # ------------------------------------------------------------------
        def _reload_events(self) -> None:
            severity = self.combo_filter_severity.currentData()
            event_type = self.combo_filter_type.currentData()
            search = self.edit_search.text().strip() or None
            rows = self.database.query_events(
                limit=1000, severity=severity, event_type=event_type, search=search
            )
            self.table_events.load_rows(rows)
            counts = self.database.event_counts_by_severity()
            total = sum(counts.values())
            self.event_cards["total"].set_value(total, "stored")
            self.event_cards["info"].set_value(counts.get("INFO", 0))
            self.event_cards["warning"].set_value(counts.get("WARNING", 0))
            self.event_cards["critical"].set_value(counts.get("CRITICAL", 0))

        def _on_export_csv(self) -> None:
            if not self.database.available:
                self._error_box("Export unavailable",
                                "The local database is not available, so there is "
                                "no stored history to export.")
                return
            default = str(Path.home() / f"sentinel_events_{int(time.time())}.csv")
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Event History", default, "CSV files (*.csv)"
            )
            if not path:
                return
            severity = self.combo_filter_severity.currentData()
            event_type = self.combo_filter_type.currentData()
            search = self.edit_search.text().strip() or None
            rows = self.database.query_events(
                limit=100000, severity=severity, event_type=event_type, search=search
            )
            try:
                count = self.database.export_csv(Path(path), rows)
                self._set_status(f"Exported {count} events to {path}")
                QMessageBox.information(
                    self, "Export complete",
                    f"Exported {count} event(s) to:\n{path}"
                )
            except RuntimeError as exc:
                self._error_box("Export failed", str(exc))

        def _on_clear_events(self) -> None:
            answer = QMessageBox.question(
                self, "Clear event history",
                "Permanently delete all stored events from the local database?\n"
                "This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer is not QMessageBox.StandardButton.Yes:
                return
            removed = self.database.clear_events()
            self.table_events.setRowCount(0)
            self.table_recent.setRowCount(0)
            self._last_events.clear()
            self._reload_events()
            self._set_status(f"Cleared {removed} stored events")

        def _reload_sessions(self) -> None:
            rows = self.database.recent_sessions()
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
                    self.table_sessions.setItem(index, column, QTableWidgetItem(str(value)))

        # ------------------------------------------------------------------
        # Zones page handlers
        # ------------------------------------------------------------------
        def _refresh_zone_table(self) -> None:
            self.table_zones.blockSignals(True)
            self.table_zones.setRowCount(len(self.zones))
            for row, zone in enumerate(self.zones):
                enabled = QTableWidgetItem()
                enabled.setFlags(Qt.ItemFlag.ItemIsUserCheckable
                                 | Qt.ItemFlag.ItemIsEnabled)
                enabled.setCheckState(Qt.CheckState.Checked if zone.enabled
                                      else Qt.CheckState.Unchecked)
                self.table_zones.setItem(row, 0, enabled)

                name_item = QTableWidgetItem(zone.name)
                name_item.setForeground(QBrush(qcolor(zone.color)))
                self.table_zones.setItem(row, 1, name_item)

                kind_item = QTableWidgetItem(zone.kind.value)
                kind_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table_zones.setItem(row, 2, kind_item)

                severity_item = QTableWidgetItem(zone.severity.value)
                severity_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                severity_item.setForeground(QBrush(qcolor(zone.severity.color)))
                self.table_zones.setItem(row, 3, severity_item)

                occupancy_item = QTableWidgetItem(
                    str(zone.max_occupancy) if zone.max_occupancy else "-"
                )
                self.table_zones.setItem(row, 4, occupancy_item)

                points_item = QTableWidgetItem(str(len(zone.polygon)))
                points_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table_zones.setItem(row, 5, points_item)
            self.table_zones.blockSignals(False)

        def _on_zone_item_changed(self, item: QTableWidgetItem) -> None:
            row = item.row()
            if not (0 <= row < len(self.zones)):
                return
            zone = self.zones[row]
            column = item.column()
            if column == 0:
                zone.enabled = item.checkState() is Qt.CheckState.Checked
            elif column == 1:
                zone.name = item.text().strip() or zone.name
            elif column == 4:
                text = item.text().strip()
                try:
                    zone.max_occupancy = max(0, int(text)) if text not in ("", "-") else 0
                except ValueError:
                    self.table_zones.blockSignals(True)
                    item.setText(str(zone.max_occupancy) if zone.max_occupancy else "-")
                    self.table_zones.blockSignals(False)
                    return
            self._push_zones()

        def _push_zones(self) -> None:
            if self._inference is not None:
                self._inference.configure_zones(
                    self.zones, self.lines,
                    self._source_info.name if self._source_info else "",
                )

        def _on_zone_draw_toggled(self, enabled: bool) -> None:
            self._pending_zone_points = []
            self.zone_view.set_edit_mode(enabled)
            self.zone_view.set_edit_points([])
            self.btn_zone_draw.setText("Stop Drawing" if enabled else "Start Drawing")
            self.btn_zone_commit.setEnabled(False)
            if enabled:
                self._set_status("Click on the preview to place zone vertices")

        def _on_zone_point(self, x: float, y: float) -> None:
            self._pending_zone_points.append((float(x), float(y)))
            self.zone_view.set_edit_points(self._pending_zone_points)
            self.btn_zone_commit.setEnabled(len(self._pending_zone_points) >= 3)
            self._set_status(f"{len(self._pending_zone_points)} point(s) placed")

        def _on_zone_undo(self) -> None:
            if self._pending_zone_points:
                self._pending_zone_points.pop()
                self.zone_view.set_edit_points(self._pending_zone_points)
                self.btn_zone_commit.setEnabled(len(self._pending_zone_points) >= 3)

        def _on_zone_commit(self) -> None:
            if len(self._pending_zone_points) < 3:
                return
            kind = ZoneKind(self.combo_zone_kind.currentText())
            zone = ZoneConfig(
                zone_id=uuid.uuid4().hex[:8],
                name=self.edit_zone_name.text().strip() or f"{kind.value} Zone",
                kind=kind,
                polygon=list(self._pending_zone_points),
                color=kind.default_color,
                severity=kind.default_severity,
                max_occupancy=self.spin_zone_occupancy.value(),
            )
            self.zones.append(zone)
            self._pending_zone_points = []
            self.zone_view.set_edit_points([])
            self.btn_zone_draw.setChecked(False)
            self.btn_zone_commit.setEnabled(False)
            self._refresh_zone_table()
            self._push_zones()
            self._set_status(f"Zone '{zone.name}' created")
            self._log_system_event(f"Zone created: {zone.name} ({kind.value})")

        def _on_zone_delete(self) -> None:
            row = self.table_zones.currentRow()
            if not (0 <= row < len(self.zones)):
                self._set_status("Select a zone row to delete")
                return
            zone = self.zones.pop(row)
            self._refresh_zone_table()
            self._push_zones()
            self._set_status(f"Zone '{zone.name}' deleted")

        def _on_zone_save(self) -> None:
            SettingsManager.save_zones(self.settings.mode, self.zones, self.lines)
            self._set_status(f"Zone layout saved to {ZONES_PATH}")

        def _on_zone_reset(self) -> None:
            self.zones = default_zones(self.settings.mode)
            self.lines = default_lines(self.settings.mode)
            self._refresh_zone_table()
            self._push_zones()
            self._set_status("Default zones restored")

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _set_status(self, message: str) -> None:
            self.status_message.setText(message)
            LOGGER.debug("status: %s", message)

        def _error_box(self, title: str, message: str) -> None:
            LOGGER.error("%s: %s", title, message)
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(title)
            box.setText(title)
            box.setInformativeText(message)
            box.exec()

        def _log_system_event(self, message: str,
                              severity: Severity = Severity.INFO) -> None:
            """Record an application-level event in the same store as CV events."""
            event = SecurityEvent(
                timestamp=time.time(), event_type=EventType.SYSTEM_WARNING,
                severity=severity, description=message,
                mode=self.settings.mode.value,
                source=self._source_info.name if self._source_info else "system",
            )
            self.database.insert_events([event])
            self._last_events.append(event)
            self.table_recent.add_events([event], limit=120)

        # ------------------------------------------------------------------
        # Shutdown
        # ------------------------------------------------------------------
        def closeEvent(self, event) -> None:  # noqa: N802
            """Deterministic teardown: threads, models, capture devices, DB."""
            if self._shutting_down:
                event.accept()
                return
            self._shutting_down = True
            LOGGER.info("Shutting down ...")
            try:
                self._render_timer.stop()
                self._slow_timer.stop()
                self._stop_pipeline(update_ui=False)
                self.settings_manager.save_settings(self.settings)
                self.settings_manager.save_geometry(self)
                SettingsManager.save_zones(self.settings.mode, self.zones, self.lines)
            except Exception as exc:  # noqa: BLE001 - shutdown must not hang
                LOGGER.exception("Error during shutdown: %s", exc)
            finally:
                self.database.close()
            LOGGER.info("Shutdown complete")
            event.accept()


# =============================================================================
# SECTION 17 - APPLICATION STARTUP
# =============================================================================

def run_self_test(frames: int = 90, mode: AppMode = AppMode.CLASSROOM) -> int:
    """Headless exercise of the full non-GUI pipeline.

    Runs source -> tracking -> zones -> events -> analytics -> SQLite without Qt,
    which makes the core verifiable in CI and on machines with no display.
    Returns a process exit code.
    """
    configure_logging()
    print(f"{APP_NAME} v{APP_VERSION} - headless self-test ({mode.value})")
    print("-" * 64)

    settings = AppSettings(mode=mode)
    source = SimulationSource(mode, people=12, vehicles=2)
    tracker = KalmanTracker(settings.max_age, settings.min_hits, settings.iou_threshold)
    events_engine = EventEngine(settings)
    zones, lines = default_zones(mode), default_lines(mode)
    events_engine.configure(zones, lines, "self-test")
    analytics = AnalyticsEngine()
    database = EventDatabase(APP_DIR / "selftest.db")
    database.start_session(mode.value, "self-test", "none", "cpu")
    run_started = time.time()   # so the readback counts THIS run only
    renderer = FrameRenderer(SupervisionAdapter())

    total_events = 0
    track_ids: set = set()
    started = time.perf_counter()
    try:
        source.open()
        for _ in range(frames):
            packet = source.read()
            if packet is None:
                continue
            assert packet.synthetic_detections is not None
            detections = [d for d in packet.synthetic_detections
                          if d.confidence >= settings.confidence]
            tracks = tracker.update(detections)
            lost = tracker.pop_lost()
            track_ids.update(t.track_id for t in tracks)

            height, width = packet.frame.shape[:2]
            new_events, zone_status, alert_ids = events_engine.process(
                tracks, lost, width, height
            )
            total_events += len(new_events)
            database.insert_events(new_events)
            analytics.record_events(new_events)
            analytics.record_frame(
                people=sum(1 for t in tracks if t.is_person),
                vehicles=sum(1 for t in tracks if t.is_vehicle),
                others=0, tracks=len(tracks), fps=25.0, timings=Timings(),
                class_names=[t.class_name for t in tracks],
            )

            canvas = packet.frame.copy()
            renderer.draw_zones(canvas, zones, {z.zone_id: z for z in zone_status})
            renderer.draw_lines(canvas, lines, events_engine.line_counts)
            renderer.draw_tracks(canvas, tracks, settings, alert_ids)
            renderer.draw_hud(
                canvas, fps=25.0, mode=mode, source=packet.source_name,
                detector="Synthetic GT",
                people=sum(1 for t in tracks if t.is_person),
                vehicles=sum(1 for t in tracks if t.is_vehicle),
                tracks=len(tracks), synthetic=True, settings=settings, critical=0,
            )
        elapsed = time.perf_counter() - started
        snapshot = analytics.snapshot(
            people=0, vehicles=0, others=0, tracks=len(track_ids),
            fps=frames / max(1e-6, elapsed), timings=Timings(),
        )
        stored = len(database.query_events(limit=100000, since=run_started))

        print(f"  frames processed     : {frames}")
        print(f"  throughput           : {frames / max(1e-6, elapsed):.1f} fps")
        print(f"  unique track IDs     : {len(track_ids)}")
        print(f"  events generated     : {total_events}")
        print(f"  events stored (db)   : {stored}")
        print(f"  entries / exits      : {snapshot.entered} / {snapshot.exited}")
        print(f"  warnings / critical  : {snapshot.warnings} / {snapshot.criticals}")
        print(f"  zones evaluated      : {len(zones)}")
        print(f"  supervision          : "
              f"{'yes' if DEP_SUPERVISION.available else 'no (OpenCV fallback)'}")

        ok = frames > 0 and len(track_ids) > 0 and total_events > 0
        print("-" * 64)
        print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 - report, don't traceback-dump
        traceback.print_exc()
        print(f"SELF-TEST FAILED: {exc}")
        return 1
    finally:
        source.release()
        database.end_session(frames, total_events)
        database.close()


def install_exception_hook() -> None:
    """Log unhandled exceptions instead of letting Qt silently swallow them.

    A surveillance console that dies without explanation is worse than one that
    reports the fault, so the hook logs and (when a GUI exists) surfaces it.
    """
    def hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        LOGGER.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
        if QT_AVAILABLE and QApplication.instance() is not None:
            try:
                QMessageBox.critical(
                    None, "Unexpected error",
                    f"{exc_type.__name__}: {exc_value}\n\n"
                    f"The error was logged to:\n{LOG_PATH}",
                )
            except Exception:  # noqa: BLE001 - never recurse inside the hook
                pass

    sys.excepthook = hook


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py", description=f"{APP_NAME} - {APP_SUBTITLE}"
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run the headless pipeline self-test and exit")
    parser.add_argument("--frames", type=int, default=90,
                        help="frames to process during --self-test")
    parser.add_argument("--mode", choices=[m.value.lower() for m in AppMode],
                        default=AppMode.CLASSROOM.value.lower(),
                        help="deployment scenario to start in")
    parser.add_argument("--source", choices=[k.value.lower().replace(" ", "-")
                                             for k in SourceKind],
                        default=None, help="auto-start with this video source")
    parser.add_argument("--video", type=str, default=None,
                        help="path to a video file (implies --source video-file)")
    parser.add_argument("--no-splash", action="store_true",
                        help="skip the startup dependency screen")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    mode = AppMode.FACTORY if args.mode.startswith("fact") else AppMode.CLASSROOM

    if args.self_test:
        return run_self_test(frames=max(1, args.frames), mode=mode)

    configure_logging(args.verbose)
    LOGGER.info("%s v%s starting on %s / Python %s", APP_NAME, APP_VERSION,
                platform.platform(), platform.python_version())

    if not QT_AVAILABLE:
        sys.stderr.write(
            f"FATAL: PyQt6 is required for the {APP_NAME} desktop console.\n"
            f"       Import error: {QT_IMPORT_ERROR}\n"
            "       Install with: pip install PyQt6\n\n"
            "       The core pipeline can still be verified without a GUI:\n"
            "           python app.py --self-test\n"
        )
        return 2

    install_exception_hook()

    # High-DPI handling: Qt6 scales by default; this only refines the policy.
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception as exc:  # noqa: BLE001 - platform dependent
        LOGGER.debug("Could not set DPI rounding policy: %s", exc)

    app = QApplication(sys.argv[:1] + list(argv or []))
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(ORG_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())

    # Dark palette so native-drawn elements match the stylesheet.
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(Palette.BG_APP))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Palette.TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(Palette.BG_INPUT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Palette.BG_CARD))
    palette.setColor(QPalette.ColorRole.Text, QColor(Palette.TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(Palette.BG_ELEV))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Palette.TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(Palette.ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#04101C"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(Palette.BG_ELEV))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(Palette.TEXT))
    app.setPalette(palette)

    window = MainWindow()
    if mode is not window.settings.mode:
        window.settings.mode = mode
        window._switch_mode(mode)

    manager = window.settings_manager
    if not args.no_splash and not manager.skip_dependency_screen:
        dialog = DependencyDialog(window)
        dialog.exec()
        if dialog.skip_next_time:
            manager.skip_dependency_screen = True

    window.show()

    # Optional auto-start from the command line.
    if args.video:
        window.edit_file_path.setText(args.video)
        window.btn_use_file.setEnabled(True)
        QTimer.singleShot(200, lambda: window._activate_source(SourceKind.FILE))
    elif args.source:
        mapping = {
            "simulation": SourceKind.SIMULATION,
            "video-file": SourceKind.FILE,
            "webcam": SourceKind.WEBCAM,
        }
        kind = mapping.get(args.source, SourceKind.SIMULATION)
        QTimer.singleShot(200, lambda: window._activate_source(kind))
    else:
        # Simulation is always ready, so the console is never empty on launch.
        QTimer.singleShot(
            250, lambda: window._activate_source(SourceKind.SIMULATION)
        )

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
