"""Dark-theme colour tokens and the Qt style sheet.

Pure data plus string templating - importing this module does not require Qt,
which keeps the palette usable by the OpenCV renderer as well as the widgets.
"""

from __future__ import annotations

from string import Template
from typing import Tuple

__all__ = ["Palette", "hex_to_bgr", "hex_to_rgb", "QSS_TEMPLATE", "build_stylesheet"]


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
