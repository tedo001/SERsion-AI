"""Qt presentation layer.

Importing this package requires PyQt6.  The entry point checks
``sentinel.deps.QT_AVAILABLE`` before importing anything from here, so a
machine without Qt can still run the headless self-test.
"""

from __future__ import annotations

from sentinel.ui.dialogs import DependencyDialog
from sentinel.ui.main_window import MainWindow

__all__ = ["MainWindow", "DependencyDialog"]
