"""GUI bootstrap: QApplication setup, theming, and the startup sequence.

Imported only after :data:`sentinel.deps.QT_AVAILABLE` has been confirmed, so
the headless path never pays for Qt.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QMessageBox

from sentinel.config import (
    APP_NAME, APP_VERSION, LOG_PATH, LOGGER, ORG_NAME,
)
from sentinel.models import AppMode
from sentinel.theme import Palette, build_stylesheet
from sentinel.ui import DependencyDialog, MainWindow

__all__ = ["run_gui", "install_exception_hook", "apply_theme"]


def install_exception_hook() -> None:
    """Log unhandled exceptions instead of letting Qt silently swallow them.

    A surveillance console that dies without explanation is worse than one that
    reports the fault, so the hook logs and surfaces it.
    """
    def hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        LOGGER.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
        if QApplication.instance() is not None:
            try:
                QMessageBox.critical(
                    None, "Unexpected error",
                    f"{exc_type.__name__}: {exc_value}\n\n"
                    f"The error was logged to:\n{LOG_PATH}",
                )
            except Exception:  # noqa: BLE001 - never recurse inside the hook
                pass

    sys.excepthook = hook


def apply_theme(app: QApplication) -> None:
    """Install the dark style sheet and a matching native palette."""
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())

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


def run_gui(args: argparse.Namespace, mode: AppMode) -> int:
    """Construct the application, show the window, and enter the event loop."""
    install_exception_hook()

    # High-DPI handling: Qt6 scales by default; this only refines the policy.
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception as exc:  # noqa: BLE001 - platform dependent
        LOGGER.debug("Could not set DPI rounding policy: %s", exc)

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(ORG_NAME)
    apply_theme(app)

    window = MainWindow()
    if mode is not window.settings.mode:
        window.switch_mode(mode)

    manager = window.settings_manager
    if not args.no_splash and not manager.skip_dependency_screen:
        dialog = DependencyDialog(window)
        dialog.exec()
        if dialog.skip_next_time:
            manager.skip_dependency_screen = True

    window.show()
    _schedule_autostart(window, args)
    return app.exec()


def _schedule_autostart(window: MainWindow, args: argparse.Namespace) -> None:
    """Pick the source to open once the event loop is running.

    Simulation is always ready, so the console is never empty on launch.
    """
    controller = window.controller
    if args.video:
        window.page_sources.set_file(args.video, MainWindow._probe_video(args.video))
        QTimer.singleShot(
            200, lambda: controller.activate_file(args.video, True)
        )
        return

    if args.source == "video-file":
        path = window.page_sources.file_path
        if path:
            QTimer.singleShot(200, lambda: controller.activate_file(path, True))
            return
        LOGGER.warning("--source video-file given without --video; using simulation")
    elif args.source == "webcam":
        QTimer.singleShot(
            200, lambda: controller.activate_webcam(window.page_sources.selected_camera())
        )
        return

    people, vehicles = window.page_sources.population()
    QTimer.singleShot(
        250, lambda: controller.activate_simulation(people, vehicles)
    )
