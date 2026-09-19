#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LOFOP Sentinel - application entry point.

Run the desktop console:

    python app.py                       # starts in Simulation mode
    python app.py --mode factory
    python app.py --video clip.mp4
    python app.py --source webcam

Verify the core pipeline without a display:

    python app.py --self-test --frames 600 --mode factory

The application itself lives in the ``sentinel`` package; this file only parses
arguments, checks that the runtime is usable, and starts either the GUI or the
headless self-test.  See ``sentinel/__init__.py`` for the module map.

This system performs object detection and tracking only.  It does not perform
facial recognition or identity inference, and stores no biometric data.
"""

from __future__ import annotations

import argparse
import platform
import sys
from typing import Optional, Sequence


def _fail(message: str) -> int:
    sys.stderr.write(message)
    return 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py", description="LOFOP Sentinel - Edge Vision Intelligence Platform"
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run the headless pipeline self-test and exit")
    parser.add_argument("--frames", type=int, default=90,
                        help="frames to process during --self-test")
    parser.add_argument("--mode", choices=["classroom", "factory"],
                        default="classroom",
                        help="deployment scenario to start in")
    parser.add_argument("--source", choices=["simulation", "video-file", "webcam"],
                        default=None, help="auto-start with this video source")
    parser.add_argument("--video", type=str, default=None,
                        help="path to a video file (implies --source video-file)")
    parser.add_argument("--no-splash", action="store_true",
                        help="skip the startup dependency screen")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # NumPy and OpenCV are hard requirements of the core pipeline.  Import them
    # through the package so a missing one is reported, not traced back.
    try:
        from sentinel.models import AppMode
    except ImportError as exc:
        return _fail(
            f"FATAL: a core dependency is missing ({exc}).\n"
            "       Install the runtime with:\n"
            "           pip install -r requirements.txt\n"
        )

    mode = AppMode.FACTORY if args.mode == "factory" else AppMode.CLASSROOM

    if args.self_test:
        from sentinel.selftest import run_self_test

        return run_self_test(frames=max(1, args.frames), mode=mode)

    from sentinel.config import APP_NAME, APP_VERSION, LOGGER, configure_logging
    from sentinel.deps import QT_AVAILABLE, QT_IMPORT_ERROR

    configure_logging(args.verbose)
    LOGGER.info("%s v%s starting on %s / Python %s", APP_NAME, APP_VERSION,
                platform.platform(), platform.python_version())

    if not QT_AVAILABLE:
        return _fail(
            f"FATAL: PyQt6 is required for the {APP_NAME} desktop console.\n"
            f"       Import error: {QT_IMPORT_ERROR}\n"
            "       Install with: pip install PyQt6\n\n"
            "       The core pipeline can still be verified without a GUI:\n"
            "           python app.py --self-test\n"
        )

    from sentinel.startup import run_gui

    return run_gui(args, mode)


if __name__ == "__main__":
    sys.exit(main())
