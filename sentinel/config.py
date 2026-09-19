"""Filesystem locations and logging configuration.

Bottom of the dependency graph: this module imports nothing from the rest of
the package, so every other module may import it freely.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from sentinel import APP_NAME, APP_SUBTITLE, APP_VERSION, ORG_NAME

__all__ = [
    "APP_NAME", "APP_SUBTITLE", "APP_VERSION", "ORG_NAME",
    "APP_DIR", "DB_PATH", "LOG_PATH", "ZONES_PATH",
    "LOGGER", "configure_logging",
]


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
