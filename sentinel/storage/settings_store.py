"""Persistence for user settings, window geometry and zone layouts.

Settings and window state live in QSettings; zone/line layouts live in a JSON
file so they can be inspected,version-controlled or shipped with a deployment.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMainWindow

from sentinel.config import APP_NAME, LOGGER, ORG_NAME, ZONES_PATH
from sentinel.models import AppMode, AppSettings, LineConfig, ZoneConfig
from sentinel.vision.zones import default_lines, default_zones

__all__ = ["SettingsManager"]


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
