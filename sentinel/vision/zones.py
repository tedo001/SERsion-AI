"""Scenario-appropriate default zone and line layouts.

Zones are stored in NORMALISED coordinates so a layout stays valid across
source resolutions.
"""

from __future__ import annotations

from typing import List, Tuple

from sentinel.models import (
    AppMode, LineConfig, Severity, ZoneConfig, ZoneKind,
)

__all__ = ["default_zones", "default_lines"]


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
