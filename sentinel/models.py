"""Enums and dataclasses shared by every layer of the application.

These types are the contract between the detection backends, the tracker, the
event engine, the renderer and the UI.  Nothing here knows which detector is
running or whether a GUI exists.
"""

from __future__ import annotations

import dataclasses
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sentinel.config import LOGGER
from sentinel.theme import Palette

__all__ = [
    "Severity", "EventType", "AppMode", "SourceKind", "BackendKind",
    "DevicePref", "ModelState", "PipelineState", "TrackPhase", "ZoneKind",
    "PERSON_CLASS_NAMES", "VEHICLE_CLASS_NAMES",
    "Detection", "TrackedObject", "ZoneConfig", "LineConfig", "SecurityEvent",
    "Timings", "FramePacket", "ZoneStatus", "FrameResult", "AppSettings",
]


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
