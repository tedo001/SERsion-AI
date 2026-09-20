"""Event engine: turns per-frame track state into graded, de-duplicated events."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentinel.models import (
    AppSettings, EventType, LineConfig, NightStatus, SecurityEvent, Severity,
    TrackedObject, ZoneConfig, ZoneKind, ZoneStatus,
)

__all__ = ["EventEngine"]


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
        EventType.LOW_LIGHT: 0.0,      # edge-triggered, not rate limited
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
        self._night_active: Optional[bool] = None
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
            self._night_active = None

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

    def note_light_conditions(self, night: NightStatus) -> List[SecurityEvent]:
        """Emit an event when the scene crosses between day and low light.

        Edge-triggered: one event per transition, not one per frame.  Severity
        is INFO because low light is a scene condition, not a safety incident -
        though it does mean detection confidence should be read with care.
        """
        with self._lock:
            if self._night_active == night.active:
                return []
            first_observation = self._night_active is None
            self._night_active = night.active
            if first_observation and not night.active:
                return []      # starting in daylight is not worth an event
            if night.active:
                description = (
                    f"Low-light conditions detected (scene luma "
                    f"{night.luminance:.0f}/255"
                    + (", monochrome/IR source" if night.monochrome else "")
                    + f"); enhancement engaged at gain {night.gain:.1f}x"
                )
            else:
                description = (
                    f"Normal lighting restored (scene luma "
                    f"{night.luminance:.0f}/255); enhancement disengaged"
                )
            return [self._make(EventType.LOW_LIGHT, Severity.INFO, description)]

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
