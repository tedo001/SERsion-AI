"""Headless verification of the core pipeline.

Runs source -> tracking -> zones -> events -> analytics -> SQLite without Qt,
which makes the core testable in CI and on machines with no display.
"""

from __future__ import annotations

import time
import traceback

from sentinel.config import APP_DIR, APP_NAME, APP_VERSION, configure_logging
from sentinel.deps import DEP_SUPERVISION
from sentinel.models import AppMode, AppSettings, Timings
from sentinel.sources import SimulationSource
from sentinel.storage import EventDatabase
from sentinel.tracking import KalmanTracker
from sentinel.vision import (
    AnalyticsEngine, EventEngine, FrameRenderer, SupervisionAdapter,
    default_lines, default_zones,
)

__all__ = ["run_self_test"]


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
