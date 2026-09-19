"""Frame annotation: zones, lines, boxes, track IDs and HUD overlays.

Uses supervision's annotators when available and a complete OpenCV
implementation otherwise, so the visual output is equivalent either way.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from sentinel.models import (
    AppMode, AppSettings, LineConfig, TrackedObject, TrackPhase, ZoneConfig,
    ZoneStatus,
)
from sentinel.theme import Palette, hex_to_bgr
from sentinel.vision.supervision_adapter import SupervisionAdapter

__all__ = ["FrameRenderer"]


class FrameRenderer:
    """Draws zones, lines, boxes, IDs and HUD overlays onto the output frame.

    Uses supervision's annotators when available and falls back to a complete
    OpenCV implementation otherwise, so the visual output is equivalent either
    way.
    """

    def __init__(self, adapter: SupervisionAdapter) -> None:
        self.adapter = adapter

    # -- zones / lines --------------------------------------------------------
    @staticmethod
    def draw_zones(
        frame: "np.ndarray", zones: Sequence[ZoneConfig],
        status: Dict[str, ZoneStatus],
    ) -> None:
        if not zones:
            return
        overlay = frame.copy()
        height, width = frame.shape[:2]
        for zone in zones:
            if not zone.enabled or len(zone.polygon) < 3:
                continue
            poly = zone.pixel_polygon(width, height)
            state = status.get(zone.zone_id)
            breached = bool(state and state.breached)
            color = hex_to_bgr(Palette.CRIT if breached else zone.color)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], True, color, 2 if not breached else 3, cv2.LINE_AA)

            label = zone.name
            if state is not None:
                label = f"{zone.name}  [{state.occupancy}]"
            x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
            y = max(14, y)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x, y - th - 7), (x + tw + 8, y), color, -1)
            cv2.putText(frame, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (16, 18, 22), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, dst=frame)

    @staticmethod
    def draw_lines(
        frame: "np.ndarray", lines: Sequence[LineConfig],
        counts: Dict[str, Tuple[int, int]],
    ) -> None:
        height, width = frame.shape[:2]
        for line in lines:
            if not line.enabled:
                continue
            p1, p2 = line.pixel_points(width, height)
            color = hex_to_bgr(line.color)
            cv2.line(frame, p1, p2, (18, 20, 24), 5, cv2.LINE_AA)
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
            cv2.circle(frame, p1, 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p2, 4, color, -1, cv2.LINE_AA)
            inbound, outbound = counts.get(line.line_id, (0, 0))
            text = f"{line.name}  IN {inbound}  OUT {outbound}"
            tx = min(p1[0], p2[0]) + 6
            ty = max(16, min(p1[1], p2[1]) - 8)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, color, 1, cv2.LINE_AA)

    # -- detections / tracks --------------------------------------------------
    def draw_tracks(
        self, frame: "np.ndarray", tracks: Sequence[TrackedObject],
        settings: AppSettings, alert_ids: Iterable[int] = (),
    ) -> None:
        if not tracks or not settings.show_boxes:
            return
        alerts = set(alert_ids)
        labels = [self._label_for(t, settings) for t in tracks]

        # Supervision path (skipped for alerting tracks, which need the custom
        # red treatment that communicates severity at a glance).
        if (settings.use_supervision_annotators and self.adapter.available
                and not alerts):
            annotated = self.adapter.annotate(
                frame, tracks, labels, settings.show_trails
            )
            if annotated is not None:
                if annotated is not frame:
                    np.copyto(frame, annotated)
                return

        # OpenCV fallback / alert path.
        for track, label in zip(tracks, labels):
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            alerting = track.track_id in alerts
            color = hex_to_bgr(Palette.CRIT) if alerting \
                else Palette.track_color_bgr(track.track_id)
            thickness = 3 if alerting else 2

            if settings.show_trails and len(track.trail) > 2:
                pts = np.array(track.trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

            if track.phase is TrackPhase.COASTING:
                # Dashed box communicates "predicted, not currently detected".
                self._dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                # Corner accents give the boxes a product-grade look.
                arm = max(6, int((x2 - x1) * 0.22))
                for cx, cy, dx, dy in (
                    (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
                ):
                    cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, thickness + 1, cv2.LINE_AA)
                    cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, thickness + 1, cv2.LINE_AA)

            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                ly = max(th + 6, y1)
                cv2.rectangle(frame, (x1, ly - th - 7), (x1 + tw + 8, ly), color, -1)
                cv2.putText(frame, label, (x1 + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (16, 18, 22), 1, cv2.LINE_AA)

            if alerting:
                cv2.putText(frame, "ALERT", (x1, min(frame.shape[0] - 4, y2 + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, hex_to_bgr(Palette.CRIT),
                            2, cv2.LINE_AA)

    @staticmethod
    def _label_for(track: TrackedObject, settings: AppSettings) -> str:
        parts: List[str] = []
        if settings.show_track_ids:
            parts.append(f"#{track.track_id}")
        parts.append(track.class_name)
        if settings.show_confidence:
            parts.append(f"{track.confidence:.2f}")
        return " ".join(parts)

    @staticmethod
    def _dashed_rect(
        frame: "np.ndarray", p1: Tuple[int, int], p2: Tuple[int, int],
        color: Tuple[int, int, int], thickness: int, dash: int = 8,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)

    # -- HUD ------------------------------------------------------------------
    @staticmethod
    def draw_hud(
        frame: "np.ndarray", *, fps: float, mode: AppMode, source: str,
        detector: str, people: int, vehicles: int, tracks: int,
        synthetic: bool, settings: AppSettings, critical: int,
    ) -> None:
        height, width = frame.shape[:2]
        pad = 10
        bar_h = 30

        # Translucent top HUD strip.
        strip = frame[0:bar_h, 0:width]
        cv2.addWeighted(np.full_like(strip, (14, 16, 20)), 0.62, strip, 0.38, 0, dst=strip)

        left_text = f"{mode.value.upper()}  |  {source}  |  {detector}"
        cv2.putText(frame, left_text, (pad, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (226, 234, 244), 1, cv2.LINE_AA)

        right_bits: List[str] = []
        if settings.show_fps_overlay:
            right_bits.append(f"{fps:5.1f} FPS")
        right_bits.append(f"P {people}")
        if vehicles:
            right_bits.append(f"V {vehicles}")
        right_bits.append(f"T {tracks}")
        right_text = "   ".join(right_bits)
        (tw, _th), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, right_text, (width - tw - pad, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 234, 244), 1, cv2.LINE_AA)

        # SIMULATION badge - always visible when the feed is synthetic so the
        # operator can never mistake it for a real camera.
        if synthetic:
            badge = "SIMULATION"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            bx, by = width - bw - 16 - pad, bar_h + 6
            cv2.rectangle(frame, (bx, by), (bx + bw + 16, by + bh + 12),
                          hex_to_bgr(Palette.VIOLET), -1)
            cv2.putText(frame, badge, (bx + 8, by + bh + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 14, 30), 2, cv2.LINE_AA)

        # Critical-alert border pulse.
        if critical > 0:
            cv2.rectangle(frame, (2, 2), (width - 3, height - 3),
                          hex_to_bgr(Palette.CRIT), 3)

    @staticmethod
    def placeholder(
        width: int, height: int, title: str, subtitle: str = "",
        accent: str = Palette.TEXT_DIM,
    ) -> "np.ndarray":
        """Render a NO SIGNAL / MODEL LOADING style placeholder frame."""
        frame = np.full((height, width, 3), 14, dtype=np.uint8)
        frame[:, :, 0] = 18
        frame[:, :, 1] = 16
        # Faint diagonal hatching.
        for x in range(-height, width, 28):
            cv2.line(frame, (x, 0), (x + height, height), (26, 30, 36), 1, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, title, ((width - tw) // 2, (height + th) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, hex_to_bgr(accent), 2, cv2.LINE_AA)
        if subtitle:
            (sw, _sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, subtitle, ((width - sw) // 2, (height + th) // 2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 144, 160), 1, cv2.LINE_AA)
        return frame
