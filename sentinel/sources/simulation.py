"""Built-in synthetic CCTV source.

Renders a stylised surveillance scene and emits ground-truth boxes alongside
each frame, which is what makes the whole product demonstrable with no model
weights and no external video files present.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sentinel.config import LOGGER
from sentinel.models import AppMode, Detection, FramePacket, SourceKind
from sentinel.sources.base import BaseVideoSource, SourceInfo
from sentinel.theme import hex_to_bgr

__all__ = ["SimActor", "SimulationSource"]


@dataclass
class SimActor:
    """One synthetic person or vehicle moving on the simulated camera floor.

    Positions live in normalised floor coordinates (0..1).  Apparent size scales
    with ``y`` so actors further "up" the frame look further away, which gives
    the rendered scene a believable CCTV perspective.
    """

    actor_id: int
    x: float
    y: float
    target_x: float
    target_y: float
    speed: float
    kind: str                       # "person" | "vehicle"
    color: Tuple[int, int, int]
    phase: float = 0.0              # gait animation phase
    dwell: float = 0.0              # seconds to stand still at the target
    loiter_zone: Optional[str] = None
    visible: bool = True

    def step(self, dt: float, rng: random.Random) -> None:
        """Advance toward the current waypoint, picking a new one on arrival."""
        if self.dwell > 0.0:
            self.dwell = max(0.0, self.dwell - dt)
            self.phase += dt * 1.5
            return
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.hypot(dx, dy)
        if distance < 0.015:
            self.pick_target(rng)
            self.dwell = rng.uniform(0.0, 2.5) if self.kind == "person" else 0.0
            return
        step = min(self.speed * dt, distance)
        self.x += dx / distance * step
        self.y += dy / distance * step
        self.phase += dt * (6.0 if self.kind == "person" else 2.0)
        # Keep actors inside the visible floor area.
        self.x = float(np.clip(self.x, 0.03, 0.97))
        self.y = float(np.clip(self.y, 0.30, 0.97))

    def pick_target(self, rng: random.Random) -> None:
        if self.kind == "vehicle":
            # Vehicles patrol a horizontal lane.
            self.target_x = rng.choice([0.06, 0.94])
            self.target_y = float(np.clip(self.y + rng.uniform(-0.03, 0.03), 0.55, 0.9))
        else:
            self.target_x = rng.uniform(0.05, 0.95)
            self.target_y = rng.uniform(0.32, 0.95)

    def scale(self) -> float:
        """Perspective scale factor: 0.45 at the horizon, 1.0 in the foreground."""
        return 0.45 + 0.55 * float(np.clip((self.y - 0.28) / 0.7, 0.0, 1.0))

    def bbox(self, width: int, height: int) -> Tuple[float, float, float, float]:
        scale = self.scale()
        if self.kind == "vehicle":
            bw = 0.20 * scale * width
            bh = 0.11 * scale * height
        else:
            bw = 0.052 * scale * width
            bh = 0.155 * scale * height
        cx = self.x * width
        cy = self.y * height
        return (cx - bw / 2.0, cy - bh, cx + bw / 2.0, cy)


class SimulationSource(BaseVideoSource):
    """Built-in synthetic CCTV feed.

    Renders a stylised surveillance scene (perspective floor, props, grain,
    scanlines, timestamp burn-in) populated with autonomous actors, and emits
    GROUND-TRUTH bounding boxes alongside each frame.  Those boxes are what
    makes the whole product demonstrable with no model weights present, and they
    are always flagged as synthetic so the UI can label them honestly.

    Realism touches that matter for the rest of the pipeline:
        * per-frame box jitter          -> exercises the Kalman smoothing
        * random detection dropout      -> exercises coasting / re-association
        * confidence sampled per actor  -> exercises the confidence threshold
    """

    def __init__(
        self,
        mode: AppMode,
        width: int = 960,
        height: int = 540,
        fps: float = 25.0,
        people: int = 10,
        vehicles: int = 2,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.people = int(np.clip(people, 1, 20))
        self.vehicles = int(np.clip(vehicles, 0, 6))
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._actors: List[SimActor] = []
        self._background: Optional[np.ndarray] = None
        self._last_step = time.time()
        self._noise_bank: List[np.ndarray] = []
        self._noise_index = 0

    # -- lifecycle ------------------------------------------------------------
    def open(self) -> None:
        self._actors = self._spawn_actors()
        self._background = self._render_background()
        # Pre-render a small bank of noise fields; generating fresh noise every
        # frame is surprisingly expensive and would distort the FPS readings.
        self._noise_bank = [
            self._np_rng.normal(0.0, 5.0, (self.height, self.width, 1)).astype(np.float32)
            for _ in range(8)
        ]
        self._opened = True
        self._index = 0
        self._last_step = time.time()
        self._info = SourceInfo(
            kind=SourceKind.SIMULATION,
            name=f"SIM-CAM-01 ({self.mode.value})",
            width=self.width, height=self.height, fps=self.fps,
            detail="Built-in synthetic scene with ground-truth annotations",
        )
        LOGGER.info("Simulation started: %d people, %d vehicles, %s mode",
                    self.people, self.vehicles, self.mode.value)

    def set_population(self, people: int, vehicles: int) -> None:
        """Re-populate the scene without restarting the pipeline."""
        self.people = int(np.clip(people, 1, 20))
        self.vehicles = int(np.clip(vehicles, 0, 6))
        self._actors = self._spawn_actors()

    def set_mode(self, mode: AppMode) -> None:
        self.mode = mode
        self._background = self._render_background()
        self._actors = self._spawn_actors()
        self._info.name = f"SIM-CAM-01 ({mode.value})"

    def release(self) -> None:
        self._actors.clear()
        self._background = None
        self._noise_bank.clear()
        self._opened = False

    def restart(self) -> bool:
        self._index = 0
        self._actors = self._spawn_actors()
        return True

    # -- scene construction ---------------------------------------------------
    def _spawn_actors(self) -> List[SimActor]:
        actors: List[SimActor] = []
        palette = [hex_to_bgr(c) for c in (
            "#D6DEE8", "#B7C4D2", "#9FB0C2", "#C8B8A6", "#A8B6A0", "#C2AEBE",
        )]
        for i in range(self.people):
            x = self._rng.uniform(0.08, 0.92)
            y = self._rng.uniform(0.35, 0.92)
            actor = SimActor(
                actor_id=i + 1, x=x, y=y, target_x=x, target_y=y,
                speed=self._rng.uniform(0.02, 0.075),
                kind="person",
                color=palette[i % len(palette)],
                phase=self._rng.uniform(0.0, 6.28),
            )
            actor.pick_target(self._rng)
            actors.append(actor)

        if self.mode is AppMode.FACTORY:
            for j in range(self.vehicles):
                y = 0.62 + 0.12 * j
                actor = SimActor(
                    actor_id=1000 + j,
                    x=self._rng.uniform(0.15, 0.85), y=min(0.9, y),
                    target_x=0.9, target_y=min(0.9, y),
                    speed=self._rng.uniform(0.05, 0.11),
                    kind="vehicle",
                    color=hex_to_bgr("#E8A33D"),
                )
                actors.append(actor)
        return actors

    def _render_background(self) -> "np.ndarray":
        """Draw the static scene: floor gradient, perspective grid and props."""
        w, h = self.width, self.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.28)

        # Back wall with a vertical gradient.
        wall_top = np.array(hex_to_bgr("#2A3038"), dtype=np.float32)
        wall_bottom = np.array(hex_to_bgr("#3A424C"), dtype=np.float32)
        for y in range(horizon):
            t = y / max(1, horizon)
            frame[y, :] = (wall_top * (1 - t) + wall_bottom * t).astype(np.uint8)

        # Floor with a depth gradient (darker far away).
        floor_far = np.array(hex_to_bgr("#4A5058"), dtype=np.float32)
        floor_near = np.array(hex_to_bgr("#6A727C"), dtype=np.float32)
        for y in range(horizon, h):
            t = (y - horizon) / max(1, h - horizon)
            frame[y, :] = (floor_far * (1 - t) + floor_near * t).astype(np.uint8)

        # Perspective grid converging on a vanishing point.  It is drawn on a
        # scratch layer and composited only BELOW the horizon, so the lines stay
        # on the floor instead of streaking across the back wall.
        grid_layer = frame.copy()
        vp = (int(w * 0.5), int(h * 0.12))
        grid_color = (96, 104, 112)
        for i in range(-12, 13):
            x_bottom = int(w * 0.5 + i * w * 0.14)
            cv2.line(grid_layer, vp, (x_bottom, h), grid_color, 1, cv2.LINE_AA)
        depth = 0.0
        for _ in range(14):
            depth += 0.055 + depth * 0.36
            y = int(horizon + (h - horizon) * min(1.0, depth))
            if y >= h:
                break
            cv2.line(grid_layer, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)
        frame[horizon:, :] = grid_layer[horizon:, :]

        # Scene props differ per deployment scenario.
        if self.mode is AppMode.FACTORY:
            self._draw_factory_props(frame)
        else:
            self._draw_classroom_props(frame)

        # Wall / floor junction line.
        cv2.line(frame, (0, horizon), (w, horizon), (70, 78, 88), 2, cv2.LINE_AA)
        return frame

    def _draw_factory_props(self, frame: "np.ndarray") -> None:
        w, h = self.width, self.height
        # Machines along the left and right.
        for cx, cy, bw, bh in (
            (0.14, 0.52, 0.16, 0.13), (0.14, 0.78, 0.18, 0.15),
            (0.86, 0.55, 0.16, 0.13), (0.86, 0.82, 0.18, 0.15),
        ):
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            depth = int((y2 - y1) * 0.45)
            # Extruded body: a lit top face plus a darker front face reads as a
            # solid machine rather than a marking painted on the floor.
            cv2.rectangle(frame, (x1, y1 - depth), (x2, y2 - depth),
                          hex_to_bgr("#6E7A8A"), -1)
            cv2.rectangle(frame, (x1, y2 - depth), (x2, y2), hex_to_bgr("#3E4652"), -1)
            cv2.rectangle(frame, (x1, y1 - depth), (x2, y2),
                          hex_to_bgr("#8C99AB"), 2)
            cv2.line(frame, (x1, y2 - depth), (x2, y2 - depth),
                     hex_to_bgr("#8C99AB"), 1, cv2.LINE_AA)
            # Status lamp on the machine housing.
            cv2.rectangle(frame, (x1 + 8, y1 - depth + 8), (x1 + 28, y1 - depth + 20),
                          hex_to_bgr("#E8B84B"), -1)
        # Hazard floor striping in front of the danger area.
        for i in range(0, w, 34):
            cv2.line(frame, (i, int(h * 0.40)), (i + 16, int(h * 0.40)),
                     hex_to_bgr("#C8A13A"), 3, cv2.LINE_AA)
        # Conveyor band.
        cv2.rectangle(frame, (int(w * 0.30), int(h * 0.33)),
                      (int(w * 0.70), int(h * 0.38)), hex_to_bgr("#4E5661"), -1)

    def _draw_classroom_props(self, frame: "np.ndarray") -> None:
        w, h = self.width, self.height
        # Whiteboard on the back wall.
        cv2.rectangle(frame, (int(w * 0.30), int(h * 0.08)),
                      (int(w * 0.70), int(h * 0.24)), hex_to_bgr("#D8DEE6"), -1)
        cv2.rectangle(frame, (int(w * 0.30), int(h * 0.08)),
                      (int(w * 0.70), int(h * 0.24)), hex_to_bgr("#8A94A2"), 2)
        # Teacher desk.
        cv2.rectangle(frame, (int(w * 0.40), int(h * 0.30)),
                      (int(w * 0.60), int(h * 0.37)), hex_to_bgr("#6B5844"), -1)
        # Student desk grid, scaled by row for perspective.
        for row in range(4):
            y = 0.50 + row * 0.115
            scale = 0.45 + 0.55 * ((y - 0.28) / 0.7)
            dw = 0.085 * scale
            dh = 0.035 * scale
            for col in range(5):
                x = 0.18 + col * 0.16
                x1 = int((x - dw / 2) * w); y1 = int((y - dh / 2) * h)
                x2 = int((x + dw / 2) * w); y2 = int((y + dh / 2) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), hex_to_bgr("#7A6A52"), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), hex_to_bgr("#95856A"), 1)

    # -- actor rendering ------------------------------------------------------
    def _draw_person(self, frame: "np.ndarray", actor: SimActor) -> None:
        w, h = self.width, self.height
        scale = actor.scale()
        cx = actor.x * w
        base_y = actor.y * h
        body_h = 0.155 * scale * h
        body_w = 0.052 * scale * w
        head_r = max(2, int(body_w * 0.36))

        # Contact shadow grounds the figure on the floor.
        cv2.ellipse(frame, (int(cx), int(base_y)),
                    (int(body_w * 0.62), max(2, int(body_w * 0.22))),
                    0, 0, 360, (40, 44, 50), -1, cv2.LINE_AA)

        torso_top = base_y - body_h * 0.62
        gait = math.sin(actor.phase) * body_w * 0.28

        # Legs.
        leg_color = tuple(int(c * 0.6) for c in actor.color)
        cv2.line(frame, (int(cx), int(torso_top + body_h * 0.30)),
                 (int(cx - gait), int(base_y)), leg_color,
                 max(2, int(body_w * 0.20)), cv2.LINE_AA)
        cv2.line(frame, (int(cx), int(torso_top + body_h * 0.30)),
                 (int(cx + gait), int(base_y)), leg_color,
                 max(2, int(body_w * 0.20)), cv2.LINE_AA)
        # Torso.
        cv2.ellipse(frame, (int(cx), int(torso_top + body_h * 0.16)),
                    (int(body_w * 0.44), int(body_h * 0.28)),
                    0, 0, 360, actor.color, -1, cv2.LINE_AA)
        # Head (a plain filled circle - deliberately featureless; this system
        # performs no facial analysis of any kind).
        head_y = torso_top - head_r * 0.7
        cv2.circle(frame, (int(cx), int(head_y)), head_r,
                   tuple(int(min(255, c * 1.08)) for c in actor.color), -1, cv2.LINE_AA)

    def _draw_vehicle(self, frame: "np.ndarray", actor: SimActor) -> None:
        w, h = self.width, self.height
        scale = actor.scale()
        cx = actor.x * w
        base_y = actor.y * h
        bw = 0.20 * scale * w
        bh = 0.11 * scale * h

        cv2.ellipse(frame, (int(cx), int(base_y)),
                    (int(bw * 0.5), max(2, int(bh * 0.12))),
                    0, 0, 360, (40, 44, 50), -1, cv2.LINE_AA)
        x1, y1 = int(cx - bw / 2), int(base_y - bh)
        x2, y2 = int(cx + bw / 2), int(base_y - bh * 0.22)
        cv2.rectangle(frame, (x1, y1), (x2, y2), actor.color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      tuple(int(c * 0.65) for c in actor.color), 2)
        # Mast and wheels give the silhouette a forklift read.
        cv2.rectangle(frame, (x2 - int(bw * 0.10), int(base_y - bh * 1.6)),
                      (x2, y2), (90, 96, 104), -1)
        radius = max(2, int(bh * 0.20))
        cv2.circle(frame, (x1 + radius * 2, int(base_y - radius)), radius, (30, 32, 36), -1)
        cv2.circle(frame, (x2 - radius * 2, int(base_y - radius)), radius, (30, 32, 36), -1)

    def _apply_camera_effects(self, frame: "np.ndarray") -> "np.ndarray":
        """Grain, scanlines, vignette and burn-in overlay -> CCTV look."""
        noise = self._noise_bank[self._noise_index % len(self._noise_bank)]
        self._noise_index += 1
        out = np.clip(frame.astype(np.float32) + noise, 0, 255)

        # Subtle horizontal scanlines.
        out[::3, :, :] *= 0.965

        # Vignette, computed once and cached on the instance.
        vignette = getattr(self, "_vignette", None)
        if vignette is None:
            h, w = frame.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = h / 2.0, w / 2.0
            radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            vignette = np.clip(1.10 - 0.34 * radius ** 2, 0.55, 1.0).astype(np.float32)[..., None]
            self._vignette = vignette  # type: ignore[attr-defined]
        out *= vignette

        result = out.astype(np.uint8)

        # Timestamp and camera-ID burn-in, like a real DVR overlay.
        stamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(result, stamp, (12, self.height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, stamp, (12, self.height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 228, 236), 1, cv2.LINE_AA)
        # Camera-ID burn-in sits below y=30 so it never collides with the
        # pipeline HUD strip the renderer draws across the top of the frame.
        label = f"SIM-CAM-01  {self.mode.value.upper()}"
        cv2.putText(result, label, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 228, 236), 1, cv2.LINE_AA)
        # Recording dot, also clear of the HUD strip.
        if (self._index // 12) % 2 == 0:
            cv2.circle(result, (self.width - 22, 46), 5, hex_to_bgr("#FF4D5E"), -1, cv2.LINE_AA)
        return result

    # -- frame production -----------------------------------------------------
    def read(self) -> Optional[FramePacket]:
        if not self._opened or self._background is None:
            return None

        now = time.time()
        dt = min(0.2, max(0.0, now - self._last_step))
        self._last_step = now

        frame = self._background.copy()

        # Painter's algorithm: far actors first so nearer ones occlude them.
        actors = sorted(self._actors, key=lambda a: a.y)
        detections: List[Detection] = []
        for actor in actors:
            actor.step(dt, self._rng)
            if actor.kind == "vehicle":
                self._draw_vehicle(frame, actor)
            else:
                self._draw_person(frame, actor)

            # Ground-truth box, jittered so tracking has real work to do.
            x1, y1, x2, y2 = actor.bbox(self.width, self.height)
            jitter = 0.012 * (x2 - x1)
            x1 += self._rng.uniform(-jitter, jitter)
            x2 += self._rng.uniform(-jitter, jitter)
            y1 += self._rng.uniform(-jitter, jitter)
            y2 += self._rng.uniform(-jitter, jitter)

            # Occasional dropout emulates a real detector missing a frame and
            # forces the Kalman filter to coast.
            if self._rng.random() < 0.04:
                continue

            detections.append(
                Detection(
                    bbox=(max(0.0, x1), max(0.0, y1),
                          min(self.width - 1.0, x2), min(self.height - 1.0, y2)),
                    confidence=round(self._rng.uniform(0.62, 0.97), 3),
                    class_id=0 if actor.kind == "person" else 2,
                    class_name="person" if actor.kind == "person" else "truck",
                )
            )

        frame = self._apply_camera_effects(frame)
        self._index += 1
        return FramePacket(
            frame=frame, index=self._index, timestamp=now,
            source_name=self._info.name,
            synthetic=True,
            synthetic_detections=detections,
        )
