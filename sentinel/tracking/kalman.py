"""Kalman filtering and multi-object tracking.

A SORT-style constant-velocity filter implemented directly on NumPy, so the
product carries no extra filtering dependency.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from typing import Deque, List, Sequence, Tuple

import numpy as np

from sentinel.config import LOGGER
from sentinel.deps import DEP_SCIPY
from sentinel.models import Detection, TrackedObject, TrackPhase

__all__ = ["iou_matrix", "KalmanBoxFilter", "Track", "KalmanTracker"]


def iou_matrix(boxes_a: "np.ndarray", boxes_b: "np.ndarray") -> "np.ndarray":
    """Vectorised IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]   # (N, 1, 4)
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]   # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area_a = np.clip(a[..., 2] - a[..., 0], 0.0, None) * np.clip(a[..., 3] - a[..., 1], 0.0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0.0, None) * np.clip(b[..., 3] - b[..., 1], 0.0, None)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-6), 0.0).astype(np.float32)


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over the bounding-box state

        x = [cx, cy, s, r, vx, vy, vs]^T

    where ``s`` is box area and ``r`` is the aspect ratio (width / height).
    This is the classic SORT parameterisation, implemented directly on NumPy so
    the application carries no extra filtering dependency.
    """

    __slots__ = ("x", "P", "F", "H", "Q", "R")

    def __init__(self, bbox: Tuple[float, float, float, float]) -> None:
        # State transition: position integrates velocity once per frame.
        self.F = np.eye(7, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement model observes (cx, cy, s, r) directly.
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Measurement noise: area and aspect are noisier than centre position.
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        # Process noise + initial covariance: velocities start highly uncertain.
        self.P = np.eye(7, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.Q = np.eye(7, dtype=np.float64)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4, 0] = self._to_z(bbox)

    # -- conversions ---------------------------------------------------------
    @staticmethod
    def _to_z(bbox: Sequence[float]) -> "np.ndarray":
        x1, y1, x2, y2 = (float(v) for v in bbox)
        w = max(1e-3, x2 - x1)
        h = max(1e-3, y2 - y1)
        return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h], dtype=np.float64)

    @staticmethod
    def _to_bbox(state: "np.ndarray") -> Tuple[float, float, float, float]:
        cx, cy, s, r = (float(state[i, 0]) for i in range(4))
        s = max(s, 1e-3)
        r = max(r, 1e-3)
        w = math.sqrt(s * r)
        h = s / w if w > 1e-6 else 1e-3
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    # -- filter steps --------------------------------------------------------
    def predict(self) -> Tuple[float, float, float, float]:
        # Guard against the area going negative under a strong negative vs.
        if self.x[6, 0] + self.x[2, 0] <= 0.0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox(self.x)

    def update(self, bbox: Sequence[float]) -> None:
        z = self._to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:          # singular innovation covariance
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        identity = np.eye(7, dtype=np.float64)
        self.P = (identity - K @ self.H) @ self.P

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._to_bbox(self.x)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self.x[4, 0]), float(self.x[5, 0]))


class Track:
    """A single tracked object: Kalman state + identity + lifecycle bookkeeping."""

    __slots__ = ("track_id", "kf", "class_id", "class_name", "confidence",
                 "age", "hits", "hit_streak", "time_since_update", "phase",
                 "last_seen", "trail", "created_at", "_class_votes")

    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.kf = KalmanBoxFilter(detection.bbox)
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.phase = TrackPhase.TENTATIVE
        self.last_seen = time.time()
        self.created_at = self.last_seen
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=48)
        # Majority vote over recent class labels keeps the label stable when the
        # detector flickers between e.g. "person" and "backpack".
        self._class_votes: Counter = Counter({detection.class_name: 1})
        self.trail.append(detection.center)

    def predict(self) -> Tuple[float, float, float, float]:
        bbox = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return bbox

    def update(self, detection: Detection) -> None:
        self.kf.update(detection.bbox)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = detection.confidence
        self.last_seen = time.time()
        self._class_votes[detection.class_name] += 1
        self.class_name = self._class_votes.most_common(1)[0][0]
        if detection.class_name == self.class_name:
            self.class_id = detection.class_id
        bbox = self.kf.bbox
        self.trail.append(((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5))

    def refresh_phase(self, min_hits: int) -> None:
        if self.time_since_update == 0:
            self.phase = (TrackPhase.CONFIRMED if self.hits >= min_hits
                          else TrackPhase.TENTATIVE)
        elif self.hits >= min_hits:
            self.phase = TrackPhase.COASTING
        else:
            self.phase = TrackPhase.TENTATIVE

    def snapshot(self) -> TrackedObject:
        bbox = self.kf.bbox
        vx, vy = self.kf.velocity
        return TrackedObject(
            track_id=self.track_id,
            bbox=bbox,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            position=((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5),
            velocity=(vx, vy),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            phase=self.phase,
            last_seen=self.last_seen,
            trail=tuple(self.trail),
        )


class KalmanTracker:
    """Multi-object tracker: Kalman prediction + IoU / distance association.

    Association strategy:
        1. Predict every existing track forward one frame.
        2. Build an IoU cost matrix against the new detections.
        3. Solve it optimally with SciPy's Hungarian algorithm when available,
           otherwise greedily by descending IoU (both give stable IDs).
        4. Reject matches below ``iou_threshold``.
        5. Unmatched detections spawn tentative tracks; unmatched tracks coast on
           Kalman prediction until ``max_age`` frames pass.

    IDs are assigned once at birth and never reshuffled, so a track keeps its
    identity for as long as it survives.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[Track] = []
        self._next_id: int = 1
        self._lost: List[TrackedObject] = []

    # -- configuration --------------------------------------------------------
    def configure(self, max_age: int, min_hits: int, iou_threshold: float) -> None:
        self.max_age = max(1, int(max_age))
        self.min_hits = max(1, int(min_hits))
        self.iou_threshold = float(np.clip(iou_threshold, 0.01, 0.95))

    def reset(self) -> None:
        self._tracks.clear()
        self._lost.clear()
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks if t.phase is not TrackPhase.TENTATIVE)

    # -- association ----------------------------------------------------------
    def _associate(
        self, detections: List[Detection], predicted: "np.ndarray"
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Return (matches as (track_idx, det_idx), unmatched_tracks, unmatched_dets)."""
        n_tracks = len(predicted)
        n_dets = len(detections)
        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        ious = iou_matrix(predicted, det_boxes)

        # Class-aware gating: matching a person track to a truck detection is
        # almost always an association error, so penalise cross-class pairs.
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                if track.class_name != det.class_name:
                    ious[ti, di] *= 0.5

        matches: List[Tuple[int, int]] = []
        if DEP_SCIPY.available:
            try:
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(-ious)
                matches = [(int(r), int(c)) for r, c in zip(rows, cols)]
            except Exception as exc:  # noqa: BLE001 - fall back to greedy
                LOGGER.debug("SciPy assignment failed (%s); using greedy", exc)
                matches = []
        if not matches:
            matches = self._greedy_match(ious)

        final: List[Tuple[int, int]] = []
        matched_tracks = set()
        matched_dets = set()
        for ti, di in matches:
            if ious[ti, di] < self.iou_threshold:
                continue
            final.append((ti, di))
            matched_tracks.add(ti)
            matched_dets.add(di)

        unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
        unmatched_dets = [i for i in range(n_dets) if i not in matched_dets]
        return final, unmatched_tracks, unmatched_dets

    @staticmethod
    def _greedy_match(ious: "np.ndarray") -> List[Tuple[int, int]]:
        """Descending-IoU greedy matching - the dependency-free fallback."""
        pairs: List[Tuple[int, int]] = []
        used_rows: set = set()
        used_cols: set = set()
        flat = np.argsort(ious, axis=None)[::-1]
        n_cols = ious.shape[1]
        for index in flat:
            r, c = divmod(int(index), n_cols)
            if ious[r, c] <= 0.0:
                break
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
        return pairs

    # -- main step ------------------------------------------------------------
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Advance the tracker by one frame and return the current snapshots."""
        self._lost = []

        # 1. Predict.  Tracks whose filter diverges into NaN are dropped.
        predicted: List[Tuple[float, float, float, float]] = []
        alive: List[Track] = []
        for track in self._tracks:
            bbox = track.predict()
            if not all(math.isfinite(v) for v in bbox):
                LOGGER.debug("Dropping diverged track %s", track.track_id)
                continue
            predicted.append(bbox)
            alive.append(track)
        self._tracks = alive
        predicted_array = (np.array(predicted, dtype=np.float32)
                           if predicted else np.zeros((0, 4), dtype=np.float32))

        # 2/3. Associate and update.
        matches, _unmatched_tracks, unmatched_dets = self._associate(
            detections, predicted_array
        )
        for track_idx, det_idx in matches:
            self._tracks[track_idx].update(detections[det_idx])

        # 4. Spawn tracks for unmatched detections.
        for det_idx in unmatched_dets:
            track = Track(self._next_id, detections[det_idx])
            self._next_id += 1
            self._tracks.append(track)

        # 5. Retire stale tracks.
        survivors: List[Track] = []
        for track in self._tracks:
            track.refresh_phase(self.min_hits)
            if track.time_since_update > self.max_age:
                track.phase = TrackPhase.LOST
                if track.hits >= self.min_hits:
                    self._lost.append(track.snapshot())
                continue
            survivors.append(track)
        self._tracks = survivors

        # Publish only tracks that are confirmed (or coasting after confirmation),
        # which prevents one-frame detector noise from flashing boxes on screen.
        return [
            t.snapshot() for t in self._tracks
            if t.hits >= self.min_hits or t.phase is TrackPhase.CONFIRMED
        ]

    def pop_lost(self) -> List[TrackedObject]:
        """Tracks that died during the last update (drives OBJECT_LOST events)."""
        lost, self._lost = self._lost, []
        return lost
