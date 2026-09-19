"""Optional bridge to the ``supervision`` library.

The application's own data model is authoritative; this adapter only borrows
supervision's detection container and annotators when the package is installed,
and degrades to ``None`` (meaning "use the OpenCV renderer") when it is not.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from sentinel.config import LOGGER
from sentinel.deps import DEP_SUPERVISION
from sentinel.models import TrackedObject
from sentinel.theme import Palette

__all__ = ["SupervisionAdapter"]


class SupervisionAdapter:
    """Thin, entirely optional bridge to the ``supervision`` library.

    The application's own data model is authoritative; this adapter only
    borrows supervision's detection container and annotators when the package is
    installed.  Every method degrades to a no-op / ``None`` when it is not, and
    the OpenCV renderer below covers the same ground, so nothing in the product
    depends on supervision being present.
    """

    def __init__(self) -> None:
        self.sv: Optional[Any] = DEP_SUPERVISION.module if DEP_SUPERVISION.available else None
        self._box_annotator: Optional[Any] = None
        self._label_annotator: Optional[Any] = None
        self._trace_annotator: Optional[Any] = None
        self._ready = False
        if self.sv is not None:
            self._build_annotators()

    @property
    def available(self) -> bool:
        return self._ready

    def _build_annotators(self) -> None:
        """Construct annotators defensively - their kwargs vary across releases."""
        sv = self.sv
        assert sv is not None
        try:
            palette = sv.ColorPalette.from_hex(list(Palette.TRACK_COLORS))
        except Exception:  # noqa: BLE001 - older/newer palette APIs
            palette = None
        try:
            kwargs: Dict[str, Any] = {"thickness": 2}
            if palette is not None:
                kwargs["color"] = palette
            self._box_annotator = sv.BoxAnnotator(**kwargs)
            label_kwargs: Dict[str, Any] = {"text_scale": 0.4, "text_padding": 4}
            if palette is not None:
                label_kwargs["color"] = palette
            self._label_annotator = sv.LabelAnnotator(**label_kwargs)
            try:
                self._trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)
            except Exception:  # noqa: BLE001 - trace annotator is a bonus only
                self._trace_annotator = None
            self._ready = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("supervision annotators unavailable: %s", exc)
            self._ready = False

    def to_sv_detections(self, tracks: Sequence[TrackedObject]) -> Optional[Any]:
        """Convert our tracks into a ``sv.Detections`` instance."""
        if self.sv is None or not tracks:
            return None
        try:
            xyxy = np.array([t.bbox for t in tracks], dtype=np.float32)
            confidence = np.array([t.confidence for t in tracks], dtype=np.float32)
            class_id = np.array([max(0, t.class_id) for t in tracks], dtype=int)
            tracker_id = np.array([t.track_id for t in tracks], dtype=int)
            return self.sv.Detections(
                xyxy=xyxy, confidence=confidence, class_id=class_id,
                tracker_id=tracker_id,
                data={"class_name": np.array([t.class_name for t in tracks])},
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("sv.Detections conversion failed: %s", exc)
            return None

    def annotate(
        self, frame: "np.ndarray", tracks: Sequence[TrackedObject],
        labels: Sequence[str], draw_traces: bool,
    ) -> Optional["np.ndarray"]:
        """Annotate with supervision; returns None to signal 'use the fallback'."""
        if not self._ready or self._box_annotator is None:
            return None
        detections = self.to_sv_detections(tracks)
        if detections is None:
            return None
        try:
            out = frame
            if draw_traces and self._trace_annotator is not None:
                out = self._trace_annotator.annotate(out, detections)
            out = self._box_annotator.annotate(out, detections)
            if self._label_annotator is not None and labels:
                out = self._label_annotator.annotate(out, detections, list(labels))
            return out
        except Exception as exc:  # noqa: BLE001 - never break rendering
            LOGGER.debug("supervision annotation failed, using fallback: %s", exc)
            self._ready = False
            return None
