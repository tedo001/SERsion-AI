"""RF-DETR backend (Roboflow ``rfdetr``).

The rfdetr package has renamed its model classes and moved ``COCO_CLASSES``
between releases.  Every one of those differences is absorbed here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sentinel.config import LOGGER
from sentinel.deps import DEP_PILLOW, DEP_RFDETR, DEP_TORCH
from sentinel.detection.base import (
    BaseDetector, DetectorError, ModelState, resolve_torch_device,
)
from sentinel.models import AppSettings, Detection

__all__ = ["RFDETRDetector"]


class RFDETRDetector(BaseDetector):
    """RF-DETR backend backed by the Roboflow ``rfdetr`` package.

    Uses the documented API:

        from rfdetr import RFDETRMedium
        from rfdetr.assets.coco_classes import COCO_CLASSES
        model = RFDETRMedium()
        detections = model.predict(image, threshold=0.5)   # -> sv.Detections

    The package has renamed its size classes across releases (``RFDETRBase`` in
    early versions, ``RFDETRNano/Small/Medium/Large`` later) and moved
    ``COCO_CLASSES`` between ``rfdetr.util`` and ``rfdetr.assets``.  All of that
    variability is absorbed here so the rest of the application never sees it.
    """

    display_name = "RF-DETR"

    #: Preferred variant order, smallest first.  Resolved against what the
    #: installed package actually exposes - never assumed.
    VARIANT_PREFERENCE: Tuple[str, ...] = (
        "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRBase", "RFDETRLarge",
    )

    def __init__(self, settings: AppSettings, variant: str = "auto") -> None:
        super().__init__(settings)
        self.variant = variant
        self._model: Optional[Any] = None
        self._class_names: Dict[int, str] = {}
        self._pil: Optional[Any] = None

    # -- introspection helpers -------------------------------------------------
    @staticmethod
    def available_variants() -> List[str]:
        """Variant class names actually exposed by the installed package."""
        module = DEP_RFDETR.module
        if module is None:
            return []
        found = [
            name for name in RFDETRDetector.VARIANT_PREFERENCE
            if hasattr(module, name)
        ]
        if found:
            return found
        # Fall back to discovering any RFDETR* detection class.
        return [
            name for name in dir(module)
            if name.startswith("RFDETR") and "Seg" not in name
            and "Keypoint" not in name and isinstance(getattr(module, name), type)
        ]

    @staticmethod
    def dependency_status() -> Tuple[bool, str]:
        if not DEP_TORCH.available:
            return False, f"PyTorch missing. {DEP_TORCH.install_hint}"
        if not DEP_RFDETR.available:
            err = DEP_RFDETR.error or "not installed"
            return False, (
                f"rfdetr not importable ({err}). Install with: pip install rfdetr"
            )
        if not RFDETRDetector.available_variants():
            return False, (
                "The installed rfdetr package exposes no RFDETR* model class. "
                "Upgrade with: pip install -U rfdetr"
            )
        return True, f"rfdetr {DEP_RFDETR.version}"

    @staticmethod
    def _load_coco_classes() -> Dict[int, str]:
        """Locate COCO_CLASSES across the package layouts rfdetr has shipped."""
        import importlib

        for path in ("rfdetr.assets.coco_classes", "rfdetr.util.coco_classes"):
            try:
                module = importlib.import_module(path)
            except Exception:  # noqa: BLE001
                continue
            table = getattr(module, "COCO_CLASSES", None)
            if isinstance(table, dict):
                return {int(k): str(v) for k, v in table.items()}
            if isinstance(table, (list, tuple)):
                return {i: str(v) for i, v in enumerate(table)}
        return {}

    def load_model(self) -> None:
        with self._lock:
            ok, reason = self.dependency_status()
            if not ok:
                self._set_state(ModelState.UNAVAILABLE, reason)
                raise DetectorError(reason)

            module = DEP_RFDETR.module
            assert module is not None
            variants = self.available_variants()
            name = self.variant if self.variant in variants else variants[0]
            self._set_state(ModelState.LOADING, f"Loading {name} ...")

            self._device = resolve_torch_device(self._settings.device)
            model_cls = getattr(module, name)

            # Constructor kwargs differ between releases; try the richest form
            # first and degrade gracefully rather than guessing a single API.
            attempts: List[Dict[str, Any]] = [
                {"device": self._device, "resolution": self._rf_resolution()},
                {"device": self._device},
                {},
            ]
            last_error: Optional[BaseException] = None
            model = None
            for kwargs in attempts:
                try:
                    model = model_cls(**kwargs)
                    break
                except TypeError as exc:
                    last_error = exc           # unsupported kwarg -> try simpler
                    continue
                except Exception as exc:  # noqa: BLE001 - weights download etc.
                    last_error = exc
                    break

            if model is None:
                message = self._explain_load_failure(last_error)
                self._set_state(ModelState.ERROR, message)
                raise DetectorError(message)

            # Optional inference optimisation offered by some releases.
            optimize = getattr(model, "optimize_for_inference", None)
            if callable(optimize):
                try:
                    optimize()
                    LOGGER.info("RF-DETR: optimize_for_inference() applied")
                except Exception as exc:  # noqa: BLE001 - purely opportunistic
                    LOGGER.debug("RF-DETR optimize_for_inference skipped: %s", exc)

            self._model = model
            self._class_names = self._load_coco_classes()
            self._pil = DEP_PILLOW.module
            self._set_state(
                ModelState.READY, f"{name} on {self._device.upper()}"
            )

    #: RF-DETR resolutions must satisfy two separate divisibility constraints:
    #: the DINOv2 patch grid (56) and, on the windowed Nano/Small variants, the
    #: window block size (patch_size 16 * num_windows 2 = 32).  Only multiples of
    #: their LCM (224) satisfy both, and a value that violates the second one is
    #: accepted by the constructor but fails later inside predict().
    RESOLUTION_STEP: int = 224

    def _rf_resolution(self) -> int:
        """Snap the configured inference size to a resolution RF-DETR accepts."""
        size = int(self._settings.inference_size)
        steps = max(1, int(round(size / float(self.RESOLUTION_STEP))))
        return steps * self.RESOLUTION_STEP

    @staticmethod
    def _explain_load_failure(exc: Optional[BaseException]) -> str:
        if exc is None:
            return "RF-DETR model could not be constructed."
        text = str(exc).lower()
        if any(k in text for k in ("connection", "offline", "resolve", "timed out",
                                   "network", "proxy", "max retries", "download")):
            return ("Could not download RF-DETR weights (no network access). "
                    "Run once with connectivity to populate the local cache.")
        return f"RF-DETR load failed: {type(exc).__name__}: {exc}"

    def predict(self, frame: "np.ndarray") -> List[Detection]:
        with self._lock:
            if self._model is None:
                return []
            height, width = frame.shape[:2]
            small, scale = self._letterbox_scale(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # A PIL image is the input form accepted by every rfdetr release;
            # fall back to the raw ndarray when Pillow is unavailable.
            image: Any = rgb
            if self._pil is not None:
                try:
                    image = self._pil.Image.fromarray(rgb)
                except Exception:  # noqa: BLE001
                    image = rgb

            try:
                raw = self._model.predict(
                    image, threshold=float(self._settings.confidence)
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("RF-DETR inference error: %s", exc)
                raise DetectorError(f"RF-DETR inference failed: {exc}") from exc

            # predict() may return a list when given a batch; take the first.
            if isinstance(raw, (list, tuple)):
                if not raw:
                    return []
                raw = raw[0]
            return self._filter(self._from_sv(raw, scale, width, height))

    def _from_sv(
        self, sv_detections: Any, scale: float, width: int, height: int
    ) -> List[Detection]:
        """Normalise a ``supervision.Detections`` object into our Detection list."""
        xyxy = getattr(sv_detections, "xyxy", None)
        if xyxy is None or len(xyxy) == 0:
            return []
        confidence = getattr(sv_detections, "confidence", None)
        class_id = getattr(sv_detections, "class_id", None)
        data = getattr(sv_detections, "data", None) or {}
        names = data.get("class_name") if isinstance(data, dict) else None

        out: List[Detection] = []
        for i in range(len(xyxy)):
            box = [float(v) * scale for v in xyxy[i]]
            clipped = self._clip_box(box, width, height)
            if clipped is None:
                continue
            cid = int(class_id[i]) if class_id is not None else -1
            if names is not None and i < len(names):
                cname = str(names[i])
            else:
                cname = self._class_names.get(cid, f"class_{cid}")
            conf = float(confidence[i]) if confidence is not None else 1.0
            out.append(
                Detection(bbox=clipped, confidence=conf, class_id=cid, class_name=cname)
            )
        return out

    def unload_model(self) -> None:
        with self._lock:
            model, self._model = self._model, None
            del model
            torch = DEP_TORCH.module
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            self._set_state(ModelState.UNLOADED, "Not loaded")
