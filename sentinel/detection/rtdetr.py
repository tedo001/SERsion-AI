"""RT-DETR backend (HuggingFace ``transformers``).

All transformers-specific behaviour is confined to this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sentinel.config import LOGGER
from sentinel.deps import DEP_TORCH, DEP_TRANSFORMERS
from sentinel.detection.base import (
    BaseDetector, DetectorError, ModelState, resolve_torch_device,
)
from sentinel.models import AppSettings, Detection

__all__ = ["RTDETRDetector"]


class RTDETRDetector(BaseDetector):
    """RT-DETR backend backed by HuggingFace ``transformers``.

    Uses the public, documented API:

        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        processor = RTDetrImageProcessor.from_pretrained(checkpoint)
        model     = RTDetrForObjectDetection.from_pretrained(checkpoint)
        inputs    = processor(images=image, return_tensors="pt")
        outputs   = model(**inputs)
        results   = processor.post_process_object_detection(
                        outputs, target_sizes=..., threshold=...)

    RT-DETRv2 checkpoints are supported through the same processor/head API and
    are selected automatically when the installed transformers build exposes
    ``RTDetrV2ForObjectDetection``.
    """

    display_name = "RT-DETR"

    #: (checkpoint, label) pairs, smallest / fastest first.
    CHECKPOINTS: Tuple[Tuple[str, str], ...] = (
        ("PekingU/rtdetr_r18vd", "RT-DETR R18 (fast)"),
        ("PekingU/rtdetr_r50vd", "RT-DETR R50 (accurate)"),
        ("PekingU/rtdetr_v2_r18vd", "RT-DETRv2 R18 (fast)"),
        ("PekingU/rtdetr_v2_r50vd", "RT-DETRv2 R50 (accurate)"),
    )

    def __init__(self, settings: AppSettings, checkpoint: str = "PekingU/rtdetr_r18vd") -> None:
        super().__init__(settings)
        self.checkpoint = checkpoint
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._id2label: Dict[int, str] = {}
        self._torch: Optional[Any] = None
        self._use_half: bool = False

    @staticmethod
    def dependency_status() -> Tuple[bool, str]:
        if not DEP_TORCH.available:
            return False, f"PyTorch missing. {DEP_TORCH.install_hint}"
        if not DEP_TRANSFORMERS.available:
            return False, f"transformers missing. {DEP_TRANSFORMERS.install_hint}"
        transformers = DEP_TRANSFORMERS.module
        assert transformers is not None
        if not hasattr(transformers, "RTDetrForObjectDetection"):
            return False, (
                "Installed transformers build has no RTDetrForObjectDetection. "
                "Upgrade with: pip install -U transformers"
            )
        return True, f"transformers {DEP_TRANSFORMERS.version}"

    def load_model(self) -> None:
        with self._lock:
            ok, reason = self.dependency_status()
            if not ok:
                self._set_state(ModelState.UNAVAILABLE, reason)
                raise DetectorError(reason)

            self._set_state(ModelState.LOADING, f"Loading {self.checkpoint} ...")
            transformers = DEP_TRANSFORMERS.module
            torch = DEP_TORCH.module
            assert transformers is not None and torch is not None
            self._torch = torch

            try:
                # v2 checkpoints need the v2 head class when it exists.
                is_v2 = "_v2_" in self.checkpoint
                model_cls = getattr(transformers, "RTDetrV2ForObjectDetection", None) \
                    if is_v2 else None
                if model_cls is None:
                    model_cls = transformers.RTDetrForObjectDetection
                processor_cls = getattr(transformers, "RTDetrImageProcessor", None) \
                    or transformers.AutoImageProcessor

                self._processor = processor_cls.from_pretrained(self.checkpoint)
                model = model_cls.from_pretrained(self.checkpoint)
            except Exception as exc:  # noqa: BLE001 - network / hub / disk errors
                message = self._explain_load_failure(exc)
                self._set_state(ModelState.ERROR, message)
                raise DetectorError(message) from exc

            self._device = resolve_torch_device(self._settings.device)
            try:
                model = model.to(self._device)
                model.eval()
                # Half precision only makes sense (and is only numerically safe
                # in this codebase) on CUDA.
                self._use_half = bool(
                    self._settings.half_precision and self._device == "cuda"
                )
                if self._use_half:
                    model = model.half()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Falling back to CPU/full precision: %s", exc)
                self._device = "cpu"
                self._use_half = False
                model = model.float().to("cpu")

            self._model = model
            raw_labels = getattr(getattr(model, "config", None), "id2label", None) or {}
            self._id2label = {int(k): str(v) for k, v in raw_labels.items()}
            self._set_state(
                ModelState.READY,
                f"{self.checkpoint} on {self._device.upper()}"
                + (" (fp16)" if self._use_half else ""),
            )

    @staticmethod
    def _explain_load_failure(exc: BaseException) -> str:
        text = str(exc).lower()
        # The hub reports an unreachable network as a plain OSError about not
        # finding the repo, so match those phrasings too rather than only the
        # obvious connection errors.
        if any(k in text for k in (
            "connection", "offline", "resolve", "timed out", "network", "proxy",
            "max retries", "huggingface.co", "can't load", "couldn't connect",
            "is not a local folder", "not a valid model identifier",
        )):
            return ("Could not download RT-DETR weights - the HuggingFace hub is "
                    "unreachable (no network, a proxy, or offline mode). "
                    "Pre-download once on a connected machine with: "
                    "huggingface-cli download PekingU/rtdetr_r18vd")
        if "no space" in text or "disk" in text:
            return "Not enough disk space to cache the RT-DETR weights."
        return f"RT-DETR load failed: {type(exc).__name__}: {exc}"

    def predict(self, frame: "np.ndarray") -> List[Detection]:
        with self._lock:
            if self._model is None or self._processor is None or self._torch is None:
                return []
            torch = self._torch
            height, width = frame.shape[:2]

            small, _scale = self._letterbox_scale(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            try:
                inputs = self._processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                if self._use_half:
                    inputs = {
                        k: (v.half() if v.dtype == torch.float32 else v)
                        for k, v in inputs.items()
                    }
                with torch.no_grad():
                    outputs = self._model(**inputs)
                # Post-process straight to ORIGINAL frame size: target_sizes is
                # (height, width), so no manual rescaling is needed.
                target_sizes = torch.tensor([(height, width)], device=self._device)
                results = self._processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=float(self._settings.confidence),
                )
            except Exception as exc:  # noqa: BLE001 - never kill the pipeline
                LOGGER.error("RT-DETR inference error: %s", exc)
                raise DetectorError(f"RT-DETR inference failed: {exc}") from exc

            if not results:
                return []
            result = results[0]
            detections: List[Detection] = []
            scores = result.get("scores")
            labels = result.get("labels")
            boxes = result.get("boxes")
            if scores is None or labels is None or boxes is None:
                return []
            for score, label, box in zip(
                scores.detach().cpu().tolist(),
                labels.detach().cpu().tolist(),
                boxes.detach().cpu().tolist(),
            ):
                clipped = self._clip_box(box, width, height)
                if clipped is None:
                    continue
                class_id = int(label)
                detections.append(
                    Detection(
                        bbox=clipped,
                        confidence=float(score),
                        class_id=class_id,
                        class_name=self._id2label.get(class_id, f"class_{class_id}"),
                    )
                )
            return self._filter(detections)

    def unload_model(self) -> None:
        with self._lock:
            model, self._model = self._model, None
            self._processor = None
            self._id2label = {}
            torch = self._torch
            del model
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            self._set_state(ModelState.UNLOADED, "Not loaded")
