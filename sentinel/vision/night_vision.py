"""Low-light ("night vision") analysis and enhancement.

What this stage actually does
-----------------------------
It measures scene luminance, decides whether the camera is in a low-light
condition, and - when it is - rebuilds local contrast so that a detector has
something to work with:

    1. CLAHE on the L channel of LAB  (recovers local contrast without
       blowing out the few bright regions a night scene has)
    2. gamma correction                (lifts the mid-tones)
    3. optional edge-preserving denoise (the gain in steps 1-2 amplifies sensor
       noise, which otherwise produces spurious detections)

SCOPE AND LIMITS - read before relying on this:
    * This is image enhancement, NOT a thermal or infrared sensor.  It can only
      amplify signal the camera actually captured; in total darkness there is
      nothing to amplify and detection will not work.
    * It does not add a detection capability of its own.  Enhancement feeds the
      SAME detector; if no model backend is loaded, nothing will be detected in
      the dark any more than in daylight.
    * IR-illuminated cameras produce near-monochrome frames.  That is reported
      as a scene property (``monochrome``), inferred from saturation - the
      application is not reading an IR sensor directly.

The green tint is a cosmetic display filter applied AFTER detection, so it can
never influence what the model sees.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from sentinel.config import LOGGER
from sentinel.models import AppSettings, NightMode, NightStatus

__all__ = ["NightVisionProcessor"]


class NightVisionProcessor:
    """Detects low-light conditions and enhances frames for detection.

    Cheap by construction: scene metrics are measured on a 64x36 thumbnail, and
    in AUTO mode no enhancement runs at all while the scene is bright.
    """

    #: Thumbnail size used for luminance/saturation metrics.
    PROBE_SIZE = (64, 36)

    #: Hysteresis band (luma units) so a scene hovering at the threshold does
    #: not flap between day and night every frame.
    HYSTERESIS = 12.0

    #: Mean saturation below which the frame is treated as monochrome/IR.
    MONOCHROME_SATURATION = 24.0

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._active = False
        self._clahe: Optional[cv2.CLAHE] = None
        self._clahe_clip = -1.0
        self._gamma_lut: Optional[np.ndarray] = None
        self._gamma_value = -1.0

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    def reset(self) -> None:
        self._active = False

    # ------------------------------------------------------------------
    # Scene measurement
    # ------------------------------------------------------------------
    def measure(self, frame: "np.ndarray") -> Tuple[float, float]:
        """Return (mean luminance, mean saturation) from a small thumbnail."""
        thumb = cv2.resize(frame, self.PROBE_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))
        luminance = float(np.mean(hsv[:, :, 2]))
        return luminance, saturation

    def _should_enhance(self, luminance: float) -> bool:
        """Decide with hysteresis whether enhancement applies this frame."""
        mode = self._settings.night_mode
        if mode is NightMode.OFF:
            return False
        if mode is NightMode.ON:
            return True
        threshold = float(self._settings.night_threshold)
        if self._active:
            # Already enhancing: only disengage once clearly bright again.
            return luminance < threshold + self.HYSTERESIS
        return luminance < threshold

    # ------------------------------------------------------------------
    # Enhancement
    # ------------------------------------------------------------------
    def _get_clahe(self, clip_limit: float) -> cv2.CLAHE:
        if self._clahe is None or abs(clip_limit - self._clahe_clip) > 0.05:
            self._clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(8, 8)
            )
            self._clahe_clip = clip_limit
        return self._clahe

    def _get_gamma_lut(self, gamma: float) -> "np.ndarray":
        """Cache the 256-entry LUT; rebuilding it per frame is wasteful.

        Follows the standard convention: out = (i/255) ** (1/gamma), so
        ``gamma > 1`` BRIGHTENS the mid-tones.
        """
        if self._gamma_lut is None or abs(gamma - self._gamma_value) > 0.01:
            inv = 1.0 / max(0.05, gamma)
            self._gamma_lut = np.clip(
                ((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255
            ).astype(np.uint8)
            self._gamma_value = gamma
        return self._gamma_lut

    def enhance(self, frame: "np.ndarray", luminance: float) -> Tuple["np.ndarray", float, float]:
        """Apply the enhancement chain.  Returns (frame, gain, clip_limit)."""
        strength = float(np.clip(self._settings.night_strength, 0.0, 1.0))

        # Darker scenes get more aggressive treatment, scaled by strength.
        darkness = float(np.clip(1.0 - luminance / 128.0, 0.0, 1.0))
        clip_limit = 1.0 + 4.0 * strength * darkness
        gamma = 1.0 + 2.2 * strength * darkness       # >1 brightens

        # 1. Local contrast on the luminance channel only, so colours survive.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = self._get_clahe(clip_limit).apply(lightness)
        out = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)),
                           cv2.COLOR_LAB2BGR)

        # 2. Mid-tone lift.
        out = cv2.LUT(out, self._get_gamma_lut(gamma))

        # 3. Edge-preserving denoise - gain amplifies sensor noise, and noise
        #    produces phantom detections.
        if self._settings.night_denoise and strength > 0.25:
            out = cv2.bilateralFilter(out, d=5, sigmaColor=45, sigmaSpace=45)

        return out, gamma, clip_limit

    @staticmethod
    def apply_tint(frame: "np.ndarray") -> "np.ndarray":
        """Cosmetic green phosphor look.

        Display only - applied after detection, so it cannot affect the model.
        """
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tinted = np.zeros_like(frame)
        tinted[:, :, 1] = grey                       # green channel
        tinted[:, :, 0] = (grey * 0.22).astype(np.uint8)
        tinted[:, :, 2] = (grey * 0.16).astype(np.uint8)
        return tinted

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------
    def process(self, frame: "np.ndarray") -> Tuple["np.ndarray", NightStatus]:
        """Analyse the frame and enhance it when the scene warrants it.

        Returns the frame to hand to the detector plus the scene status.  The
        input frame is never mutated.
        """
        status = NightStatus(mode=self._settings.night_mode)
        try:
            luminance, saturation = self.measure(frame)
        except cv2.error as exc:           # malformed frame: pass it through
            LOGGER.debug("Night-vision measurement failed: %s", exc)
            return frame, status

        status.luminance = luminance
        status.saturation = saturation
        status.monochrome = saturation < self.MONOCHROME_SATURATION

        if not self._should_enhance(luminance):
            self._active = False
            return frame, status

        try:
            enhanced, gain, clip_limit = self.enhance(frame, luminance)
        except cv2.error as exc:           # never break the pipeline on this
            LOGGER.warning("Night-vision enhancement failed: %s", exc)
            self._active = False
            return frame, status

        self._active = True
        status.active = True
        status.gain = gain
        status.clip_limit = clip_limit
        return enhanced, status
