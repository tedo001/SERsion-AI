"""Thermal / false-colour imaging stage.

TWO DIFFERENT THINGS LIVE HERE - the distinction matters:

1. PALETTE mode (works with any camera)
   Maps pixel BRIGHTNESS through a thermal-style colour ramp (iron, inferno,
   rainbow, white-hot, black-hot).  This is a display filter.  It produces the
   familiar thermal look, and "hotspots" it reports are bright regions, NOT hot
   regions.  No temperature is derived, because none exists in the data: a
   visible-light camera measures light, not heat.

2. RADIOMETRIC mode (requires a real thermal camera)
   Only available when the operator declares that the active source is a
   calibrated thermal camera and supplies its temperature span.  Pixel values
   are then mapped linearly onto that span, which is how the AGC-normalised
   greyscale streams from common USB thermal cameras (FLIR Lepton/PureThermal,
   Seek, Hikvision IR) are meant to be read.

Temperatures are reported ONLY in radiometric mode.  In palette mode every
temperature field is None and the UI shows "n/a".  Inventing a °C figure from
an ordinary webcam would be fabricated telemetry, and a surveillance product
that fabricates safety readings is worse than one that omits them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sentinel.config import LOGGER
from sentinel.models import AppSettings, ThermalMode, ThermalPalette

__all__ = ["ThermalProcessor", "Hotspot", "ThermalStatus"]


@dataclass(slots=True)
class Hotspot:
    """A contiguous high-intensity region."""

    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2
    peak: float                          # peak pixel intensity, 0-255
    mean: float                          # mean intensity within the region
    area_fraction: float                 # share of the frame
    temperature_c: Optional[float] = None  # None unless radiometric


@dataclass(slots=True)
class ThermalStatus:
    """Per-frame thermal analysis.

    ``*_temp_c`` fields are None whenever the source is not a calibrated
    thermal camera - the UI must render "n/a" rather than substituting a
    number.
    """

    active: bool = False
    mode: ThermalMode = ThermalMode.OFF
    palette: ThermalPalette = ThermalPalette.IRON
    radiometric: bool = False

    peak_intensity: float = 0.0          # 0-255
    mean_intensity: float = 0.0
    min_intensity: float = 0.0
    hotspot_fraction: float = 0.0        # share of frame above the threshold
    hotspots: Tuple[Hotspot, ...] = ()

    max_temp_c: Optional[float] = None
    avg_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None

    @property
    def unit(self) -> str:
        return "°C" if self.radiometric else "%"

    def format_max(self) -> str:
        if self.radiometric and self.max_temp_c is not None:
            return f"{self.max_temp_c:.1f}°C"
        return f"{self.peak_intensity / 2.55:.0f}%"

    def format_avg(self) -> str:
        if self.radiometric and self.avg_temp_c is not None:
            return f"{self.avg_temp_c:.1f}°C"
        return f"{self.mean_intensity / 2.55:.0f}%"


class ThermalProcessor:
    """Applies the thermal palette and measures intensity hotspots."""

    #: OpenCV colormaps behind each palette name.  IRON is built by hand
    #: because OpenCV has no direct equivalent of the classic iron ramp.
    _CV_MAPS = {
        ThermalPalette.INFERNO: cv2.COLORMAP_INFERNO,
        ThermalPalette.RAINBOW: cv2.COLORMAP_JET,
        ThermalPalette.PLASMA: cv2.COLORMAP_PLASMA,
    }

    #: Hotspot contours smaller than this fraction of the frame are noise.
    MIN_HOTSPOT_FRACTION = 0.0008

    #: At most this many hotspots are reported, largest first.
    MAX_HOTSPOTS = 6

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._iron_lut: Optional[np.ndarray] = None

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Palette
    # ------------------------------------------------------------------
    def _iron_colormap(self) -> "np.ndarray":
        """Classic iron ramp: black -> purple -> red -> orange -> yellow -> white."""
        if self._iron_lut is not None:
            return self._iron_lut
        stops = [
            (0.00, (0, 0, 0)),
            (0.18, (40, 0, 70)),
            (0.36, (120, 0, 130)),
            (0.52, (200, 20, 110)),
            (0.68, (255, 90, 30)),
            (0.84, (255, 180, 0)),
            (0.94, (255, 240, 120)),
            (1.00, (255, 255, 255)),
        ]
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            for index in range(len(stops) - 1):
                t0, c0 = stops[index]
                t1, c1 = stops[index + 1]
                if t0 <= t <= t1:
                    span = max(1e-6, t1 - t0)
                    f = (t - t0) / span
                    r = c0[0] + (c1[0] - c0[0]) * f
                    g = c0[1] + (c1[1] - c0[1]) * f
                    b = c0[2] + (c1[2] - c0[2]) * f
                    lut[i, 0] = (int(b), int(g), int(r))   # BGR
                    break
        self._iron_lut = lut
        return lut

    def colorize(self, grey: "np.ndarray", palette: ThermalPalette) -> "np.ndarray":
        """Map a single-channel image through the selected palette."""
        if palette is ThermalPalette.WHITE_HOT:
            return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        if palette is ThermalPalette.BLACK_HOT:
            return cv2.cvtColor(cv2.bitwise_not(grey), cv2.COLOR_GRAY2BGR)
        if palette is ThermalPalette.IRON:
            return cv2.applyColorMap(grey, self._iron_colormap())
        return cv2.applyColorMap(grey, self._CV_MAPS.get(palette, cv2.COLORMAP_INFERNO))

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def _to_temperature(self, intensity: float) -> Optional[float]:
        """Linear pixel -> temperature map, only for a declared thermal source."""
        if self._settings.thermal_mode is not ThermalMode.RADIOMETRIC:
            return None
        low = float(self._settings.thermal_min_c)
        high = float(self._settings.thermal_max_c)
        if high <= low:
            return None
        return low + (intensity / 255.0) * (high - low)

    def analyse(self, grey: "np.ndarray") -> ThermalStatus:
        """Measure intensity statistics and locate hotspots."""
        settings = self._settings
        radiometric = settings.thermal_mode is ThermalMode.RADIOMETRIC
        status = ThermalStatus(
            active=settings.thermal_mode is not ThermalMode.OFF,
            mode=settings.thermal_mode,
            palette=settings.thermal_palette,
            radiometric=radiometric,
        )

        peak = float(grey.max())
        status.peak_intensity = peak
        status.mean_intensity = float(grey.mean())
        status.min_intensity = float(grey.min())
        status.max_temp_c = self._to_temperature(peak)
        status.avg_temp_c = self._to_temperature(status.mean_intensity)
        status.min_temp_c = self._to_temperature(status.min_intensity)

        # Hotspots: threshold on the configured intensity percentile.
        cutoff = int(np.clip(settings.thermal_hotspot_threshold, 1, 99) * 2.55)
        mask = (grey >= cutoff).astype(np.uint8) * 255
        total = float(grey.shape[0] * grey.shape[1])
        status.hotspot_fraction = float(np.count_nonzero(mask)) / max(1.0, total)

        try:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        except cv2.error as exc:   # never break the pipeline on analysis
            LOGGER.debug("Hotspot contouring failed: %s", exc)
            return status

        spots: List[Hotspot] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            fraction = (w * h) / max(1.0, total)
            if fraction < self.MIN_HOTSPOT_FRACTION:
                continue
            region = grey[y:y + h, x:x + w]
            if region.size == 0:
                continue
            spot_peak = float(region.max())
            spots.append(Hotspot(
                bbox=(x, y, x + w, y + h),
                peak=spot_peak,
                mean=float(region.mean()),
                area_fraction=fraction,
                temperature_c=self._to_temperature(spot_peak),
            ))
        spots.sort(key=lambda s: s.area_fraction, reverse=True)
        status.hotspots = tuple(spots[:self.MAX_HOTSPOTS])
        return status

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------
    def process(self, frame: "np.ndarray") -> Tuple["np.ndarray", ThermalStatus]:
        """Return the display frame and the thermal analysis.

        The returned frame is for DISPLAY only; the detector keeps receiving the
        unmodified (or night-enhanced) image, because a palette-mapped frame is
        far outside a COCO-trained model's input distribution and would wreck
        detection accuracy.
        """
        if self._settings.thermal_mode is ThermalMode.OFF:
            return frame, ThermalStatus(mode=ThermalMode.OFF)

        try:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            status = self.analyse(grey)
            display = self.colorize(grey, self._settings.thermal_palette)
        except cv2.error as exc:
            LOGGER.warning("Thermal processing failed: %s", exc)
            return frame, ThermalStatus(mode=self._settings.thermal_mode)
        return display, status
