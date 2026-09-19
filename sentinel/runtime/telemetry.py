"""Host CPU / RAM / GPU telemetry for the edge system panel."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from sentinel.config import LOGGER
from sentinel.deps import DEP_PSUTIL, DEP_TORCH

__all__ = ["SystemTelemetry", "PerformanceMonitor"]


@dataclass(slots=True)
class SystemTelemetry:
    """Host telemetry for the Edge System panel."""

    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_percent: float = 0.0
    process_mb: float = 0.0
    gpu_name: Optional[str] = None
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_util_percent: Optional[float] = None
    device: str = "cpu"
    available: bool = False


class PerformanceMonitor:
    """Samples host CPU / RAM / GPU telemetry on a cheap cadence."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = float(interval_s)
        self._last_sample = 0.0
        self._cached = SystemTelemetry()
        self._psutil = DEP_PSUTIL.module if DEP_PSUTIL.available else None
        self._process: Optional[Any] = None
        if self._psutil is not None:
            try:
                self._process = self._psutil.Process(os.getpid())
                self._psutil.cpu_percent(interval=None)   # prime the counter
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("psutil init failed: %s", exc)
                self._psutil = None

    def sample(self, device: str = "cpu", force: bool = False) -> SystemTelemetry:
        now = time.time()
        if not force and (now - self._last_sample) < self.interval_s:
            return self._cached
        self._last_sample = now

        telemetry = SystemTelemetry(device=device)
        if self._psutil is not None:
            try:
                telemetry.cpu_percent = float(self._psutil.cpu_percent(interval=None))
                memory = self._psutil.virtual_memory()
                telemetry.ram_total_mb = memory.total / (1024 ** 2)
                telemetry.ram_used_mb = (memory.total - memory.available) / (1024 ** 2)
                telemetry.ram_percent = float(memory.percent)
                if self._process is not None:
                    telemetry.process_mb = self._process.memory_info().rss / (1024 ** 2)
                telemetry.available = True
            except Exception as exc:  # noqa: BLE001 - telemetry is best effort
                LOGGER.debug("psutil sample failed: %s", exc)

        torch = DEP_TORCH.module
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    telemetry.gpu_name = torch.cuda.get_device_name(0)
                    free_b, total_b = torch.cuda.mem_get_info()
                    telemetry.gpu_mem_total_mb = total_b / (1024 ** 2)
                    telemetry.gpu_mem_used_mb = (total_b - free_b) / (1024 ** 2)
                    try:
                        telemetry.gpu_util_percent = float(torch.cuda.utilization())
                    except Exception:  # noqa: BLE001 - needs pynvml on some builds
                        telemetry.gpu_util_percent = None
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("CUDA telemetry failed: %s", exc)

        self._cached = telemetry
        return telemetry
