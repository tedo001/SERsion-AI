"""Optional-dependency probing.

Every optional package in the product goes through :class:`DependencyProbe`, so
the startup system-check screen reports what is *actually* importable rather
than what the code hopes is installed.  Nothing here raises: a missing package
is a reported state, not a crash.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

__all__ = [
    "DependencyProbe",
    "DEP_TORCH", "DEP_SUPERVISION", "DEP_TRANSFORMERS", "DEP_RFDETR",
    "DEP_PILLOW", "DEP_PSUTIL", "DEP_SCIPY", "OPTIONAL_DEPENDENCIES",
    "QT_AVAILABLE", "QT_IMPORT_ERROR", "core_requirements",
]


class DependencyProbe:
    """Import a module lazily and record availability + version + error.

    Every optional dependency in the application goes through this class so the
    startup dependency screen can report an accurate, non-fabricated status.
    """

    __slots__ = ("name", "module_name", "install_hint", "purpose", "_module",
                 "_checked", "_error")

    def __init__(self, name: str, module_name: str, install_hint: str, purpose: str) -> None:
        self.name = name
        self.module_name = module_name
        self.install_hint = install_hint
        self.purpose = purpose
        self._module: Optional[Any] = None
        self._checked: bool = False
        self._error: Optional[str] = None

    def _probe(self) -> None:
        if self._checked:
            return
        self._checked = True
        try:
            import importlib

            self._module = importlib.import_module(self.module_name)
        except BaseException as exc:  # noqa: BLE001 - some libs raise non-Exception
            self._module = None
            self._error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        self._probe()
        return self._module is not None

    @property
    def module(self) -> Optional[Any]:
        self._probe()
        return self._module

    @property
    def error(self) -> Optional[str]:
        self._probe()
        return self._error

    @property
    def version(self) -> str:
        self._probe()
        if self._module is None:
            return "-"
        for attr in ("__version__", "VERSION", "version"):
            value = getattr(self._module, attr, None)
            if isinstance(value, str):
                return value
        return "installed"

DEP_TORCH = DependencyProbe(
    "PyTorch", "torch", "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    "Neural network runtime for both detection backends.")
DEP_SUPERVISION = DependencyProbe(
    "Supervision", "supervision", "pip install supervision",
    "Detection container, annotators, zone / line-crossing utilities.")
DEP_TRANSFORMERS = DependencyProbe(
    "Transformers (RT-DETR)", "transformers", "pip install transformers",
    "Provides RTDetrForObjectDetection + RTDetrImageProcessor.")
DEP_RFDETR = DependencyProbe(
    "RF-DETR", "rfdetr", "pip install rfdetr",
    "Roboflow RF-DETR real-time detection transformer.")
DEP_PILLOW = DependencyProbe(
    "Pillow", "PIL", "pip install pillow",
    "Image container accepted by both model backends.")
DEP_PSUTIL = DependencyProbe(
    "psutil", "psutil", "pip install psutil",
    "CPU / RAM telemetry for the edge system panel.")
DEP_SCIPY = DependencyProbe(
    "SciPy", "scipy", "pip install scipy",
    "Optimal (Hungarian) track association; greedy fallback used when absent.")

OPTIONAL_DEPENDENCIES: Tuple[DependencyProbe, ...] = (
    DEP_TORCH, DEP_SUPERVISION, DEP_TRANSFORMERS, DEP_RFDETR,
    DEP_PILLOW, DEP_PSUTIL, DEP_SCIPY,
)


# -- PyQt6 ---------------------------------------------------------------------
# Qt is probed rather than imported here so that the headless core (self-test,
# CI, a machine with no display) never pays for it.  Modules under sentinel.ui
# and sentinel.runtime import PyQt6 directly; the entry point checks this flag
# before importing them.
try:  # pragma: no cover - environment dependent
    import PyQt6.QtCore as _qtcore  # noqa: F401

    QT_AVAILABLE = True
    QT_IMPORT_ERROR: Optional[str] = None
except BaseException as _exc:  # noqa: BLE001 - broken Qt installs raise widely
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = str(_exc)


def core_requirements() -> Tuple[Tuple[str, bool, str, str], ...]:
    """Report the mandatory runtime stack as (name, ok, version, install hint).

    NumPy and OpenCV are hard requirements of the core pipeline; PyQt6 is a hard
    requirement of the desktop console but not of the headless self-test.
    """
    import platform

    rows: list[Tuple[str, bool, str, str]] = [
        ("Python", True, platform.python_version(), ""),
    ]
    for name, module_name, hint in (
        ("NumPy", "numpy", "pip install numpy"),
        ("OpenCV", "cv2", "pip install opencv-python"),
    ):
        try:
            import importlib

            module = importlib.import_module(module_name)
            rows.append((name, True, getattr(module, "__version__", "installed"), ""))
        except ImportError:
            rows.append((name, False, "-", hint))
    rows.append((
        "PyQt6", QT_AVAILABLE, "installed" if QT_AVAILABLE else "-",
        "" if QT_AVAILABLE else "pip install PyQt6",
    ))
    return tuple(rows)
