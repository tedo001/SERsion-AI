"""LOFOP Sentinel - Edge Vision Intelligence Platform.

A modular PyQt6 desktop application for edge computer-vision surveillance in
two deployment scenarios: classroom safety monitoring and factory/workplace
safety monitoring.

The package performs OBJECT DETECTION and OBJECT TRACKING only.  It does not
perform facial recognition, biometric identification, or any form of identity
inference, and it never stores biometric data.

Package layout
--------------
    sentinel.config       paths, logging, application metadata
    sentinel.theme        colour tokens and the Qt style sheet
    sentinel.deps         optional-dependency probing (never assumes a package)
    sentinel.models       enums and dataclasses shared by every layer
    sentinel.detection    BaseDetector interface + RT-DETR / RF-DETR backends
    sentinel.tracking     Kalman filter, tracks, and the multi-object tracker
    sentinel.sources      video file, webcam and synthetic simulation sources
    sentinel.vision       zones, supervision adapter, renderer, events, analytics
    sentinel.storage      SQLite event store and settings/zone persistence
    sentinel.runtime      telemetry, bounded frame queue, workers, controller
    sentinel.ui           Qt widgets, pages and the main window
    sentinel.selftest     headless pipeline verification

Layering rule: imports flow one way only, from the bottom of that list upward.
Only ``sentinel.ui`` and ``sentinel.runtime.workers``/``controller`` import Qt,
so the entire core pipeline runs (and is testable) without a display.
"""

from __future__ import annotations

APP_NAME = "LOFOP Sentinel"
APP_SUBTITLE = "Edge Vision Intelligence Platform"
APP_VERSION = "1.0.0"
ORG_NAME = "LOFOP"

__all__ = ["APP_NAME", "APP_SUBTITLE", "APP_VERSION", "ORG_NAME"]
