"""Page widgets shown in the main window's stack.

Each page owns its widgets, exposes intent as Qt signals and offers explicit
update methods.  No page reaches into the pipeline or another page.
"""

from __future__ import annotations

from sentinel.ui.pages.analytics import AnalyticsPage
from sentinel.ui.pages.dashboard import DashboardPage
from sentinel.ui.pages.events import EventsPage
from sentinel.ui.pages.live_monitor import LiveMonitorPage
from sentinel.ui.pages.models import ModelsPage
from sentinel.ui.pages.settings import SettingsPage
from sentinel.ui.pages.sources import SourcesPage
from sentinel.ui.pages.zones import ZonesPage

__all__ = [
    "DashboardPage", "LiveMonitorPage", "SourcesPage", "AnalyticsPage",
    "EventsPage", "ZonesPage", "ModelsPage", "SettingsPage",
]
