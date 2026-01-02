"""Browser watchdogs for event-driven browser monitoring.

Watchdogs are modular components that monitor specific aspects of browser
state and react to events. They follow the browser-use pattern for
event-driven architecture.

Architecture:
    - Each watchdog inherits from BaseWatchdog
    - Watchdogs declare events they listen to (LISTENS_TO) and emit (EMITS)
    - Watchdogs register handlers via attach_to_session()
    - Communication happens through the shared EventBus

Available Watchdogs:
    BaseWatchdog: Abstract base class for all watchdogs.
    AboutBlankWatchdog: Handles about:blank tab detection.
    CrashWatchdog: Monitors browser crashes and recovery.
    DefaultActionWatchdog: Handles default UI action behaviors.
    DOMWatchdog: Manages DOM tree caching and selector maps.
    DownloadsWatchdog: Monitors file download events.
    LocalBrowserWatchdog: Manages local browser subprocess lifecycle.
    PermissionsWatchdog: Grants browser permissions on connection.
    PopupsWatchdog: Handles JavaScript dialogs automatically.
    RecordingWatchdog: Manages video recording via CDP screencast.
    ScreenshotWatchdog: Handles screenshot requests.
    SecurityWatchdog: Enforces URL access policies.
    StorageStateWatchdog: Persists cookies and storage state.

Example:
    >>> watchdog = DOMWatchdog(
    ...     event_bus=session.event_bus,
    ...     browser_session=session
    ... )
    >>> watchdog.attach_to_session()
"""

from src.openbrowser.browser.watchdogs.base import BaseWatchdog
from src.openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog
from src.openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog
from src.openbrowser.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
from src.openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog
from src.openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog
from src.openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
from src.openbrowser.browser.watchdogs.permissions_watchdog import PermissionsWatchdog
from src.openbrowser.browser.watchdogs.popups_watchdog import PopupsWatchdog
from src.openbrowser.browser.watchdogs.recording_watchdog import RecordingWatchdog
from src.openbrowser.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog
from src.openbrowser.browser.watchdogs.security_watchdog import SecurityWatchdog
from src.openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

__all__ = [
    "BaseWatchdog",
    "AboutBlankWatchdog",
    "CrashWatchdog",
    "DefaultActionWatchdog",
    "DOMWatchdog",
    "DownloadsWatchdog",
    "LocalBrowserWatchdog",
    "PermissionsWatchdog",
    "PopupsWatchdog",
    "RecordingWatchdog",
    "ScreenshotWatchdog",
    "SecurityWatchdog",
    "StorageStateWatchdog",
]
