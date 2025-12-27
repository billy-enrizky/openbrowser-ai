"""Browser watchdogs for event-driven browser monitoring."""

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
