"""Crash watchdog following browser-use pattern."""

import logging
from typing import TYPE_CHECKING

from src.openbrowser.browser.watchdogs.base import BaseWatchdog
from src.openbrowser.browser.events import BrowserErrorEvent

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class CrashWatchdog(BaseWatchdog):
    """
    Watchdog that monitors for browser crashes and handles recovery.
    """

    browser_session: "BrowserSession"
    crash_count: int = 0
    max_crashes: int = 3

    def attach(self) -> None:
        """Attach event handlers."""
        self.browser_session.event_bus.on(BrowserErrorEvent, self.on_BrowserErrorEvent)
        logger.info("CrashWatchdog attached")

    async def on_BrowserErrorEvent(self, event: BrowserErrorEvent) -> None:
        """Handle browser error events."""
        self.crash_count += 1
        logger.error(f"Browser error detected: {event.error} (crash {self.crash_count}/{self.max_crashes})")

        if self.crash_count >= self.max_crashes:
            logger.error("Max crashes reached, stopping browser session")
            await self.browser_session.stop()

