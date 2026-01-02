"""Crash watchdog following browser-use pattern.

This module provides the CrashWatchdog which monitors for browser crashes
and errors, handling recovery or graceful shutdown.

Classes:
    CrashWatchdog: Monitors browser crashes and handles recovery.
"""

import logging
from typing import TYPE_CHECKING

from src.openbrowser.browser.watchdogs.base import BaseWatchdog
from src.openbrowser.browser.events import BrowserErrorEvent

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class CrashWatchdog(BaseWatchdog):
    """Watchdog that monitors for browser crashes and handles recovery.

    Tracks crash count and stops the browser session if max_crashes is
    exceeded to prevent infinite crash loops.

    Attributes:
        crash_count: Number of crashes observed in current session.
        max_crashes: Maximum crashes before forcing session stop.

    Listens to:
        BrowserErrorEvent: Detects browser errors and crashes.

    Example:
        >>> watchdog = CrashWatchdog(
        ...     event_bus=bus,
        ...     browser_session=session,
        ...     max_crashes=5
        ... )
    """

    browser_session: "BrowserSession"
    crash_count: int = 0
    max_crashes: int = 3

    def attach(self) -> None:
        """Attach event handlers.

        Registers BrowserErrorEvent handler for crash detection.
        """
        self.browser_session.event_bus.on(BrowserErrorEvent, self.on_BrowserErrorEvent)
        logger.info("CrashWatchdog attached")

    async def on_BrowserErrorEvent(self, event: BrowserErrorEvent) -> None:
        """Handle browser error events.

        Increments crash count and stops session if max_crashes exceeded.

        Args:
            event: BrowserErrorEvent with error details.
        """
        self.crash_count += 1
        logger.error(f"Browser error detected: {event.error} (crash {self.crash_count}/{self.max_crashes})")

        if self.crash_count >= self.max_crashes:
            logger.error("Max crashes reached, stopping browser session")
            await self.browser_session.stop()

