"""AboutBlank watchdog following browser-use pattern."""

import logging
from typing import TYPE_CHECKING

from src.openbrowser.browser.watchdogs.base import BaseWatchdog
from src.openbrowser.browser.events import TabCreatedEvent

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class AboutBlankWatchdog(BaseWatchdog):
    """
    Watchdog that handles about:blank tabs.
    Shows a placeholder when the browser is on about:blank.
    """

    browser_session: "BrowserSession"

    def attach(self) -> None:
        """Attach event handlers."""
        self.browser_session.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)
        logger.info("AboutBlankWatchdog attached")

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Handle new tab creation."""
        if event.url == "about:blank":
            logger.debug(f"New about:blank tab detected: {event.target_id}")
            # Could inject placeholder content or navigate to a default page
            # For now, just log it

