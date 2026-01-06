"""AboutBlank watchdog following browser-use pattern.

This module provides the AboutBlankWatchdog which monitors for about:blank
tabs and can inject placeholder content or navigate to default pages.

Classes:
    AboutBlankWatchdog: Handles about:blank tab detection and placeholder display.
"""

import logging
from typing import TYPE_CHECKING

from openbrowser.browser.watchdogs.base import BaseWatchdog
from openbrowser.browser.events import TabCreatedEvent

if TYPE_CHECKING:
    from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class AboutBlankWatchdog(BaseWatchdog):
    """Watchdog that handles about:blank tabs.

    Monitors for new tabs with about:blank URLs and can show placeholder
    content or navigate to a configured default page.

    Listens to:
        TabCreatedEvent: Detects new about:blank tabs.

    Example:
        >>> watchdog = AboutBlankWatchdog(
        ...     event_bus=bus,
        ...     browser_session=session
        ... )
        >>> watchdog.attach()
    """

    browser_session: "BrowserSession"

    def attach(self) -> None:
        """Attach event handlers.

        Registers TabCreatedEvent handler for about:blank detection.
        """
        self.browser_session.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)
        logger.info("AboutBlankWatchdog attached")

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Handle new tab creation.

        Called when a new tab is created. Detects about:blank tabs
        and can inject placeholder content or navigate elsewhere.

        Args:
            event: TabCreatedEvent with target_id and url.
        """
        if event.url == "about:blank":
            logger.debug(f"New about:blank tab detected: {event.target_id}")
            # Could inject placeholder content or navigate to a default page
            # For now, just log it

