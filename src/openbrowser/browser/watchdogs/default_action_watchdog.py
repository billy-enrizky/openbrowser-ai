"""Default action watchdog following browser-use pattern."""

import logging
from typing import TYPE_CHECKING

from src.openbrowser.browser.watchdogs.base import BaseWatchdog

if TYPE_CHECKING:
    from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class DefaultActionWatchdog(BaseWatchdog):
    """
    Watchdog that handles default action behaviors.
    This includes handling dropdown interactions and other common UI patterns.
    """

    browser_session: "BrowserSession"

    def attach(self) -> None:
        """Attach event handlers."""
        # No specific events to handle yet
        # This watchdog can be extended for default action behaviors
        logger.info("DefaultActionWatchdog attached")

