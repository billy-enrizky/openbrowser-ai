"""DOM watchdog for browser DOM tree management using CDP."""

import logging
from typing import TYPE_CHECKING, ClassVar

from bubus import BaseEvent

from src.openbrowser.browser.dom import DomState
from src.openbrowser.browser.events import TabCreatedEvent
from src.openbrowser.browser.watchdogs.base import BaseWatchdog

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DOMWatchdog(BaseWatchdog):
    """Handles DOM tree building and element access via CDP.
    
    This watchdog acts as a bridge between the event-driven browser session
    and the DomService implementation, maintaining cached state and providing
    helper methods for other watchdogs.
    
    Following browser-use pattern, this watchdog caches DOM state and provides
    access to selector maps for other watchdogs and tools.
    """

    LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [TabCreatedEvent]
    EMITS: ClassVar[list[type[BaseEvent]]] = []

    # Public properties for other watchdogs
    current_dom_state: DomState | None = None
    selector_map: dict[int, int] | None = None

    def attach_to_session(self) -> None:
        """Register event handlers."""
        self.event_bus.on(TabCreatedEvent, self.on_TabCreatedEvent)

    async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
        """Handle tab creation - no special setup needed."""
        return None

    def update_dom_state(self, dom_state: DomState) -> None:
        """Update cached DOM state.
        
        Args:
            dom_state: The new DOM state to cache
        """
        self.current_dom_state = dom_state
        self.selector_map = dom_state.selector_map

        # Update browser session's cached selector map if it has that method
        if hasattr(self.browser_session, "update_cached_selector_map"):
            self.browser_session.update_cached_selector_map(self.selector_map)

        self.logger.debug(f"DOM state updated: {len(self.selector_map)} elements")

