"""Base watchdog class for browser monitoring components."""

import logging
from typing import Any, ClassVar

from bubus import BaseEvent, EventBus
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class BaseWatchdog(BaseModel):
    """Base class for all browser watchdogs.

    Watchdogs monitor browser state and emit events based on changes.
    They automatically register event handlers based on method names.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid',
        validate_assignment=False,
        revalidate_instances='never',
    )

    # Class variables to statically define the list of events relevant to each watchdog
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = []
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

    # Core dependencies
    event_bus: EventBus = Field()
    browser_session: Any = Field()  # BrowserSession type

    @property
    def logger(self) -> logging.Logger:
        """Get the logger from the browser session."""
        return self.browser_session.logger

    def attach_to_session(self) -> None:
        """Attach event handlers to the event bus.

        Subclasses should override this to register their specific handlers.
        """
        pass

