"""Browser view models following browser-use pattern.

This module provides data models for browser-related information that is
passed between components. These models follow the browser-use pattern
for structured data representation.

Classes:
    TabInfo: Information about an open browser tab.
    BrowserError: Exception with structured memory for LLM context.
    URLNotAllowedError: Exception for blocked URL navigation.

Example:
    >>> tab = TabInfo(url='https://example.com', title='Example', target_id='ABC123')
    >>> print(tab.url)
    'https://example.com'
"""

from typing import TYPE_CHECKING, Any

from cdp_use.cdp.target import TargetID
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer

if TYPE_CHECKING:
    from bubus import BaseEvent


class TabInfo(BaseModel):
    """Represents information about a browser tab following browser-use pattern.

    Encapsulates tab metadata including URL, title, and CDP target identifiers.
    Used for tab listing and management operations.

    Attributes:
        url: Current URL of the tab.
        title: Current title of the tab.
        target_id: CDP target ID for this tab (serialized as last 4 chars).
        parent_target_id: CDP target ID of parent page (for popups/iframes).

    Example:
        >>> tab = TabInfo(
        ...     url='https://example.com',
        ...     title='Example Page',
        ...     target_id='ABC123DEF456',
        ... )
        >>> tab.model_dump()['tab_id']  # Uses serialization alias
        'F456'
    """

    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True,
        populate_by_name=True,
    )

    url: str
    title: str
    target_id: TargetID = Field(serialization_alias='tab_id', validation_alias=AliasChoices('tab_id', 'target_id'))
    parent_target_id: TargetID | None = Field(
        default=None,
        serialization_alias='parent_tab_id',
        validation_alias=AliasChoices('parent_tab_id', 'parent_target_id')
    )  # parent page that contains this popup or cross-origin iframe

    @field_serializer('target_id')
    def serialize_target_id(self, target_id: TargetID, _info: Any) -> str:
        """Serialize target_id to last 4 characters for compact display.

        Args:
            target_id: Full CDP target ID.
            _info: Pydantic serialization info (unused).

        Returns:
            Last 4 characters of target ID for display.
        """
        return target_id[-4:]

    @field_serializer('parent_target_id')
    def serialize_parent_target_id(self, parent_target_id: TargetID | None, _info: Any) -> str | None:
        """Serialize parent_target_id to last 4 characters for compact display.

        Args:
            parent_target_id: Full CDP parent target ID or None.
            _info: Pydantic serialization info (unused).

        Returns:
            Last 4 characters of parent target ID, or None if not set.
        """
        return parent_target_id[-4:] if parent_target_id else None


class BrowserError(Exception):
    """Browser error with structured memory for LLM context management.

    This exception class provides separate memory contexts for browser actions,
    allowing LLM agents to receive both immediate context and persistent error
    information for better decision-making.

    Attributes:
        message: Technical error message for logging and debugging.
        short_term_memory: Context shown once to LLM (e.g., available actions).
        long_term_memory: Persistent error info stored across steps.
        details: Additional metadata for debugging.
        while_handling_event: The browser event that triggered this error.

    Example:
        >>> raise BrowserError(
        ...     message='Element not found',
        ...     short_term_memory='Try using a different selector',
        ...     long_term_memory='Element index 5 does not exist',
        ... )
    """

    def __init__(
        self,
        message: str,
        short_term_memory: str | None = None,
        long_term_memory: str | None = None,
        details: dict[str, Any] | None = None,
        event: 'BaseEvent[Any] | None' = None,
    ):
        """Initialize a BrowserError with structured memory contexts.

        Args:
            message: Technical error message for logging and debugging.
            short_term_memory: Context shown once to LLM for next action.
                Example: list of available elements, alternative approaches.
            long_term_memory: Persistent error info stored in agent memory.
                Example: 'Element index 5 was not found on step 3'.
            details: Additional metadata dict for debugging.
            event: The browser event that triggered this error.
        """
        self.message = message
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.details = details
        self.while_handling_event = event
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f'{self.message} ({self.details}) during: {self.while_handling_event}'
        elif self.while_handling_event:
            return f'{self.message} (while handling: {self.while_handling_event})'
        else:
            return self.message


class URLNotAllowedError(BrowserError):
    """Error raised when a URL is not allowed by security policy.

    Thrown by SecurityWatchdog when navigation to a URL is blocked
    due to allowed_domains or prohibited_domains restrictions.

    Example:
        >>> raise URLNotAllowedError(
        ...     message='Navigation blocked to http://evil.com',
        ...     details={'url': 'http://evil.com', 'reason': 'prohibited_domain'},
        ... )
    """
    pass

