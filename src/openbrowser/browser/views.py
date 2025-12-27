"""Browser view models following browser-use pattern."""

from typing import TYPE_CHECKING, Any

from cdp_use.cdp.target import TargetID
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer

if TYPE_CHECKING:
    from bubus import BaseEvent


class TabInfo(BaseModel):
    """Represents information about a browser tab following browser-use pattern."""

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
        return target_id[-4:]

    @field_serializer('parent_target_id')
    def serialize_parent_target_id(self, parent_target_id: TargetID | None, _info: Any) -> str | None:
        return parent_target_id[-4:] if parent_target_id else None


class BrowserError(Exception):
    """Browser error with structured memory for LLM context management.

    This exception class provides separate memory contexts for browser actions:
    - short_term_memory: Immediate context shown once to the LLM for the next action
    - long_term_memory: Persistent error information stored across steps
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
            message: Technical error message for logging and debugging
            short_term_memory: Context shown once to LLM (e.g., available actions, options)
            long_term_memory: Persistent error info stored in agent memory
            details: Additional metadata for debugging
            event: The browser event that triggered this error
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
    """Error raised when a URL is not allowed"""
    pass

