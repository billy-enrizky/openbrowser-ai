"""Views and models for message manager."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field


class HistoryItem(BaseModel):
    """A single item in agent history."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    step_number: int | None = None
    evaluation_previous_goal: str | None = None
    memory: str | None = None
    next_goal: str | None = None
    action_results: str | None = None
    error: str | None = None
    system_message: str | None = None

    def model_post_init(self, __context) -> None:
        """Validate that error and system_message are not both provided."""
        if self.error is not None and self.system_message is not None:
            raise ValueError('Cannot have both error and system_message at the same time')

    def to_string(self) -> str:
        """Convert history item to string format."""
        step_str = f'step_{self.step_number}' if self.step_number is not None else 'step_unknown'
        
        if self.error:
            return f'<{step_str}>\n{self.error}'
        elif self.system_message:
            return self.system_message
        else:
            content_parts = []

            if self.evaluation_previous_goal:
                content_parts.append(f'{self.evaluation_previous_goal}')
            if self.memory:
                content_parts.append(f'{self.memory}')
            if self.next_goal:
                content_parts.append(f'{self.next_goal}')
            if self.action_results:
                content_parts.append(self.action_results)

            content = '\n'.join(content_parts)
            return f'<{step_str}>\n{content}'


class MessageHistory(BaseModel):
    """Message history container."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    system_message: BaseMessage | None = None
    state_message: BaseMessage | None = None
    context_messages: list[BaseMessage] = Field(default_factory=list)

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages in order."""
        messages = []
        if self.system_message:
            messages.append(self.system_message)
        if self.state_message:
            messages.append(self.state_message)
        messages.extend(self.context_messages)
        return messages


class MessageManagerState(BaseModel):
    """Serializable state for message manager - holds all message manager state for checkpointing."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    history: MessageHistory = Field(default_factory=MessageHistory)
    tool_id: int = 1
    agent_history_items: list[HistoryItem] = Field(
        default_factory=lambda: [HistoryItem(step_number=0, system_message='Agent initialized')]
    )
    read_state_description: str = ''
