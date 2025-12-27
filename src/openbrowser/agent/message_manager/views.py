"""Views and models for message manager."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field


class HistoryItem(BaseModel):
    """A single item in agent history."""
    
    step_number: int | None = None
    evaluation_previous_goal: str | None = None
    memory: str | None = None
    next_goal: str | None = None
    action_results: str | None = None
    error: str | None = None
    system_message: str | None = None

    def to_string(self) -> str:
        """Convert history item to string format."""
        if self.system_message:
            return f'<sys>{self.system_message}</sys>'
        
        if self.step_number is None:
            return ''
            
        parts = [f'<step_{self.step_number}>']
        
        if self.evaluation_previous_goal:
            parts.append(f'Evaluation: {self.evaluation_previous_goal}')
        if self.memory:
            parts.append(f'Memory: {self.memory}')
        if self.next_goal:
            parts.append(f'Next Goal: {self.next_goal}')
        if self.action_results:
            parts.append(f'{self.action_results}')
        if self.error:
            parts.append(f'Error: {self.error}')
            
        parts.append(f'</step_{self.step_number}>')
        
        return '\n'.join(parts)


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
    """Serializable state for message manager."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    agent_history_items: list[HistoryItem] = Field(default_factory=list)
    read_state_description: str = ''
    history: MessageHistory = Field(default_factory=MessageHistory)

