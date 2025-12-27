"""Conversation saving and loading functionality."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConversationMetadata(BaseModel):
    """Metadata for a saved conversation."""

    task: str
    created_at: str
    total_messages: int
    total_steps: int
    is_successful: Optional[bool] = None
    final_result: Optional[str] = None
    llm_provider: str = "unknown"
    llm_model: str = "unknown"


class SavedConversation(BaseModel):
    """Saved conversation with messages and metadata."""

    metadata: ConversationMetadata
    messages: list[dict[str, Any]]


def save_conversation(
    messages: list[BaseMessage],
    file_path: str | Path,
    task: str = "",
    total_steps: int = 0,
    is_successful: Optional[bool] = None,
    final_result: Optional[str] = None,
    llm_provider: str = "unknown",
    llm_model: str = "unknown",
    encoding: str = "utf-8",
) -> Path:
    """
    Save conversation to a file.

    Args:
        messages: List of LangChain messages
        file_path: Path to save the conversation
        task: The task description
        total_steps: Number of steps taken
        is_successful: Whether the task was successful
        final_result: Final result message
        llm_provider: LLM provider used
        llm_model: LLM model used
        encoding: File encoding

    Returns:
        Path to the saved file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert messages to serializable format
    serialized_messages = []
    for msg in messages:
        serialized = _serialize_message(msg)
        if serialized:
            serialized_messages.append(serialized)

    # Create metadata
    metadata = ConversationMetadata(
        task=task,
        created_at=datetime.now().isoformat(),
        total_messages=len(serialized_messages),
        total_steps=total_steps,
        is_successful=is_successful,
        final_result=final_result,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Create saved conversation
    conversation = SavedConversation(
        metadata=metadata,
        messages=serialized_messages,
    )

    # Save to file
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(conversation.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved conversation to {file_path}")
    return file_path


def load_conversation(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> tuple[list[BaseMessage], ConversationMetadata]:
    """
    Load conversation from a file.

    Args:
        file_path: Path to the conversation file
        encoding: File encoding

    Returns:
        Tuple of (messages, metadata)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Conversation file not found: {file_path}")

    with open(file_path, "r", encoding=encoding) as f:
        data = json.load(f)

    conversation = SavedConversation(**data)

    # Convert back to LangChain messages
    messages = []
    for msg_data in conversation.messages:
        msg = _deserialize_message(msg_data)
        if msg:
            messages.append(msg)

    logger.info(f"Loaded conversation from {file_path} ({len(messages)} messages)")
    return messages, conversation.metadata


def _serialize_message(message: BaseMessage) -> Optional[dict[str, Any]]:
    """Serialize a LangChain message to a dictionary."""
    try:
        msg_type = type(message).__name__

        # Handle content (can be string or list)
        content = message.content
        if isinstance(content, list):
            # Filter out image data to reduce file size
            content = [
                item if not (isinstance(item, dict) and item.get("type") == "image_url")
                else {"type": "image_url", "image_url": {"url": "[IMAGE REMOVED]"}}
                for item in content
            ]

        result = {
            "type": msg_type,
            "content": content,
        }

        # Add tool_calls for AIMessage
        if isinstance(message, AIMessage) and message.tool_calls:
            result["tool_calls"] = message.tool_calls

        # Add tool_call_id for ToolMessage
        if isinstance(message, ToolMessage):
            result["tool_call_id"] = message.tool_call_id

        # Add additional kwargs if present
        if hasattr(message, "additional_kwargs") and message.additional_kwargs:
            result["additional_kwargs"] = message.additional_kwargs

        return result

    except Exception as e:
        logger.warning(f"Failed to serialize message: {e}")
        return None


def _deserialize_message(data: dict[str, Any]) -> Optional[BaseMessage]:
    """Deserialize a dictionary to a LangChain message."""
    try:
        msg_type = data.get("type", "")
        content = data.get("content", "")

        if msg_type == "SystemMessage":
            return SystemMessage(content=content)
        elif msg_type == "HumanMessage":
            return HumanMessage(content=content)
        elif msg_type == "AIMessage":
            return AIMessage(
                content=content,
                tool_calls=data.get("tool_calls") or [],
            )
        elif msg_type == "ToolMessage":
            return ToolMessage(
                content=content,
                tool_call_id=data.get("tool_call_id", ""),
            )
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return None

    except Exception as e:
        logger.warning(f"Failed to deserialize message: {e}")
        return None


def conversation_to_text(messages: list[BaseMessage]) -> str:
    """
    Convert conversation to human-readable text.

    Args:
        messages: List of LangChain messages

    Returns:
        Text representation of the conversation
    """
    lines = []

    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content

        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )

        # Truncate long content
        if len(content) > 500:
            content = content[:497] + "..."

        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)

