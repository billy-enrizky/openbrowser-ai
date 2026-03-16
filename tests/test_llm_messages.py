"""Comprehensive tests for openbrowser.llm.messages and openbrowser.llm.base modules.

Covers: _truncate, _format_image_url, ContentPartTextParam, ContentPartRefusalParam,
ImageURL, ContentPartImageParam, Function, ToolCall, UserMessage, SystemMessage,
AssistantMessage, BaseChatModel Protocol.
"""

import logging
from unittest.mock import MagicMock

import pytest

from openbrowser.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    Function,
    ImageURL,
    SystemMessage,
    ToolCall,
    UserMessage,
    _format_image_url,
    _truncate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 50) == "hello"

    def test_exact_length_unchanged(self):
        text = "x" * 50
        assert _truncate(text, 50) == text

    def test_long_text_truncated(self):
        text = "x" * 60
        result = _truncate(text, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_very_short_max_length(self):
        result = _truncate("hello world", 5)
        assert len(result) == 5
        assert result.endswith("...")

    def test_empty_string(self):
        assert _truncate("", 50) == ""

    def test_default_max_length(self):
        text = "x" * 60
        result = _truncate(text)
        assert len(result) == 50
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# _format_image_url
# ---------------------------------------------------------------------------


class TestFormatImageUrl:
    def test_base64_image(self):
        url = "data:image/png;base64,iVBORw0KGgoAAAANSUh"
        result = _format_image_url(url)
        assert "<base64 image/png>" == result

    def test_base64_no_semicolon(self):
        url = "data:image/png"
        result = _format_image_url(url)
        # Should still handle gracefully
        assert "base64" in result or "image" in result

    def test_regular_url_short(self):
        url = "https://example.com/img.png"
        result = _format_image_url(url)
        assert result == url

    def test_regular_url_long(self):
        url = "https://example.com/" + "a" * 100
        result = _format_image_url(url, max_length=30)
        assert result.endswith("...")
        assert len(result) == 30


# ---------------------------------------------------------------------------
# ContentPartTextParam
# ---------------------------------------------------------------------------


class TestContentPartTextParam:
    def test_creation(self):
        part = ContentPartTextParam(text="hello world")
        assert part.text == "hello world"
        assert part.type == "text"

    def test_str(self):
        part = ContentPartTextParam(text="hello")
        assert "Text:" in str(part)
        assert "hello" in str(part)

    def test_repr(self):
        part = ContentPartTextParam(text="hello")
        assert "ContentPartTextParam" in repr(part)

    def test_long_text_str_truncated(self):
        part = ContentPartTextParam(text="x" * 100)
        result = str(part)
        assert "..." in result


# ---------------------------------------------------------------------------
# ContentPartRefusalParam
# ---------------------------------------------------------------------------


class TestContentPartRefusalParam:
    def test_creation(self):
        part = ContentPartRefusalParam(refusal="I cannot do that")
        assert part.refusal == "I cannot do that"
        assert part.type == "refusal"

    def test_str(self):
        part = ContentPartRefusalParam(refusal="refused")
        assert "Refusal:" in str(part)

    def test_repr(self):
        part = ContentPartRefusalParam(refusal="refused")
        assert "ContentPartRefusalParam" in repr(part)


# ---------------------------------------------------------------------------
# ImageURL
# ---------------------------------------------------------------------------


class TestImageURL:
    def test_defaults(self):
        img = ImageURL(url="https://example.com/img.png")
        assert img.detail == "auto"
        assert img.media_type == "image/jpeg"

    def test_custom_values(self):
        img = ImageURL(url="https://example.com/img.png", detail="high", media_type="image/png")
        assert img.detail == "high"
        assert img.media_type == "image/png"

    def test_str(self):
        img = ImageURL(url="https://example.com/img.png")
        result = str(img)
        assert "Image" in result
        assert "detail=auto" in result

    def test_repr(self):
        img = ImageURL(url="https://example.com/img.png")
        result = repr(img)
        assert "ImageURL" in result

    def test_base64_url_str(self):
        img = ImageURL(url="data:image/png;base64,abc123")
        result = str(img)
        assert "base64" in result


# ---------------------------------------------------------------------------
# ContentPartImageParam
# ---------------------------------------------------------------------------


class TestContentPartImageParam:
    def test_creation(self):
        img_url = ImageURL(url="https://example.com/img.png")
        part = ContentPartImageParam(image_url=img_url)
        assert part.type == "image_url"

    def test_str(self):
        img_url = ImageURL(url="https://example.com/img.png")
        part = ContentPartImageParam(image_url=img_url)
        result = str(part)
        assert "Image" in result

    def test_repr(self):
        img_url = ImageURL(url="https://example.com/img.png")
        part = ContentPartImageParam(image_url=img_url)
        result = repr(part)
        assert "ContentPartImageParam" in result


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------


class TestFunction:
    def test_creation(self):
        func = Function(name="click", arguments='{"index": 5}')
        assert func.name == "click"
        assert func.arguments == '{"index": 5}'

    def test_str(self):
        func = Function(name="click", arguments='{"index": 5}')
        result = str(func)
        assert "click" in result
        assert "index" in result

    def test_repr(self):
        func = Function(name="click", arguments='{"index": 5}')
        result = repr(func)
        assert "Function" in result
        assert "click" in result

    def test_long_arguments_truncated(self):
        func = Function(name="func", arguments="x" * 200)
        result = str(func)
        assert "..." in result


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_creation(self):
        func = Function(name="click", arguments='{}')
        tc = ToolCall(id="call_123", function=func)
        assert tc.id == "call_123"
        assert tc.type == "function"

    def test_str(self):
        func = Function(name="click", arguments='{}')
        tc = ToolCall(id="call_123", function=func)
        result = str(tc)
        assert "ToolCall" in result
        assert "call_123" in result

    def test_repr(self):
        func = Function(name="click", arguments='{}')
        tc = ToolCall(id="call_123", function=func)
        result = repr(tc)
        assert "ToolCall" in result


# ---------------------------------------------------------------------------
# UserMessage
# ---------------------------------------------------------------------------


class TestUserMessage:
    def test_string_content(self):
        msg = UserMessage(content="Hello")
        assert msg.text == "Hello"
        assert msg.role == "user"

    def test_list_content(self):
        parts = [
            ContentPartTextParam(text="first"),
            ContentPartTextParam(text="second"),
        ]
        msg = UserMessage(content=parts)
        assert "first" in msg.text
        assert "second" in msg.text

    def test_list_content_with_image(self):
        parts = [
            ContentPartTextParam(text="text"),
            ContentPartImageParam(image_url=ImageURL(url="https://img.com/a.png")),
        ]
        msg = UserMessage(content=parts)
        # Image parts are not text, so should only include text part
        assert "text" in msg.text

    def test_str(self):
        msg = UserMessage(content="Hello")
        assert "UserMessage" in str(msg)
        assert "Hello" in str(msg)

    def test_repr(self):
        msg = UserMessage(content="Hello")
        assert "UserMessage" in repr(msg)

    def test_cache_field(self):
        msg = UserMessage(content="Hello", cache=True)
        assert msg.cache is True

    def test_name_field(self):
        msg = UserMessage(content="Hello", name="user1")
        assert msg.name == "user1"

    def test_empty_content_text(self):
        # Non-str, non-list content should return empty
        msg = UserMessage(content="")
        assert msg.text == ""


# ---------------------------------------------------------------------------
# SystemMessage
# ---------------------------------------------------------------------------


class TestSystemMessage:
    def test_string_content(self):
        msg = SystemMessage(content="You are a helper")
        assert msg.text == "You are a helper"
        assert msg.role == "system"

    def test_list_content(self):
        parts = [ContentPartTextParam(text="system instruction")]
        msg = SystemMessage(content=parts)
        assert "system instruction" in msg.text

    def test_str(self):
        msg = SystemMessage(content="test")
        assert "SystemMessage" in str(msg)

    def test_repr(self):
        msg = SystemMessage(content="test")
        assert "SystemMessage" in repr(msg)

    def test_empty_text(self):
        msg = SystemMessage(content="")
        assert msg.text == ""


# ---------------------------------------------------------------------------
# AssistantMessage
# ---------------------------------------------------------------------------


class TestAssistantMessage:
    def test_string_content(self):
        msg = AssistantMessage(content="I can help")
        assert msg.text == "I can help"
        assert msg.role == "assistant"

    def test_none_content(self):
        msg = AssistantMessage(content=None)
        assert msg.text == ""

    def test_list_content_with_text(self):
        parts = [ContentPartTextParam(text="response")]
        msg = AssistantMessage(content=parts)
        assert "response" in msg.text

    def test_list_content_with_refusal(self):
        parts = [ContentPartRefusalParam(refusal="I cannot do that")]
        msg = AssistantMessage(content=parts)
        assert "[Refusal]" in msg.text
        assert "I cannot do that" in msg.text

    def test_list_content_mixed(self):
        parts = [
            ContentPartTextParam(text="Here is some text"),
            ContentPartRefusalParam(refusal="Cannot provide more"),
        ]
        msg = AssistantMessage(content=parts)
        text = msg.text
        assert "Here is some text" in text
        assert "[Refusal]" in text

    def test_tool_calls(self):
        func = Function(name="click", arguments='{"index": 5}')
        tc = ToolCall(id="call_123", function=func)
        msg = AssistantMessage(content="Calling tool", tool_calls=[tc])
        assert len(msg.tool_calls) == 1

    def test_refusal_field(self):
        msg = AssistantMessage(content=None, refusal="I cannot do that")
        assert msg.refusal == "I cannot do that"

    def test_str(self):
        msg = AssistantMessage(content="test")
        assert "AssistantMessage" in str(msg)

    def test_repr(self):
        msg = AssistantMessage(content="test")
        assert "AssistantMessage" in repr(msg)

    def test_default_tool_calls_empty(self):
        msg = AssistantMessage(content="test")
        assert msg.tool_calls == []


# ---------------------------------------------------------------------------
# BaseChatModel Protocol
# ---------------------------------------------------------------------------


class TestBaseChatModel:
    def test_protocol_is_runtime_checkable(self):
        from openbrowser.llm.base import BaseChatModel

        # BaseChatModel is a runtime_checkable Protocol
        assert hasattr(BaseChatModel, "__protocol_attrs__") or hasattr(BaseChatModel, "__abstractmethods__")
        # The protocol is defined and importable
        assert BaseChatModel is not None

    def test_protocol_has_annotations(self):
        from openbrowser.llm.base import BaseChatModel

        # The protocol should have annotations for 'model'
        annotations = getattr(BaseChatModel, "__annotations__", {})
        assert "model" in annotations

    def test_get_pydantic_core_schema(self):
        from openbrowser.llm.base import BaseChatModel

        # This should not raise
        schema = BaseChatModel.__get_pydantic_core_schema__(BaseChatModel, lambda x: x)
        assert schema is not None

    def test_model_name_property(self):
        from openbrowser.llm.base import BaseChatModel

        # model_name property should be accessible on the Protocol
        assert hasattr(BaseChatModel, "model_name")
