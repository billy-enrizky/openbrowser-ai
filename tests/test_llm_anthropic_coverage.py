"""Tests for Anthropic LLM provider modules - chat.py and serializer.py.

Covers:
  src/openbrowser/llm/anthropic/chat.py
  src/openbrowser/llm/anthropic/serializer.py
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

# Guard: skip entire module if anthropic is not installed
anthropic_mod = pytest.importorskip("anthropic")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
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
)
from openbrowser.llm.views import ChatInvokeCompletion, ChatInvokeUsage


def _make_usage(input_tokens=10, output_tokens=5, cache_read=0, cache_creation=0):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_creation
    return usage


def _make_text_block(text="hello"):
    from anthropic.types.text_block import TextBlock

    return TextBlock(text=text, type="text")


def _make_tool_use_block(tool_id="tool_1", name="MyModel", input_data=None):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data or {"field": "value"}
    return block


def _make_response(content=None, usage=None, stop_reason="end_turn"):
    from anthropic.types import Message

    if content is None:
        content = [_make_text_block("test response")]
    if usage is None:
        usage = _make_usage()

    resp = MagicMock(spec=Message)
    resp.content = content
    resp.usage = usage
    resp.stop_reason = stop_reason
    # Make isinstance checks pass
    resp.__class__ = Message
    return resp


# ===========================================================================
# Tests for AnthropicMessageSerializer
# ===========================================================================
class TestAnthropicSerializer:
    """Tests for serializer.py coverage."""

    def test_is_base64_image_true(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        assert AnthropicMessageSerializer._is_base64_image("data:image/png;base64,abc") is True

    def test_is_base64_image_false(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        assert AnthropicMessageSerializer._is_base64_image("https://example.com/img.png") is False

    def test_parse_base64_url_valid_jpeg(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        media_type, data = AnthropicMessageSerializer._parse_base64_url(
            "data:image/jpeg;base64,SGVsbG8="
        )
        assert media_type == "image/jpeg"
        assert data == "SGVsbG8="

    def test_parse_base64_url_valid_png(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        media_type, data = AnthropicMessageSerializer._parse_base64_url(
            "data:image/png;base64,SGVsbG8="
        )
        assert media_type == "image/png"
        assert data == "SGVsbG8="

    def test_parse_base64_url_unsupported_type_defaults_to_jpeg(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        media_type, data = AnthropicMessageSerializer._parse_base64_url(
            "data:image/bmp;base64,SGVsbG8="
        )
        assert media_type == "image/jpeg"

    def test_parse_base64_url_invalid_raises(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        with pytest.raises(ValueError, match="Invalid base64 URL"):
            AnthropicMessageSerializer._parse_base64_url("https://example.com/img.png")

    def test_serialize_cache_control_true(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        result = AnthropicMessageSerializer._serialize_cache_control(True)
        assert result is not None
        assert result["type"] == "ephemeral"

    def test_serialize_cache_control_false(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        result = AnthropicMessageSerializer._serialize_cache_control(False)
        assert result is None

    def test_serialize_content_part_text(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        part = ContentPartTextParam(text="hello")
        result = AnthropicMessageSerializer._serialize_content_part_text(part, use_cache=False)
        assert result["text"] == "hello"
        assert result["type"] == "text"

    def test_serialize_content_part_text_with_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        part = ContentPartTextParam(text="hello")
        result = AnthropicMessageSerializer._serialize_content_part_text(part, use_cache=True)
        assert result["cache_control"] is not None

    def test_serialize_content_part_image_base64(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        part = ContentPartImageParam(
            image_url=ImageURL(url="data:image/png;base64,SGVsbG8=")
        )
        result = AnthropicMessageSerializer._serialize_content_part_image(part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"

    def test_serialize_content_part_image_url(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        part = ContentPartImageParam(
            image_url=ImageURL(url="https://example.com/img.png")
        )
        result = AnthropicMessageSerializer._serialize_content_part_image(part)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"

    def test_serialize_content_to_str_string_no_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        result = AnthropicMessageSerializer._serialize_content_to_str("hello", use_cache=False)
        assert result == "hello"

    def test_serialize_content_to_str_string_with_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        result = AnthropicMessageSerializer._serialize_content_to_str("hello", use_cache=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_serialize_content_to_str_list(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        parts = [ContentPartTextParam(text="part1"), ContentPartTextParam(text="part2")]
        result = AnthropicMessageSerializer._serialize_content_to_str(parts, use_cache=False)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_serialize_content_string_no_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        result = AnthropicMessageSerializer._serialize_content("hello world", use_cache=False)
        assert result == "hello world"

    def test_serialize_content_string_with_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        result = AnthropicMessageSerializer._serialize_content("hello", use_cache=True)
        assert isinstance(result, list)
        assert result[0]["text"] == "hello"

    def test_serialize_content_list_mixed(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        parts = [
            ContentPartTextParam(text="text part"),
            ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png")),
        ]
        result = AnthropicMessageSerializer._serialize_content(parts, use_cache=False)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_serialize_tool_calls_to_content(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="my_func", arguments='{"a": 1}'))
        result = AnthropicMessageSerializer._serialize_tool_calls_to_content([tc], use_cache=False)
        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["id"] == "tc1"
        assert result[0]["input"] == {"a": 1}

    def test_serialize_tool_calls_invalid_json(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        tc = ToolCall(id="tc2", function=Function(name="my_func", arguments="not-json"))
        result = AnthropicMessageSerializer._serialize_tool_calls_to_content([tc])
        assert result[0]["input"] == {"arguments": "not-json"}

    def test_serialize_user_message_string(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msg = UserMessage(content="hello")
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_serialize_user_message_with_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msg = UserMessage(content="hello", cache=True)
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "user"

    def test_serialize_system_message_returns_system(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msg = SystemMessage(content="you are helpful")
        result = AnthropicMessageSerializer.serialize(msg)
        assert isinstance(result, SystemMessage)

    def test_serialize_assistant_message_string(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msg = AssistantMessage(content="response text")
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        # Single text block without cache -> simplified to string
        assert result["content"] == "response text"

    def test_serialize_assistant_message_with_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msg = AssistantMessage(content="cached response", cache=True)
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        # With cache, content should be a list of blocks
        assert isinstance(result["content"], list)

    def test_serialize_assistant_message_with_tool_calls(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="func1", arguments='{"x": 1}'))
        msg = AssistantMessage(content="text", tool_calls=[tc])
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        # Multiple blocks (text + tool_use)
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2

    def test_serialize_assistant_message_no_content_no_tools(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msg = AssistantMessage(content=None)
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        # Should have empty text block
        assert isinstance(result["content"], str) or isinstance(result["content"], list)

    def test_serialize_assistant_message_list_content(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        parts = [ContentPartTextParam(text="part1"), ContentPartTextParam(text="part2")]
        msg = AssistantMessage(content=parts)
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"

    def test_serialize_assistant_message_single_tool_block_no_cache(self):
        """Cover line 232: single non-text block without cache -> content = blocks."""
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="func1", arguments='{"x": 1}'))
        msg = AssistantMessage(content=None, tool_calls=[tc])
        result = AnthropicMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        # Should be a list (the blocks list), not simplified to string
        assert isinstance(result["content"], list)

    def test_serialize_unknown_message_raises(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        with pytest.raises(ValueError, match="Unknown message type"):
            AnthropicMessageSerializer.serialize(MagicMock())

    def test_clean_cache_messages_empty(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        result = AnthropicMessageSerializer._clean_cache_messages([])
        assert result == []

    def test_clean_cache_messages_single_cached(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msgs = [UserMessage(content="a", cache=True)]
        result = AnthropicMessageSerializer._clean_cache_messages(msgs)
        assert result[0].cache is True

    def test_clean_cache_messages_multiple_cached(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msgs = [
            UserMessage(content="a", cache=True),
            UserMessage(content="b", cache=True),
            UserMessage(content="c", cache=True),
        ]
        result = AnthropicMessageSerializer._clean_cache_messages(msgs)
        # Only the last one should remain cached
        assert result[0].cache is False
        assert result[1].cache is False
        assert result[2].cache is True

    def test_clean_cache_messages_no_cached(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msgs = [UserMessage(content="a"), UserMessage(content="b")]
        result = AnthropicMessageSerializer._clean_cache_messages(msgs)
        assert result[0].cache is False
        assert result[1].cache is False

    def test_serialize_messages_basic(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msgs = [
            SystemMessage(content="sys prompt"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        serialized, system = AnthropicMessageSerializer.serialize_messages(msgs)
        assert system == "sys prompt"
        assert len(serialized) == 2

    def test_serialize_messages_no_system(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msgs = [UserMessage(content="hi")]
        serialized, system = AnthropicMessageSerializer.serialize_messages(msgs)
        assert system is None
        assert len(serialized) == 1

    def test_serialize_messages_system_with_cache(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msgs = [
            SystemMessage(content="sys prompt", cache=True),
            UserMessage(content="hi"),
        ]
        serialized, system = AnthropicMessageSerializer.serialize_messages(msgs)
        assert isinstance(system, list)
        assert system[0]["text"] == "sys prompt"

    def test_serialize_messages_system_list_content(self):
        from openbrowser.llm.anthropic.serializer import AnthropicMessageSerializer

        msgs = [
            SystemMessage(content=[ContentPartTextParam(text="p1"), ContentPartTextParam(text="p2")]),
            UserMessage(content="hi"),
        ]
        serialized, system = AnthropicMessageSerializer.serialize_messages(msgs)
        assert isinstance(system, list)
        assert len(system) == 2


# ===========================================================================
# Tests for ChatAnthropic
# ===========================================================================
class TestChatAnthropic:
    """Tests for chat.py coverage."""

    def _make_chat(self, **kwargs):
        from openbrowser.llm.anthropic.chat import ChatAnthropic

        defaults = {"model": "claude-sonnet-4-20250514", "api_key": "test-key"}
        defaults.update(kwargs)
        return ChatAnthropic(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "anthropic"

    def test_name(self):
        chat = self._make_chat(model="claude-sonnet-4-20250514")
        assert chat.name == "claude-sonnet-4-20250514"

    def test_get_client_params_defaults(self):
        chat = self._make_chat()
        params = chat._get_client_params()
        assert "api_key" in params
        assert params["api_key"] == "test-key"

    def test_get_client_params_filters_none(self):
        chat = self._make_chat(auth_token=None, base_url=None)
        params = chat._get_client_params()
        assert "auth_token" not in params
        assert "base_url" not in params

    def test_get_client_params_includes_set_values(self):
        chat = self._make_chat(base_url="https://example.com", default_headers={"X-Custom": "val"})
        params = chat._get_client_params()
        assert params["base_url"] == "https://example.com"
        assert params["default_headers"] == {"X-Custom": "val"}

    def test_get_client_params_for_invoke_all_set(self):
        chat = self._make_chat(temperature=0.5, max_tokens=1024, top_p=0.9, seed=42)
        params = chat._get_client_params_for_invoke()
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 1024
        assert params["top_p"] == 0.9
        assert params["seed"] == 42

    def test_get_client_params_for_invoke_defaults(self):
        chat = self._make_chat()
        params = chat._get_client_params_for_invoke()
        assert "max_tokens" in params
        assert "temperature" not in params  # default is None

    def test_get_client(self):
        chat = self._make_chat()
        client = chat.get_client()
        from anthropic import AsyncAnthropic

        assert isinstance(client, AsyncAnthropic)

    def test_get_usage(self):
        chat = self._make_chat()
        resp = _make_response()
        usage = chat._get_usage(resp)
        assert isinstance(usage, ChatInvokeUsage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_get_usage_with_cache(self):
        chat = self._make_chat()
        resp = _make_response(usage=_make_usage(cache_read=3, cache_creation=2))
        usage = chat._get_usage(resp)
        assert usage.prompt_tokens == 13  # 10 + 3
        assert usage.prompt_cached_tokens == 3
        assert usage.prompt_cache_creation_tokens == 2

    @pytest.mark.asyncio
    async def test_ainvoke_text_response(self):
        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = _make_response()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hello")])

        assert isinstance(result, ChatInvokeCompletion)
        assert result.completion == "test response"
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_ainvoke_text_non_textblock_content(self):
        """Test when first content block is not a TextBlock."""
        chat = self._make_chat()
        mock_client = AsyncMock()

        non_text_block = MagicMock()
        non_text_block.__str__ = lambda self: "non-text-content"
        resp = _make_response(content=[non_text_block])
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hello")])

        assert "non-text-content" in result.completion

    @pytest.mark.asyncio
    async def test_ainvoke_unexpected_response_type(self):
        """Test when API returns a non-Message response."""
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value="not a message")

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Unexpected response type"):
                await chat.ainvoke([UserMessage(content="hello")])

    @pytest.mark.asyncio
    async def test_ainvoke_with_system_message(self):
        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = _make_response()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([
                SystemMessage(content="be helpful"),
                UserMessage(content="hello"),
            ])

        assert isinstance(result, ChatInvokeCompletion)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        tool_block = _make_tool_use_block(name="MyOutput", input_data={"field": "value"})
        resp = _make_response(content=[tool_block])
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)
        assert result.completion.field == "value"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_unexpected_response(self):
        """Test structured output when API returns non-Message."""
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value="not a message")

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Unexpected response type"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_json_string_input(self):
        """Test structured output when tool_use input is a JSON string."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()

        # Create a tool block with string input that fails model_validate but succeeds as JSON
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = '{"field": "from_json"}'

        resp = _make_response(content=[tool_block])
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert result.completion.field == "from_json"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_no_tool_use(self):
        """Test error when no tool_use block found in response."""
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()

        text_block = _make_text_block("no tool here")
        resp = _make_response(content=[text_block])
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_schema_has_title(self):
        """Cover line 180: title removed from schema before tool creation."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        tool_block = _make_tool_use_block(name="MyOutput", input_data={"field": "value"})
        resp = _make_response(content=[tool_block])
        mock_client.messages.create = AsyncMock(return_value=resp)

        # Patch SchemaOptimizer to return a schema WITH title
        schema_with_title = {
            "type": "object",
            "title": "MyOutput",
            "properties": {"field": {"type": "string"}},
            "required": ["field"],
            "additionalProperties": False,
        }
        with patch.object(chat, "get_client", return_value=mock_client), \
             patch("openbrowser.llm.anthropic.chat.SchemaOptimizer") as mock_optimizer:
            mock_optimizer.create_optimized_json_schema.return_value = schema_with_title
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_api_connection_error(self):
        from anthropic import APIConnectionError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        error = APIConnectionError(request=MagicMock())
        mock_client.messages.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hello")])

    @pytest.mark.asyncio
    async def test_ainvoke_rate_limit_error(self):
        from anthropic import RateLimitError
        from openbrowser.llm.exceptions import ModelRateLimitError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )
        mock_client.messages.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelRateLimitError):
                await chat.ainvoke([UserMessage(content="hello")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_status_error(self):
        from anthropic import APIStatusError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        error = APIStatusError(
            message="server error",
            response=mock_response,
            body=None,
        )
        mock_client.messages.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hello")])

    @pytest.mark.asyncio
    async def test_ainvoke_generic_exception(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="hello")])

    @pytest.mark.asyncio
    async def test_ainvoke_structured_validation_error_non_string_input(self):
        """Test that validation errors re-raise when input is not a string."""
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: int  # expects int

        chat = self._make_chat()
        mock_client = AsyncMock()

        # input is a dict but with wrong type -> validation fails and input is not str
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"field": "not_an_int_but_coercible_wait_no"}  # pydantic may fail

        # Actually, pydantic v2 may coerce. Let's use something that truly fails.
        tool_block.input = {"wrong_field": "value"}  # missing required 'field'

        resp = _make_response(content=[tool_block])
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
