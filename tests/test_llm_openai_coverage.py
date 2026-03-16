"""Tests for OpenAI LLM provider modules - chat.py and serializer.py.

Covers:
  src/openbrowser/llm/openai/chat.py
  src/openbrowser/llm/openai/serializer.py
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

openai_mod = pytest.importorskip("openai")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_usage(prompt=10, completion=5, total=15, cached=None, reasoning=None):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = total

    # prompt_tokens_details
    if cached is not None:
        details = MagicMock()
        details.cached_tokens = cached
        usage.prompt_tokens_details = details
    else:
        usage.prompt_tokens_details = None

    # completion_tokens_details
    if reasoning is not None:
        comp_details = MagicMock()
        comp_details.reasoning_tokens = reasoning
        usage.completion_tokens_details = comp_details
    else:
        usage.completion_tokens_details = None

    return usage


def _make_choice(content="hello", finish_reason="stop"):
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    return choice


def _make_response(content="hello", usage=None, finish_reason="stop"):
    from openai.types.chat.chat_completion import ChatCompletion

    resp = MagicMock(spec=ChatCompletion)
    resp.choices = [_make_choice(content, finish_reason)]
    resp.usage = usage if usage is not None else _make_usage()
    return resp


# ===========================================================================
# Tests for OpenAIMessageSerializer
# ===========================================================================
class TestOpenAISerializer:
    """Tests for openai/serializer.py coverage."""

    def test_serialize_content_part_text(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        part = ContentPartTextParam(text="hello")
        result = OpenAIMessageSerializer._serialize_content_part_text(part)
        assert result["text"] == "hello"
        assert result["type"] == "text"

    def test_serialize_content_part_image(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        part = ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png", detail="high"))
        result = OpenAIMessageSerializer._serialize_content_part_image(part)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/img.png"
        assert result["image_url"]["detail"] == "high"

    def test_serialize_content_part_refusal(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        part = ContentPartRefusalParam(refusal="I cannot do that")
        result = OpenAIMessageSerializer._serialize_content_part_refusal(part)
        assert result["refusal"] == "I cannot do that"
        assert result["type"] == "refusal"

    def test_serialize_user_content_string(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        result = OpenAIMessageSerializer._serialize_user_content("hello")
        assert result == "hello"

    def test_serialize_user_content_list(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        parts = [
            ContentPartTextParam(text="text"),
            ContentPartImageParam(image_url=ImageURL(url="https://img.com/a.png")),
        ]
        result = OpenAIMessageSerializer._serialize_user_content(parts)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_serialize_system_content_string(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        result = OpenAIMessageSerializer._serialize_system_content("system prompt")
        assert result == "system prompt"

    def test_serialize_system_content_list(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        parts = [ContentPartTextParam(text="p1"), ContentPartTextParam(text="p2")]
        result = OpenAIMessageSerializer._serialize_system_content(parts)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_serialize_assistant_content_none(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        result = OpenAIMessageSerializer._serialize_assistant_content(None)
        assert result is None

    def test_serialize_assistant_content_string(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        result = OpenAIMessageSerializer._serialize_assistant_content("response")
        assert result == "response"

    def test_serialize_assistant_content_list(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        parts = [
            ContentPartTextParam(text="text"),
            ContentPartRefusalParam(refusal="no"),
        ]
        result = OpenAIMessageSerializer._serialize_assistant_content(parts)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_serialize_tool_call(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{"a": 1}'))
        result = OpenAIMessageSerializer._serialize_tool_call(tc)
        assert result["id"] == "tc1"
        assert result["function"]["name"] == "fn"
        assert result["type"] == "function"

    def test_serialize_user_message(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        msg = UserMessage(content="hi", name="user1")
        result = OpenAIMessageSerializer.serialize(msg)
        assert result["role"] == "user"
        assert result["content"] == "hi"
        assert result["name"] == "user1"

    def test_serialize_user_message_no_name(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        msg = UserMessage(content="hi")
        result = OpenAIMessageSerializer.serialize(msg)
        assert "name" not in result

    def test_serialize_system_message(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        msg = SystemMessage(content="sys", name="system1")
        result = OpenAIMessageSerializer.serialize(msg)
        assert result["role"] == "system"
        assert result["name"] == "system1"

    def test_serialize_system_message_no_name(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        msg = SystemMessage(content="sys")
        result = OpenAIMessageSerializer.serialize(msg)
        assert "name" not in result

    def test_serialize_assistant_message_full(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{}'))
        msg = AssistantMessage(content="resp", name="bot", refusal="no", tool_calls=[tc])
        result = OpenAIMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "resp"
        assert result["name"] == "bot"
        assert result["refusal"] == "no"
        assert len(result["tool_calls"]) == 1

    def test_serialize_assistant_message_none_content(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        msg = AssistantMessage(content=None)
        result = OpenAIMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        assert "content" not in result

    def test_serialize_unknown_message_raises(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        with pytest.raises(ValueError, match="Unknown message type"):
            OpenAIMessageSerializer.serialize(MagicMock())

    def test_serialize_messages(self):
        from openbrowser.llm.openai.serializer import OpenAIMessageSerializer

        msgs = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        result = OpenAIMessageSerializer.serialize_messages(msgs)
        assert len(result) == 3


# ===========================================================================
# Tests for ChatOpenAI
# ===========================================================================
class TestChatOpenAI:
    """Tests for openai/chat.py coverage."""

    def _make_chat(self, **kwargs):
        from openbrowser.llm.openai.chat import ChatOpenAI

        defaults = {"model": "gpt-4o", "api_key": "test-key"}
        defaults.update(kwargs)
        return ChatOpenAI(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "openai"

    def test_name(self):
        chat = self._make_chat(model="gpt-4o-mini")
        assert chat.name == "gpt-4o-mini"

    def test_get_client_params(self):
        chat = self._make_chat(
            organization="org1",
            project="proj1",
            base_url="https://api.example.com",
        )
        params = chat._get_client_params()
        assert params["api_key"] == "test-key"
        assert params["organization"] == "org1"
        assert params["project"] == "proj1"

    def test_get_client_params_filters_none(self):
        chat = self._make_chat()
        params = chat._get_client_params()
        assert "organization" not in params
        assert "project" not in params

    def test_get_client_params_with_http_client(self):
        import httpx

        http_client = httpx.AsyncClient()
        chat = self._make_chat(http_client=http_client)
        params = chat._get_client_params()
        assert params["http_client"] is http_client

    def test_get_client(self):
        chat = self._make_chat()
        client = chat.get_client()
        from openai import AsyncOpenAI

        assert isinstance(client, AsyncOpenAI)

    def test_get_usage_with_details(self):
        chat = self._make_chat()
        resp = _make_response(usage=_make_usage(cached=5, reasoning=3))
        usage = chat._get_usage(resp)
        assert usage.prompt_tokens == 10
        assert usage.prompt_cached_tokens == 5
        assert usage.completion_tokens == 8  # 5 + 3 reasoning

    def test_get_usage_no_details(self):
        chat = self._make_chat()
        resp = _make_response(usage=_make_usage())
        usage = chat._get_usage(resp)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5

    def test_get_usage_none(self):
        chat = self._make_chat()
        resp = _make_response()
        resp.usage = None
        usage = chat._get_usage(resp)
        assert usage is None

    def test_get_usage_completion_details_none_reasoning(self):
        chat = self._make_chat()
        u = _make_usage()
        comp_details = MagicMock()
        comp_details.reasoning_tokens = None
        u.completion_tokens_details = comp_details
        resp = _make_response(usage=u)
        usage = chat._get_usage(resp)
        assert usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_ainvoke_text(self):
        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = _make_response(content="test reply")
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "test reply"

    @pytest.mark.asyncio
    async def test_ainvoke_text_with_all_params(self):
        chat = self._make_chat(
            temperature=0.5,
            frequency_penalty=0.2,
            max_completion_tokens=2048,
            top_p=0.9,
            seed=42,
            service_tier="auto",
        )
        mock_client = AsyncMock()
        resp = _make_response()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert isinstance(result, ChatInvokeCompletion)

    @pytest.mark.asyncio
    async def test_ainvoke_reasoning_model(self):
        """Test that reasoning models remove temperature and frequency_penalty."""
        chat = self._make_chat(model="o4-mini", temperature=0.5, frequency_penalty=0.3)
        mock_client = AsyncMock()
        resp = _make_response()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        # The call should have gone through
        assert isinstance(result, ChatInvokeCompletion)
        call_kwargs = mock_client.chat.completions.create.call_args
        assert "reasoning_effort" in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = _make_response(content='{"field": "value"}')
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)
        assert result.completion.field == "value"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_none_content(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = _make_response(content=None)
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Failed to parse"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_with_schema_in_system(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(add_schema_to_system_prompt=True)
        mock_client = AsyncMock()
        resp = _make_response(content='{"field": "value"}')
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        msgs = [SystemMessage(content="be helpful"), UserMessage(content="extract")]

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke(msgs, output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_with_schema_in_system_list_content(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(add_schema_to_system_prompt=True)
        mock_client = AsyncMock()
        resp = _make_response(content='{"field": "value"}')
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        msgs = [
            SystemMessage(content=[ContentPartTextParam(text="be helpful")]),
            UserMessage(content="extract"),
        ]

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke(msgs, output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_rate_limit_error(self):
        from openai import RateLimitError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "rate limited"}}
        error = RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_connection_error(self):
        from openai import APIConnectionError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        error = APIConnectionError(request=MagicMock())
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_status_error(self):
        from openai import APIStatusError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "server error"}}
        error = APIStatusError(
            message="server error",
            response=mock_response,
            body=None,
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_status_error_non_json(self):
        """Test APIStatusError when json() raises."""
        from openai import APIStatusError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.json.side_effect = Exception("not json")
        mock_response.text = "plain error text"
        error = APIStatusError(
            message="server error",
            response=mock_response,
            body=None,
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_generic_exception(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_rate_limit_error_dict_message(self):
        """Test rate limit error with dict error message."""
        from openai import RateLimitError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": "just a string"}
        error = RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])
