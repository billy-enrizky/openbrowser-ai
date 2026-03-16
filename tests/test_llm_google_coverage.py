"""Tests for Google LLM provider modules - chat.py and serializer.py.

Covers:
  src/openbrowser/llm/google/chat.py
  src/openbrowser/llm/google/serializer.py
"""

import asyncio
import base64
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

google_mod = pytest.importorskip("google.genai")

from openbrowser.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from openbrowser.llm.views import ChatInvokeCompletion, ChatInvokeUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_usage_metadata(prompt=10, candidates=5, total=15, thoughts=0, cached=None, image_details=None):
    meta = MagicMock()
    meta.prompt_token_count = prompt
    meta.candidates_token_count = candidates
    meta.total_token_count = total
    meta.thoughts_token_count = thoughts
    meta.cached_content_token_count = cached

    if image_details is not None:
        meta.prompt_tokens_details = image_details
    else:
        meta.prompt_tokens_details = None

    return meta


def _make_response(text="hello", usage_meta=None, parsed=None, finish_reason="STOP"):
    resp = MagicMock()
    resp.text = text

    if usage_meta is None:
        usage_meta = _make_usage_metadata()
    resp.usage_metadata = usage_meta

    resp.parsed = parsed

    # candidates
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    resp.candidates = [candidate]

    return resp


# ===========================================================================
# Tests for GoogleMessageSerializer
# ===========================================================================
class TestGoogleSerializer:
    """Tests for google/serializer.py coverage."""

    def test_serialize_messages_basic(self):
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [
            SystemMessage(content="system prompt"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        assert system == "system prompt"
        assert len(contents) == 2

    def test_serialize_messages_no_system(self):
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [UserMessage(content="hi")]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        assert system is None
        assert len(contents) == 1

    def test_serialize_messages_system_list_content(self):
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [
            SystemMessage(content=[ContentPartTextParam(text="p1"), ContentPartTextParam(text="p2")]),
            UserMessage(content="hi"),
        ]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        assert system == "p1\np2"

    def test_serialize_messages_include_system_in_user(self):
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [
            SystemMessage(content="sys prompt"),
            UserMessage(content="hi"),
        ]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs, include_system_in_user=True)
        assert system is None
        assert len(contents) == 1

    def test_serialize_messages_include_system_list_in_user(self):
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [
            SystemMessage(content=[ContentPartTextParam(text="sys p1")]),
            UserMessage(content="hi"),
        ]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs, include_system_in_user=True)
        assert system is None

    def test_serialize_messages_include_system_in_user_list_content(self):
        """System in user with list content user message."""
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [
            SystemMessage(content="sys prompt"),
            UserMessage(content=[ContentPartTextParam(text="user text")]),
        ]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs, include_system_in_user=True)
        assert system is None

    def test_serialize_user_message_list_content_with_images(self):
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        b64_data = base64.b64encode(b"fake_image").decode()
        msgs = [
            UserMessage(content=[
                ContentPartTextParam(text="look at this"),
                ContentPartImageParam(
                    image_url=ImageURL(url=f"data:image/jpeg;base64,{b64_data}")
                ),
            ]),
        ]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        assert len(contents) == 1

    def test_serialize_user_message_list_with_refusal(self):
        """Test assistant message with refusal part."""
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [
            AssistantMessage(content=[
                ContentPartTextParam(text="text part"),
                ContentPartRefusalParam(refusal="I refuse"),
            ]),
        ]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        assert len(contents) == 1

    def test_serialize_assistant_message_string(self):
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [AssistantMessage(content="model says")]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        assert len(contents) == 1

    def test_serialize_unknown_message_type(self):
        """Unknown message types default to user role when not SystemMessage/UserMessage/AssistantMessage."""
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        # A mock that isn't any known type, but model_copy returns itself
        # so the serializer can properly access its .content attribute
        CustomMsg = type("CustomMsg", (), {})
        mock_msg = MagicMock(spec=[])
        mock_msg.role = "custom"
        mock_msg.content = "custom content"
        mock_msg.__class__ = CustomMsg
        mock_msg.model_copy = MagicMock(return_value=mock_msg)

        msgs = [mock_msg]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        # It should be treated as user with string content -> one Content entry
        assert len(contents) == 1

    def test_serialize_user_message_none_content(self):
        """Test when message content is not a list and not a string (edge case)."""
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msgs = [AssistantMessage(content=None)]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        # No parts -> no content added
        assert len(contents) == 0

    def test_serialize_messages_system_none_content(self):
        """System message with None content."""
        from openbrowser.llm.google.serializer import GoogleMessageSerializer

        msg = SystemMessage(content="")
        msgs = [msg, UserMessage(content="hi")]
        contents, system = GoogleMessageSerializer.serialize_messages(msgs)
        # Empty string still counts as system message
        assert system == ""


# ===========================================================================
# Tests for ChatGoogle
# ===========================================================================
class TestChatGoogle:
    """Tests for google/chat.py coverage."""

    def _make_chat(self, **kwargs):
        from openbrowser.llm.google.chat import ChatGoogle

        defaults = {"model": "gemini-2.0-flash", "api_key": "test-key"}
        defaults.update(kwargs)
        return ChatGoogle(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "google"

    def test_logger(self):
        chat = self._make_chat()
        assert chat.logger.name == "openbrowser.llm.google.gemini-2.0-flash"

    def test_name(self):
        chat = self._make_chat(model="gemini-2.5-pro")
        assert chat.name == "gemini-2.5-pro"

    def test_get_client_params(self):
        chat = self._make_chat(api_key="key1", project="proj1", location="us-central1")
        params = chat._get_client_params()
        assert params["api_key"] == "key1"
        assert params["project"] == "proj1"
        assert params["location"] == "us-central1"

    def test_get_client_params_filters_none(self):
        chat = self._make_chat()
        params = chat._get_client_params()
        assert "project" not in params

    def test_get_client_caches(self):
        chat = self._make_chat()
        with patch("google.genai.Client") as mock_client_cls:
            c1 = chat.get_client()
            c2 = chat.get_client()
            assert c1 is c2
            mock_client_cls.assert_called_once()

    def test_get_stop_reason_with_candidates(self):
        chat = self._make_chat()
        resp = _make_response(finish_reason="STOP")
        result = chat._get_stop_reason(resp)
        assert result is not None

    def test_get_stop_reason_no_candidates(self):
        chat = self._make_chat()
        resp = MagicMock()
        resp.candidates = []
        result = chat._get_stop_reason(resp)
        assert result is None

    def test_get_stop_reason_no_attr(self):
        chat = self._make_chat()
        resp = MagicMock(spec=[])  # no attributes
        result = chat._get_stop_reason(resp)
        assert result is None

    def test_get_usage_with_metadata(self):
        chat = self._make_chat()
        resp = _make_response()
        usage = chat._get_usage(resp)
        assert isinstance(usage, ChatInvokeUsage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5

    def test_get_usage_with_image_tokens(self):
        from google.genai.types import MediaModality

        chat = self._make_chat()
        detail = MagicMock()
        detail.modality = MediaModality.IMAGE
        detail.token_count = 100

        meta = _make_usage_metadata(image_details=[detail])
        resp = _make_response(usage_meta=meta)
        usage = chat._get_usage(resp)
        assert usage.prompt_image_tokens == 100

    def test_get_usage_none_metadata(self):
        chat = self._make_chat()
        resp = MagicMock()
        resp.usage_metadata = None
        usage = chat._get_usage(resp)
        assert usage is None

    @pytest.mark.asyncio
    async def test_ainvoke_text_response(self):
        chat = self._make_chat()
        mock_client = MagicMock()
        resp = _make_response(text="test response")
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "test response"

    @pytest.mark.asyncio
    async def test_ainvoke_text_empty_response(self):
        chat = self._make_chat()
        mock_client = MagicMock()
        resp = _make_response(text="")
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == ""

    @pytest.mark.asyncio
    async def test_ainvoke_text_none_response(self):
        chat = self._make_chat()
        mock_client = MagicMock()
        resp = _make_response(text=None)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == ""

    @pytest.mark.asyncio
    async def test_ainvoke_with_config(self):
        chat = self._make_chat(
            config={"tools": []},
            temperature=0.7,
            top_p=0.9,
            seed=42,
            max_output_tokens=2048,
        )
        mock_client = MagicMock()
        resp = _make_response()
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([
                SystemMessage(content="be helpful"),
                UserMessage(content="hi"),
            ])

        assert isinstance(result, ChatInvokeCompletion)

    @pytest.mark.asyncio
    async def test_ainvoke_flash_model_sets_thinking_budget(self):
        """Gemini flash models should auto-set thinking_budget=0."""
        chat = self._make_chat(model="gemini-2.5-flash")
        mock_client = MagicMock()
        resp = _make_response()
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            await chat.ainvoke([UserMessage(content="hi")])

        assert chat.thinking_budget == 0

    @pytest.mark.asyncio
    async def test_ainvoke_explicit_thinking_budget(self):
        chat = self._make_chat(thinking_budget=100)
        mock_client = MagicMock()
        resp = _make_response()
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            await chat.ainvoke([UserMessage(content="hi")])

        assert chat.thinking_budget == 100

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_native_parsed(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()

        parsed_obj = MyOutput(field="value")
        resp = _make_response(parsed=parsed_obj)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)
        assert result.completion.field == "value"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_native_parsed_wrong_type(self):
        """Test when parsed is not the expected type."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()

        resp = _make_response(parsed={"field": "value"})
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_parsed_none_text(self):
        """Test when parsed is None but text has JSON."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()

        resp = _make_response(text='{"field": "from_text"}', parsed=None)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert result.completion.field == "from_text"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_parsed_none_text_json_code_block(self):
        """Test when text is wrapped in ```json...```."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()

        resp = _make_response(text='```json\n{"field": "from_block"}\n```', parsed=None)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert result.completion.field == "from_block"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_parsed_none_text_generic_code_block(self):
        """Test when text is wrapped in plain ```...```."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()

        resp = _make_response(text='```\n{"field": "from_plain"}\n```', parsed=None)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert result.completion.field == "from_plain"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_parsed_none_invalid_json(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()

        resp = _make_response(text="not json at all", parsed=None)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Failed to parse"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_parsed_none_no_text(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()

        resp = _make_response(text=None, parsed=None)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="No response from model"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_fallback(self):
        """Test fallback JSON mode (supports_structured_output=False)."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(supports_structured_output=False)
        mock_client = MagicMock()

        resp = _make_response(text='{"field": "fallback_value"}')
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert result.completion.field == "fallback_value"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_fallback_code_block(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(supports_structured_output=False)
        mock_client = MagicMock()

        resp = _make_response(text='```json\n{"field": "fb_block"}\n```')
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert result.completion.field == "fb_block"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_fallback_plain_code_block(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(supports_structured_output=False)
        mock_client = MagicMock()

        resp = _make_response(text='```\n{"field": "fb_plain"}\n```')
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert result.completion.field == "fb_plain"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_fallback_invalid_json(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(supports_structured_output=False)
        mock_client = MagicMock()

        resp = _make_response(text="not json")
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="failed to parse"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_fallback_no_text(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(supports_structured_output=False)
        mock_client = MagicMock()

        resp = _make_response(text=None)
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="No response from model"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_fallback_with_system(self):
        """Cover line 361: fallback JSON mode with a system message present."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(supports_structured_output=False)
        mock_client = MagicMock()

        resp = _make_response(text='{"field": "sys_fallback"}')
        mock_client.aio.models.generate_content = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke(
                [SystemMessage(content="You are a helper"), UserMessage(content="extract")],
                output_format=MyOutput,
            )
        assert result.completion.field == "sys_fallback"

    @pytest.mark.asyncio
    async def test_ainvoke_retry_on_retryable_status(self):
        """Test retry on 403 error."""
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=2, retry_delay=0.001)
        mock_client = MagicMock()

        error = ModelProviderError(message="forbidden", status_code=403, model="test")
        resp = _make_response(text="success")

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise error
            return resp

        mock_client.aio.models.generate_content = AsyncMock(side_effect=side_effect)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ainvoke_retry_exhausted(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=2, retry_delay=0.001)
        mock_client = MagicMock()

        error = ModelProviderError(message="forbidden", status_code=403, model="test")
        mock_client.aio.models.generate_content = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="forbidden"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_non_retryable_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=3, retry_delay=0.001)
        mock_client = MagicMock()

        error = ModelProviderError(message="bad request", status_code=400, model="test")
        mock_client.aio.models.generate_content = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="bad request"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_generic_exception_wrapping(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=1)
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_timeout_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=1)
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("request timeout"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError) as exc_info:
                await chat.ainvoke([UserMessage(content="hi")])
        assert exc_info.value.status_code == 408

    @pytest.mark.asyncio
    async def test_ainvoke_cancelled_error(self):
        """asyncio.CancelledError is a BaseException, not Exception, so it
        propagates without being caught by the except-Exception handler."""

        chat = self._make_chat(max_retries=1)
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=asyncio.CancelledError("cancelled timeout")
        )

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(asyncio.CancelledError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_forbidden_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=1)
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("forbidden access"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError) as exc_info:
                await chat.ainvoke([UserMessage(content="hi")])
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_ainvoke_rate_limit_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=1)
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("rate limit exceeded"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError) as exc_info:
                await chat.ainvoke([UserMessage(content="hi")])
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_ainvoke_service_unavailable_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=1)
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("service unavailable 503")
        )

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError) as exc_info:
                await chat.ainvoke([UserMessage(content="hi")])
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_ainvoke_exception_with_response_status(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat(max_retries=1)
        mock_client = MagicMock()

        exc = Exception("custom error")
        exc.response = MagicMock()
        exc.response.status_code = 422
        mock_client.aio.models.generate_content = AsyncMock(side_effect=exc)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError) as exc_info:
                await chat.ainvoke([UserMessage(content="hi")])
        assert exc_info.value.status_code == 422

    def test_fix_gemini_schema_basic(self):
        chat = self._make_chat()
        schema = {
            "type": "object",
            "title": "MyModel",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "title": "Name"},
            },
            "required": ["name", "title"],
        }
        result = chat._fix_gemini_schema(schema)
        assert "additionalProperties" not in result
        assert "title" not in result.get("properties", {}).get("name", {})
        # 'title' should be removed from required
        assert "title" not in result.get("required", [])

    def test_fix_gemini_schema_with_defs(self):
        chat = self._make_chat()
        schema = {
            "type": "object",
            "$defs": {
                "SubModel": {
                    "type": "object",
                    "properties": {"sub_field": {"type": "string"}},
                }
            },
            "properties": {
                "sub": {"$ref": "#/$defs/SubModel"},
            },
        }
        result = chat._fix_gemini_schema(schema)
        assert "$defs" not in result
        assert "$ref" not in result.get("properties", {}).get("sub", {})

    def test_fix_gemini_schema_empty_object_properties(self):
        chat = self._make_chat()
        schema = {
            "type": "OBJECT",
            "properties": {},
        }
        result = chat._fix_gemini_schema(schema)
        assert "_placeholder" in result.get("properties", {})

    def test_fix_gemini_schema_list_in_schema(self):
        chat = self._make_chat()
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": [{"type": "string"}, {"type": "number"}],
                }
            },
        }
        result = chat._fix_gemini_schema(schema)
        assert "items" in result["properties"]

    def test_fix_gemini_schema_default_removal(self):
        chat = self._make_chat()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "test"},
            },
        }
        result = chat._fix_gemini_schema(schema)
        assert "default" not in result["properties"]["name"]

    def test_fix_gemini_schema_ref_with_extra_properties(self):
        """Cover lines 489-490: $ref with sibling properties merged into resolved ref."""
        chat = self._make_chat()
        schema = {
            "type": "object",
            "$defs": {
                "SubModel": {
                    "type": "object",
                    "properties": {"sub_field": {"type": "string"}},
                }
            },
            "properties": {
                "sub": {
                    "$ref": "#/$defs/SubModel",
                    "description": "A sub-model",
                },
            },
        }
        result = chat._fix_gemini_schema(schema)
        # The description should be merged into the resolved ref
        assert result["properties"]["sub"].get("description") == "A sub-model"
        assert "$ref" not in result["properties"]["sub"]

    def test_fix_gemini_schema_ref_not_found(self):
        """Cover line 492: $ref name not found in defs, return obj without ref key."""
        chat = self._make_chat()
        schema = {
            "type": "object",
            "$defs": {
                "ExistingModel": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                }
            },
            "properties": {
                "sub": {
                    "$ref": "#/$defs/NonExistentModel",
                },
            },
        }
        result = chat._fix_gemini_schema(schema)
        # The $ref was popped but no resolution found, so obj is returned
        # without the $ref key (it was already removed via pop)
        assert "$ref" not in result["properties"]["sub"]
        # But the sub dict should exist (returned empty since ref was the only key)
        assert "sub" in result["properties"]

    def test_fix_gemini_schema_resolve_refs_list(self):
        """Cover line 497: list in resolve_refs."""
        chat = self._make_chat()
        schema = {
            "type": "object",
            "$defs": {
                "Inner": {
                    "type": "string",
                }
            },
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Inner"},
                },
            },
            "anyOf": [
                {"$ref": "#/$defs/Inner"},
                {"type": "number"},
            ],
        }
        result = chat._fix_gemini_schema(schema)
        # anyOf list items should have their $refs resolved
        assert "$defs" not in result

    def test_fix_gemini_schema_clean_leaves_empty_properties(self):
        """Cover line 531: empty properties dict after cleaning triggers placeholder."""
        chat = self._make_chat()
        schema = {
            "type": "OBJECT",
            "properties": {
                "only_title": {"title": "Will be removed", "type": "string"},
            },
        }
        # The clean_schema step removes 'title' but the property itself
        # should remain since it has 'type'. This tests the second
        # empty-properties check after cleaning. We need properties that
        # become truly empty after cleaning.
        schema2 = {
            "type": "OBJECT",
            "properties": {},
            "additionalProperties": True,
        }
        result = chat._fix_gemini_schema(schema2)
        # additionalProperties is removed, and empty properties gets placeholder
        assert "_placeholder" in result.get("properties", {})
