"""Tests for Groq LLM provider modules - chat.py, parser.py, and serializer.py.

Covers:
  src/openbrowser/llm/groq/chat.py
  src/openbrowser/llm/groq/parser.py
  src/openbrowser/llm/groq/serializer.py
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

groq_mod = pytest.importorskip("groq")

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
def _make_usage(prompt=10, completion=5, total=15):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = total
    return usage


def _make_choice(content="hello", finish_reason="stop"):
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    return choice


def _make_response(content="hello", usage=None):
    resp = MagicMock()
    resp.choices = [_make_choice(content)]
    resp.usage = usage if usage is not None else _make_usage()
    return resp


# ===========================================================================
# Tests for GroqMessageSerializer
# ===========================================================================
class TestGroqSerializer:
    """Tests for groq/serializer.py coverage."""

    def test_serialize_content_part_text(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        part = ContentPartTextParam(text="hello")
        result = GroqMessageSerializer._serialize_content_part_text(part)
        assert result["text"] == "hello"
        assert result["type"] == "text"

    def test_serialize_content_part_image(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        part = ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png", detail="low"))
        result = GroqMessageSerializer._serialize_content_part_image(part)
        assert result["type"] == "image_url"
        assert result["image_url"]["detail"] == "low"

    def test_serialize_user_content_string(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        result = GroqMessageSerializer._serialize_user_content("hello")
        assert result == "hello"

    def test_serialize_user_content_list(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        parts = [
            ContentPartTextParam(text="text"),
            ContentPartImageParam(image_url=ImageURL(url="https://img.com/a.png")),
        ]
        result = GroqMessageSerializer._serialize_user_content(parts)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_serialize_system_content_string(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        result = GroqMessageSerializer._serialize_system_content("system")
        assert result == "system"

    def test_serialize_system_content_list(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        parts = [ContentPartTextParam(text="p1"), ContentPartTextParam(text="p2")]
        result = GroqMessageSerializer._serialize_system_content(parts)
        assert result == "p1\np2"

    def test_serialize_assistant_content_none(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        result = GroqMessageSerializer._serialize_assistant_content(None)
        assert result is None

    def test_serialize_assistant_content_string(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        result = GroqMessageSerializer._serialize_assistant_content("resp")
        assert result == "resp"

    def test_serialize_assistant_content_list(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        parts = [ContentPartTextParam(text="text1"), ContentPartTextParam(text="text2")]
        result = GroqMessageSerializer._serialize_assistant_content(parts)
        assert result == "text1\ntext2"

    def test_serialize_tool_call(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{"a": 1}'))
        result = GroqMessageSerializer._serialize_tool_call(tc)
        assert result["id"] == "tc1"
        assert result["function"]["name"] == "fn"
        assert result["type"] == "function"

    def test_serialize_user_message(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        msg = UserMessage(content="hi", name="user1")
        result = GroqMessageSerializer.serialize(msg)
        assert result["role"] == "user"
        assert result["name"] == "user1"

    def test_serialize_user_message_no_name(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        msg = UserMessage(content="hi")
        result = GroqMessageSerializer.serialize(msg)
        assert "name" not in result

    def test_serialize_system_message(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        msg = SystemMessage(content="sys", name="sys1")
        result = GroqMessageSerializer.serialize(msg)
        assert result["role"] == "system"
        assert result["name"] == "sys1"

    def test_serialize_system_message_no_name(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        msg = SystemMessage(content="sys")
        result = GroqMessageSerializer.serialize(msg)
        assert "name" not in result

    def test_serialize_assistant_message(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{}'))
        msg = AssistantMessage(content="resp", name="bot", tool_calls=[tc])
        result = GroqMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        assert result["name"] == "bot"
        assert len(result["tool_calls"]) == 1

    def test_serialize_assistant_message_none_content(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        msg = AssistantMessage(content=None)
        result = GroqMessageSerializer.serialize(msg)
        assert "content" not in result

    def test_serialize_unknown_message_raises(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        with pytest.raises(ValueError, match="Unknown message type"):
            GroqMessageSerializer.serialize(MagicMock())

    def test_serialize_messages(self):
        from openbrowser.llm.groq.serializer import GroqMessageSerializer

        msgs = [SystemMessage(content="sys"), UserMessage(content="hi")]
        result = GroqMessageSerializer.serialize_messages(msgs)
        assert len(result) == 2


# ===========================================================================
# Tests for try_parse_groq_failed_generation (parser.py)
# ===========================================================================
class TestGroqParser:
    """Tests for groq/parser.py coverage."""

    def _make_api_status_error(self, failed_gen):
        mock_error = MagicMock()
        mock_error.body = {"error": {"failed_generation": failed_gen}}
        mock_error.response = MagicMock()
        mock_error.response.text = "error text"
        return mock_error

    def test_parse_valid_json(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation

        class MyOutput(PydanticBaseModel):
            field: str

        error = self._make_api_status_error('{"field": "value"}')
        result = try_parse_groq_failed_generation(error, MyOutput)
        assert result.field == "value"

    def test_parse_json_in_code_block(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation

        class MyOutput(PydanticBaseModel):
            field: str

        error = self._make_api_status_error('```json\n{"field": "value"}\n```')
        result = try_parse_groq_failed_generation(error, MyOutput)
        assert result.field == "value"

    def test_parse_json_with_html_prefix(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation

        class MyOutput(PydanticBaseModel):
            field: str

        error = self._make_api_status_error('<|header_start|>assistant<|header_end|>{"field": "value"}')
        result = try_parse_groq_failed_generation(error, MyOutput)
        assert result.field == "value"

    def test_parse_json_with_html_suffix(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation

        class MyOutput(PydanticBaseModel):
            field: str

        error = self._make_api_status_error('{"field": "value"}</function>')
        result = try_parse_groq_failed_generation(error, MyOutput)
        assert result.field == "value"

    def test_parse_json_with_html_pipe_suffix(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation

        class MyOutput(PydanticBaseModel):
            field: str

        error = self._make_api_status_error('{"field": "value"}<|end|>')
        result = try_parse_groq_failed_generation(error, MyOutput)
        assert result.field == "value"

    def test_parse_invalid_json_brace_counting(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation

        class MyOutput(PydanticBaseModel):
            field: str

        # Content that ends with '}' but has invalid trailing braces so
        # brace-counting logic kicks in to find the balanced end position.
        error = self._make_api_status_error('{"field": "value"} extra junk}')
        result = try_parse_groq_failed_generation(error, MyOutput)
        assert result.field == "value"

    def test_parse_list_response(self):
        """Test the list-unwrap branch (line 78-79) where json.loads returns
        a single-element list containing a dict."""
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation

        class MyOutput(PydanticBaseModel):
            field: str

        # The regex cleanup makes it hard to pass a raw JSON list through
        # to json.loads, so we patch json.loads to return a list on the
        # final successful parse call (line 75 in parser.py).
        error = self._make_api_status_error('{"field": "value"}')
        original_loads = json.loads

        call_count = 0
        def patched_loads(s, **kwargs):
            nonlocal call_count
            call_count += 1
            result = original_loads(s, **kwargs)
            # Call #3 is the final json.loads at line 75:
            #  Call 1 = line 52 (endswith-'}' validity check)
            #  Call 2 = line 99 inside _fix_control_characters_in_json
            #  Call 3 = line 75 (the actual parse)
            if call_count == 3 and isinstance(result, dict):
                return [result]
            return result

        with patch("openbrowser.llm.groq.parser.json.loads", side_effect=patched_loads):
            result = try_parse_groq_failed_generation(error, MyOutput)
        assert result.field == "value"

    def test_parse_key_error(self):
        from openbrowser.llm.groq.parser import ParseFailedGenerationError, try_parse_groq_failed_generation
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        mock_error = MagicMock()
        mock_error.body = {"error": {}}  # no 'failed_generation' key
        mock_error.response = MagicMock()
        mock_error.response.text = "error"

        with pytest.raises(ParseFailedGenerationError):
            try_parse_groq_failed_generation(mock_error, MyOutput)

    def test_parse_invalid_json_raises(self):
        from openbrowser.llm.groq.parser import try_parse_groq_failed_generation
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        error = self._make_api_status_error("completely not json at all {{{")
        with pytest.raises(ValueError, match="Could not parse"):
            try_parse_groq_failed_generation(error, MyOutput)

    def test_parse_generic_exception(self):
        from openbrowser.llm.groq.parser import ParseFailedGenerationError, try_parse_groq_failed_generation
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: int  # will fail validation

        error = self._make_api_status_error('{"field": "not_an_int_really"}')
        # pydantic v2 may coerce, but let's use a truly wrong type
        error = self._make_api_status_error('{"wrong_key": 1}')
        with pytest.raises(ParseFailedGenerationError):
            try_parse_groq_failed_generation(error, MyOutput)

    def test_fix_control_characters_valid_json(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        valid = '{"key": "value"}'
        assert _fix_control_characters_in_json(valid) == valid

    def test_fix_control_characters_newline(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        content = '{"key": "line1\nline2"}'
        result = _fix_control_characters_in_json(content)
        assert "\\n" in result
        # Should be valid JSON now
        parsed = json.loads(result)
        assert "line1" in parsed["key"]

    def test_fix_control_characters_tab(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        content = '{"key": "col1\tcol2"}'
        result = _fix_control_characters_in_json(content)
        assert "\\t" in result

    def test_fix_control_characters_carriage_return(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        content = '{"key": "line1\rline2"}'
        result = _fix_control_characters_in_json(content)
        assert "\\r" in result

    def test_fix_control_characters_backspace(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        content = '{"key": "ab\bcd"}'
        result = _fix_control_characters_in_json(content)
        assert "\\b" in result

    def test_fix_control_characters_formfeed(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        content = '{"key": "ab\fcd"}'
        result = _fix_control_characters_in_json(content)
        assert "\\f" in result

    def test_fix_control_characters_other_control(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        # Use a control character like BEL (0x07)
        content = '{"key": "ab' + chr(7) + 'cd"}'
        result = _fix_control_characters_in_json(content)
        assert "\\u0007" in result

    def test_fix_control_characters_escaped_chars(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        content = '{"key": "escaped\\nnewline"}'
        result = _fix_control_characters_in_json(content)
        # Already valid JSON, should not double-escape
        assert result == content

    def test_fix_control_characters_with_escape_and_control_char(self):
        """Cover lines 124-125 and 128-129: backslash escape handling inside strings
        that also have literal control characters."""
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        # Build a string with BOTH a literal newline (makes it invalid JSON)
        # AND a backslash-escaped character like \" inside the string value.
        # This forces the function to enter the character-by-character processing
        # and encounter both the escape char (\\ at line 128) and the escaped
        # char (the next character after backslash, at line 124).
        content = '{"key": "has \\"quote\\" and literal\nnewline"}'
        result = _fix_control_characters_in_json(content)
        # The literal newline should be escaped to \\n, and \\" should be preserved
        parsed = json.loads(result)
        assert "quote" in parsed["key"]
        assert "literal" in parsed["key"]

    def test_fix_control_characters_normal_string(self):
        from openbrowser.llm.groq.parser import _fix_control_characters_in_json

        content = '{"key": "normal text"}'
        result = _fix_control_characters_in_json(content)
        assert result == content


# ===========================================================================
# Tests for ChatGroq
# ===========================================================================
class TestChatGroq:
    """Tests for groq/chat.py coverage."""

    def _make_chat(self, **kwargs):
        from openbrowser.llm.groq.chat import ChatGroq

        defaults = {"model": "meta-llama/llama-4-scout-17b-16e-instruct", "api_key": "test-key"}
        defaults.update(kwargs)
        return ChatGroq(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "groq"

    def test_name(self):
        chat = self._make_chat()
        assert "llama" in chat.name

    def test_get_client(self):
        chat = self._make_chat()
        client = chat.get_client()
        from groq import AsyncGroq

        assert isinstance(client, AsyncGroq)

    def test_get_usage(self):
        chat = self._make_chat()
        resp = _make_response()
        usage = chat._get_usage(resp)
        assert usage.prompt_tokens == 10

    def test_get_usage_none(self):
        chat = self._make_chat()
        resp = MagicMock()
        resp.usage = None
        usage = chat._get_usage(resp)
        assert usage is None

    @pytest.mark.asyncio
    async def test_ainvoke_regular(self):
        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = _make_response(content="test reply")
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "test reply"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_json_schema(self):
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

    @pytest.mark.asyncio
    async def test_ainvoke_structured_tool_calling(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat(model="moonshotai/kimi-k2-instruct")
        mock_client = AsyncMock()
        resp = _make_response(content='{"field": "value"}')
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_no_content(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = _make_response(content=None)
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="No content"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_rate_limit_error(self):
        from groq import RateLimitError
        from openbrowser.llm.exceptions import ModelRateLimitError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"
        mock_response.headers = {}
        error = RateLimitError(message="rate limited", response=mock_response, body=None)
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelRateLimitError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_response_validation_error(self):
        from groq import APIResponseValidationError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "validation error"
        mock_response.headers = {}
        error = APIResponseValidationError(response=mock_response, body=None, message="bad")
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_status_error_no_format(self):
        from groq import APIStatusError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"
        mock_response.headers = {}
        error = APIStatusError(message="server error", response=mock_response, body=None)
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_status_error_with_format_fallback_success(self):
        from pydantic import BaseModel as PydanticBaseModel
        from groq import APIStatusError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "error"
        mock_response.headers = {}
        error = APIStatusError(message="bad", response=mock_response, body=None)
        error.body = {"error": {"failed_generation": '{"field": "fallback"}'}}
        error.response = mock_response
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert result.completion.field == "fallback"

    @pytest.mark.asyncio
    async def test_ainvoke_api_status_error_with_format_fallback_failure(self):
        from pydantic import BaseModel as PydanticBaseModel
        from groq import APIStatusError
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "error"
        mock_response.headers = {}
        error = APIStatusError(message="bad", response=mock_response, body=None)
        error.body = {"error": {}}  # no failed_generation
        error.response = mock_response
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_api_error(self):
        """Cover line 145: Groq APIError properly caught."""
        import httpx
        from groq import APIError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        req = httpx.Request("POST", "http://test.com")
        error = APIError("api error", request=req, body=None)
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="api error"):
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
