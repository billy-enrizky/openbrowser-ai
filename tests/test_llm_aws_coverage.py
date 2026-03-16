"""Tests for AWS LLM provider modules.

Covers:
  src/openbrowser/llm/aws/__init__.py
  src/openbrowser/llm/aws/chat_anthropic.py
  src/openbrowser/llm/aws/chat_bedrock.py
  src/openbrowser/llm/aws/serializer.py
"""

import base64
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

anthropic_mod = pytest.importorskip("anthropic")

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


# ===========================================================================
# Tests for AWS __init__.py (lazy imports)
# ===========================================================================
class TestAWSInit:
    """Tests for aws/__init__.py coverage."""

    def test_lazy_import_chat_aws_bedrock(self):
        from openbrowser.llm import aws

        cls = getattr(aws, "ChatAWSBedrock")
        assert cls is not None

    def test_lazy_import_chat_anthropic_bedrock(self):
        from openbrowser.llm import aws

        cls = getattr(aws, "ChatAnthropicBedrock")
        assert cls is not None

    def test_lazy_import_unknown_raises(self):
        from openbrowser.llm import aws

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(aws, "NonExistentClass")

    def test_lazy_import_caches(self):
        from openbrowser.llm import aws

        cls1 = getattr(aws, "ChatAWSBedrock")
        cls2 = getattr(aws, "ChatAWSBedrock")
        assert cls1 is cls2


# ===========================================================================
# Tests for AWSBedrockMessageSerializer
# ===========================================================================
class TestAWSBedrockSerializer:
    """Tests for aws/serializer.py coverage."""

    def test_is_base64_image(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        assert AWSBedrockMessageSerializer._is_base64_image("data:image/png;base64,abc") is True
        assert AWSBedrockMessageSerializer._is_base64_image("https://example.com/img.png") is False

    def test_is_url_image(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        assert AWSBedrockMessageSerializer._is_url_image("https://example.com/img.png") is True
        assert AWSBedrockMessageSerializer._is_url_image("https://example.com/img.jpg") is True
        assert AWSBedrockMessageSerializer._is_url_image("https://example.com/img.gif") is True
        assert AWSBedrockMessageSerializer._is_url_image("https://example.com/img.webp") is True
        assert AWSBedrockMessageSerializer._is_url_image("data:image/png;base64,abc") is False
        assert AWSBedrockMessageSerializer._is_url_image("https://example.com/page.html") is False

    def test_parse_base64_url_jpeg(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        b64_data = base64.b64encode(b"fake").decode()
        fmt, data = AWSBedrockMessageSerializer._parse_base64_url(f"data:image/jpeg;base64,{b64_data}")
        assert fmt == "jpeg"
        assert isinstance(data, bytes)

    def test_parse_base64_url_png(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        b64_data = base64.b64encode(b"fake").decode()
        fmt, data = AWSBedrockMessageSerializer._parse_base64_url(f"data:image/png;base64,{b64_data}")
        assert fmt == "png"

    def test_parse_base64_url_gif(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        b64_data = base64.b64encode(b"fake").decode()
        fmt, data = AWSBedrockMessageSerializer._parse_base64_url(f"data:image/gif;base64,{b64_data}")
        assert fmt == "gif"

    def test_parse_base64_url_unknown_format(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        b64_data = base64.b64encode(b"fake").decode()
        fmt, data = AWSBedrockMessageSerializer._parse_base64_url(f"data:application/octet;base64,{b64_data}")
        assert fmt == "jpeg"  # default

    def test_parse_base64_url_invalid_raises(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        with pytest.raises(ValueError, match="Invalid base64 URL"):
            AWSBedrockMessageSerializer._parse_base64_url("https://example.com/img.png")

    def test_parse_base64_url_bad_data(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        with pytest.raises(ValueError, match="Failed to decode"):
            AWSBedrockMessageSerializer._parse_base64_url("data:image/jpeg;base64,!!!invalid!!!")

    def test_download_and_convert_image(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = b"image_data"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            fmt, data = AWSBedrockMessageSerializer._download_and_convert_image("https://example.com/img.png")
        assert fmt == "png"
        assert data == b"image_data"

    def test_download_and_convert_image_jpeg_url(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        mock_response = MagicMock()
        mock_response.headers = {"content-type": ""}
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            fmt, _ = AWSBedrockMessageSerializer._download_and_convert_image("https://example.com/photo.jpg")
        assert fmt == "jpeg"

    def test_download_and_convert_image_gif(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/gif"}
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            fmt, _ = AWSBedrockMessageSerializer._download_and_convert_image("https://example.com/img.gif")
        assert fmt == "gif"

    def test_download_and_convert_image_webp(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/webp"}
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            fmt, _ = AWSBedrockMessageSerializer._download_and_convert_image("https://example.com/img.webp")
        assert fmt == "webp"

    def test_download_and_convert_image_unknown(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            fmt, _ = AWSBedrockMessageSerializer._download_and_convert_image("https://example.com/img.bmp")
        assert fmt == "jpeg"  # default

    def test_download_and_convert_image_failure(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        with patch("httpx.get", side_effect=Exception("network error")):
            with pytest.raises(ValueError, match="Failed to download"):
                AWSBedrockMessageSerializer._download_and_convert_image("https://example.com/img.png")

    def test_serialize_content_part_text(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        part = ContentPartTextParam(text="hello")
        result = AWSBedrockMessageSerializer._serialize_content_part_text(part)
        assert result == {"text": "hello"}

    def test_serialize_content_part_image_base64(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        b64 = base64.b64encode(b"fake").decode()
        part = ContentPartImageParam(image_url=ImageURL(url=f"data:image/png;base64,{b64}"))
        result = AWSBedrockMessageSerializer._serialize_content_part_image(part)
        assert "image" in result

    def test_serialize_content_part_image_url(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()

        part = ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.jpeg"))
        with patch("httpx.get", return_value=mock_response):
            result = AWSBedrockMessageSerializer._serialize_content_part_image(part)
        assert "image" in result

    def test_serialize_content_part_image_unsupported(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        part = ContentPartImageParam(image_url=ImageURL(url="ftp://example.com/img.png"))
        with pytest.raises(ValueError, match="Unsupported image URL format"):
            AWSBedrockMessageSerializer._serialize_content_part_image(part)

    def test_serialize_user_content_string(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        result = AWSBedrockMessageSerializer._serialize_user_content("hello")
        assert result == [{"text": "hello"}]

    def test_serialize_user_content_list(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        b64 = base64.b64encode(b"fake").decode()
        parts = [
            ContentPartTextParam(text="text"),
            ContentPartImageParam(image_url=ImageURL(url=f"data:image/png;base64,{b64}")),
        ]
        result = AWSBedrockMessageSerializer._serialize_user_content(parts)
        assert len(result) == 2

    def test_serialize_system_content_string(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        result = AWSBedrockMessageSerializer._serialize_system_content("sys prompt")
        assert result == [{"text": "sys prompt"}]

    def test_serialize_system_content_list(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        parts = [ContentPartTextParam(text="p1"), ContentPartTextParam(text="p2")]
        result = AWSBedrockMessageSerializer._serialize_system_content(parts)
        assert len(result) == 2

    def test_serialize_assistant_content_none(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        result = AWSBedrockMessageSerializer._serialize_assistant_content(None)
        assert result == []

    def test_serialize_assistant_content_string(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        result = AWSBedrockMessageSerializer._serialize_assistant_content("resp")
        assert result == [{"text": "resp"}]

    def test_serialize_assistant_content_list(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        parts = [ContentPartTextParam(text="text"), ContentPartRefusalParam(refusal="no")]
        result = AWSBedrockMessageSerializer._serialize_assistant_content(parts)
        # Only text parts are serialized
        assert len(result) == 1

    def test_serialize_tool_call(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{"a": 1}'))
        result = AWSBedrockMessageSerializer._serialize_tool_call(tc)
        assert result["toolUse"]["toolUseId"] == "tc1"
        assert result["toolUse"]["input"] == {"a": 1}

    def test_serialize_tool_call_invalid_json(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        tc = ToolCall(id="tc2", function=Function(name="fn", arguments="not json"))
        result = AWSBedrockMessageSerializer._serialize_tool_call(tc)
        assert result["toolUse"]["input"] == {"arguments": "not json"}

    def test_serialize_user_message(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        msg = UserMessage(content="hello")
        result = AWSBedrockMessageSerializer.serialize(msg)
        assert result["role"] == "user"

    def test_serialize_system_message(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        msg = SystemMessage(content="sys")
        result = AWSBedrockMessageSerializer.serialize(msg)
        assert isinstance(result, SystemMessage)

    def test_serialize_assistant_message(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        msg = AssistantMessage(content="resp")
        result = AWSBedrockMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"

    def test_serialize_assistant_message_with_tools(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{}'))
        msg = AssistantMessage(content="resp", tool_calls=[tc])
        result = AWSBedrockMessageSerializer.serialize(msg)
        assert len(result["content"]) == 2  # text + tool_use

    def test_serialize_assistant_message_no_content(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        msg = AssistantMessage(content=None)
        result = AWSBedrockMessageSerializer.serialize(msg)
        assert result["content"] == [{"text": ""}]

    def test_serialize_unknown_message_raises(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        with pytest.raises(ValueError, match="Unknown message type"):
            AWSBedrockMessageSerializer.serialize(MagicMock())

    def test_serialize_messages(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        msgs = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        bedrock_msgs, system = AWSBedrockMessageSerializer.serialize_messages(msgs)
        assert system == [{"text": "sys"}]
        assert len(bedrock_msgs) == 2

    def test_serialize_messages_no_system(self):
        from openbrowser.llm.aws.serializer import AWSBedrockMessageSerializer

        msgs = [UserMessage(content="hi")]
        bedrock_msgs, system = AWSBedrockMessageSerializer.serialize_messages(msgs)
        assert system is None
        assert len(bedrock_msgs) == 1


# ===========================================================================
# Tests for ChatAWSBedrock
# ===========================================================================
class TestChatAWSBedrock:
    """Tests for aws/chat_bedrock.py coverage."""

    def _make_chat(self, **kwargs):
        from openbrowser.llm.aws.chat_bedrock import ChatAWSBedrock

        defaults = {"model": "anthropic.claude-3-5-sonnet-20240620-v1:0"}
        defaults.update(kwargs)
        return ChatAWSBedrock(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "aws_bedrock"

    def test_name(self):
        chat = self._make_chat()
        assert chat.name == "anthropic.claude-3-5-sonnet-20240620-v1:0"

    def test_get_inference_config(self):
        chat = self._make_chat(
            max_tokens=1024, temperature=0.5, top_p=0.9,
            stop_sequences=["END"], seed=42,
        )
        config = chat._get_inference_config()
        assert config["maxTokens"] == 1024
        assert config["temperature"] == 0.5
        assert config["topP"] == 0.9
        assert config["stopSequences"] == ["END"]
        assert config["seed"] == 42

    def test_get_inference_config_empty(self):
        chat = self._make_chat(max_tokens=None, temperature=None, top_p=None, seed=None)
        config = chat._get_inference_config()
        assert config == {}

    def test_format_tools_for_request(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str
            count: int

        chat = self._make_chat()
        tools = chat._format_tools_for_request(MyOutput)
        assert len(tools) == 1
        assert tools[0]["toolSpec"]["name"] == "extract_myoutput"

    def test_get_usage_with_data(self):
        chat = self._make_chat()
        response = {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}
        usage = chat._get_usage(response)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_get_usage_no_data(self):
        chat = self._make_chat()
        response = {}
        usage = chat._get_usage(response)
        assert usage is None

    def test_get_client_with_session(self):
        chat = self._make_chat()
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        chat.session = mock_session

        with patch("openbrowser.llm.aws.chat_bedrock.ChatAWSBedrock._get_client") as mock:
            mock.return_value = MagicMock()
            client = chat._get_client()
            assert client is not None

    def test_get_client_with_credentials(self):
        chat = self._make_chat(
            aws_access_key_id="key", aws_secret_access_key="secret",
            aws_region="us-east-1",
        )
        with patch("boto3.client") as mock_client:
            mock_client.return_value = MagicMock()
            client = chat._get_client()
            assert client is not None

    def test_get_client_sso_auth(self):
        chat = self._make_chat(aws_sso_auth=True, aws_region="us-east-1")
        with patch("boto3.client") as mock_client:
            mock_client.return_value = MagicMock()
            client = chat._get_client()
            assert client is not None

    def test_get_client_no_credentials_raises(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        with patch.dict("os.environ", {}, clear=True):
            # Remove any env vars
            with patch("os.getenv", return_value=None):
                with pytest.raises(ModelProviderError, match="AWS credentials not found"):
                    chat._get_client()

    @pytest.mark.asyncio
    async def test_ainvoke_text_response(self):
        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "hello"

    @pytest.mark.asyncio
    async def test_ainvoke_text_no_output(self):
        """Test when response has no output."""
        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == ""

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t1", "name": "extract", "input": {"field": "val"}}}
            ]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_json_string_input(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t1", "name": "extract", "input": '{"field": "val"}'}}
            ]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            # First validation fails (string input), then JSON parse succeeds
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert result.completion.field == "val"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_validation_error(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: int

        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t1", "name": "extract", "input": {"wrong": "val"}}}
            ]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Failed to validate"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_no_tool_use(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "no tool"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Expected structured output"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_no_output_raises(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="No valid content"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_client_error_throttling(self):
        from openbrowser.llm.exceptions import ModelRateLimitError

        chat = self._make_chat()
        mock_client = MagicMock()

        # Create a ClientError-like exception
        error = MagicMock()
        error.response = {"Error": {"Code": "ThrottlingException", "Message": "throttled"}}

        # We need to create an actual exception class for botocore
        boto_mod = pytest.importorskip("botocore")
        from botocore.exceptions import ClientError

        client_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "throttled"}},
            "converse",
        )
        mock_client.converse.side_effect = client_error

        with patch.object(chat, "_get_client", return_value=mock_client):
            with pytest.raises(ModelRateLimitError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_client_error_other(self):
        from openbrowser.llm.exceptions import ModelProviderError

        boto_mod = pytest.importorskip("botocore")
        from botocore.exceptions import ClientError

        chat = self._make_chat()
        mock_client = MagicMock()
        client_error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad input"}},
            "converse",
        )
        mock_client.converse.side_effect = client_error

        with patch.object(chat, "_get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="bad input"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_generic_exception(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.side_effect = RuntimeError("boom")

        with patch.object(chat, "_get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_with_request_params(self):
        chat = self._make_chat(request_params={"extra": "param"})
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])
        assert result.completion == "hello"

    @pytest.mark.asyncio
    async def test_ainvoke_text_with_system_message(self):
        """Cover line 192: system message is added to request body."""
        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            result = await chat.ainvoke([
                SystemMessage(content="You are a helper"),
                UserMessage(content="hi"),
            ])
        assert result.completion == "hello"
        # Verify system message was passed in the call
        call_kwargs = mock_client.converse.call_args
        assert "system" in call_kwargs.kwargs or any(
            "system" in str(a) for a in call_kwargs.args
        )

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_string_tool_input_invalid_json(self):
        """Cover lines 255-256: tool_input is a string with invalid JSON."""
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t1", "name": "extract", "input": "not valid json at all"}}
            ]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }

        with patch.object(chat, "_get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Failed to validate"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)


# ===========================================================================
# Tests for ChatAnthropicBedrock
# ===========================================================================
class TestChatAnthropicBedrock:
    """Tests for aws/chat_anthropic.py coverage."""

    def _make_chat(self, **kwargs):
        from openbrowser.llm.aws.chat_anthropic import ChatAnthropicBedrock

        defaults = {"aws_access_key": "test-key", "aws_secret_key": "test-secret", "aws_region": "us-east-1"}
        defaults.update(kwargs)
        return ChatAnthropicBedrock(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "anthropic_bedrock"

    def test_name(self):
        chat = self._make_chat()
        assert "anthropic" in chat.name

    def test_get_client_params_with_credentials(self):
        chat = self._make_chat()
        params = chat._get_client_params()
        assert params["aws_access_key"] == "test-key"
        assert params["aws_secret_key"] == "test-secret"
        assert params["aws_region"] == "us-east-1"

    def test_get_client_params_with_session(self):
        chat = self._make_chat()
        mock_session = MagicMock()
        mock_creds = MagicMock()
        mock_creds.access_key = "session-key"
        mock_creds.secret_key = "session-secret"
        mock_creds.token = "session-token"
        mock_session.get_credentials.return_value = mock_creds
        mock_session.region_name = "eu-west-1"
        chat.session = mock_session

        params = chat._get_client_params()
        assert params["aws_access_key"] == "session-key"
        assert params["aws_region"] == "eu-west-1"

    def test_get_client_params_optional(self):
        chat = self._make_chat(
            max_retries=5,
            default_headers={"X-Custom": "val"},
            default_query={"q": "v"},
        )
        params = chat._get_client_params()
        assert params["max_retries"] == 5
        assert params["default_headers"] == {"X-Custom": "val"}
        assert params["default_query"] == {"q": "v"}

    def test_get_client_params_for_invoke(self):
        chat = self._make_chat(
            temperature=0.5, max_tokens=2048, top_p=0.9, top_k=40,
            seed=42, stop_sequences=["END"],
        )
        params = chat._get_client_params_for_invoke()
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 2048
        assert params["top_p"] == 0.9
        assert params["top_k"] == 40
        assert params["seed"] == 42
        assert params["stop_sequences"] == ["END"]

    def test_get_client_params_for_invoke_defaults(self):
        chat = self._make_chat()
        params = chat._get_client_params_for_invoke()
        assert "max_tokens" in params

    def test_get_client(self):
        chat = self._make_chat()
        client = chat.get_client()
        from anthropic import AsyncAnthropicBedrock

        assert isinstance(client, AsyncAnthropicBedrock)

    def test_get_usage(self):
        chat = self._make_chat()
        usage_mock = MagicMock()
        usage_mock.input_tokens = 10
        usage_mock.output_tokens = 5
        usage_mock.cache_read_input_tokens = 2
        usage_mock.cache_creation_input_tokens = 1

        resp = MagicMock()
        resp.usage = usage_mock
        usage = chat._get_usage(resp)
        assert usage.prompt_tokens == 12  # 10 + 2
        assert usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_ainvoke_text_response(self):
        from anthropic.types import Message
        from anthropic.types.text_block import TextBlock

        chat = self._make_chat()
        mock_client = AsyncMock()

        resp = MagicMock(spec=Message)
        resp.content = [TextBlock(text="response", type="text")]
        usage_mock = MagicMock()
        usage_mock.input_tokens = 10
        usage_mock.output_tokens = 5
        usage_mock.cache_read_input_tokens = 0
        usage_mock.cache_creation_input_tokens = 0
        resp.usage = usage_mock

        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "response"

    @pytest.mark.asyncio
    async def test_ainvoke_text_non_textblock(self):
        from anthropic.types import Message

        chat = self._make_chat()
        mock_client = AsyncMock()

        non_text = MagicMock()
        non_text.__str__ = lambda s: "non-text"
        resp = MagicMock(spec=Message)
        resp.content = [non_text]
        usage_mock = MagicMock()
        usage_mock.input_tokens = 10
        usage_mock.output_tokens = 5
        usage_mock.cache_read_input_tokens = 0
        usage_mock.cache_creation_input_tokens = 0
        resp.usage = usage_mock
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])
        assert "non-text" in result.completion

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output(self):
        from pydantic import BaseModel as PydanticBaseModel
        from anthropic.types import Message

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"field": "value"}

        resp = MagicMock(spec=Message)
        resp.content = [tool_block]
        usage_mock = MagicMock()
        usage_mock.input_tokens = 10
        usage_mock.output_tokens = 5
        usage_mock.cache_read_input_tokens = 0
        usage_mock.cache_creation_input_tokens = 0
        resp.usage = usage_mock
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_json_string_input(self):
        from pydantic import BaseModel as PydanticBaseModel
        from anthropic.types import Message

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = '{"field": "json_val"}'

        resp = MagicMock(spec=Message)
        resp.content = [tool_block]
        usage_mock = MagicMock()
        usage_mock.input_tokens = 10
        usage_mock.output_tokens = 5
        usage_mock.cache_read_input_tokens = 0
        usage_mock.cache_creation_input_tokens = 0
        resp.usage = usage_mock
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert result.completion.field == "json_val"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_no_tool(self):
        from pydantic import BaseModel as PydanticBaseModel
        from anthropic.types import Message
        from anthropic.types.text_block import TextBlock
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()

        resp = MagicMock(spec=Message)
        resp.content = [TextBlock(text="no tool", type="text")]
        usage_mock = MagicMock()
        usage_mock.input_tokens = 10
        usage_mock.output_tokens = 5
        usage_mock.cache_read_input_tokens = 0
        usage_mock.cache_creation_input_tokens = 0
        resp.usage = usage_mock
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

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
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_rate_limit_error(self):
        from anthropic import RateLimitError
        from openbrowser.llm.exceptions import ModelRateLimitError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = RateLimitError(message="rate limited", response=mock_response, body=None)
        mock_client.messages.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelRateLimitError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_api_status_error(self):
        from anthropic import APIStatusError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        error = APIStatusError(message="server error", response=mock_response, body=None)
        mock_client.messages.create = AsyncMock(side_effect=error)

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_generic_exception(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="hi")])
