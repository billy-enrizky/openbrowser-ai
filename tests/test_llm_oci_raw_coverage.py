"""Tests for OCI Raw LLM provider modules - chat.py and serializer.py.

Covers:
  src/openbrowser/llm/oci_raw/chat.py
  src/openbrowser/llm/oci_raw/serializer.py
"""

import json
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock the entire OCI SDK before importing our modules.
# This lets the tests run even when `oci` is not installed.
# ---------------------------------------------------------------------------
_mock_oci = MagicMock()
_mock_oci_genai = MagicMock()
_mock_oci_genai_models = MagicMock()

# Create sentinel classes for OCI SDK models so isinstance/type checks work
_MockMessage = type("Message", (), {"__init__": lambda self: None})
_MockTextContent = type("TextContent", (), {"__init__": lambda self: None})
_MockImageContent = type("ImageContent", (), {"__init__": lambda self, **kw: None})
_MockImageUrl = type("ImageUrl", (), {"__init__": lambda self, **kw: None})

_mock_oci_genai_models.Message = _MockMessage
_mock_oci_genai_models.TextContent = _MockTextContent
_mock_oci_genai_models.ImageContent = _MockImageContent
_mock_oci_genai_models.ImageUrl = _MockImageUrl
_mock_oci_genai_models.BaseChatRequest = MagicMock()
_mock_oci_genai_models.ChatDetails = MagicMock()
_mock_oci_genai_models.CohereChatRequest = MagicMock()
_mock_oci_genai_models.GenericChatRequest = MagicMock()
_mock_oci_genai_models.OnDemandServingMode = MagicMock()

# Install OCI mocks permanently in sys.modules so all imports share the same
# module objects. Using patch.dict context manager would undo the injection and
# cause separate module identities for classes like ModelProviderError.
for _key, _val in {
    "oci": _mock_oci,
    "oci.generative_ai_inference": _mock_oci_genai,
    "oci.generative_ai_inference.models": _mock_oci_genai_models,
    "oci.config": _mock_oci.config,
    "oci.retry": _mock_oci.retry,
    "oci.auth": _mock_oci.auth,
    "oci.auth.signers": _mock_oci.auth.signers,
}.items():
    sys.modules.setdefault(_key, _val)

from openbrowser.llm.oci_raw.chat import ChatOCIRaw  # noqa: E402
from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer  # noqa: E402

from openbrowser.llm.exceptions import ModelProviderError, ModelRateLimitError  # noqa: E402
from openbrowser.llm.messages import (  # noqa: E402
    AssistantMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from openbrowser.llm.views import ChatInvokeCompletion, ChatInvokeUsage  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chat_oci_raw(**overrides):
    """Create a ChatOCIRaw instance with sensible defaults."""
    defaults = dict(
        model_id="ocid1.generativeaimodel.oc1.us-chicago-1.test-model",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..test-compartment",
        provider="meta",
    )
    defaults.update(overrides)
    return ChatOCIRaw(**defaults)


def _make_oci_response(text="hello", has_usage=True, response_type="generic"):
    """Build a mock OCI response object.

    Args:
        text: The response text.
        has_usage: Whether to include usage info.
        response_type: 'generic' for choices-based, 'cohere' for text-based.
    """
    response = MagicMock()

    if response_type == "cohere":
        chat_response = MagicMock()
        chat_response.text = text
        # Remove choices attr so hasattr returns False
        del chat_response.choices
    else:
        # Generic response with choices
        text_part = MagicMock()
        text_part.text = text

        message = MagicMock()
        message.content = [text_part]

        choice = MagicMock()
        choice.message = message

        chat_response = MagicMock()
        # Remove text attr so hasattr returns False for cohere check
        del chat_response.text
        chat_response.choices = [choice]

    if has_usage:
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        chat_response.usage = usage
    else:
        del chat_response.usage

    response.data = MagicMock()
    response.data.chat_response = chat_response
    return response


# ===========================================================================
# Tests for OCIRawMessageSerializer (serializer.py)
# ===========================================================================
class TestOCIRawMessageSerializer:
    """Tests for oci_raw/serializer.py coverage."""

    # ---- _is_base64_image ----
    def test_is_base64_image_true(self):
        assert OCIRawMessageSerializer._is_base64_image("data:image/png;base64,abc123") is True

    def test_is_base64_image_false(self):
        assert OCIRawMessageSerializer._is_base64_image("https://example.com/img.png") is False

    # ---- _parse_base64_url ----
    def test_parse_base64_url_valid(self):
        url = "data:image/png;base64,abc123XYZ"
        result = OCIRawMessageSerializer._parse_base64_url(url)
        assert result == "abc123XYZ"

    def test_parse_base64_url_not_base64(self):
        with pytest.raises(ValueError, match="Not a base64 image URL"):
            OCIRawMessageSerializer._parse_base64_url("https://example.com/img.png")

    def test_parse_base64_url_invalid_format_no_comma(self):
        with pytest.raises(ValueError, match="Invalid base64 image URL format"):
            OCIRawMessageSerializer._parse_base64_url("data:image/png;base64")

    # ---- _create_image_content ----
    @patch("openbrowser.llm.oci_raw.serializer.ImageUrl")
    @patch("openbrowser.llm.oci_raw.serializer.ImageContent")
    def test_create_image_content_base64(self, mock_image_content_cls, mock_image_url_cls):
        part = ContentPartImageParam(
            image_url=ImageURL(url="data:image/png;base64,abc123")
        )
        mock_image_url_inst = MagicMock()
        mock_image_url_cls.return_value = mock_image_url_inst
        mock_image_content_inst = MagicMock()
        mock_image_content_cls.return_value = mock_image_content_inst

        result = OCIRawMessageSerializer._create_image_content(part)
        mock_image_url_cls.assert_called_once_with(url="data:image/png;base64,abc123")
        mock_image_content_cls.assert_called_once_with(image_url=mock_image_url_inst)
        assert result is mock_image_content_inst

    @patch("openbrowser.llm.oci_raw.serializer.ImageUrl")
    @patch("openbrowser.llm.oci_raw.serializer.ImageContent")
    def test_create_image_content_regular_url(self, mock_image_content_cls, mock_image_url_cls):
        part = ContentPartImageParam(
            image_url=ImageURL(url="https://example.com/img.png")
        )
        mock_image_url_inst = MagicMock()
        mock_image_url_cls.return_value = mock_image_url_inst
        mock_image_content_inst = MagicMock()
        mock_image_content_cls.return_value = mock_image_content_inst

        result = OCIRawMessageSerializer._create_image_content(part)
        mock_image_url_cls.assert_called_once_with(url="https://example.com/img.png")
        assert result is mock_image_content_inst

    # ---- serialize_messages ----
    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_user_message_string(self, mock_msg_cls, mock_tc_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        msgs = [UserMessage(content="hello")]
        result = OCIRawMessageSerializer.serialize_messages(msgs)

        assert mock_msg.role == "USER"
        mock_tc_cls.assert_called()
        assert mock_tc.text == "hello"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_user_message_list_text(self, mock_msg_cls, mock_tc_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        msgs = [UserMessage(content=[ContentPartTextParam(text="hi")])]
        result = OCIRawMessageSerializer.serialize_messages(msgs)

        assert mock_msg.role == "USER"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.ImageUrl")
    @patch("openbrowser.llm.oci_raw.serializer.ImageContent")
    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_user_message_list_image(self, mock_msg_cls, mock_tc_cls, mock_ic_cls, mock_iu_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg

        msgs = [
            UserMessage(
                content=[
                    ContentPartImageParam(
                        image_url=ImageURL(url="https://example.com/img.png")
                    )
                ]
            )
        ]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert mock_msg.role == "USER"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_system_message_string(self, mock_msg_cls, mock_tc_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        msgs = [SystemMessage(content="You are a helper")]
        result = OCIRawMessageSerializer.serialize_messages(msgs)

        assert mock_msg.role == "SYSTEM"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_system_message_list_text(self, mock_msg_cls, mock_tc_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        msgs = [SystemMessage(content=[ContentPartTextParam(text="system instruction")])]
        result = OCIRawMessageSerializer.serialize_messages(msgs)

        assert mock_msg.role == "SYSTEM"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.ImageUrl")
    @patch("openbrowser.llm.oci_raw.serializer.ImageContent")
    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_system_message_list_image(self, mock_msg_cls, mock_tc_cls, mock_ic_cls, mock_iu_cls):
        """System messages with image_url content parts (line 113-116 coverage)."""
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg

        # Use model_construct to bypass Pydantic validation (SystemMessage normally
        # only accepts ContentPartTextParam, but the serializer handles image_url).
        sys_msg = SystemMessage.model_construct(role="system", content="placeholder", cache=False)
        image_part = MagicMock()
        image_part.type = "image_url"
        image_part.image_url = MagicMock()
        image_part.image_url.url = "https://example.com/img.png"
        sys_msg.content = [image_part]

        result = OCIRawMessageSerializer.serialize_messages([sys_msg])
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_assistant_message_string(self, mock_msg_cls, mock_tc_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        msgs = [AssistantMessage(content="Sure, here you go")]
        result = OCIRawMessageSerializer.serialize_messages(msgs)

        assert mock_msg.role == "ASSISTANT"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_assistant_message_list_text(self, mock_msg_cls, mock_tc_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        msgs = [AssistantMessage(content=[ContentPartTextParam(text="response text")])]
        result = OCIRawMessageSerializer.serialize_messages(msgs)

        assert mock_msg.role == "ASSISTANT"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_assistant_message_list_refusal(self, mock_msg_cls, mock_tc_cls):
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        msgs = [AssistantMessage(content=[ContentPartRefusalParam(refusal="I cannot do that")])]
        result = OCIRawMessageSerializer.serialize_messages(msgs)

        assert mock_msg.role == "ASSISTANT"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.ImageUrl")
    @patch("openbrowser.llm.oci_raw.serializer.ImageContent")
    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_assistant_message_list_image(self, mock_msg_cls, mock_tc_cls, mock_ic_cls, mock_iu_cls):
        """Assistant messages with image_url content parts (line 135-139 coverage)."""
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg

        # AssistantMessage content type doesn't include ContentPartImageParam,
        # but the serializer handles it. Use model_construct to bypass validation.
        asst_msg = AssistantMessage.model_construct(
            role="assistant", content="placeholder", cache=False, refusal=None, tool_calls=[]
        )
        image_part = MagicMock()
        image_part.type = "image_url"
        image_part.image_url = MagicMock()
        image_part.image_url.url = "https://example.com/img.png"
        asst_msg.content = [image_part]

        result = OCIRawMessageSerializer.serialize_messages([asst_msg])
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.TextContent")
    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_unknown_message_type_fallback(self, mock_msg_cls, mock_tc_cls):
        """Unknown message type falls back to USER role (line 146-151)."""
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg_cls.return_value = mock_msg
        mock_tc = MagicMock()
        mock_tc_cls.return_value = mock_tc

        # Use a plain class so isinstance checks against message types fail
        class _UnknownMsg:
            def __str__(self):
                return "unknown content"

        result = OCIRawMessageSerializer.serialize_messages([_UnknownMsg()])
        assert mock_msg.role == "USER"
        assert len(result) == 1

    @patch("openbrowser.llm.oci_raw.serializer.Message")
    def test_serialize_message_with_no_content_skipped(self, mock_msg_cls):
        """Messages that result in no content should be skipped (line 154-155)."""
        mock_msg = MagicMock(spec=[])  # No content attr -> hasattr returns False
        mock_msg_cls.return_value = mock_msg

        # UserMessage with empty list content -> no contents list -> message skipped
        user_msg = UserMessage.model_construct(role="user", content=[], cache=False)
        result = OCIRawMessageSerializer.serialize_messages([user_msg])
        assert len(result) == 0

    # ---- serialize_messages_for_cohere ----
    def test_cohere_user_message_string(self):
        msgs = [UserMessage(content="Hello")]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "User: Hello"

    def test_cohere_user_message_list_text(self):
        msgs = [UserMessage(content=[ContentPartTextParam(text="Hi there")])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "User: Hi there"

    def test_cohere_user_message_list_image_base64(self):
        msgs = [
            UserMessage(
                content=[
                    ContentPartImageParam(
                        image_url=ImageURL(url="data:image/png;base64,abc123")
                    )
                ]
            )
        ]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "User: [Image: base64_data]"

    def test_cohere_user_message_list_image_external(self):
        msgs = [
            UserMessage(
                content=[
                    ContentPartImageParam(
                        image_url=ImageURL(url="https://example.com/img.png")
                    )
                ]
            )
        ]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "User: [Image: external_url]"

    def test_cohere_system_message_string(self):
        msgs = [SystemMessage(content="Be helpful")]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "System: Be helpful"

    def test_cohere_system_message_list_text(self):
        msgs = [SystemMessage(content=[ContentPartTextParam(text="system instruction")])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "System: system instruction"

    def test_cohere_assistant_message_string(self):
        msgs = [AssistantMessage(content="response")]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "Assistant: response"

    def test_cohere_assistant_message_list_text(self):
        msgs = [AssistantMessage(content=[ContentPartTextParam(text="answer")])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "Assistant: answer"

    def test_cohere_assistant_message_list_refusal(self):
        msgs = [AssistantMessage(content=[ContentPartRefusalParam(refusal="I refuse")])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "Assistant: [Refusal] I refuse"

    def test_cohere_unknown_message_fallback(self):
        """Unknown message type falls back to 'User: ...' (line 225-227)."""

        class _UnknownMsg:
            def __str__(self):
                return "fallback content"

        result = OCIRawMessageSerializer.serialize_messages_for_cohere([_UnknownMsg()])
        assert result == "User: fallback content"

    def test_cohere_multiple_messages(self):
        msgs = [
            SystemMessage(content="Be helpful"),
            UserMessage(content="What is 2+2?"),
            AssistantMessage(content="4"),
        ]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "System: Be helpful" in result
        assert "User: What is 2+2?" in result
        assert "Assistant: 4" in result
        # Messages separated by double newline
        parts = result.split("\n\n")
        assert len(parts) == 3

    def test_cohere_user_message_list_mixed_text_and_image(self):
        msgs = [
            UserMessage(
                content=[
                    ContentPartTextParam(text="Look at this"),
                    ContentPartImageParam(
                        image_url=ImageURL(url="https://example.com/img.png")
                    ),
                ]
            )
        ]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert result == "User: Look at this [Image: external_url]"


# ===========================================================================
# Tests for ChatOCIRaw (chat.py)
# ===========================================================================
class TestChatOCIRawProperties:
    """Tests for ChatOCIRaw properties and static config."""

    def test_provider_name(self):
        model = _make_chat_oci_raw()
        assert model.provider_name == "oci-raw"

    def test_model_property(self):
        model = _make_chat_oci_raw(model_id="test-model-id")
        assert model.model == "test-model-id"

    def test_name_short_model_id(self):
        model = _make_chat_oci_raw(model_id="short-model")
        assert model.name == "short-model"

    def test_name_long_model_id_with_parts(self):
        # Model ID longer than 90 chars with 4+ parts
        long_id = "ocid1.generativeaimodel.oc1.us-chicago-1." + "x" * 60
        model = _make_chat_oci_raw(model_id=long_id)
        assert model.name == "oci-meta-us-chicago-1"

    def test_name_long_model_id_few_parts(self):
        # Model ID longer than 90 chars but fewer than 4 parts
        long_id = "verylongprefix." + "x" * 90
        model = _make_chat_oci_raw(model_id=long_id, provider="meta")
        assert model.name == "oci-meta-model"

    def test_model_name_short(self):
        model = _make_chat_oci_raw(model_id="short-model")
        assert model.model_name == "short-model"

    def test_model_name_long_with_parts(self):
        long_id = "ocid1.generativeaimodel.oc1.us-chicago-1." + "x" * 60
        model = _make_chat_oci_raw(model_id=long_id)
        assert model.model_name == "oci-meta-us-chicago-1"

    def test_model_name_long_few_parts(self):
        long_id = "verylongprefix." + "x" * 90
        model = _make_chat_oci_raw(model_id=long_id, provider="cohere")
        assert model.model_name == "oci-cohere-model"


class TestChatOCIRawUsesCohere:
    """Tests for _uses_cohere_format."""

    def test_cohere_provider(self):
        model = _make_chat_oci_raw(provider="cohere")
        assert model._uses_cohere_format() is True

    def test_cohere_provider_uppercase(self):
        model = _make_chat_oci_raw(provider="Cohere")
        assert model._uses_cohere_format() is True

    def test_meta_provider(self):
        model = _make_chat_oci_raw(provider="meta")
        assert model._uses_cohere_format() is False

    def test_xai_provider(self):
        model = _make_chat_oci_raw(provider="xai")
        assert model._uses_cohere_format() is False


class TestChatOCIRawSupportedParams:
    """Tests for _get_supported_parameters with all provider branches."""

    def test_meta_params(self):
        model = _make_chat_oci_raw(provider="meta")
        params = model._get_supported_parameters()
        assert params["temperature"] is True
        assert params["frequency_penalty"] is True
        assert params["presence_penalty"] is True
        assert params["top_k"] is False

    def test_cohere_params(self):
        model = _make_chat_oci_raw(provider="cohere")
        params = model._get_supported_parameters()
        assert params["temperature"] is True
        assert params["frequency_penalty"] is True
        assert params["presence_penalty"] is False
        assert params["top_k"] is True

    def test_xai_params(self):
        model = _make_chat_oci_raw(provider="xai")
        params = model._get_supported_parameters()
        assert params["temperature"] is True
        assert params["frequency_penalty"] is False
        assert params["presence_penalty"] is False
        assert params["top_k"] is True

    def test_unknown_provider_params(self):
        model = _make_chat_oci_raw(provider="custom")
        params = model._get_supported_parameters()
        # All should be True for unknown providers
        assert all(v is True for v in params.values())


class TestChatOCIRawGetClient:
    """Tests for _get_oci_client with all auth types."""

    @patch("openbrowser.llm.oci_raw.chat.oci")
    @patch("openbrowser.llm.oci_raw.chat.GenerativeAiInferenceClient")
    def test_api_key_auth(self, mock_client_cls, mock_oci):
        model = _make_chat_oci_raw(auth_type="API_KEY", auth_profile="DEFAULT")
        mock_config = MagicMock()
        mock_oci.config.from_file.return_value = mock_config
        mock_oci.retry.NoneRetryStrategy.return_value = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst

        result = model._get_oci_client()
        mock_oci.config.from_file.assert_called_once_with("~/.oci/config", "DEFAULT")
        assert result is mock_client_inst

    @patch("openbrowser.llm.oci_raw.chat.oci")
    @patch("openbrowser.llm.oci_raw.chat.GenerativeAiInferenceClient")
    def test_instance_principal_auth(self, mock_client_cls, mock_oci):
        model = _make_chat_oci_raw(auth_type="INSTANCE_PRINCIPAL")
        mock_signer = MagicMock()
        mock_oci.auth.signers.InstancePrincipalsSecurityTokenSigner.return_value = mock_signer
        mock_oci.retry.NoneRetryStrategy.return_value = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst

        result = model._get_oci_client()
        mock_oci.auth.signers.InstancePrincipalsSecurityTokenSigner.assert_called_once()
        assert result is mock_client_inst

    @patch("openbrowser.llm.oci_raw.chat.oci")
    @patch("openbrowser.llm.oci_raw.chat.GenerativeAiInferenceClient")
    def test_resource_principal_auth(self, mock_client_cls, mock_oci):
        model = _make_chat_oci_raw(auth_type="RESOURCE_PRINCIPAL")
        mock_signer = MagicMock()
        mock_oci.auth.signers.get_resource_principals_signer.return_value = mock_signer
        mock_oci.retry.NoneRetryStrategy.return_value = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst

        result = model._get_oci_client()
        mock_oci.auth.signers.get_resource_principals_signer.assert_called_once()
        assert result is mock_client_inst

    @patch("openbrowser.llm.oci_raw.chat.oci")
    @patch("openbrowser.llm.oci_raw.chat.GenerativeAiInferenceClient")
    def test_unknown_auth_fallback_to_api_key(self, mock_client_cls, mock_oci):
        model = _make_chat_oci_raw(auth_type="UNKNOWN_TYPE")
        mock_config = MagicMock()
        mock_oci.config.from_file.return_value = mock_config
        mock_oci.retry.NoneRetryStrategy.return_value = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst

        result = model._get_oci_client()
        mock_oci.config.from_file.assert_called_once_with("~/.oci/config", "DEFAULT")
        assert result is mock_client_inst

    @patch("openbrowser.llm.oci_raw.chat.oci")
    @patch("openbrowser.llm.oci_raw.chat.GenerativeAiInferenceClient")
    def test_client_cached(self, mock_client_cls, mock_oci):
        """Client should be cached after first creation."""
        model = _make_chat_oci_raw(auth_type="API_KEY")
        mock_config = MagicMock()
        mock_oci.config.from_file.return_value = mock_config
        mock_oci.retry.NoneRetryStrategy.return_value = MagicMock()
        mock_client_inst = MagicMock()
        mock_client_cls.return_value = mock_client_inst

        result1 = model._get_oci_client()
        result2 = model._get_oci_client()
        # Should only create client once
        assert mock_client_cls.call_count == 1
        assert result1 is result2


class TestChatOCIRawExtractUsage:
    """Tests for _extract_usage."""

    def test_extract_usage_success(self):
        model = _make_chat_oci_raw()
        response = _make_oci_response(has_usage=True)
        result = model._extract_usage(response)

        assert result is not None
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15

    def test_extract_usage_no_data(self):
        model = _make_chat_oci_raw()
        response = MagicMock(spec=[])  # No data attr
        result = model._extract_usage(response)
        assert result is None

    def test_extract_usage_no_chat_response(self):
        model = _make_chat_oci_raw()
        response = MagicMock()
        response.data = MagicMock(spec=[])  # No chat_response attr
        result = model._extract_usage(response)
        assert result is None

    def test_extract_usage_no_usage_attr(self):
        model = _make_chat_oci_raw()
        response = _make_oci_response(has_usage=False)
        result = model._extract_usage(response)
        assert result is None

    def test_extract_usage_exception(self):
        model = _make_chat_oci_raw()
        response = MagicMock()
        response.data = MagicMock()
        response.data.chat_response = MagicMock()
        response.data.chat_response.usage = MagicMock()
        # Force an exception by making prompt_tokens raise
        type(response.data.chat_response.usage).prompt_tokens = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("oops"))
        )
        result = model._extract_usage(response)
        assert result is None


class TestChatOCIRawExtractContent:
    """Tests for _extract_content."""

    def test_extract_content_generic_response(self):
        model = _make_chat_oci_raw()
        response = _make_oci_response(text="Hello world", response_type="generic")
        result = model._extract_content(response)
        assert result == "Hello world"

    def test_extract_content_cohere_response(self):
        model = _make_chat_oci_raw(provider="cohere")
        response = _make_oci_response(text="Cohere answer", response_type="cohere")
        result = model._extract_content(response)
        assert result == "Cohere answer"

    def test_extract_content_cohere_none_text(self):
        model = _make_chat_oci_raw(provider="cohere")
        response = MagicMock()
        chat_response = MagicMock()
        chat_response.text = None
        del chat_response.choices
        response.data.chat_response = chat_response
        result = model._extract_content(response)
        assert result == ""

    def test_extract_content_no_data_attribute(self):
        model = _make_chat_oci_raw()
        response = MagicMock(spec=[])  # No data attribute
        with pytest.raises(ModelProviderError, match="Failed to extract content"):
            model._extract_content(response)

    def test_extract_content_unsupported_format(self):
        model = _make_chat_oci_raw()
        response = MagicMock()
        chat_response = MagicMock(spec=[])  # No text, no choices
        response.data.chat_response = chat_response
        with pytest.raises(ModelProviderError, match="Failed to extract content"):
            model._extract_content(response)

    def test_extract_content_generic_no_text_parts(self):
        """Generic response where parts don't have text attr."""
        model = _make_chat_oci_raw()
        response = MagicMock()
        part = MagicMock(spec=[])  # No text attribute
        message = MagicMock()
        message.content = [part]
        choice = MagicMock()
        choice.message = message
        chat_response = MagicMock()
        del chat_response.text  # Not cohere
        chat_response.choices = [choice]
        response.data.chat_response = chat_response
        result = model._extract_content(response)
        assert result == ""

    def test_extract_content_generic_multiple_text_parts(self):
        """Generic response with multiple text parts joined by newline."""
        model = _make_chat_oci_raw()
        response = MagicMock()
        part1 = MagicMock()
        part1.text = "Line 1"
        part2 = MagicMock()
        part2.text = "Line 2"
        message = MagicMock()
        message.content = [part1, part2]
        choice = MagicMock()
        choice.message = message
        chat_response = MagicMock()
        del chat_response.text
        chat_response.choices = [choice]
        response.data.chat_response = chat_response
        result = model._extract_content(response)
        assert result == "Line 1\nLine 2"

    def test_extract_content_generic_empty_choices(self):
        """Generic response with empty choices list."""
        model = _make_chat_oci_raw()
        response = MagicMock()
        chat_response = MagicMock()
        del chat_response.text
        chat_response.choices = []
        response.data.chat_response = chat_response
        with pytest.raises(ModelProviderError, match="Failed to extract content"):
            model._extract_content(response)


class TestChatOCIRawMakeRequest:
    """Tests for _make_request."""

    @pytest.mark.asyncio
    @patch("openbrowser.llm.oci_raw.chat.OnDemandServingMode")
    @patch("openbrowser.llm.oci_raw.chat.ChatDetails")
    @patch("openbrowser.llm.oci_raw.chat.GenericChatRequest")
    @patch("openbrowser.llm.oci_raw.chat.OCIRawMessageSerializer")
    async def test_make_request_meta_provider(self, mock_serializer, mock_generic_cls, mock_details_cls, mock_serving_cls):
        model = _make_chat_oci_raw(provider="meta")
        mock_serializer.serialize_messages.return_value = [MagicMock()]

        mock_generic = MagicMock()
        mock_generic_cls.return_value = mock_generic
        mock_details = MagicMock()
        mock_details_cls.return_value = mock_details
        mock_serving = MagicMock()
        mock_serving_cls.return_value = mock_serving

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.return_value = mock_response

        with patch.object(model, "_get_oci_client", return_value=mock_client):
            result = await model._make_request([UserMessage(content="hi")])

        assert result is mock_response
        # Meta should set frequency_penalty and presence_penalty
        assert mock_generic.frequency_penalty is not None
        assert mock_generic.presence_penalty is not None

    @pytest.mark.asyncio
    @patch("openbrowser.llm.oci_raw.chat.OnDemandServingMode")
    @patch("openbrowser.llm.oci_raw.chat.ChatDetails")
    @patch("openbrowser.llm.oci_raw.chat.CohereChatRequest")
    @patch("openbrowser.llm.oci_raw.chat.OCIRawMessageSerializer")
    async def test_make_request_cohere_provider(self, mock_serializer, mock_cohere_cls, mock_details_cls, mock_serving_cls):
        model = _make_chat_oci_raw(provider="cohere")
        mock_serializer.serialize_messages_for_cohere.return_value = "User: hi"

        mock_cohere = MagicMock()
        mock_cohere_cls.return_value = mock_cohere
        mock_details = MagicMock()
        mock_details_cls.return_value = mock_details
        mock_serving = MagicMock()
        mock_serving_cls.return_value = mock_serving

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.return_value = mock_response

        with patch.object(model, "_get_oci_client", return_value=mock_client):
            result = await model._make_request([UserMessage(content="hi")])

        assert result is mock_response
        mock_serializer.serialize_messages_for_cohere.assert_called_once()

    @pytest.mark.asyncio
    @patch("openbrowser.llm.oci_raw.chat.OnDemandServingMode")
    @patch("openbrowser.llm.oci_raw.chat.ChatDetails")
    @patch("openbrowser.llm.oci_raw.chat.GenericChatRequest")
    @patch("openbrowser.llm.oci_raw.chat.OCIRawMessageSerializer")
    async def test_make_request_xai_provider(self, mock_serializer, mock_generic_cls, mock_details_cls, mock_serving_cls):
        model = _make_chat_oci_raw(provider="xai")
        mock_serializer.serialize_messages.return_value = [MagicMock()]

        mock_generic = MagicMock()
        mock_generic_cls.return_value = mock_generic
        mock_details = MagicMock()
        mock_details_cls.return_value = mock_details
        mock_serving = MagicMock()
        mock_serving_cls.return_value = mock_serving

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.return_value = mock_response

        with patch.object(model, "_get_oci_client", return_value=mock_client):
            result = await model._make_request([UserMessage(content="hi")])

        assert result is mock_response
        # xAI should set top_k
        assert mock_generic.top_k is not None

    @pytest.mark.asyncio
    @patch("openbrowser.llm.oci_raw.chat.OnDemandServingMode")
    @patch("openbrowser.llm.oci_raw.chat.ChatDetails")
    @patch("openbrowser.llm.oci_raw.chat.GenericChatRequest")
    @patch("openbrowser.llm.oci_raw.chat.OCIRawMessageSerializer")
    async def test_make_request_unknown_provider(self, mock_serializer, mock_generic_cls, mock_details_cls, mock_serving_cls):
        model = _make_chat_oci_raw(provider="custom_provider")
        mock_serializer.serialize_messages.return_value = [MagicMock()]

        mock_generic = MagicMock()
        mock_generic_cls.return_value = mock_generic
        mock_details = MagicMock()
        mock_details_cls.return_value = mock_details
        mock_serving = MagicMock()
        mock_serving_cls.return_value = mock_serving

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.return_value = mock_response

        with patch.object(model, "_get_oci_client", return_value=mock_client):
            result = await model._make_request([UserMessage(content="hi")])

        assert result is mock_response
        # Default provider should set frequency_penalty and presence_penalty
        assert mock_generic.frequency_penalty is not None
        assert mock_generic.presence_penalty is not None

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_error(self):
        model = _make_chat_oci_raw(provider="meta")
        exc = Exception("rate limited")
        exc.status = 429

        mock_client = MagicMock()
        mock_client.chat.side_effect = exc

        with (
            patch.object(model, "_get_oci_client", return_value=mock_client),
            patch("openbrowser.llm.oci_raw.chat.OCIRawMessageSerializer") as mock_ser,
            patch("openbrowser.llm.oci_raw.chat.GenericChatRequest"),
            patch("openbrowser.llm.oci_raw.chat.OnDemandServingMode"),
            patch("openbrowser.llm.oci_raw.chat.ChatDetails"),
        ):
            mock_ser.serialize_messages.return_value = [MagicMock()]
            with pytest.raises(ModelRateLimitError, match="Rate limit exceeded"):
                await model._make_request([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_make_request_generic_error(self):
        model = _make_chat_oci_raw(provider="meta")
        exc = Exception("server error")
        exc.status = 500

        mock_client = MagicMock()
        mock_client.chat.side_effect = exc

        with (
            patch.object(model, "_get_oci_client", return_value=mock_client),
            patch("openbrowser.llm.oci_raw.chat.OCIRawMessageSerializer") as mock_ser,
            patch("openbrowser.llm.oci_raw.chat.GenericChatRequest"),
            patch("openbrowser.llm.oci_raw.chat.OnDemandServingMode"),
            patch("openbrowser.llm.oci_raw.chat.ChatDetails"),
        ):
            mock_ser.serialize_messages.return_value = [MagicMock()]
            with pytest.raises(ModelProviderError, match="server error"):
                await model._make_request([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_make_request_error_no_status_attr(self):
        """Error without status attribute defaults to 500."""
        model = _make_chat_oci_raw(provider="meta")
        exc = RuntimeError("something broke")

        mock_client = MagicMock()
        mock_client.chat.side_effect = exc

        with (
            patch.object(model, "_get_oci_client", return_value=mock_client),
            patch("openbrowser.llm.oci_raw.chat.OCIRawMessageSerializer") as mock_ser,
            patch("openbrowser.llm.oci_raw.chat.GenericChatRequest"),
            patch("openbrowser.llm.oci_raw.chat.OnDemandServingMode"),
            patch("openbrowser.llm.oci_raw.chat.ChatDetails"),
        ):
            mock_ser.serialize_messages.return_value = [MagicMock()]
            with pytest.raises(ModelProviderError) as exc_info:
                await model._make_request([UserMessage(content="hi")])
            assert exc_info.value.status_code == 500


class TestChatOCIRawAinvoke:
    """Tests for ainvoke method."""

    @pytest.mark.asyncio
    async def test_ainvoke_string_response(self):
        model = _make_chat_oci_raw()
        mock_response = _make_oci_response(text="Hello!")

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            result = await model.ainvoke([UserMessage(content="Hi")])

        assert isinstance(result, ChatInvokeCompletion)
        assert result.completion == "Hello!"
        assert result.usage is not None

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            answer: str
            score: int

        model = _make_chat_oci_raw()
        json_text = json.dumps({"answer": "42", "score": 100})
        mock_response = _make_oci_response(text=json_text)

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            result = await model.ainvoke(
                [UserMessage(content="What is the answer?")],
                output_format=TestOutput,
            )

        assert isinstance(result.completion, TestOutput)
        assert result.completion.answer == "42"
        assert result.completion.score == 100

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_with_markdown_code_block(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        json_text = '```json\n{"value": "test"}\n```'
        mock_response = _make_oci_response(text=json_text)

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            result = await model.ainvoke(
                [UserMessage(content="test")],
                output_format=TestOutput,
            )

        assert result.completion.value == "test"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_with_plain_code_block(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        json_text = '```\n{"value": "test2"}\n```'
        mock_response = _make_oci_response(text=json_text)

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            result = await model.ainvoke(
                [UserMessage(content="test")],
                output_format=TestOutput,
            )

        assert result.completion.value == "test2"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_json_embedded_in_text(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        json_text = 'Here is the result: {"value": "embedded"} as requested.'
        mock_response = _make_oci_response(text=json_text)

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            result = await model.ainvoke(
                [UserMessage(content="test")],
                output_format=TestOutput,
            )

        assert result.completion.value == "embedded"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_invalid_json(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        mock_response = _make_oci_response(text="not json at all")

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(ModelProviderError, match="Failed to parse structured output"):
                await model.ainvoke(
                    [UserMessage(content="test")],
                    output_format=TestOutput,
                )

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_modifies_existing_system_message_str(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        json_text = json.dumps({"value": "test"})
        mock_response = _make_oci_response(text=json_text)

        msgs = [
            SystemMessage(content="Be helpful"),
            UserMessage(content="What?"),
        ]

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response) as mock_req:
            result = await model.ainvoke(msgs, output_format=TestOutput)

        assert result.completion.value == "test"
        # Check the modified messages were passed
        call_args = mock_req.call_args[0][0]
        assert "Be helpful" in call_args[0].content
        assert "JSON" in call_args[0].content

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_modifies_existing_system_message_list(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        json_text = json.dumps({"value": "test"})
        mock_response = _make_oci_response(text=json_text)

        # Create a SystemMessage with list content, then override to non-string
        sys_msg = SystemMessage.model_construct(
            role="system", content=["some", "list"], cache=False
        )

        msgs = [sys_msg, UserMessage(content="What?")]

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response) as mock_req:
            result = await model.ainvoke(msgs, output_format=TestOutput)

        assert result.completion.value == "test"
        # The list content should have been converted to string and appended
        call_args = mock_req.call_args[0][0]
        assert "JSON" in str(call_args[0].content)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_inserts_new_system_message(self):
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        json_text = json.dumps({"value": "test"})
        mock_response = _make_oci_response(text=json_text)

        # No system message present
        msgs = [UserMessage(content="What?")]

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response) as mock_req:
            result = await model.ainvoke(msgs, output_format=TestOutput)

        assert result.completion.value == "test"
        call_args = mock_req.call_args[0][0]
        # First message should be the newly inserted system message
        assert call_args[0].role == "system"
        assert "JSON" in call_args[0].content

    @pytest.mark.asyncio
    async def test_ainvoke_reraises_rate_limit_error(self):
        model = _make_chat_oci_raw()

        with patch.object(
            model, "_make_request", new_callable=AsyncMock,
            side_effect=ModelRateLimitError(message="rate limited", model="test"),
        ):
            with pytest.raises(ModelRateLimitError):
                await model.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_reraises_provider_error(self):
        model = _make_chat_oci_raw()

        with patch.object(
            model, "_make_request", new_callable=AsyncMock,
            side_effect=ModelProviderError(message="provider error", status_code=500, model="test"),
        ):
            with pytest.raises(ModelProviderError):
                await model.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_wraps_unexpected_error(self):
        model = _make_chat_oci_raw()

        with patch.object(
            model, "_make_request", new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected"),
        ):
            with pytest.raises(ModelProviderError, match="Unexpected error"):
                await model.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_empty_messages(self):
        """Test structured output with empty message list (no system message to modify)."""
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        json_text = json.dumps({"value": "test"})
        mock_response = _make_oci_response(text=json_text)

        msgs: list = []

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response) as mock_req:
            result = await model.ainvoke(msgs, output_format=TestOutput)

        assert result.completion.value == "test"
        # Should have inserted a system message
        call_args = mock_req.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].role == "system"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_json_decode_error(self):
        """Test that json.JSONDecodeError is caught for structured output."""
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        # Response that looks like it could have JSON but is malformed
        mock_response = _make_oci_response(text='{"value": broken}')

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(ModelProviderError, match="Failed to parse structured output"):
                await model.ainvoke(
                    [UserMessage(content="test")],
                    output_format=TestOutput,
                )

    @pytest.mark.asyncio
    async def test_ainvoke_structured_validation_error(self):
        """Test that ValueError from model_validate is caught."""
        from pydantic import BaseModel as PydanticBaseModel

        class StrictOutput(PydanticBaseModel):
            count: int

        model = _make_chat_oci_raw()
        # Valid JSON but wrong type for field
        mock_response = _make_oci_response(text='{"count": "not_a_number"}')

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(ModelProviderError, match="Failed to parse structured output|Unexpected error"):
                await model.ainvoke(
                    [UserMessage(content="test")],
                    output_format=StrictOutput,
                )

    @pytest.mark.asyncio
    async def test_ainvoke_structured_output_json_no_braces(self):
        """JSON response that doesn't start with { and has no { at all."""
        from pydantic import BaseModel as PydanticBaseModel

        class TestOutput(PydanticBaseModel):
            value: str

        model = _make_chat_oci_raw()
        mock_response = _make_oci_response(text="just plain text no json here")

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(ModelProviderError, match="Failed to parse structured output"):
                await model.ainvoke(
                    [UserMessage(content="test")],
                    output_format=TestOutput,
                )

    @pytest.mark.asyncio
    async def test_ainvoke_string_response_no_usage(self):
        """String response without usage info."""
        model = _make_chat_oci_raw()
        mock_response = _make_oci_response(text="Hello!", has_usage=False)

        with patch.object(model, "_make_request", new_callable=AsyncMock, return_value=mock_response):
            result = await model.ainvoke([UserMessage(content="Hi")])

        assert result.completion == "Hello!"
        assert result.usage is None


class TestChatOCIRawInit:
    """Tests for ChatOCIRaw __init__ / dataclass defaults."""

    def test_default_values(self):
        model = _make_chat_oci_raw()
        assert model.provider == "meta"
        assert model.temperature == 1.0
        assert model.max_tokens == 600
        assert model.frequency_penalty == 0.0
        assert model.presence_penalty == 0.0
        assert model.top_p == 0.75
        assert model.top_k == 0
        assert model.auth_type == "API_KEY"
        assert model.auth_profile == "DEFAULT"
        assert model.timeout == 60.0

    def test_custom_values(self):
        model = _make_chat_oci_raw(
            provider="xai",
            temperature=0.5,
            max_tokens=1000,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            top_p=0.9,
            top_k=10,
            auth_type="INSTANCE_PRINCIPAL",
            auth_profile="CUSTOM",
            timeout=120.0,
        )
        assert model.provider == "xai"
        assert model.temperature == 0.5
        assert model.max_tokens == 1000
        assert model.top_k == 10
        assert model.auth_type == "INSTANCE_PRINCIPAL"
        assert model.auth_profile == "CUSTOM"
        assert model.timeout == 120.0
