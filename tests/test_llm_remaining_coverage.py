"""Tests for remaining LLM provider modules.

Covers:
  src/openbrowser/llm/ollama/chat.py
  src/openbrowser/llm/ollama/serializer.py
  src/openbrowser/llm/cerebras/chat.py
  src/openbrowser/llm/cerebras/serializer.py
  src/openbrowser/llm/oci_raw/chat.py
  src/openbrowser/llm/oci_raw/serializer.py
  src/openbrowser/llm/azure/chat.py
  src/openbrowser/llm/browser_use/chat.py
  src/openbrowser/llm/schema.py
  src/openbrowser/llm/models.py
  src/openbrowser/llm/base.py
  src/openbrowser/llm/__init__.py
"""

import base64
import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

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
# Tests for SchemaOptimizer (schema.py)
# ===========================================================================
class TestSchemaOptimizer:
    """Tests for llm/schema.py coverage."""

    def test_create_optimized_json_schema_basic(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.schema import SchemaOptimizer

        class Simple(PydanticBaseModel):
            name: str
            count: int

        schema = SchemaOptimizer.create_optimized_json_schema(Simple)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert schema["additionalProperties"] is False
        # All properties should be required
        assert "name" in schema["required"]
        assert "count" in schema["required"]

    def test_create_optimized_json_schema_with_refs(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.schema import SchemaOptimizer

        class Inner(PydanticBaseModel):
            value: str

        class Outer(PydanticBaseModel):
            inner: Inner
            name: str

        schema = SchemaOptimizer.create_optimized_json_schema(Outer)
        assert "$defs" not in schema
        assert "$ref" not in json.dumps(schema)

    def test_create_optimized_json_schema_with_anyof(self):
        from typing import Union
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.schema import SchemaOptimizer

        class TypeA(PydanticBaseModel):
            a: str

        class TypeB(PydanticBaseModel):
            b: int

        class Container(PydanticBaseModel):
            item: Union[TypeA, TypeB]

        schema = SchemaOptimizer.create_optimized_json_schema(Container)
        assert "properties" in schema

    def test_create_optimized_json_schema_description_handling(self):
        from pydantic import BaseModel as PydanticBaseModel, Field
        from openbrowser.llm.schema import SchemaOptimizer

        class Described(PydanticBaseModel):
            name: str = Field(description="The name field")
            empty_desc: str = Field(description="")

        schema = SchemaOptimizer.create_optimized_json_schema(Described)
        # Non-empty description should be preserved
        name_props = schema["properties"]["name"]
        assert "description" in name_props

    def test_create_optimized_json_schema_with_items(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.schema import SchemaOptimizer

        class WithList(PydanticBaseModel):
            items: list[str]

        schema = SchemaOptimizer.create_optimized_json_schema(WithList)
        assert "items" in schema["properties"]

    def test_make_strict_compatible(self):
        from openbrowser.llm.schema import SchemaOptimizer

        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
            "required": ["a"],
        }
        SchemaOptimizer._make_strict_compatible(schema)
        # All properties should now be required
        assert set(schema["required"]) == {"a", "b"}

    def test_make_strict_compatible_nested(self):
        from openbrowser.llm.schema import SchemaOptimizer

        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string"},
                    },
                    "required": [],
                },
            },
            "required": [],
        }
        SchemaOptimizer._make_strict_compatible(schema)
        assert "x" in schema["properties"]["nested"]["required"]

    def test_make_strict_compatible_list(self):
        from openbrowser.llm.schema import SchemaOptimizer

        schema_list = [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": []},
        ]
        SchemaOptimizer._make_strict_compatible(schema_list)
        assert "a" in schema_list[0]["required"]

    def test_create_gemini_optimized_schema(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.schema import SchemaOptimizer

        class Simple(PydanticBaseModel):
            name: str

        schema = SchemaOptimizer.create_gemini_optimized_schema(Simple)
        assert "required" not in schema

    def test_ensure_additional_properties_false(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.schema import SchemaOptimizer

        class Nested(PydanticBaseModel):
            inner: dict

        schema = SchemaOptimizer.create_optimized_json_schema(Nested)
        # All object types should have additionalProperties: false
        assert schema.get("additionalProperties") is False

    def test_create_optimized_json_schema_ref_with_description_merge(self):
        """Cover lines 96-99: description merging when flattening $ref."""
        from openbrowser.llm.schema import SchemaOptimizer

        # Manually build a schema with $defs and $ref that also has a description
        # sibling (which should be merged into the flattened ref)
        raw_schema = {
            "type": "object",
            "properties": {
                "item": {
                    "$ref": "#/$defs/Inner",
                    "description": "Override description",
                },
            },
            "required": ["item"],
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                    },
                    "required": ["value"],
                },
            },
        }

        # We need to call the internal optimize_schema logic.
        # Since create_optimized_json_schema uses model_json_schema(),
        # we patch it to return our raw schema.
        from pydantic import BaseModel as PydanticBaseModel

        class Dummy(PydanticBaseModel):
            x: str

        with patch.object(Dummy, "model_json_schema", return_value=raw_schema):
            result = SchemaOptimizer.create_optimized_json_schema(Dummy)
        # The description should have been carried through
        assert result["properties"]["item"]["description"] == "Override description"

    def test_create_gemini_optimized_schema_with_list(self):
        """Cover line 187: list in remove_required_arrays for Gemini."""
        from openbrowser.llm.schema import SchemaOptimizer

        raw_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": [
                        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
                    ],
                },
            },
            "required": ["items"],
        }

        from pydantic import BaseModel as PydanticBaseModel

        class Dummy(PydanticBaseModel):
            x: str

        with patch.object(Dummy, "model_json_schema", return_value=raw_schema):
            result = SchemaOptimizer.create_gemini_optimized_schema(Dummy)
        # required arrays should have been removed
        assert "required" not in result


# ===========================================================================
# Tests for BaseChatModel (base.py)
# ===========================================================================
class TestBaseChatModel:
    """Tests for llm/base.py coverage."""

    def test_model_name_property(self):
        """Test that model_name returns self.model (line 32)."""
        from openbrowser.llm.base import BaseChatModel

        class MockModel:
            _verified_api_keys = False
            model = "test-model"

            @property
            def provider(self):
                return "test"

            @property
            def name(self):
                return "test"

            async def ainvoke(self, messages, output_format=None):
                pass

        m = MockModel()
        assert m.model == "test-model"
        # Test the model_name property from BaseChatModel
        assert BaseChatModel.model_name.fget(m) == "test-model"

    def test_get_pydantic_core_schema(self):
        from openbrowser.llm.base import BaseChatModel

        schema = BaseChatModel.__get_pydantic_core_schema__(BaseChatModel, None)
        assert schema is not None

    def test_protocol_isinstance(self):
        from openbrowser.llm.base import BaseChatModel

        class ValidModel:
            _verified_api_keys = False
            model = "test"

            @property
            def provider(self):
                return "test"

            @property
            def name(self):
                return "test"

            @property
            def model_name(self):
                return "test"

            async def ainvoke(self, messages, output_format=None):
                pass

            @classmethod
            def __get_pydantic_core_schema__(cls, source_type, handler):
                pass

        assert isinstance(ValidModel(), BaseChatModel)


# ===========================================================================
# Tests for llm/messages.py - __str__, __repr__, text properties, helpers
# ===========================================================================
class TestMessages:
    """Tests for llm/messages.py coverage."""

    # ---- helper functions ----

    def test_truncate_short(self):
        from openbrowser.llm.messages import _truncate
        assert _truncate("hello", 50) == "hello"

    def test_truncate_long(self):
        from openbrowser.llm.messages import _truncate
        result = _truncate("a" * 100, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_format_image_url_base64(self):
        from openbrowser.llm.messages import _format_image_url
        result = _format_image_url("data:image/png;base64,abc123")
        assert result == "<base64 image/png>"

    def test_format_image_url_regular(self):
        from openbrowser.llm.messages import _format_image_url
        result = _format_image_url("https://example.com/image.png")
        assert "example.com" in result

    def test_format_image_url_regular_long(self):
        from openbrowser.llm.messages import _format_image_url
        long_url = "https://example.com/" + "x" * 200
        result = _format_image_url(long_url, 30)
        assert len(result) == 30
        assert result.endswith("...")

    # ---- ContentPartTextParam ----

    def test_content_part_text_str(self):
        part = ContentPartTextParam(text="hello world")
        s = str(part)
        assert "Text:" in s
        assert "hello world" in s

    def test_content_part_text_repr(self):
        part = ContentPartTextParam(text="hello world")
        r = repr(part)
        assert "ContentPartTextParam" in r

    # ---- ContentPartRefusalParam ----

    def test_content_part_refusal_str(self):
        part = ContentPartRefusalParam(refusal="I cannot do that")
        s = str(part)
        assert "Refusal:" in s

    def test_content_part_refusal_repr(self):
        part = ContentPartRefusalParam(refusal="I cannot do that")
        r = repr(part)
        assert "ContentPartRefusalParam" in r

    # ---- ImageURL ----

    def test_image_url_str(self):
        img = ImageURL(url="https://example.com/img.png")
        s = str(img)
        assert "Image" in s

    def test_image_url_repr(self):
        img = ImageURL(url="https://example.com/img.png")
        r = repr(img)
        assert "ImageURL" in r

    def test_image_url_str_base64(self):
        img = ImageURL(url="data:image/jpeg;base64,abc123")
        s = str(img)
        assert "base64" in s

    # ---- ContentPartImageParam ----

    def test_content_part_image_str(self):
        img = ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png"))
        s = str(img)
        assert "Image" in s

    def test_content_part_image_repr(self):
        img = ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png"))
        r = repr(img)
        assert "ContentPartImageParam" in r

    # ---- Function ----

    def test_function_str(self):
        func = Function(arguments='{"x": 1}', name="my_func")
        s = str(func)
        assert "my_func" in s

    def test_function_repr(self):
        func = Function(arguments='{"x": 1}', name="my_func")
        r = repr(func)
        assert "Function" in r

    # ---- ToolCall ----

    def test_tool_call_str(self):
        tc = ToolCall(id="tc_1", function=Function(arguments='{}', name="fn"))
        s = str(tc)
        assert "ToolCall" in s
        assert "tc_1" in s

    def test_tool_call_repr(self):
        tc = ToolCall(id="tc_1", function=Function(arguments='{}', name="fn"))
        r = repr(tc)
        assert "ToolCall" in r

    # ---- UserMessage ----

    def test_user_message_text_string(self):
        msg = UserMessage(content="hello")
        assert msg.text == "hello"

    def test_user_message_text_list(self):
        msg = UserMessage(content=[ContentPartTextParam(text="a"), ContentPartTextParam(text="b")])
        assert "a" in msg.text
        assert "b" in msg.text

    def test_user_message_str(self):
        msg = UserMessage(content="hello")
        s = str(msg)
        assert "UserMessage" in s

    def test_user_message_repr(self):
        msg = UserMessage(content="hello")
        r = repr(msg)
        assert "UserMessage" in r

    # ---- SystemMessage ----

    def test_system_message_text_string(self):
        msg = SystemMessage(content="system prompt")
        assert msg.text == "system prompt"

    def test_system_message_text_list(self):
        msg = SystemMessage(content=[ContentPartTextParam(text="a")])
        assert "a" in msg.text

    def test_system_message_str(self):
        msg = SystemMessage(content="system prompt")
        s = str(msg)
        assert "SystemMessage" in s

    def test_system_message_repr(self):
        msg = SystemMessage(content="system prompt")
        r = repr(msg)
        assert "SystemMessage" in r

    # ---- AssistantMessage ----

    def test_assistant_message_text_string(self):
        msg = AssistantMessage(content="response")
        assert msg.text == "response"

    def test_assistant_message_text_list_with_refusal(self):
        msg = AssistantMessage(content=[
            ContentPartTextParam(text="hello"),
            ContentPartRefusalParam(refusal="nope"),
        ])
        assert "hello" in msg.text
        assert "[Refusal] nope" in msg.text

    def test_assistant_message_text_none(self):
        msg = AssistantMessage(content=None)
        assert msg.text == ""

    def test_assistant_message_str(self):
        msg = AssistantMessage(content="response")
        s = str(msg)
        assert "AssistantMessage" in s

    def test_assistant_message_repr(self):
        msg = AssistantMessage(content="response")
        r = repr(msg)
        assert "AssistantMessage" in r


# ===========================================================================
# Tests for llm/__init__.py
# ===========================================================================
class TestLLMInit:
    """Tests for llm/__init__.py coverage."""

    def test_lazy_import_chat_class(self):
        """Test lazy import of a chat class."""
        openai_mod = pytest.importorskip("openai")
        from openbrowser import llm

        cls = llm.ChatOpenAI
        assert cls is not None

    def test_lazy_import_failure(self):
        """Test that lazy import handles ImportError gracefully."""
        from openbrowser import llm

        with pytest.raises(AttributeError, match="has no attribute"):
            llm.totally_nonexistent_attribute_xyz

    def test_lazy_import_import_error(self):
        """Cover lines 109-110: ImportError during lazy import of a known class."""
        from openbrowser import llm

        # Temporarily add a fake entry to _LAZY_IMPORTS
        llm._LAZY_IMPORTS["FakeClass"] = ("nonexistent.module.path", "FakeClass")
        try:
            with pytest.raises(ImportError, match="Failed to import FakeClass"):
                llm.FakeClass
        finally:
            del llm._LAZY_IMPORTS["FakeClass"]

    def test_lazy_import_model_instance(self):
        """Test lazy loading of model instances."""
        openai_mod = pytest.importorskip("openai")
        from openbrowser import llm

        model = llm.openai_gpt_4o
        assert model is not None

    def test_model_instance_caching(self):
        """Test that model instances are cached."""
        openai_mod = pytest.importorskip("openai")
        from openbrowser import llm

        # Clear cache first
        llm._model_cache.clear()

        m1 = llm.openai_gpt_4o
        m2 = llm.openai_gpt_4o
        # Second access should come from cache
        assert m2 is not None


# ===========================================================================
# Tests for llm/models.py
# ===========================================================================
class TestLLMModels:
    """Tests for llm/models.py coverage."""

    def test_get_llm_by_name_empty(self):
        from openbrowser.llm.models import get_llm_by_name

        with pytest.raises(ValueError, match="cannot be empty"):
            get_llm_by_name("")

    def test_get_llm_by_name_no_underscore(self):
        from openbrowser.llm.models import get_llm_by_name

        with pytest.raises(ValueError, match="Invalid model name format"):
            get_llm_by_name("nounderscores")

    def test_get_llm_by_name_openai(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("openai_gpt_4o")
        assert model is not None

    def test_get_llm_by_name_openai_gpt_4_1_mini(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("openai_gpt_4_1_mini")
        assert model.model == "gpt-4.1-mini"

    def test_get_llm_by_name_openai_gpt_4o_mini(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("openai_gpt_4o_mini")
        assert "4o-mini" in model.model

    def test_get_llm_by_name_google(self):
        pytest.importorskip("google.genai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("google_gemini_2_0_flash")
        assert "gemini-2.0" in model.model

    def test_get_llm_by_name_google_25(self):
        pytest.importorskip("google.genai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("google_gemini_2_5_pro")
        assert "gemini-2.5" in model.model

    def test_get_llm_by_name_azure(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("azure_gpt_4o")
        assert model is not None

    def test_get_llm_by_name_cerebras(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_llama3_1_8b")
        assert "llama3.1" in model.model

    def test_get_llm_by_name_cerebras_llama3_3(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_llama3_3_70b")
        assert "llama-3.3" in model.model

    def test_get_llm_by_name_cerebras_llama_4_scout(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_llama_4_scout_17b_16e_instruct")
        assert "llama-4-scout" in model.model

    def test_get_llm_by_name_cerebras_llama_4_maverick(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_llama_4_maverick_17b_128e_instruct")
        assert "llama-4-maverick" in model.model

    def test_get_llm_by_name_cerebras_gpt_oss(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_gpt_oss_120b")
        assert "gpt-oss-120b" in model.model

    def test_get_llm_by_name_cerebras_qwen_3_32b(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_qwen_3_32b")
        assert "qwen-3-32b" in model.model

    def test_get_llm_by_name_cerebras_qwen_3_235b_instruct(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_qwen_3_235b_a22b_instruct_2507")
        assert "qwen-3-235b-a22b-instruct-2507" in model.model

    def test_get_llm_by_name_cerebras_qwen_3_235b_thinking(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_qwen_3_235b_a22b_thinking_2507")
        assert "qwen-3-235b-a22b-thinking-2507" in model.model

    def test_get_llm_by_name_cerebras_qwen_3_coder_480b(self):
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_qwen_3_coder_480b")
        assert "qwen-3-coder-480b" in model.model

    def test_get_llm_by_name_bu(self):
        from openbrowser.llm.models import get_llm_by_name

        with patch.dict(os.environ, {"BROWSER_USE_API_KEY": "test-key"}):
            model = get_llm_by_name("bu_latest")
        assert model is not None

    def test_get_llm_by_name_oci(self):
        from openbrowser.llm.models import get_llm_by_name

        with pytest.raises(ValueError, match="OCI models require manual"):
            get_llm_by_name("oci_some_model")

    def test_get_llm_by_name_unknown_provider(self):
        from openbrowser.llm.models import get_llm_by_name

        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm_by_name("unknown_model")

    def test_getattr_chat_classes(self):
        pytest.importorskip("openai")
        from openbrowser.llm import models

        assert models.__getattr__("ChatOpenAI") is not None
        assert models.__getattr__("ChatGoogle") is not None
        assert models.__getattr__("ChatCerebras") is not None
        assert models.__getattr__("ChatBrowserUse") is not None

    def test_getattr_azure(self):
        pytest.importorskip("openai")
        from openbrowser.llm import models

        assert models.__getattr__("ChatAzureOpenAI") is not None

    def test_getattr_model_instance(self):
        pytest.importorskip("openai")
        from openbrowser.llm import models

        model = models.__getattr__("openai_gpt_4o")
        assert model is not None

    def test_getattr_unknown(self):
        from openbrowser.llm import models

        with pytest.raises(AttributeError, match="has no attribute"):
            models.__getattr__("totally_nonexistent")

    def test_getattr_oci_not_available(self):
        from openbrowser.llm import models

        with patch.object(models, "OCI_AVAILABLE", False):
            with patch.object(models, "ChatOCIRaw", None):
                with pytest.raises(ImportError, match="OCI integration"):
                    models.__getattr__("ChatOCIRaw")

    def test_get_llm_by_name_fallback_model_name(self):
        """Test model name that doesn't match any special pattern."""
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("openai_o3")
        assert model.model == "o3"

    def test_get_llm_by_name_cerebras_qwen_instruct_no_2507(self):
        """Test qwen instruct model without _2507 suffix."""
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_qwen_3_235b_a22b_instruct")
        assert "qwen-3-235b-a22b-instruct-2507" in model.model

    def test_get_llm_by_name_cerebras_qwen_thinking_no_2507(self):
        """Test qwen thinking model without _2507 suffix."""
        pytest.importorskip("openai")
        from openbrowser.llm.models import get_llm_by_name

        model = get_llm_by_name("cerebras_qwen_3_235b_a22b_thinking")
        assert "qwen-3-235b-a22b-thinking-2507" in model.model


# ===========================================================================
# Tests for Ollama serializer and chat
# ===========================================================================
class TestOllamaSerializer:
    """Tests for ollama/serializer.py coverage."""

    def setup_method(self):
        pytest.importorskip("ollama")

    def test_extract_text_content_none(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        assert OllamaMessageSerializer._extract_text_content(None) == ""

    def test_extract_text_content_string(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        assert OllamaMessageSerializer._extract_text_content("hello") == "hello"

    def test_extract_text_content_list(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        parts = [
            ContentPartTextParam(text="text1"),
            ContentPartRefusalParam(refusal="refuse"),
        ]
        result = OllamaMessageSerializer._extract_text_content(parts)
        assert "text1" in result
        assert "[Refusal] refuse" in result

    def test_extract_images_none(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        assert OllamaMessageSerializer._extract_images(None) == []

    def test_extract_images_string(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        assert OllamaMessageSerializer._extract_images("text") == []

    def test_extract_images_base64(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        b64 = base64.b64encode(b"fake").decode()
        parts = [
            ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}")),
        ]
        result = OllamaMessageSerializer._extract_images(parts)
        assert len(result) == 1

    def test_extract_images_url(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        parts = [
            ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png")),
        ]
        result = OllamaMessageSerializer._extract_images(parts)
        assert len(result) == 1

    def test_serialize_tool_calls(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        tcs = [ToolCall(id="tc1", function=Function(name="fn", arguments='{"a": 1}'))]
        result = OllamaMessageSerializer._serialize_tool_calls(tcs)
        assert len(result) == 1
        assert result[0].function.name == "fn"

    def test_serialize_tool_calls_invalid_json(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        tcs = [ToolCall(id="tc1", function=Function(name="fn", arguments="not json"))]
        result = OllamaMessageSerializer._serialize_tool_calls(tcs)
        assert result[0].function.arguments == {"arguments": "not json"}

    def test_serialize_user_message_string(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        msg = UserMessage(content="hello")
        result = OllamaMessageSerializer.serialize(msg)
        assert result.role == "user"
        assert result.content == "hello"

    def test_serialize_user_message_with_images(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        b64 = base64.b64encode(b"fake").decode()
        msg = UserMessage(content=[
            ContentPartTextParam(text="look"),
            ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}")),
        ])
        result = OllamaMessageSerializer.serialize(msg)
        assert result.role == "user"
        assert result.images is not None

    def test_serialize_system_message(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        msg = SystemMessage(content="sys")
        result = OllamaMessageSerializer.serialize(msg)
        assert result.role == "system"

    def test_serialize_assistant_message(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{}'))
        msg = AssistantMessage(content="resp", tool_calls=[tc])
        result = OllamaMessageSerializer.serialize(msg)
        assert result.role == "assistant"
        assert result.tool_calls is not None

    def test_serialize_assistant_none_content(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        msg = AssistantMessage(content=None)
        result = OllamaMessageSerializer.serialize(msg)
        assert result.content is None

    def test_serialize_unknown_raises(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        with pytest.raises(ValueError, match="Unknown message type"):
            OllamaMessageSerializer.serialize(MagicMock())

    def test_serialize_messages(self):
        from openbrowser.llm.ollama.serializer import OllamaMessageSerializer

        msgs = [SystemMessage(content="sys"), UserMessage(content="hi")]
        result = OllamaMessageSerializer.serialize_messages(msgs)
        assert len(result) == 2


class TestChatOllama:
    """Tests for ollama/chat.py coverage."""

    def setup_method(self):
        pytest.importorskip("ollama")

    def _make_chat(self, **kwargs):
        from openbrowser.llm.ollama.chat import ChatOllama

        defaults = {"model": "llama3", "host": "http://localhost:11434"}
        defaults.update(kwargs)
        return ChatOllama(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "ollama"

    def test_name(self):
        chat = self._make_chat()
        assert chat.name == "llama3"

    def test_get_client_params(self):
        chat = self._make_chat()
        params = chat._get_client_params()
        assert params["host"] == "http://localhost:11434"

    def test_get_client(self):
        chat = self._make_chat()
        client = chat.get_client()
        from ollama import AsyncClient

        assert isinstance(client, AsyncClient)

    @pytest.mark.asyncio
    async def test_ainvoke_text(self):
        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.message.content = "test reply"
        mock_client.chat = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "test reply"

    @pytest.mark.asyncio
    async def test_ainvoke_text_none_content(self):
        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.message.content = None
        mock_client.chat = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == ""

    @pytest.mark.asyncio
    async def test_ainvoke_structured(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.message.content = '{"field": "value"}'
        mock_client.chat = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)
        assert result.completion.field == "value"

    @pytest.mark.asyncio
    async def test_ainvoke_structured_none_content(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.message.content = None
        mock_client.chat = AsyncMock(return_value=resp)

        with patch.object(chat, "get_client", return_value=mock_client):
            # None content -> '' -> validation fails
            from openbrowser.llm.exceptions import ModelProviderError

            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_exception(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(chat, "get_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="hi")])


# ===========================================================================
# Tests for Cerebras serializer and chat
# ===========================================================================
class TestCerebrasSerializer:
    """Tests for cerebras/serializer.py coverage."""

    def test_serialize_text_part(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        part = ContentPartTextParam(text="hello")
        assert CerebrasMessageSerializer._serialize_text_part(part) == "hello"

    def test_serialize_image_part_data_url(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        part = ContentPartImageParam(image_url=ImageURL(url="data:image/png;base64,abc"))
        result = CerebrasMessageSerializer._serialize_image_part(part)
        assert result["type"] == "image_url"

    def test_serialize_image_part_regular_url(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        part = ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png"))
        result = CerebrasMessageSerializer._serialize_image_part(part)
        assert result["type"] == "image_url"

    def test_serialize_content_none(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        assert CerebrasMessageSerializer._serialize_content(None) == ""

    def test_serialize_content_string(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        assert CerebrasMessageSerializer._serialize_content("hello") == "hello"

    def test_serialize_content_list(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        parts = [
            ContentPartTextParam(text="text"),
            ContentPartImageParam(image_url=ImageURL(url="https://img.com/a.png")),
            ContentPartRefusalParam(refusal="no"),
        ]
        result = CerebrasMessageSerializer._serialize_content(parts)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_serialize_tool_calls(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        tcs = [ToolCall(id="tc1", function=Function(name="fn", arguments='{"a": 1}'))]
        result = CerebrasMessageSerializer._serialize_tool_calls(tcs)
        assert len(result) == 1
        assert result[0]["function"]["arguments"] == {"a": 1}

    def test_serialize_tool_calls_invalid_json(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        tcs = [ToolCall(id="tc1", function=Function(name="fn", arguments="not json"))]
        result = CerebrasMessageSerializer._serialize_tool_calls(tcs)
        assert result[0]["function"]["arguments"] == {"arguments": "not json"}

    def test_serialize_user(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        msg = UserMessage(content="hi")
        result = CerebrasMessageSerializer.serialize(msg)
        assert result["role"] == "user"

    def test_serialize_system(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        msg = SystemMessage(content="sys")
        result = CerebrasMessageSerializer.serialize(msg)
        assert result["role"] == "system"

    def test_serialize_assistant(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        tc = ToolCall(id="tc1", function=Function(name="fn", arguments='{}'))
        msg = AssistantMessage(content="resp", tool_calls=[tc])
        result = CerebrasMessageSerializer.serialize(msg)
        assert result["role"] == "assistant"
        assert "tool_calls" in result

    def test_serialize_unknown_raises(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        with pytest.raises(ValueError, match="Unknown message type"):
            CerebrasMessageSerializer.serialize(MagicMock())

    def test_serialize_messages(self):
        pytest.importorskip("openai")
        from openbrowser.llm.cerebras.serializer import CerebrasMessageSerializer

        msgs = [SystemMessage(content="sys"), UserMessage(content="hi")]
        result = CerebrasMessageSerializer.serialize_messages(msgs)
        assert len(result) == 2


class TestChatCerebras:
    """Tests for cerebras/chat.py coverage."""

    def setup_method(self):
        pytest.importorskip("openai")

    def _make_chat(self, **kwargs):
        from openbrowser.llm.cerebras.chat import ChatCerebras

        defaults = {"model": "llama3.1-8b", "api_key": "test-key"}
        defaults.update(kwargs)
        return ChatCerebras(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "cerebras"

    def test_name(self):
        chat = self._make_chat()
        assert chat.name == "llama3.1-8b"

    def test_client(self):
        chat = self._make_chat()
        client = chat._client()
        from openai import AsyncOpenAI

        assert isinstance(client, AsyncOpenAI)

    def test_get_usage_with_data(self):
        chat = self._make_chat()
        resp = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        usage = chat._get_usage(resp)
        assert usage.prompt_tokens == 10

    def test_get_usage_none(self):
        chat = self._make_chat()
        resp = MagicMock()
        resp.usage = None
        usage = chat._get_usage(resp)
        assert usage is None

    @pytest.mark.asyncio
    async def test_ainvoke_text(self):
        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "test reply"
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "test reply"

    @pytest.mark.asyncio
    async def test_ainvoke_text_with_params(self):
        chat = self._make_chat(temperature=0.5, max_tokens=1024, top_p=0.9, seed=42)
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "reply"
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "reply"

    @pytest.mark.asyncio
    async def test_ainvoke_text_rate_limit(self):
        from openai import RateLimitError
        from openbrowser.llm.exceptions import ModelRateLimitError

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = RateLimitError(message="rate limited", response=mock_response, body=None)
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelRateLimitError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_text_api_error(self):
        from openai import APIError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        error = MagicMock(spec=APIError)
        error.__str__ = lambda s: "api error"
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_text_generic_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_structured(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"field": "value"}'
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_json_in_text(self):
        """Test when JSON is embedded in text."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = 'Here is the result: {"field": "value"} done.'
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "_client", return_value=mock_client):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_empty_content(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = ""
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="Empty JSON"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_rate_limit(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openai import RateLimitError
        from openbrowser.llm.exceptions import ModelRateLimitError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = RateLimitError(message="rate limited", response=mock_response, body=None)
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelRateLimitError):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_generic_error(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="boom"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_list_content(self):
        """Test structured output modifying list content in last message."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"field": "value"}'
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        msgs = [
            SystemMessage(content="sys"),
            UserMessage(content=[ContentPartTextParam(text="extract data")]),
        ]

        with patch.object(chat, "_client", return_value=mock_client):
            result = await chat.ainvoke(msgs, output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_no_user_msg(self):
        """Test structured output with no user message at end."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"field": "value"}'
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        msgs = [SystemMessage(content="sys")]

        with patch.object(chat, "_client", return_value=mock_client):
            result = await chat.ainvoke(msgs, output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_text_api_connection_error(self):
        """Cover line 128: APIConnectionError in text path."""
        import httpx
        from openai import APIConnectionError
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = AsyncMock()
        req = httpx.Request("GET", "http://test.com")
        error = APIConnectionError(message="connection error", request=req)
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="connection error"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_structured_no_json_match(self):
        """Cover line 179: no regex match so json_str = content directly."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        # Content with no curly braces -- regex won't match, falls back to raw content
        # This will fail validation, but we're testing the code path
        resp.choices[0].message.content = 'no json here at all'
        resp.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(Exception):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_api_connection_error(self):
        """Cover line 189: APIConnectionError in structured path."""
        import httpx
        from pydantic import BaseModel as PydanticBaseModel
        from openai import APIConnectionError
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_client = AsyncMock()
        req = httpx.Request("GET", "http://test.com")
        error = APIConnectionError(message="connection error", request=req)
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        with patch.object(chat, "_client", return_value=mock_client):
            with pytest.raises(ModelProviderError, match="connection error"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)


# ===========================================================================
# Tests for OCI Raw serializer and chat
# ===========================================================================
class TestOCIRawSerializer:
    """Tests for oci_raw/serializer.py coverage."""

    def setup_method(self):
        pytest.importorskip("oci")

    def test_is_base64_image(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        assert OCIRawMessageSerializer._is_base64_image("data:image/png;base64,abc") is True
        assert OCIRawMessageSerializer._is_base64_image("https://example.com") is False

    def test_parse_base64_url(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        result = OCIRawMessageSerializer._parse_base64_url("data:image/png;base64,SGVsbG8=")
        assert result == "SGVsbG8="

    def test_parse_base64_url_invalid(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        with pytest.raises(ValueError, match="Not a base64"):
            OCIRawMessageSerializer._parse_base64_url("https://example.com")

    def test_parse_base64_url_no_comma(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        with pytest.raises(ValueError, match="Invalid base64"):
            OCIRawMessageSerializer._parse_base64_url("data:image/pngbase64nocomma")

    def test_create_image_content_base64(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        part = ContentPartImageParam(image_url=ImageURL(url="data:image/png;base64,abc"))
        result = OCIRawMessageSerializer._create_image_content(part)
        assert result is not None

    def test_create_image_content_url(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        part = ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png"))
        result = OCIRawMessageSerializer._create_image_content(part)
        assert result is not None

    def test_serialize_messages_user_string(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [UserMessage(content="hello")]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 1
        assert result[0].role == "USER"

    def test_serialize_messages_user_list(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [UserMessage(content=[
            ContentPartTextParam(text="text"),
            ContentPartImageParam(image_url=ImageURL(url="data:image/png;base64,abc")),
        ])]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 1

    def test_serialize_messages_system(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [SystemMessage(content="sys")]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 1
        assert result[0].role == "SYSTEM"

    def test_serialize_messages_system_list(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [SystemMessage(content=[
            ContentPartTextParam(text="text"),
        ])]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 1

    def test_serialize_messages_assistant(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [AssistantMessage(content="reply")]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 1
        assert result[0].role == "ASSISTANT"

    def test_serialize_messages_assistant_list(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [AssistantMessage(content=[
            ContentPartTextParam(text="text"),
            ContentPartRefusalParam(refusal="no"),
        ])]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 1

    def test_serialize_messages_assistant_none(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [AssistantMessage(content=None)]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 0  # No content -> not appended

    def test_serialize_messages_unknown_type(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        mock_msg = MagicMock()
        mock_msg.__class__ = type("CustomMsg", (), {})
        mock_msg.__str__ = lambda s: "custom msg"
        msgs = [mock_msg]
        result = OCIRawMessageSerializer.serialize_messages(msgs)
        assert len(result) == 1

    def test_serialize_messages_for_cohere_user_string(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [UserMessage(content="hello")]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "User: hello" in result

    def test_serialize_messages_for_cohere_user_list(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [UserMessage(content=[
            ContentPartTextParam(text="text"),
            ContentPartImageParam(image_url=ImageURL(url="data:image/png;base64,abc")),
        ])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "User:" in result
        assert "[Image: base64_data]" in result

    def test_serialize_messages_for_cohere_user_url_image(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [UserMessage(content=[
            ContentPartImageParam(image_url=ImageURL(url="https://example.com/img.png")),
        ])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "[Image: external_url]" in result

    def test_serialize_messages_for_cohere_system(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [SystemMessage(content="sys")]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "System: sys" in result

    def test_serialize_messages_for_cohere_system_list(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [SystemMessage(content=[ContentPartTextParam(text="sys text")])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "System: sys text" in result

    def test_serialize_messages_for_cohere_assistant(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [AssistantMessage(content="reply")]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "Assistant: reply" in result

    def test_serialize_messages_for_cohere_assistant_list(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        msgs = [AssistantMessage(content=[
            ContentPartTextParam(text="text"),
            ContentPartRefusalParam(refusal="no"),
        ])]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "text" in result
        assert "[Refusal] no" in result

    def test_serialize_messages_for_cohere_unknown(self):
        from openbrowser.llm.oci_raw.serializer import OCIRawMessageSerializer

        mock_msg = MagicMock()
        mock_msg.__class__ = type("CustomMsg", (), {})
        mock_msg.__str__ = lambda s: "custom"
        msgs = [mock_msg]
        result = OCIRawMessageSerializer.serialize_messages_for_cohere(msgs)
        assert "User: custom" in result


class TestChatOCIRaw:
    """Tests for oci_raw/chat.py coverage."""

    def setup_method(self):
        pytest.importorskip("oci")

    def _make_chat(self, **kwargs):
        from openbrowser.llm.oci_raw.chat import ChatOCIRaw

        defaults = {
            "model_id": "ocid1.generativeaimodel.oc1.test",
            "service_endpoint": "https://inference.generativeai.test.oci.oraclecloud.com",
            "compartment_id": "ocid1.compartment.oc1.test",
        }
        defaults.update(kwargs)
        return ChatOCIRaw(**defaults)

    def test_provider_name(self):
        chat = self._make_chat()
        assert chat.provider_name == "oci-raw"

    def test_name_short(self):
        chat = self._make_chat(model_id="short-model")
        assert chat.name == "short-model"

    def test_name_long(self):
        long_id = "a" * 91
        chat = self._make_chat(model_id=long_id)
        assert len(chat.name) < 100

    def test_name_long_with_parts(self):
        long_id = "ocid1.generativeaimodel.oc1.us-chicago-1." + "a" * 60
        chat = self._make_chat(model_id=long_id)
        assert "oci-meta" in chat.name

    def test_name_long_fewer_parts(self):
        long_id = "part1.part2." + "a" * 90
        chat = self._make_chat(model_id=long_id)
        assert "oci-meta-model" in chat.name

    def test_model(self):
        chat = self._make_chat()
        assert chat.model == chat.model_id

    def test_model_name_short(self):
        chat = self._make_chat(model_id="short")
        assert chat.model_name == "short"

    def test_model_name_long(self):
        long_id = "a" * 91
        chat = self._make_chat(model_id=long_id)
        assert len(chat.model_name) < 100

    def test_uses_cohere_format(self):
        chat = self._make_chat(provider="cohere")
        assert chat._uses_cohere_format() is True
        chat2 = self._make_chat(provider="meta")
        assert chat2._uses_cohere_format() is False

    def test_get_supported_parameters_meta(self):
        chat = self._make_chat(provider="meta")
        params = chat._get_supported_parameters()
        assert params["temperature"] is True
        assert params["top_k"] is False

    def test_get_supported_parameters_cohere(self):
        chat = self._make_chat(provider="cohere")
        params = chat._get_supported_parameters()
        assert params["top_k"] is True
        assert params["presence_penalty"] is False

    def test_get_supported_parameters_xai(self):
        chat = self._make_chat(provider="xai")
        params = chat._get_supported_parameters()
        assert params["top_k"] is True
        assert params["frequency_penalty"] is False

    def test_get_supported_parameters_default(self):
        chat = self._make_chat(provider="unknown")
        params = chat._get_supported_parameters()
        assert params["temperature"] is True

    def test_extract_usage_with_data(self):
        chat = self._make_chat()
        response = MagicMock()
        response.data.chat_response.usage.prompt_tokens = 10
        response.data.chat_response.usage.completion_tokens = 5
        response.data.chat_response.usage.total_tokens = 15
        usage = chat._extract_usage(response)
        assert usage.prompt_tokens == 10

    def test_extract_usage_no_data(self):
        chat = self._make_chat()
        response = MagicMock(spec=[])
        usage = chat._extract_usage(response)
        assert usage is None

    def test_extract_usage_no_usage_attr(self):
        chat = self._make_chat()
        response = MagicMock()
        response.data = MagicMock()
        response.data.chat_response = MagicMock(spec=[])  # no usage
        usage = chat._extract_usage(response)
        assert usage is None

    def test_extract_content_cohere(self):
        chat = self._make_chat(provider="cohere")
        response = MagicMock()
        response.data.chat_response.text = "cohere response"
        result = chat._extract_content(response)
        assert result == "cohere response"

    def test_extract_content_generic(self):
        chat = self._make_chat()
        response = MagicMock()
        part = MagicMock()
        part.text = "text content"
        response.data.chat_response.choices = [MagicMock()]
        response.data.chat_response.choices[0].message.content = [part]
        # Simulate: hasattr(chat_response, 'text') returns False
        del response.data.chat_response.text
        result = chat._extract_content(response)
        assert result == "text content"

    def test_extract_content_no_data(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        response = MagicMock(spec=[])  # no data attr
        with pytest.raises(ModelProviderError, match="Failed to extract"):
            chat._extract_content(response)

    @pytest.mark.asyncio
    async def test_ainvoke_text(self):
        chat = self._make_chat()
        mock_response = MagicMock()
        mock_response.data.chat_response.text = "reply"
        mock_response.data.chat_response.usage.prompt_tokens = 10
        mock_response.data.chat_response.usage.completion_tokens = 5
        mock_response.data.chat_response.usage.total_tokens = 15

        with patch.object(chat, "_make_request", AsyncMock(return_value=mock_response)):
            result = await chat.ainvoke([UserMessage(content="hi")])
        assert result.completion == "reply"

    @pytest.mark.asyncio
    async def test_ainvoke_structured(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_response = MagicMock()
        mock_response.data.chat_response.text = '{"field": "value"}'
        mock_response.data.chat_response.usage.prompt_tokens = 10
        mock_response.data.chat_response.usage.completion_tokens = 5
        mock_response.data.chat_response.usage.total_tokens = 15

        with patch.object(chat, "_make_request", AsyncMock(return_value=mock_response)):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_code_block(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_response = MagicMock()
        mock_response.data.chat_response.text = '```json\n{"field": "value"}\n```'
        mock_response.data.chat_response.usage.prompt_tokens = 10
        mock_response.data.chat_response.usage.completion_tokens = 5
        mock_response.data.chat_response.usage.total_tokens = 15

        with patch.object(chat, "_make_request", AsyncMock(return_value=mock_response)):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_json_in_text(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_response = MagicMock()
        mock_response.data.chat_response.text = 'Result: {"field": "value"}'
        mock_response.data.chat_response.usage.prompt_tokens = 10
        mock_response.data.chat_response.usage.completion_tokens = 5
        mock_response.data.chat_response.usage.total_tokens = 15

        with patch.object(chat, "_make_request", AsyncMock(return_value=mock_response)):
            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_structured_invalid_json(self):
        from pydantic import BaseModel as PydanticBaseModel
        from openbrowser.llm.exceptions import ModelProviderError

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_response = MagicMock()
        mock_response.data.chat_response.text = "not json at all"
        mock_response.data.chat_response.usage.prompt_tokens = 10
        mock_response.data.chat_response.usage.completion_tokens = 5
        mock_response.data.chat_response.usage.total_tokens = 15

        with patch.object(chat, "_make_request", AsyncMock(return_value=mock_response)):
            with pytest.raises(ModelProviderError, match="Failed to parse"):
                await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_rate_limit_error(self):
        from openbrowser.llm.exceptions import ModelRateLimitError

        chat = self._make_chat()

        with patch.object(chat, "_make_request", AsyncMock(side_effect=ModelRateLimitError("rate limited"))):
            with pytest.raises(ModelRateLimitError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_provider_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()

        with patch.object(chat, "_make_request", AsyncMock(side_effect=ModelProviderError("provider error"))):
            with pytest.raises(ModelProviderError):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_generic_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()

        with patch.object(chat, "_make_request", AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(ModelProviderError, match="Unexpected error"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_structured_existing_system(self):
        """Test structured output with existing system message."""
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()
        mock_response = MagicMock()
        mock_response.data.chat_response.text = '{"field": "value"}'
        mock_response.data.chat_response.usage.prompt_tokens = 10
        mock_response.data.chat_response.usage.completion_tokens = 5
        mock_response.data.chat_response.usage.total_tokens = 15

        with patch.object(chat, "_make_request", AsyncMock(return_value=mock_response)):
            result = await chat.ainvoke([
                SystemMessage(content="existing system"),
                UserMessage(content="extract"),
            ], output_format=MyOutput)
        assert isinstance(result.completion, MyOutput)

    def test_get_oci_client_api_key(self):
        chat = self._make_chat(auth_type="API_KEY")
        with patch("oci.config.from_file", return_value={"user": "test"}):
            with patch("oci.generative_ai_inference.GenerativeAiInferenceClient") as mock_cls:
                mock_cls.return_value = MagicMock()
                client = chat._get_oci_client()
                assert client is not None

    def test_get_oci_client_instance_principal(self):
        chat = self._make_chat(auth_type="INSTANCE_PRINCIPAL")
        with patch("oci.auth.signers.InstancePrincipalsSecurityTokenSigner", return_value=MagicMock()):
            with patch("oci.generative_ai_inference.GenerativeAiInferenceClient") as mock_cls:
                mock_cls.return_value = MagicMock()
                client = chat._get_oci_client()
                assert client is not None

    def test_get_oci_client_resource_principal(self):
        chat = self._make_chat(auth_type="RESOURCE_PRINCIPAL")
        with patch("oci.auth.signers.get_resource_principals_signer", return_value=MagicMock()):
            with patch("oci.generative_ai_inference.GenerativeAiInferenceClient") as mock_cls:
                mock_cls.return_value = MagicMock()
                client = chat._get_oci_client()
                assert client is not None

    def test_get_oci_client_fallback(self):
        chat = self._make_chat(auth_type="UNKNOWN")
        with patch("oci.config.from_file", return_value={"user": "test"}):
            with patch("oci.generative_ai_inference.GenerativeAiInferenceClient") as mock_cls:
                mock_cls.return_value = MagicMock()
                client = chat._get_oci_client()
                assert client is not None

    @pytest.mark.asyncio
    async def test_make_request_cohere(self):
        chat = self._make_chat(provider="cohere")
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()

        with patch.object(chat, "_get_oci_client", return_value=mock_client):
            result = await chat._make_request([UserMessage(content="hi")])
        assert result is not None

    @pytest.mark.asyncio
    async def test_make_request_meta(self):
        chat = self._make_chat(provider="meta")
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()

        with patch.object(chat, "_get_oci_client", return_value=mock_client):
            result = await chat._make_request([UserMessage(content="hi")])
        assert result is not None

    @pytest.mark.asyncio
    async def test_make_request_xai(self):
        chat = self._make_chat(provider="xai")
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()

        with patch.object(chat, "_get_oci_client", return_value=mock_client):
            result = await chat._make_request([UserMessage(content="hi")])
        assert result is not None

    @pytest.mark.asyncio
    async def test_make_request_default_provider(self):
        chat = self._make_chat(provider="unknown")
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock()

        with patch.object(chat, "_get_oci_client", return_value=mock_client):
            result = await chat._make_request([UserMessage(content="hi")])
        assert result is not None

    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self):
        from openbrowser.llm.exceptions import ModelRateLimitError

        chat = self._make_chat()
        mock_client = MagicMock()
        error = Exception("rate limit")
        error.status = 429
        mock_client.chat.side_effect = error

        with patch.object(chat, "_get_oci_client", return_value=mock_client):
            with pytest.raises(ModelRateLimitError):
                await chat._make_request([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_make_request_error(self):
        from openbrowser.llm.exceptions import ModelProviderError

        chat = self._make_chat()
        mock_client = MagicMock()
        error = Exception("server error")
        error.status = 500
        mock_client.chat.side_effect = error

        with patch.object(chat, "_get_oci_client", return_value=mock_client):
            with pytest.raises(ModelProviderError):
                await chat._make_request([UserMessage(content="hi")])


# ===========================================================================
# Tests for Azure chat
# ===========================================================================
class TestChatAzureOpenAI:
    """Tests for azure/chat.py coverage."""

    def setup_method(self):
        pytest.importorskip("openai")

    def _make_chat(self, **kwargs):
        from openbrowser.llm.azure.chat import ChatAzureOpenAI

        defaults = {
            "model": "gpt-4o",
            "api_key": "test-key",
            "azure_endpoint": "https://test.openai.azure.com",
        }
        defaults.update(kwargs)
        return ChatAzureOpenAI(**defaults)

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "azure"

    def test_get_client_params(self):
        chat = self._make_chat(
            api_version="2024-01-01",
            azure_deployment="my-deployment",
            default_headers={"X-Custom": "val"},
            default_query={"q": "v"},
        )
        params = chat._get_client_params()
        assert params["api_key"] == "test-key"
        assert params["azure_endpoint"] == "https://test.openai.azure.com"
        assert params["api_version"] == "2024-01-01"
        assert "default_headers" in params
        assert "default_query" in params

    def test_get_client_params_from_env(self):
        chat = self._make_chat(api_key=None, azure_endpoint=None)
        with patch.dict(os.environ, {
            "AZURE_OPENAI_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
        }):
            params = chat._get_client_params()
        assert "api_key" in params

    def test_get_client_cached(self):
        from openai import AsyncAzureOpenAI as Client

        chat = self._make_chat()
        mock_client = MagicMock(spec=Client)
        chat.client = mock_client
        result = chat.get_client()
        assert result is mock_client

    def test_get_client_new(self):
        chat = self._make_chat()
        client = chat.get_client()
        from openai import AsyncAzureOpenAI

        assert isinstance(client, AsyncAzureOpenAI)

    def test_get_client_with_http_client(self):
        import httpx

        http_client = httpx.AsyncClient()
        chat = self._make_chat(http_client=http_client)
        client = chat.get_client()
        assert client is not None


# ===========================================================================
# Tests for ChatBrowserUse
# ===========================================================================
class TestChatBrowserUse:
    """Tests for browser_use/chat.py coverage."""

    def _make_chat(self, **kwargs):
        from openbrowser.llm.browser_use.chat import ChatBrowserUse

        defaults = {"model": "bu-1-0", "api_key": "test-key"}
        defaults.update(kwargs)
        return ChatBrowserUse(**defaults)

    def test_init_valid(self):
        chat = self._make_chat()
        assert chat.model == "bu-1-0"

    def test_init_latest(self):
        chat = self._make_chat(model="bu-latest")
        assert chat.model == "bu-1-0"

    def test_init_invalid_model(self):
        from openbrowser.llm.browser_use.chat import ChatBrowserUse

        with pytest.raises(ValueError, match="Invalid model"):
            ChatBrowserUse(model="invalid", api_key="key")

    def test_init_no_api_key(self):
        from openbrowser.llm.browser_use.chat import ChatBrowserUse

        with patch.dict(os.environ, {}, clear=True):
            with patch("os.getenv", return_value=None):
                with pytest.raises(ValueError, match="BROWSER_USE_API_KEY"):
                    ChatBrowserUse(model="bu-1-0")

    def test_provider(self):
        chat = self._make_chat()
        assert chat.provider == "browser-use"

    def test_name(self):
        chat = self._make_chat()
        assert chat.name == "bu-1-0"

    @pytest.mark.asyncio
    async def test_ainvoke_text(self):
        chat = self._make_chat()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "completion": "text response",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_cached_tokens": None,
                "prompt_cache_creation_tokens": None,
                "prompt_image_tokens": None,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await chat.ainvoke([UserMessage(content="hi")])

        assert result.completion == "text response"

    @pytest.mark.asyncio
    async def test_ainvoke_structured(self):
        from pydantic import BaseModel as PydanticBaseModel

        class MyOutput(PydanticBaseModel):
            field: str

        chat = self._make_chat()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "completion": {"field": "value"},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await chat.ainvoke([UserMessage(content="extract")], output_format=MyOutput)

        assert isinstance(result.completion, MyOutput)

    @pytest.mark.asyncio
    async def test_ainvoke_http_401_error(self):
        import httpx

        chat = self._make_chat()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "bad key"}
        error = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=error)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Invalid API key"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_http_402_error(self):
        import httpx

        chat = self._make_chat()

        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.json.return_value = {"detail": "no credits"}
        error = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=error)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Insufficient credits"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_http_500_error(self):
        import httpx

        chat = self._make_chat()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("not json")
        error = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_response)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=error)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="API request failed"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_timeout_error(self):
        import httpx

        chat = self._make_chat()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="timed out"):
                await chat.ainvoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_ainvoke_generic_error(self):
        chat = self._make_chat()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=ConnectionError("connection failed"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Failed to connect"):
                await chat.ainvoke([UserMessage(content="hi")])

    def test_serialize_message(self):
        chat = self._make_chat()
        msg = UserMessage(content="hello")
        result = chat._serialize_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"
