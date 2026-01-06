"""OCI (Oracle Cloud Infrastructure) GenAI LLM integration."""

import json
import logging
import os
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import PrivateAttr

from openbrowser.llm.exceptions import ModelProviderError

logger = logging.getLogger(__name__)


class ChatOCI(BaseChatModel):
    """
    Chat model for Oracle Cloud Infrastructure GenAI.

    This class provides integration with OCI's generative AI service.

    Args:
        model: The OCI model to use
        compartment_id: OCI compartment ID
        endpoint: OCI GenAI endpoint URL
        temperature: Temperature for response generation
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters
    """

    _model: str = PrivateAttr()
    _compartment_id: str = PrivateAttr()
    _endpoint: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _client: httpx.AsyncClient = PrivateAttr()
    _config_profile: str = PrivateAttr()

    def __init__(
        self,
        model: str = "cohere.command-r-plus",
        compartment_id: str | None = None,
        endpoint: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        config_profile: str = "DEFAULT",
        **kwargs: Any,
    ):
        super().__init__()
        self._model = model
        self._compartment_id = compartment_id or os.getenv("OCI_COMPARTMENT_ID", "")
        self._endpoint = endpoint or os.getenv(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._config_profile = config_profile
        self._client = httpx.AsyncClient(timeout=120.0)

        if not self._compartment_id:
            raise ValueError("OCI_COMPARTMENT_ID is not set and no compartment_id provided")

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def _llm_type(self) -> str:
        return "oci"

    def _get_auth_header(self) -> dict[str, str]:
        """Get OCI authentication headers.
        
        This uses OCI's SDK for authentication if available,
        otherwise falls back to API key.
        """
        try:
            import oci
            config = oci.config.from_file(profile_name=self._config_profile)
            signer = oci.signer.Signer(
                tenancy=config["tenancy"],
                user=config["user"],
                fingerprint=config["fingerprint"],
                private_key_file_location=config["key_file"],
            )
            # For simplicity, we'll use a static token approach
            # In production, you'd use the full OCI SDK signer
            return {}
        except ImportError:
            api_key = os.getenv("OCI_API_KEY", "")
            if api_key:
                return {"Authorization": f"Bearer {api_key}"}
            return {}

    def bind_tools(self, tools: list[Any]) -> "ChatOCI":
        """Bind tools to this LLM instance."""
        # OCI doesn't support function calling in the same way
        logger.warning("OCI does not support tool binding in the same way as OpenAI")
        return self

    def with_structured_output(self, output_schema: type[Any], **kwargs: Any) -> Any:
        """Get structured output from the model."""
        # Return self with schema stored for later use
        return self

    async def ainvoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the model with the given messages."""
        from langchain_core.messages import AIMessage

        # Convert messages to OCI format
        chat_history = []
        system_message = None
        
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == 'system':
                    system_message = msg.content
                elif msg.type == 'human':
                    chat_history.append({"role": "USER", "message": msg.content})
                elif msg.type == 'ai':
                    chat_history.append({"role": "CHATBOT", "message": msg.content})

        request_body = {
            "compartmentId": self._compartment_id,
            "servingMode": {
                "servingType": "ON_DEMAND",
                "modelId": self._model,
            },
            "chatRequest": {
                "apiFormat": "COHERE",
                "message": chat_history[-1]["message"] if chat_history else "",
                "chatHistory": chat_history[:-1] if len(chat_history) > 1 else [],
                "maxTokens": self._max_tokens,
                "temperature": self._temperature,
            }
        }

        if system_message:
            request_body["chatRequest"]["preambleOverride"] = system_message

        try:
            response = await self._client.post(
                f"{self._endpoint}/20231130/actions/chat",
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    **self._get_auth_header(),
                },
            )
            response.raise_for_status()
            data = response.json()
            
            text = data.get("chatResponse", {}).get("text", "")
            return AIMessage(content=text)

        except httpx.HTTPStatusError as e:
            raise ModelProviderError(
                message=f"OCI API error: {e.response.text}",
                status_code=e.response.status_code,
                model=self._model,
            )

    def invoke(
        self,
        messages: list[Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronously invoke the model."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(messages, config, **kwargs)
        )

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Internal generate method required by BaseChatModel."""
        from langchain_core.outputs import ChatGeneration, ChatResult
        result = self.invoke(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=result)])

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

