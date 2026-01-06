"""LLM integrations for OpenBrowser.

This module provides a unified interface for interacting with various Large Language Model
(LLM) providers. It includes support for OpenAI, Anthropic, Google, Groq, AWS Bedrock,
Azure OpenAI, Ollama, OpenRouter, OCI, Cerebras, DeepSeek, and browser-use's cloud service.

The module uses lazy loading for provider-specific implementations to minimize import
time and memory usage when only specific providers are needed.

Example:
    >>> from openbrowser.llm import get_llm_by_name
    >>> llm = get_llm_by_name('openai', model='gpt-4o')
    >>> response = await llm.ainvoke([HumanMessage(content="Hello!")])

Attributes:
    BaseChatModel: Abstract base class for all chat models.
    LangChainChatModelWrapper: Wrapper for LangChain chat models.
    LLMException: Generic exception for LLM errors.
    ModelAuthenticationError: Exception for authentication failures.
    ModelRateLimitError: Exception for rate limit errors.
"""

from typing import TYPE_CHECKING

from openbrowser.llm.base import BaseChatModel, LangChainChatModelWrapper
from openbrowser.llm.exceptions import LLMException, ModelAuthenticationError, ModelRateLimitError

# Type stubs for lazy imports
if TYPE_CHECKING:
    from openbrowser.llm.anthropic.chat import ChatAnthropic
    from openbrowser.llm.aws.chat import ChatAWSBedrock
    from openbrowser.llm.azure.chat import ChatAzureOpenAI
    from openbrowser.llm.browser_use.chat import ChatBrowserUse
    from openbrowser.llm.cerebras.chat import ChatCerebras
    from openbrowser.llm.deepseek.chat import ChatDeepSeek
    from openbrowser.llm.google.chat import ChatGoogle
    from openbrowser.llm.groq.chat import ChatGroq
    from openbrowser.llm.oci.chat import ChatOCI
    from openbrowser.llm.ollama.chat import ChatOllama
    from openbrowser.llm.openai.chat import ChatOpenAI
    from openbrowser.llm.openrouter.chat import ChatOpenRouter

# Lazy imports mapping for heavy chat models
_LAZY_IMPORTS = {
    'ChatAnthropic': ('openbrowser.llm.anthropic.chat', 'ChatAnthropic'),
    'ChatAWSBedrock': ('openbrowser.llm.aws.chat', 'ChatAWSBedrock'),
    'ChatAzureOpenAI': ('openbrowser.llm.azure.chat', 'ChatAzureOpenAI'),
    'ChatBrowserUse': ('openbrowser.llm.browser_use.chat', 'ChatBrowserUse'),
    'ChatCerebras': ('openbrowser.llm.cerebras.chat', 'ChatCerebras'),
    'ChatDeepSeek': ('openbrowser.llm.deepseek.chat', 'ChatDeepSeek'),
    'ChatGoogle': ('openbrowser.llm.google.chat', 'ChatGoogle'),
    'ChatGroq': ('openbrowser.llm.groq.chat', 'ChatGroq'),
    'ChatOCI': ('openbrowser.llm.oci.chat', 'ChatOCI'),
    'ChatOllama': ('openbrowser.llm.ollama.chat', 'ChatOllama'),
    'ChatOpenAI': ('openbrowser.llm.openai.chat', 'ChatOpenAI'),
    'ChatOpenRouter': ('openbrowser.llm.openrouter.chat', 'ChatOpenRouter'),
}


def __getattr__(name: str):
    """Lazy import mechanism for heavy chat model imports.
    
    This function enables lazy loading of provider-specific chat model classes.
    When a chat model class is accessed (e.g., `ChatOpenAI`), it is dynamically
    imported from its respective module.
    
    Args:
        name: The name of the attribute being accessed. Should be one of the
            registered chat model class names (e.g., 'ChatOpenAI', 'ChatAnthropic').
    
    Returns:
        The requested chat model class.
    
    Raises:
        AttributeError: If the requested attribute is not a known chat model class.
    
    Example:
        >>> from openbrowser.llm import ChatOpenAI
        >>> llm = ChatOpenAI(model='gpt-4o')
    """
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_llm_by_name(provider: str, model: str | None = None, **kwargs) -> BaseChatModel:
    """Factory function to get an LLM instance by provider name.
    
    This is the recommended way to instantiate LLM providers when the provider
    name is determined at runtime (e.g., from configuration). It handles the
    mapping between provider names and their corresponding chat model classes.
    
    Args:
        provider: The provider name. Supported values are:
            - 'openai': OpenAI GPT models
            - 'anthropic': Anthropic Claude models
            - 'google': Google Gemini models
            - 'groq': Groq-hosted models (fast inference)
            - 'ollama': Local Ollama models
            - 'openrouter': OpenRouter multi-provider gateway
            - 'aws': AWS Bedrock (Claude models)
            - 'azure': Azure OpenAI
            - 'oci': Oracle Cloud Infrastructure GenAI
            - 'cerebras': Cerebras fast inference
            - 'deepseek': DeepSeek models with reasoning
            - 'browser_use': Browser-use cloud service
        model: Optional model name. If not specified, uses the provider's default.
        **kwargs: Additional keyword arguments passed to the LLM constructor.
            Common arguments include:
            - temperature (float): Controls randomness (0.0 to 1.0)
            - max_tokens (int): Maximum tokens in response
            - api_key (str): API key (overrides environment variable)
    
    Returns:
        BaseChatModel: An initialized chat model instance for the specified provider.
    
    Raises:
        ValueError: If the provider name is not recognized.
        ImportError: If the provider's dependencies are not installed.
    
    Example:
        >>> llm = get_llm_by_name('openai', model='gpt-4o', temperature=0.7)
        >>> llm = get_llm_by_name('anthropic')  # Uses default claude model
    """
    provider = provider.lower()
    
    provider_map = {
        'openai': 'ChatOpenAI',
        'google': 'ChatGoogle',
        'anthropic': 'ChatAnthropic',
        'groq': 'ChatGroq',
        'ollama': 'ChatOllama',
        'openrouter': 'ChatOpenRouter',
        'aws': 'ChatAWSBedrock',
        'azure': 'ChatAzureOpenAI',
        'oci': 'ChatOCI',
        'cerebras': 'ChatCerebras',
        'deepseek': 'ChatDeepSeek',
        'browser_use': 'ChatBrowserUse',
    }
    
    if provider not in provider_map:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(provider_map.keys())}")
    
    chat_class = __getattr__(provider_map[provider])
    
    if model:
        return chat_class(model=model, **kwargs)
    return chat_class(**kwargs)


__all__ = [
    # Base classes
    "BaseChatModel",
    "LangChainChatModelWrapper",
    # Exceptions
    "LLMException",
    "ModelAuthenticationError",
    "ModelRateLimitError",
    # Factory function
    "get_llm_by_name",
    # Chat models (lazy loaded)
    "ChatAnthropic",
    "ChatAWSBedrock",
    "ChatAzureOpenAI",
    "ChatBrowserUse",
    "ChatCerebras",
    "ChatDeepSeek",
    "ChatGoogle",
    "ChatGroq",
    "ChatOCI",
    "ChatOllama",
    "ChatOpenAI",
    "ChatOpenRouter",
]
