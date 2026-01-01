"""LLM integrations for OpenBrowser."""

from typing import TYPE_CHECKING

from src.openbrowser.llm.base import BaseChatModel, LangChainChatModelWrapper
from src.openbrowser.llm.exceptions import LLMException, ModelAuthenticationError, ModelRateLimitError

# Type stubs for lazy imports
if TYPE_CHECKING:
    from src.openbrowser.llm.anthropic.chat import ChatAnthropic
    from src.openbrowser.llm.aws.chat import ChatAWSBedrock
    from src.openbrowser.llm.azure.chat import ChatAzureOpenAI
    from src.openbrowser.llm.browser_use.chat import ChatBrowserUse
    from src.openbrowser.llm.cerebras.chat import ChatCerebras
    from src.openbrowser.llm.deepseek.chat import ChatDeepSeek
    from src.openbrowser.llm.google.chat import ChatGoogle
    from src.openbrowser.llm.groq.chat import ChatGroq
    from src.openbrowser.llm.oci.chat import ChatOCI
    from src.openbrowser.llm.ollama.chat import ChatOllama
    from src.openbrowser.llm.openai.chat import ChatOpenAI
    from src.openbrowser.llm.openrouter.chat import ChatOpenRouter

# Lazy imports mapping for heavy chat models
_LAZY_IMPORTS = {
    'ChatAnthropic': ('src.openbrowser.llm.anthropic.chat', 'ChatAnthropic'),
    'ChatAWSBedrock': ('src.openbrowser.llm.aws.chat', 'ChatAWSBedrock'),
    'ChatAzureOpenAI': ('src.openbrowser.llm.azure.chat', 'ChatAzureOpenAI'),
    'ChatBrowserUse': ('src.openbrowser.llm.browser_use.chat', 'ChatBrowserUse'),
    'ChatCerebras': ('src.openbrowser.llm.cerebras.chat', 'ChatCerebras'),
    'ChatDeepSeek': ('src.openbrowser.llm.deepseek.chat', 'ChatDeepSeek'),
    'ChatGoogle': ('src.openbrowser.llm.google.chat', 'ChatGoogle'),
    'ChatGroq': ('src.openbrowser.llm.groq.chat', 'ChatGroq'),
    'ChatOCI': ('src.openbrowser.llm.oci.chat', 'ChatOCI'),
    'ChatOllama': ('src.openbrowser.llm.ollama.chat', 'ChatOllama'),
    'ChatOpenAI': ('src.openbrowser.llm.openai.chat', 'ChatOpenAI'),
    'ChatOpenRouter': ('src.openbrowser.llm.openrouter.chat', 'ChatOpenRouter'),
}


def __getattr__(name: str):
    """Lazy import mechanism for heavy chat model imports."""
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_llm_by_name(provider: str, model: str | None = None, **kwargs) -> BaseChatModel:
    """
    Factory function to get an LLM instance by provider name.
    
    Args:
        provider: Provider name (openai, google, anthropic, groq, ollama, openrouter, aws, azure, oci, cerebras, deepseek, browser_use)
        model: Model name (optional, uses provider default if not specified)
        **kwargs: Additional arguments passed to the LLM constructor
        
    Returns:
        BaseChatModel instance
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
