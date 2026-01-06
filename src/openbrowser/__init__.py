"""OpenBrowser - A browser automation framework using CDP and LangGraph."""

from typing import TYPE_CHECKING

__version__ = "0.1.2"

# Core browser components - always available
from openbrowser.browser.profile import BrowserProfile
from openbrowser.browser.session import BrowserSession

# Browser alias for cleaner API (matching browser-use pattern)
Browser = BrowserSession

# Agent components
from openbrowser.agent import (
    BrowserAgent,
    AgentState,
    ActionResult,
    AgentBrain,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentStepInfo,
    BrowserStateHistory,
    StepMetadata,
    SystemPrompt,
    AgentMessagePrompt,
    MessageManager,
    MessageManagerState,
    DEFAULT_INCLUDE_ATTRIBUTES,
)

# Tools
from openbrowser.tools.actions import Tools

# Controller alias for backward compatibility
Controller = Tools

# DOM Service
from openbrowser.browser.dom import DomService

# Lazy imports for heavier modules
if TYPE_CHECKING:
    from openbrowser.llm import (
        BaseChatModel,
        ChatAnthropic,
        ChatAWSBedrock,
        ChatAzureOpenAI,
        ChatBrowserUse,
        ChatCerebras,
        ChatDeepSeek,
        ChatGoogle,
        ChatGroq,
        ChatOCI,
        ChatOllama,
        ChatOpenAI,
        ChatOpenRouter,
        get_llm_by_name,
    )
    from openbrowser.code_use import CodeAgent, create_namespace
    from openbrowser.filesystem import FileSystem
    from openbrowser.tokens import TokenCost
    from openbrowser.screenshots import ScreenshotService
    from openbrowser.telemetry import ProductTelemetry

_LAZY_IMPORTS = {
    # LLM
    'BaseChatModel': ('openbrowser.llm', 'BaseChatModel'),
    'ChatAnthropic': ('openbrowser.llm', 'ChatAnthropic'),
    'ChatAWSBedrock': ('openbrowser.llm', 'ChatAWSBedrock'),
    'ChatAzureOpenAI': ('openbrowser.llm', 'ChatAzureOpenAI'),
    'ChatBrowserUse': ('openbrowser.llm', 'ChatBrowserUse'),
    'ChatCerebras': ('openbrowser.llm', 'ChatCerebras'),
    'ChatDeepSeek': ('openbrowser.llm', 'ChatDeepSeek'),
    'ChatGoogle': ('openbrowser.llm', 'ChatGoogle'),
    'ChatGroq': ('openbrowser.llm', 'ChatGroq'),
    'ChatOCI': ('openbrowser.llm', 'ChatOCI'),
    'ChatOllama': ('openbrowser.llm', 'ChatOllama'),
    'ChatOpenAI': ('openbrowser.llm', 'ChatOpenAI'),
    'ChatOpenRouter': ('openbrowser.llm', 'ChatOpenRouter'),
    'get_llm_by_name': ('openbrowser.llm', 'get_llm_by_name'),
    # Code use
    'CodeAgent': ('openbrowser.code_use', 'CodeAgent'),
    'create_namespace': ('openbrowser.code_use', 'create_namespace'),
    # Filesystem
    'FileSystem': ('openbrowser.filesystem', 'FileSystem'),
    # Tokens
    'TokenCost': ('openbrowser.tokens', 'TokenCost'),
    # Screenshots
    'ScreenshotService': ('openbrowser.screenshots', 'ScreenshotService'),
    # Telemetry
    'ProductTelemetry': ('openbrowser.telemetry', 'ProductTelemetry'),
}


def __getattr__(name: str):
    """Lazy import mechanism for heavy modules.
    
    This function enables lazy loading of resource-intensive modules like LLM
    providers, code execution tools, and telemetry. Modules are only imported
    when first accessed, reducing startup time and memory usage.
    
    Args:
        name: The name of the attribute being accessed.
        
    Returns:
        The requested module attribute.
        
    Raises:
        AttributeError: If the attribute is not found in the module.
        
    Example:
        >>> from openbrowser import ChatOpenAI  # Only imports when accessed
        >>> llm = ChatOpenAI(model="gpt-4o")
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Browser
    "BrowserProfile",
    "BrowserSession",
    "Browser",  # Alias for BrowserSession
    # Agent
    "BrowserAgent",
    "AgentState",
    "ActionResult",
    "AgentBrain",
    "AgentError",
    "AgentHistory",
    "AgentHistoryList",
    "AgentOutput",
    "AgentSettings",
    "AgentStepInfo",
    "BrowserStateHistory",
    "StepMetadata",
    "SystemPrompt",
    "AgentMessagePrompt",
    "MessageManager",
    "MessageManagerState",
    "DEFAULT_INCLUDE_ATTRIBUTES",
    # Tools
    "Tools",
    "Controller",  # Alias for Tools
    # DOM
    "DomService",
    # LLM (lazy)
    "BaseChatModel",
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
    "get_llm_by_name",
    # Code use (lazy)
    "CodeAgent",
    "create_namespace",
    # Filesystem (lazy)
    "FileSystem",
    # Tokens (lazy)
    "TokenCost",
    # Screenshots (lazy)
    "ScreenshotService",
    # Telemetry (lazy)
    "ProductTelemetry",
]
