"""OpenBrowser - A browser automation framework using CDP and LangGraph."""

from typing import TYPE_CHECKING

__version__ = "0.1.68"

# Core browser components - always available
from src.openbrowser.browser.profile import BrowserProfile
from src.openbrowser.browser.session import BrowserSession

# Browser alias for cleaner API (matching browser-use pattern)
Browser = BrowserSession

# Agent components
from src.openbrowser.agent import (
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
from src.openbrowser.tools.actions import Tools

# Controller alias for backward compatibility
Controller = Tools

# DOM Service
from src.openbrowser.browser.dom import DomService

# Lazy imports for heavier modules
if TYPE_CHECKING:
    from src.openbrowser.llm import (
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
    from src.openbrowser.code_use import CodeAgent, create_namespace
    from src.openbrowser.filesystem import FileSystem
    from src.openbrowser.tokens import TokenCost
    from src.openbrowser.screenshots import ScreenshotService
    from src.openbrowser.telemetry import ProductTelemetry

_LAZY_IMPORTS = {
    # LLM
    'BaseChatModel': ('src.openbrowser.llm', 'BaseChatModel'),
    'ChatAnthropic': ('src.openbrowser.llm', 'ChatAnthropic'),
    'ChatAWSBedrock': ('src.openbrowser.llm', 'ChatAWSBedrock'),
    'ChatAzureOpenAI': ('src.openbrowser.llm', 'ChatAzureOpenAI'),
    'ChatBrowserUse': ('src.openbrowser.llm', 'ChatBrowserUse'),
    'ChatCerebras': ('src.openbrowser.llm', 'ChatCerebras'),
    'ChatDeepSeek': ('src.openbrowser.llm', 'ChatDeepSeek'),
    'ChatGoogle': ('src.openbrowser.llm', 'ChatGoogle'),
    'ChatGroq': ('src.openbrowser.llm', 'ChatGroq'),
    'ChatOCI': ('src.openbrowser.llm', 'ChatOCI'),
    'ChatOllama': ('src.openbrowser.llm', 'ChatOllama'),
    'ChatOpenAI': ('src.openbrowser.llm', 'ChatOpenAI'),
    'ChatOpenRouter': ('src.openbrowser.llm', 'ChatOpenRouter'),
    'get_llm_by_name': ('src.openbrowser.llm', 'get_llm_by_name'),
    # Code use
    'CodeAgent': ('src.openbrowser.code_use', 'CodeAgent'),
    'create_namespace': ('src.openbrowser.code_use', 'create_namespace'),
    # Filesystem
    'FileSystem': ('src.openbrowser.filesystem', 'FileSystem'),
    # Tokens
    'TokenCost': ('src.openbrowser.tokens', 'TokenCost'),
    # Screenshots
    'ScreenshotService': ('src.openbrowser.screenshots', 'ScreenshotService'),
    # Telemetry
    'ProductTelemetry': ('src.openbrowser.telemetry', 'ProductTelemetry'),
}


def __getattr__(name: str):
    """Lazy import mechanism for heavy modules."""
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
