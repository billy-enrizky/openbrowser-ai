"""OpenBrowser - Browser automation with AI agents.

Performance optimizations:
- Lazy imports with caching (imports only when accessed)
- Optimized __getattr__ with dict lookup instead of repeated conditionals
- Minimal module-level imports
- Event loop error handling patched at module load
"""

import os
from typing import TYPE_CHECKING

from openbrowser.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('OPENBROWSER_SETUP_LOGGING', 'true').lower() != 'false':
	from openbrowser.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'OPENBROWSER_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'OPENBROWSER_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('openbrowser')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from openbrowser.agent.prompts import SystemPrompt
	from openbrowser.agent.service import Agent
	from openbrowser.agent.service import Agent as BrowserAgent  # Alias

	# from openbrowser.agent.service import Agent
	from openbrowser.agent.views import ActionModel, ActionResult, AgentHistoryList
	from openbrowser.browser import BrowserProfile, BrowserSession
	from openbrowser.browser import BrowserSession as Browser
	from openbrowser.code_use.service import CodeAgent
	from openbrowser.dom.service import DomService
	from openbrowser.llm import models
	from openbrowser.llm.anthropic.chat import ChatAnthropic
	from openbrowser.llm.azure.chat import ChatAzureOpenAI
	from openbrowser.llm.browser_use.chat import ChatBrowserUse
	from openbrowser.llm.google.chat import ChatGoogle
	from openbrowser.llm.groq.chat import ChatGroq
	from openbrowser.llm.oci_raw.chat import ChatOCIRaw
	from openbrowser.llm.ollama.chat import ChatOllama
	from openbrowser.llm.openai.chat import ChatOpenAI
	from openbrowser.tools.service import Controller, Tools


# Lazy imports mapping - only import when actually accessed
# Format: name -> (module_path, attr_name) where attr_name=None means return module itself
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
	# Agent service (heavy due to dependencies)
	'Agent': ('openbrowser.agent.service', 'Agent'),
	'BrowserAgent': ('openbrowser.agent.service', 'Agent'),  # Alias for backward compatibility
	# Code-use agent (Jupyter notebook-like execution)
	'CodeAgent': ('openbrowser.code_use.service', 'CodeAgent'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('openbrowser.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('openbrowser.agent.views', 'ActionModel'),
	'ActionResult': ('openbrowser.agent.views', 'ActionResult'),
	'AgentHistoryList': ('openbrowser.agent.views', 'AgentHistoryList'),
	# Browser session
	'BrowserSession': ('openbrowser.browser', 'BrowserSession'),
	'Browser': ('openbrowser.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('openbrowser.browser', 'BrowserProfile'),
	# Tools (moderate weight)
	'Tools': ('openbrowser.tools.service', 'Tools'),
	'Controller': ('openbrowser.tools.service', 'Controller'),  # alias
	# DOM service (moderate weight)
	'DomService': ('openbrowser.dom.service', 'DomService'),
	# Chat models (very heavy imports)
	'ChatOpenAI': ('openbrowser.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('openbrowser.llm.google.chat', 'ChatGoogle'),
	'ChatAnthropic': ('openbrowser.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatBrowserUse': ('openbrowser.llm.browser_use.chat', 'ChatBrowserUse'),
	'ChatGroq': ('openbrowser.llm.groq.chat', 'ChatGroq'),
	'ChatAzureOpenAI': ('openbrowser.llm.azure.chat', 'ChatAzureOpenAI'),
	'ChatOCIRaw': ('openbrowser.llm.oci_raw.chat', 'ChatOCIRaw'),
	'ChatOllama': ('openbrowser.llm.ollama.chat', 'ChatOllama'),
	# LLM models module
	'models': ('openbrowser.llm.models', None),
}

# Cache for imported modules/attributes
_import_cache: dict[str, object] = {}


def __getattr__(name: str):
	"""Lazy import mechanism - only import modules when they're actually accessed.
	
	Optimized with caching to avoid repeated imports.
	"""
	# Check cache first (fast path)
	if name in _import_cache:
		return _import_cache[name]
	
	# Check if it's a lazy import
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			if attr_name is None:
				# For modules like 'models', return the module itself
				attr = module
			else:
				attr = getattr(module, attr_name)
			# Cache the imported attribute
			_import_cache[name] = attr
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'BrowserAgent',  # Alias for Agent
	'CodeAgent',
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	# Chat models
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatBrowserUse',
	'ChatGroq',
	'ChatAzureOpenAI',
	'ChatOCIRaw',
	'ChatOllama',
	'Tools',
	'Controller',
	# LLM models module
	'models',
]
