"""OpenBrowser - Browser automation with AI agents.

Performance optimizations:
- Lazy imports with caching (imports only when accessed)
- Optimized __getattr__ with dict lookup instead of repeated conditionals
- Minimal module-level imports
- Event loop error handling installed when a browser session loads
- CLI fast paths skip logging setup, asyncio, and configuration imports
"""

import os
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

_CLI_PROGRAM_NAMES = frozenset({'openbrowser-ai', 'openbrowser', 'cli.py', 'cli_entry.py'})


class _LazyLogger:
	"""Logger-shaped object that imports logging on first use."""

	name = 'openbrowser'

	def __getattr__(self, item: str):
		import logging

		return getattr(logging.getLogger(self.name), item)


def _is_openbrowser_cli(argv: list[str]) -> bool:
	program = os.path.basename(argv[0] or '') if argv else ''
	return program in _CLI_PROGRAM_NAMES or program.startswith('openbrowser-ai')


def _should_setup_logging(argv: list[str], env: Mapping[str, str]) -> bool:
	"""Return False for an explicit CLI fast path that owns its output.

	Library imports and Click commands keep the existing logging setup. Only a
	recognized OpenBrowser CLI invocation in an explicit fast mode skips it.
	"""
	if env.get('OPENBROWSER_SETUP_LOGGING', 'true').lower() == 'false':
		return False
	if not _is_openbrowser_cli(argv):
		return True
	from openbrowser.cli_modes import select_cli_mode

	return select_cli_mode(argv[1:]) is None


logger = _LazyLogger()

# Only set up logging if this is not an explicit CLI fast path.
if _should_setup_logging(sys.argv, os.environ):
	try:
		from openbrowser.config import CONFIG
		from openbrowser.logging_config import setup_logging

		debug_log_file = getattr(CONFIG, 'OPENBROWSER_DEBUG_LOG_FILE', None)
		info_log_file = getattr(CONFIG, 'OPENBROWSER_INFO_LOG_FILE', None)

		logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
	except Exception as e:
		import warnings
		warnings.warn(f'Failed to set up openbrowser logging: {e}', stacklevel=2)

# Event loop error handling is installed when a browser session loads. Keeping
# asyncio out of this module keeps the CLI warm path to the standard library.
_original_del = None
_asyncio_guard_installed = False


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without noisy errors."""
	global _original_del
	original = _original_del
	if original is None:
		from asyncio import base_subprocess

		original = base_subprocess.BaseSubprocessTransport.__del__
		_original_del = original
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		original(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


def _install_asyncio_del_guard() -> None:
	"""Install the subprocess transport guard once, after asyncio is loaded."""
	global _original_del, _asyncio_guard_installed
	if _asyncio_guard_installed:
		return
	from asyncio import base_subprocess

	_original_del = base_subprocess.BaseSubprocessTransport.__del__
	base_subprocess.BaseSubprocessTransport.__del__ = _patched_del
	_asyncio_guard_installed = True


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from openbrowser.agent.prompts import SystemPrompt
	from openbrowser.agent.service import Agent
	from openbrowser.agent.service import Agent as BrowserAgent  # Alias

	# from openbrowser.agent.service import Agent
	from openbrowser.agent.views import (
		AgentHistoryList,
		AgentRunInterrupted,
		AgentRunPhase,
		AgentRunStateError,
		AgentRunStatus,
	)
	from openbrowser.browser import BrowserProfile, BrowserSession
	from openbrowser.browser import BrowserSession as Browser
	from openbrowser.code_use.service import CodeAgent, CodeAgentStateError
	from openbrowser.code_use.views import CodeAgentPhase, CodeAgentRunStatus
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
	from openbrowser.models import ActionResult
	from openbrowser.tools.registry.views import ActionModel
	from openbrowser.tools.service import Controller, Tools


# Lazy imports mapping - only import when actually accessed
# Format: name -> (module_path, attr_name) where attr_name=None means return module itself
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
	# Agent service (heavy due to dependencies)
	'Agent': ('openbrowser.agent.service', 'Agent'),
	'BrowserAgent': ('openbrowser.agent.service', 'Agent'),  # Alias for backward compatibility
	'AgentRunPhase': ('openbrowser.agent.views', 'AgentRunPhase'),
	'AgentRunStatus': ('openbrowser.agent.views', 'AgentRunStatus'),
	'AgentRunStateError': ('openbrowser.agent.views', 'AgentRunStateError'),
	'AgentRunInterrupted': ('openbrowser.agent.views', 'AgentRunInterrupted'),
	# Code-use agent (Jupyter notebook-like execution)
	'CodeAgent': ('openbrowser.code_use.service', 'CodeAgent'),
	'CodeAgentStateError': ('openbrowser.code_use.service', 'CodeAgentStateError'),
	'CodeAgentPhase': ('openbrowser.code_use.views', 'CodeAgentPhase'),
	'CodeAgentRunStatus': ('openbrowser.code_use.views', 'CodeAgentRunStatus'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('openbrowser.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('openbrowser.tools.registry.views', 'ActionModel'),
	'ActionResult': ('openbrowser.models', 'ActionResult'),
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
	'AgentRunPhase',
	'AgentRunStatus',
	'AgentRunStateError',
	'AgentRunInterrupted',
	'CodeAgent',
	'CodeAgentPhase',
	'CodeAgentRunStatus',
	'CodeAgentStateError',
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
