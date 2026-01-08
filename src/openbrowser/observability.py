"""Observability module for browser-use.

This module provides observability decorators that optionally integrate with lmnr (Laminar) for tracing.
If lmnr is not installed, it provides no-op wrappers that accept the same parameters.

Features:
- Optional lmnr integration - works with or without lmnr installed
- Debug mode support - observe_debug only traces when in debug mode
- Full parameter compatibility with lmnr observe decorator
- No-op fallbacks when lmnr is unavailable

Performance optimizations:
- Cached debug mode check (only computed once at module load)
- Cached lmnr availability check
- Optimized no-op decorator (identity function when possible)
- Module-level constants for fast access
"""

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, TypeVar, cast

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

# Type definitions
F = TypeVar('F', bound=Callable[..., Any])


# Cache debug mode check at module load time (environment rarely changes)
def _compute_debug_mode() -> bool:
	"""Compute debug mode once at module load."""
	return os.getenv('LMNR_LOGGING_LEVEL', '').lower() == 'debug'


_DEBUG_MODE: bool = _compute_debug_mode()


def _is_debug_mode() -> bool:
	"""Check if we're in debug mode (cached)."""
	return _DEBUG_MODE


# Try to import lmnr observe - cache result at module load
_LMNR_AVAILABLE: bool = False
_lmnr_observe: Callable | None = None

try:
	from lmnr import observe as _lmnr_observe  # type: ignore

	if os.environ.get('OPENBROWSER_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
		logger.debug('Lmnr is available for observability')
	_LMNR_AVAILABLE = True
except ImportError:
	if os.environ.get('OPENBROWSER_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
		logger.debug('Lmnr is not available for observability')
	_LMNR_AVAILABLE = False


def _identity_decorator(func: F) -> F:
	"""Identity decorator - returns function unchanged (zero overhead)."""
	return func


def _create_no_op_decorator(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	**kwargs: Any,
) -> Callable[[F], F]:
	"""Create a no-op decorator that accepts all lmnr observe parameters but does nothing.
	
	Optimized: For sync functions, returns identity. For async, minimal wrapper.
	"""
	import asyncio

	def decorator(func: F) -> F:
		# For async functions, we need a wrapper to preserve async behavior
		if asyncio.iscoroutinefunction(func):
			@wraps(func)
			async def async_wrapper(*args, **kwargs):
				return await func(*args, **kwargs)
			return cast(F, async_wrapper)
		else:
			# For sync functions, just return the function unchanged (zero overhead)
			return func

	return decorator


def observe(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	span_type: Literal['DEFAULT', 'LLM', 'TOOL'] = 'DEFAULT',
	**kwargs: Any,
) -> Callable[[F], F]:
	"""
	Observability decorator that traces function execution when lmnr is available.

	This decorator will use lmnr's observe decorator if lmnr is installed,
	otherwise it will be a no-op that accepts the same parameters.

	Args:
	    name: Name of the span/trace
	    ignore_input: Whether to ignore function input parameters in tracing
	    ignore_output: Whether to ignore function output in tracing
	    metadata: Additional metadata to attach to the span
	    **kwargs: Additional parameters passed to lmnr observe

	Returns:
	    Decorated function that may be traced depending on lmnr availability

	Example:
	    @observe(name="my_function", metadata={"version": "1.0"})
	    def my_function(param1, param2):
	        return param1 + param2
	"""
	# Fast path: if lmnr not available, return identity decorator
	if not _LMNR_AVAILABLE or _lmnr_observe is None:
		return _create_no_op_decorator(name=name, ignore_input=ignore_input, ignore_output=ignore_output, metadata=metadata)

	# lmnr is available - use real decorator
	observe_kwargs = {
		'name': name,
		'ignore_input': ignore_input,
		'ignore_output': ignore_output,
		'metadata': metadata,
		'span_type': span_type,
		'tags': ['observe', 'observe_debug'],  # important: tags need to be created on laminar first
		**kwargs,
	}
	return cast(Callable[[F], F], _lmnr_observe(**observe_kwargs))


def observe_debug(
	name: str | None = None,
	ignore_input: bool = False,
	ignore_output: bool = False,
	metadata: dict[str, Any] | None = None,
	span_type: Literal['DEFAULT', 'LLM', 'TOOL'] = 'DEFAULT',
	**kwargs: Any,
) -> Callable[[F], F]:
	"""
	Debug-only observability decorator that only traces when in debug mode.

	This decorator will use lmnr's observe decorator if both lmnr is installed
	AND we're in debug mode, otherwise it will be a no-op.

	Debug mode is determined by:
	- LMNR_LOGGING_LEVEL environment variable set to 'debug'

	Args:
	    name: Name of the span/trace
	    ignore_input: Whether to ignore function input parameters in tracing
	    ignore_output: Whether to ignore function output in tracing
	    metadata: Additional metadata to attach to the span
	    **kwargs: Additional parameters passed to lmnr observe

	Returns:
	    Decorated function that may be traced only in debug mode

	Example:
	    @observe_debug(ignore_input=True, ignore_output=True, name="debug_function", metadata={"debug": True})
	    def debug_function(param1, param2):
	        return param1 + param2
	"""
	# Fast path: if lmnr not available or not in debug mode, return identity decorator
	if not _LMNR_AVAILABLE or _lmnr_observe is None or not _DEBUG_MODE:
		return _create_no_op_decorator(name=name, ignore_input=ignore_input, ignore_output=ignore_output, metadata=metadata)

	# lmnr is available and debug mode is on - use real decorator
	observe_kwargs = {
		'name': name,
		'ignore_input': ignore_input,
		'ignore_output': ignore_output,
		'metadata': metadata,
		'span_type': span_type,
		'tags': ['observe_debug'],  # important: tags need to be created on laminar first
		**kwargs,
	}
	return cast(Callable[[F], F], _lmnr_observe(**observe_kwargs))


# Convenience functions for checking availability and debug status
def is_lmnr_available() -> bool:
	"""Check if lmnr is available for tracing."""
	return _LMNR_AVAILABLE


def is_debug_mode() -> bool:
	"""Check if we're currently in debug mode."""
	return _DEBUG_MODE


def get_observability_status() -> dict[str, bool]:
	"""Get the current status of observability features."""
	return {
		'lmnr_available': _LMNR_AVAILABLE,
		'debug_mode': _DEBUG_MODE,
		'observe_active': _LMNR_AVAILABLE,
		'observe_debug_active': _LMNR_AVAILABLE and _DEBUG_MODE,
	}
