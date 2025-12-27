"""
Observability module for openbrowser.

This module provides observability decorators that optionally integrate with lmnr (Laminar) for tracing.
If lmnr is not installed, it provides no-op wrappers that accept the same parameters.

Features:
- Optional lmnr integration - works with or without lmnr installed
- Debug mode support - observe_debug only traces when in debug mode
- Full parameter compatibility with lmnr observe decorator
- No-op fallbacks when lmnr is unavailable
"""

import asyncio
import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, TypeVar, cast

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Type definitions
F = TypeVar('F', bound=Callable[..., Any])


def _is_debug_mode() -> bool:
    """Check if we're in debug mode based on environment variables or logging level."""
    lmnr_debug_mode = os.getenv('LMNR_LOGGING_LEVEL', '').lower()
    if lmnr_debug_mode == 'debug':
        return True
    
    openbrowser_debug = os.getenv('OPENBROWSER_DEBUG', '').lower()
    if openbrowser_debug in ('1', 'true', 'yes', 'on'):
        return True
    
    # Check root logger level
    if logging.root.level <= logging.DEBUG:
        return True
    
    return False


# Try to import lmnr observe
_LMNR_AVAILABLE = False
_lmnr_observe = None

try:
    from lmnr import observe as _lmnr_observe  # type: ignore

    if os.environ.get('OPENBROWSER_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
        logger.debug('Lmnr is available for observability')
    _LMNR_AVAILABLE = True
except ImportError:
    if os.environ.get('OPENBROWSER_VERBOSE_OBSERVABILITY', 'false').lower() == 'true':
        logger.debug('Lmnr is not available for observability')
    _LMNR_AVAILABLE = False


def _create_no_op_decorator(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Create a no-op decorator that accepts all lmnr observe parameters but does nothing."""

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return cast(F, sync_wrapper)

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
        span_type: Type of span (DEFAULT, LLM, TOOL)
        **kwargs: Additional parameters passed to lmnr observe

    Returns:
        Decorated function that may be traced depending on lmnr availability

    Example:
        @observe(name="my_function", metadata={"version": "1.0"})
        def my_function(param1, param2):
            return param1 + param2
    """
    observe_kwargs = {
        'name': name,
        'ignore_input': ignore_input,
        'ignore_output': ignore_output,
        'metadata': metadata,
        'span_type': span_type,
        'tags': ['observe'],
        **kwargs,
    }

    if _LMNR_AVAILABLE and _lmnr_observe:
        return cast(Callable[[F], F], _lmnr_observe(**observe_kwargs))
    else:
        return _create_no_op_decorator(**observe_kwargs)


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
    - LMNR_LOGGING_LEVEL environment variable set to debug
    - OPENBROWSER_DEBUG environment variable set to 1/true/yes/on
    - Root logging level set to DEBUG or lower

    Args:
        name: Name of the span/trace
        ignore_input: Whether to ignore function input parameters in tracing
        ignore_output: Whether to ignore function output in tracing
        metadata: Additional metadata to attach to the span
        span_type: Type of span (DEFAULT, LLM, TOOL)
        **kwargs: Additional parameters passed to lmnr observe

    Returns:
        Decorated function that may be traced only in debug mode

    Example:
        @observe_debug(ignore_input=True, ignore_output=True, name="debug_function")
        def debug_function(param1, param2):
            return param1 + param2
    """
    observe_kwargs = {
        'name': name,
        'ignore_input': ignore_input,
        'ignore_output': ignore_output,
        'metadata': metadata,
        'span_type': span_type,
        'tags': ['observe_debug'],
        **kwargs,
    }

    if _LMNR_AVAILABLE and _lmnr_observe and _is_debug_mode():
        return cast(Callable[[F], F], _lmnr_observe(**observe_kwargs))
    else:
        return _create_no_op_decorator(**observe_kwargs)


# Convenience functions for checking availability and debug status
def is_lmnr_available() -> bool:
    """Check if lmnr is available for tracing."""
    return _LMNR_AVAILABLE


def is_debug_mode() -> bool:
    """Check if we're currently in debug mode."""
    return _is_debug_mode()


def get_observability_status() -> dict[str, bool]:
    """Get the current status of observability features."""
    return {
        'lmnr_available': _LMNR_AVAILABLE,
        'debug_mode': _is_debug_mode(),
        'observe_active': _LMNR_AVAILABLE,
        'observe_debug_active': _LMNR_AVAILABLE and _is_debug_mode(),
    }

