"""Comprehensive tests for openbrowser.observability module.

Covers: observe, observe_debug, is_lmnr_available, is_debug_mode,
get_observability_status, _identity_decorator, _create_no_op_decorator,
_compute_debug_mode, _is_debug_mode.
"""

import asyncio
import logging
import os
from unittest.mock import patch

import pytest

from openbrowser.observability import (
    _compute_debug_mode,
    _create_no_op_decorator,
    _identity_decorator,
    _is_debug_mode,
    get_observability_status,
    is_debug_mode,
    is_lmnr_available,
    observe,
    observe_debug,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _compute_debug_mode
# ---------------------------------------------------------------------------


class TestComputeDebugMode:
    def test_debug_when_env_set(self):
        with patch.dict(os.environ, {"LMNR_LOGGING_LEVEL": "debug"}):
            assert _compute_debug_mode() is True

    def test_not_debug_when_env_missing(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LMNR_LOGGING_LEVEL", None)
            assert _compute_debug_mode() is False

    def test_not_debug_when_env_is_info(self):
        with patch.dict(os.environ, {"LMNR_LOGGING_LEVEL": "info"}):
            assert _compute_debug_mode() is False

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"LMNR_LOGGING_LEVEL": "DEBUG"}):
            assert _compute_debug_mode() is True


# ---------------------------------------------------------------------------
# _is_debug_mode
# ---------------------------------------------------------------------------


class TestIsDebugMode:
    def test_returns_cached_value(self):
        result = _is_debug_mode()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _identity_decorator
# ---------------------------------------------------------------------------


class TestIdentityDecorator:
    def test_returns_same_function(self):
        def my_func():
            return 42

        result = _identity_decorator(my_func)
        assert result is my_func
        assert result() == 42


# ---------------------------------------------------------------------------
# _create_no_op_decorator
# ---------------------------------------------------------------------------


class TestCreateNoOpDecorator:
    def test_sync_function_returned_unchanged(self):
        decorator = _create_no_op_decorator(name="test")

        def my_func(x):
            return x * 2

        decorated = decorator(my_func)
        assert decorated is my_func
        assert decorated(5) == 10

    @pytest.mark.asyncio
    async def test_async_function_wrapped(self):
        decorator = _create_no_op_decorator(name="test")

        async def my_func(x):
            return x * 2

        decorated = decorator(my_func)
        assert asyncio.iscoroutinefunction(decorated)
        assert await decorated(5) == 10

    def test_accepts_all_params(self):
        decorator = _create_no_op_decorator(
            name="test",
            ignore_input=True,
            ignore_output=True,
            metadata={"key": "value"},
            extra_kwarg="extra",
        )
        assert callable(decorator)


# ---------------------------------------------------------------------------
# observe
# ---------------------------------------------------------------------------


class TestObserve:
    def test_sync_function_works(self):
        @observe(name="test_func")
        def my_func(x):
            return x + 1

        assert my_func(5) == 6

    @pytest.mark.asyncio
    async def test_async_function_works(self):
        @observe(name="test_func")
        async def my_func(x):
            return x + 1

        assert await my_func(5) == 6

    def test_accepts_all_params(self):
        @observe(
            name="test",
            ignore_input=True,
            ignore_output=True,
            metadata={"version": "1.0"},
            span_type="TOOL",
        )
        def my_func():
            return 42

        assert my_func() == 42

    def test_no_args(self):
        @observe()
        def my_func():
            return 42

        assert my_func() == 42


# ---------------------------------------------------------------------------
# observe_debug
# ---------------------------------------------------------------------------


class TestObserveDebug:
    def test_sync_function_works(self):
        @observe_debug(name="debug_test")
        def my_func(x):
            return x * 3

        assert my_func(2) == 6

    @pytest.mark.asyncio
    async def test_async_function_works(self):
        @observe_debug(name="debug_test")
        async def my_func(x):
            return x * 3

        assert await my_func(2) == 6

    def test_accepts_all_params(self):
        @observe_debug(
            name="debug",
            ignore_input=True,
            ignore_output=True,
            metadata={"debug": True},
            span_type="LLM",
        )
        def my_func():
            return "ok"

        assert my_func() == "ok"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_is_lmnr_available(self):
        result = is_lmnr_available()
        assert isinstance(result, bool)

    def test_is_debug_mode(self):
        result = is_debug_mode()
        assert isinstance(result, bool)

    def test_get_observability_status(self):
        status = get_observability_status()
        assert isinstance(status, dict)
        assert "lmnr_available" in status
        assert "debug_mode" in status
        assert "observe_active" in status
        assert "observe_debug_active" in status
        assert isinstance(status["lmnr_available"], bool)
        assert isinstance(status["debug_mode"], bool)
        assert isinstance(status["observe_active"], bool)
        assert isinstance(status["observe_debug_active"], bool)

    def test_observe_debug_active_requires_both(self):
        status = get_observability_status()
        if status["observe_debug_active"]:
            assert status["lmnr_available"] is True
            assert status["debug_mode"] is True
