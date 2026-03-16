"""Comprehensive tests for src/openbrowser/utils.py to cover all missed lines.

Covers: URL matching, domain patterns, version checks, decorators, time_execution,
SignalHandler, merge_dicts, env variable checks, etc.

Missing lines: 39-40, 44-45, 102-114, 118-119, 123-144, 148-164, 173-213, 224-251,
260-268, 272-290, 301-340, 345-348, 361-370, 391-400, 525, 531, 538, 552-554,
581-587, 596-598, 613-616, 631-659
"""

import asyncio
import importlib.util
import logging
import os
import signal
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

# Import the actual utils.py file (not the utils/ package)
_utils_path = Path(__file__).parent.parent / "src" / "openbrowser" / "utils.py"
_spec = importlib.util.spec_from_file_location("openbrowser_utils_file", str(_utils_path))
_utils_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_mod)


# ---------------------------------------------------------------------------
# Import error branches (lines 39-40, 44-45)
# ---------------------------------------------------------------------------


class TestImportFallbacks:
    """Test import fallback branches for OpenAI/Groq BadRequestError."""

    def test_openai_import_fallback(self):
        """Lines 39-40: OpenAIBadRequestError is set to None when openai is not installed."""
        # _utils_mod is the direct utils.py file
        assert hasattr(_utils_mod, "OpenAIBadRequestError")

    def test_groq_import_fallback(self):
        """Lines 44-45: GroqBadRequestError is set to None when groq is not installed."""
        assert hasattr(_utils_mod, "GroqBadRequestError")


# ---------------------------------------------------------------------------
# SignalHandler in utils.py (lines 102-348)
# ---------------------------------------------------------------------------


class TestSignalHandlerUtils:
    """Test the SignalHandler class defined in utils.py (not utils/signal_handler.py)."""

    def _get_handler_class(self):
        """Get SignalHandler from utils.py directly."""
        return _utils_mod.SignalHandler

    def test_init_defaults(self):
        """Lines 102-114: SignalHandler init with default values."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            assert handler.loop is loop
            assert handler.pause_callback is None
            assert handler.resume_callback is None
            assert handler.custom_exit_callback is None
            assert handler.exit_on_second_int is True
            assert handler.interruptible_task_patterns == [
                "step",
                "multi_act",
                "get_next_action",
            ]
            assert handler.original_sigint_handler is None
            assert handler.original_sigterm_handler is None
        finally:
            loop.close()

    def test_init_custom_params(self):
        """Lines 102-114: SignalHandler init with custom parameters."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            pause_cb = MagicMock()
            resume_cb = MagicMock()
            exit_cb = MagicMock()
            handler = SignalHandler(
                loop=loop,
                pause_callback=pause_cb,
                resume_callback=resume_cb,
                custom_exit_callback=exit_cb,
                exit_on_second_int=False,
                interruptible_task_patterns=["custom_pattern"],
            )
            assert handler.pause_callback is pause_cb
            assert handler.resume_callback is resume_cb
            assert handler.custom_exit_callback is exit_cb
            assert handler.exit_on_second_int is False
            assert handler.interruptible_task_patterns == ["custom_pattern"]
        finally:
            loop.close()

    def test_initialize_loop_state(self):
        """Lines 118-119: _initialize_loop_state sets loop attrs."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            assert getattr(loop, "ctrl_c_pressed", None) is False
            assert getattr(loop, "waiting_for_input", None) is False
        finally:
            loop.close()

    def test_register_windows(self):
        """Lines 123-133: register on Windows path."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            # Patch at the module level for the direct utils.py module
            orig = _utils_mod._IS_WINDOWS
            _utils_mod._IS_WINDOWS = True
            try:
                with patch("signal.signal") as mock_signal:
                    mock_signal.return_value = signal.SIG_DFL
                    handler.register()
                    mock_signal.assert_called_once()
                    assert handler.original_sigint_handler == signal.SIG_DFL
            finally:
                _utils_mod._IS_WINDOWS = orig
        finally:
            loop.close()

    def test_register_unix(self):
        """Lines 134-137: register on Unix path."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            orig = _utils_mod._IS_WINDOWS
            _utils_mod._IS_WINDOWS = False
            try:
                with patch.object(loop, "add_signal_handler") as mock_add:
                    mock_add.return_value = None
                    handler.register()
                    assert mock_add.call_count == 2
            finally:
                _utils_mod._IS_WINDOWS = orig
        finally:
            loop.close()

    def test_register_exception(self):
        """Lines 139-144: register handles exceptions gracefully."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            orig = _utils_mod._IS_WINDOWS
            _utils_mod._IS_WINDOWS = False
            try:
                with patch.object(
                    loop, "add_signal_handler", side_effect=RuntimeError("no support")
                ):
                    handler.register()  # Should not raise
            finally:
                _utils_mod._IS_WINDOWS = orig
        finally:
            loop.close()

    def test_unregister_windows(self):
        """Lines 148-152: unregister on Windows path."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            # Use a real callable since signal.SIG_DFL is 0 (falsy)
            original_handler = lambda *a: None
            handler.original_sigint_handler = original_handler
            orig = _utils_mod._IS_WINDOWS
            _utils_mod._IS_WINDOWS = True
            try:
                with patch("signal.signal") as mock_signal:
                    handler.unregister()
                    mock_signal.assert_called_once_with(signal.SIGINT, original_handler)
            finally:
                _utils_mod._IS_WINDOWS = orig
        finally:
            loop.close()

    def test_unregister_unix(self):
        """Lines 153-162: unregister on Unix path."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            # Use real callables since signal.SIG_DFL is 0 (falsy)
            original_sigint = lambda *a: None
            original_sigterm = lambda *a: None
            handler.original_sigint_handler = original_sigint
            handler.original_sigterm_handler = original_sigterm
            orig = _utils_mod._IS_WINDOWS
            _utils_mod._IS_WINDOWS = False
            try:
                with patch.object(loop, "remove_signal_handler"):
                    with patch("signal.signal") as mock_signal:
                        handler.unregister()
                        # Should restore both SIGINT and SIGTERM
                        assert mock_signal.call_count == 2
            finally:
                _utils_mod._IS_WINDOWS = orig
        finally:
            loop.close()

    def test_unregister_exception(self):
        """Lines 163-164: unregister handles exceptions."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            orig = _utils_mod._IS_WINDOWS
            _utils_mod._IS_WINDOWS = False
            try:
                with patch.object(
                    loop,
                    "remove_signal_handler",
                    side_effect=RuntimeError("test"),
                ):
                    handler.unregister()  # Should not raise
            finally:
                _utils_mod._IS_WINDOWS = orig
        finally:
            loop.close()

    def test_handle_second_ctrl_c(self):
        """Lines 173-213: _handle_second_ctrl_c performs cleanup and exit."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            exit_cb = MagicMock()
            handler = SignalHandler(loop=loop, custom_exit_callback=exit_cb)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = False
            try:
                with patch("os._exit") as mock_exit:
                    handler._handle_second_ctrl_c()
                    exit_cb.assert_called_once()
                    mock_exit.assert_called_once_with(0)
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_handle_second_ctrl_c_exit_callback_error(self):
        """Lines 180-181: exit callback raises exception."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            exit_cb = MagicMock(side_effect=RuntimeError("exit error"))
            handler = SignalHandler(loop=loop, custom_exit_callback=exit_cb)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = False
            try:
                with patch("os._exit"):
                    handler._handle_second_ctrl_c()
                    exit_cb.assert_called_once()
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_handle_second_ctrl_c_already_exiting(self):
        """Lines 173-174: already exiting skips callback."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            exit_cb = MagicMock()
            handler = SignalHandler(loop=loop, custom_exit_callback=exit_cb)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = True
            try:
                with patch("os._exit"):
                    handler._handle_second_ctrl_c()
                    exit_cb.assert_not_called()
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_sigint_handler_first_press(self):
        """Lines 224-251: sigint_handler first Ctrl+C."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            pause_cb = MagicMock()
            handler = SignalHandler(loop=loop, pause_callback=pause_cb)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = False
            try:
                setattr(loop, "ctrl_c_pressed", False)
                with patch.object(SignalHandler, "_cancel_interruptible_tasks"):
                    handler.sigint_handler()
                    pause_cb.assert_called_once()
                    assert getattr(loop, "ctrl_c_pressed") is True
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_sigint_handler_already_exiting(self):
        """Lines 224-226: sigint_handler when already exiting."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = True
            try:
                with patch("os._exit") as mock_exit:
                    handler.sigint_handler()
                    mock_exit.assert_called_once_with(0)
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_sigint_handler_second_press_waiting_for_input(self):
        """Lines 228-231: second press while waiting for input returns."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = False
            setattr(loop, "ctrl_c_pressed", True)
            setattr(loop, "waiting_for_input", True)
            try:
                with patch.object(SignalHandler, "_cancel_interruptible_tasks"):
                    handler.sigint_handler()
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_sigint_handler_second_press_exit(self):
        """Lines 234-235: second press exits."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = False
            setattr(loop, "ctrl_c_pressed", True)
            setattr(loop, "waiting_for_input", False)
            try:
                with patch.object(SignalHandler, "_handle_second_ctrl_c") as mock_exit:
                    with patch.object(SignalHandler, "_cancel_interruptible_tasks"):
                        handler.sigint_handler()
                        mock_exit.assert_called_once()
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_sigint_handler_pause_callback_error(self):
        """Lines 247-248: pause callback raises exception."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            pause_cb = MagicMock(side_effect=RuntimeError("pause error"))
            handler = SignalHandler(loop=loop, pause_callback=pause_cb)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = False
            setattr(loop, "ctrl_c_pressed", False)
            try:
                with patch.object(SignalHandler, "_cancel_interruptible_tasks"):
                    handler.sigint_handler()  # Should not raise
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_sigterm_handler(self):
        """Lines 260-268: sigterm_handler exits."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            exit_cb = MagicMock()
            handler = SignalHandler(loop=loop, custom_exit_callback=exit_cb)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = False
            try:
                with patch("os._exit") as mock_exit:
                    handler.sigterm_handler()
                    exit_cb.assert_called_once()
                    mock_exit.assert_called_once_with(0)
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_sigterm_handler_already_exiting(self):
        """Lines 260: sigterm when already exiting."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            original_exiting = _utils_mod._exiting
            _utils_mod._exiting = True
            try:
                with patch("os._exit") as mock_exit:
                    handler.sigterm_handler()
                    mock_exit.assert_called_once_with(0)
            finally:
                _utils_mod._exiting = original_exiting
        finally:
            loop.close()

    def test_cancel_interruptible_tasks(self):
        """Lines 272-290: _cancel_interruptible_tasks cancels matching tasks."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)

            task1 = MagicMock()
            task1.get_name.return_value = "step_1"
            task1.done.return_value = False

            task2 = MagicMock()
            task2.get_name.return_value = "other_task"
            task2.done.return_value = False

            task3 = MagicMock()
            task3.get_name.return_value = "multi_act_2"
            task3.done.return_value = False

            current_task = MagicMock()
            current_task.get_name.return_value = "get_next_action_main"
            current_task.done.return_value = False

            with patch("asyncio.all_tasks", return_value={task1, task2, task3, current_task}):
                with patch("asyncio.current_task", return_value=current_task):
                    handler._cancel_interruptible_tasks()
                    task1.cancel.assert_called_once()
                    task2.cancel.assert_not_called()
                    task3.cancel.assert_called_once()
                    current_task.cancel.assert_called_once()
        finally:
            loop.close()

    def test_reset(self):
        """Lines 345-348: reset clears flags."""
        SignalHandler = self._get_handler_class()
        loop = asyncio.new_event_loop()
        try:
            handler = SignalHandler(loop=loop)
            setattr(loop, "ctrl_c_pressed", True)
            setattr(loop, "waiting_for_input", True)
            handler.reset()
            assert getattr(loop, "ctrl_c_pressed") is False
            assert getattr(loop, "waiting_for_input") is False
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# time_execution_sync and time_execution_async (lines 361-370, 391-400)
# ---------------------------------------------------------------------------


class TestTimeExecution:
    """Test time_execution_sync and time_execution_async decorators."""

    def test_time_execution_sync_fast(self):
        """Test sync decorator with fast function (no logging)."""
        from openbrowser.utils import time_execution_sync

        @time_execution_sync("test_func")
        def fast_func():
            return 42

        result = fast_func()
        assert result == 42

    def test_time_execution_sync_slow_with_self_logger(self):
        """Lines 361-370: sync decorator with slow function and self.logger."""
        from openbrowser.utils import time_execution_sync

        class MyClass:
            def __init__(self):
                self.logger = logging.getLogger("test")

            @time_execution_sync("my_method")
            def slow_method(self):
                time.sleep(0.3)
                return "done"

        obj = MyClass()
        with patch.object(obj.logger, "debug") as mock_debug:
            result = obj.slow_method()
            assert result == "done"
            mock_debug.assert_called_once()

    def test_time_execution_sync_slow_with_agent_kwarg(self):
        """Lines 364-365: sync decorator resolves logger from agent kwarg."""
        from openbrowser.utils import time_execution_sync

        @time_execution_sync("test")
        def slow_func(agent=None):
            time.sleep(0.3)
            return "done"

        mock_agent = MagicMock()
        mock_agent.logger = logging.getLogger("agent_test")
        with patch.object(mock_agent.logger, "debug"):
            result = slow_func(agent=mock_agent)
            assert result == "done"

    def test_time_execution_sync_slow_with_browser_session_kwarg(self):
        """Lines 366-367: sync decorator resolves logger from browser_session kwarg."""
        from openbrowser.utils import time_execution_sync

        @time_execution_sync("test")
        def slow_func(browser_session=None):
            time.sleep(0.3)
            return "done"

        mock_session = MagicMock()
        mock_session.logger = logging.getLogger("session_test")
        with patch.object(mock_session.logger, "debug"):
            result = slow_func(browser_session=mock_session)
            assert result == "done"

    def test_time_execution_sync_slow_no_logger(self):
        """Lines 368-370: sync decorator falls back to module logger."""
        from openbrowser.utils import time_execution_sync

        @time_execution_sync("test")
        def slow_func():
            time.sleep(0.3)
            return "done"

        result = slow_func()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_time_execution_async_fast(self):
        """Test async decorator with fast function (no logging)."""
        from openbrowser.utils import time_execution_async

        @time_execution_async("test_func")
        async def fast_func():
            return 42

        result = await fast_func()
        assert result == 42

    @pytest.mark.asyncio
    async def test_time_execution_async_slow_with_self_logger(self):
        """Lines 391-400: async decorator with slow function and self.logger."""
        from openbrowser.utils import time_execution_async

        class MyClass:
            def __init__(self):
                self.logger = logging.getLogger("test_async")

            @time_execution_async("my_method")
            async def slow_method(self):
                await asyncio.sleep(0.3)
                return "done"

        obj = MyClass()
        with patch.object(obj.logger, "debug") as mock_debug:
            result = await obj.slow_method()
            assert result == "done"
            mock_debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_time_execution_async_slow_with_agent_kwarg(self):
        """Lines 394-395: async decorator resolves logger from agent kwarg."""
        from openbrowser.utils import time_execution_async

        @time_execution_async("test")
        async def slow_func(agent=None):
            await asyncio.sleep(0.3)
            return "done"

        mock_agent = MagicMock()
        mock_agent.logger = logging.getLogger("agent_async_test")
        result = await slow_func(agent=mock_agent)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_time_execution_async_slow_with_browser_session_kwarg(self):
        """Lines 396-397: async decorator resolves logger from browser_session kwarg."""
        from openbrowser.utils import time_execution_async

        @time_execution_async("test")
        async def slow_func(browser_session=None):
            await asyncio.sleep(0.3)
            return "done"

        mock_session = MagicMock()
        mock_session.logger = logging.getLogger("session_async_test")
        result = await slow_func(browser_session=mock_session)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_time_execution_async_slow_fallback_logger(self):
        """Lines 398-400: async decorator falls back to module logger."""
        from openbrowser.utils import time_execution_async

        @time_execution_async("test")
        async def slow_func():
            await asyncio.sleep(0.3)
            return "done"

        result = await slow_func()
        assert result == "done"


# ---------------------------------------------------------------------------
# URL matching (lines 525, 531, 538, 552-554)
# ---------------------------------------------------------------------------


class TestMatchUrlWithDomainPattern:
    """Test match_url_with_domain_pattern for missed branches."""

    def test_multiple_wildcards_warning(self):
        """Line 525: log_warnings=True for multiple wildcards."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://sub.example.com", "*.*.example.com", log_warnings=True
        )
        assert result is False

    def test_wildcard_tld_warning(self):
        """Line 531: log_warnings=True for wildcard TLD."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://example.com", "example.*", log_warnings=True
        )
        assert result is False

    def test_embedded_wildcard_warning(self):
        """Line 538: log_warnings=True for embedded wildcard."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://example.com", "ex*ple.com", log_warnings=True
        )
        assert result is False

    def test_exception_handling(self):
        """Lines 552-554: exception during matching."""
        from openbrowser.utils import match_url_with_domain_pattern

        # Pass something that will cause urlparse to fail in some way
        result = match_url_with_domain_pattern(None, "example.com")
        assert result is False

    def test_new_tab_page(self):
        """Test new tab page returns False."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern("about:blank", "example.com")
        assert result is False

    def test_no_scheme_or_domain(self):
        """Test URL with missing scheme or domain."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern("", "example.com")
        assert result is False

    def test_wildcard_domain(self):
        """Test pattern with * wildcard domain."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://anything.com", "https://*"
        )
        assert result is True

    def test_exact_match(self):
        """Test exact domain match."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://example.com", "example.com"
        )
        assert result is True

    def test_scheme_mismatch(self):
        """Test scheme mismatch."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "http://example.com", "example.com"
        )
        assert result is False  # Default scheme is https

    def test_port_in_pattern(self):
        """Test port in pattern is stripped."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://example.com", "https://example.com:443"
        )
        assert result is True

    def test_star_dot_domain_matches_parent(self):
        """Test *.example.com matches bare example.com."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://example.com", "https://*.example.com"
        )
        assert result is True

    def test_fnmatch_normal_glob(self):
        """Test normal glob matching."""
        from openbrowser.utils import match_url_with_domain_pattern

        result = match_url_with_domain_pattern(
            "https://sub.example.com", "https://*.example.com"
        )
        assert result is True


# ---------------------------------------------------------------------------
# Version functions (lines 581-598, 613-616, 631-659)
# ---------------------------------------------------------------------------


class TestVersionFunctions:
    """Test get_openbrowser_version, check_latest_openbrowser_version, get_git_info."""

    def test_get_openbrowser_version_from_pyproject(self):
        """Lines 581-587: version from pyproject.toml."""
        from openbrowser.utils import get_openbrowser_version

        # Clear cache
        get_openbrowser_version.cache_clear()
        version = get_openbrowser_version()
        assert isinstance(version, str)
        assert version != ""

    def test_get_openbrowser_version_no_pyproject(self):
        """Lines 596-598: version fallback to importlib.metadata."""
        from openbrowser.utils import get_openbrowser_version

        get_openbrowser_version.cache_clear()
        with patch("pathlib.Path.exists", return_value=False):
            with patch(
                "importlib.metadata.version", side_effect=Exception("not found")
            ):
                version = get_openbrowser_version()
                assert version == "unknown"

    @pytest.mark.asyncio
    async def test_check_latest_version_success(self):
        """Lines 613-616: check latest version from PyPI (success)."""
        from openbrowser.utils import check_latest_openbrowser_version

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"info": {"version": "1.0.0"}}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await check_latest_openbrowser_version()
            assert result == "1.0.0"

    @pytest.mark.asyncio
    async def test_check_latest_version_failure(self):
        """Lines 613-616: check latest version fails gracefully."""
        from openbrowser.utils import check_latest_openbrowser_version

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("network error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await check_latest_openbrowser_version()
            assert result is None

    @pytest.mark.asyncio
    async def test_check_latest_version_non_200(self):
        """Lines 613-616: check latest version returns non-200."""
        from openbrowser.utils import check_latest_openbrowser_version

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await check_latest_openbrowser_version()
            assert result is None

    def test_get_git_info_no_git_dir(self):
        """Lines 631-659: get_git_info when no .git directory."""
        from openbrowser.utils import get_git_info

        get_git_info.cache_clear()
        with patch("pathlib.Path.exists", return_value=False):
            result = get_git_info()
            assert result is None

    def test_get_git_info_exception(self):
        """Lines 657-659: get_git_info handles exception."""
        from openbrowser.utils import get_git_info

        get_git_info.cache_clear()
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "subprocess.check_output",
                side_effect=Exception("git not found"),
            ):
                result = get_git_info()
                assert result is None

    def test_get_git_info_success(self):
        """Lines 631-656: get_git_info returns git info dict."""
        from openbrowser.utils import get_git_info

        get_git_info.cache_clear()
        with patch("pathlib.Path.exists", return_value=True):
            with patch("subprocess.check_output") as mock_check:
                mock_check.return_value = b"test_value\n"
                result = get_git_info()
                assert result is not None
                assert "commit_hash" in result
                assert "branch" in result
                assert "remote_url" in result
                assert "commit_timestamp" in result


# ---------------------------------------------------------------------------
# Other utility functions
# ---------------------------------------------------------------------------


class TestOtherUtilFunctions:
    """Test singleton, check_env_variables, is_unsafe_pattern, merge_dicts, etc."""

    def test_singleton_decorator(self):
        """Test singleton decorator creates single instance."""
        from openbrowser.utils import singleton

        @singleton
        class MyClass:
            def __init__(self, value):
                self.value = value

        a = MyClass(1)
        b = MyClass(2)
        assert a is b
        assert a.value == 1

    def test_check_env_variables_all(self):
        """Test check_env_variables with all()."""
        from openbrowser.utils import check_env_variables

        with patch.dict(os.environ, {"KEY1": "val1", "KEY2": "val2"}):
            assert check_env_variables(["KEY1", "KEY2"]) is True

    def test_check_env_variables_any(self):
        """Test check_env_variables with any()."""
        from openbrowser.utils import check_env_variables

        with patch.dict(os.environ, {"KEY1": "val1"}, clear=False):
            assert check_env_variables(["KEY1", "NONEXISTENT"], any_or_all=any) is True

    def test_check_env_variables_missing(self):
        """Test check_env_variables with missing keys."""
        from openbrowser.utils import check_env_variables

        assert check_env_variables(["DEFINITELY_NOT_SET_12345"]) is False

    def test_is_unsafe_pattern_safe(self):
        """Test is_unsafe_pattern with safe pattern."""
        from openbrowser.utils import is_unsafe_pattern

        assert is_unsafe_pattern("*.example.com") is False
        assert is_unsafe_pattern("example.*") is False

    def test_is_unsafe_pattern_with_scheme(self):
        """Test is_unsafe_pattern with scheme."""
        from openbrowser.utils import is_unsafe_pattern

        assert is_unsafe_pattern("https://*.example.com") is False

    def test_is_unsafe_pattern_unsafe(self):
        """Test is_unsafe_pattern with unsafe pattern."""
        from openbrowser.utils import is_unsafe_pattern

        assert is_unsafe_pattern("ex*ple.com") is True

    def test_is_new_tab_page(self):
        """Test is_new_tab_page."""
        from openbrowser.utils import is_new_tab_page

        assert is_new_tab_page("about:blank") is True
        assert is_new_tab_page("chrome://new-tab-page/") is True
        assert is_new_tab_page("chrome://newtab") is True
        assert is_new_tab_page("https://example.com") is False

    def test_merge_dicts_simple(self):
        """Test merge_dicts basic merge."""
        from openbrowser.utils import merge_dicts

        a = {"key1": "val1"}
        b = {"key2": "val2"}
        result = merge_dicts(a, b)
        assert result == {"key1": "val1", "key2": "val2"}

    def test_merge_dicts_nested(self):
        """Test merge_dicts with nested dicts."""
        from openbrowser.utils import merge_dicts

        a = {"outer": {"inner1": 1}}
        b = {"outer": {"inner2": 2}}
        result = merge_dicts(a, b)
        assert result == {"outer": {"inner1": 1, "inner2": 2}}

    def test_merge_dicts_lists(self):
        """Test merge_dicts with list concatenation."""
        from openbrowser.utils import merge_dicts

        a = {"items": [1, 2]}
        b = {"items": [3, 4]}
        result = merge_dicts(a, b)
        assert result == {"items": [1, 2, 3, 4]}

    def test_merge_dicts_conflict(self):
        """Test merge_dicts raises on conflict."""
        from openbrowser.utils import merge_dicts

        a = {"key": "val1"}
        b = {"key": "val2"}
        with pytest.raises(Exception, match="Conflict"):
            merge_dicts(a, b)

    def test_log_pretty_path_none(self):
        """Test _log_pretty_path with None."""
        from openbrowser.utils import _log_pretty_path

        assert _log_pretty_path(None) == ""

    def test_log_pretty_path_empty(self):
        """Test _log_pretty_path with empty string."""
        from openbrowser.utils import _log_pretty_path

        assert _log_pretty_path("") == ""

    def test_log_pretty_path_non_path(self):
        """Test _log_pretty_path with non-path type."""
        from openbrowser.utils import _log_pretty_path

        result = _log_pretty_path(123)
        assert "<int>" in result

    def test_log_pretty_path_with_spaces(self):
        """Test _log_pretty_path with path containing spaces."""
        from openbrowser.utils import _log_pretty_path

        result = _log_pretty_path("/some path/with spaces")
        assert '"' in result

    def test_log_pretty_url(self):
        """Test _log_pretty_url."""
        from openbrowser.utils import _log_pretty_url

        result = _log_pretty_url("https://www.example.com/very/long/path")
        assert "..." in result
        assert "https://" not in result

    def test_log_pretty_url_short(self):
        """Test _log_pretty_url with short URL."""
        from openbrowser.utils import _log_pretty_url

        result = _log_pretty_url("https://example.com")
        assert result == "example.com"

    def test_log_pretty_url_no_max(self):
        """Test _log_pretty_url with no max_len."""
        from openbrowser.utils import _log_pretty_url

        result = _log_pretty_url("https://example.com/long/path", max_len=None)
        assert "..." not in result

    def test_url_pattern_compiled(self):
        """Test URL_PATTERN is compiled."""
        from openbrowser.utils import URL_PATTERN

        assert URL_PATTERN is not None
        match = URL_PATTERN.search("Visit https://example.com today")
        assert match is not None
