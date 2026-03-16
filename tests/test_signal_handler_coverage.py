"""Comprehensive tests for src/openbrowser/utils/signal_handler.py to cover all missed lines.

Missing lines: 63, 72, 77, 84-107, 111-114, 158-164, 168-176, 180-182,
186-192, 196-202, 206-211, 215-216, 221, 226, 229-230, 233
"""

import asyncio
import logging
import signal
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

logger = logging.getLogger(__name__)


class TestSignalHandlerBasic:
    """Test SignalHandler from signal_handler.py basic functionality."""

    def test_init_defaults(self):
        """Test default initialization."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        assert handler._loop is None
        assert handler._pause_callback is None
        assert handler._resume_callback is None
        assert handler._custom_exit_callback is None
        assert handler._exit_on_second_int is True
        assert handler._sigint_count == 0
        assert handler._is_paused is False
        assert handler._original_handler is None
        assert handler._registered is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        from openbrowser.utils.signal_handler import SignalHandler

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
            )
            assert handler._loop is loop
            assert handler._pause_callback is pause_cb
            assert handler._resume_callback is resume_cb
            assert handler._custom_exit_callback is exit_cb
            assert handler._exit_on_second_int is False
        finally:
            loop.close()


class TestSignalHandlerRegister:
    """Test register/unregister methods."""

    def test_register(self):
        """Line 63: register skips if already registered."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        with patch("signal.signal", return_value=signal.SIG_DFL) as mock_signal:
            handler.register()
            assert handler._registered is True
            mock_signal.assert_called_once()

            # Second call should be no-op (line 63)
            mock_signal.reset_mock()
            handler.register()
            mock_signal.assert_not_called()

        # Clean up
        handler._registered = False

    def test_unregister_not_registered(self):
        """Line 72: unregister when not registered."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        handler.unregister()  # Should be no-op

    def test_unregister_with_original_handler(self):
        """Lines 74-75: unregister restores original handler."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        handler._registered = True
        handler._original_handler = signal.SIG_DFL

        with patch("signal.signal") as mock_signal:
            handler.unregister()
            mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_DFL)
            assert handler._registered is False

    def test_unregister_no_original_handler(self):
        """Lines 76-77: unregister with no original handler restores SIG_DFL."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        handler._registered = True
        handler._original_handler = None

        with patch("signal.signal") as mock_signal:
            handler.unregister()
            mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_DFL)
            assert handler._registered is False


class TestSignalHandlerSigint:
    """Test _handle_sigint method."""

    def test_first_sigint_pauses(self):
        """Lines 84-98: first SIGINT pauses the agent."""
        from openbrowser.utils.signal_handler import SignalHandler

        pause_cb = MagicMock()
        handler = SignalHandler(pause_callback=pause_cb)

        handler._handle_sigint(signal.SIGINT, None)

        assert handler._sigint_count == 1
        assert handler._is_paused is True
        pause_cb.assert_called_once()

    def test_first_sigint_no_callback(self):
        """Lines 84-98: first SIGINT without callback."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        handler._handle_sigint(signal.SIGINT, None)

        assert handler._sigint_count == 1
        assert handler._is_paused is True

    def test_first_sigint_resumes_when_paused(self):
        """Lines 87-92: first SIGINT when paused resumes."""
        from openbrowser.utils.signal_handler import SignalHandler

        resume_cb = MagicMock()
        handler = SignalHandler(resume_callback=resume_cb)
        handler._is_paused = True

        handler._handle_sigint(signal.SIGINT, None)

        assert handler._sigint_count == 1
        assert handler._is_paused is False
        resume_cb.assert_called_once()

    def test_first_sigint_resumes_when_paused_no_callback(self):
        """Lines 87-90: first SIGINT when paused resumes without callback."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        handler._is_paused = True

        handler._handle_sigint(signal.SIGINT, None)

        assert handler._sigint_count == 1
        assert handler._is_paused is False

    def test_second_sigint_exits_with_callback(self):
        """Lines 100-103: second SIGINT calls exit callback."""
        from openbrowser.utils.signal_handler import SignalHandler

        exit_cb = MagicMock()
        handler = SignalHandler(custom_exit_callback=exit_cb)
        handler._sigint_count = 1  # Already received one

        handler._handle_sigint(signal.SIGINT, None)

        assert handler._sigint_count == 2
        exit_cb.assert_called_once()

    def test_second_sigint_exits_without_callback(self):
        """Lines 104-107: second SIGINT raises KeyboardInterrupt without callback."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        handler._sigint_count = 1
        handler._registered = True
        handler._original_handler = signal.SIG_DFL

        with patch("signal.signal"):
            with pytest.raises(KeyboardInterrupt):
                handler._handle_sigint(signal.SIGINT, None)

    def test_second_sigint_no_exit(self):
        """Test second SIGINT when exit_on_second_int=False."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler(exit_on_second_int=False)
        handler._sigint_count = 1
        handler._handle_sigint(signal.SIGINT, None)
        assert handler._sigint_count == 2


class TestScheduleCallback:
    """Test _schedule_callback method."""

    def test_schedule_with_loop(self):
        """Lines 111-112: schedule callback with event loop."""
        from openbrowser.utils.signal_handler import SignalHandler

        loop = MagicMock()
        callback = MagicMock()
        handler = SignalHandler(loop=loop)

        handler._schedule_callback(callback)
        loop.call_soon_threadsafe.assert_called_once_with(callback)

    def test_schedule_without_loop(self):
        """Lines 113-114: schedule callback without event loop (direct call)."""
        from openbrowser.utils.signal_handler import SignalHandler

        callback = MagicMock()
        handler = SignalHandler(loop=None)

        handler._schedule_callback(callback)
        callback.assert_called_once()


class TestSignalHandlerReset:
    """Test reset method."""

    def test_reset(self):
        """Test reset clears counters."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        handler._sigint_count = 5
        handler._is_paused = True

        handler.reset()

        assert handler._sigint_count == 0
        assert handler._is_paused is False


class TestSignalHandlerProperties:
    """Test is_paused and interrupt_count properties."""

    def test_is_paused_property(self):
        """Test is_paused property."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        assert handler.is_paused is False
        handler._is_paused = True
        assert handler.is_paused is True

    def test_interrupt_count_property(self):
        """Test interrupt_count property."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        assert handler.interrupt_count == 0
        handler._sigint_count = 3
        assert handler.interrupt_count == 3


class TestSignalHandlerContextManager:
    """Test context manager protocol."""

    def test_context_manager(self):
        """Test __enter__ and __exit__."""
        from openbrowser.utils.signal_handler import SignalHandler

        handler = SignalHandler()
        with patch.object(SignalHandler, "register") as mock_reg:
            with patch.object(SignalHandler, "unregister") as mock_unreg:
                with handler as h:
                    assert h is handler
                    mock_reg.assert_called_once()
                mock_unreg.assert_called_once()


# ---------------------------------------------------------------------------
# AsyncSignalHandler tests (lines 158-233)
# ---------------------------------------------------------------------------


class TestAsyncSignalHandler:
    """Test AsyncSignalHandler class."""

    def test_init(self):
        """Lines 158-164: AsyncSignalHandler initialization."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        pause_cb = MagicMock()
        resume_cb = MagicMock()
        stop_cb = MagicMock()

        handler = AsyncSignalHandler(
            pause_callback=pause_cb,
            resume_callback=resume_cb,
            stop_callback=stop_cb,
        )

        assert handler._pause_callback is pause_cb
        assert handler._resume_callback is resume_cb
        assert handler._stop_callback is stop_cb
        assert handler._is_paused is False
        assert handler._handler is None

    def test_start(self):
        """Lines 168-176: start creates and registers inner SignalHandler."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.get_event_loop", return_value=loop):
                with patch(
                    "openbrowser.utils.signal_handler.SignalHandler"
                ) as mock_sh:
                    mock_instance = MagicMock()
                    mock_sh.return_value = mock_instance
                    handler.start()
                    mock_sh.assert_called_once()
                    mock_instance.register.assert_called_once()
                    assert handler._handler is mock_instance
        finally:
            loop.close()

    def test_stop(self):
        """Lines 180-182: stop unregisters and clears handler."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        handler._handler = MagicMock()

        handler.stop()

        handler._handler is None  # noqa: B015 (intentional)

    def test_stop_no_handler(self):
        """Lines 180: stop when handler is None."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        handler.stop()  # Should be no-op

    def test_on_pause_sync_callback(self):
        """Lines 186-192: _on_pause with sync callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        pause_cb = MagicMock()
        handler = AsyncSignalHandler(pause_callback=pause_cb)

        handler._on_pause()

        assert handler._is_paused is True
        pause_cb.assert_called_once()

    def test_on_pause_async_callback(self):
        """Lines 189-190: _on_pause with async callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        async def async_pause():
            pass

        handler = AsyncSignalHandler(pause_callback=async_pause)

        loop = asyncio.new_event_loop()
        try:
            with patch("asyncio.create_task") as mock_create_task:
                with patch("asyncio.iscoroutinefunction", return_value=True):
                    handler._on_pause()
                    assert handler._is_paused is True
                    mock_create_task.assert_called_once()
        finally:
            loop.close()

    def test_on_pause_no_callback(self):
        """Lines 186-187: _on_pause without callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        handler._on_pause()
        assert handler._is_paused is True

    def test_on_resume_sync_callback(self):
        """Lines 196-202: _on_resume with sync callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        resume_cb = MagicMock()
        handler = AsyncSignalHandler(resume_callback=resume_cb)
        handler._is_paused = True

        handler._on_resume()

        assert handler._is_paused is False
        resume_cb.assert_called_once()

    def test_on_resume_async_callback(self):
        """Lines 199-200: _on_resume with async callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        async def async_resume():
            pass

        handler = AsyncSignalHandler(resume_callback=async_resume)
        handler._is_paused = True

        with patch("asyncio.create_task") as mock_create_task:
            with patch("asyncio.iscoroutinefunction", return_value=True):
                handler._on_resume()
                assert handler._is_paused is False
                mock_create_task.assert_called_once()

    def test_on_resume_no_callback(self):
        """Lines 196-197: _on_resume without callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        handler._is_paused = True
        handler._on_resume()
        assert handler._is_paused is False

    def test_on_stop_sync_callback(self):
        """Lines 206-211: _on_stop with sync callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        stop_cb = MagicMock()
        handler = AsyncSignalHandler(stop_callback=stop_cb)

        handler._on_stop()

        assert handler._stop_event.is_set()
        stop_cb.assert_called_once()

    def test_on_stop_async_callback(self):
        """Lines 208-209: _on_stop with async callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        async def async_stop():
            pass

        handler = AsyncSignalHandler(stop_callback=async_stop)

        with patch("asyncio.create_task") as mock_create_task:
            with patch("asyncio.iscoroutinefunction", return_value=True):
                handler._on_stop()
                assert handler._stop_event.is_set()
                mock_create_task.assert_called_once()

    def test_on_stop_no_callback(self):
        """Lines 206: _on_stop without callback."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        handler._on_stop()
        assert handler._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_wait_if_paused_not_paused(self):
        """Lines 215-216: wait_if_paused when not paused returns immediately."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        await handler.wait_if_paused()  # Should return immediately

    @pytest.mark.asyncio
    async def test_wait_if_paused_when_paused(self):
        """Lines 215-216: wait_if_paused when paused waits for event."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        handler._is_paused = True

        # Set the event after a short delay
        async def set_event():
            await asyncio.sleep(0.05)
            handler._pause_event.set()

        asyncio.create_task(set_event())
        await handler.wait_if_paused()

    def test_should_stop(self):
        """Line 221: should_stop property."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        assert handler.should_stop is False
        handler._stop_event.set()
        assert handler.should_stop is True

    def test_is_paused(self):
        """Line 226: is_paused property."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        assert handler.is_paused is False
        handler._is_paused = True
        assert handler.is_paused is True

    def test_context_manager(self):
        """Lines 229-233: context manager protocol."""
        from openbrowser.utils.signal_handler import AsyncSignalHandler

        handler = AsyncSignalHandler()
        with patch.object(AsyncSignalHandler, "start") as mock_start:
            with patch.object(AsyncSignalHandler, "stop") as mock_stop:
                with handler as h:
                    assert h is handler
                    mock_start.assert_called_once()
                mock_stop.assert_called_once()
