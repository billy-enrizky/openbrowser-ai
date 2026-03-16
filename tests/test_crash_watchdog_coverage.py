"""Tests for openbrowser.browser.watchdogs.crash_watchdog module -- 100% coverage target."""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser_session():
    """Create a mock BrowserSession for CrashWatchdog tests."""
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_crash_watchdog")
    session.event_bus = MagicMock()  # mock dispatch to prevent unawaited coroutines
    session._cdp_client_root = MagicMock()
    session.agent_focus = MagicMock()
    session.agent_focus.target_id = "AAAA1111BBBB2222CCCC3333DDDD4444"
    session.agent_focus.session_id = "sess-1234"
    session.agent_focus.cdp_client = MagicMock()
    session.get_or_create_cdp_session = AsyncMock()
    session.cdp_client = MagicMock()
    session.cdp_client.send = MagicMock()
    session.cdp_client.send.Target = MagicMock()
    session.cdp_client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})
    session._local_browser_watchdog = None
    session.id = "test-crash-session"
    session.is_local = True
    return session


def _make_crash_watchdog(session=None, event_bus=None):
    from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog

    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = MagicMock()
    session.event_bus = event_bus
    return CrashWatchdog.model_construct(
        event_bus=event_bus,
        browser_session=session,
        network_timeout_seconds=10.0,
        check_interval_seconds=5.0,
    )


# ---------------------------------------------------------------------------
# NetworkRequestTracker
# ---------------------------------------------------------------------------

class TestNetworkRequestTracker:
    def test_init(self):
        from openbrowser.browser.watchdogs.crash_watchdog import NetworkRequestTracker

        tracker = NetworkRequestTracker(
            request_id="r1",
            start_time=1000.0,
            url="https://example.com",
            method="GET",
            resource_type="Document",
        )
        assert tracker.request_id == "r1"
        assert tracker.url == "https://example.com"
        assert tracker.method == "GET"
        assert tracker.resource_type == "Document"

    def test_init_defaults(self):
        from openbrowser.browser.watchdogs.crash_watchdog import NetworkRequestTracker

        tracker = NetworkRequestTracker(
            request_id="r2",
            start_time=0.0,
            url="",
            method="POST",
        )
        assert tracker.resource_type is None


# ---------------------------------------------------------------------------
# CrashWatchdog class-level attributes
# ---------------------------------------------------------------------------

class TestCrashWatchdogClassAttrs:
    def test_listens_to_events(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog

        assert len(CrashWatchdog.LISTENS_TO) == 4

    def test_emits_events(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog
        from openbrowser.browser.events import BrowserErrorEvent

        assert BrowserErrorEvent in CrashWatchdog.EMITS


# ---------------------------------------------------------------------------
# on_BrowserConnectedEvent
# ---------------------------------------------------------------------------

class TestOnBrowserConnectedEvent:
    @pytest.mark.asyncio
    async def test_starts_monitoring(self):
        session = _make_mock_browser_session()
        session.cdp_client = MagicMock()
        watchdog = _make_crash_watchdog(session=session)

        event = MagicMock()
        with patch("asyncio.create_task") as mock_create_task:
            await watchdog.on_BrowserConnectedEvent(event)
            # on_BrowserConnectedEvent should call asyncio.create_task(_start_monitoring())
            mock_create_task.assert_called_once()
            # Verify the coroutine passed to create_task is from _start_monitoring
            coro = mock_create_task.call_args[0][0]
            assert coro is not None, "Expected a coroutine to be passed to create_task"
            # Clean up the coroutine to avoid RuntimeWarning
            coro.close()


# ---------------------------------------------------------------------------
# on_BrowserStoppedEvent
# ---------------------------------------------------------------------------

class TestOnBrowserStoppedEvent:
    @pytest.mark.asyncio
    async def test_stops_monitoring(self):
        watchdog = _make_crash_watchdog()
        # Provide a done task to skip cancel path
        mock_task = MagicMock()
        mock_task.done.return_value = True
        watchdog._monitoring_task = mock_task
        # Pre-populate tracking state to verify cleanup
        watchdog._targets_with_listeners.add("target-1")
        watchdog._last_responsive_checks["target-1"] = 1000.0

        event = MagicMock()
        await watchdog.on_BrowserStoppedEvent(event)
        # _stop_monitoring should clear all tracking state
        assert len(watchdog._targets_with_listeners) == 0, "targets_with_listeners not cleaned up"
        assert len(watchdog._last_responsive_checks) == 0, "last_responsive_checks not cleaned up"


# ---------------------------------------------------------------------------
# on_TabCreatedEvent
# ---------------------------------------------------------------------------

class TestOnTabCreatedEvent:
    @pytest.mark.asyncio
    async def test_attaches_to_target(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog
        session = _make_mock_browser_session()
        watchdog = _make_crash_watchdog(session=session)

        event = MagicMock()
        with patch.object(CrashWatchdog, "attach_to_target", new_callable=AsyncMock) as mock_attach:
            await watchdog.on_TabCreatedEvent(event)
            mock_attach.assert_awaited_once_with(session.agent_focus.target_id)

    @pytest.mark.asyncio
    async def test_asserts_agent_focus(self):
        session = _make_mock_browser_session()
        session.agent_focus = None
        watchdog = _make_crash_watchdog(session=session)

        event = MagicMock()
        with pytest.raises(AssertionError):
            await watchdog.on_TabCreatedEvent(event)


# ---------------------------------------------------------------------------
# on_TabClosedEvent
# ---------------------------------------------------------------------------

class TestOnTabClosedEvent:
    @pytest.mark.asyncio
    async def test_removes_target_from_listeners(self):
        watchdog = _make_crash_watchdog()
        target_id = "target-abc"
        watchdog._targets_with_listeners.add(target_id)

        event = MagicMock()
        event.target_id = target_id
        await watchdog.on_TabClosedEvent(event)

        assert target_id not in watchdog._targets_with_listeners

    @pytest.mark.asyncio
    async def test_noop_if_target_not_tracked(self):
        watchdog = _make_crash_watchdog()
        event = MagicMock()
        event.target_id = "not-tracked"
        await watchdog.on_TabClosedEvent(event)
        # no error


# ---------------------------------------------------------------------------
# attach_to_target
# ---------------------------------------------------------------------------

class TestAttachToTarget:
    @pytest.mark.asyncio
    async def test_skips_already_registered(self):
        watchdog = _make_crash_watchdog()
        target_id = "AAAA1111BBBB2222"
        watchdog._targets_with_listeners.add(target_id)
        await watchdog.attach_to_target(target_id)
        # should return early, no CDP calls

    @pytest.mark.asyncio
    async def test_registers_crash_handler(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Target = MagicMock()
        cdp_session_mock.cdp_client.register.Target.targetCrashed = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Target = MagicMock()
        cdp_session_mock.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [
                    {"targetId": "target123456", "url": "https://example.com", "type": "page"}
                ]
            }
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_crash_watchdog(session=session)
        await watchdog.attach_to_target("target123456")

        assert "target123456" in watchdog._targets_with_listeners
        cdp_session_mock.cdp_client.register.Target.targetCrashed.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("CDP error"))
        watchdog = _make_crash_watchdog(session=session)

        # Should not raise
        await watchdog.attach_to_target("target-fail")
        assert "target-fail" not in watchdog._targets_with_listeners

    @pytest.mark.asyncio
    async def test_no_target_info_found(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.register = MagicMock()
        cdp_session_mock.cdp_client.register.Target = MagicMock()
        cdp_session_mock.cdp_client.register.Target.targetCrashed = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Target = MagicMock()
        cdp_session_mock.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": []}
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)

        watchdog = _make_crash_watchdog(session=session)
        await watchdog.attach_to_target("target-no-info")
        assert "target-no-info" in watchdog._targets_with_listeners


# ---------------------------------------------------------------------------
# CDP event handlers (_on_request_cdp, _on_response_cdp, etc.)
# ---------------------------------------------------------------------------

class TestCDPRequestTracking:
    @pytest.mark.asyncio
    async def test_on_request_cdp(self):
        watchdog = _make_crash_watchdog()
        event = {
            "requestId": "req-1",
            "request": {"url": "https://example.com/api", "method": "GET"},
            "type": "XHR",
        }
        await watchdog._on_request_cdp(event)
        assert "req-1" in watchdog._active_requests
        assert watchdog._active_requests["req-1"].url == "https://example.com/api"
        assert watchdog._active_requests["req-1"].method == "GET"
        assert watchdog._active_requests["req-1"].resource_type == "XHR"

    def test_on_response_cdp(self):
        from openbrowser.browser.watchdogs.crash_watchdog import NetworkRequestTracker

        watchdog = _make_crash_watchdog()
        watchdog._active_requests["req-2"] = NetworkRequestTracker(
            request_id="req-2",
            start_time=time.time() - 1.0,
            url="https://example.com/data",
            method="POST",
        )
        event = {"requestId": "req-2", "response": {"url": "https://example.com/data"}}
        watchdog._on_response_cdp(event)
        # Should NOT remove from tracking (waits for loadingFinished)
        assert "req-2" in watchdog._active_requests

    def test_on_response_cdp_unknown_request(self):
        watchdog = _make_crash_watchdog()
        event = {"requestId": "unknown", "response": {"url": "http://x.com"}}
        watchdog._on_response_cdp(event)  # No error

    def test_on_request_failed_cdp(self):
        from openbrowser.browser.watchdogs.crash_watchdog import NetworkRequestTracker

        watchdog = _make_crash_watchdog()
        watchdog._active_requests["req-3"] = NetworkRequestTracker(
            request_id="req-3",
            start_time=time.time() - 2.0,
            url="https://example.com/fail",
            method="GET",
        )
        event = {"requestId": "req-3"}
        watchdog._on_request_failed_cdp(event)
        assert "req-3" not in watchdog._active_requests

    def test_on_request_failed_cdp_unknown(self):
        watchdog = _make_crash_watchdog()
        event = {"requestId": "not-tracked"}
        watchdog._on_request_failed_cdp(event)

    def test_on_request_finished_cdp(self):
        from openbrowser.browser.watchdogs.crash_watchdog import NetworkRequestTracker

        watchdog = _make_crash_watchdog()
        watchdog._active_requests["req-4"] = NetworkRequestTracker(
            request_id="req-4",
            start_time=time.time(),
            url="https://example.com/done",
            method="GET",
        )
        event = {"requestId": "req-4"}
        watchdog._on_request_finished_cdp(event)
        assert "req-4" not in watchdog._active_requests

    def test_on_request_finished_cdp_unknown(self):
        watchdog = _make_crash_watchdog()
        event = {"requestId": "nonexistent"}
        watchdog._on_request_finished_cdp(event)  # pop with default, no error


# ---------------------------------------------------------------------------
# _on_target_crash_cdp
# ---------------------------------------------------------------------------

class TestOnTargetCrashCDP:
    @pytest.mark.asyncio
    async def test_crash_agent_focus(self):
        session = _make_mock_browser_session()
        target_id = "target-crashed1"
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [{"targetId": target_id, "url": "https://crashed.com"}]
            }
        )
        session.agent_focus.target_id = target_id
        event_bus = MagicMock()
        watchdog = _make_crash_watchdog(session=session, event_bus=event_bus)

        await watchdog._on_target_crash_cdp(target_id)
        event_bus.dispatch.assert_called_once()
        args = event_bus.dispatch.call_args[0][0]
        assert args.error_type == "TargetCrash"

    @pytest.mark.asyncio
    async def test_crash_not_agent_focus(self):
        session = _make_mock_browser_session()
        target_id = "target-crashed2"
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [{"targetId": target_id, "url": "https://other.com"}]
            }
        )
        session.agent_focus.target_id = "different-target"
        event_bus = MagicMock()
        watchdog = _make_crash_watchdog(session=session, event_bus=event_bus)

        await watchdog._on_target_crash_cdp(target_id)
        event_bus.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_crash_no_target_info(self):
        session = _make_mock_browser_session()
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": []}
        )
        event_bus = MagicMock()
        watchdog = _make_crash_watchdog(session=session, event_bus=event_bus)

        await watchdog._on_target_crash_cdp("not-found")
        event_bus.dispatch.assert_called_once()


# ---------------------------------------------------------------------------
# _start_monitoring / _stop_monitoring
# ---------------------------------------------------------------------------

class TestMonitoringLifecycle:
    @pytest.mark.asyncio
    async def test_start_monitoring_asserts_cdp_client(self):
        session = _make_mock_browser_session()
        session.cdp_client = None
        watchdog = _make_crash_watchdog(session=session)

        with pytest.raises(AssertionError):
            await watchdog._start_monitoring()

    @pytest.mark.asyncio
    async def test_start_monitoring_skips_if_already_running(self):
        watchdog = _make_crash_watchdog()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        watchdog._monitoring_task = mock_task

        await watchdog._start_monitoring()
        # Should not create another task

    @pytest.mark.asyncio
    async def test_start_monitoring_creates_task(self):
        session = _make_mock_browser_session()
        session.cdp_client = MagicMock()
        watchdog = _make_crash_watchdog(session=session)

        # Patch _monitoring_loop to avoid actual loop
        async def mock_loop():
            pass

        with patch.object(watchdog, "_monitoring_loop", side_effect=mock_loop):
            await watchdog._start_monitoring()
            assert watchdog._monitoring_task is not None

        # Cleanup
        if watchdog._monitoring_task and not watchdog._monitoring_task.done():
            watchdog._monitoring_task.cancel()
            try:
                await watchdog._monitoring_task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_stop_monitoring_cancels_task(self):
        watchdog = _make_crash_watchdog()

        async def long_loop():
            await asyncio.sleep(100)

        watchdog._monitoring_task = asyncio.create_task(long_loop())
        await asyncio.sleep(0.01)  # let task start

        await watchdog._stop_monitoring()
        assert watchdog._monitoring_task.done() or watchdog._monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_monitoring_cancels_cdp_event_tasks(self):
        watchdog = _make_crash_watchdog()
        watchdog._monitoring_task = None  # no monitoring task

        async def slow():
            await asyncio.sleep(100)

        task = asyncio.create_task(slow())
        watchdog._cdp_event_tasks.add(task)

        await watchdog._stop_monitoring()
        assert len(watchdog._cdp_event_tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_monitoring_clears_state(self):
        watchdog = _make_crash_watchdog()
        watchdog._monitoring_task = None
        watchdog._active_requests["r1"] = MagicMock()
        watchdog._targets_with_listeners.add("t1")
        watchdog._last_responsive_checks["url"] = 1.0

        await watchdog._stop_monitoring()
        assert len(watchdog._active_requests) == 0
        assert len(watchdog._targets_with_listeners) == 0
        assert len(watchdog._last_responsive_checks) == 0


# ---------------------------------------------------------------------------
# _monitoring_loop
# ---------------------------------------------------------------------------

class TestMonitoringLoop:
    @pytest.mark.asyncio
    async def test_loop_catches_cancelled(self):
        watchdog = _make_crash_watchdog()

        with patch.object(watchdog, "_check_network_timeouts", new_callable=AsyncMock) as mock_net, \
             patch.object(watchdog, "_check_browser_health", new_callable=AsyncMock) as mock_health:
            # Immediately cancel after first sleep
            call_count = 0

            async def cancel_after_first_call(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise asyncio.CancelledError()

            mock_net.side_effect = cancel_after_first_call
            # Need to also handle the initial sleep
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                mock_sleep.side_effect = [None, asyncio.CancelledError()]
                await watchdog._monitoring_loop()

    @pytest.mark.asyncio
    async def test_loop_catches_generic_exception(self):
        watchdog = _make_crash_watchdog()

        call_count = 0

        async def fail_then_cancel(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return  # initial sleep
            elif call_count == 2:
                raise RuntimeError("test error")
            else:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=fail_then_cancel):
            with patch.object(watchdog, "_check_network_timeouts", new_callable=AsyncMock):
                with patch.object(watchdog, "_check_browser_health", new_callable=AsyncMock):
                    await watchdog._monitoring_loop()


# ---------------------------------------------------------------------------
# _check_network_timeouts
# ---------------------------------------------------------------------------

class TestCheckNetworkTimeouts:
    @pytest.mark.asyncio
    async def test_no_active_requests(self):
        watchdog = _make_crash_watchdog()
        await watchdog._check_network_timeouts()
        # No error, no dispatch

    @pytest.mark.asyncio
    async def test_request_not_timed_out(self):
        from openbrowser.browser.watchdogs.crash_watchdog import NetworkRequestTracker

        watchdog = _make_crash_watchdog()
        watchdog._active_requests["r1"] = NetworkRequestTracker(
            request_id="r1",
            start_time=time.time(),
            url="https://example.com",
            method="GET",
        )
        event_bus = MagicMock()
        watchdog.event_bus = event_bus

        await watchdog._check_network_timeouts()
        event_bus.dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_timed_out_request_dispatches_event(self):
        from openbrowser.browser.watchdogs.crash_watchdog import NetworkRequestTracker

        watchdog = _make_crash_watchdog()
        watchdog._active_requests["r2"] = NetworkRequestTracker(
            request_id="r2",
            start_time=time.time() - 20.0,  # 20 seconds ago, well past 10s timeout
            url="https://slow.example.com/api",
            method="POST",
            resource_type="XHR",
        )
        event_bus = MagicMock()
        watchdog.event_bus = event_bus

        await watchdog._check_network_timeouts()
        event_bus.dispatch.assert_called_once()
        args = event_bus.dispatch.call_args[0][0]
        assert args.error_type == "NetworkTimeout"
        assert "r2" not in watchdog._active_requests


# ---------------------------------------------------------------------------
# _check_browser_health
# ---------------------------------------------------------------------------

class TestCheckBrowserHealth:
    @pytest.mark.asyncio
    async def test_health_check_passes(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.navigate = AsyncMock()
        cdp_session_mock.session_id = "session-123"
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": []}
        )
        session._local_browser_watchdog = None

        watchdog = _make_crash_watchdog(session=session)
        await watchdog._check_browser_health()

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        session = _make_mock_browser_session()
        session.get_or_create_cdp_session = AsyncMock(side_effect=RuntimeError("connection lost"))
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={"targetInfos": []}
        )
        session._local_browser_watchdog = None

        watchdog = _make_crash_watchdog(session=session)
        # Should not raise
        await watchdog._check_browser_health()

    @pytest.mark.asyncio
    async def test_health_check_redirects_new_tab_page(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})
        cdp_session_mock.cdp_client.send.Page = MagicMock()
        cdp_session_mock.cdp_client.send.Page.navigate = AsyncMock()
        cdp_session_mock.session_id = "session-456"

        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session.cdp_client.send.Target.getTargets = AsyncMock(
            return_value={
                "targetInfos": [
                    {"targetId": "tid1", "type": "page", "url": "chrome://new-tab-page/"}
                ]
            }
        )
        session._local_browser_watchdog = None

        watchdog = _make_crash_watchdog(session=session)
        await watchdog._check_browser_health()
        cdp_session_mock.cdp_client.send.Page.navigate.assert_awaited()

    @pytest.mark.asyncio
    async def test_health_check_browser_process_crashed(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})
        cdp_session_mock.session_id = "session-789"

        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

        # Simulate zombie browser process
        mock_proc = MagicMock()
        mock_proc.status.return_value = "zombie"
        mock_proc.pid = 12345
        mock_local_watchdog = MagicMock()
        mock_local_watchdog._subprocess = mock_proc
        session._local_browser_watchdog = mock_local_watchdog

        event_bus = MagicMock()
        watchdog = _make_crash_watchdog(session=session, event_bus=event_bus)

        with patch.object(watchdog, "_stop_monitoring", new_callable=AsyncMock) as mock_stop:
            await watchdog._check_browser_health()
            event_bus.dispatch.assert_called_once()
            args = event_bus.dispatch.call_args[0][0]
            assert args.error_type == "BrowserProcessCrashed"
            mock_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_health_check_process_status_exception(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})
        cdp_session_mock.session_id = "session-abc"

        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

        # Process raises an exception on .status()
        mock_proc = MagicMock()
        mock_proc.status.side_effect = Exception("psutil error")
        mock_local_watchdog = MagicMock()
        mock_local_watchdog._subprocess = mock_proc
        session._local_browser_watchdog = mock_local_watchdog

        watchdog = _make_crash_watchdog(session=session)
        # Should not raise
        await watchdog._check_browser_health()

    @pytest.mark.asyncio
    async def test_health_check_running_process(self):
        session = _make_mock_browser_session()
        cdp_session_mock = MagicMock()
        cdp_session_mock.cdp_client = MagicMock()
        cdp_session_mock.cdp_client.send = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime = MagicMock()
        cdp_session_mock.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})
        cdp_session_mock.session_id = "session-def"

        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session_mock)
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={"targetInfos": []})

        # Process is healthy
        mock_proc = MagicMock()
        mock_proc.status.return_value = "running"
        mock_proc.pid = 99999
        mock_local_watchdog = MagicMock()
        mock_local_watchdog._subprocess = mock_proc
        session._local_browser_watchdog = mock_local_watchdog

        event_bus = MagicMock()
        watchdog = _make_crash_watchdog(session=session, event_bus=event_bus)
        await watchdog._check_browser_health()
        event_bus.dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# _is_new_tab_page (static)
# ---------------------------------------------------------------------------

class TestIsNewTabPage:
    def test_about_blank(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog

        assert CrashWatchdog._is_new_tab_page("about:blank") is True

    def test_chrome_new_tab_page(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog

        assert CrashWatchdog._is_new_tab_page("chrome://new-tab-page/") is True

    def test_chrome_newtab(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog

        assert CrashWatchdog._is_new_tab_page("chrome://newtab/") is True

    def test_regular_url(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog

        assert CrashWatchdog._is_new_tab_page("https://example.com") is False

    def test_empty_string(self):
        from openbrowser.browser.watchdogs.crash_watchdog import CrashWatchdog

        assert CrashWatchdog._is_new_tab_page("") is False
