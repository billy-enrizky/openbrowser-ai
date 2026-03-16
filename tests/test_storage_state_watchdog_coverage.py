"""Tests for openbrowser.browser.watchdogs.storage_state_watchdog module -- 100% coverage target."""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser_session():
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger("test_storage_state_watchdog")
    session.event_bus = MagicMock()
    session._cdp_client_root = MagicMock()
    session.agent_focus = MagicMock()
    session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock())
    session.cdp_client = MagicMock()
    session._cdp_get_cookies = AsyncMock(return_value=[])
    session._cdp_set_cookies = AsyncMock()
    session._cdp_get_storage_state = AsyncMock(return_value={"cookies": [], "origins": []})
    session._cdp_add_init_script = AsyncMock()
    session.id = "test-storage-session"
    session.is_local = True

    profile = MagicMock()
    profile.storage_state = None
    session.browser_profile = profile

    return session


def _make_storage_watchdog(session=None, event_bus=None):
    from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

    if session is None:
        session = _make_mock_browser_session()
    if event_bus is None:
        event_bus = MagicMock()
    session.event_bus = event_bus
    return StorageStateWatchdog.model_construct(
        event_bus=event_bus,
        browser_session=session,
        auto_save_interval=30.0,
        save_on_change=True,
    )


# ---------------------------------------------------------------------------
# on_BrowserConnectedEvent
# ---------------------------------------------------------------------------

class TestOnBrowserConnectedEvent:
    @pytest.mark.asyncio
    async def test_starts_monitoring_and_loads(self):
        session = _make_mock_browser_session()
        session.cdp_client = MagicMock()
        event_bus = MagicMock()
        event_bus.dispatch = AsyncMock()
        watchdog = _make_storage_watchdog(session=session, event_bus=event_bus)

        with patch.object(watchdog, "_start_monitoring", new_callable=AsyncMock) as mock_start:
            event = MagicMock()
            await watchdog.on_BrowserConnectedEvent(event)
            mock_start.assert_awaited_once()
            event_bus.dispatch.assert_awaited()


# ---------------------------------------------------------------------------
# on_BrowserStopEvent
# ---------------------------------------------------------------------------

class TestOnBrowserStopEvent:
    @pytest.mark.asyncio
    async def test_stops_monitoring(self):
        watchdog = _make_storage_watchdog()
        with patch.object(watchdog, "_stop_monitoring", new_callable=AsyncMock) as mock_stop:
            event = MagicMock()
            await watchdog.on_BrowserStopEvent(event)
            mock_stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# on_SaveStorageStateEvent
# ---------------------------------------------------------------------------

class TestOnSaveStorageStateEvent:
    @pytest.mark.asyncio
    async def test_uses_event_path(self):
        watchdog = _make_storage_watchdog()
        event = MagicMock()
        event.path = "/tmp/test_state.json"

        with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
            await watchdog.on_SaveStorageStateEvent(event)
            mock_save.assert_awaited_once_with("/tmp/test_state.json")

    @pytest.mark.asyncio
    async def test_falls_back_to_profile_path(self):
        session = _make_mock_browser_session()
        session.browser_profile.storage_state = "/tmp/profile_state.json"
        watchdog = _make_storage_watchdog(session=session)

        event = MagicMock()
        event.path = None

        with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
            await watchdog.on_SaveStorageStateEvent(event)
            mock_save.assert_awaited_once_with("/tmp/profile_state.json")

    @pytest.mark.asyncio
    async def test_no_path_available(self):
        session = _make_mock_browser_session()
        session.browser_profile.storage_state = None
        watchdog = _make_storage_watchdog(session=session)

        event = MagicMock()
        event.path = None

        with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
            await watchdog.on_SaveStorageStateEvent(event)
            mock_save.assert_awaited_once_with(None)


# ---------------------------------------------------------------------------
# on_LoadStorageStateEvent
# ---------------------------------------------------------------------------

class TestOnLoadStorageStateEvent:
    @pytest.mark.asyncio
    async def test_uses_event_path(self):
        watchdog = _make_storage_watchdog()
        event = MagicMock()
        event.path = "/tmp/load_state.json"

        with patch.object(watchdog, "_load_storage_state", new_callable=AsyncMock) as mock_load:
            await watchdog.on_LoadStorageStateEvent(event)
            mock_load.assert_awaited_once_with("/tmp/load_state.json")

    @pytest.mark.asyncio
    async def test_falls_back_to_profile_path(self):
        session = _make_mock_browser_session()
        session.browser_profile.storage_state = "/tmp/profile_load.json"
        watchdog = _make_storage_watchdog(session=session)

        event = MagicMock()
        event.path = None

        with patch.object(watchdog, "_load_storage_state", new_callable=AsyncMock) as mock_load:
            await watchdog.on_LoadStorageStateEvent(event)
            mock_load.assert_awaited_once_with("/tmp/profile_load.json")

    @pytest.mark.asyncio
    async def test_no_path_available(self):
        session = _make_mock_browser_session()
        session.browser_profile.storage_state = None
        watchdog = _make_storage_watchdog(session=session)

        event = MagicMock()
        event.path = None

        with patch.object(watchdog, "_load_storage_state", new_callable=AsyncMock) as mock_load:
            await watchdog.on_LoadStorageStateEvent(event)
            mock_load.assert_awaited_once_with(None)


# ---------------------------------------------------------------------------
# _start_monitoring / _stop_monitoring
# ---------------------------------------------------------------------------

class TestMonitoringLifecycle:
    @pytest.mark.asyncio
    async def test_start_monitoring_skips_if_running(self):
        watchdog = _make_storage_watchdog()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        watchdog._monitoring_task = mock_task

        await watchdog._start_monitoring()
        # Should not create another task

    @pytest.mark.asyncio
    async def test_start_monitoring_asserts_cdp_client(self):
        session = _make_mock_browser_session()
        session.cdp_client = None
        watchdog = _make_storage_watchdog(session=session)

        with pytest.raises(AssertionError):
            await watchdog._start_monitoring()

    @pytest.mark.asyncio
    async def test_start_monitoring_creates_task(self):
        session = _make_mock_browser_session()
        session.cdp_client = MagicMock()
        watchdog = _make_storage_watchdog(session=session)

        async def mock_monitor():
            pass

        with patch.object(watchdog, "_monitor_storage_changes", side_effect=mock_monitor):
            await watchdog._start_monitoring()
            assert watchdog._monitoring_task is not None

        if watchdog._monitoring_task and not watchdog._monitoring_task.done():
            watchdog._monitoring_task.cancel()
            try:
                await watchdog._monitoring_task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_stop_monitoring_cancels_task(self):
        watchdog = _make_storage_watchdog()

        async def long_task():
            await asyncio.sleep(100)

        watchdog._monitoring_task = asyncio.create_task(long_task())
        await asyncio.sleep(0.01)

        await watchdog._stop_monitoring()
        assert watchdog._monitoring_task.done() or watchdog._monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_monitoring_noop_if_no_task(self):
        watchdog = _make_storage_watchdog()
        watchdog._monitoring_task = None
        await watchdog._stop_monitoring()


# ---------------------------------------------------------------------------
# _check_for_cookie_changes_cdp
# ---------------------------------------------------------------------------

class TestCheckForCookieChanges:
    @pytest.mark.asyncio
    async def test_detects_set_cookie_header(self):
        watchdog = _make_storage_watchdog()
        event = {"headers": {"Set-Cookie": "session=abc123"}}

        with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
            await watchdog._check_for_cookie_changes_cdp(event)
            mock_save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_detects_lowercase_set_cookie(self):
        watchdog = _make_storage_watchdog()
        event = {"headers": {"set-cookie": "test=1"}}

        with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
            await watchdog._check_for_cookie_changes_cdp(event)
            mock_save.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_cookie_change(self):
        watchdog = _make_storage_watchdog()
        event = {"headers": {"Content-Type": "text/html"}}

        with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
            await watchdog._check_for_cookie_changes_cdp(event)
            mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_on_change_disabled(self):
        watchdog = _make_storage_watchdog()
        watchdog.save_on_change = False
        event = {"headers": {"Set-Cookie": "x=1"}}

        with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
            await watchdog._check_for_cookie_changes_cdp(event)
            mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        watchdog = _make_storage_watchdog()
        event = {"headers": None}  # Will cause attribute error
        # Should not raise
        await watchdog._check_for_cookie_changes_cdp(event)


# ---------------------------------------------------------------------------
# _monitor_storage_changes
# ---------------------------------------------------------------------------

class TestMonitorStorageChanges:
    @pytest.mark.asyncio
    async def test_saves_on_cookie_change(self):
        watchdog = _make_storage_watchdog()
        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=mock_sleep):
            with patch.object(watchdog, "_have_cookies_changed", new_callable=AsyncMock, return_value=True):
                with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
                    await watchdog._monitor_storage_changes()
                    mock_save.assert_awaited()

    @pytest.mark.asyncio
    async def test_no_save_when_unchanged(self):
        watchdog = _make_storage_watchdog()
        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=mock_sleep):
            with patch.object(watchdog, "_have_cookies_changed", new_callable=AsyncMock, return_value=False):
                with patch.object(watchdog, "_save_storage_state", new_callable=AsyncMock) as mock_save:
                    await watchdog._monitor_storage_changes()
                    mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handles_exception_in_loop(self):
        watchdog = _make_storage_watchdog()
        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return
            raise asyncio.CancelledError()

        with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=mock_sleep):
            with patch.object(watchdog, "_have_cookies_changed", new_callable=AsyncMock, side_effect=RuntimeError("err")):
                await watchdog._monitor_storage_changes()


# ---------------------------------------------------------------------------
# _have_cookies_changed
# ---------------------------------------------------------------------------

class TestHaveCookiesChanged:
    @pytest.mark.asyncio
    async def test_no_cdp_client(self):
        session = _make_mock_browser_session()
        session.cdp_client = None
        watchdog = _make_storage_watchdog(session=session)
        result = await watchdog._have_cookies_changed()
        assert result is False

    @pytest.mark.asyncio
    async def test_cookies_unchanged(self):
        session = _make_mock_browser_session()
        session._cdp_get_cookies = AsyncMock(return_value=[
            {"name": "a", "domain": ".example.com", "path": "/", "value": "1"}
        ])
        watchdog = _make_storage_watchdog(session=session)
        watchdog._last_cookie_state = [
            {"name": "a", "domain": ".example.com", "path": "/", "value": "1"}
        ]
        result = await watchdog._have_cookies_changed()
        assert result is False

    @pytest.mark.asyncio
    async def test_cookies_changed(self):
        session = _make_mock_browser_session()
        session._cdp_get_cookies = AsyncMock(return_value=[
            {"name": "a", "domain": ".example.com", "path": "/", "value": "2"}
        ])
        watchdog = _make_storage_watchdog(session=session)
        watchdog._last_cookie_state = [
            {"name": "a", "domain": ".example.com", "path": "/", "value": "1"}
        ]
        result = await watchdog._have_cookies_changed()
        assert result is True

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        session = _make_mock_browser_session()
        session._cdp_get_cookies = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_storage_watchdog(session=session)
        result = await watchdog._have_cookies_changed()
        assert result is False


# ---------------------------------------------------------------------------
# _save_storage_state
# ---------------------------------------------------------------------------

class TestSaveStorageState:
    @pytest.mark.asyncio
    async def test_no_save_path(self):
        session = _make_mock_browser_session()
        session.browser_profile.storage_state = None
        watchdog = _make_storage_watchdog(session=session)
        await watchdog._save_storage_state(path=None)
        # Should return early

    @pytest.mark.asyncio
    async def test_dict_save_path_skipped(self):
        session = _make_mock_browser_session()
        session.browser_profile.storage_state = {"cookies": []}
        watchdog = _make_storage_watchdog(session=session)
        await watchdog._save_storage_state(path=None)
        # Should skip because it's a dict

    @pytest.mark.asyncio
    async def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "state.json")
            session = _make_mock_browser_session()
            session._cdp_get_storage_state = AsyncMock(return_value={
                "cookies": [{"name": "a", "domain": ".x.com", "path": "/", "value": "1"}],
                "origins": []
            })
            event_bus = MagicMock()
            watchdog = _make_storage_watchdog(session=session, event_bus=event_bus)

            await watchdog._save_storage_state(path=save_path)

            assert os.path.exists(save_path)
            with open(save_path) as f:
                data = json.load(f)
            assert len(data["cookies"]) == 1
            event_bus.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_merges_with_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "state.json")
            # Write existing state
            existing = {
                "cookies": [{"name": "old", "domain": ".x.com", "path": "/", "value": "old_val"}],
                "origins": [{"origin": "https://x.com", "localStorage": []}]
            }
            with open(save_path, "w") as f:
                json.dump(existing, f)

            session = _make_mock_browser_session()
            session._cdp_get_storage_state = AsyncMock(return_value={
                "cookies": [{"name": "new", "domain": ".y.com", "path": "/", "value": "new_val"}],
                "origins": [{"origin": "https://y.com", "localStorage": []}]
            })
            event_bus = MagicMock()
            watchdog = _make_storage_watchdog(session=session, event_bus=event_bus)

            await watchdog._save_storage_state(path=save_path)

            with open(save_path) as f:
                data = json.load(f)
            assert len(data["cookies"]) == 2
            assert len(data["origins"]) == 2

    @pytest.mark.asyncio
    async def test_save_exception(self):
        session = _make_mock_browser_session()
        session._cdp_get_storage_state = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_storage_watchdog(session=session)

        # Should not raise
        await watchdog._save_storage_state(path="/tmp/impossible_path_test/state.json")

    @pytest.mark.asyncio
    async def test_merge_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "state.json")
            # Write invalid existing state
            with open(save_path, "w") as f:
                f.write("not json")

            session = _make_mock_browser_session()
            session._cdp_get_storage_state = AsyncMock(return_value={
                "cookies": [], "origins": []
            })
            event_bus = MagicMock()
            watchdog = _make_storage_watchdog(session=session, event_bus=event_bus)

            await watchdog._save_storage_state(path=save_path)
            # Should still succeed, just not merge
            assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# _load_storage_state
# ---------------------------------------------------------------------------

class TestLoadStorageState:
    @pytest.mark.asyncio
    async def test_no_cdp_client(self):
        session = _make_mock_browser_session()
        session.cdp_client = None
        watchdog = _make_storage_watchdog(session=session)
        await watchdog._load_storage_state(path="/tmp/state.json")

    @pytest.mark.asyncio
    async def test_no_path(self):
        session = _make_mock_browser_session()
        session.browser_profile.storage_state = None
        watchdog = _make_storage_watchdog(session=session)
        await watchdog._load_storage_state(path=None)

    @pytest.mark.asyncio
    async def test_nonexistent_file(self):
        watchdog = _make_storage_watchdog()
        await watchdog._load_storage_state(path="/tmp/nonexistent_state_file_xyz.json")

    @pytest.mark.asyncio
    async def test_loads_cookies_and_origins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            state = {
                "cookies": [{"name": "a", "domain": ".x.com", "path": "/", "value": "1"}],
                "origins": [
                    {
                        "origin": "https://x.com",
                        "localStorage": [{"name": "key1", "value": "val1"}],
                        "sessionStorage": [{"name": "sk1", "value": "sv1"}]
                    }
                ]
            }
            with open(state_path, "w") as f:
                json.dump(state, f)

            session = _make_mock_browser_session()
            event_bus = MagicMock()
            watchdog = _make_storage_watchdog(session=session, event_bus=event_bus)

            await watchdog._load_storage_state(path=state_path)

            session._cdp_set_cookies.assert_awaited_once()
            assert session._cdp_add_init_script.await_count == 2  # localStorage + sessionStorage
            event_bus.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_loads_empty_cookies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            state = {"cookies": [], "origins": []}
            with open(state_path, "w") as f:
                json.dump(state, f)

            session = _make_mock_browser_session()
            event_bus = MagicMock()
            watchdog = _make_storage_watchdog(session=session, event_bus=event_bus)

            await watchdog._load_storage_state(path=state_path)
            session._cdp_set_cookies.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_load_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "bad_state.json")
            with open(state_path, "w") as f:
                f.write("{invalid json")

            watchdog = _make_storage_watchdog()
            # Should not raise
            await watchdog._load_storage_state(path=state_path)


# ---------------------------------------------------------------------------
# _merge_storage_states (static)
# ---------------------------------------------------------------------------

class TestMergeStorageStates:
    def test_merge_new_cookies(self):
        from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

        existing = {
            "cookies": [{"name": "a", "domain": ".x.com", "path": "/", "value": "1"}],
            "origins": [{"origin": "https://x.com"}]
        }
        new = {
            "cookies": [{"name": "b", "domain": ".y.com", "path": "/", "value": "2"}],
            "origins": [{"origin": "https://y.com"}]
        }
        result = StorageStateWatchdog._merge_storage_states(existing, new)
        assert len(result["cookies"]) == 2
        assert len(result["origins"]) == 2

    def test_merge_overwrites_existing(self):
        from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

        existing = {
            "cookies": [{"name": "a", "domain": ".x.com", "path": "/", "value": "old"}],
            "origins": []
        }
        new = {
            "cookies": [{"name": "a", "domain": ".x.com", "path": "/", "value": "new"}],
            "origins": []
        }
        result = StorageStateWatchdog._merge_storage_states(existing, new)
        assert len(result["cookies"]) == 1
        assert result["cookies"][0]["value"] == "new"

    def test_merge_empty_states(self):
        from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

        result = StorageStateWatchdog._merge_storage_states({}, {})
        assert result["cookies"] == []
        assert result["origins"] == []


# ---------------------------------------------------------------------------
# get_current_cookies
# ---------------------------------------------------------------------------

class TestGetCurrentCookies:
    @pytest.mark.asyncio
    async def test_no_cdp_client(self):
        session = _make_mock_browser_session()
        session.cdp_client = None
        watchdog = _make_storage_watchdog(session=session)
        result = await watchdog.get_current_cookies()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_cookies(self):
        session = _make_mock_browser_session()
        session._cdp_get_cookies = AsyncMock(return_value=[
            {"name": "a", "value": "1", "domain": ".x.com"}
        ])
        watchdog = _make_storage_watchdog(session=session)
        result = await watchdog.get_current_cookies()
        assert len(result) == 1
        assert result[0]["name"] == "a"

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        session = _make_mock_browser_session()
        session._cdp_get_cookies = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_storage_watchdog(session=session)
        result = await watchdog.get_current_cookies()
        assert result == []


# ---------------------------------------------------------------------------
# add_cookies
# ---------------------------------------------------------------------------

class TestAddCookies:
    @pytest.mark.asyncio
    async def test_no_cdp_client(self):
        session = _make_mock_browser_session()
        session.cdp_client = None
        watchdog = _make_storage_watchdog(session=session)
        await watchdog.add_cookies([{"name": "a", "value": "1"}])

    @pytest.mark.asyncio
    async def test_adds_cookies(self):
        session = _make_mock_browser_session()
        watchdog = _make_storage_watchdog(session=session)
        cookies = [{"name": "a", "value": "1", "domain": ".x.com", "path": "/"}]
        await watchdog.add_cookies(cookies)
        session._cdp_set_cookies.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_cookies_exception(self):
        session = _make_mock_browser_session()
        session._cdp_set_cookies = AsyncMock(side_effect=RuntimeError("fail"))
        watchdog = _make_storage_watchdog(session=session)
        # Should not raise
        await watchdog.add_cookies([{"name": "a", "value": "1"}])
