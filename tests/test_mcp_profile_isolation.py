"""Tests for per-instance MCP browser profiles."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import DummyServer


@pytest.fixture
def isolated_mcp(monkeypatch, tmp_path):
    from openbrowser.mcp import server as mcp_mod

    monkeypatch.setenv('OPENBROWSER_CONFIG_DIR', str(tmp_path / 'config'))
    profiles_dir = tmp_path / 'config' / 'profiles'
    default_state = profiles_dir / 'default' / 'storage_state.json'
    default_state.parent.mkdir(parents=True)
    default_state.write_text(json.dumps({'cookies': [{'name': 'seed'}], 'origins': []}), encoding='utf-8')

    monkeypatch.setattr(mcp_mod, 'Server', DummyServer)
    monkeypatch.setattr(mcp_mod, 'TELEMETRY_AVAILABLE', False)
    monkeypatch.setattr(mcp_mod, 'load_openbrowser_config', lambda: {'browser_profile': {}})
    import openbrowser.config as config_mod
    config_mod.CONFIG._old_config = None
    config_mod.CONFIG._env_config = None
    return mcp_mod, profiles_dir, default_state


def test_default_mcp_servers_get_distinct_stable_profiles(isolated_mcp):
    mcp_mod, profiles_dir, default_state = isolated_mcp
    first = mcp_mod.OpenBrowserServer()
    second = mcp_mod.OpenBrowserServer()

    first_profile = first._build_browser_profile()
    first_again = first._build_browser_profile()
    second_profile = second._build_browser_profile()

    assert first_profile.user_data_dir != second_profile.user_data_dir
    assert first_profile.user_data_dir == first_again.user_data_dir
    assert str(first_profile.user_data_dir).startswith(str(profiles_dir.resolve()))
    assert str(first_profile.user_data_dir).endswith(first._instance_id)
    assert str(first_profile.storage_state).endswith('storage_state.json')
    assert Path(first_profile.storage_state).read_text(encoding='utf-8') == default_state.read_text(encoding='utf-8')
    assert first_profile.storage_state != second_profile.storage_state


def test_default_storage_state_is_seeded_only_once(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    profile = server._build_browser_profile()
    state_path = Path(profile.storage_state)
    state_path.write_text(json.dumps({'cookies': [{'name': 'changed'}]}), encoding='utf-8')

    server._build_browser_profile()

    assert json.loads(state_path.read_text(encoding='utf-8'))['cookies'][0]['name'] == 'changed'


def test_configured_storage_state_is_copied_per_instance(isolated_mcp, tmp_path):
    mcp_mod, profiles_dir, _ = isolated_mcp
    configured_state = tmp_path / 'configured-storage-state.json'
    configured_state.write_text(json.dumps({'cookies': [{'name': 'configured'}]}), encoding='utf-8')
    with patch.object(
        mcp_mod,
        'load_openbrowser_config',
        return_value={'browser_profile': {'storage_state': str(configured_state)}},
    ):
        first = mcp_mod.OpenBrowserServer()
        second = mcp_mod.OpenBrowserServer()

        first_profile = first._build_browser_profile()
        second_profile = second._build_browser_profile()

    assert Path(first_profile.storage_state).parent == Path(first_profile.user_data_dir)
    assert Path(second_profile.storage_state).parent == Path(second_profile.user_data_dir)
    assert first_profile.storage_state != second_profile.storage_state
    assert Path(first_profile.storage_state).read_text(encoding='utf-8') == configured_state.read_text(encoding='utf-8')
    assert Path(second_profile.storage_state).read_text(encoding='utf-8') == configured_state.read_text(encoding='utf-8')
    assert str(first_profile.user_data_dir).startswith(str(profiles_dir.resolve()))


def test_in_memory_storage_state_is_preserved_per_instance(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    state = {'cookies': [{'name': 'in-memory'}], 'origins': []}
    with patch.object(
        mcp_mod,
        'load_openbrowser_config',
        return_value={'browser_profile': {'storage_state': state}},
    ):
        first = mcp_mod.OpenBrowserServer()
        second = mcp_mod.OpenBrowserServer()

        first_profile = first._build_browser_profile()
        second_profile = second._build_browser_profile()

    assert first_profile.storage_state == state
    assert second_profile.storage_state == state
    assert first_profile.storage_state is not state
    assert second_profile.storage_state is not state
    assert first_profile.storage_state is not second_profile.storage_state


def test_explicit_profile_path_is_preserved(isolated_mcp, tmp_path, monkeypatch):
    mcp_mod, _, _ = isolated_mcp
    shared_profile = tmp_path / 'shared'
    monkeypatch.setattr(mcp_mod, 'load_openbrowser_config', lambda: {'browser_profile': {'user_data_dir': str(shared_profile)}})
    server = mcp_mod.OpenBrowserServer()

    profile = server._build_browser_profile()

    assert Path(profile.user_data_dir).resolve() == shared_profile.resolve()


@pytest.mark.asyncio
async def test_recovery_does_not_run_unscoped_profile_kill(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    old_session = MagicMock()
    old_session.kill = AsyncMock()
    old_session.browser_profile.user_data_dir = '/tmp/owned-profile'
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    new_session = MagicMock()
    new_session.start = AsyncMock()
    with (
        patch.object(mcp_mod, 'BrowserSession', return_value=new_session),
        patch.object(mcp_mod, 'CodeAgentTools', return_value=MagicMock()),
        patch.object(mcp_mod, 'create_namespace', return_value={}),
        patch(
            'openbrowser.browser.watchdogs.local_browser_watchdog.LocalBrowserWatchdog._kill_stale_chrome_for_profile',
            new_callable=AsyncMock,
        ) as stale_kill,
    ):
        await server._recover_browser_session()

    stale_kill.assert_not_awaited()
    assert server.browser_session is new_session


@pytest.mark.asyncio
async def test_cdp_health_probe_is_bounded(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session._cdp_client_root = MagicMock()
    session._cdp_client_root.send.Browser.getVersion = MagicMock()
    server.browser_session = session

    with patch.object(mcp_mod.asyncio, 'wait_for', new_callable=AsyncMock, side_effect=asyncio.TimeoutError) as wait_for:
        assert await server._is_cdp_alive() is False

    wait_for.assert_awaited_once()
    assert wait_for.await_args.kwargs['timeout'] == mcp_mod._CDP_HEALTH_CHECK_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_namespace_initialization_is_serialized(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    start_calls = 0

    async def start_session():
        nonlocal start_calls
        start_calls += 1
        await asyncio.sleep(0)

    session = MagicMock()
    session.start = start_session
    with (
        patch.object(mcp_mod, 'BrowserSession', return_value=session),
        patch.object(mcp_mod, 'CodeAgentTools', return_value=MagicMock()),
        patch.object(mcp_mod, 'create_namespace', return_value={}),
    ):
        await asyncio.gather(server._ensure_namespace(), server._ensure_namespace())

    assert start_calls == 1


@pytest.mark.asyncio
async def test_browser_recovery_is_serialized_and_rechecks_health(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    old_session = MagicMock()
    old_session.kill = AsyncMock()
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    new_session = MagicMock()
    new_session.start = AsyncMock()
    with patch.object(server, '_is_cdp_alive', new_callable=AsyncMock, side_effect=[False, True]) as is_cdp_alive:
        with (
            patch.object(mcp_mod, 'BrowserSession', return_value=new_session),
            patch.object(mcp_mod, 'CodeAgentTools', return_value=MagicMock()),
            patch.object(mcp_mod, 'create_namespace', return_value={}),
        ):
            await asyncio.gather(server._recover_browser_session(), server._recover_browser_session())

    old_session.kill.assert_awaited_once()
    new_session.start.assert_awaited_once()
    assert is_cdp_alive.await_count == 2


@pytest.mark.asyncio
async def test_recovery_cleans_up_a_failed_replacement(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    old_session = MagicMock()
    old_session.kill = AsyncMock()
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    replacement = MagicMock()
    replacement.start = AsyncMock(side_effect=RuntimeError('replacement failed'))
    replacement.kill = AsyncMock()
    with (
        patch.object(server, '_is_cdp_alive', new_callable=AsyncMock, return_value=False),
        patch.object(mcp_mod, 'BrowserSession', return_value=replacement),
    ):
        with pytest.raises(RuntimeError, match='replacement failed'):
            await server._recover_browser_session()

    replacement.kill.assert_awaited_once()
    assert server.browser_session is None
    assert server._namespace is None


@pytest.mark.asyncio
async def test_recovery_cleans_up_namespace_build_failure(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    old_session = MagicMock()
    old_session.kill = AsyncMock()
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    replacement = MagicMock()
    replacement.start = AsyncMock()
    replacement.kill = AsyncMock()
    with (
        patch.object(server, '_is_cdp_alive', new_callable=AsyncMock, return_value=False),
        patch.object(mcp_mod, 'BrowserSession', return_value=replacement),
        patch.object(mcp_mod, 'create_namespace', side_effect=RuntimeError('namespace failed')),
    ):
        with pytest.raises(RuntimeError, match='namespace failed'):
            await server._recover_browser_session()

    replacement.kill.assert_awaited_once()
    assert server.browser_session is None
    assert server._namespace is None


@pytest.mark.asyncio
async def test_initialization_retains_session_when_cleanup_fails(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session.start = AsyncMock()
    session.kill = AsyncMock(side_effect=RuntimeError('cleanup failed'))

    with (
        patch.object(mcp_mod, 'BrowserSession', return_value=session),
        patch.object(mcp_mod, 'CodeAgentTools', return_value=MagicMock()),
        patch.object(mcp_mod, 'create_namespace', side_effect=RuntimeError('namespace failed')),
    ):
        with pytest.raises(RuntimeError, match='namespace failed'):
            await server._ensure_namespace()

    assert server.browser_session is session
    assert server._namespace is None


@pytest.mark.asyncio
async def test_cancelled_initialization_clears_session_after_cleanup(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session.start = AsyncMock(side_effect=RuntimeError('startup failed'))

    with (
        patch.object(mcp_mod, 'BrowserSession', return_value=session),
        patch.object(server, '_kill_session_safely', new_callable=AsyncMock, return_value=(True, True)),
    ):
        with pytest.raises(asyncio.CancelledError):
            await server._ensure_namespace()

    assert server.browser_session is None
    assert server._namespace is None


@pytest.mark.asyncio
async def test_recovery_retains_replacement_when_cleanup_fails(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    old_session = MagicMock()
    old_session.kill = AsyncMock()
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    replacement = MagicMock()
    replacement.start = AsyncMock()
    replacement.kill = AsyncMock(side_effect=RuntimeError('replacement cleanup failed'))
    with (
        patch.object(server, '_is_cdp_alive', new_callable=AsyncMock, return_value=False),
        patch.object(mcp_mod, 'BrowserSession', return_value=replacement),
        patch.object(mcp_mod, 'CodeAgentTools', return_value=MagicMock()),
        patch.object(mcp_mod, 'create_namespace', side_effect=RuntimeError('namespace failed')),
    ):
        with pytest.raises(RuntimeError, match='namespace failed'):
            await server._recover_browser_session()

    assert server.browser_session is replacement
    assert server._namespace is None


@pytest.mark.asyncio
async def test_recovery_profile_build_failure_clears_old_namespace(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    old_session = MagicMock()
    old_session.kill = AsyncMock()
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    with (
        patch.object(server, '_is_cdp_alive', new_callable=AsyncMock, return_value=False),
        patch.object(server, '_build_browser_profile', side_effect=RuntimeError('profile failed')),
    ):
        with pytest.raises(RuntimeError, match='profile failed'):
            await server._recover_browser_session()

    assert server.browser_session is None
    assert server._namespace is None


@pytest.mark.asyncio
async def test_recovery_aborts_when_old_session_cleanup_fails(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    old_session = MagicMock()
    old_session.kill = AsyncMock(side_effect=RuntimeError('old cleanup failed'))
    old_session.reset = AsyncMock()
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    with patch.object(server, '_is_cdp_alive', new_callable=AsyncMock, return_value=False):
        with pytest.raises(RuntimeError, match='Unable to safely close'):
            await server._recover_browser_session()

    old_session.reset.assert_not_awaited()
    assert server.browser_session is old_session


@pytest.mark.asyncio
async def test_cleanup_waits_for_active_execution(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer(session_timeout_minutes=0)
    session = MagicMock()
    session.kill = AsyncMock()
    server.browser_session = session
    server._namespace = {'value': 1}
    server._is_cdp_alive = AsyncMock(return_value=True)
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()

    async def execute(_code):
        execution_started.set()
        await release_execution.wait()
        return MagicMock(success=True, error=None, output='done')

    server._executor = MagicMock(initialized=True, execute=execute)
    execute_task = asyncio.create_task(server._execute_code('value = 1'))
    await execution_started.wait()
    server._last_activity = 0
    cleanup_task = asyncio.create_task(server._cleanup_expired_session())
    await asyncio.sleep(0)

    session.kill.assert_not_awaited()
    release_execution.set()
    await execute_task
    await cleanup_task
    session.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_waits_for_active_execution(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session.kill = AsyncMock()
    server.browser_session = session
    server._namespace = {'value': 1}
    server._is_cdp_alive = AsyncMock(return_value=True)
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()

    async def execute(_code):
        execution_started.set()
        await release_execution.wait()
        return MagicMock(success=True, error=None, output='done')

    server._executor = MagicMock(initialized=True, execute=execute)
    execute_task = asyncio.create_task(server._execute_code('value = 1'))
    await execution_started.wait()
    shutdown_task = asyncio.create_task(server._shutdown())
    await asyncio.sleep(0)

    session.kill.assert_not_awaited()
    release_execution.set()
    await execute_task
    await shutdown_task
    session.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_idle_cleanup_finishes_kill_before_releasing_reference(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer(session_timeout_minutes=0)
    session = MagicMock()
    kill_started = asyncio.Event()
    allow_kill = asyncio.Event()
    kill_completed = False

    async def kill():
        nonlocal kill_completed
        kill_started.set()
        await allow_kill.wait()
        kill_completed = True

    session.kill = kill
    server.browser_session = session
    server._namespace = {'value': 1}
    server._last_activity = 0

    cleanup_task = asyncio.create_task(server._cleanup_expired_session())
    await kill_started.wait()
    cleanup_task.cancel()
    await asyncio.sleep(0)

    assert not kill_completed
    assert server.browser_session is session
    allow_kill.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    assert kill_completed
    assert server.browser_session is None


@pytest.mark.asyncio
async def test_cancelled_recovery_finishes_old_kill_before_replacement(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    kill_started = asyncio.Event()
    allow_kill = asyncio.Event()
    kill_completed = False

    async def kill():
        nonlocal kill_completed
        kill_started.set()
        await allow_kill.wait()
        kill_completed = True

    old_session = MagicMock()
    old_session.kill = kill
    old_session.reset = AsyncMock()
    server.browser_session = old_session
    server._namespace = {'user_value': 42}

    with patch.object(server, '_is_cdp_alive', new_callable=AsyncMock, return_value=False):
        recovery_task = asyncio.create_task(server._recover_browser_session())
        await kill_started.wait()
        recovery_task.cancel()
        await asyncio.sleep(0)

        assert not kill_completed
        allow_kill.set()
        with pytest.raises(asyncio.CancelledError):
            await recovery_task

    assert kill_completed


@pytest.mark.asyncio
async def test_kill_session_safely_drains_after_repeated_cancellation(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    kill_started = asyncio.Event()
    allow_kill = asyncio.Event()

    async def kill():
        kill_started.set()
        await allow_kill.wait()

    session = MagicMock()
    session.kill = kill
    cleanup_task = asyncio.create_task(server._kill_session_safely(session))
    await kill_started.wait()

    cleanup_task.cancel()
    await asyncio.sleep(0)
    cleanup_task.cancel()
    allow_kill.set()

    assert await cleanup_task == (True, True)


@pytest.mark.asyncio
async def test_sync_kill_compatibility_dispatches_forced_stop_and_checks_result(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    stop_event = MagicMock()
    stop_event.event_result.return_value = None
    session = MagicMock()
    session.kill.return_value = None
    session.event_bus.dispatch.return_value = stop_event

    assert await server._kill_session_safely(session) == (True, False)

    dispatched_event = session.event_bus.dispatch.call_args.args[0]
    assert dispatched_event.force is True
    stop_event.event_result.assert_called_once_with(raise_if_any=True, raise_if_none=False)


@pytest.mark.asyncio
async def test_expired_session_uses_kill_to_save_state(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session.kill = AsyncMock()
    server.browser_session = session
    server._namespace = {'value': 1}
    server._last_activity = 0

    with patch.object(mcp_mod.time, 'time', return_value=server.session_timeout_minutes * 60 + 1):
        await server._cleanup_expired_session()

    session.kill.assert_awaited_once()
    assert server.browser_session is None


@pytest.mark.asyncio
async def test_shutdown_cancels_cleanup_and_kills_owned_session(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session.kill = AsyncMock()
    server.browser_session = session
    server._namespace = {'value': 1}
    server._cleanup_task = asyncio.create_task(asyncio.sleep(60))

    await server._shutdown()

    session.kill.assert_awaited_once()
    assert server._cleanup_task is None
    assert server.browser_session is None
    assert server._namespace is None


@pytest.mark.asyncio
async def test_shutdown_propagates_cleanup_failure_and_keeps_ownership(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session.kill = AsyncMock(side_effect=RuntimeError('cleanup failed'))
    server.browser_session = session
    server._namespace = {'value': 1}

    with pytest.raises(RuntimeError, match='Unable to safely close'):
        await server._shutdown()

    assert server.browser_session is session
    assert server._namespace == {'value': 1}


@pytest.mark.asyncio
async def test_cancelled_shutdown_finishes_after_waiting_for_lifecycle_lock(isolated_mcp):
    mcp_mod, _, _ = isolated_mcp
    server = mcp_mod.OpenBrowserServer()
    session = MagicMock()
    session.kill = AsyncMock()
    server.browser_session = session
    server._namespace = {'value': 1}
    lock_acquired = asyncio.Event()
    release_lock = asyncio.Event()

    async def hold_execution_lock():
        async with server._execution_lock:
            lock_acquired.set()
            await release_lock.wait()

    holder = asyncio.create_task(hold_execution_lock())
    await lock_acquired.wait()
    shutdown_task = asyncio.create_task(server._shutdown())
    await asyncio.sleep(0)
    shutdown_task.cancel()
    release_lock.set()
    await holder

    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    session.kill.assert_awaited_once()
    assert server.browser_session is None
