"""Tests for per-instance MCP browser profiles."""

from __future__ import annotations

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
