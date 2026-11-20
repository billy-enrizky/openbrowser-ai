"""Tests for daemon lifecycle, profile isolation, and recovery."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.daemon.server import DaemonServer


@pytest.fixture
def daemon_server():
	return DaemonServer()


def test_status_before_browser_start(daemon_server):
	result = daemon_server._status_payload()

	assert result['state'] == 'stopped'
	assert result['initialized'] is False
	assert result['browser']['phase'] == 'stopped'
	assert 'socket' in result
	assert 'idle_timeout' in result


def test_status_resolves_profile_without_starting_browser(daemon_server):
	def build_profile():
		daemon_server._profile_summary = {
			'user_data_dir': '/tmp/daemon-profile',
			'headless': True,
			'storage_state': None,
		}
		return MagicMock()

	with patch.object(daemon_server, '_build_browser_profile', side_effect=build_profile) as build:
		result = daemon_server._status_payload()

	build.assert_called_once()
	assert result['profile']['user_data_dir'] == '/tmp/daemon-profile'
	assert daemon_server._session is None


def test_status_includes_browser_tabs(daemon_server):
	daemon_server._lifecycle.transition(LifecyclePhase.STARTING)
	daemon_server._lifecycle.transition(LifecyclePhase.READY)
	daemon_server._executor = MagicMock()
	session = MagicMock()
	session.lifecycle_snapshot.return_value = {
		'phase': 'ready',
		'generation': 3,
		'tabs': [{'target_id': 'abc', 'state': 'open'}],
	}
	daemon_server._session = session

	result = daemon_server._status_payload()

	assert result['state'] == 'ready'
	assert result['initialized'] is True
	assert result['browser']['generation'] == 3
	assert result['browser']['tabs'][0]['target_id'] == 'abc'


def test_default_profile_for_default_socket(daemon_server, tmp_path, monkeypatch):
	monkeypatch.delenv('OPENBROWSER_DAEMON_PROFILE_DIR', raising=False)
	monkeypatch.delenv('OPENBROWSER_SOCKET', raising=False)
	with patch(
		'openbrowser.config.CONFIG',
		new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / 'profiles'),
	):
		assert daemon_server._default_profile_dir() == tmp_path / 'profiles' / 'daemon'


def test_custom_socket_gets_isolated_profile(daemon_server, tmp_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_SOCKET', str(tmp_path / 'custom.sock'))
	monkeypatch.setenv('OPENBROWSER_DAEMON_PROFILE_ISOLATION', 'true')
	monkeypatch.delenv('OPENBROWSER_DAEMON_PROFILE_DIR', raising=False)
	with (
		patch(
			'openbrowser.config.CONFIG',
			new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / 'profiles'),
		),
		patch('openbrowser.daemon.server.get_socket_path', return_value=tmp_path / 'custom.sock'),
	):
		profile_dir = daemon_server._default_profile_dir()

	assert profile_dir.parent == tmp_path / 'profiles'
	assert profile_dir.name.startswith('daemon-')
	assert profile_dir.name != 'daemon'


def test_custom_socket_shares_default_profile_without_opt_in(daemon_server, tmp_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_SOCKET', str(tmp_path / 'custom.sock'))
	monkeypatch.delenv('OPENBROWSER_DAEMON_PROFILE_ISOLATION', raising=False)
	monkeypatch.delenv('OPENBROWSER_DAEMON_PROFILE_DIR', raising=False)
	with patch(
		'openbrowser.config.CONFIG',
		new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / 'profiles'),
	):
		assert daemon_server._default_profile_dir() == tmp_path / 'profiles' / 'daemon'


def test_explicit_profile_override_wins(daemon_server, tmp_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_SOCKET', str(tmp_path / 'custom.sock'))
	monkeypatch.setenv('OPENBROWSER_DAEMON_PROFILE_DIR', str(tmp_path / 'explicit'))
	assert daemon_server._default_profile_dir() == tmp_path / 'explicit'


def test_daemon_profile_env_wins_over_saved_config(daemon_server, tmp_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_DAEMON_PROFILE_DIR', str(tmp_path / 'explicit'))
	monkeypatch.delenv('OPENBROWSER_USER_DATA_DIR', raising=False)
	monkeypatch.delenv('OPENBROWSER_SOCKET', raising=False)
	saved_profile = {'user_data_dir': str(tmp_path / 'saved'), 'headless': False}

	with (
		patch('openbrowser.config.CONFIG', new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / 'profiles')),
		patch('openbrowser.config.load_openbrowser_config', return_value={'browser_profile': saved_profile}),
		patch('openbrowser.config.get_default_profile', return_value=saved_profile),
		patch(
			'openbrowser.config.apply_managed_browser_profile_defaults',
			side_effect=lambda profile_data, default_user_data_dir=None: dict(profile_data),
		),
	):
		profile = daemon_server._build_browser_profile()

	assert Path(profile.user_data_dir).expanduser().resolve() == (tmp_path / 'explicit').resolve()


def test_opted_in_isolation_wins_over_saved_config(daemon_server, tmp_path, monkeypatch):
	custom_socket = tmp_path / 'custom.sock'
	monkeypatch.setenv('OPENBROWSER_SOCKET', str(custom_socket))
	monkeypatch.setenv('OPENBROWSER_DAEMON_PROFILE_ISOLATION', 'true')
	monkeypatch.delenv('OPENBROWSER_DAEMON_PROFILE_DIR', raising=False)
	monkeypatch.delenv('OPENBROWSER_USER_DATA_DIR', raising=False)
	saved_profile = {'user_data_dir': str(tmp_path / 'saved'), 'headless': False}

	with (
		patch('openbrowser.config.CONFIG', new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / 'profiles')),
		patch('openbrowser.config.load_openbrowser_config', return_value={'browser_profile': saved_profile}),
		patch('openbrowser.config.get_default_profile', return_value=saved_profile),
		patch('openbrowser.daemon.server.get_socket_path', return_value=custom_socket),
		patch(
			'openbrowser.config.apply_managed_browser_profile_defaults',
			side_effect=lambda profile_data, default_user_data_dir=None: dict(profile_data),
		),
	):
		profile = daemon_server._build_browser_profile()

	assert Path(profile.user_data_dir).expanduser().resolve().name.startswith('daemon-')


@pytest.mark.asyncio
async def test_recovery_preserves_user_variables(daemon_server):
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	daemon_server._session = old_session

	old_namespace = {
		'user_value': 42,
		'browser': object(),
		'navigate': object(),
		'stale': old_session,
		'pages': [old_session],
	}
	executor = MagicMock()
	executor.namespace = old_namespace
	daemon_server._executor = executor

	new_session = MagicMock()
	new_session.start = AsyncMock()
	daemon_server._build_browser_profile = MagicMock(return_value=MagicMock())

	with (
		patch('openbrowser.browser.BrowserSession', return_value=new_session),
		patch('openbrowser.tools.service.CodeAgentTools', return_value=MagicMock()),
		patch('openbrowser.code_use.namespace.create_namespace', return_value={}),
	):
		await daemon_server._recover_browser_session()

	new_namespace = executor.set_namespace.call_args.args[0]
	assert new_namespace['user_value'] == 42
	assert 'stale' not in new_namespace
	assert 'pages' not in new_namespace
	dropped = daemon_server._last_namespace_transfer['dropped']
	assert 'previous browser session' in dropped['stale']
	assert 'previous browser session' in dropped['pages']
	assert daemon_server._status_payload()['namespace_transfer']['preserved'] == ['user_value']
	assert daemon_server._session is new_session
	assert daemon_server._browser_generation == 1
	assert new_session.seed_generation.call_args.args[0] == 1


@pytest.mark.asyncio
async def test_concurrent_recovery_builds_one_replacement_session(daemon_server):
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	daemon_server._session = old_session

	executor = MagicMock()
	executor.namespace = {}
	daemon_server._executor = executor

	new_session = MagicMock()
	new_session.start = AsyncMock()
	daemon_server._build_browser_profile = MagicMock(return_value=MagicMock())

	with (
		patch('openbrowser.browser.BrowserSession', return_value=new_session) as session_ctor,
		patch('openbrowser.tools.service.CodeAgentTools', return_value=MagicMock()),
		patch('openbrowser.code_use.namespace.create_namespace', return_value={}),
	):
		first = asyncio.create_task(daemon_server._recover_browser_session(expected_session=old_session))
		second = asyncio.create_task(daemon_server._recover_browser_session(expected_session=old_session))
		await asyncio.gather(first, second)

	old_session.kill.assert_awaited_once()
	assert session_ctor.call_count == 1
	new_session.start.assert_awaited_once()
	assert daemon_server._session is new_session


@pytest.mark.asyncio
async def test_execution_waits_for_connection_recovery(daemon_server):
	failed = MagicMock()
	failed.success = False
	failed.output = 'ConnectionClosedError'
	failed.error = 'ConnectionClosedError'
	succeeded = MagicMock()
	succeeded.success = True
	succeeded.output = 'ok'
	succeeded.error = None

	executor = MagicMock()
	executor.execute = AsyncMock(side_effect=[failed, succeeded])
	daemon_server._executor = executor
	daemon_server._session = MagicMock()

	recovery_started = asyncio.Event()
	release_recovery = asyncio.Event()

	async def blocking_recovery(*, expected_session):
		assert expected_session is daemon_server._session
		recovery_started.set()
		await release_recovery.wait()

	with patch.object(daemon_server, '_recover_browser_session', side_effect=blocking_recovery):
		first = asyncio.create_task(daemon_server._handle_request({'action': 'execute', 'code': 'first', 'id': 1}))
		await asyncio.wait_for(recovery_started.wait(), timeout=1)

		second = asyncio.create_task(daemon_server._handle_request({'action': 'execute', 'code': 'second', 'id': 2}))
		await asyncio.sleep(0.05)
		assert executor.execute.await_count == 1

		release_recovery.set()
		first_response, second_response = await asyncio.gather(first, second)

	assert first_response['error_code'] == 'cdp_connection_lost'
	assert second_response['success'] is True
	assert executor.execute.await_count == 2


@pytest.mark.asyncio
async def test_execution_returns_a_structured_timeout(daemon_server):
	async def slow_execute(code):
		await asyncio.sleep(1)

	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock(side_effect=slow_execute)
	daemon_server._session = MagicMock()
	daemon_server._exec_timeout = 0.01

	response = await daemon_server._handle_request({'action': 'execute', 'code': 'slow', 'id': 1})

	assert response['success'] is False
	assert response['error_code'] == 'execution_timeout'
	assert 'may still be running' in response['hint']


@pytest.mark.asyncio
async def test_execution_recovers_when_executor_survives_a_missing_session(daemon_server):
	daemon_server._session = None
	result = MagicMock()
	result.success = True
	result.output = 'ok'
	result.error = None
	daemon_server._needs_browser_recovery = True
	daemon_server._executor = MagicMock()
	daemon_server._executor.namespace = {'browser': object()}
	daemon_server._executor.execute = AsyncMock(return_value=result)
	replacement = MagicMock()

	async def recover():
		daemon_server._session = replacement

	with patch.object(daemon_server, '_recover_browser_session', side_effect=recover) as recovery:
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 1})

	recovery.assert_awaited_once()
	assert response['success'] is True
	assert daemon_server._session is replacement


@pytest.mark.asyncio
async def test_execution_refuses_to_run_when_a_previous_recovery_failed(daemon_server):
	daemon_server._session = MagicMock()
	daemon_server._needs_browser_recovery = True
	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock()

	with patch.object(
		daemon_server,
		'_recover_browser_session',
		new=AsyncMock(side_effect=RuntimeError('close failed')),
	) as recovery:
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 1})

	recovery.assert_awaited_once()
	daemon_server._executor.execute.assert_not_awaited()
	assert response['success'] is False
	assert response['error_code'] == 'cdp_recovery_failed'


@pytest.mark.asyncio
async def test_execution_does_not_recover_an_injected_code_only_executor(daemon_server):
	daemon_server._session = None
	result = MagicMock()
	result.success = True
	result.output = 'ok'
	result.error = None
	daemon_server._executor = MagicMock()
	daemon_server._executor.namespace = {'browser': object()}
	daemon_server._executor.execute = AsyncMock(return_value=result)

	with patch.object(daemon_server, '_recover_browser_session', new=AsyncMock()) as recovery:
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 1})

	recovery.assert_not_awaited()
	assert response['success'] is True


def test_error_response_is_structured(daemon_server):
	response = daemon_server._error_response(
		7,
		'boom',
		code='test_error',
		hint='do the thing',
	)

	assert response['id'] == 7
	assert response['success'] is False
	assert response['error_code'] == 'test_error'
	assert response['hint'] == 'do the thing'
	assert 'state' in response
