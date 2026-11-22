"""Tests for daemon lifecycle, profile isolation, and recovery."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.focus import (
	FocusChangeReason,
	FocusPhase,
	FocusSnapshot,
	FocusTargetCode,
	FocusTargetError,
)
from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.browser.session import CDPSession
from openbrowser.browser.tabs import TabState
from openbrowser.config_lifecycle import ConfigStartupError
from openbrowser.daemon.server import DaemonServer


@pytest.fixture
def daemon_server(isolated_config_path):
	return DaemonServer()


def test_status_before_browser_start(daemon_server):
	result = daemon_server._status_payload()

	assert result['state'] == 'stopped'
	assert result['initialized'] is False
	assert result['browser']['phase'] == 'stopped'
	assert result['browser']['focus']['phase'] == 'uninitialized'
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
	session = BrowserSession(browser_profile=BrowserProfile(headless=True))
	session._focus_state = FocusSnapshot(
		phase=FocusPhase.FOCUSED,
		target_id='abc',
		session_id='session-1',
		previous_target_id=None,
		generation=2,
		reason=FocusChangeReason.SWITCH,
		last_error=None,
		updated_at=1.0,
	)
	focus_session = CDPSession.model_construct(
		cdp_client=MagicMock(),
		target_id='abc',
		session_id='session-1',
	)
	session.agent_focus = focus_session
	session._tab_registry.register('abc', url='https://example.com', state=TabState.OPEN, title='Example')
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.READY, increment_generation=True)
	daemon_server._session = session

	result = daemon_server._status_payload()

	assert result['state'] == 'ready'
	assert result['initialized'] is True
	assert result['browser']['generation'] == 1
	assert result['browser']['active_target_id'] == 'abc'
	assert result['browser']['tabs'][0]['target_id'] == 'abc'
	assert result['browser']['focus']['phase'] == 'focused'
	assert result['browser']['focus']['target_id'] == 'abc'
	assert result['browser']['focus']['generation'] == 2


def test_status_includes_lost_browser_focus(daemon_server):
	session = BrowserSession(browser_profile=BrowserProfile(headless=True))
	session._focus_lost_locked(
		FocusTargetError(
			'No live page target was available.',
			code=FocusTargetCode.RECOVERY_FAILED,
			phase=FocusPhase.LOST,
			target_id='tab-1',
			target_type='page',
		),
		FocusChangeReason.CRASH_RECOVERY,
		previous_target_id='tab-1',
	)
	daemon_server._session = session

	result = daemon_server._status_payload()

	assert result['browser']['focus']['phase'] == 'lost'
	assert result['browser']['focus']['previous_target_id'] == 'tab-1'
	assert result['browser']['focus']['last_error']['code'] == 'recovery_failed'


def test_status_recovers_focus_when_lifecycle_snapshot_fails(daemon_server):
	session = BrowserSession(browser_profile=BrowserProfile(headless=True))
	session._focus_lost_locked(
		FocusTargetError(
			'No live page target was available.',
			code=FocusTargetCode.RECOVERY_FAILED,
			phase=FocusPhase.LOST,
			target_id='tab-1',
			target_type='page',
		),
		FocusChangeReason.DETACH_RECOVERY,
		previous_target_id='tab-1',
	)
	daemon_server._session = session

	with patch.object(BrowserSession, 'lifecycle_snapshot', side_effect=RuntimeError('status unavailable')):
		result = daemon_server._status_payload()

	assert result['browser']['focus']['phase'] == 'lost'
	assert result['browser']['focus']['previous_target_id'] == 'tab-1'


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

	daemon_server._startup_config = {'browser_profile': saved_profile}
	with (
		patch('openbrowser.config.CONFIG', new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / 'profiles')),
		patch('openbrowser.daemon.server.get_default_profile', return_value=saved_profile),
		patch(
			'openbrowser.daemon.server.apply_managed_browser_profile_defaults',
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

	daemon_server._startup_config = {'browser_profile': saved_profile}
	with (
		patch('openbrowser.config.CONFIG', new=SimpleNamespace(OPENBROWSER_PROFILES_DIR=tmp_path / 'profiles')),
		patch('openbrowser.daemon.server.get_default_profile', return_value=saved_profile),
		patch('openbrowser.daemon.server.get_socket_path', return_value=custom_socket),
		patch(
			'openbrowser.daemon.server.apply_managed_browser_profile_defaults',
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
	status_transfer = daemon_server._status_payload()['namespace_transfer']
	dropped = status_transfer['dropped']
	assert 'previous browser session' in dropped['stale']
	assert 'previous browser session' in dropped['pages']
	assert status_transfer['preserved'] == ['user_value']
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


@pytest.mark.asyncio
async def test_failed_recovery_keeps_only_scalars_in_the_live_namespace(daemon_server):
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	daemon_server._session = old_session

	executor = MagicMock()
	executor.namespace = {
		'user_value': 42,
		'stale': old_session,
		'pages': [old_session],
		'browser': object(),
	}
	daemon_server._executor = executor

	new_session = MagicMock()
	new_session.kill = AsyncMock()
	new_session.start = AsyncMock(side_effect=RuntimeError('start failed'))
	daemon_server._build_browser_profile = MagicMock(return_value=MagicMock())

	with patch('openbrowser.browser.BrowserSession', return_value=new_session):
		with pytest.raises(RuntimeError, match='start failed'):
			await daemon_server._recover_browser_session()

	sanitized = executor.set_namespace.call_args.args[0]
	assert sanitized == {'user_value': 42}
	assert daemon_server._session is None
	assert 'stale' in daemon_server._last_namespace_transfer['dropped']
	assert 'previous browser session' in daemon_server._last_namespace_transfer['dropped']['stale']


def _write_valid_config(path: Path, *, headless: bool = True, model: str = 'gpt-5-mini') -> None:
	import json

	path.write_text(
		json.dumps(
			{
				'browser_profile': {
					'profile-a': {
						'id': 'profile-a',
						'default': True,
						'headless': headless,
						'user_data_dir': '/tmp/openbrowser-daemon-test-profile',
					}
				},
				'llm': {'llm-a': {'id': 'llm-a', 'default': True, 'model': model}},
				'agent': {'agent-a': {'id': 'agent-a', 'default': True}},
			}
		)
	)


@pytest.mark.asyncio
async def test_blocked_config_prevents_browser_start_and_reports_status(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	daemon = DaemonServer()

	with patch('openbrowser.browser.BrowserSession.start') as start:
		response = await daemon._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 1})

	assert response['success'] is False
	assert response['error_code'] == 'config_restart_required'
	assert response['hint'] == 'Fix the config, then restart the daemon or MCP server.'
	assert start.called is False

	status = daemon._status_payload()
	assert status['config']['startup']['phase'] == 'invalid'
	assert status['config']['startup']['error_code'] == 'config_invalid_json'
	assert status['config']['current']['phase'] == 'invalid'
	assert status['config']['restart_required'] is True
	assert status['state'] == 'stopped'


@pytest.mark.asyncio
async def test_valid_to_valid_config_change_requires_restart(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path, headless=True)
	daemon = DaemonServer()

	_write_valid_config(isolated_config_path, headless=False, model='gpt-5-mini-extra')

	with patch('openbrowser.browser.BrowserSession.start') as start, pytest.raises(ConfigStartupError) as error:
		await daemon._ensure_executor()

	assert error.value.code == 'config_restart_required'
	assert start.called is False
	status = daemon._status_payload()
	assert status['config']['startup']['phase'] == 'ready'
	assert status['config']['current']['phase'] == 'ready'
	assert status['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_recovery_gate_rejects_changed_config_before_cleanup(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path, headless=True)
	daemon = DaemonServer()
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	daemon._session = old_session
	daemon._executor = MagicMock()

	_write_valid_config(isolated_config_path, headless=False, model='gpt-5-mini-extra')

	with pytest.raises(ConfigStartupError) as error:
		await daemon._recover_browser_session(expected_session=old_session)

	assert error.value.code == 'config_restart_required'
	old_session.kill.assert_not_awaited()
	assert daemon._session is old_session
	status = daemon._status_payload()
	assert status['config']['startup']['phase'] == 'ready'
	assert status['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_status_does_not_start_browser_and_preserves_existing_keys(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	daemon = DaemonServer()

	with patch('openbrowser.browser.BrowserSession.start') as start:
		status = daemon._status_payload()

	assert start.called is False
	assert {
		'pid',
		'state',
		'initialized',
		'needs_browser_recovery',
		'socket',
		'profile',
		'idle_timeout',
		'init_timeout',
		'exec_timeout',
		'last_error',
		'generation',
		'uptime_seconds',
		'version',
		'browser',
		'startup',
		'storage_state',
		'storage_state_save',
		'namespace_transfer',
		'tools',
		'downloads',
	} <= status.keys()
	assert status['config']['startup']['phase'] == 'invalid'
	assert status['config']['current']['phase'] == 'invalid'
	assert status['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_repair_after_daemon_construction_still_requires_restart(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	daemon = DaemonServer()
	_write_valid_config(isolated_config_path)

	with patch('openbrowser.browser.BrowserSession.start') as start:
		response = await daemon._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 2})

	assert start.called is False
	assert response['error_code'] == 'config_restart_required'
	status = daemon._status_payload()
	assert status['config']['startup']['phase'] == 'invalid'
	assert status['config']['current']['phase'] == 'ready'
	assert status['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_connection_error_recovery_preserves_config_error_code(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	daemon = DaemonServer()
	daemon._executor = MagicMock()
	daemon._executor.execute = AsyncMock(
		return_value=SimpleNamespace(success=False, output='', error='websocket connection closed')
	)
	daemon._session = MagicMock()
	daemon._recover_browser_session = AsyncMock(
		side_effect=ConfigStartupError(
			daemon._startup_config_result.snapshot,
			code='config_restart_required',
			hint='Fix the config, then restart the daemon or MCP server.',
		)
	)

	response = await daemon._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 3})

	assert response['success'] is False
	assert response['error_code'] == 'config_restart_required'
	assert 'websocket connection closed' in response['error']
	assert 'config_restart_required' in response['error']
	assert 'may have partially run' in response['hint']


@pytest.mark.asyncio
async def test_recovery_gate_rejects_deleted_config_before_cleanup(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	daemon = DaemonServer()
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	daemon._session = old_session
	daemon._executor = MagicMock()
	isolated_config_path.unlink()

	with pytest.raises(ConfigStartupError) as error:
		await daemon._recover_browser_session(expected_session=old_session)

	assert error.value.code == 'config_restart_required'
	old_session.kill.assert_not_awaited()
	assert daemon._session is old_session
