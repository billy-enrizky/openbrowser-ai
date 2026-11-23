"""Tests for daemon lifecycle, profile isolation, and recovery."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.dom_cache import DocumentIdentity
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
	assert 'cdp' not in result['browser']
	assert 'socket' in result
	assert 'idle_timeout' in result


def test_status_uses_last_known_snapshot_without_probing_or_starting(daemon_server):
	session = MagicMock()
	session.lifecycle_snapshot = MagicMock(
		return_value={
			'phase': 'degraded',
			'cdp': {'phase': 'degraded', 'client_present': True, 'healthy': False},
		}
	)
	session.probe_cdp_connection = AsyncMock(side_effect=AssertionError('status must not probe'))
	session.start = AsyncMock(side_effect=AssertionError('status must not start'))
	daemon_server._session = session

	result = daemon_server._status_payload()

	assert result['browser']['cdp']['phase'] == 'degraded'
	session.probe_cdp_connection.assert_not_called()
	session.start.assert_not_called()


def test_status_falls_back_to_cdp_health_when_snapshot_fails(daemon_server):
	session = BrowserSession()
	root = MagicMock()
	root.send.Browser.getVersion = AsyncMock(return_value={})
	session._cdp_client_root = root
	session._cdp_health.mark_cleanup_failure(RuntimeError('retained cleanup'))
	daemon_server._session = session

	with patch.object(BrowserSession, 'lifecycle_snapshot', side_effect=RuntimeError('snapshot failed')):
		result = daemon_server._status_payload()

	assert result['browser']['cdp']['phase'] == 'disconnected'
	assert result['browser']['cdp']['healthy'] is None
	assert result['browser']['cdp']['cleanup_pending'] is True
	assert result['browser']['cdp_connected'] is True
	root.send.Browser.getVersion.assert_not_awaited()


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
		await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)

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
		first_response, second_response = await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)

	assert first_response['error_code'] == 'cdp_connection_lost'
	assert second_response['success'] is True
	assert executor.execute.await_count == 2


@pytest.mark.asyncio
async def test_execution_failure_sanitizes_executor_result(daemon_server):
	daemon_server._session = MagicMock()
	daemon_server._executor = MagicMock()
	daemon_server._executor.max_output_chars = 12_000
	daemon_server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='retained-context-' + 'x' * 10_500 + ' Authorization: Bearer daemon-executor-output-secret',
			error='RuntimeError: Authorization: Bearer daemon-executor-error-secret',
		)
	)

	response = await daemon_server._handle_request({'action': 'execute', 'code': 'raise RuntimeError()', 'id': 9})

	assert response['success'] is False
	payload = json.dumps(response)
	assert len(response['output']) > 10_000
	assert 'retained-context-' in response['output']
	assert 'daemon-executor-output-secret' not in payload
	assert 'daemon-executor-error-secret' not in payload
	assert 'authorization=<redacted>' in response['output'].lower()
	assert 'authorization=<redacted>' in response['error'].lower()


@pytest.mark.asyncio
async def test_execution_real_executor_failure_preserves_traceback_after_redaction(daemon_server):
	from openbrowser.code_use.executor import CodeExecutor

	executor = CodeExecutor()
	namespace = {'__builtins__': __builtins__}
	executor.set_namespace(namespace)
	daemon_server._executor = executor
	daemon_server._session = MagicMock()

	response = await daemon_server._handle_request(
		{
			'action': 'execute',
			'code': (
				"print('Authorization:')\n"
				"print('  Bearer real-daemon-executor-secret second-token')\n"
				"print('Proxy-Authorization: Bearer')\n"
				"print('  q7')\n"
				"print('  request_id=123')\n"
				"print('Cookie: session=')\n"
				"print('  c9')\n"
				"print('  a=b')\n"
				"print('  csrf=xyz')\n"
				"print('  captured diagnostic = expected')\n"
				"raise RuntimeError('boom')\n"
			),
			'id': 10,
		}
	)

	assert response['success'] is False
	assert 'Traceback (most recent call last):' in response['output']
	assert '  File "<execute_code>"' in response['output']
	assert '  captured diagnostic = expected' in response['output']
	assert '  request_id=123' in response['output']
	assert 'real-daemon-executor-secret' not in response['output']
	assert 'second-token' not in response['output']
	assert 'q7' not in response['output']
	assert 'c9' not in response['output']
	assert 'a=b' not in response['output']
	assert 'csrf=xyz' not in response['output']
	assert 'authorization=<redacted>' in response['output'].lower()


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
		new=AsyncMock(side_effect=RuntimeError('Authorization: Bearer daemon-recovery-secret')),
	) as recovery:
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 1})

	recovery.assert_awaited_once()
	daemon_server._executor.execute.assert_not_awaited()
	assert response['success'] is False
	assert response['error_code'] == 'cdp_recovery_failed'
	payload = json.dumps(response)
	assert 'daemon-recovery-secret' not in payload
	assert 'authorization=<redacted>' in response['error'].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize('recovery_outcome', ['success', 'generic', 'typed'])
async def test_post_execution_recovery_output_only_is_sanitized(daemon_server, recovery_outcome):
	daemon_server._session = MagicMock()
	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='websocket connection closed\npartial output context\nAuthorization: Bearer output-only-secret',
			error=None,
		)
	)
	if recovery_outcome == 'success':
		recovery = AsyncMock(return_value=None)
		expected_code = 'cdp_connection_lost'
	elif recovery_outcome == 'generic':
		recovery = AsyncMock(side_effect=RuntimeError('Authorization: Bearer generic-recovery-secret'))
		expected_code = 'cdp_recovery_failed'
	else:
		error = ConfigStartupError(
			SimpleNamespace(
				path='config.toml',
				error_code='config_invalid',
				hint='Restart the daemon after fixing config.toml.',
			),
			hint='Restart the daemon after fixing config.toml.',
			detail='recovery-specific diagnostic detail',
		)
		recovery = AsyncMock(side_effect=error)
		expected_code = 'config_invalid'

	with patch.object(daemon_server, '_recover_browser_session', new=recovery):
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 4})

	assert response['error_code'] == expected_code
	payload = json.dumps(response)
	assert 'output-only-secret' not in payload
	assert 'authorization=<redacted>' in response['error'].lower()
	if recovery_outcome == 'generic':
		assert 'generic-recovery-secret' not in payload
		assert 'may have partially run' in response['hint']
		assert 'verify the page before retrying' in response['hint']
	else:
		assert 'partial output context' in response['error']
	if recovery_outcome == 'typed':
		assert 'recovery-specific diagnostic detail' in response['error']
		assert 'Restart the daemon after fixing config.toml.' in response['hint']
		assert 'may have partially run' in response['hint']
		assert 'verify the page before retrying' in response['hint']


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


def test_status_reports_dom_cache_without_starting_browser(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	daemon = DaemonServer()
	daemon._session = BrowserSession()

	with patch('openbrowser.browser.BrowserSession.start') as start:
		status = daemon._status_payload()

	assert start.called is False
	assert status['browser']['dom_cache']['phase'] == 'empty'
	assert status['browser']['dom_cache']['selector_count'] == 0


def test_status_reports_a_ready_dom_cache_without_page_content(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	daemon = DaemonServer()
	session = BrowserSession()
	session.agent_focus = CDPSession.model_construct(
		cdp_client=MagicMock(),
		target_id='tab-a',
		session_id='sess-a',
		url='https://secret.test/',
		title='Secret',
	)
	assert (
		session.update_cached_selector_map(
			{1: object()},
			identity=DocumentIdentity('loader-1', 'https://secret.test/'),
		)
		is True
	)
	daemon._session = session

	with patch('openbrowser.browser.BrowserSession.start') as start:
		status = daemon._status_payload()

	assert start.called is False
	assert status['browser']['dom_cache']['phase'] == 'ready'
	assert status['browser']['dom_cache']['selector_count'] == 1
	assert 'secret.test' not in json.dumps(status)


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


def test_profile_resolution_reports_config_and_environment_sources(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	monkeypatch.setenv('OPENBROWSER_HEADLESS', 'false')
	daemon = DaemonServer()

	with patch('openbrowser.browser.BrowserSession.start') as start:
		status = daemon._status_payload()

	assert start.called is False
	assert status['profile']['headless'] is False
	assert status['profile']['storage_state'] is None
	resolution = status['profile']['resolution']
	assert resolution['user_data_dir']['source'] == 'config'
	assert resolution['user_data_dir']['value'].endswith('/openbrowser-daemon-test-profile')
	assert resolution['headless'] == {'value': False, 'source': 'environment'}
	assert resolution['storage_state'] == {'mode': 'none', 'source': 'default'}
	assert resolution['downloads_path']['source'] == 'default'


def test_profile_resolution_reports_a_managed_path(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	monkeypatch.delenv('OPENBROWSER_HEADLESS', raising=False)
	daemon = DaemonServer()
	daemon._startup_config['browser_profile'].pop('user_data_dir', None)
	daemon._profile_summary = None

	with patch('openbrowser.browser.BrowserSession.start') as start:
		status = daemon._status_payload()

	assert start.called is False
	resolution = status['profile']['resolution']
	assert resolution['user_data_dir']['source'] == 'managed'
	assert resolution['headless'] == {'value': True, 'source': 'default'}


def test_profile_resolution_attributes_environment_paths(isolated_config_path, monkeypatch, tmp_path):
	from openbrowser.config import CONFIG

	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	env_profile = tmp_path / 'environment-profile'
	env_state = tmp_path / 'environment-state.json'
	env_state.write_text('{"cookies": [], "origins": []}')
	monkeypatch.setenv('OPENBROWSER_USER_DATA_DIR', str(env_profile))
	monkeypatch.setenv('OPENBROWSER_STORAGE_STATE', str(env_state))
	CONFIG._env_config = None
	daemon = DaemonServer()

	with patch('openbrowser.browser.BrowserSession.start') as start:
		status = daemon._status_payload()

	assert start.called is False
	resolution = status['profile']['resolution']
	assert resolution['user_data_dir'] == {
		'value': str(env_profile.resolve()),
		'source': 'environment',
	}
	assert resolution['storage_state'] == {'mode': 'file', 'source': 'environment'}


def test_profile_resolution_reports_validated_effective_values(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	config = json.loads(isolated_config_path.read_text())
	config['browser_profile']['profile-a'].update(
		{
			'keep_alive': 'false',
			'disable_security': 1,
			'wait_between_actions': '0.25',
			'device_scale_factor': '1.5',
		}
	)
	isolated_config_path.write_text(json.dumps(config))
	daemon = DaemonServer()

	status = daemon._status_payload()

	resolution = status['profile']['resolution']
	assert resolution['keep_alive'] == {'value': False, 'source': 'config'}
	assert resolution['disable_security'] == {'value': True, 'source': 'config'}
	assert resolution['wait_between_actions'] == {'value': 0.25, 'source': 'config'}
	assert resolution['device_scale_factor'] == {'value': 1.5, 'source': 'config'}


def test_profile_resolution_reports_default_headless_over_config(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path, headless=False)
	daemon = DaemonServer()

	status = daemon._status_payload()

	assert status['profile']['headless'] is True
	assert status['profile']['resolution']['headless'] == {'value': True, 'source': 'default'}


def test_profile_resolution_reports_profile_dir_override_as_managed(isolated_config_path, monkeypatch, tmp_path):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	override = tmp_path / 'daemon-profile-override'
	monkeypatch.setenv('OPENBROWSER_DAEMON_PROFILE_DIR', str(override))
	daemon = DaemonServer()

	status = daemon._status_payload()

	assert status['profile']['resolution']['user_data_dir'] == {
		'value': str(override.resolve()),
		'source': 'managed',
	}


def test_profile_resolution_reports_environment_proxy_without_credentials(isolated_config_path, monkeypatch):
	from openbrowser.config import CONFIG

	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	monkeypatch.setenv('OPENBROWSER_PROXY_URL', 'http://user:secret-password@proxy.test:8080')
	CONFIG._env_config = None
	daemon = DaemonServer()

	status = daemon._status_payload()

	assert status['profile']['resolution']['proxy'] == {
		'mode': 'configured',
		'source': 'environment',
	}
	assert 'secret-password' not in json.dumps(status)


def test_profile_resolution_reports_configured_storage_state(isolated_config_path, monkeypatch, tmp_path):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_config(isolated_config_path)
	state_path = tmp_path / 'configured-state.json'
	state_path.write_text('{"cookies": [], "origins": []}')
	config = json.loads(isolated_config_path.read_text())
	config['browser_profile']['profile-a']['storage_state'] = str(state_path)
	isolated_config_path.write_text(json.dumps(config))
	daemon = DaemonServer()

	status = daemon._status_payload()

	assert status['profile']['storage_state'] == str(state_path)
	assert status['profile']['resolution']['storage_state'] == {
		'mode': 'file',
		'source': 'config',
	}


@pytest.mark.asyncio
async def test_start_failure_sanitizes_daemon_status(daemon_server):
	session = MagicMock()
	session.start = AsyncMock(side_effect=RuntimeError('Authorization: Bearer daemon-lifecycle-secret'))
	session.kill = AsyncMock()
	daemon_server._build_browser_profile = MagicMock(return_value=MagicMock())

	with (
		patch('openbrowser.browser.BrowserSession', return_value=session),
		pytest.raises(RuntimeError, match='daemon-lifecycle-secret'),
	):
		await daemon_server._ensure_executor()

	status = daemon_server._status_payload()
	assert 'daemon-lifecycle-secret' not in str(status)
	assert 'authorization=<redacted>' in (daemon_server._browser_error or '').lower()


@pytest.mark.asyncio
async def test_execute_start_failure_sanitizes_client_response(daemon_server):
	with patch.object(
		daemon_server,
		'_ensure_executor',
		AsyncMock(side_effect=RuntimeError('Authorization: Bearer daemon-response-secret')),
	):
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 1})

	payload = json.dumps(response)
	assert 'daemon-response-secret' not in payload
	assert 'authorization=<redacted>' in response['error'].lower()
	assert response['error_code'] == 'daemon_start_failed'


@pytest.mark.asyncio
async def test_failed_cleanup_retains_daemon_session_owner(daemon_server):
	session = MagicMock()
	session.start = AsyncMock(side_effect=RuntimeError('start failed'))
	session.kill = AsyncMock(side_effect=RuntimeError('kill failed'))
	session.lifecycle_snapshot = MagicMock(
		return_value={
			'phase': 'failed',
			'cdp': {'phase': 'failed', 'client_present': True, 'healthy': False, 'cleanup_pending': True},
		}
	)
	daemon_server._build_browser_profile = MagicMock(return_value=MagicMock())

	with (
		patch('openbrowser.browser.BrowserSession', return_value=session),
		pytest.raises(RuntimeError, match='start failed'),
	):
		await daemon_server._ensure_executor()

	assert daemon_server._session is session
	status = daemon_server._status_payload()
	assert status['browser']['cdp']['cleanup_pending'] is True

	with (
		patch('openbrowser.browser.BrowserSession') as factory,
		pytest.raises(RuntimeError, match='failed browser session still requires cleanup'),
	):
		await daemon_server._ensure_executor()
	factory.assert_not_called()
	assert daemon_server._session is session


@pytest.mark.asyncio
async def test_retained_session_request_hint_names_reset(daemon_server):
	daemon_server._session = MagicMock()
	daemon_server._executor = None

	response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 11})

	assert response['success'] is False
	assert response['error_code'] == 'daemon_reset_required'
	assert 'daemon reset' in response['hint']
	assert 'failed browser session still requires cleanup' in response['error']


@pytest.mark.asyncio
async def test_typed_startup_error_response_is_sanitized(daemon_server):
	error = ConfigStartupError(
		SimpleNamespace(
			path='config.toml',
			error_code='config_invalid',
			hint='Authorization: Bearer typed-daemon-hint',
		),
		hint='Authorization: Bearer typed-daemon-hint',
		detail='Authorization: Bearer typed-daemon-detail',
	)
	with patch.object(daemon_server, '_ensure_executor', AsyncMock(side_effect=error)):
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 1})

	payload = json.dumps(response)
	assert 'typed-daemon' not in payload
	assert 'authorization=<redacted>' in response['error'].lower()
	assert 'authorization=<redacted>' in response['hint'].lower()


@pytest.mark.asyncio
async def test_daemon_reset_releases_retained_failed_session_for_retry(daemon_server):
	failed = MagicMock()
	failed.start = AsyncMock(side_effect=RuntimeError('start failed'))
	failed.kill = AsyncMock(side_effect=[RuntimeError('kill failed'), None])
	failed.lifecycle_snapshot = MagicMock(
		return_value={
			'phase': 'failed',
			'cdp': {
				'phase': 'failed',
				'client_present': True,
				'healthy': False,
				'cleanup_pending': True,
			},
		}
	)
	daemon_server._build_browser_profile = MagicMock(return_value=MagicMock())

	with (
		patch('openbrowser.browser.BrowserSession', return_value=failed),
		pytest.raises(RuntimeError, match='start failed'),
	):
		await daemon_server._ensure_executor()
	assert daemon_server._session is failed

	with (
		patch('openbrowser.browser.BrowserSession') as factory,
		pytest.raises(RuntimeError, match='failed browser session still requires cleanup'),
	):
		await daemon_server._ensure_executor()
	factory.assert_not_called()

	reset = await daemon_server._handle_request({'action': 'reset', 'id': 2})
	assert reset['success'] is True
	assert failed.kill.await_count == 2
	assert daemon_server._session is None

	replacement = MagicMock()
	replacement.start = AsyncMock()
	replacement.seed_generation = MagicMock()
	with (
		patch('openbrowser.browser.BrowserSession', return_value=replacement),
		patch('openbrowser.tools.service.CodeAgentTools', return_value=MagicMock()),
		patch('openbrowser.code_use.namespace.create_namespace', return_value={}),
	):
		await daemon_server._ensure_executor()
	assert daemon_server._session is replacement


@pytest.mark.asyncio
async def test_failed_reset_blocks_existing_executor_until_retry(daemon_server):
	failed = MagicMock()
	failed.kill = AsyncMock(side_effect=RuntimeError('kill failed'))
	daemon_server._session = failed
	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(success=True, output='must not run', error=None)
	)

	with patch.object(daemon_server, '_close_session_bounded', AsyncMock(return_value=False)):
		reset = await daemon_server._handle_request({'action': 'reset', 'id': 20})

	assert reset['success'] is False
	assert reset['error_code'] == 'reset_failed'

	execute = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 21})

	assert execute['success'] is False
	assert execute['error_code'] == 'daemon_reset_required'
	daemon_server._executor.execute.assert_not_awaited()
	assert daemon_server._session is failed

	with patch.object(daemon_server, '_close_session_bounded', AsyncMock(return_value=True)):
		retry = await daemon_server._handle_request({'action': 'reset', 'id': 22})
	assert retry['success'] is True
	assert daemon_server._session is None
	assert daemon_server._executor is None
	assert daemon_server._cleanup_required is False


@pytest.mark.asyncio
async def test_startup_cleanup_cancellation_propagates_and_retains_session(daemon_server):
	session = MagicMock()
	cleanup_started = asyncio.Event()
	release = asyncio.Event()

	async def fail_start():
		raise RuntimeError('startup failed')

	cleanup_finished = asyncio.Event()
	kill_calls = 0

	async def stubborn_kill():
		nonlocal kill_calls
		kill_calls += 1
		if kill_calls == 1:
			cleanup_started.set()
			try:
				await asyncio.sleep(60)
			except asyncio.CancelledError:
				await asyncio.wait_for(release.wait(), timeout=5.0)
			cleanup_finished.set()

	session.start = AsyncMock(side_effect=fail_start)
	session.kill = AsyncMock(side_effect=stubborn_kill)
	task = None

	try:
		with (
			patch('openbrowser.browser.BrowserSession', return_value=session),
			patch.object(daemon_server, '_build_browser_profile', return_value=MagicMock()),
			patch.object(daemon_server, '_init_timeout', 1.0),
		):
			task = asyncio.create_task(daemon_server._ensure_executor())
			await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
			task.cancel()
			with pytest.raises(asyncio.CancelledError):
				await asyncio.wait_for(task, timeout=2.0)
	finally:
		release.set()
		if task is not None and not task.done():
			task.cancel()
			await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=2.0)

	assert daemon_server._session is session
	assert daemon_server._cleanup_required is True
	await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
	with patch('openbrowser.daemon.server.DAEMON_CLEANUP_TIMEOUT', 1.0):
		retry = await daemon_server._handle_request({'action': 'reset', 'id': 33})
	assert retry['success'] is True
	assert daemon_server._session is None
	assert daemon_server._cleanup_required is False


@pytest.mark.asyncio
async def test_reset_cancellation_retains_cleanup_gate_until_retry(daemon_server):
	session = MagicMock()
	cleanup_started = asyncio.Event()
	cleanup_finished = asyncio.Event()
	release = asyncio.Event()
	kill_calls = 0

	async def kill():
		nonlocal kill_calls
		kill_calls += 1
		if kill_calls == 1:
			cleanup_started.set()
			try:
				await asyncio.sleep(60)
			except asyncio.CancelledError:
				await asyncio.wait_for(release.wait(), timeout=5.0)
			cleanup_finished.set()

	session.kill = AsyncMock(side_effect=kill)
	daemon_server._session = session
	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock()
	reset_task = None

	try:
		reset_task = asyncio.create_task(daemon_server._handle_request({'action': 'reset', 'id': 40}))
		await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
		reset_task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await asyncio.wait_for(reset_task, timeout=2.0)

		assert daemon_server._session is session
		assert daemon_server._cleanup_required is True
		execute = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 41})
		assert execute['error_code'] == 'daemon_reset_required'
		daemon_server._executor.execute.assert_not_awaited()
	finally:
		release.set()
		if reset_task is not None and not reset_task.done():
			reset_task.cancel()
			await asyncio.wait_for(asyncio.gather(reset_task, return_exceptions=True), timeout=2.0)
		if kill_calls:
			await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)

	with patch('openbrowser.daemon.server.DAEMON_CLEANUP_TIMEOUT', 1.0):
		retry = await daemon_server._handle_request({'action': 'reset', 'id': 42})

	assert retry['success'] is True
	assert daemon_server._session is None
	assert daemon_server._cleanup_required is False


@pytest.mark.asyncio
async def test_recovery_cleanup_cancellation_retains_session_until_reset(daemon_server):
	session = MagicMock()
	cleanup_started = asyncio.Event()
	cleanup_finished = asyncio.Event()
	release = asyncio.Event()
	kill_calls = 0

	async def fail_start():
		raise RuntimeError('replacement startup failed')

	async def kill():
		nonlocal kill_calls
		kill_calls += 1
		if kill_calls == 1:
			cleanup_started.set()
			try:
				await asyncio.sleep(60)
			except asyncio.CancelledError:
				await asyncio.wait_for(release.wait(), timeout=5.0)
			cleanup_finished.set()

	session.start = AsyncMock(side_effect=fail_start)
	session.kill = AsyncMock(side_effect=kill)
	daemon_server._session = None
	daemon_server._last_closed_session = None
	daemon_server._executor = MagicMock()
	daemon_server._executor.namespace = {}
	recovery_task = None

	try:
		with (
			patch('openbrowser.browser.BrowserSession', return_value=session),
			patch.object(daemon_server, '_build_browser_profile', return_value=MagicMock()),
			patch.object(daemon_server, '_init_timeout', 1.0),
			patch.object(daemon_server, '_startup_config_result', daemon_server._startup_config_result),
		):
			recovery_task = asyncio.create_task(daemon_server._recover_browser_session())
			await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
			recovery_task.cancel()
			with pytest.raises(asyncio.CancelledError):
				await asyncio.wait_for(recovery_task, timeout=2.0)
	finally:
		release.set()
		if recovery_task is not None and not recovery_task.done():
			recovery_task.cancel()
			await asyncio.wait_for(asyncio.gather(recovery_task, return_exceptions=True), timeout=2.0)
		if kill_calls:
			await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)

	assert daemon_server._session is session
	assert daemon_server._cleanup_required is True
	with patch('openbrowser.daemon.server.DAEMON_CLEANUP_TIMEOUT', 1.0):
		retry = await daemon_server._handle_request({'action': 'reset', 'id': 43})
	assert retry['success'] is True
	assert daemon_server._session is None
	assert daemon_server._cleanup_required is False


@pytest.mark.asyncio
async def test_reset_timeout_retains_cleanup_without_a_forced_event(daemon_server):
	release = asyncio.Event()
	first_done = asyncio.Event()
	kill_calls = 0
	session = MagicMock()
	session.event_bus.dispatch = MagicMock()

	async def kill():
		nonlocal kill_calls
		kill_calls += 1
		if kill_calls == 1:
			try:
				await asyncio.sleep(60)
			except asyncio.CancelledError:
				await asyncio.wait_for(release.wait(), timeout=5.0)
			first_done.set()

	session.kill = AsyncMock(side_effect=kill)
	daemon_server._session = session
	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock()

	try:
		with patch('openbrowser.daemon.server.DAEMON_CLEANUP_TIMEOUT', 0.01):
			reset = await daemon_server._handle_request({'action': 'reset', 'id': 30})

		assert reset['success'] is False
		assert reset['error_code'] == 'reset_failed'
		session.event_bus.dispatch.assert_not_called()
		assert daemon_server._session is session
		assert daemon_server._cleanup_required is True

		execute = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 31})
		assert execute['error_code'] == 'daemon_reset_required'
		daemon_server._executor.execute.assert_not_awaited()
	finally:
		release.set()
		if kill_calls:
			await asyncio.wait_for(first_done.wait(), timeout=1.0)

	with patch('openbrowser.daemon.server.DAEMON_CLEANUP_TIMEOUT', 1.0):
		retry = await daemon_server._handle_request({'action': 'reset', 'id': 32})

	assert retry['success'] is True
	assert daemon_server._session is None
	assert daemon_server._executor is None
	assert daemon_server._cleanup_required is False


@pytest.mark.asyncio
async def test_post_execution_recovery_failure_sanitizes_hint(daemon_server):
	daemon_server._session = MagicMock()
	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='Authorization: Bearer executor-output-secret',
			error='Authorization: Bearer executor-error-secret websocket connection closed',
		)
	)

	with patch.object(
		daemon_server,
		'_recover_browser_session',
		new=AsyncMock(side_effect=RuntimeError('Authorization: Bearer daemon-post-recovery-secret')),
	):
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 2})

	assert response['success'] is False
	assert response['error_code'] == 'cdp_recovery_failed'
	payload = json.dumps(response)
	assert 'daemon-post-recovery-secret' not in payload
	assert 'executor-output-secret' not in payload
	assert 'executor-error-secret' not in payload
	assert 'authorization=<redacted>' in response['error'].lower()
	assert 'authorization=<redacted>' in response['hint'].lower()


@pytest.mark.asyncio
async def test_typed_post_execution_recovery_failure_sanitizes_executor_text(daemon_server):
	daemon_server._session = MagicMock()
	daemon_server._executor = MagicMock()
	daemon_server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='partial executor context\nAuthorization: Bearer typed-executor-output-secret',
			error='Authorization: Bearer typed-executor-error-secret websocket connection closed',
		)
	)
	error = ConfigStartupError(
		SimpleNamespace(
			path='config.toml',
			error_code='config_invalid',
			hint='Restart the daemon after fixing config.toml. Authorization: Bearer typed-recovery-hint',
		),
		hint='Restart the daemon after fixing config.toml. Authorization: Bearer typed-recovery-hint',
		detail='typed recovery detail',
	)

	with patch.object(daemon_server, '_recover_browser_session', new=AsyncMock(side_effect=error)):
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 3})

	assert response['success'] is False
	assert response['error_code'] == 'config_invalid'
	payload = json.dumps(response)
	for sentinel in (
		'typed-executor-output-secret',
		'typed-executor-error-secret',
		'typed-recovery-hint',
	):
		assert sentinel not in payload
	assert 'partial executor context' in response['error']
	assert 'typed recovery detail' in response['error']
	assert 'Restart the daemon after fixing config.toml' in response['hint']
	assert 'may have partially run' in response['hint']
	assert 'authorization=<redacted>' in response['error'].lower()
	assert 'authorization=<redacted>' in response['hint'].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize('output_cap', [1, 2, 3, 64])
async def test_typed_recovery_bounds_error_and_reserves_warning(daemon_server, output_cap):
	from openbrowser.browser.cdp_connection import MAX_ERROR_CHARS

	daemon_server._session = MagicMock()
	daemon_server._executor = MagicMock()
	daemon_server._executor.max_output_chars = output_cap
	daemon_server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='partial output ' * 50,
			error='websocket connection closed',
		)
	)
	long_hint = 'Fix config.toml. ' + 'Follow the recovery guide. ' * 30
	error = ConfigStartupError(
		SimpleNamespace(
			path='config.toml',
			error_code='config_restart_required',
			hint=long_hint,
		),
		code='config_restart_required',
		hint=long_hint,
		detail='recovery specific detail ' * 30,
	)

	with patch.object(daemon_server, '_recover_browser_session', new=AsyncMock(side_effect=error)):
		response = await daemon_server._handle_request({'action': 'execute', 'code': 'print(1)', 'id': 5})

	assert response['success'] is False
	assert response['error_code'] == 'config_restart_required'
	assert len(response['error']) <= output_cap
	if output_cap in (1, 2):
		assert response['error'] == 'r'
	elif output_cap == 3:
		assert response['error'] == 'p\nr'
	else:
		assert 'partial output' in response['error']
		assert 'recovery spec' in response['error']
	assert len(response['hint']) <= MAX_ERROR_CHARS
	assert 'Fix config.toml' in response['hint']
	assert 'may have partially run' in response['hint']
	assert 'verify the page before retrying' in response['hint']
