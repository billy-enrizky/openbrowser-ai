"""Tests for MCP lifecycle, status, and reset tools."""

import asyncio
import json
import time
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
from openbrowser.config import CONFIG
from openbrowser.config_lifecycle import ConfigStartupError
from openbrowser.mcp.server import OpenBrowserContext, OpenBrowserServer, _CleanupOutcome


@pytest.mark.asyncio
async def test_get_status_does_not_start_browser(mcp_server):
	result = await mcp_server._handle_tool('get_status', {})
	payload = json.loads(result)

	assert mcp_server.browser_session is None
	assert payload['browser']['phase'] == 'stopped'
	assert 'cdp' not in payload['browser']
	assert 'health' not in payload['browser']
	assert payload['namespace_variables'] == 0
	assert payload['profile'] is None
	assert payload['tools']['phase'] == 'not_initialized'
	assert payload['downloads']['pdf_detection']['phase'] == 'idle'


@pytest.mark.asyncio
async def test_get_status_reports_dom_cache_without_starting_browser(mcp_server):
	mcp_server.browser_session = BrowserSession()

	with patch.object(BrowserSession, 'start') as start:
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	assert payload['browser']['dom_cache']['phase'] == 'empty'
	assert payload['browser']['dom_cache']['selector_count'] == 0
	assert 'url' not in payload['browser']['dom_cache']


@pytest.mark.asyncio
async def test_get_status_reports_a_ready_dom_cache_without_page_content(mcp_server):
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
	mcp_server.browser_session = session

	with patch.object(BrowserSession, 'start') as start:
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	assert payload['browser']['dom_cache']['phase'] == 'ready'
	assert payload['browser']['dom_cache']['selector_count'] == 1
	assert 'secret.test' not in json.dumps(payload)


@pytest.mark.asyncio
async def test_get_status_reports_initialized_tool_availability(mcp_server):
	from openbrowser.tools.service import CodeAgentTools

	mcp_server._tools = CodeAgentTools()
	mcp_server._tools_generation = mcp_server._browser_generation
	mcp_server._namespace = {}
	mcp_server.browser_session = MagicMock()
	mcp_server.browser_session.download_outcomes.return_value = []
	mcp_server.browser_session.download_outcomes_error = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['tools']['phase'] == 'ready'
	assert payload['tools']['available_count'] > 0
	assert 'screenshot' in payload['tools']['excluded']
	assert payload['tools']['page_url'] is None


@pytest.mark.asyncio
async def test_get_status_reports_filtered_tools_for_the_active_tab(mcp_server):
	from openbrowser.tools.service import CodeAgentTools

	mcp_server._tools = CodeAgentTools()
	mcp_server._tools_generation = mcp_server._browser_generation
	mcp_server._namespace = {}

	@mcp_server._tools.registry.action('Google', domains=['*.google.com'])
	async def google_action():
		return 'ok'

	mcp_server._profile_summary = {
		'user_data_dir': '/tmp/profile',
		'headless': True,
		'storage_state': None,
	}
	browser = MagicMock()
	browser.lifecycle_snapshot.return_value = {
		'phase': 'ready',
		'active_target_id': 'tab-1',
		'tabs': [
			{
				'target_id': 'tab-1',
				'url': 'https://user:pass@example.com:8443/private?token=secret#frag',
			}
		],
	}
	browser.download_outcomes.return_value = []
	browser.download_outcomes_error = None
	mcp_server.browser_session = browser

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['tools']['phase'] == 'page_filtered'
	assert payload['tools']['page_url'] == 'https://example.com:8443'
	assert payload['tools']['filtered_count'] == 1
	assert payload['tools']['filtered'][0]['name'] == 'google_action'
	assert payload['tools']['filtered'][0]['reason'] == 'domain_filtered'
	assert 'secret' not in json.dumps(payload['tools'])


@pytest.mark.asyncio
async def test_get_status_reports_storage_state_previews_without_browser(mcp_server, tmp_path):
	state_path = tmp_path / 'state.json'
	state_path.write_text('{"cookies": [], "origins": []}')
	mcp_server._profile_summary = {'storage_state': str(state_path)}

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['storage_state']['phase'] == 'pending'
	assert payload['storage_state']['path'] == str(state_path.resolve())
	assert payload['storage_state_save']['phase'] == 'idle'
	assert payload['storage_state_save']['path'] == str(state_path.resolve())


@pytest.mark.asyncio
async def test_get_status_uses_config_storage_state_before_browser(mcp_server, tmp_path):
	state_path = tmp_path / 'config-state.json'
	state_path.write_text('{"cookies": [], "origins": []}')
	mcp_server.config = {
		'browser_profile': {
			'storage_state': str(state_path),
			'user_data_dir': str(tmp_path / 'profile'),
		}
	}
	mcp_server._profile_summary = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['storage_state']['phase'] == 'pending'
	assert payload['storage_state']['path'] == str(state_path.resolve())
	assert payload['storage_state_save']['phase'] == 'idle'
	assert payload['storage_state_save']['path'] == str(state_path.resolve())


@pytest.mark.asyncio
async def test_get_status_reports_managed_copy_of_a_configured_source(mcp_server, tmp_path):
	source = tmp_path / 'source-state.json'
	source.write_text('{"cookies": [], "origins": []}')
	mcp_server.config = {'browser_profile': {'storage_state': str(source)}}
	mcp_server._profile_summary = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	expected_suffix = f'mcp-{mcp_server.owner_id}/storage_state.json'
	assert payload['storage_state_save']['phase'] == 'idle'
	assert payload['storage_state_save']['path'].endswith(expected_suffix)
	assert not payload['storage_state_save']['path'].endswith('source-state.json')


@pytest.mark.asyncio
async def test_get_status_reports_invalid_storage_state_as_failed(mcp_server):
	mcp_server.config = {'browser_profile': {'storage_state': 123}}
	mcp_server._profile_summary = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['storage_state']['phase'] == 'failed'
	assert 'Unsupported storage-state source type' in (payload['storage_state']['error'] or '')
	assert payload['storage_state_save']['phase'] == 'failed'


@pytest.mark.asyncio
async def test_get_status_reports_dict_storage_state_as_skipped(mcp_server):
	mcp_server.config = {'browser_profile': {'storage_state': {'cookies': [], 'origins': []}}}
	mcp_server._profile_summary = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['storage_state']['phase'] == 'skipped'
	assert payload['storage_state_save']['phase'] == 'skipped'


@pytest.mark.asyncio
async def test_get_status_reports_managed_storage_state_path_before_browser(mcp_server):
	mcp_server.config = {}
	mcp_server._profile_summary = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	expected_suffix = f'mcp-{mcp_server.owner_id}/storage_state.json'
	managed_dir = CONFIG.XDG_CONFIG_HOME / 'openbrowser' / 'profiles' / f'mcp-{mcp_server.owner_id}'
	assert not managed_dir.exists()
	assert payload['storage_state_save']['phase'] == 'idle'
	assert payload['storage_state_save']['path'].endswith(expected_suffix)
	assert payload['storage_state']['phase'] == 'skipped'


@pytest.mark.asyncio
async def test_reset_browser_preserves_user_variables(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.browser_session = session

	async def navigate():
		return None

	from openbrowser.tools.service import CodeAgentTools

	mcp_server._tools = CodeAgentTools()
	mcp_server._tools_generation = mcp_server._browser_generation
	mcp_server._namespace = {
		'user_value': 42,
		'browser': object(),
		'navigate': navigate,
	}
	mcp_server._browser_lifecycle.transition(LifecyclePhase.STARTING)
	mcp_server._browser_lifecycle.transition(LifecyclePhase.READY)

	result = await mcp_server._handle_tool('reset_browser', {})
	payload = json.loads(result)

	assert payload['ok'] is True
	assert payload['preserved_variables'] == 1
	assert mcp_server.browser_session is None
	assert mcp_server._namespace is None
	assert 'user_value' in mcp_server._pending_variables
	assert 'browser' not in mcp_server._pending_variables
	assert 'browser' in payload['namespace_transfer']['framework']
	assert 'navigate' in payload['namespace_transfer']['framework']
	assert payload['browser']['phase'] == 'stopped'
	assert mcp_server._status_payload()['tools']['phase'] == 'not_initialized'


@pytest.mark.asyncio
async def test_reset_browser_drops_stale_browser_references(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.browser_session = session
	mcp_server._namespace = {
		'count': 2,
		'old_browser': session,
		'pages': [session],
		'page': object(),
	}

	result = await mcp_server._handle_tool('reset_browser', {})
	payload = json.loads(result)

	assert payload['preserved_variables'] == 1
	assert mcp_server._pending_variables == {'count': 2}
	dropped = payload['namespace_transfer']['dropped']
	assert 'previous browser session' in dropped['old_browser']
	assert 'previous browser session' in dropped['pages']
	assert 'page' in dropped
	assert len(payload['namespace_transfer']['warnings']) == 3


@pytest.mark.asyncio
async def test_reset_browser_clears_variables_when_requested(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.browser_session = session
	mcp_server._namespace = {'user_value': 42}

	result = await mcp_server._handle_tool('reset_browser', {'clear_variables': True})
	payload = json.loads(result)

	assert payload['ok'] is True
	assert payload['preserved_variables'] == 0
	assert mcp_server._pending_variables == {}


@pytest.mark.asyncio
async def test_repeated_reset_browser_keeps_pending_variables(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.browser_session = session
	mcp_server._namespace = {'user_value': 42}

	await mcp_server._handle_tool('reset_browser', {})
	await mcp_server._handle_tool('reset_browser', {})

	assert mcp_server._pending_variables['user_value'] == 42


@pytest.mark.asyncio
async def test_reset_browser_finishes_state_changes_before_cancellation(mcp_server):
	session = MagicMock()
	mcp_server.browser_session = session
	mcp_server._namespace = {'user_value': 42}

	with patch.object(
		mcp_server,
		'_kill_session_safely',
		new_callable=AsyncMock,
		return_value=_CleanupOutcome(True, asyncio.CancelledError()),
	):
		with pytest.raises(asyncio.CancelledError):
			await mcp_server._handle_tool('reset_browser', {})

	assert mcp_server.browser_session is None
	assert mcp_server._namespace is None
	assert mcp_server._pending_variables['user_value'] == 42
	assert mcp_server._executor.namespace is None


@pytest.mark.asyncio
async def test_reset_browser_keeps_ownership_when_cleanup_fails(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock(side_effect=RuntimeError('cleanup failed'))
	mcp_server.browser_session = session
	mcp_server._namespace = {'user_value': 42}

	result = await mcp_server._handle_tool('reset_browser', {})
	payload = json.loads(result)

	assert payload['ok'] is False
	assert mcp_server.browser_session is session


@pytest.mark.asyncio
async def test_namespace_start_timeout_cleans_up(mcp_server):
	from openbrowser.mcp import server as mcp_mod

	session = MagicMock()

	async def hanging_start():
		await asyncio.sleep(60)

	session.start = AsyncMock(side_effect=hanging_start)
	mcp_server.browser_session = None

	with (
		patch.object(mcp_mod, 'BrowserSession', return_value=session),
		patch.object(mcp_mod, '_start_timeout_seconds', return_value=0.01),
		patch.object(mcp_server, '_kill_session_safely', new_callable=AsyncMock, return_value=_CleanupOutcome(True)),
	):
		with pytest.raises(asyncio.TimeoutError):
			await mcp_server._ensure_namespace()

	assert mcp_server.browser_session is None
	assert mcp_server._namespace is None
	assert mcp_server._browser_lifecycle.phase is LifecyclePhase.FAILED


@pytest.mark.asyncio
async def test_namespace_start_failure_sanitizes_browser_status(mcp_server):
	from openbrowser.mcp import server as mcp_mod

	session = MagicMock()
	session.start = AsyncMock(side_effect=RuntimeError('Authorization: Bearer mcp-lifecycle-secret'))
	mcp_server.browser_session = None

	with (
		patch.object(mcp_mod, 'BrowserSession', return_value=session),
		patch.object(mcp_server, '_kill_session_safely', new_callable=AsyncMock, return_value=_CleanupOutcome(True)),
		pytest.raises(RuntimeError, match='mcp-lifecycle-secret'),
	):
		await mcp_server._ensure_namespace()

	status = mcp_server._status_payload()
	assert 'mcp-lifecycle-secret' not in str(status)
	assert 'authorization=<redacted>' in (status['browser']['last_error'] or '').lower()


@pytest.mark.asyncio
async def test_profile_failure_during_recovery_marks_browser_failed(mcp_server):
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	mcp_server.browser_session = old_session
	mcp_server._build_browser_profile = MagicMock(
		side_effect=RuntimeError('Authorization: Bearer recovery-lifecycle-secret')
	)

	with (
		patch.object(mcp_server, '_is_cdp_alive', AsyncMock(return_value=False)),
		patch.object(mcp_server, '_kill_session_safely', AsyncMock(return_value=_CleanupOutcome(True))),
		pytest.raises(RuntimeError, match='recovery-lifecycle-secret'),
	):
		await mcp_server._recover_browser_session()

	status = mcp_server._status_payload()
	assert status['browser']['phase'] == 'failed'
	assert 'recovery-lifecycle-secret' not in str(status)
	assert 'authorization=<redacted>' in (status['browser']['last_error'] or '').lower()


@pytest.mark.asyncio
async def test_profile_failure_marks_browser_lifecycle_failed(mcp_server):
	mcp_server._build_browser_profile = MagicMock(side_effect=RuntimeError('profile failed'))

	with pytest.raises(RuntimeError, match='profile failed'):
		await mcp_server._ensure_namespace()

	assert mcp_server._browser_lifecycle.phase is LifecyclePhase.FAILED
	assert 'profile failed' in (mcp_server._status_payload()['browser']['last_error'] or '')


@pytest.mark.asyncio
async def test_status_reports_download_outcomes(mcp_server):
	mcp_server.browser_session = MagicMock()
	mcp_server.browser_session.download_outcomes.return_value = [
		{'download_id': 'a', 'phase': 'completed', 'terminal': True},
		{'download_id': 'b', 'phase': 'failed', 'terminal': True, 'stage': 'fetch'},
		{'download_id': 'c', 'phase': 'started', 'terminal': False},
	]

	status = mcp_server._status_payload()

	assert status['downloads']['count'] == 3
	assert status['downloads']['pending'] == 1
	assert status['downloads']['recent'][-1]['phase'] == 'started'
	failed = [entry for entry in status['downloads']['recent'] if entry['phase'] == 'failed']
	assert failed[0]['stage'] == 'fetch'


@pytest.mark.asyncio
async def test_reset_then_execute_restores_pending_variables(mcp_server):
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	mcp_server.browser_session = old_session
	mcp_server._namespace = {'user_value': 42}
	await mcp_server._handle_tool('reset_browser', {})
	assert mcp_server._pending_variables['user_value'] == 42

	new_session = MagicMock()
	new_session.start = AsyncMock()
	mcp_server._build_browser_profile = MagicMock(return_value=MagicMock())
	with (
		patch('openbrowser.mcp.server.BrowserSession', return_value=new_session),
		patch('openbrowser.mcp.server.CodeAgentTools', return_value=MagicMock()),
		patch('openbrowser.mcp.server.create_namespace', return_value={}),
	):
		await mcp_server._ensure_namespace()

	assert mcp_server._namespace['user_value'] == 42
	assert mcp_server._pending_variables == {}


@pytest.mark.asyncio
async def test_idle_expiry_preserves_user_variables(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.session_timeout_minutes = 0
	mcp_server.browser_session = session
	mcp_server._namespace = {'user_value': 42}
	mcp_server._last_activity = 0

	await mcp_server._cleanup_expired_session()

	assert mcp_server.browser_session is None
	assert mcp_server._namespace is None
	assert mcp_server._pending_variables['user_value'] == 42


@pytest.mark.asyncio
async def test_execute_code_reports_browser_start_timeout(mcp_server, monkeypatch):
	from openbrowser.mcp import server as mcp_mod

	session = MagicMock()

	async def hanging_start():
		await asyncio.sleep(60)

	session.start = AsyncMock(side_effect=hanging_start)
	mcp_server.browser_session = None
	with (
		patch.object(mcp_mod, 'BrowserSession', return_value=session),
		patch.object(mcp_mod, '_start_timeout_seconds', return_value=0.01),
		patch.object(
			mcp_server,
			'_kill_session_safely',
			new_callable=AsyncMock,
			return_value=_CleanupOutcome(True),
		),
	):
		result = await mcp_server._execute_code('x = 1')

	assert 'Browser startup timed out' in result
	assert 'get_status' in result
	assert mcp_server.browser_session is None


@pytest.mark.asyncio
async def test_kill_session_timeout_is_a_cleanup_failure(mcp_server, monkeypatch):
	from openbrowser.mcp import server as mcp_mod

	async def hanging_kill():
		await asyncio.sleep(60)

	session = MagicMock()
	session.kill = hanging_kill
	monkeypatch.setattr(mcp_mod, '_shutdown_timeout_seconds', lambda: 0.01)
	outcome = await mcp_server._kill_session_safely(session)
	assert outcome.succeeded is False
	assert outcome.caller_cancellation is None


@pytest.mark.asyncio
async def test_recovery_drops_stale_browser_values_and_keeps_scalars(mcp_server):
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	mcp_server.browser_session = old_session
	mcp_server._namespace = {
		'answer': 42,
		'old_page': old_session,
		'pages': [old_session],
	}
	mcp_server._browser_lifecycle.transition(LifecyclePhase.STARTING)
	mcp_server._browser_lifecycle.transition(LifecyclePhase.READY)

	new_session = MagicMock()
	new_session.start = AsyncMock()
	new_session.seed_generation = MagicMock()

	with (
		patch.object(mcp_server, '_is_cdp_alive', new_callable=AsyncMock, return_value=False),
		patch.object(
			mcp_server,
			'_kill_session_safely',
			new_callable=AsyncMock,
			return_value=_CleanupOutcome(True),
		),
		patch.object(mcp_server, '_build_browser_profile', return_value=MagicMock()),
		patch('openbrowser.mcp.server.BrowserSession', return_value=new_session),
		patch('openbrowser.mcp.server.create_namespace', return_value={}),
		patch('openbrowser.mcp.server.CodeAgentTools', return_value=MagicMock()),
		patch.object(mcp_server, '_create_owner_file_system', return_value=None),
	):
		await mcp_server._recover_browser_session()

	assert mcp_server.browser_session is new_session
	assert mcp_server._namespace == {'answer': 42}
	assert 'old_page' not in mcp_server._namespace
	assert 'pages' not in mcp_server._namespace
	dropped = mcp_server._last_namespace_transfer['dropped']
	assert 'previous browser session' in dropped['old_page']
	assert 'previous browser session' in dropped['pages']


@pytest.mark.asyncio
async def test_reset_browser_clear_variables_resets_transfer_summary(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.browser_session = session
	mcp_server._namespace = {'answer': 42, 'old_page': session}

	await mcp_server._handle_tool('reset_browser', {})

	second = MagicMock()
	second.kill = AsyncMock()
	mcp_server.browser_session = second
	mcp_server._namespace = {'answer': 43}

	result = await mcp_server._handle_tool('reset_browser', {'clear_variables': True})
	payload = json.loads(result)

	assert payload['namespace_transfer']['preserved_count'] == 0
	assert payload['namespace_transfer']['dropped_count'] == 0
	assert mcp_server._pending_variables == {}


@pytest.mark.asyncio
async def test_get_status_reports_uninitialized_focus_without_browser(mcp_server):
	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['browser']['focus']['phase'] == 'uninitialized'
	assert payload['browser']['focus']['target_id'] is None
	assert payload['browser']['focus']['generation'] == 0


@pytest.mark.asyncio
async def test_get_status_inherits_focused_browser_snapshot(mcp_server):
	browser = BrowserSession(browser_profile=BrowserProfile(headless=True))
	browser._focus_state = FocusSnapshot(
		phase=FocusPhase.FOCUSED,
		target_id='tab-1',
		session_id='session-1',
		previous_target_id=None,
		generation=2,
		reason=FocusChangeReason.SWITCH,
		last_error=None,
		updated_at=1.0,
	)
	focus_session = CDPSession.model_construct(
		cdp_client=MagicMock(),
		target_id='tab-1',
		session_id='session-1',
	)
	browser.agent_focus = focus_session
	browser._tab_registry.register('tab-1', url='https://example.com', state=TabState.OPEN, title='Example')
	mcp_server.browser_session = browser

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['browser']['active_target_id'] == 'tab-1'
	assert payload['browser']['tabs'][0]['target_id'] == 'tab-1'
	assert payload['browser']['focus']['phase'] == 'focused'
	assert payload['browser']['focus']['target_id'] == 'tab-1'
	assert payload['browser']['focus']['generation'] == 2


@pytest.mark.asyncio
async def test_get_status_inherits_lost_browser_snapshot(mcp_server):
	browser = BrowserSession(browser_profile=BrowserProfile(headless=True))
	browser._focus_lost_locked(
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
	mcp_server.browser_session = browser

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['browser']['focus']['phase'] == 'lost'
	assert payload['browser']['focus']['previous_target_id'] == 'tab-1'
	assert payload['browser']['focus']['last_error']['code'] == 'recovery_failed'
	assert payload['browser']['focus']['last_error']['phase'] == 'lost'


@pytest.mark.asyncio
async def test_get_status_recovers_focus_when_lifecycle_snapshot_fails(mcp_server):
	browser = BrowserSession(browser_profile=BrowserProfile(headless=True))
	browser._focus_lost_locked(
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
	mcp_server.browser_session = browser

	with patch.object(BrowserSession, 'lifecycle_snapshot', side_effect=RuntimeError('status unavailable')):
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['browser']['focus']['phase'] == 'lost'
	assert payload['browser']['focus']['previous_target_id'] == 'tab-1'


def _write_valid_mcp_config(path, *, headless=True, model='gpt-5-mini'):
	path.write_text(
		json.dumps(
			{
				'browser_profile': {
					'profile-a': {
						'id': 'profile-a',
						'default': True,
						'headless': headless,
						'user_data_dir': '/tmp/openbrowser-mcp-test-profile',
					}
				},
				'llm': {'llm-a': {'id': 'llm-a', 'default': True, 'model': model}},
				'agent': {'agent-a': {'id': 'agent-a', 'default': True}},
			}
		)
	)


@pytest.mark.asyncio
async def test_blocked_config_prevents_mcp_namespace_start_and_reports_status(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	server = OpenBrowserServer()

	with patch('openbrowser.browser.BrowserSession.start') as start, pytest.raises(ConfigStartupError) as error:
		await server._ensure_namespace()

	assert error.value.code == 'config_restart_required'
	assert start.called is False

	payload = server._status_payload()
	assert payload['config']['startup']['phase'] == 'invalid'
	assert payload['config']['startup']['error_code'] == 'config_invalid_json'
	assert payload['config']['current']['phase'] == 'invalid'
	assert payload['config']['restart_required'] is True
	assert payload['browser']['phase'] == 'stopped'
	assert payload['downloads']['pdf_detection']['phase'] == 'idle'


@pytest.mark.asyncio
async def test_modern_context_uses_server_startup_snapshot(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	server = OpenBrowserServer()
	owner = OpenBrowserContext(server, 'ctx-test')

	with patch('openbrowser.browser.BrowserSession.start') as start, pytest.raises(ConfigStartupError) as error:
		await owner._ensure_namespace()

	assert error.value.code == 'config_restart_required'
	assert start.called is False


@pytest.mark.asyncio
async def test_mcp_valid_to_valid_config_change_requires_restart(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_mcp_config(isolated_config_path, headless=True)
	server = OpenBrowserServer()
	_write_valid_mcp_config(isolated_config_path, headless=False, model='gpt-5-mini-extra')

	with patch('openbrowser.browser.BrowserSession.start') as start, pytest.raises(ConfigStartupError) as error:
		await server._ensure_namespace()

	assert error.value.code == 'config_restart_required'
	assert start.called is False
	payload = server._status_payload()
	assert payload['config']['startup']['phase'] == 'ready'
	assert payload['config']['current']['phase'] == 'ready'
	assert payload['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_mcp_owned_tool_raises_blocked_config_before_browser_start(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	server = OpenBrowserServer()

	with patch('openbrowser.browser.BrowserSession.start') as start, pytest.raises(ConfigStartupError) as error:
		await server._run_owned_tool(server, 'execute_code', {'code': 'print(1)'})

	assert error.value.code == 'config_restart_required'
	assert start.called is False


@pytest.mark.asyncio
async def test_mcp_status_does_not_start_browser_and_preserves_existing_keys(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	server = OpenBrowserServer()

	with patch('openbrowser.browser.BrowserSession.start') as start:
		payload = server._status_payload()

	assert start.called is False
	assert {
		'server',
		'browser',
		'profile',
		'instance_id',
		'context_id',
		'namespace_variables',
		'pending_variables',
		'namespace_transfer',
		'session_timeout_minutes',
		'exec_timeout',
		'uptime_seconds',
		'version',
		'tools',
		'downloads',
		'storage_state',
		'storage_state_save',
	} <= payload.keys()
	assert payload['config']['startup']['phase'] == 'invalid'
	assert payload['config']['current']['phase'] == 'invalid'
	assert payload['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_mcp_repair_after_construction_still_requires_restart(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	isolated_config_path.write_text('NOT JSON {{{')
	server = OpenBrowserServer()
	_write_valid_mcp_config(isolated_config_path)

	with patch('openbrowser.browser.BrowserSession.start') as start, pytest.raises(ConfigStartupError) as error:
		await server._ensure_namespace()

	assert error.value.code == 'config_restart_required'
	assert start.called is False
	payload = server._status_payload()
	assert payload['config']['startup']['phase'] == 'invalid'
	assert payload['config']['current']['phase'] == 'ready'
	assert payload['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_mcp_recovery_gate_rejects_blocked_config_before_cleanup(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_mcp_config(isolated_config_path)
	server = OpenBrowserServer()
	old_session = MagicMock()
	old_session.kill = AsyncMock()
	server.browser_session = old_session
	server._executor = MagicMock()
	isolated_config_path.write_text('NOT JSON {{{')

	with (
		patch.object(server, '_is_cdp_alive', AsyncMock(return_value=False)),
		patch('openbrowser.browser.BrowserSession.start') as start,
		pytest.raises(ConfigStartupError) as error,
	):
		await server._recover_browser_session()

	assert error.value.code == 'config_restart_required'
	old_session.kill.assert_not_awaited()
	assert server.browser_session is old_session
	assert start.called is False
	payload = server._status_payload()
	assert payload['config']['startup']['phase'] == 'ready'
	assert payload['config']['restart_required'] is True


@pytest.mark.asyncio
async def test_mcp_blocked_recovery_preserves_partial_output_and_warning(isolated_config_path, monkeypatch):
	monkeypatch.setenv('OPENBROWSER_CONFIG_PATH', str(isolated_config_path))
	_write_valid_mcp_config(isolated_config_path)
	server = OpenBrowserServer()
	server._executor = MagicMock()
	server._executor.initialized = True
	server._namespace = {'__builtins__': __builtins__}
	server.browser_session = MagicMock()
	server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='partial user output',
			error='websocket connection closed',
		)
	)
	server._recover_browser_session = AsyncMock(
		side_effect=ConfigStartupError(
			server._startup_config_result.snapshot,
			code='config_restart_required',
			hint='Fix the config, then restart the daemon or MCP server.',
		)
	)

	with (
		patch.object(server, '_is_cdp_alive', AsyncMock(return_value=True)),
		pytest.raises(ConfigStartupError) as error,
	):
		await server._execute_code_locked('print(1)')

	assert 'partial user output' in str(error.value)
	assert 'config_restart_required' in str(error.value)
	assert 'may have partially run' in error.value.hint


@pytest.mark.asyncio
async def test_mcp_failed_execution_sanitizes_executor_result():
	from openbrowser.mcp.server import OpenBrowserServer

	server = OpenBrowserServer()
	server._executor = MagicMock()
	server._executor.initialized = True
	server._executor.max_output_chars = 12_000
	server._namespace = {'__builtins__': __builtins__}
	server.browser_session = None
	server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='retained-context-' + 'x' * 10_500 + ' Authorization: Bearer mcp-executor-output-secret',
			error='RuntimeError: Authorization: Bearer mcp-executor-error-secret',
		)
	)

	result = await server._execute_code_locked('raise RuntimeError()')

	assert len(result) > 10_000
	assert 'retained-context-' in result
	assert 'mcp-executor-output-secret' not in result
	assert 'mcp-executor-error-secret' not in result
	assert 'authorization=<redacted>' in result.lower()


@pytest.mark.asyncio
async def test_mcp_real_executor_failure_preserves_traceback_after_redaction():
	from openbrowser.code_use.executor import CodeExecutor
	from openbrowser.mcp.server import OpenBrowserServer

	server = OpenBrowserServer()
	executor = CodeExecutor()
	namespace = {'__builtins__': __builtins__}
	executor.set_namespace(namespace)
	server._executor = executor
	server._namespace = namespace
	server.browser_session = None

	result = await server._execute_code_locked(
		"print('Authorization:')\n"
		"print('  Bearer real-mcp-executor-secret second-token')\n"
		"print('Proxy-Authorization: Bearer')\n"
		"print('  q7')\n"
		"print('  request_id=123')\n"
		"print('Cookie: session=')\n"
		"print('  c9')\n"
		"print('  a=b')\n"
		"print('  csrf=xyz')\n"
		"print('  captured diagnostic = expected')\n"
		"raise RuntimeError('boom')\n"
	)

	assert 'Traceback (most recent call last):' in result
	assert '  File "<execute_code>"' in result
	assert '  captured diagnostic = expected' in result
	assert '  request_id=123' in result
	assert 'real-mcp-executor-secret' not in result
	assert 'second-token' not in result
	assert 'q7' not in result
	assert 'c9' not in result
	assert 'a=b' not in result
	assert 'csrf=xyz' not in result
	assert 'authorization=<redacted>' in result.lower()


@pytest.mark.asyncio
async def test_mcp_failed_execution_respects_small_output_cap():
	from openbrowser.mcp.server import OpenBrowserServer

	server = OpenBrowserServer()
	server._executor = MagicMock()
	server._executor.initialized = True
	server._executor.max_output_chars = 2
	server._namespace = {'__builtins__': __builtins__}
	server.browser_session = None
	server._executor.execute = AsyncMock(return_value=SimpleNamespace(success=False, output='x' * 1000, error=None))

	result = await server._execute_code_locked('raise RuntimeError()')

	assert len(result) <= 2


@pytest.mark.asyncio
async def test_mcp_recovery_failure_returns_sanitized_error():
	from openbrowser.mcp.server import OpenBrowserServer

	server = OpenBrowserServer()
	server._executor = MagicMock()
	server._executor.initialized = True
	server._namespace = {'__builtins__': __builtins__}
	server.browser_session = MagicMock()
	server._executor.execute = AsyncMock(
		return_value=SimpleNamespace(
			success=False,
			output='websocket connection closed\nAuthorization: Bearer mcp-partial-secret',
			error=None,
		)
	)
	server._recover_browser_session = AsyncMock(side_effect=RuntimeError('Authorization: Bearer mcp-recovery-secret'))

	with patch.object(server, '_is_cdp_alive', AsyncMock(return_value=True)):
		result = await server._execute_code_locked('print(1)')

	assert 'mcp-recovery-secret' not in result
	server._recover_browser_session.assert_awaited_once()
	assert 'mcp-partial-secret' not in result
	assert 'authorization=<redacted>' in result.lower()
	assert 'websocket connection closed' in result.lower()
	assert 'The code may have partially run' in result
	assert 'run get_status before retrying' in result.lower()


@pytest.mark.asyncio
async def test_get_status_falls_back_to_cdp_health_when_snapshot_fails(mcp_server):
	session = BrowserSession()
	root = MagicMock()
	root.send.Browser.getVersion = AsyncMock(return_value={})
	session._cdp_client_root = root
	session._cdp_health.mark_cleanup_failure(RuntimeError('retained cleanup'))
	mcp_server.browser_session = session

	with patch.object(BrowserSession, 'lifecycle_snapshot', side_effect=RuntimeError('snapshot failed')):
		payload = mcp_server._status_payload()

	assert payload['browser']['cdp']['phase'] == 'disconnected'
	assert payload['browser']['cdp']['healthy'] is None
	assert payload['browser']['cdp']['cleanup_pending'] is True
	assert payload['browser']['cdp_connected'] is True
	assert payload['browser']['health']['phase'] == 'unknown'
	assert payload['browser']['health']['reason'] == 'initial'
	root.send.Browser.getVersion.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_status_reports_profile_resolution_sources(mcp_server, tmp_path):
	mcp_server.config = {
		'browser_profile': {
			'user_data_dir': str(tmp_path / 'explicit-profile'),
			'headless': False,
		}
	}
	mcp_server._profile_summary = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	resolution = payload['profile']['resolution']
	assert resolution['user_data_dir'] == {
		'value': str(tmp_path / 'explicit-profile'),
		'source': 'config',
	}
	assert resolution['headless'] == {'value': False, 'source': 'config'}
	assert resolution['storage_state'] == {'mode': 'none', 'source': 'default'}
	assert payload['profile']['user_data_dir'] == str(tmp_path / 'explicit-profile')
	assert payload['profile']['headless'] is False
	assert payload['profile']['storage_state'] is None


@pytest.mark.asyncio
async def test_get_status_reports_managed_profile_and_storage(mcp_server):
	mcp_server.config = {'browser_profile': {}}
	mcp_server._profile_summary = None
	mcp_server._managed_profile_dir = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	resolution = payload['profile']['resolution']
	assert resolution['user_data_dir']['source'] == 'managed'
	assert resolution['storage_state'] == {'mode': 'file', 'source': 'managed'}
	assert resolution['args'] == {'mode': 'configured', 'source': 'managed', 'count': 1}
	assert payload['profile']['headless'] is True
	assert payload['profile']['storage_state'].endswith('/storage_state.json')
	assert payload['profile']['instance_id'] == mcp_server.owner_id


@pytest.mark.asyncio
async def test_get_status_reports_environment_headless(mcp_server, monkeypatch):
	mcp_server.config = {'browser_profile': {'headless': True}}
	mcp_server._profile_summary = None
	monkeypatch.setenv('OPENBROWSER_HEADLESS', 'false')

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	assert payload['profile']['resolution']['headless'] == {
		'value': False,
		'source': 'environment',
	}


@pytest.mark.asyncio
async def test_get_status_reports_in_memory_storage_state_provenance(mcp_server):
	mcp_server.config = {
		'browser_profile': {
			'storage_state': {
				'cookies': [{'name': 'session', 'value': 'secret-cookie-value'}],
				'origins': [],
			}
		}
	}
	mcp_server._profile_summary = None
	mcp_server._managed_profile_dir = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	assert payload['profile']['resolution']['storage_state'] == {
		'mode': 'memory',
		'source': 'config',
	}
	assert payload['profile']['storage_state'] is None
	assert payload['profile']['storage_state_mode'] == 'memory'
	assert 'secret-cookie-value' not in json.dumps(payload)


@pytest.mark.asyncio
async def test_get_status_reports_environment_paths(mcp_server, monkeypatch, tmp_path, isolated_config_path):
	from openbrowser.config import CONFIG

	_write_valid_mcp_config(isolated_config_path)
	configured_profile = tmp_path / 'configured-profile'
	configured_state = tmp_path / 'configured-state.json'
	configured_state.write_text('{"cookies": [], "origins": []}')
	config = json.loads(isolated_config_path.read_text())
	config['browser_profile']['profile-a'].update(
		{
			'user_data_dir': str(configured_profile),
			'storage_state': str(configured_state),
			'proxy': {'server': 'http://configured-proxy.test:8080'},
		}
	)
	isolated_config_path.write_text(json.dumps(config))
	env_profile = tmp_path / 'environment-profile'
	env_state = tmp_path / 'environment-state.json'
	env_state.write_text('{"cookies": [], "origins": []}')
	monkeypatch.setenv('OPENBROWSER_USER_DATA_DIR', str(env_profile))
	monkeypatch.setenv('OPENBROWSER_STORAGE_STATE', str(env_state))
	monkeypatch.setenv('OPENBROWSER_PROXY_URL', 'http://user:secret-password@proxy.test:8080')
	CONFIG._env_config = None
	mcp_server.config = CONFIG.config_dict_from_result(CONFIG.load_result())
	mcp_server._profile_summary = None
	mcp_server._managed_profile_dir = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	resolution = payload['profile']['resolution']
	assert resolution['user_data_dir'] == {
		'value': str(env_profile.resolve()),
		'source': 'environment',
	}
	assert resolution['storage_state'] == {'mode': 'file', 'source': 'environment'}
	assert resolution['proxy'] == {'mode': 'configured', 'source': 'environment'}
	assert payload['profile']['user_data_dir'] == str(env_profile.resolve())
	assert payload['profile']['storage_state'] == str(env_state)
	assert 'secret-password' not in json.dumps(payload)


@pytest.mark.asyncio
async def test_get_status_reports_validated_effective_values(mcp_server, tmp_path):
	mcp_server.config = {
		'browser_profile': {
			'user_data_dir': str(tmp_path / 'validated-profile'),
			'keep_alive': 'false',
			'disable_security': 1,
			'wait_between_actions': '0.25',
			'device_scale_factor': '1.5',
			'proxy': {},
		}
	}
	mcp_server._profile_summary = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	resolution = payload['profile']['resolution']
	assert resolution['keep_alive'] == {'value': False, 'source': 'config'}
	assert resolution['disable_security'] == {'value': True, 'source': 'config'}
	assert resolution['wait_between_actions'] == {'value': 0.25, 'source': 'config'}
	assert resolution['device_scale_factor'] == {'value': 1.5, 'source': 'config'}
	assert resolution['proxy'] == {'mode': 'none', 'source': 'config'}


@pytest.mark.asyncio
async def test_get_status_reports_configured_storage_state(mcp_server, tmp_path):
	state_path = tmp_path / 'configured-state.json'
	state_path.write_text('{"cookies": [], "origins": []}')
	mcp_server.config = {
		'browser_profile': {
			'user_data_dir': str(tmp_path / 'configured-profile'),
			'storage_state': str(state_path),
		}
	}
	mcp_server._profile_summary = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	assert payload['profile']['resolution']['storage_state'] == {
		'mode': 'file',
		'source': 'config',
	}
	assert payload['profile']['storage_state'] == str(state_path)
	assert payload['profile']['storage_state_mode'] == 'file'


@pytest.mark.asyncio
async def test_get_status_preserves_explicit_config_none_storage_state(mcp_server, tmp_path):
	mcp_server.config = {
		'browser_profile': {
			'user_data_dir': str(tmp_path / 'configured-profile'),
			'storage_state': None,
		}
	}
	mcp_server._profile_summary = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	assert payload['profile']['resolution']['storage_state'] == {
		'mode': 'none',
		'source': 'config',
	}


@pytest.mark.asyncio
async def test_get_status_reports_configured_storage_state_copied_to_managed_path(mcp_server, tmp_path):
	source = tmp_path / 'source-state.json'
	source.write_text('{"cookies": [], "origins": []}')
	mcp_server.config = {'browser_profile': {'storage_state': str(source)}}
	mcp_server._profile_summary = None
	mcp_server._managed_profile_dir = None

	with patch.object(BrowserSession, 'start') as start:
		mcp_server._build_browser_profile()
		payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert start.called is False
	assert payload['profile']['resolution']['storage_state'] == {
		'mode': 'file',
		'source': 'managed',
	}
	managed_path = payload['profile']['storage_state']
	assert managed_path.endswith(f'mcp-{mcp_server.owner_id}/storage_state.json')
	assert managed_path != str(source)


@pytest.mark.asyncio
async def test_get_status_reports_cdp_health_without_probing(mcp_server):
	class _NoProbeSession(BrowserSession):
		async def probe_cdp_connection(self, *, timeout_seconds=2.0):
			raise AssertionError('get_status must not probe CDP health')

	mcp_server.browser_session = _NoProbeSession()

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['browser']['cdp_connected'] is False
	assert payload['browser']['cdp']['phase'] == 'disconnected'
	assert payload['browser']['cdp']['healthy'] is None


@pytest.mark.asyncio
async def test_is_cdp_alive_delegates_to_the_session_probe(mcp_server):
	session = BrowserSession()
	root = MagicMock()
	root.send = MagicMock()
	root.send.Browser = MagicMock()
	root.send.Browser.getVersion = AsyncMock(return_value={'product': 'Chrome'})
	session._cdp_client_root = root
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.READY)
	mcp_server.browser_session = session

	alive = await mcp_server._is_cdp_alive()
	health = session.lifecycle_snapshot()['cdp']

	assert alive is True
	assert health['probe_attempts'] == 1
	assert health['healthy'] is True


@pytest.mark.asyncio
async def test_get_status_exposes_the_last_mcp_probe(mcp_server):
	session = BrowserSession()
	root = MagicMock()
	root.send = MagicMock()
	root.send.Browser = MagicMock()
	root.send.Browser.getVersion = AsyncMock(side_effect=TimeoutError('timed out'))
	session._cdp_client_root = root
	session._lifecycle.transition(LifecyclePhase.STARTING)
	session._lifecycle.transition(LifecyclePhase.READY)
	mcp_server.browser_session = session

	assert await mcp_server._is_cdp_alive() is False
	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['browser']['cdp_connected'] is True
	assert payload['browser']['cdp']['phase'] == 'degraded'
	assert payload['browser']['cdp']['healthy'] is False
	assert payload['browser']['cdp']['last_error_code'] == 'timeout'


@pytest.mark.asyncio
async def test_is_cdp_alive_requires_client_presence_and_health(mcp_server):
	class _ProbeSession:
		def __init__(self, *, client_present, healthy):
			self.client_present = client_present
			self.healthy = healthy
			self.timeouts = []

		async def probe_cdp_connection(self, *, timeout_seconds):
			self.timeouts.append(timeout_seconds)
			return SimpleNamespace(client_present=self.client_present, healthy=self.healthy)

	not_connected = _ProbeSession(client_present=False, healthy=True)
	mcp_server.browser_session = not_connected
	assert await mcp_server._is_cdp_alive() is False

	unhealthy = _ProbeSession(client_present=True, healthy=False)
	mcp_server.browser_session = unhealthy
	assert await mcp_server._is_cdp_alive() is False

	healthy = _ProbeSession(client_present=True, healthy=True)
	mcp_server.browser_session = healthy
	assert await mcp_server._is_cdp_alive(deadline=time.monotonic() + 0.25) is True
	assert 0 < healthy.timeouts[0] <= 0.25


@pytest.mark.asyncio
async def test_is_cdp_alive_skips_an_expired_deadline(mcp_server):
	class _ProbeSession:
		def __init__(self):
			self.calls = 0

		async def probe_cdp_connection(self, *, timeout_seconds):
			self.calls += 1
			return SimpleNamespace(client_present=True, healthy=True)

	session = _ProbeSession()
	mcp_server.browser_session = session

	assert await mcp_server._is_cdp_alive(deadline=time.monotonic() - 1) is False
	assert session.calls == 0


@pytest.mark.asyncio
async def test_is_cdp_alive_passes_two_second_default_timeout_to_session_probe(mcp_server):
	class _ProbeSession:
		def __init__(self):
			self.timeouts = []

		async def probe_cdp_connection(self, *, timeout_seconds):
			self.timeouts.append(timeout_seconds)
			return SimpleNamespace(client_present=True, healthy=True)

	session = _ProbeSession()
	mcp_server.browser_session = session

	assert await mcp_server._is_cdp_alive() is True
	assert session.timeouts == [2.0]


@pytest.mark.asyncio
async def test_is_cdp_alive_returns_false_when_the_session_probe_raises(mcp_server):
	class _ProbeSession:
		async def probe_cdp_connection(self, *, timeout_seconds):
			raise RuntimeError('probe failed')

	mcp_server.browser_session = _ProbeSession()

	assert await mcp_server._is_cdp_alive() is False
