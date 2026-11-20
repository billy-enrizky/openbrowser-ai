"""Tests for MCP lifecycle, status, and reset tools."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.mcp.server import _CleanupOutcome


@pytest.mark.asyncio
async def test_get_status_does_not_start_browser(mcp_server):
	result = await mcp_server._handle_tool('get_status', {})
	payload = json.loads(result)

	assert mcp_server.browser_session is None
	assert payload['browser']['phase'] == 'stopped'
	assert payload['namespace_variables'] == 0
	assert payload['profile'] is None


@pytest.mark.asyncio
async def test_get_status_reports_storage_state_previews_without_browser(mcp_server, tmp_path):
	state_path = tmp_path / 'state.json'
	state_path.write_text('{"cookies": [], "origins": []}')
	mcp_server._profile_summary = {'storage_state': str(state_path)}

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['storage_state']['phase'] == 'pending'
	assert payload['storage_state']['path'] == str(state_path)
	assert payload['storage_state_save']['phase'] == 'idle'
	assert payload['storage_state_save']['path'] == str(state_path)


@pytest.mark.asyncio
async def test_get_status_uses_config_storage_state_before_browser(mcp_server, tmp_path):
	state_path = tmp_path / 'config-state.json'
	state_path.write_text('{"cookies": [], "origins": []}')
	mcp_server.config = {'browser_profile': {'storage_state': str(state_path)}}
	mcp_server._profile_summary = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['storage_state']['phase'] == 'pending'
	assert payload['storage_state']['path'] == str(state_path)
	assert payload['storage_state_save']['phase'] == 'idle'
	assert payload['storage_state_save']['path'] == str(state_path)


@pytest.mark.asyncio
async def test_get_status_reports_dict_storage_state_as_skipped(mcp_server):
	mcp_server.config = {
		'browser_profile': {'storage_state': {'cookies': [], 'origins': []}}
	}
	mcp_server._profile_summary = None

	payload = json.loads(await mcp_server._handle_tool('get_status', {}))

	assert payload['storage_state']['phase'] == 'skipped'
	assert payload['storage_state_save']['phase'] == 'skipped'


@pytest.mark.asyncio
async def test_reset_browser_preserves_user_variables(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.browser_session = session
	async def navigate():
		return None

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

	with patch.object(mcp_server, '_kill_session_safely', new_callable=AsyncMock, return_value=_CleanupOutcome(True, asyncio.CancelledError())):
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
