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
async def test_reset_browser_preserves_user_variables(mcp_server):
	session = MagicMock()
	session.kill = AsyncMock()
	mcp_server.browser_session = session
	mcp_server._namespace = {
		'user_value': 42,
		'browser': object(),
		'navigate': object(),
	}
	mcp_server._browser_lifecycle.transition(LifecyclePhase.STARTING)
	mcp_server._browser_lifecycle.transition(LifecyclePhase.READY)

	result = await mcp_server._handle_tool('reset_browser', {})
	payload = json.loads(result)

	assert payload['ok'] is True
	assert payload['preserved_variables'] == 3
	assert mcp_server.browser_session is None
	assert mcp_server._namespace is None
	assert 'user_value' in mcp_server._pending_variables
	assert 'browser' in mcp_server._pending_variables
	assert payload['browser']['phase'] == 'stopped'


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
	assert mcp_server._executor.namespace == {}


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
