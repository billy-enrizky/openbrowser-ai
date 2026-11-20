"""Backend propagation of explicit Browser Agent terminal state."""

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import openbrowser

ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(ROOT / 'backend')
sys.path.insert(0, _BACKEND)
try:
	from app.services.agent_service import AgentSession  # noqa: E402
finally:
	sys.path.remove(_BACKEND)

from openbrowser.agent.views import AgentRunPhase, AgentRunStatus  # noqa: E402


def make_session(status, *, history_success=None, final_result='final'):
	session = AgentSession(
		task_id='task-1',
		task='Do the thing',
		agent_type='browser',
		enable_vnc=False,
		use_current_browser=False,
	)
	session.started_at = datetime.now(UTC).replace(tzinfo=None)
	session.completed_at = None
	history = SimpleNamespace(
		final_result=lambda: final_result,
		is_successful=lambda: history_success,
		total_duration_seconds=lambda: 1.5,
		history=[SimpleNamespace(result=[])],
	)
	session.agent = SimpleNamespace(run=AsyncMock(return_value=history), status=status)
	session.agent.history = history
	session._collect_browser_agent_attachments = lambda history: []
	session.on_output_callback = MagicMock()
	return session


@pytest.mark.asyncio
async def test_succeeded_browser_status_is_success():
	session = make_session(
		AgentRunStatus(
			phase=AgentRunPhase.SUCCEEDED,
			success=True,
			terminal_reason='done',
			result='final',
		),
		history_success=True,
	)

	result = await session._run_browser_agent()

	assert result['success'] is True
	assert result['cancelled'] is False
	assert result['phase'] == 'succeeded'
	assert result['terminal_phase'] == 'succeeded'
	assert result['terminal_reason'] == 'done'
	session.on_output_callback.assert_called_once_with('final', True)


@pytest.mark.asyncio
async def test_failed_and_max_step_browser_status_are_not_success():
	for phase, reason in [
		(AgentRunPhase.FAILED, 'max_failures'),
		(AgentRunPhase.MAX_STEPS, 'max_steps'),
	]:
		session = make_session(
			AgentRunStatus(
				phase=phase,
				success=False,
				terminal_reason=reason,
				result='partial',
				error='failed' if phase is AgentRunPhase.FAILED else None,
			),
			history_success=None,
		)

		result = await session._run_browser_agent()

		assert result['success'] is False
		assert result['terminal_phase'] == phase.value
		assert result['terminal_reason'] == reason
		session.on_output_callback.assert_called_once_with('partial', False)


@pytest.mark.asyncio
async def test_stopped_browser_status_is_cancellation():
	session = make_session(
		AgentRunStatus(
			phase=AgentRunPhase.STOPPED,
			success=False,
			terminal_reason='stopped',
			result='partial data',
		),
		history_success=None,
		final_result='partial data',
	)

	result = await session._run_browser_agent()

	assert result['success'] is False
	assert result['cancelled'] is True
	assert result['terminal_phase'] == 'stopped'
	assert result['terminal_reason'] == 'stopped'
	assert result['result'] == 'partial data'
	assert session.task_status == 'cancelled'


@pytest.mark.asyncio
async def test_stopped_follow_up_does_not_reuse_previous_history_result():
	session = make_session(
		AgentRunStatus(
			phase=AgentRunPhase.STOPPED,
			success=False,
			terminal_reason='stopped_before_start',
			result=None,
		),
		history_success=True,
		final_result='previous result',
	)

	result = await session._run_browser_agent()

	assert result['result'] == 'Task was cancelled'
	assert result['cancelled'] is True


def test_paused_browser_session_reports_paused_status():
	session = make_session(AgentRunStatus(phase=AgentRunPhase.PAUSED))
	session.is_running = True
	session.is_paused = False

	assert session.task_status == 'paused'


def test_terminal_agent_phase_is_visible_before_session_finalization():
	for phase, expected in [
		(AgentRunPhase.SUCCEEDED, 'completed'),
		(AgentRunPhase.FAILED, 'failed'),
		(AgentRunPhase.MAX_STEPS, 'failed'),
		(AgentRunPhase.STOPPED, 'cancelled'),
		(AgentRunPhase.CANCELLED, 'cancelled'),
	]:
		session = make_session(AgentRunStatus(phase=phase))
		session.is_running = True

		assert session.task_status == expected


@pytest.mark.asyncio
async def test_session_pause_and_resume_follow_agent_acceptance():
    session = make_session(AgentRunStatus(phase=AgentRunPhase.RUNNING))
    session.is_running = True
    session.agent.pause = MagicMock(return_value=True)
    session.agent.resume = MagicMock(return_value=True)

    assert await session.pause() is True
    assert session.is_paused is True
    assert session.task_status == 'paused'
    assert await session.resume() is True
    assert session.is_paused is False

    session.agent.pause = MagicMock(return_value=False)
    assert await session.pause() is False
    assert session.is_paused is False


@pytest.mark.asyncio
async def test_cancel_clears_paused_status_and_is_accepted_before_first_step():
    session = make_session(AgentRunStatus(phase=AgentRunPhase.IDLE))
    session.is_running = True
    session.is_paused = True
    session.agent.stop = MagicMock(return_value=True)

    assert await session.cancel() is True
    assert session.is_cancelled is True
    assert session.is_paused is False
    assert session.task_status == 'stopping'


@pytest.mark.asyncio
async def test_legacy_browser_agent_without_status_uses_history():
	session = make_session(None, history_success=True, final_result='legacy')

	result = await session._run_browser_agent()

	assert result['success'] is True
	assert result['phase'] is None
	assert result['terminal_phase'] is None
	assert result['result'] == 'legacy'


@pytest.mark.asyncio
async def test_browser_agent_run_error_maps_terminal_status():
	session = make_session(
		AgentRunStatus(
			phase=AgentRunPhase.FAILED,
			success=False,
			terminal_reason='run_error',
			error='RuntimeError: boom',
		),
		history_success=None,
		final_result=None,
	)
	session.agent.run = AsyncMock(side_effect=RuntimeError('boom'))

	result = await session._run_browser_agent()

	assert result['success'] is False
	assert result['terminal_phase'] == 'failed'
	assert result['terminal_reason'] == 'run_error'
	assert result['result'] == 'Task failed: RuntimeError: boom'


@pytest.mark.asyncio
async def test_cancellation_after_committed_success_preserves_result():
	session = make_session(
		AgentRunStatus(
			phase=AgentRunPhase.SUCCEEDED,
			success=True,
			terminal_reason='done',
			result='accepted',
		),
		history_success=True,
		final_result='accepted',
	)
	session.agent.run = AsyncMock(side_effect=asyncio.CancelledError())

	result = await session._run_browser_agent()

	assert result['success'] is True
	assert result['cancelled'] is False
	assert result['terminal_phase'] == 'succeeded'
	assert result['result'] == 'accepted'


@pytest.mark.asyncio
async def test_cancellation_after_committed_max_steps_preserves_the_outcome():
	session = make_session(
		AgentRunStatus(
			phase=AgentRunPhase.MAX_STEPS,
			success=False,
			terminal_reason='max_steps',
			result='partial',
		),
		history_success=None,
		final_result='partial',
	)
	session.agent.run = AsyncMock(side_effect=asyncio.CancelledError())

	result = await session._run_browser_agent()

	assert result['success'] is False
	assert result['cancelled'] is False
	assert result['terminal_phase'] == 'max_steps'
	assert result['terminal_reason'] == 'max_steps'
	assert result['result'] == 'partial'


@pytest.mark.asyncio
async def test_task_status_prefers_a_terminal_phase_over_pending_cancellation():
	session = make_session(
		AgentRunStatus(
			phase=AgentRunPhase.STOPPED,
			success=False,
			terminal_reason='stopped',
			result='partial',
		),
		history_success=None,
	)
	session.is_cancelled = True
	session.is_running = True
	session.completed_at = None

	assert session.task_status == 'cancelled'

	session.agent.status = AgentRunStatus(
		phase=AgentRunPhase.SUCCEEDED,
		success=True,
		terminal_reason='done',
		result='accepted',
	)

	assert session.task_status == 'completed'


@pytest.mark.asyncio
async def test_browser_agent_start_disables_process_signal_handler(monkeypatch):
	session = AgentSession(
		task_id='task-1',
		task='Do the thing',
		agent_type='browser',
		enable_vnc=False,
		use_current_browser=False,
	)
	captured = {}

	class FakeAgent:
		def __init__(self, **kwargs):
			captured.update(kwargs)

		def stop(self):
			return True

	monkeypatch.setattr(openbrowser, 'Agent', FakeAgent)
	monkeypatch.setattr(session, '_setup_vnc', AsyncMock(return_value=None))
	monkeypatch.setattr(session, '_setup_new_browser', AsyncMock())
	monkeypatch.setattr(session, '_create_llm', MagicMock(return_value=MagicMock()))
	monkeypatch.setattr(session, '_cleanup', AsyncMock())
	monkeypatch.setattr(
		session,
		'_run_browser_agent',
		AsyncMock(
			return_value={
				'task_id': session.task_id,
				'result': 'done',
				'success': True,
				'cancelled': False,
				'terminal_phase': 'succeeded',
				'terminal_reason': 'done',
				'total_steps': 1,
				'duration_seconds': 1,
				'attachments': [],
			}
		),
	)

	await session.start()

	assert captured['use_signal_handler'] is False

@pytest.mark.asyncio
async def test_cleanup_cancellation_preserves_a_committed_result(monkeypatch):
	session = AgentSession(
		task_id='task-1',
		task='Do the thing',
		agent_type='browser',
		enable_vnc=False,
		use_current_browser=False,
	)

	class FakeAgent:
		def __init__(self, **kwargs):
			pass

		def stop(self):
			return True

	async def committed_result():
		session.completed_at = datetime.now(UTC).replace(tzinfo=None)
		session.result = 'accepted'
		session.success = True
		return {
			'task_id': session.task_id,
			'result': 'accepted',
			'success': True,
			'cancelled': False,
			'terminal_phase': 'succeeded',
			'terminal_reason': 'done',
			'total_steps': 1,
			'duration_seconds': 1,
			'attachments': [],
		}

	monkeypatch.setattr(openbrowser, 'Agent', FakeAgent)
	monkeypatch.setattr(session, '_setup_vnc', AsyncMock(return_value=None))
	monkeypatch.setattr(session, '_setup_new_browser', AsyncMock())
	monkeypatch.setattr(session, '_create_llm', MagicMock(return_value=MagicMock()))
	cleanup = AsyncMock(side_effect=asyncio.CancelledError())
	monkeypatch.setattr(session, '_cleanup', cleanup)
	monkeypatch.setattr(session, '_run_browser_agent', committed_result)

	result = await session.start()

	assert result['success'] is True
	assert result['result'] == 'accepted'
	cleanup.assert_awaited_once()

@pytest.mark.asyncio
async def test_cleanup_cancellation_still_raises_before_a_committed_result(monkeypatch):
	session = AgentSession(
		task_id='task-1',
		task='Do the thing',
		agent_type='browser',
		enable_vnc=False,
		use_current_browser=False,
	)

	class FakeAgent:
		def __init__(self, **kwargs):
			pass

		def stop(self):
			return True

	cleanup = AsyncMock(side_effect=asyncio.CancelledError())
	monkeypatch.setattr(openbrowser, 'Agent', FakeAgent)
	monkeypatch.setattr(session, '_setup_vnc', AsyncMock(return_value=None))
	monkeypatch.setattr(session, '_setup_new_browser', AsyncMock())
	monkeypatch.setattr(session, '_create_llm', MagicMock(return_value=MagicMock()))
	monkeypatch.setattr(session, '_cleanup', cleanup)
	monkeypatch.setattr(
		session,
		'_run_browser_agent',
		AsyncMock(side_effect=asyncio.CancelledError()),
	)

	with pytest.raises(asyncio.CancelledError):
		await session.start()

	cleanup.assert_awaited_once()
