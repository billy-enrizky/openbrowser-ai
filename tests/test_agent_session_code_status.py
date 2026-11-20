"""Backend propagation of the explicit CodeAgent terminal status."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import openbrowser
import openbrowser.tools.service as openbrowser_tools

ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(ROOT / 'backend')
sys.path.insert(0, _BACKEND)
try:
    from app.services.agent_service import AgentSession  # noqa: E402
finally:
    sys.path.remove(_BACKEND)
from openbrowser.code_use.views import CodeAgentPhase, CodeAgentRunStatus  # noqa: E402


def make_session(phase, *, result=None, error=None, reason=None, success=None):
    session = AgentSession(
        task_id='task-1',
        task='Do the thing',
        agent_type='code',
        enable_vnc=False,
        use_current_browser=True,
    )
    session.started_at = datetime.now()
    session.completed_at = session.started_at + timedelta(seconds=1)
    status = CodeAgentRunStatus(
        phase=phase,
        step=2,
        max_steps=5,
        success=success,
        terminal_reason=reason,
        result=result,
        error=error,
    )
    session.agent = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace()),
        status=status,
        namespace={},
        complete_history=[],
        _is_task_done=lambda: phase is CodeAgentPhase.SUCCEEDED,
    )
    session._collect_code_agent_attachments = lambda: []
    session.on_output_callback = MagicMock()
    return session


@pytest.mark.asyncio
async def test_start_replays_cancellation_before_running_code_agent(monkeypatch):
    session = AgentSession(
        task_id='task-1',
        task='Do the thing',
        agent_type='code',
        enable_vnc=False,
        use_current_browser=False,
    )
    created = {}

    class FakeCodeAgent:
        def __init__(self, **kwargs):
            self.stop_calls = 0
            created['agent'] = self
            session.is_cancelled = True

        def stop(self):
            self.stop_calls += 1
            return True

    monkeypatch.setattr(openbrowser, 'CodeAgent', FakeCodeAgent)
    monkeypatch.setattr(openbrowser_tools, 'CodeAgentTools', lambda **kwargs: MagicMock())
    monkeypatch.setattr(session, '_setup_new_browser', AsyncMock())
    monkeypatch.setattr(session, '_create_llm', MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(session, '_cleanup', AsyncMock())

    async def run_code():
        assert created['agent'].stop_calls == 1
        session.result = 'Task was cancelled'
        session.success = False
        return {
            'task_id': session.task_id,
            'result': session.result,
            'success': False,
            'cancelled': True,
            'terminal_phase': 'cancelled',
            'terminal_reason': 'cancelled',
            'total_steps': 0,
            'duration_seconds': 0,
            'attachments': [],
        }

    monkeypatch.setattr(session, '_run_code_agent', run_code)

    result = await session.start()

    assert result['cancelled'] is True
    assert created['agent'].stop_calls == 1
    session._cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_cancelled_before_setup_allocates_no_resources(monkeypatch):
    session = AgentSession(
        task_id='task-1',
        task='Do the thing',
        agent_type='code',
        enable_vnc=False,
        use_current_browser=False,
    )
    await session.cancel()
    assert session.task_status == 'cancelled'
    session._setup_vnc = AsyncMock()
    session._setup_new_browser = AsyncMock()
    session._create_llm = MagicMock()
    session._cleanup = AsyncMock()

    result = await session.start()

    assert result['cancelled'] is True
    assert result['terminal_reason'] == 'cancelled_before_start'
    session._setup_vnc.assert_not_awaited()
    session._setup_new_browser.assert_not_awaited()
    session._create_llm.assert_not_called()
    session._cleanup.assert_awaited_once()
    assert session.task_status == 'cancelled'


@pytest.mark.asyncio
async def test_start_cancelled_during_setup_error_reports_cancelled(monkeypatch):
    session = AgentSession(
        task_id='task-1',
        task='Do the thing',
        agent_type='code',
        enable_vnc=False,
        use_current_browser=False,
    )
    session._setup_vnc = AsyncMock()

    async def fail_after_cancel(*args):
        session.is_cancelled = True
        raise RuntimeError('browser failed')

    session._setup_new_browser = fail_after_cancel
    session._create_llm = MagicMock()
    session._cleanup = AsyncMock()

    result = await session.start()

    assert result['cancelled'] is True
    assert result['terminal_reason'] == 'cancelled_during_setup'
    session._create_llm.assert_not_called()
    session._cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_cancelled_during_setup_skips_agent_creation():
    session = AgentSession(
        task_id='task-1',
        task='Do the thing',
        agent_type='code',
        enable_vnc=False,
        use_current_browser=False,
    )
    session._setup_vnc = AsyncMock()

    async def finish_after_cancel(*args):
        session.is_cancelled = True

    session._setup_new_browser = finish_after_cancel
    session._create_llm = MagicMock()
    session._cleanup = AsyncMock()

    result = await session.start()

    assert result['cancelled'] is True
    assert result['terminal_reason'] == 'cancelled_during_setup'
    session._create_llm.assert_not_called()
    session._cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_reports_whether_the_agent_accepted_the_stop():
    session = make_session(CodeAgentPhase.RUNNING)
    session.is_running = True
    session.completed_at = None

    session.agent.stop = MagicMock(return_value=False)
    assert await session.cancel() is False
    assert session.is_cancelled is False

    session.agent.stop = MagicMock(return_value=True)
    assert await session.cancel() is True
    assert session.is_cancelled is True

    session.agent = None
    assert await session.cancel() is True


@pytest.mark.asyncio
async def test_late_cancel_is_rejected_without_signalling_the_agent():
    session = make_session(CodeAgentPhase.SUCCEEDED, result='done', reason='done', success=True)
    session.completed_at = datetime.now()
    session.is_running = True
    session.agent.stop = MagicMock(return_value=True)

    assert await session.cancel() is False
    session.agent.stop.assert_not_called()
    assert session.is_cancelled is False


def test_session_task_status_is_explicit():
    session = AgentSession(task_id='task-1', task='Do the thing', enable_vnc=False)
    assert session.task_status == 'pending'

    session.is_running = True
    assert session.task_status == 'running'

    session.completed_at = datetime.now()
    session.success = True
    assert session.task_status == 'completed'

    session.success = False
    session.is_cancelled = True
    assert session.task_status == 'cancelled'

    session.is_cancelled = False
    assert session.task_status == 'failed'


@pytest.mark.asyncio
async def test_cancelled_code_status_is_not_completion():
    session = make_session(
        CodeAgentPhase.CANCELLED,
        result='partial data',
        reason='cancelled',
        success=False,
    )

    result = await session._run_code_agent()

    assert result['success'] is False
    assert result['cancelled'] is True
    assert result['terminal_phase'] == 'cancelled'
    assert result['result'] == 'partial data'
    session.on_output_callback.assert_called_once_with('partial data', False)


@pytest.mark.asyncio
async def test_failed_code_status_carries_the_error():
    session = make_session(
        CodeAgentPhase.FAILED,
        error='step_error: boom',
        reason='step_error',
        success=False,
    )

    result = await session._run_code_agent()

    assert result['success'] is False
    assert result['cancelled'] is False
    assert result['terminal_phase'] == 'failed'
    assert 'boom' in result['result']
    session.on_output_callback.assert_called_once_with(result['result'], False)


@pytest.mark.asyncio
async def test_max_steps_is_incomplete_not_success():
    session = make_session(
        CodeAgentPhase.MAX_STEPS,
        result='Task incomplete - reached step limit (5 steps).',
        reason='max_steps',
        success=False,
    )

    result = await session._run_code_agent()

    assert result['success'] is False
    assert result['terminal_phase'] == 'max_steps'
    assert 'incomplete' in result['result']


@pytest.mark.asyncio
async def test_succeeded_code_status_reports_success():
    session = make_session(
        CodeAgentPhase.SUCCEEDED,
        result='final',
        reason='done',
        success=True,
    )

    result = await session._run_code_agent()

    assert result['success'] is True
    assert result['cancelled'] is False
    assert result['terminal_phase'] == 'succeeded'
    assert result['result'] == 'final'
    session.on_output_callback.assert_called_once_with('final', True)


@pytest.mark.asyncio
async def test_code_status_reports_elapsed_time():
    session = make_session(
        CodeAgentPhase.SUCCEEDED,
        result='final',
        reason='done',
        success=True,
    )
    session.started_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=2)
    session.completed_at = None

    result = await session._run_code_agent()

    assert result['duration_seconds'] >= 1
    assert session.completed_at is not None


@pytest.mark.asyncio
async def test_nonterminal_status_does_not_fall_back_to_namespace_success():
    session = make_session(CodeAgentPhase.RUNNING)
    session.agent.namespace = {'_task_done': True, '_task_success': True, '_task_result': 'stale'}
    session.agent._is_task_done = lambda: True

    result = await session._run_code_agent()

    assert result['success'] is False
    assert result['phase'] == 'running'
    assert result['terminal_phase'] is None
    assert result['result'] != 'Task completed'


@pytest.mark.asyncio
async def test_legacy_agent_without_status_uses_namespace_flags():
    session = make_session(CodeAgentPhase.SUCCEEDED)
    session.agent.status = None
    session.agent.namespace = {'_task_done': True, '_task_success': True, '_task_result': 'legacy'}
    session.agent._is_task_done = lambda: True

    result = await session._run_code_agent()

    assert result['success'] is True
    assert result['terminal_phase'] is None
    assert result['result'] == 'legacy'


async def test_attachment_collection_prefers_the_completion_snapshot(tmp_path):
    snapshot_file = tmp_path / 'snapshot.txt'
    stale_file = tmp_path / 'stale.txt'
    snapshot_file.write_text('snapshot', encoding='utf-8')
    stale_file.write_text('stale', encoding='utf-8')
    session = make_session(
        CodeAgentPhase.SUCCEEDED,
        result='done',
        reason='done',
        success=True,
    )
    session.agent.status = CodeAgentRunStatus(
        phase=CodeAgentPhase.SUCCEEDED,
        success=True,
        terminal_reason='done',
        result='done',
        attachments=[str(snapshot_file)],
    )
    session.agent.namespace = {'_task_attachments': [str(stale_file)]}
    del session._collect_code_agent_attachments

    attachments = session._collect_code_agent_attachments()

    assert [item['name'] for item in attachments] == ['snapshot.txt']


@pytest.mark.asyncio
async def test_browser_agent_result_carries_accepted_cancellation():
    session = make_session(CodeAgentPhase.RUNNING)
    session.agent_type = 'browser'
    session.is_running = True
    session.completed_at = None
    session.is_cancelled = True
    history = SimpleNamespace(
        final_result=lambda: 'partial',
        is_successful=lambda: True,
        total_duration_seconds=lambda: 1.5,
        history=[SimpleNamespace(result=[])],
    )
    session.agent = SimpleNamespace(run=AsyncMock(return_value=history))
    session._collect_browser_agent_attachments = lambda history: []

    result = await session._run_browser_agent()

    assert result['cancelled'] is True
    assert result['success'] is False
    assert result['result'] == 'partial'
    assert session.completed_at is not None


@pytest.mark.asyncio
async def test_cleanup_only_detaches_a_debugger_this_session_attached(monkeypatch):
    extension_manager = SimpleNamespace(
        get_any_extension_id=MagicMock(return_value='extension-id'),
        send_command=AsyncMock(),
    )
    extension_handler = ModuleType('app.websocket.extension_handler')
    extension_handler.extension_manager = extension_manager
    monkeypatch.setitem(sys.modules, 'app.websocket.extension_handler', extension_handler)
    session = AgentSession(
        task_id='task-1',
        task='Do the thing',
        enable_vnc=False,
        use_current_browser=True,
    )

    await session._cleanup()
    extension_manager.send_command.assert_not_awaited()

    session._debugger_attached = True
    await session._cleanup()
    extension_manager.send_command.assert_awaited_once_with(
        'extension-id', {'type': 'detach_debugger'}
    )
