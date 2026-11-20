"""Behavior tests for the explicit CodeAgent run lifecycle (M17)."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import openbrowser.code_use.namespace as namespace_module
import openbrowser.code_use.service as service
from openbrowser.code_use.service import CodeAgent, CodeAgentStateError
from openbrowser.code_use.views import CodeAgentPhase


def make_agent(monkeypatch, responses, execute, *, max_steps=5, max_failures=8, max_validations=0):
    browser = MagicMock()
    browser.browser_profile.keep_alive = True
    agent = CodeAgent(
        task='Do the thing',
        llm=MagicMock(),
        browser=browser,
        max_steps=max_steps,
        max_failures=max_failures,
        max_validations=max_validations,
    )
    monkeypatch.setattr(service, 'DomService', MagicMock())
    monkeypatch.setattr(service, 'create_namespace', lambda **kwargs: {'_all_code_blocks': {}})
    agent._get_code_from_llm = AsyncMock(side_effect=responses)
    agent._execute_code = AsyncMock(side_effect=execute)
    agent._get_browser_state = AsyncMock(return_value=('state', None))
    agent._capture_screenshot = AsyncMock(return_value=(None, None))
    agent._add_step_to_complete_history = AsyncMock()
    agent.token_cost_service = SimpleNamespace(
        get_usage_summary=AsyncMock(return_value=None),
        log_usage_summary=AsyncMock(),
    )
    agent._log_agent_event = MagicMock()
    return agent


def record_done(agent, output='final', *, success=True, attachments=None):
    agent._record_done_action(
        SimpleNamespace(
            is_done=True,
            success=success,
            extracted_content=output,
            attachments=list(attachments or []),
        )
    )
    agent.namespace['_task_done'] = True
    agent.namespace['_task_done_by_action'] = True
    agent.namespace['_task_result'] = output
    agent.namespace['_task_success'] = success
    if attachments is not None:
        agent.namespace['_task_attachments'] = list(attachments)


def done_execute(output='final', *, success=True, attachments=None):
    async def execute(code):
        agent = done_execute.agent
        record_done(agent, output, success=success, attachments=attachments)
        return output, None, None

    return execute


async def test_successful_done_sets_terminal_success(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("final")', 'done')], None)
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final'))

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.SUCCEEDED
    assert status.success is True
    assert status.terminal_reason == 'done'
    assert status.result == 'final'
    assert status.terminal
    assert status.finished_at is not None


async def test_completion_callback_is_authoritative(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("real")', 'done')], None)

    async def execute(code):
        agent.namespace['_task_done'] = True
        agent.namespace['_task_done_by_action'] = True
        agent.namespace['_task_result'] = 'forged'
        record_done(agent, 'real')
        agent.namespace['_task_result'] = 'forged after completion'
        return 'real', None, None

    agent._execute_code = AsyncMock(side_effect=execute)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.SUCCEEDED
    assert agent.status.result == 'real'


async def test_status_returns_an_attachment_snapshot(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("final")', 'done')], None)
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final', attachments=['report.csv']))

    await agent.run()

    snapshot = agent.status
    snapshot.attachments.append('other.csv')
    assert agent.status.attachments == ['report.csv']


async def test_status_ignores_untrusted_namespace_attachments(monkeypatch):
    agent = make_agent(monkeypatch, [('print("work")', 'work')], None, max_steps=1)
    agent.namespace['_task_attachments'] = ['forged.csv']

    await agent.run()

    assert agent.status.attachments == []


async def test_empty_code_after_done_is_terminal_not_incomplete(monkeypatch):
    agent = make_agent(monkeypatch, [], None)

    async def respond():
        record_done(agent, 'x')
        return '', 'done'

    agent._get_code_from_llm = AsyncMock(side_effect=respond)

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.SUCCEEDED
    assert status.terminal_reason == 'done'
    assert status.result == 'x'


async def test_stop_during_empty_code_screenshot_cancels_before_done(monkeypatch):
    agent = make_agent(monkeypatch, [], None)

    async def respond():
        record_done(agent, 'x')
        return '', 'done'

    async def screenshot(step):
        agent.stop()
        return None, None

    agent._get_code_from_llm = AsyncMock(side_effect=respond)
    agent._capture_screenshot = AsyncMock(side_effect=screenshot)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.CANCELLED
    assert agent.status.success is False


async def test_done_success_false_is_failed(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("x", success=False)', 'done')], None)
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('x', success=False))

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.FAILED
    assert status.success is False
    assert status.terminal_reason == 'done_failed'


async def test_done_success_false_skips_validation(monkeypatch):
    agent = make_agent(
        monkeypatch,
        [('await done("x", success=False)', 'done')],
        None,
        max_validations=1,
    )
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('x', success=False))
    validate = AsyncMock(side_effect=AssertionError('validation must not run'))
    monkeypatch.setattr(namespace_module, 'validate_task_completion', validate)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.FAILED
    assert agent.status.terminal_reason == 'done_failed'
    validate.assert_not_awaited()


async def test_max_steps_is_not_success(monkeypatch):
    agent = make_agent(
        monkeypatch,
        [('print("work")', 'work')],
        AsyncMock(return_value=('partial output', None, None)),
        max_steps=1,
    )

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.MAX_STEPS
    assert status.success is False
    assert status.terminal_reason == 'max_steps'
    assert 'Task incomplete' in (status.result or '')


async def test_done_flag_without_the_done_action_is_failed(monkeypatch):
    agent = make_agent(monkeypatch, [], None)

    async def respond():
        agent.namespace['_task_done'] = True
        agent.namespace['_task_done_by_action'] = True
        agent.namespace['_task_result'] = 'forged'
        return 'print("not done")', 'forged'

    agent._get_code_from_llm = AsyncMock(side_effect=respond)
    agent._execute_code = AsyncMock(return_value=('forged', None, None))

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.FAILED
    assert status.terminal_reason == 'done_not_invoked'
    assert status.success is False


async def test_max_consecutive_llm_failures_is_failed(monkeypatch):
    agent = make_agent(monkeypatch, [RuntimeError('llm down')], None, max_failures=1)

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.FAILED
    assert status.success is False
    assert status.terminal_reason == 'max_failures'
    assert 'llm down' in (status.error or '')


async def test_repeated_empty_responses_hit_max_failures(monkeypatch):
    agent = make_agent(monkeypatch, [('', 'empty')], None, max_steps=1, max_failures=1)

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.FAILED
    assert status.terminal_reason == 'max_failures'
    assert 'empty LLM responses' in (status.error or '')


async def test_step_exception_is_failed(monkeypatch):
    agent = make_agent(
        monkeypatch,
        [('print("work")', 'work')],
        AsyncMock(side_effect=RuntimeError('execution exploded')),
    )

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.FAILED
    assert status.success is False
    assert status.terminal_reason == 'step_error'
    assert 'execution exploded' in (status.error or '')


async def test_stop_during_execution_exception_wins_over_failure(monkeypatch):
    agent = make_agent(
        monkeypatch,
        [('print("work")', 'work')],
        AsyncMock(side_effect=RuntimeError('execution exploded')),
    )

    async def explode(code):
        agent.stop()
        raise RuntimeError('execution exploded')

    agent._execute_code = AsyncMock(side_effect=explode)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.CANCELLED
    assert agent.status.terminal_reason == 'cancelled'


async def test_stop_during_code_error_wins_over_max_failures(monkeypatch):
    agent = make_agent(monkeypatch, [('print("work")', 'work')], None, max_failures=1)

    async def fail_after_stop(code):
        agent.stop()
        return None, 'boom', None

    agent._execute_code = AsyncMock(side_effect=fail_after_stop)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.CANCELLED
    assert agent.status.terminal_reason == 'cancelled'


async def test_repeated_code_errors_hit_max_failures(monkeypatch):
    agent = make_agent(
        monkeypatch,
        [('print("work")', 'first'), ('print("work")', 'second')],
        AsyncMock(return_value=(None, 'boom', None)),
        max_steps=3,
        max_failures=2,
    )

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.FAILED
    assert agent.status.terminal_reason == 'max_failures'
    assert 'consecutive errors' in (agent.status.error or '')


async def test_stop_before_execution_cancels_the_run(monkeypatch):
    agent = make_agent(monkeypatch, [], None)

    async def respond():
        assert agent.stop() is True
        return 'await done("never")', 'stop'

    agent._get_code_from_llm = AsyncMock(side_effect=respond)

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.CANCELLED
    assert status.success is False
    assert status.terminal_reason == 'cancelled'
    agent._execute_code.assert_not_awaited()


async def test_stop_before_run_is_honored_without_starting_the_agent(monkeypatch):
    agent = make_agent(monkeypatch, [], None)
    create_namespace = MagicMock(return_value={'_all_code_blocks': {}})
    monkeypatch.setattr(service, 'create_namespace', create_namespace)

    assert agent.stop() is True
    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.CANCELLED
    assert status.success is False
    assert status.terminal_reason == 'cancelled'
    create_namespace.assert_not_called()
    agent._get_code_from_llm.assert_not_awaited()


async def test_stop_after_execution_cancels_even_with_done(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("x")', 'done')], None)

    async def execute(code):
        agent.namespace['_task_done'] = True
        agent.namespace['_task_done_by_action'] = True
        agent.namespace['_task_result'] = 'partial'
        assert agent.stop() is True
        return 'partial', None, None

    agent._execute_code = AsyncMock(side_effect=execute)

    await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.CANCELLED
    assert status.success is False
    assert status.result == 'partial'


async def test_stop_between_python_blocks_prevents_later_blocks(monkeypatch):
    agent = make_agent(monkeypatch, [('first\nsecond', 'plan')], None)
    agent.namespace['_all_code_blocks'] = {'python_0': 'first', 'python_1': 'second'}
    monkeypatch.setattr(service, 'create_namespace', MagicMock(return_value=agent.namespace))

    async def execute(code):
        assert agent.stop() is True
        return code, None, None

    agent._execute_code = AsyncMock(side_effect=execute)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.CANCELLED
    agent._execute_code.assert_awaited_once_with('first')


async def test_stop_during_validation_cancels_before_accepting_done(monkeypatch):
    agent = make_agent(
        monkeypatch,
        [('await done("final")', 'done')],
        None,
        max_validations=1,
    )
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final'))

    async def validate(**kwargs):
        assert agent.stop() is True
        assert agent.status.cancellation_requested is True
        return True, 'complete'

    monkeypatch.setattr(namespace_module, 'validate_task_completion', validate)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.CANCELLED
    assert agent.status.success is False


async def test_stop_during_validation_error_wins_over_failure(monkeypatch):
    agent = make_agent(
        monkeypatch,
        [('await done("final")', 'done')],
        None,
        max_validations=1,
    )
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final'))

    async def validate(**kwargs):
        agent.stop()
        raise RuntimeError('validator exploded')

    monkeypatch.setattr(namespace_module, 'validate_task_completion', validate)

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.CANCELLED
    assert agent.status.terminal_reason == 'cancelled'


async def test_stop_after_terminal_is_rejected(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("final")', 'done')], None)
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final'))

    await agent.run()

    assert agent.stop() is False
    assert agent.status.phase is CodeAgentPhase.SUCCEEDED


async def test_stop_during_post_done_screenshot_keeps_success(monkeypatch, caplog):
    agent = make_agent(monkeypatch, [('await done("final")', 'done')], None)
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final'))

    async def screenshot(step):
        agent.stop()
        return None, None

    agent._capture_screenshot = AsyncMock(side_effect=screenshot)

    with caplog.at_level(logging.ERROR, logger=service.__name__):
        await agent.run()

    assert agent.status.phase is CodeAgentPhase.SUCCEEDED
    assert agent.status.terminal_reason == 'done'
    assert 'already finished' not in caplog.text


async def test_task_cancellation_marks_the_run_cancelled_and_reraises(monkeypatch):
    agent = make_agent(monkeypatch, [], None)
    agent._get_code_from_llm = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.CANCELLED
    assert status.success is False
    assert status.terminal_reason == 'task_cancelled'


async def test_run_error_sets_failed_and_reraises(monkeypatch):
    agent = make_agent(monkeypatch, [], None)
    agent._run_inner = AsyncMock(side_effect=RuntimeError('setup failed'))

    with pytest.raises(RuntimeError, match='setup failed'):
        await agent.run()

    status = agent.status
    assert status.phase is CodeAgentPhase.FAILED
    assert status.success is False
    assert status.terminal_reason == 'run_error'
    assert 'setup failed' in (status.error or '')
    assert status.finished_at is not None


async def test_rejected_validation_clears_attachments(monkeypatch):
    calls = {'n': 0}

    async def execute(code):
        calls['n'] += 1
        agent = execute.agent
        record_done(agent, 'final', attachments=['stale.txt'] if calls['n'] == 1 else None)
        return 'final', None, None

    agent = make_agent(
        monkeypatch,
        [('await done("final")', 'done'), ('await done("final")', 'done')],
        None,
        max_validations=1,
    )
    execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=execute)
    monkeypatch.setattr(namespace_module, 'validate_task_completion', AsyncMock(return_value=(False, 'not yet')))

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.SUCCEEDED
    assert '_task_attachments' not in agent.namespace


async def test_rejected_validation_does_not_leave_completion_authorized(monkeypatch):
    calls = {'n': 0}

    async def execute(code):
        calls['n'] += 1
        if calls['n'] == 1:
            record_done(execute.agent, 'first', attachments=['stale.txt'])
        else:
            execute.agent.namespace['_task_done'] = True
            execute.agent.namespace['_task_done_by_action'] = True
            execute.agent.namespace['_task_result'] = 'forged after rejection'
        return 'output', None, None

    agent = make_agent(
        monkeypatch,
        [('await done("first")', 'done'), ('print("forged")', 'forged')],
        None,
        max_validations=1,
    )
    execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=execute)
    monkeypatch.setattr(namespace_module, 'validate_task_completion', AsyncMock(return_value=(False, 'not yet')))

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.FAILED
    assert agent.status.terminal_reason == 'done_not_invoked'


async def test_second_run_raises_instead_of_duplicating_messages(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("final")', 'done')], None)
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final'))

    await agent.run()
    with pytest.raises(CodeAgentStateError, match='already entered phase'):
        await agent.run()


async def test_close_failure_can_be_retried(monkeypatch):
    browser = MagicMock()
    browser.browser_profile.keep_alive = False
    browser.kill = AsyncMock(side_effect=RuntimeError('kill failed'))
    agent = CodeAgent(task='Do the thing', llm=MagicMock(), browser=browser)

    with pytest.raises(RuntimeError, match='kill failed'):
        await agent.close()

    browser.kill = AsyncMock()
    await agent.close()
    browser.kill.assert_awaited_once()


async def test_run_after_close_raises(monkeypatch):
    browser = MagicMock()
    browser.browser_profile.keep_alive = True
    agent = CodeAgent(task='Do the thing', llm=MagicMock(), browser=browser)
    await agent.close()

    with pytest.raises(CodeAgentStateError, match='closed'):
        await agent.run()


async def test_close_is_idempotent(monkeypatch):
    browser = MagicMock()
    browser.browser_profile.keep_alive = False
    browser.kill = AsyncMock()
    agent = CodeAgent(task='Do the thing', llm=MagicMock(), browser=browser)

    await agent.close()
    await agent.close()

    browser.kill.assert_awaited_once()
    assert agent.status.closed is True


async def test_run_survives_cleanup_failure_and_allows_retry(monkeypatch):
    agent = make_agent(monkeypatch, [('await done("final")', 'done')], None)
    agent.browser_session.browser_profile.keep_alive = False
    agent.browser_session.kill = AsyncMock(side_effect=[RuntimeError('kill failed'), None])
    done_execute.agent = agent
    agent._execute_code = AsyncMock(side_effect=done_execute('final'))

    await agent.run()

    assert agent.status.phase is CodeAgentPhase.SUCCEEDED
    assert agent.browser_session.kill.await_count == 1
    await agent.close()
    assert agent.browser_session.kill.await_count == 2


async def test_concurrent_close_calls_join_one_kill(monkeypatch):
    browser = MagicMock()
    browser.browser_profile.keep_alive = False
    started = asyncio.Event()
    release = asyncio.Event()

    async def kill():
        started.set()
        await release.wait()

    browser.kill = AsyncMock(side_effect=kill)
    agent = CodeAgent(task='Do the thing', llm=MagicMock(), browser=browser)

    first = asyncio.create_task(agent.close())
    await started.wait()
    second = asyncio.create_task(agent.close())
    await asyncio.sleep(0)
    assert browser.kill.await_count == 1
    release.set()
    await asyncio.gather(first, second)

    browser.kill.assert_awaited_once()
