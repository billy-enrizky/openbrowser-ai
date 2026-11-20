"""Behavior tests for the explicit Browser Agent run state (M18)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.agent.graph import AgentGraphBuilder
from openbrowser.agent.service import Agent
from openbrowser.agent.views import (
	AgentHistoryList,
	AgentRunInterrupted,
	AgentRunPhase,
	AgentRunStateError,
	AgentState,
	AgentStepInfo,
)
from openbrowser.models import ActionResult


def make_graph_agent(*, max_failures=2, final_response_after_failure=True):
	agent = SimpleNamespace()
	agent.logger = MagicMock()
	agent.has_downloads_path = False
	agent.include_recent_events = False
	agent.sensitive_data = None
	agent.available_file_paths = None
	agent.settings = SimpleNamespace(
		max_failures=max_failures,
		final_response_after_failure=final_response_after_failure,
		use_vision=True,
		llm_timeout=60,
	)
	agent.state = AgentState()
	agent.history = AgentHistoryList(history=[])
	agent._run_phase = AgentRunPhase.RUNNING
	agent._run_max_steps = 5
	agent._run_terminal_reason = None
	agent._run_success = None
	agent._run_result = None
	agent._run_error = None
	agent._run_started_at = None
	agent._run_finished_at = None
	agent._run_step_base = 0
	agent._run_history_start = 0
	agent._pending_stop_reason = None
	agent._external_pause_event = asyncio.Event()
	agent._external_pause_event.set()
	agent.register_should_stop_callback = None
	agent.register_external_agent_status_raise_error_callback = None
	agent._current_step_number = Agent._current_step_number.__get__(agent, SimpleNamespace)
	agent._finish_run = Agent._finish_run.__get__(agent, SimpleNamespace)
	agent._done_terminal_phase = Agent._done_terminal_phase.__get__(agent, SimpleNamespace)
	agent._run_final_result = Agent._run_final_result.__get__(agent, SimpleNamespace)
	agent._current_run_last_result = Agent._current_run_last_result.__get__(agent, SimpleNamespace)
	agent._run_has_done = Agent._run_has_done.__get__(agent, SimpleNamespace)
	agent._neutralize_unaccepted_done = Agent._neutralize_unaccepted_done.__get__(agent, SimpleNamespace)
	agent._publish_completion = Agent._publish_completion.__get__(agent, SimpleNamespace)
	agent._check_stop_or_pause = AsyncMock()
	agent.pause = Agent.pause.__get__(agent, SimpleNamespace)
	agent.resume = Agent.resume.__get__(agent, SimpleNamespace)
	agent._update_action_models_for_page = AsyncMock()
	agent._force_done_after_last_step = AsyncMock()
	agent._force_done_after_failure = AsyncMock()
	agent._get_model_output_with_retry = AsyncMock()
	agent._handle_post_llm_processing = AsyncMock()
	agent._make_history_item = AsyncMock()
	agent.multi_act = AsyncMock()
	agent.save_file_system_state = MagicMock()
	agent.log_completion = AsyncMock()
	agent.register_done_callback = None
	agent.browser_session = MagicMock()
	agent.browser_session.get_browser_state_summary = AsyncMock()
	agent._message_manager = MagicMock()
	agent._message_manager.get_messages.return_value = []
	agent._message_manager.last_state_message_text = 'state'
	agent.tools = MagicMock()
	agent.tools.registry.get_prompt_description.return_value = 'actions'
	return agent


def append_history(agent, *, done=None, success=None, content='result'):
	result = ActionResult(is_done=done, success=success, extracted_content=content)
	agent.history.history.append(SimpleNamespace(result=[result], metadata=None))
	return result


def make_live_agent():
	agent = object.__new__(Agent)
	agent.state = AgentState()
	agent.history = AgentHistoryList(history=[])
	agent.task_id = 'task'
	agent.browser_session = MagicMock()
	agent.browser_session.id = 'browser'
	agent.browser_session.agent_focus = None
	agent._external_pause_event = asyncio.Event()
	agent._external_pause_event.set()
	agent.register_should_stop_callback = None
	agent.register_external_agent_status_raise_error_callback = None
	agent._run_phase = AgentRunPhase.IDLE
	agent._run_max_steps = 0
	agent._run_terminal_reason = None
	agent._run_success = None
	agent._run_result = None
	agent._run_error = None
	agent._run_started_at = None
	agent._run_finished_at = None
	agent._run_step_base = 0
	agent._run_history_start = 0
	agent._run_active = False
	agent._step_sequence_initialized = False
	agent._pending_stop_reason = None
	agent._force_exit_requested = False
	agent.token_cost_service = SimpleNamespace(usage_history=[])
	agent._force_exit_telemetry_logged = False
	agent.use_signal_handler = False
	agent.register_should_stop_callback = None
	agent.register_external_agent_status_raise_error_callback = None
	return agent


def make_full_live_agent():
	agent = make_live_agent()
	agent.use_signal_handler = False
	agent.session_id = 'session'
	agent.browser_session.cdp_url = None
	agent.browser_session.start = AsyncMock()
	agent._log_agent_run = AsyncMock()
	agent._log_first_step_startup = MagicMock()
	agent._log_agent_event = MagicMock()
	agent._log_final_outcome_messages = MagicMock()
	agent.token_cost_service = SimpleNamespace(
		usage_history=[],
		get_usage_summary=AsyncMock(return_value=None),
		log_usage_summary=AsyncMock(),
	)
	agent.eventbus = SimpleNamespace(stop=AsyncMock())
	agent.close = AsyncMock()
	agent.settings = SimpleNamespace(generate_gif=False)
	agent.output_model_schema = None
	agent.log_completion = AsyncMock()
	agent.register_done_callback = None
	return agent


def test_run_state_snapshot_and_control_states():
	agent = make_live_agent()
	assert agent.status.phase is AgentRunPhase.IDLE
	assert agent.status.step == 0
	assert agent.pause() is False

	agent._begin_run(5)
	assert agent.status.phase is AgentRunPhase.STARTING
	assert agent.status.step == 0
	assert agent.status.max_steps == 5
	assert agent.pause() is True
	assert agent.status.phase is AgentRunPhase.PAUSED
	assert agent.status.pause_requested is True
	assert agent.resume() is True
	assert agent.status.phase is AgentRunPhase.RUNNING
	assert agent.stop() is True
	assert agent.status.phase is AgentRunPhase.STOPPING
	assert agent.status.stop_requested is True


def test_force_exit_uses_the_stop_transition():
	agent = make_live_agent()
	agent._run_phase = AgentRunPhase.RUNNING
	agent._log_agent_event = MagicMock()
	agent.telemetry = MagicMock()

	agent._on_force_exit()
	agent._on_force_exit()

	assert agent.state.stopped is True
	assert agent.status.phase is AgentRunPhase.STOPPING
	assert agent._pending_stop_reason == 'sigint'
	assert agent._force_exit_requested is True
	assert agent._force_exit_telemetry_logged is False
	agent._log_agent_event.assert_not_called()
	agent.telemetry.flush.assert_not_called()


def test_a_later_stop_request_does_not_overwrite_the_accepted_reason():
	agent = make_live_agent()
	agent._run_phase = AgentRunPhase.RUNNING

	agent._on_force_exit()
	assert agent.stop() is True

	assert agent._pending_stop_reason == 'sigint'


def test_force_exit_after_a_terminal_phase_does_nothing():
	agent = make_live_agent()
	agent._run_phase = AgentRunPhase.SUCCEEDED

	agent._on_force_exit()

	assert agent._force_exit_requested is False
	assert agent._pending_stop_reason is None
	assert agent.state.stopped is False


async def test_force_exit_flushes_telemetry_once_after_the_terminal_event():
	agent = make_full_live_agent()
	agent.telemetry = MagicMock()
	agent._on_force_exit()

	await agent.run(max_steps=5)

	agent._log_agent_event.assert_called_once()
	agent.telemetry.flush.assert_called_once()


async def test_finalizer_logs_telemetry_before_a_cancelled_usage_summary():
	agent = make_full_live_agent()
	agent.stop()
	agent.token_cost_service.log_usage_summary = AsyncMock(side_effect=asyncio.CancelledError())

	with pytest.raises(asyncio.CancelledError):
		await agent.run(max_steps=5)

	agent._log_agent_event.assert_called_once()
	assert agent.status.phase is AgentRunPhase.STOPPED


async def test_stop_checkpoint_raises_the_accepted_reason():
	agent = make_live_agent()
	agent.state.stopped = True
	agent._pending_stop_reason = 'sigint'

	with pytest.raises(AgentRunInterrupted) as exc_info:
		await agent._check_stop_or_pause()

	assert exc_info.value.reason == 'sigint'


async def test_stop_before_run_is_retained_without_starting_a_browser():
	agent = make_full_live_agent()

	assert agent.stop() is True
	history = await agent.run(max_steps=5)

	assert history is agent.history
	agent.browser_session.start.assert_not_awaited()
	agent.eventbus.stop.assert_awaited_once()
	agent.close.assert_awaited_once()
	assert agent.status.phase is AgentRunPhase.STOPPED
	assert agent.status.terminal_reason == 'stopped_before_start'
	assert agent.status.success is False


async def test_initial_done_success_finishes_without_running_the_graph():
	agent = make_full_live_agent()
	callback = AsyncMock()
	agent.register_done_callback = callback

	async def initial_done():
		append_history(agent, done=True, success=True, content='startup done')

	agent._execute_initial_actions = AsyncMock(side_effect=initial_done)

	with patch('openbrowser.agent.graph.create_agent_graph') as graph_factory:
		await agent.run(max_steps=5)

	graph_factory.assert_not_called()
	assert agent.status.phase is AgentRunPhase.SUCCEEDED
	assert agent.status.terminal_reason == 'done'
	assert agent.status.result == 'startup done'
	agent.log_completion.assert_awaited_once()
	callback.assert_awaited_once_with(agent.history)


async def test_initial_failed_done_finishes_without_completion():
	agent = make_full_live_agent()
	callback = AsyncMock()
	agent.register_done_callback = callback

	async def initial_done():
		append_history(agent, done=True, success=False, content='startup failed')

	agent._execute_initial_actions = AsyncMock(side_effect=initial_done)

	with patch('openbrowser.agent.graph.create_agent_graph') as graph_factory:
		await agent.run(max_steps=5)

	graph_factory.assert_not_called()
	assert agent.status.phase is AgentRunPhase.FAILED
	assert agent.status.terminal_reason == 'done_failed'
	agent.log_completion.assert_not_awaited()
	callback.assert_not_awaited()


async def test_initial_actions_stop_keeps_the_accepted_reason():
	agent = make_full_live_agent()

	async def initial_stop():
		append_history(agent, content='partial')
		agent.state.stopped = True
		agent._pending_stop_reason = 'external_stop'

	agent._execute_initial_actions = AsyncMock(side_effect=initial_stop)

	with patch('openbrowser.agent.graph.create_agent_graph') as graph_factory:
		await agent.run(max_steps=5)

	graph_factory.assert_not_called()
	assert agent.status.phase is AgentRunPhase.STOPPED
	assert agent.status.terminal_reason == 'external_stop'
	agent.log_completion.assert_not_awaited()


async def test_initial_done_waits_for_an_accepted_pause():
	agent = make_full_live_agent()
	reached_pause = asyncio.Event()

	async def initial_done():
		append_history(agent, done=True, success=True, content='startup done')
		agent.state.paused = True
		agent._external_pause_event.clear()
		reached_pause.set()

	agent._execute_initial_actions = AsyncMock(side_effect=initial_done)

	with patch('openbrowser.agent.graph.create_agent_graph') as graph_factory:
		run_task = asyncio.create_task(agent.run(max_steps=5))
		await asyncio.wait_for(reached_pause.wait(), timeout=1)
		await asyncio.sleep(0)
		assert not run_task.done()

		assert agent.resume() is True
		await asyncio.wait_for(run_task, timeout=1)

	graph_factory.assert_not_called()
	assert agent.status.phase is AgentRunPhase.SUCCEEDED
	assert agent.status.terminal_reason == 'done'
	agent.log_completion.assert_awaited_once()


async def test_initial_done_honors_an_external_stop_callback():
	agent = make_full_live_agent()

	async def initial_done():
		append_history(agent, done=True, success=True, content='startup done')

	agent._execute_initial_actions = AsyncMock(side_effect=initial_done)
	agent.register_should_stop_callback = AsyncMock(return_value=True)

	with patch('openbrowser.agent.graph.create_agent_graph') as graph_factory:
		await agent.run(max_steps=5)

	graph_factory.assert_not_called()
	assert agent.status.phase is AgentRunPhase.STOPPED
	assert agent.status.terminal_reason == 'external_stop'
	assert agent.history.history[-1].result[-1].is_done is False
	agent.log_completion.assert_not_awaited()


async def test_add_new_task_rejects_an_in_flight_run():
	agent = make_live_agent()
	started = asyncio.Event()
	release = asyncio.Event()

	async def wait_for_release(max_steps, on_step_start=None, on_step_end=None):
		started.set()
		await release.wait()
		return agent.history

	agent._run_with_langgraph = wait_for_release
	run_task = asyncio.create_task(agent.run(max_steps=5))
	await asyncio.wait_for(started.wait(), timeout=1)

	with pytest.raises(AgentRunStateError, match='run is active'):
		agent.add_new_task('Follow up')

	release.set()
	await asyncio.wait_for(run_task, timeout=1)


async def test_stopped_follow_up_does_not_report_the_previous_task_result():
	agent = make_full_live_agent()
	agent._message_manager = MagicMock()
	agent.id = 'agent-id'
	append_history(agent, done=True, success=True, content='previous result')
	agent.state.n_steps = 1
	agent.add_new_task('Follow up')
	agent.stop()

	await agent.run(max_steps=5)

	assert agent.status.phase is AgentRunPhase.STOPPED
	assert agent.status.result is None


async def test_check_stop_or_pause_waits_for_resume_and_stop_wins(monkeypatch):
	agent = make_live_agent()
	agent._begin_run(5)
	agent._run_phase = AgentRunPhase.RUNNING
	assert agent.pause() is True
	check = asyncio.create_task(agent._check_stop_or_pause())
	await asyncio.sleep(0)
	assert not check.done()
	monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
	assert agent.resume() is True
	await asyncio.wait_for(check, timeout=1)

	assert agent.pause() is True
	check = asyncio.create_task(agent._check_stop_or_pause())
	await asyncio.sleep(0)
	assert agent.stop() is True
	assert agent.resume() is False
	with pytest.raises(AgentRunInterrupted, match='stopped'):
		await asyncio.wait_for(check, timeout=1)


async def test_stop_during_status_callback_is_observed_before_execution():
	agent = make_live_agent()
	agent._run_phase = AgentRunPhase.RUNNING
	started = asyncio.Event()
	release = asyncio.Event()

	async def status_callback():
		started.set()
		await release.wait()
		return False

	agent.register_external_agent_status_raise_error_callback = status_callback
	check = asyncio.create_task(agent._check_stop_or_pause())
	await asyncio.wait_for(started.wait(), timeout=1)
	assert agent.stop() is True
	release.set()

	with pytest.raises(AgentRunInterrupted, match='stopped'):
		await asyncio.wait_for(check, timeout=1)


async def test_second_run_requires_add_new_task():
	agent = make_live_agent()
	agent._begin_run(5)
	agent._finish_run(AgentRunPhase.FAILED, 'done_failed', False)
	with pytest.raises(AgentRunStateError, match='add_new_task'):
		agent._begin_run(5)


def test_success_requires_an_accepted_done_history_item():
	agent = make_live_agent()
	agent._begin_run(5)

	with pytest.raises(AgentRunStateError, match='accepted successful done'):
		agent._finish_run(AgentRunPhase.SUCCEEDED, 'done', True)

	append_history(agent, done=True, success=True, content='done')
	agent._finish_run(AgentRunPhase.SUCCEEDED, 'done', True)
	assert agent.status.phase is AgentRunPhase.SUCCEEDED


def test_stopped_terminal_neutralizes_current_success_history():
	agent = make_live_agent()
	agent._begin_run(5)
	append_history(agent, done=True, success=True, content='startup done')

	agent._finish_run(AgentRunPhase.STOPPED, 'stopped', False)

	assert agent.status.phase is AgentRunPhase.STOPPED
	assert agent.history.is_done() is False
	assert agent.history.is_successful() is None


async def test_run_error_and_task_cancellation_are_terminal():
	agent = make_live_agent()
	agent._run_with_langgraph = AsyncMock(side_effect=RuntimeError('setup failed'))
	with pytest.raises(RuntimeError, match='setup failed'):
		await agent.run(max_steps=5)
	assert agent.status.phase is AgentRunPhase.FAILED
	assert agent.status.terminal_reason == 'run_error'
	assert agent.status.success is False

	agent = make_live_agent()
	agent._run_with_langgraph = AsyncMock(side_effect=asyncio.CancelledError())
	with pytest.raises(asyncio.CancelledError):
		await agent.run(max_steps=5)
	assert agent.status.phase is AgentRunPhase.CANCELLED
	assert agent.status.terminal_reason == 'task_cancelled'
	assert agent.status.success is False
	assert agent.status.stop_requested is False


async def test_cancellation_after_history_write_neutralizes_success():
	agent = make_live_agent()

	async def cancel_after_history(max_steps, on_step_start=None, on_step_end=None):
		append_history(agent, done=True, success=True, content='unaccepted')
		raise asyncio.CancelledError()

	agent._run_with_langgraph = cancel_after_history
	with pytest.raises(asyncio.CancelledError):
		await agent.run(max_steps=5)

	assert agent.status.phase is AgentRunPhase.CANCELLED
	assert agent.history.is_done() is False
	assert agent.history.is_successful() is None


async def test_stop_wins_over_a_run_error():
	agent = make_live_agent()

	async def stop_then_fail(max_steps, on_step_start=None, on_step_end=None):
		agent.stop()
		raise RuntimeError('setup failed after stop')

	agent._run_with_langgraph = stop_then_fail
	with pytest.raises(RuntimeError, match='setup failed after stop'):
		await agent.run(max_steps=5)

	assert agent.status.phase is AgentRunPhase.STOPPED
	assert agent.status.terminal_reason == 'stopped_during_error'
	assert agent.status.stop_requested is True


def test_add_new_task_resets_run_terminal_state():
	agent = make_live_agent()
	agent.id = 'agent-id'
	agent._message_manager = MagicMock()
	agent.eventbus = MagicMock()
	agent._begin_run(5)
	agent._finish_run(AgentRunPhase.FAILED, 'max_failures', False)
	agent.state.consecutive_failures = 3

	agent.add_new_task('Follow up')

	assert agent.status.phase is AgentRunPhase.IDLE
	assert agent.state.consecutive_failures == 0
	assert agent.state.stopped is False
	assert agent.state.paused is False


def test_follow_up_run_continues_the_public_step_counter():
	agent = make_live_agent()
	append_history(agent, done=False)
	agent.state.n_steps = 2

	agent._begin_run(5)

	assert agent.status.step == 2
	assert agent._run_step_base == 2
	assert agent._current_step_number(0) == 3


def test_follow_up_continues_after_a_historyless_run():
	agent = make_live_agent()
	agent._message_manager = MagicMock()
	agent.eventbus = MagicMock()
	agent.id = 'agent-id'
	agent._begin_run(5)
	agent.state.n_steps = 1
	agent._finish_run(AgentRunPhase.FAILED, 'run_error', False)

	agent.add_new_task('Follow up')
	agent._begin_run(5)

	assert agent._current_step_number(0) == 2


def test_restored_step_sequence_does_not_reset_without_history():
	agent = make_live_agent()
	agent._step_sequence_initialized = True
	agent.state.n_steps = 12

	agent._begin_run(5)

	assert agent._run_step_base == 12
	assert agent._current_step_number(0) == 13


def test_manual_step_advances_the_next_graph_run():
	agent = make_live_agent()
	agent.state.step_sequence_started = True
	agent.state.step_sequence_is_next = True
	agent.state.n_steps = 2

	agent._begin_run(5)

	assert agent._run_step_base == 1
	assert agent._current_step_number(0) == 2


async def test_step_counter_and_failure_counter_have_one_authority():
	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(error='boom')
	agent.multi_act.return_value = [result]

	async def add_history(*args, **kwargs):
		metadata = args[3] if len(args) > 3 else kwargs.get('metadata')
		agent.history.history.append(SimpleNamespace(result=[result], metadata=metadata))

	agent._make_history_item = AsyncMock(side_effect=add_history)
	agent.state.consecutive_failures = 1

	await builder._step_node({
		'step_number': 0,
		'max_steps': 5,
		'is_done': False,
	})

	assert agent.state.consecutive_failures == 2
	assert agent.state.n_steps == 1
	assert agent.history.history[-1].metadata.step_number == 1


async def test_bookkeeping_failure_on_a_clean_step_marks_the_next_prompt_failed():
	agent = make_graph_agent(max_failures=5, final_response_after_failure=False)
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	agent.multi_act.return_value = [ActionResult(extracted_content='clicked')]

	async def add_history(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))

	agent._make_history_item = AsyncMock(side_effect=add_history)
	agent.save_file_system_state.side_effect = RuntimeError('disk full')

	await builder._step_node({'step_number': 0, 'max_steps': 5, 'is_done': False})

	assert agent.state.consecutive_failures == 1
	assert agent.state.last_result[-1].error == 'File system state save failed: disk full'


async def test_action_error_and_bookkeeping_failure_increment_once():
	agent = make_graph_agent(max_failures=5, final_response_after_failure=False)
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(error='boom')
	agent.multi_act.return_value = [result]

	async def add_history(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))

	agent._make_history_item = AsyncMock(side_effect=add_history)
	agent.save_file_system_state.side_effect = RuntimeError('disk full')

	await builder._step_node({'step_number': 0, 'max_steps': 5, 'is_done': False})

	assert agent.state.consecutive_failures == 1


async def test_manual_step_defers_accepted_success_result_to_the_observer():
	agent = make_live_agent()
	agent._check_and_update_downloads = AsyncMock()
	agent.state.last_result = [
		ActionResult(is_done=True, success=True, extracted_content='accepted answer')
	]
	logger = agent.logger

	with patch.object(logger, 'info') as log_info:
		await agent._post_process()

	log_info.assert_not_called()


async def test_manual_follow_up_does_not_reuse_the_previous_done():
	agent = make_live_agent()
	agent.log_completion = AsyncMock()
	append_history(agent, done=True, success=True, content='previous result')
	agent._run_history_start = len(agent.history.history)
	agent.step = AsyncMock()

	is_done, is_valid = await agent.take_step(AgentStepInfo(step_number=5, max_steps=10))
	step_done = await agent._execute_step(0, 10, AgentStepInfo(step_number=0, max_steps=10))

	assert (is_done, is_valid) == (False, False)
	assert step_done is False
	agent.log_completion.assert_not_awaited()


async def test_manual_multi_action_failure_increments_once():
	agent = make_live_agent()
	agent._check_and_update_downloads = AsyncMock()
	agent.state.consecutive_failures = 1
	agent.state.last_result = [
		ActionResult(extracted_content='ok'),
		ActionResult(error='boom'),
	]

	await agent._post_process()

	assert agent.state.consecutive_failures == 2


async def test_manual_pause_checkpoint_does_not_enter_a_run_phase():
	agent = make_live_agent()
	agent._check_stop_or_pause = Agent._check_stop_or_pause.__get__(agent, SimpleNamespace)
	agent.resume = Agent.resume.__get__(agent, SimpleNamespace)
	agent.state.paused = True
	agent._external_pause_event.clear()

	checkpoint = asyncio.create_task(agent._check_stop_or_pause())
	await asyncio.sleep(0)
	assert not checkpoint.done()

	assert agent.resume() is True
	await asyncio.wait_for(checkpoint, timeout=1)

	assert agent._run_phase is AgentRunPhase.IDLE


async def test_pause_after_done_history_waits_before_routing(monkeypatch):
	agent = make_graph_agent()
	agent._check_stop_or_pause = Agent._check_stop_or_pause.__get__(agent, SimpleNamespace)
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(is_done=True, success=True, extracted_content='done')
	agent.multi_act.return_value = [result]
	paused = asyncio.Event()

	async def add_history_then_pause(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))
		agent.state.paused = True
		agent._external_pause_event.clear()
		paused.set()

	agent._make_history_item = AsyncMock(side_effect=add_history_then_pause)
	monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
	step_task = asyncio.create_task(
		builder._step_node({'step_number': 0, 'max_steps': 5, 'is_done': False})
	)
	await asyncio.wait_for(paused.wait(), timeout=1)
	await asyncio.sleep(0)
	assert not step_task.done()

	assert agent.resume() is True
	result_state = await asyncio.wait_for(step_task, timeout=1)

	assert result_state['is_done'] is True
	assert agent.history.is_successful() is True


async def test_real_graph_reaches_success_once_and_keeps_global_step_numbers():
	agent = make_graph_agent()
	callback = AsyncMock()
	agent.register_done_callback = callback
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	agent.multi_act.side_effect = [
		[ActionResult(extracted_content='working')],
		[ActionResult(is_done=True, success=True, extracted_content='done')],
	]

	async def add_history(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))

	agent._make_history_item = AsyncMock(side_effect=add_history)

	await builder.run(max_steps=5)

	assert agent._run_phase is AgentRunPhase.SUCCEEDED
	assert agent._run_terminal_reason == 'done'
	assert agent._run_success is True
	assert agent.state.n_steps == 2
	assert [item.metadata.step_number for item in agent.history.history] == [1, 2]
	assert agent.history.is_successful() is True
	agent.log_completion.assert_awaited_once()
	callback.assert_awaited_once_with(agent.history)


async def test_real_graph_failed_done_is_not_completion():
	agent = make_graph_agent()
	callback = AsyncMock()
	agent.register_done_callback = callback
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(is_done=True, success=False, extracted_content='failed')
	agent.multi_act.return_value = [result]

	async def add_history(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))

	agent._make_history_item = AsyncMock(side_effect=add_history)

	await builder.run(max_steps=5)

	assert agent._run_phase is AgentRunPhase.FAILED
	assert agent._run_terminal_reason == 'done_failed'
	assert agent._run_success is False
	agent.log_completion.assert_not_awaited()
	callback.assert_not_awaited()


async def test_real_graph_stop_after_history_invalidates_done():
	agent = make_graph_agent()
	callback = AsyncMock()
	agent.register_done_callback = callback
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(is_done=True, success=True, extracted_content='unaccepted')
	agent.multi_act.return_value = [result]

	async def add_history_then_stop(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))
		agent.state.stopped = True

	agent._make_history_item = AsyncMock(side_effect=add_history_then_stop)

	await builder.run(max_steps=5)

	assert agent._run_phase is AgentRunPhase.STOPPED
	assert agent._run_success is False
	assert agent.history.is_done() is False
	assert agent.history.is_successful() is None
	agent.save_file_system_state.assert_called_once()
	agent.log_completion.assert_not_awaited()
	callback.assert_not_awaited()


async def test_real_graph_retains_accepted_done_when_bookkeeping_fails():
	agent = make_graph_agent()
	callback = AsyncMock()
	agent.register_done_callback = callback
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(is_done=True, success=True, extracted_content='done')
	agent.multi_act.return_value = [result]

	async def add_history(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))

	agent._make_history_item = AsyncMock(side_effect=add_history)
	agent.save_file_system_state.side_effect = RuntimeError('disk full')

	await builder.run(max_steps=5)

	assert agent._run_phase is AgentRunPhase.SUCCEEDED
	assert agent.history.is_successful() is True
	agent.log_completion.assert_awaited_once()
	callback.assert_awaited_once_with(agent.history)


def test_terminal_success_and_failure_classification():
	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	append_history(agent, done=True, success=True, content='done')
	assert builder._should_continue({'is_done': True, 'step_number': 1, 'max_steps': 5}) == 'done'
	assert agent._run_phase is AgentRunPhase.SUCCEEDED
	assert agent._run_success is True

	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	append_history(agent, done=True, success=False, content='failed')
	assert builder._should_continue({'is_done': True, 'step_number': 1, 'max_steps': 5}) == 'error'
	assert agent._run_phase is AgentRunPhase.FAILED
	assert agent._run_terminal_reason == 'done_failed'


def test_stop_after_done_before_routing_neutralizes_the_done():
	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	append_history(agent, done=True, success=True, content='done')
	agent.state.stopped = True

	assert builder._should_continue({'is_done': True, 'step_number': 1, 'max_steps': 5}) == 'done'

	assert agent._run_phase is AgentRunPhase.STOPPED
	assert agent._run_success is False
	assert agent.history.is_done() is False
	assert agent.history.is_successful() is None


def test_max_failures_and_max_steps_are_distinct_terminals():
	agent = make_graph_agent(max_failures=2, final_response_after_failure=False)
	builder = AgentGraphBuilder(agent)
	agent.state.consecutive_failures = 2
	assert builder._should_continue({'is_done': False, 'step_number': 1, 'max_steps': 5}) == 'error'
	assert agent._run_phase is AgentRunPhase.FAILED
	assert agent._run_terminal_reason == 'max_failures'

	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	assert builder._should_continue({'is_done': False, 'step_number': 5, 'max_steps': 5}) == 'done'
	assert agent._run_phase is AgentRunPhase.MAX_STEPS
	assert agent._run_terminal_reason == 'max_steps'


async def test_stop_during_history_neutralizes_unaccepted_done():
	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(is_done=True, success=True, extracted_content='done')
	agent.multi_act.return_value = [result]

	async def write_then_stop(*args, **kwargs):
		agent.history.history.append(SimpleNamespace(result=[result], metadata=None))
		agent.state.stopped = True

	agent._make_history_item = AsyncMock(side_effect=write_then_stop)

	step_result = await builder._step_node({
		'step_number': 0,
		'max_steps': 5,
		'is_done': False,
	})

	assert step_result['is_done'] is False
	assert agent._run_terminal_reason == 'stopped'
	assert agent._run_phase is AgentRunPhase.STOPPED
	assert agent.history.history[-1].result[-1].is_done is False
	assert agent.history.history[-1].result[-1].success is False


def test_stop_routing_does_not_mutate_previous_task_history():
	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	append_history(agent, done=True, success=True, content='previous')
	agent._run_history_start = 1
	agent.state.stopped = True

	assert builder._should_continue({'is_done': True, 'step_number': 1, 'max_steps': 5}) == 'done'

	assert agent._run_phase is AgentRunPhase.STOPPED
	assert agent.history.is_done() is True
	assert agent.history.is_successful() is True


async def test_stop_after_actions_retains_partial_history():
	agent = make_graph_agent()
	builder = AgentGraphBuilder(agent)
	browser_state = MagicMock()
	browser_state.url = 'https://example.com'
	agent.browser_session.get_browser_state_summary.return_value = browser_state
	model_output = MagicMock()
	model_output.action = [MagicMock(), MagicMock()]
	agent._get_model_output_with_retry.return_value = model_output
	result = ActionResult(extracted_content='clicked')

	async def stop_after_action(actions):
		agent.state.stopped = True
		return [result]

	agent.multi_act = AsyncMock(side_effect=stop_after_action)

	async def add_history(model_output, browser_state, results, metadata=None, state_message=None):
		agent.history.history.append(SimpleNamespace(result=results, metadata=metadata))

	agent._make_history_item = AsyncMock(side_effect=add_history)

	await builder._step_node({'step_number': 0, 'max_steps': 5, 'is_done': False})

	assert agent._run_phase is AgentRunPhase.STOPPED
	assert len(agent.history.history) == 1
	assert agent.history.history[0].result == [result]
	assert agent._run_final_result() == 'clicked'


async def test_multi_act_returns_completed_actions_when_stop_arrives_between_actions():
	agent = make_live_agent()
	agent.browser_session.browser_profile = SimpleNamespace(wait_between_actions=0)
	agent.file_system = None
	agent.settings = SimpleNamespace(page_extraction_llm=None)
	agent.sensitive_data = None
	agent.available_file_paths = []
	agent.tools = MagicMock()
	first = ActionResult(extracted_content='clicked')
	agent.tools.act = AsyncMock(return_value=first)
	agent._check_stop_or_pause = AsyncMock(
		side_effect=[None, AgentRunInterrupted('stopped')]
	)
	actions = [MagicMock(), MagicMock()]
	actions[0].model_dump.return_value = {}
	actions[1].model_dump.return_value = {}

	results = await agent.multi_act(actions)

	assert results == [first]
	assert agent.state.stopped is True
	assert agent._pending_stop_reason == 'stopped'
	agent.tools.act.assert_awaited_once()


async def test_multi_act_returns_completed_actions_for_external_interrupt():
	agent = make_live_agent()
	agent.browser_session.browser_profile = SimpleNamespace(wait_between_actions=0)
	agent.file_system = None
	agent.settings = SimpleNamespace(page_extraction_llm=None)
	agent.sensitive_data = None
	agent.available_file_paths = []
	agent.tools = MagicMock()
	first = ActionResult(extracted_content='clicked')
	agent.tools.act = AsyncMock(return_value=first)
	agent._check_stop_or_pause = AsyncMock(
		side_effect=[None, AgentRunInterrupted('external_interrupt')]
	)
	actions = [MagicMock(), MagicMock()]
	actions[0].model_dump.return_value = {}
	actions[1].model_dump.return_value = {}

	results = await agent.multi_act(actions)

	assert results == [first]
	assert agent.state.stopped is True
	assert agent._pending_stop_reason == 'external_interrupt'


async def test_multi_act_preserves_a_stop_reason_recorded_before_the_checkpoint():
	agent = make_live_agent()
	agent.browser_session.browser_profile = SimpleNamespace(wait_between_actions=0)
	agent.file_system = None
	agent.settings = SimpleNamespace(page_extraction_llm=None)
	agent.sensitive_data = None
	agent.available_file_paths = []
	agent.tools = MagicMock()
	first = ActionResult(extracted_content='clicked')
	agent.tools.act = AsyncMock(return_value=first)
	agent._check_stop_or_pause = AsyncMock(
		side_effect=[None, AgentRunInterrupted('sigint')]
	)
	agent._pending_stop_reason = 'sigint'
	actions = [MagicMock(), MagicMock()]
	actions[0].model_dump.return_value = {}
	actions[1].model_dump.return_value = {}

	results = await agent.multi_act(actions)

	assert results == [first]
	assert agent._pending_stop_reason == 'sigint'


async def test_multi_act_does_not_overwrite_an_accepted_reason_with_a_raw_interrupt():
	agent = make_live_agent()
	agent.browser_session.browser_profile = SimpleNamespace(wait_between_actions=0)
	agent.file_system = None
	agent.settings = SimpleNamespace(page_extraction_llm=None)
	agent.sensitive_data = None
	agent.available_file_paths = []
	agent.tools = MagicMock()
	first = ActionResult(extracted_content='clicked')
	agent.tools.act = AsyncMock(return_value=first)
	agent._check_stop_or_pause = AsyncMock(side_effect=[None, InterruptedError()])
	agent._pending_stop_reason = 'sigint'
	actions = [MagicMock(), MagicMock()]
	actions[0].model_dump.return_value = {}
	actions[1].model_dump.return_value = {}

	results = await agent.multi_act(actions)

	assert results == [first]
	assert agent._pending_stop_reason == 'sigint'


async def test_log_completion_reports_only_an_accepted_final_result():
	agent = make_live_agent()
	append_history(agent, done=True, success=True, content='accepted answer')
	logger = agent.logger

	with patch.object(logger, 'info') as log_info:
		await agent.log_completion()

	messages = [str(call.args[0]) for call in log_info.call_args_list if call.args]
	assert any('Task completed successfully' in message for message in messages)
	assert any('Final Result' in message and 'accepted answer' in message for message in messages)


async def test_log_completion_stays_quiet_without_accepted_success():
	agent = make_live_agent()
	append_history(agent, done=True, success=False, content='failed answer')
	logger = agent.logger

	with patch.object(logger, 'info') as log_info:
		await agent.log_completion()

	log_info.assert_not_called()


async def test_stopped_and_max_step_runs_do_not_call_done_callback():
	for phase, reason, success in [
		(AgentRunPhase.STOPPED, 'stopped', False),
		(AgentRunPhase.MAX_STEPS, 'max_steps', False),
	]:
		agent = make_graph_agent()
		callback = AsyncMock()
		agent.register_done_callback = callback
		agent._run_phase = phase
		agent._run_terminal_reason = reason
		agent._run_success = success
		builder = AgentGraphBuilder(agent)
		builder.graph = MagicMock()
		builder.graph.ainvoke = AsyncMock()

		await builder.run(max_steps=5)

		callback.assert_not_awaited()
		agent.log_completion.assert_not_awaited()


async def test_failed_done_does_not_call_done_callback():
	for reason in ['done_failed', 'done_without_success']:
		agent = make_graph_agent()
		callback = AsyncMock()
		agent.register_done_callback = callback
		agent._run_phase = AgentRunPhase.FAILED
		agent._run_terminal_reason = reason
		agent._run_success = False
		builder = AgentGraphBuilder(agent)
		builder.graph = MagicMock()
		builder.graph.ainvoke = AsyncMock()

		await builder.run(max_steps=5)

		callback.assert_not_awaited()
		agent.log_completion.assert_not_awaited()


async def test_manual_step_paths_publish_completion_only_for_success():
	agent = make_live_agent()
	agent.log_completion = AsyncMock()
	callback = AsyncMock()
	agent.register_done_callback = callback

	async def failed_done_step(_step_info=None):
		append_history(agent, done=True, success=False, content='failed')

	agent.step = AsyncMock(side_effect=failed_done_step)

	is_done, _ = await agent.take_step(AgentStepInfo(step_number=5, max_steps=10))
	assert is_done is True

	result = await agent._execute_step(0, 10, AgentStepInfo(step_number=0, max_steps=10))
	assert result is True

	agent.log_completion.assert_not_awaited()
	callback.assert_not_awaited()


async def test_successful_done_calls_done_callback_once():
	agent = make_graph_agent()
	callback = AsyncMock()
	agent.register_done_callback = callback
	agent._run_phase = AgentRunPhase.SUCCEEDED
	agent._run_terminal_reason = 'done'
	agent._run_success = True
	builder = AgentGraphBuilder(agent)
	builder.graph = MagicMock()
	builder.graph.ainvoke = AsyncMock()

	await builder.run(max_steps=5)

	agent.log_completion.assert_awaited_once()
	callback.assert_awaited_once_with(agent.history)


async def test_done_callback_failure_does_not_change_success():
	agent = make_graph_agent()
	callback = AsyncMock(side_effect=RuntimeError('callback failed'))
	agent.register_done_callback = callback
	agent._run_phase = AgentRunPhase.SUCCEEDED
	agent._run_terminal_reason = 'done'
	agent._run_success = True
	builder = AgentGraphBuilder(agent)
	builder.graph = MagicMock()
	builder.graph.ainvoke = AsyncMock()

	await builder.run(max_steps=5)

	assert agent._run_phase is AgentRunPhase.SUCCEEDED
	callback.assert_awaited_once_with(agent.history)
	agent.logger.exception.assert_called_once()


async def test_done_callback_cancellation_does_not_change_success():
	agent = make_graph_agent()
	callback = AsyncMock(side_effect=asyncio.CancelledError())
	agent.register_done_callback = callback
	agent._run_phase = AgentRunPhase.SUCCEEDED
	agent._run_terminal_reason = 'done'
	agent._run_success = True
	builder = AgentGraphBuilder(agent)
	builder.graph = MagicMock()
	builder.graph.ainvoke = AsyncMock()

	await builder.run(max_steps=5)

	assert agent._run_phase is AgentRunPhase.SUCCEEDED
	callback.assert_awaited_once_with(agent.history)


async def test_caller_cancellation_during_completion_propagates():
	agent = make_graph_agent()
	started = asyncio.Event()

	async def slow_completion():
		started.set()
		await asyncio.sleep(60)

	agent.log_completion = AsyncMock(side_effect=slow_completion)
	agent._run_phase = AgentRunPhase.SUCCEEDED
	agent._run_terminal_reason = 'done'
	agent._run_success = True
	builder = AgentGraphBuilder(agent)
	builder.graph = MagicMock()
	builder.graph.ainvoke = AsyncMock()

	task = asyncio.create_task(builder.run(max_steps=5))
	await asyncio.wait_for(started.wait(), timeout=1)
	task.cancel()

	with pytest.raises(asyncio.CancelledError):
		await task

	assert agent._run_phase is AgentRunPhase.SUCCEEDED


def test_manual_steps_reject_an_active_run():
	agent = make_live_agent()
	agent._run_active = True

	with pytest.raises(AgentRunStateError, match='operation is already active'):
		asyncio.run(agent.take_step(AgentStepInfo(step_number=1, max_steps=2)))
	with pytest.raises(AgentRunStateError, match='operation is already active'):
		asyncio.run(agent._execute_step(0, 2, AgentStepInfo(step_number=0, max_steps=2)))


async def test_manual_step_holds_the_operation_slot_across_hooks():
	agent = make_live_agent()
	agent.step = AsyncMock()
	started = asyncio.Event()
	release = asyncio.Event()

	async def on_step_start(_agent):
		started.set()
		await release.wait()

	manual_task = asyncio.create_task(
		agent._execute_step(0, 2, AgentStepInfo(step_number=0, max_steps=2), on_step_start=on_step_start)
	)
	await asyncio.wait_for(started.wait(), timeout=1)

	with pytest.raises(AgentRunStateError, match='run is already active'):
		await agent.run(max_steps=2)

	release.set()
	await asyncio.wait_for(manual_task, timeout=1)
	assert agent._run_active is False


def test_final_failure_message_uses_current_run_success():
	agent = make_live_agent()
	append_history(agent, done=True, success=True, content='previous')
	agent._run_success = False
	agent._run_phase = AgentRunPhase.FAILED

	with patch.object(agent.logger, 'info') as log_info:
		agent._log_final_outcome_messages()

	assert any('Did the Agent not work as expected?' in str(call) for call in log_info.call_args_list)
