"""Tests for the explicit lifecycle state machine."""

import pytest

from openbrowser.browser.lifecycle import (
	LifecycleMachine,
	LifecyclePhase,
	LifecycleTransitionError,
)


def test_initial_state_is_stopped():
	machine = LifecycleMachine('browser')
	assert machine.phase is LifecyclePhase.STOPPED
	assert machine.generation == 0
	assert machine.snapshot().started_at is None


def test_start_ready_stop_transition():
	machine = LifecycleMachine('browser')
	machine.transition(LifecyclePhase.STARTING, reason='start requested')
	machine.transition(LifecyclePhase.READY, reason='browser ready')
	machine.transition(LifecyclePhase.STOPPING, reason='stop requested')
	snapshot = machine.transition(LifecyclePhase.STOPPED, reason='stopped', increment_generation=True)

	assert snapshot.phase is LifecyclePhase.STOPPED
	assert snapshot.previous_phase is LifecyclePhase.STOPPING
	assert snapshot.generation == 1
	assert snapshot.reason == 'stopped'


def test_same_phase_is_idempotent():
	machine = LifecycleMachine('browser')
	machine.transition(LifecyclePhase.STARTING)
	machine.transition(LifecyclePhase.READY, reason='ready')
	first = machine.snapshot()
	second = machine.transition(LifecyclePhase.READY, reason='ready again')

	assert second.phase is LifecyclePhase.READY
	assert second.generation == first.generation
	assert second.reason == 'ready again'


def test_illegal_transition_raises_with_context():
	machine = LifecycleMachine('browser')
	machine.transition(LifecyclePhase.STARTING)

	with pytest.raises(LifecycleTransitionError) as error:
		machine.transition(LifecyclePhase.DEGRADED, reason='not connected yet')

	message = str(error.value)
	assert 'browser' in message
	assert 'starting' in message
	assert 'degraded' in message
	assert 'not connected yet' in message


def test_failed_transition_records_error_and_allows_retry():
	machine = LifecycleMachine('daemon')
	machine.transition(LifecyclePhase.STARTING)
	machine.fail(RuntimeError('profile in use'), reason='start failed')

	snapshot = machine.snapshot()
	assert snapshot.phase is LifecyclePhase.FAILED
	assert snapshot.last_error == 'RuntimeError: profile in use'

	machine.transition(LifecyclePhase.STARTING, reason='retry')
	assert machine.snapshot().last_error is None
	machine.transition(LifecyclePhase.READY)
	assert machine.snapshot().last_error is None


def test_recovery_generation_increments_only_when_requested():
	machine = LifecycleMachine('browser')
	machine.transition(LifecyclePhase.STARTING)
	machine.transition(LifecyclePhase.READY)
	machine.transition(LifecyclePhase.RECOVERING, reason='cdp lost')
	machine.transition(LifecyclePhase.READY, reason='recovered', increment_generation=True)

	assert machine.generation == 1
	assert machine.snapshot().previous_phase is LifecyclePhase.RECOVERING


def test_snapshot_to_dict_uses_plain_values():
	machine = LifecycleMachine('mcp')
	machine.transition(LifecyclePhase.STARTING, reason='tool call')
	data = machine.snapshot().to_dict()

	assert data['name'] == 'mcp'
	assert data['phase'] == 'starting'
	assert data['previous_phase'] == 'stopped'
	assert isinstance(data['generation'], int)
