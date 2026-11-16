import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from test_navigation import policy_handler_fixture

from openbrowser.browser.lifecycle import (
    LifecycleMachine,
    LifecyclePhase,
    LifecycleTransitionError,
)
from openbrowser.browser.navigation import (
    ContainmentAttempt,
    ContainmentPhase,
    NavigationCommit,
    NavigationError,
    NavigationOperation,
)


def test_lifecycle_revision_tracks_writes_without_changing_public_snapshot(monkeypatch):
    monkeypatch.setattr('openbrowser.browser.lifecycle.time.time', lambda: 1234567890.0)
    lifecycle = LifecycleMachine('revision-test')
    assert lifecycle.revision == 0
    initial = lifecycle.snapshot()
    lifecycle.transition(LifecyclePhase.STOPPED)
    assert lifecycle.snapshot() == initial
    assert lifecycle.revision == 1
    lifecycle.seed_generation(0)
    lifecycle.seed_generation('invalid')
    assert lifecycle.revision == 1
    lifecycle.seed_generation(4)
    assert lifecycle.revision == 2 and lifecycle.generation == 4
    with pytest.raises(LifecycleTransitionError):
        lifecycle.transition(LifecyclePhase.READY)
    assert lifecycle.revision == 2
    with pytest.raises(AttributeError):
        lifecycle.revision = 50
    lifecycle.transition(LifecyclePhase.STARTING, increment_generation=True)
    assert lifecycle.revision == 3 and lifecycle.generation == 5
    assert 'revision' not in lifecycle.snapshot().to_dict()


@asynccontextmanager
async def retained_policy(initial_phase=LifecyclePhase.READY):
    session, client, focus, frame, patches, controls = policy_handler_fixture()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    if initial_phase is LifecyclePhase.DEGRADED:
        session._lifecycle.transition(initial_phase, reason='existing health warning')
    controls.hold_action = True
    task = None
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://allowed.test/start'))
            await asyncio.wait_for(controls.action.wait(), 2)
            operation = session._navigation_monitor.active
            operation.containment.deadline = asyncio.get_running_loop().time()
            with pytest.raises(NavigationError, match='retir'):
                await session._join_navigation_policy(operation)
            assert session.lifecycle.phase is LifecyclePhase.FAILED
            yield session, client, focus, frame, controls, operation, task
    finally:
        controls.release_action.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        session._navigation_policies.clear()
        session._local_browser_watchdog = None
        session._stop_task = None
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


async def settle_policy(controls, operation, task):
    controls.release_action.set()
    await asyncio.wait_for(asyncio.shield(operation.policy_task), 2)
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('initial_phase', [LifecyclePhase.READY, LifecyclePhase.DEGRADED])
@pytest.mark.parametrize('outcome', ['contained', 'skipped'])
async def test_retired_policy_failure_becomes_degraded_without_claiming_ready(initial_phase, outcome):
    async with retained_policy(initial_phase) as (session, client, focus, frame, controls, operation, task):
        generation = session.lifecycle.generation
        if outcome == 'skipped':
            frame.update(loaderId='successor', url='https://allowed.test/successor')
        await settle_policy(controls, operation, task)
        assert operation.containment.phase.value == outcome
        assert session.lifecycle.phase is LifecyclePhase.DEGRADED
        assert session.lifecycle.generation == generation
        assert session._cdp_client_root is client and session.agent_focus is focus
        assert session.lifecycle.last_error is not None


async def test_local_retired_policy_allows_explicit_reset_without_killing_owned_browser():
    async with retained_policy() as (session, _client, _focus, _, controls, operation, task):
        session.browser_profile.is_local = True
        watchdog = SimpleNamespace(_subprocess=object(), _spawned_subprocess=None, _profile_lease=None)
        session._local_browser_watchdog = watchdog
        generation = session.lifecycle.generation
        await settle_policy(controls, operation, task)
        await session.reset(stop_event_bus=False)
        assert session.lifecycle.phase is LifecyclePhase.STOPPED
        assert session.lifecycle.generation > generation
        assert session._cdp_client_root is None and session.agent_focus is None
        assert session._local_browser_watchdog is watchdog
        assert watchdog._subprocess is not None


@pytest.mark.parametrize('interference', ['unrelated_failure', 'stopping', 'root', 'generation', 'shutdown'])
async def test_policy_retirement_cannot_resurrect_another_lifecycle_owner(interference):
    async with retained_policy() as (session, _, _, _, controls, operation, task):
        shutdown = None
        if interference == 'unrelated_failure':
            session._lifecycle.fail('independent failure', reason='not policy-owned')
        elif interference == 'stopping':
            session._lifecycle.transition(LifecyclePhase.STOPPING, reason='user stop')
        elif interference == 'root':
            session._cdp_client_root = None
        elif interference == 'generation':
            session.seed_generation(session.lifecycle.generation + 1)
        else:
            shutdown = asyncio.create_task(asyncio.Event().wait())
            session._stop_task = shutdown
        expected = session.lifecycle
        try:
            await settle_policy(controls, operation, task)
            if interference == 'unrelated_failure':
                assert session.lifecycle.reason == expected.reason
                assert session.lifecycle.last_error == expected.last_error
            assert session.lifecycle.phase in {LifecyclePhase.FAILED, LifecyclePhase.STOPPING}
        finally:
            if shutdown is not None:
                shutdown.cancel()
                await asyncio.gather(shutdown, return_exceptions=True)


async def test_unconfirmed_policy_failure_stays_failed_after_its_task_finishes():
    async with retained_policy() as (session, _, _, _, controls, operation, task):
        controls.action_error = ConnectionError('sent response lost')
        await settle_policy(controls, operation, task)
        assert operation.containment.phase is ContainmentPhase.UNCONFIRMED
        assert session.lifecycle.phase is LifecyclePhase.FAILED


async def test_other_unretired_policy_blocks_lifecycle_reconciliation():
    async with retained_policy() as (session, _, _, _, controls, operation, task):
        other = NavigationOperation('other', operation.generation, 'https://allowed.test/', 'load')
        other.containment = ContainmentAttempt(
            NavigationCommit('other-frame', 'other-loader', 'https://blocked.test/'),
            phase=ContainmentPhase.UNCONFIRMED,
        )
        session._navigation_policies[other.navigation_id] = other
        await settle_policy(controls, operation, task)
        assert session.lifecycle.phase is LifecyclePhase.FAILED


async def test_bus_pump_retirement_retries_a_completed_policy_reconciliation():
    async with retained_policy() as (session, _, _, _, controls, operation, task):
        release_pump = asyncio.Event()
        pump = asyncio.create_task(release_pump.wait())
        session._navigation_event_pumps[pump] = session.event_bus
        pump.add_done_callback(session._navigation_event_pump_done)
        try:
            await settle_policy(controls, operation, task)
            assert session.lifecycle.phase is LifecyclePhase.FAILED
            release_pump.set()
            await pump
            await asyncio.sleep(0)
            assert session.lifecycle.phase is LifecyclePhase.DEGRADED
        finally:
            release_pump.set()
            await asyncio.gather(pump, return_exceptions=True)


async def test_frozen_clock_cannot_hide_an_unrelated_identical_lifecycle_failure(monkeypatch):
    monkeypatch.setattr('openbrowser.browser.lifecycle.time.time', lambda: 1234567890.0)
    async with retained_policy() as (session, _, _, _, controls, operation, task):
        failure = session.lifecycle
        session._lifecycle.fail(failure.last_error, reason=failure.reason)
        assert session.lifecycle == failure
        await settle_policy(controls, operation, task)
        assert operation.containment.phase is ContainmentPhase.CONTAINED
        assert session.lifecycle.phase is LifecyclePhase.FAILED


async def test_repeated_owned_policy_failure_keeps_its_recovery_authority(monkeypatch):
    monkeypatch.setattr('openbrowser.browser.lifecycle.time.time', lambda: 1234567890.0)
    async with retained_policy() as (session, _, _, _, controls, operation, task):
        for _ in range(3):
            with pytest.raises(NavigationError):
                await session._join_navigation_policy(operation)
        await settle_policy(controls, operation, task)
        assert session.lifecycle.phase is LifecyclePhase.DEGRADED
