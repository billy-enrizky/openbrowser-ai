"""Behavior tests for explicit agent focus and focus recovery."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bubus import EventBus
from test_session_part1 import _make_browser_session, _make_cdp_session, _setup_session_with_cdp

from openbrowser.browser.events import (
    AgentFocusChangedEvent,
    BrowserErrorEvent,
    CloseTabEvent,
    NavigationCompleteEvent,
    SwitchTabEvent,
    TabClosedEvent,
)
from openbrowser.browser.focus import (
    FOCUS_TEXT_LIMIT,
    FocusChangeReason,
    FocusErrorSnapshot,
    FocusPhase,
    FocusSnapshot,
    FocusTargetCode,
    FocusTargetError,
    FocusTransitionBudget,
    FocusTransitionResult,
    default_focus_hint,
)
from openbrowser.browser.profile import BrowserProfile
from openbrowser.browser.session import BrowserSession
from openbrowser.browser.tabs import TabState
from openbrowser.tools.service import Tools


def _focused_session(target_id: str, session_id: str):
    session = _make_browser_session()
    client, focus, manager = _setup_session_with_cdp(session)
    replacement = _make_cdp_session(
        client=client, target_id=target_id, session_id=session_id, url="https://new.example"
    )

    async def target_info(**kwargs):
        requested = kwargs["params"]["targetId"]
        return {
            "targetInfo": {
                "targetId": requested,
                "type": "page",
                "url": "about:blank",
                "title": "",
            }
        }

    async def targets():
        return {
            "targetInfos": [
                {"targetId": focus.target_id, "type": "page", "url": focus.url, "title": focus.title},
                {"targetId": replacement.target_id, "type": "page", "url": replacement.url, "title": replacement.title},
            ]
        }

    client.send.Target.getTargetInfo = AsyncMock(side_effect=target_info)
    client.send.Target.getTargets = AsyncMock(side_effect=targets)
    return session, client, focus, manager, replacement


async def _stop_bus(session: BrowserSession) -> None:
    if isinstance(session.event_bus, EventBus):
        await session.event_bus.stop(clear=True, timeout=0.2)


async def _drain_focus_tasks(session: BrowserSession, seconds: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while session._focus_observer_tasks:
        tasks = list(session._focus_observer_tasks)
        if not tasks:
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Focus tasks did not drain")
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=remaining)


def test_focus_snapshot_nullability_and_bounded_error_text():
    uninitialized = FocusSnapshot.uninitialized()
    assert tuple(uninitialized.to_dict()) == (
        "phase",
        "target_id",
        "session_id",
        "previous_target_id",
        "generation",
        "reason",
        "last_error",
        "updated_at",
    )
    assert uninitialized.to_dict() == {
        "phase": "uninitialized",
        "target_id": None,
        "session_id": None,
        "previous_target_id": None,
        "generation": 0,
        "reason": None,
        "last_error": None,
        "updated_at": uninitialized.updated_at,
    }

    error = FocusTargetError(
        "x" * (FOCUS_TEXT_LIMIT + 50),
        code=FocusTargetCode.RECOVERY_FAILED,
        phase=FocusPhase.LOST,
        target_id="target",
        hint="y" * (FOCUS_TEXT_LIMIT + 50),
    )
    snapshot = error.to_snapshot()
    assert len(snapshot.message) == FOCUS_TEXT_LIMIT
    assert len(snapshot.hint) == FOCUS_TEXT_LIMIT
    assert snapshot.message.endswith("...")
    assert snapshot.hint.endswith("...")


def test_focus_snapshot_and_result_reject_illegal_phase_shapes():
    with pytest.raises(ValueError):
        FocusSnapshot(
            phase=FocusPhase.FOCUSED,
            target_id=None,
            session_id=None,
            previous_target_id=None,
            generation=1,
            reason=FocusChangeReason.SWITCH,
            last_error=None,
            updated_at=0.0,
        )
    with pytest.raises(ValueError):
        FocusSnapshot(
            phase=FocusPhase.LOST,
            target_id=None,
            session_id=None,
            previous_target_id="target",
            generation=1,
            reason=FocusChangeReason.CRASH_RECOVERY,
            last_error=None,
            updated_at=0.0,
        )

    session = _make_cdp_session(target_id="TARGET-0000", session_id="session-1")
    focused = FocusSnapshot(
        phase=FocusPhase.FOCUSED,
        target_id=session.target_id,
        session_id=session.session_id,
        previous_target_id=None,
        generation=1,
        reason=FocusChangeReason.SWITCH,
        last_error=None,
        updated_at=0.0,
    )
    assert FocusTransitionResult(session=session, snapshot=focused).session is session


def test_focus_event_fields_are_optional_and_deadlines_stay_in_process():
    switched = SwitchTabEvent(target_id=None)
    assert switched.focus_deadline_monotonic is None
    assert "focus_deadline_monotonic" not in switched.model_dump()

    closed = CloseTabEvent(target_id="target")
    assert closed.focus_deadline_monotonic is None
    assert "focus_deadline_monotonic" not in closed.model_dump()

    observed = AgentFocusChangedEvent(target_id="target", url="about:blank")
    assert observed.focus_generation is None
    assert observed.focus_session_id is None
    assert observed.focus_reason is None
    assert observed.focus_budget_seconds is None
    assert observed.focus_deadline_monotonic is None
    assert "focus_deadline_monotonic" not in observed.model_dump()

    stamped = AgentFocusChangedEvent(
        target_id="target",
        url="about:blank",
        focus_generation=2,
        focus_session_id="session",
        focus_reason=FocusChangeReason.SWITCH,
        focus_budget_seconds=1.0,
        focus_deadline_monotonic=10.0,
    )
    assert stamped.focus_reason is FocusChangeReason.SWITCH
    assert "focus_deadline_monotonic" not in stamped.model_dump()
    round_trip = AgentFocusChangedEvent.model_validate(stamped.model_dump())
    assert round_trip.focus_generation == 2
    assert round_trip.focus_session_id == "session"
    assert round_trip.focus_reason is FocusChangeReason.SWITCH
    assert round_trip.focus_budget_seconds == 1.0
    assert round_trip.focus_deadline_monotonic is None
    with pytest.raises(ValueError):
        AgentFocusChangedEvent(target_id="target", url="about:blank", focus_budget_seconds=float("nan"))


async def test_explicit_zero_event_budget_is_expired():
    budget = FocusTransitionBudget.from_event(deadline_monotonic=None, budget_seconds=0)
    assert budget.remaining() == 0.0
    assert budget.expired() is True


def test_focus_transition_budget_uses_one_absolute_deadline(monkeypatch):
    from openbrowser.browser import focus as focus_module

    monkeypatch.setattr(focus_module, "FOCUS_TRANSITION_BUDGET_SECONDS", 0.05)

    async def run():
        budget = FocusTransitionBudget.start()
        first = budget.remaining()
        await asyncio.sleep(0.02)
        second = budget.remaining()
        assert second < first <= 0.05
        with pytest.raises(ValueError):
            FocusTransitionBudget(deadline=float("nan"))

    asyncio.run(run())


async def test_lock_contention_consumes_the_focus_deadline_without_mutation():
    session, _client, _focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    before = session.focus_snapshot()
    await session._focus_lock.acquire()
    try:
        with pytest.raises(FocusTargetError) as error:
            await session._focus_transition_to_target(
                replacement.target_id,
                reason=FocusChangeReason.SWITCH,
                budget=FocusTransitionBudget.start(0.05),
                activate=False,
                probe=False,
            )
    finally:
        session._focus_lock.release()

    assert error.value.code is FocusTargetCode.TIMEOUT
    assert session.focus_snapshot() == before
    manager.get_live_session_for_target.assert_not_awaited()


async def test_non_page_focus_is_rejected_without_mutation():
    session, client, focus, manager, iframe = _focused_session("PAGE-TARGET-000000", "page-session")
    client.send.Target.getTargetInfo = AsyncMock(
        return_value={"targetInfo": {"targetId": iframe.target_id, "type": "iframe", "url": "https://frame.example"}}
    )
    manager.get_live_session_for_target = AsyncMock(return_value=iframe)
    before = session.focus_snapshot()

    with pytest.raises(FocusTargetError) as error:
        await session.get_or_create_cdp_session(iframe.target_id, focus=True)

    assert error.value.code is FocusTargetCode.NOT_A_PAGE
    assert session.agent_focus is focus
    assert session.focus_snapshot() == before


async def test_closing_tab_cannot_receive_focus_and_prior_state_is_restored():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(replacement.target_id, url=replacement.url, state=TabState.OPEN)
    session._tab_registry.mark_closing(replacement.target_id)
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    before = session.focus_snapshot()

    with pytest.raises(FocusTargetError) as error:
        await session.on_SwitchTabEvent(SwitchTabEvent(target_id=replacement.target_id))

    assert error.value.code is FocusTargetCode.TAB_NOT_LIVE
    assert session.agent_focus is focus
    assert session.focus_snapshot() == before


async def test_activation_failure_restores_prior_focus_and_cache():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    client.send.Target.activateTarget = AsyncMock(side_effect=RuntimeError("activation failed"))
    session._cached_browser_state_summary = "cached"
    before = session.focus_snapshot()

    with pytest.raises(FocusTargetError) as error:
        await session.on_SwitchTabEvent(SwitchTabEvent(target_id=replacement.target_id))

    assert error.value.code is FocusTargetCode.ACTIVATION_FAILED
    assert session.agent_focus is focus
    assert session.focus_snapshot() == before
    assert session._cached_browser_state_summary == "cached"


async def test_candidate_detach_before_commit_restores_a_live_prior_focus():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    candidate_calls = 0

    async def live_session(target_id, *, budget=None):
        nonlocal candidate_calls
        if target_id == focus.target_id:
            return focus
        if target_id == replacement.target_id:
            candidate_calls += 1
            return replacement if candidate_calls == 1 else None
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    with (
        patch.object(session, "_schedule_focus_recovery") as schedule,
        pytest.raises(FocusTargetError) as error,
    ):
        await session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(),
            activate=False,
            probe=False,
        )

    assert error.value.code is FocusTargetCode.TARGET_DETACHED
    assert session.agent_focus is focus
    assert session.focus_snapshot().target_id == focus.target_id
    schedule.assert_not_called()


async def test_failed_switch_does_not_restore_a_detached_checkpoint():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")

    async def live_session(target_id, *, budget=None):
        if target_id == replacement.target_id:
            return replacement
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    manager.active_session_ids = AsyncMock(return_value=())
    client.send.Runtime.evaluate = AsyncMock(side_effect=RuntimeError("probe failed"))
    with (
        patch.object(session, "_schedule_focus_recovery") as schedule,
        pytest.raises(FocusTargetError),
    ):
        await session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(),
            activate=False,
            probe=True,
        )

    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.LOST
    assert snapshot.previous_target_id == focus.target_id
    assert snapshot.reason is FocusChangeReason.DETACH_RECOVERY
    assert session.agent_focus is None
    schedule.assert_called_once_with(
        focus.target_id,
        FocusChangeReason.DETACH_RECOVERY,
        expected_generation=2,
    )


async def test_unresponsive_target_restores_prior_focus():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 1}})
    before = session.focus_snapshot()

    with pytest.raises(FocusTargetError) as error:
        await session.on_SwitchTabEvent(SwitchTabEvent(target_id=replacement.target_id))

    assert error.value.code is FocusTargetCode.UNRESPONSIVE
    assert session.agent_focus is focus
    assert session.focus_snapshot() == before


async def test_same_pair_observation_is_a_cache_and_generation_noop():
    session, _client, _focus, _manager, _ = _focused_session("PAGE-TARGET-000000", "page-session")
    dom_watchdog = MagicMock()
    session._dom_watchdog = dom_watchdog
    before = session.focus_snapshot()

    await session.on_AgentFocusChangedEvent(
        AgentFocusChangedEvent(
            target_id=before.target_id,
            url="https://new.example",
            focus_generation=before.generation,
            focus_session_id=before.session_id,
        )
    )

    assert session.focus_snapshot() == before
    dom_watchdog.clear_cache.assert_not_called()


async def test_stale_generation_observation_cannot_undo_newer_focus():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    result = await session._focus_transition_to_target(
        replacement.target_id,
        reason=FocusChangeReason.SWITCH,
        budget=FocusTransitionBudget.start(),
        activate=False,
        probe=False,
    )
    committed = session.focus_snapshot()
    assert committed.session_id == replacement.session_id

    await session.on_AgentFocusChangedEvent(
        AgentFocusChangedEvent(
            target_id=focus.target_id,
            url="https://old.example",
            focus_generation=result.snapshot.generation - 1,
            focus_session_id=focus.session_id,
        )
    )

    assert session.focus_snapshot() == committed
    assert session.agent_focus is replacement


async def test_failed_initial_external_focus_commits_lost():
    session = BrowserSession(browser_profile=BrowserProfile(cdp_url="ws://localhost:9222", is_local=False))
    client = MagicMock()
    client.send.Target.getTargetInfo = AsyncMock(side_effect=RuntimeError("missing target"))
    session._cdp_client_root = client

    with pytest.raises(FocusTargetError):
        await session.on_AgentFocusChangedEvent(
            AgentFocusChangedEvent(target_id="MISSING-TARGET-0000", url="about:blank")
        )

    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.LOST
    assert snapshot.reason is FocusChangeReason.INITIAL
    assert snapshot.last_error is not None
    assert snapshot.last_error.code is FocusTargetCode.TARGET_MISSING
    assert session.agent_focus is None


async def test_switch_observation_failure_is_typed_after_commit():
    session, _client, _focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()

    async def failing_observer(event: AgentFocusChangedEvent):
        raise RuntimeError("observer failed")

    session.event_bus.on(AgentFocusChangedEvent, failing_observer)
    try:
        with pytest.raises(FocusTargetError) as error:
            await session.on_SwitchTabEvent(SwitchTabEvent(target_id=replacement.target_id))
        assert error.value.code is FocusTargetCode.OBSERVATION_FAILED
        assert session.agent_focus is replacement
        assert session.focus_snapshot().target_id == replacement.target_id
    finally:
        await _stop_bus(session)


async def test_slow_navigation_observer_does_not_delay_focus_observation():
    session, _client, _focus, _manager, _ = _focused_session("PAGE-TARGET-000000", "page-session")
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_observer(event: NavigationCompleteEvent):
        entered.set()
        await release.wait()

    session.event_bus.on(NavigationCompleteEvent, slow_observer)
    try:
        snapshot = session.focus_snapshot()
        started = asyncio.get_running_loop().time()
        await asyncio.wait_for(
            session._notify_focus_changed(
                snapshot.target_id,
                "https://example.com",
                FocusChangeReason.SWITCH,
                FocusTransitionBudget.start(),
                snapshot,
            ),
            0.5,
        )
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.5
        await asyncio.wait_for(entered.wait(), 0.5)
        assert not release.is_set()
    finally:
        release.set()
        await _stop_bus(session)


async def test_same_target_attach_rebinds_once_and_old_detach_is_noop():
    session, client, focus, manager, _ = _focused_session("OTHER-TARGET-000000", "other-session")
    replacement = _make_cdp_session(
        client=client,
        target_id=focus.target_id,
        session_id="rebound-session",
        url=focus.url,
    )
    before_generation = session.focus_snapshot().generation
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    session._cdp_session_pool[replacement.target_id] = replacement
    dom_watchdog = MagicMock()
    session._dom_watchdog = dom_watchdog

    await session._handle_attached_session(replacement.target_id, replacement)

    assert session.agent_focus is replacement
    assert session.focus_snapshot().session_id == replacement.session_id
    assert session.focus_snapshot().generation == before_generation + 1
    dom_watchdog.clear_cache.assert_called_once()

    await session._handle_detached_session(replacement.target_id, focus.session_id, (replacement.session_id,))
    assert session.agent_focus is replacement


async def test_partial_focused_session_detach_rebinds_deterministically():
    session, client, focus, manager, _ = _focused_session("OTHER-TARGET-000000", "other-session")
    replacement = _make_cdp_session(
        client=client,
        target_id=focus.target_id,
        session_id="rebound-session",
        url=focus.url,
    )
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)

    with patch.object(BrowserSession, "_set_agent_focus", AsyncMock(return_value=session.focus_snapshot())) as commit:
        await session._handle_detached_session(focus.target_id, focus.session_id, ("a-session", "b-session"))

    commit.assert_awaited_once_with(
        replacement,
        FocusChangeReason.SESSION_REBIND,
        expected_generation=1,
    )


async def test_delayed_partial_rebind_cannot_override_newer_focus():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    rebound = _make_cdp_session(
        client=client,
        target_id=focus.target_id,
        session_id="rebound-session",
        url=focus.url,
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def live_session(target_id, *, budget=None):
        if target_id == focus.target_id:
            entered.set()
            await release.wait()
            return rebound
        if target_id == replacement.target_id:
            return replacement
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    task = asyncio.create_task(
        session._handle_detached_session(focus.target_id, focus.session_id, (rebound.session_id,))
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        await session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(),
            activate=False,
            probe=False,
        )
    finally:
        release.set()
    try:
        with pytest.raises(FocusTargetError) as error:
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session.agent_focus is replacement
    assert session.focus_snapshot().target_id == replacement.target_id


async def test_delayed_accessor_rebind_cannot_override_newer_focus():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    rebound = _make_cdp_session(
        client=client,
        target_id=focus.target_id,
        session_id="rebound-session",
        url=focus.url,
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def live_session(target_id, *, budget=None):
        if target_id == focus.target_id:
            entered.set()
            await release.wait()
            return rebound
        if target_id == replacement.target_id:
            return replacement
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    task = asyncio.create_task(session.get_or_create_cdp_session(focus.target_id, focus=True))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        await session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(),
            activate=False,
            probe=False,
        )
    finally:
        release.set()
    try:
        with pytest.raises(FocusTargetError) as error:
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session.agent_focus is replacement
    assert session.focus_snapshot().target_id == replacement.target_id


async def test_implicit_accessor_during_switch_cannot_restore_old_focus():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    entered, release = asyncio.Event(), asyncio.Event()

    async def live_session(target_id, *, budget=None):
        if target_id == replacement.target_id:
            entered.set()
            await release.wait()
            return replacement
        if target_id == focus.target_id:
            return focus
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    switch_task = asyncio.create_task(
        session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(),
            activate=False,
            probe=False,
        )
    )
    await asyncio.wait_for(entered.wait(), 1.0)
    accessor_task = asyncio.create_task(session.get_or_create_cdp_session())
    for _attempt in range(100):
        if getattr(session._focus_lock, "_waiters", ()):
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("Accessor never reached the focus lock")
    release.set()
    try:
        await asyncio.wait_for(switch_task, 1.0)
        with pytest.raises(FocusTargetError) as error:
            await asyncio.wait_for(accessor_task, 1.0)
    finally:
        for task in (switch_task, accessor_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(switch_task, accessor_task, return_exceptions=True)

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session.agent_focus is replacement
    assert session.focus_snapshot().target_id == replacement.target_id


async def test_focused_session_fast_path_uses_the_shared_budget():
    session, _client, focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")

    async def slow_lookup(target_id, *, budget=None):
        await asyncio.sleep(0.05)
        return focus

    manager.get_live_session_for_target = AsyncMock(side_effect=slow_lookup)
    with pytest.raises(FocusTargetError) as error:
        await session.get_or_create_cdp_session(
            budget=FocusTransitionBudget.start(0.02),
        )

    assert error.value.code is FocusTargetCode.TIMEOUT


async def test_implicit_accessor_rejects_a_replacement_target_after_missing_session():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    focus.cdp_client.send.Runtime.runIfWaitingForDebugger = AsyncMock(
        side_effect=RuntimeError("Session with given id not found")
    )
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]
    )

    async def live_session(target_id, *, budget=None):
        if target_id == replacement.target_id:
            return replacement
        if target_id == focus.target_id:
            return focus
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    with pytest.raises(FocusTargetError) as error:
        await asyncio.wait_for(session.get_or_create_cdp_session(), timeout=1.0)

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.FOCUSED
    assert snapshot.target_id == replacement.target_id
    assert session.agent_focus is replacement


async def test_implicit_accessor_cannot_return_a_new_focus_after_a_blocked_missing_session():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(
        side_effect=lambda target_id, *, budget=None: focus if target_id == focus.target_id else replacement
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_missing_resume(**kwargs):
        entered.set()
        await release.wait()
        raise RuntimeError("Session with given id not found")

    focus.cdp_client.send.Runtime.runIfWaitingForDebugger = AsyncMock(side_effect=slow_missing_resume)
    task = asyncio.create_task(session.get_or_create_cdp_session())
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    await session._focus_transition_to_target(
        replacement.target_id,
        reason=FocusChangeReason.SWITCH,
        budget=FocusTransitionBudget.start(),
        activate=False,
        probe=False,
    )
    release.set()
    with pytest.raises(FocusTargetError) as error:
        await asyncio.wait_for(task, timeout=1.0)

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session.focus_snapshot().target_id == replacement.target_id
    assert session.agent_focus is replacement


async def test_explicit_missing_session_returns_a_validated_replacement():
    session, client, focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    requested_target = "REQUESTED-TARGET-000"
    first = _make_cdp_session(
        client=client,
        target_id=requested_target,
        session_id="requested-session-1",
        url="https://requested.example",
    )
    second = _make_cdp_session(
        client=client,
        target_id=requested_target,
        session_id="requested-session-2",
        url="https://requested.example",
    )
    use_replacement = False

    async def live_session(target_id, *, budget=None):
        if target_id == requested_target:
            return second if use_replacement else first
        return focus

    async def missing_resume(**kwargs):
        nonlocal use_replacement
        use_replacement = True
        raise RuntimeError("Session with given id not found")

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    manager.active_session_ids = AsyncMock(
        side_effect=lambda target_id, *, budget=None: (
            (first.session_id, second.session_id) if target_id == requested_target else (focus.session_id,)
        )
    )
    client.send.Runtime.runIfWaitingForDebugger = AsyncMock(side_effect=missing_resume)
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": 2}})

    result = await asyncio.wait_for(
        session.get_or_create_cdp_session(requested_target, focus=True),
        timeout=1.0,
    )

    assert result is second
    assert session.agent_focus is second
    assert session.focus_snapshot().session_id == second.session_id


async def test_implicit_focus_false_lookup_rejects_a_newer_switch():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    entered, release = asyncio.Event(), asyncio.Event()

    async def gated_live(target_id, *, budget=None):
        if target_id == replacement.target_id:
            return replacement
        entered.set()
        await release.wait()
        return focus

    manager.get_live_session_for_target = AsyncMock(side_effect=gated_live)
    task = asyncio.create_task(session.get_or_create_cdp_session(focus=False))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    await session._focus_transition_to_target(
        replacement.target_id,
        reason=FocusChangeReason.SWITCH,
        budget=FocusTransitionBudget.start(),
        activate=False,
        probe=False,
    )
    release.set()
    with pytest.raises(FocusTargetError) as error:
        await asyncio.wait_for(task, timeout=1.0)

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session.focus_snapshot().target_id == replacement.target_id


async def test_close_cleanup_rejects_a_new_attempt_before_terminal_write():
    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session._tab_registry.mark_closing(focus.target_id)
    first_attempt = session._tab_registry.get(focus.target_id).close_attempt
    await session._focus_lock.acquire()
    with patch.object(session, "_target_liveness", AsyncMock(return_value=False)):
        session._schedule_close_reconciliation(focus.target_id)
        for _attempt in range(100):
            if getattr(session._focus_lock, "_waiters", ()):
                break
            await asyncio.sleep(0)
        else:
            session._focus_lock.release()
            raise AssertionError("Cleanup never waited for the focus lock")
        session._tab_registry.mark_closing(focus.target_id)
        session._focus_lock.release()
        await _drain_focus_tasks(session)

    record = session._tab_registry.get(focus.target_id)
    assert record is not None
    assert record.close_attempt > first_attempt
    assert record.state is TabState.CLOSING
    assert session.focus_snapshot().phase is FocusPhase.FOCUSED


async def test_recovery_expected_generation_rejects_a_newer_commit():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")

    async def live_session(target_id, *, budget=None):
        if target_id == replacement.target_id:
            return replacement
        if target_id == focus.target_id:
            return focus
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    await session._focus_transition_to_target(
        replacement.target_id,
        reason=FocusChangeReason.SWITCH,
        budget=FocusTransitionBudget.start(),
        activate=False,
        probe=False,
    )

    with pytest.raises(FocusTargetError) as error:
        await session._recover_agent_focus(
            focus.target_id,
            reason=FocusChangeReason.DETACH_RECOVERY,
            budget=FocusTransitionBudget.start(),
            expected_generation=1,
        )

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session.focus_snapshot().target_id == replacement.target_id


async def test_focused_full_detach_recovers_to_live_page():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "old-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    with patch.object(
        BrowserSession,
        "_cdp_get_all_pages",
        AsyncMock(return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]),
    ):
        await session._handle_detached_session(focus.target_id, focus.session_id, ())

    assert session.agent_focus is replacement
    assert session.focus_snapshot().reason is FocusChangeReason.DETACH_RECOVERY


async def test_recovery_excludes_crashed_and_closing_pages():
    session, _client, _focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    crashed = "CRASHED-TARGET-0000"
    closing = "CLOSING-TARGET-0000"
    for target_id, state in ((crashed, TabState.CRASHED), (closing, TabState.CLOSING)):
        session._tab_registry.register(target_id, url="https://bad.example", state=TabState.OPEN)
        session._tab_registry.update(target_id, state=state)
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    pages = [
        {"targetId": crashed, "type": "page", "url": "https://bad.example"},
        {"targetId": closing, "type": "page", "url": "https://bad.example"},
        {"targetId": replacement.target_id, "type": "page", "url": replacement.url},
    ]
    with patch.object(BrowserSession, "_cdp_get_all_pages", AsyncMock(return_value=pages)):
        result = await session._focus_transition_to_target(
            None,
            reason=FocusChangeReason.DETACH_RECOVERY,
            budget=FocusTransitionBudget.start(),
            create_if_missing=True,
        )

    assert result.session is replacement
    assert result.created_target_id is None


async def test_recovery_creates_one_page_and_reports_created_target():
    session, client, _focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    created = _make_cdp_session(
        client=client, target_id="CREATED-TARGET-0000", session_id="created-session", url="about:blank"
    )
    created_targets = []

    async def create_page(url="about:blank", **kwargs):
        created_targets.append(url)
        return {"targetId": created.target_id}

    client.send.Target.createTarget = AsyncMock(side_effect=create_page)
    manager.get_live_session_for_target = AsyncMock(return_value=created)
    with patch.object(BrowserSession, "_cdp_get_all_pages", AsyncMock(return_value=[])):
        result = await session._focus_transition_to_target(
            None,
            reason=FocusChangeReason.DETACH_RECOVERY,
            budget=FocusTransitionBudget.start(),
            create_if_missing=True,
        )

    assert created_targets == ["about:blank"]
    assert result.created_target_id == created.target_id
    assert session.agent_focus is created


async def test_expired_switch_budget_restores_a_live_prior_focus():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(
        side_effect=lambda target_id, *, budget=None: replacement if target_id == replacement.target_id else focus
    )
    manager.active_session_ids = AsyncMock(return_value=(focus.session_id,))

    async def slow_probe(**kwargs):
        await asyncio.sleep(0.05)
        return {"result": {"value": 2}}

    client.send.Runtime.evaluate = AsyncMock(side_effect=slow_probe)
    with pytest.raises(FocusTargetError) as error:
        await session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(0.02),
            activate=False,
            probe=True,
        )

    assert error.value.code is FocusTargetCode.TIMEOUT
    assert session.agent_focus is focus
    assert session.focus_snapshot().target_id == focus.target_id


async def test_unknown_rollback_liveness_publishes_lost_without_false_recovery():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(
        side_effect=lambda target_id, *, budget=None: replacement if target_id == replacement.target_id else focus
    )
    manager.active_session_ids = AsyncMock(side_effect=RuntimeError("liveness unavailable"))

    async def slow_probe(**kwargs):
        await asyncio.sleep(0.05)
        return {"result": {"value": 2}}

    client.send.Runtime.evaluate = AsyncMock(side_effect=slow_probe)
    with pytest.raises(FocusTargetError):
        await session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(0.02),
            activate=False,
            probe=True,
        )

    snapshot = session.focus_snapshot()
    assert session.agent_focus is None
    assert snapshot.phase is FocusPhase.LOST
    assert snapshot.previous_target_id == focus.target_id
    assert snapshot.reason is FocusChangeReason.LIVENESS_RECOVERY
    assert snapshot.last_error is not None
    assert snapshot.last_error.code is FocusTargetCode.TARGET_LIVENESS_UNKNOWN


async def test_exact_session_liveness_preserves_a_timeout():
    session, _client, focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")

    async def slow_active_ids(target_id, *, budget=None):
        await asyncio.sleep(0.05)
        return (focus.session_id,)

    manager.active_session_ids = AsyncMock(side_effect=slow_active_ids)
    with pytest.raises(FocusTargetError) as error:
        await session._focus_exact_session_is_live(
            focus.target_id,
            focus.session_id,
            FocusTransitionBudget.start(0.01),
            phase=FocusPhase.FOCUSED,
            use_active_ids=True,
        )

    assert error.value.code is FocusTargetCode.TIMEOUT


async def test_focus_session_lookup_timeout_is_typed():
    session, _client, focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(side_effect=TimeoutError)

    with pytest.raises(FocusTargetError) as error:
        await session._get_focus_session_locked(
            focus.target_id,
            FocusTransitionBudget.start(1.0),
        )

    assert error.value.code is FocusTargetCode.TIMEOUT


async def test_recovery_retry_create_failure_is_typed():
    session, _client, _focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    recovery_error = FocusTargetError(
        "No page available.",
        code=FocusTargetCode.NO_PAGE_AVAILABLE,
        phase=FocusPhase.RECOVERING,
    )
    with (
        patch.object(session, "_focus_transition_to_target", AsyncMock(side_effect=recovery_error)),
        patch.object(session, "_cdp_create_new_page", AsyncMock(side_effect=RuntimeError("create failed"))),
        pytest.raises(FocusTargetError) as error,
    ):
        await session._recover_agent_focus(
            "CRASHED-TARGET-0000",
            reason=FocusChangeReason.CRASH_RECOVERY,
            budget=FocusTransitionBudget.start(),
        )

    assert error.value.code is FocusTargetCode.RECOVERY_FAILED
    assert "create failed" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)


async def test_recovery_retry_rejects_a_reset_after_page_creation():
    session, _client, _focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    recovery_error = FocusTargetError(
        "No page available.",
        code=FocusTargetCode.NO_PAGE_AVAILABLE,
        phase=FocusPhase.RECOVERING,
    )

    async def create_after_reset(*args, **kwargs):
        session._focus_epoch += 1
        return "RESET-TARGET-00000"

    with (
        patch.object(session, "_focus_transition_to_target", AsyncMock(side_effect=recovery_error)),
        patch.object(session, "_cdp_create_new_page", AsyncMock(side_effect=create_after_reset)),
        pytest.raises(FocusTargetError) as error,
    ):
        await session._recover_agent_focus(
            "CRASHED-TARGET-0000",
            reason=FocusChangeReason.CRASH_RECOVERY,
            budget=FocusTransitionBudget.start(),
        )

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session._tab_registry.get("RESET-TARGET-00000") is None


async def test_close_reconciliation_ignores_a_newer_close_attempt():
    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session._tab_registry.mark_closing(focus.target_id)
    first_attempt = session._tab_registry.get(focus.target_id).close_attempt
    session._tab_registry.mark_closing(focus.target_id)

    with patch.object(session, "_target_liveness", AsyncMock(return_value=True)):
        await session._reconcile_uncertain_close(
            focus.target_id,
            FocusTransitionBudget.start(),
            expected_generation=session.focus_snapshot().generation,
            expected_close_attempt=first_attempt,
        )

    record = session._tab_registry.get(focus.target_id)
    assert record is not None and record.state is TabState.CLOSING
    assert session.focus_snapshot().phase is FocusPhase.FOCUSED


async def test_reopen_restore_timeout_schedules_a_bounded_retry():
    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session._focus_lost_locked(
        FocusTargetError(
            "Close outcome was unknown.",
            code=FocusTargetCode.TARGET_DETACHED,
            phase=FocusPhase.FOCUSED,
            target_id=focus.target_id,
        ),
        FocusChangeReason.CLOSE_REPAIR,
        previous_target_id=focus.target_id,
    )
    expected_generation = session.focus_snapshot().generation
    timeout_error = FocusTargetError(
        "Restore timed out.",
        code=FocusTargetCode.TIMEOUT,
        phase=FocusPhase.RECOVERING,
        target_id=focus.target_id,
    )

    async def timeout_after_recovering(*args, **kwargs):
        session._begin_focus_change_locked(
            FocusChangeReason.CLOSE_REPAIR,
            None,
            FocusPhase.RECOVERING,
        )
        session._focus_lost_locked(
            FocusTargetError(
                "Restore timed out.",
                code=FocusTargetCode.TIMEOUT,
                phase=FocusPhase.RECOVERING,
                target_id=focus.target_id,
            ),
            FocusChangeReason.CLOSE_REPAIR,
            previous_target_id=focus.target_id,
        )
        raise timeout_error

    with (
        patch.object(session, "_focus_transition_to_target", side_effect=timeout_after_recovering),
        patch.object(session, "_schedule_reopened_focus_restore") as schedule,
    ):
        await session._restore_focus_after_close_reopen(
            focus.target_id,
            FocusTransitionBudget.start(0.01),
        )

    schedule.assert_called_once_with(
        focus.target_id,
        expected_generation=session.focus_snapshot().generation,
    )
    assert session.focus_snapshot().generation > expected_generation


async def test_implicit_session_return_rechecks_focus_after_resume():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(
        side_effect=lambda target_id, *, budget=None: focus if target_id == focus.target_id else replacement
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_resume(**kwargs):
        entered.set()
        await release.wait()

    focus.cdp_client.send.Runtime.runIfWaitingForDebugger = AsyncMock(side_effect=slow_resume)
    task = asyncio.create_task(session.get_or_create_cdp_session())
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    await session._focus_transition_to_target(
        replacement.target_id,
        reason=FocusChangeReason.SWITCH,
        budget=FocusTransitionBudget.start(),
        activate=False,
        probe=False,
    )
    release.set()
    with pytest.raises(FocusTargetError) as error:
        await asyncio.wait_for(task, timeout=1.0)

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    assert session.focus_snapshot().target_id == replacement.target_id


async def test_recovery_creates_an_emergency_page_after_attach_failure(monkeypatch):
    session, client, _focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    existing_target = "EXISTING-TARGET-000"
    emergency_target = "EMERGENCY-TARGET-00"
    emergency = _make_cdp_session(
        client=client,
        target_id=emergency_target,
        session_id="emergency-session",
        url="about:blank",
    )

    async def live_session(target_id, *, budget=None):
        if target_id == existing_target:
            return None
        if target_id == emergency_target:
            return emergency
        return None

    manager.get_live_session_for_target = AsyncMock(side_effect=live_session)
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": existing_target, "type": "page", "url": "https://existing.example"}]
    )
    session._cdp_create_new_page = AsyncMock(return_value=emergency_target)
    monkeypatch.setattr("openbrowser.browser.session.asyncio.sleep", AsyncMock(return_value=None))

    result = await session._recover_agent_focus(
        "CRASHED-TARGET-0000",
        reason=FocusChangeReason.CRASH_RECOVERY,
        budget=FocusTransitionBudget.start(),
    )

    assert result.target_id == emergency_target
    assert session.agent_focus is emergency
    assert session.focus_snapshot().target_id == emergency_target
    session._cdp_create_new_page.assert_awaited_once_with("about:blank")


async def test_reset_during_gated_recovery_cannot_restore_or_late_commit():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    entered, release = asyncio.Event(), asyncio.Event()

    async def gated_live(target_id, *, budget=None):
        entered.set()
        await release.wait()
        return replacement

    manager.get_live_session_for_target = AsyncMock(side_effect=gated_live)
    with patch.object(
        BrowserSession,
        "_cdp_get_all_pages",
        AsyncMock(return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]),
    ):
        task = asyncio.create_task(
            session._recover_agent_focus(
                focus.target_id,
                reason=FocusChangeReason.RESPONSIVENESS_RECOVERY,
                budget=FocusTransitionBudget.start(),
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        session._force_reset_runtime_state(replace_event_bus=False)
        release.set()
        with pytest.raises(FocusTargetError):
            await task

    assert session.focus_snapshot().phase is FocusPhase.LOST
    assert session.focus_snapshot().reason is FocusChangeReason.RESET
    assert session.agent_focus is None


async def test_reset_during_switch_reports_the_committed_prior_target():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    entered, release = asyncio.Event(), asyncio.Event()

    async def gated_probe(**_kwargs):
        entered.set()
        await release.wait()
        return {"result": {"value": 2}}

    client.send.Runtime.evaluate = AsyncMock(side_effect=gated_probe)
    task = asyncio.create_task(
        session._focus_transition_to_target(
            replacement.target_id,
            reason=FocusChangeReason.SWITCH,
            budget=FocusTransitionBudget.start(),
            activate=False,
            probe=True,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    session._focus_epoch += 1
    session._focus_lost_on_reset()
    release.set()
    with pytest.raises(FocusTargetError) as error:
        await task

    assert error.value.code is FocusTargetCode.STALE_TRANSITION
    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.LOST
    assert snapshot.reason is FocusChangeReason.RESET
    assert snapshot.previous_target_id == focus.target_id


async def test_normal_reset_overrides_a_cancelled_recovery_reason():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    entered, release = asyncio.Event(), asyncio.Event()

    async def gated_live(target_id, *, budget=None):
        entered.set()
        await release.wait()
        return replacement

    manager.get_live_session_for_target = AsyncMock(side_effect=gated_live)
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]
    )
    session._retire_event_bus = AsyncMock()
    session._cancel_navigation = AsyncMock()
    session._capture_download_history = AsyncMock(return_value=[])
    task = asyncio.create_task(
        session._recover_agent_focus(
            focus.target_id,
            reason=FocusChangeReason.CRASH_RECOVERY,
            budget=FocusTransitionBudget.start(),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    async def clear():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    manager.clear = AsyncMock(side_effect=clear)
    await session._reset_locked(stop_event_bus=False)

    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.LOST
    assert snapshot.reason is FocusChangeReason.RESET
    assert snapshot.previous_target_id == focus.target_id


async def test_close_repair_moves_focus_and_records_reason():
    session, _client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    with patch.object(
        BrowserSession,
        "_cdp_get_all_pages",
        AsyncMock(return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]),
    ):
        await session.on_TabClosedEvent(TabClosedEvent(target_id=focus.target_id))

    assert session.agent_focus is replacement
    assert session.focus_snapshot().reason is FocusChangeReason.CLOSE_REPAIR


async def test_close_repair_failure_returns_typed_error_and_lost_state():
    session, client, focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    client.send.Target.createTarget = AsyncMock(side_effect=RuntimeError("no page"))
    manager.get_live_session_for_target = AsyncMock(return_value=None)
    with (
        patch.object(BrowserSession, "_cdp_get_all_pages", AsyncMock(return_value=[])),
        pytest.raises(FocusTargetError) as error,
    ):
        await session.on_TabClosedEvent(TabClosedEvent(target_id=focus.target_id))

    assert error.value.code is FocusTargetCode.RECOVERY_FAILED
    assert session.focus_snapshot().phase is FocusPhase.LOST
    assert session.agent_focus is None


async def test_close_post_event_timeout_marks_focus_lost_before_raising():
    session, client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    client.send.Target.closeTarget = AsyncMock(return_value={})
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)

    async def timeout_event(*args, **kwargs):
        raise TimeoutError

    with (
        patch.object(session, "_await_event_result", side_effect=timeout_event),
        pytest.raises(TimeoutError),
    ):
        await session.on_CloseTabEvent(CloseTabEvent(target_id=focus.target_id))

    assert session.focus_snapshot().phase is FocusPhase.LOST
    assert session.focus_snapshot().previous_target_id == focus.target_id
    assert session.agent_focus is None
    record = session._tab_registry.get(focus.target_id)
    assert record is not None and record.state is TabState.CLOSED
    await _drain_focus_tasks(session)


async def test_expired_tab_closed_event_marks_focus_lost_before_raising():
    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    event = TabClosedEvent(
        target_id=focus.target_id,
        focus_deadline_monotonic=asyncio.get_running_loop().time() - 1.0,
    )

    with pytest.raises(FocusTargetError):
        await session.on_TabClosedEvent(event)

    assert session.focus_snapshot().phase is FocusPhase.LOST
    assert session.focus_snapshot().previous_target_id == focus.target_id
    assert session.agent_focus is None
    record = session._tab_registry.get(focus.target_id)
    assert record is not None and record.state is TabState.CLOSED
    await _drain_focus_tasks(session)


def test_unstarted_session_status_reports_uninitialized_focus():
    session = BrowserSession()
    focus = session.lifecycle_snapshot()["focus"]
    assert focus["phase"] == "uninitialized"
    assert focus["target_id"] is None
    assert focus["generation"] == 0


def test_focus_error_snapshot_serialization_is_deterministic():
    snapshot = FocusErrorSnapshot(
        code=FocusTargetCode.TIMEOUT,
        message="  timed\n out  ",
        hint="retry",
        target_id="target",
        target_type="page",
        phase=FocusPhase.LOST,
    )
    assert snapshot.to_dict() == {
        "code": "timeout",
        "message": "timed out",
        "hint": "retry",
        "target_id": "target",
        "target_type": "page",
        "phase": "lost",
    }


def test_default_focus_hints_are_surface_neutral_and_bounded():
    for code in FocusTargetCode:
        hint = default_focus_hint(code)
        assert hint
        assert "get_status" not in hint
        assert len(hint) <= FOCUS_TEXT_LIMIT


def test_focus_recovery_failure_publishes_a_bounded_browser_error():
    session = BrowserSession()
    session.event_bus.dispatch = MagicMock()
    error = FocusTargetError(
        "No live page target was available.",
        code=FocusTargetCode.RECOVERY_FAILED,
        phase=FocusPhase.LOST,
        target_id="target-1",
        target_type="page",
    )

    session._report_focus_recovery_failure("target-1", error)

    event = session.event_bus.dispatch.call_args.args[0]
    assert isinstance(event, BrowserErrorEvent)
    assert event.error_type == "FocusRecoveryError"
    assert event.details["code"] == "recovery_failed"
    assert event.details["target_id"] == "target-1"


def test_focus_loss_clears_focus_owned_caches():
    session = BrowserSession()
    session._cached_browser_state_summary = "cached"
    session._cached_selector_map = {1: MagicMock()}
    session._dom_watchdog = MagicMock()

    session._focus_lost_locked(
        FocusTargetError(
            "No live page target was available.",
            code=FocusTargetCode.RECOVERY_FAILED,
            phase=FocusPhase.LOST,
            target_id="target-1",
            target_type="page",
        ),
        FocusChangeReason.CRASH_RECOVERY,
        previous_target_id="target-1",
    )

    assert session._cached_browser_state_summary is None
    assert session._cached_selector_map == {}
    session._dom_watchdog.clear_cache.assert_called_once()


async def test_focus_page_setup_failure_is_logged():
    session = BrowserSession()

    class FailingEvent:
        def __await__(self):
            yield
            return self

        async def event_result(self, *, raise_if_any, raise_if_none):
            if raise_if_any:
                raise RuntimeError("setup failed")
            return None

    session.event_bus.dispatch = MagicMock(return_value=FailingEvent())
    with patch("logging.Logger.warning") as warning:
        await session._initialize_focus_page("CREATED-TARGET-00", FocusTransitionBudget.start(1.0))

    warning.assert_called_once()
    assert "TabCreatedEvent setup failed" in warning.call_args.args[0]
    assert "setup failed" in warning.call_args.args[0]


async def test_service_switch_times_out_behind_queue_and_cannot_commit_late(monkeypatch):
    from bubus import BaseEvent

    from openbrowser.browser import focus as focus_module
    from openbrowser.tools.service import Tools

    session, _client, focus, manager, replacement = _focused_session("QUEUED-TARGET-0000", "queued-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    before = session.focus_snapshot()
    monkeypatch.setattr(focus_module, "FOCUS_TRANSITION_BUDGET_SECONDS", 0.05)

    class Blocker(BaseEvent):
        pass

    entered, release = asyncio.Event(), asyncio.Event()

    async def block(event: Blocker):
        entered.set()
        await release.wait()

    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    session.event_bus.on(Blocker, block)
    tools = Tools()
    tools.browser_session = session
    object.__setattr__(
        session,
        "get_target_id_from_tab_id",
        AsyncMock(return_value=replacement.target_id),
    )
    pending = session.event_bus.dispatch(Blocker())
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    try:
        result = await tools.registry.execute_action(
            "switch",
            {"tab_id": replacement.target_id[-4:]},
            browser_session=session,
        )
        assert result.success is False
        error = (result.error or "").lower()
        assert "timed out" in error
        assert "browser status" in error
        manager.get_live_session_for_target.assert_not_awaited()
        assert session.focus_snapshot() == before
        release.set()
        await pending
        await session.event_bus.wait_until_idle()
        assert session.focus_snapshot() == before
        assert session.agent_focus is focus
    finally:
        release.set()
        await _stop_bus(session)


async def test_close_through_the_serial_event_bus_does_not_deadlock():
    session = BrowserSession(browser_profile=BrowserProfile(cdp_url="ws://localhost:9222", is_local=False))
    target_id = "AAAA1111BBBB2222CCCC3333DDDD4444"
    session._tab_registry.register(target_id, state=TabState.OPEN)
    client = MagicMock()
    client.send.Target.closeTarget = AsyncMock(return_value={})
    session._cdp_client_root = client
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    try:
        event = session.event_bus.dispatch(
            CloseTabEvent(
                target_id=target_id,
                focus_deadline_monotonic=FocusTransitionBudget.start().deadline,
            )
        )
        await asyncio.wait_for(event, timeout=1.0)
        await event.event_result(timeout=0.5, raise_if_any=True, raise_if_none=False)

        record = session._tab_registry.get(target_id)
        assert record is not None
        assert record.state is TabState.CLOSED
    finally:
        await _stop_bus(session)


async def test_switch_action_reports_observation_failure_after_commit():
    session, _client, _focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()

    async def failing_observer(event: AgentFocusChangedEvent):
        raise RuntimeError("observer failed")

    session.event_bus.on(AgentFocusChangedEvent, failing_observer)
    object.__setattr__(
        session,
        "get_target_id_from_tab_id",
        AsyncMock(return_value=replacement.target_id),
    )
    try:
        result = await Tools().registry.execute_action(
            "switch",
            {"tab_id": replacement.target_id[-4:]},
            browser_session=session,
        )

        assert result.success is False
        assert "observation failed" in (result.error or "").lower()
        assert session.agent_focus is replacement
        assert session.focus_snapshot().target_id == replacement.target_id
    finally:
        await _stop_bus(session)


async def test_close_action_reports_focus_repair_failure_after_the_tab_closed():
    session, client, focus, manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    client.send.Target.closeTarget = AsyncMock(return_value={})
    manager.get_live_session_for_target = AsyncMock(return_value=None)
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session._cdp_get_all_pages = AsyncMock(return_value=[])
    session._cdp_create_new_page = AsyncMock(side_effect=RuntimeError("no page"))
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    object.__setattr__(
        session,
        "get_target_id_from_tab_id",
        AsyncMock(return_value=focus.target_id),
    )
    try:
        result = await Tools().registry.execute_action(
            "close",
            {"tab_id": focus.target_id[-4:]},
            browser_session=session,
        )

        assert result.success is False
        error = (result.error or "").lower()
        assert "closed" in error
        assert "focus repair failed" in error
        assert "failed to close tab" not in error
        assert session.focus_snapshot().phase is FocusPhase.LOST
        client.send.Target.closeTarget.assert_awaited_once_with(params={"targetId": focus.target_id})
        record = session._tab_registry.get(focus.target_id)
        assert record is not None
        assert record.state is TabState.CLOSED
    finally:
        await _stop_bus(session)


async def test_close_action_reports_a_post_close_observation_failure_truthfully():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    client.send.Target.closeTarget = AsyncMock(return_value={})
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]
    )
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()

    async def failing_observer(event: AgentFocusChangedEvent):
        raise RuntimeError("observer failed")

    session.event_bus.on(AgentFocusChangedEvent, failing_observer)
    object.__setattr__(
        session,
        "get_target_id_from_tab_id",
        AsyncMock(return_value=focus.target_id),
    )
    try:
        result = await Tools().registry.execute_action(
            "close",
            {"tab_id": focus.target_id[-4:]},
            browser_session=session,
        )

        assert result.success is False
        error = (result.error or "").lower()
        assert "closed" in error
        assert "focus observation failed" in error
        assert "failed to close tab" not in error
        record = session._tab_registry.get(focus.target_id)
        assert record is not None and record.state is TabState.CLOSED
        assert session.agent_focus is replacement
    finally:
        await _stop_bus(session)


async def test_close_timeout_reconciles_with_a_separate_cleanup_budget():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(
        side_effect=lambda target_id, *, budget=None: replacement if target_id == replacement.target_id else None
    )
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]
    )
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    client.send.Target.getTargets = AsyncMock(
        return_value={
            "targetInfos": [
                {"targetId": replacement.target_id, "type": "page", "url": replacement.url, "title": replacement.title}
            ]
        }
    )
    close_entered = asyncio.Event()
    liveness_budgets: list[FocusTransitionBudget] = []
    actual_liveness = session._target_liveness

    async def delayed_close(**kwargs):
        close_entered.set()
        await asyncio.sleep(0.15)
        raise TimeoutError

    async def record_liveness(target_id, budget):
        liveness_budgets.append(budget)
        return await actual_liveness(target_id, budget)

    client.send.Target.closeTarget = AsyncMock(side_effect=delayed_close)
    event = CloseTabEvent(
        target_id=focus.target_id,
        focus_deadline_monotonic=asyncio.get_running_loop().time() + 0.1,
    )
    with patch.object(session, "_target_liveness", side_effect=record_liveness):
        with pytest.raises(FocusTargetError):
            await session.on_CloseTabEvent(event)
        await _drain_focus_tasks(session)

    assert close_entered.is_set()
    assert liveness_budgets
    assert liveness_budgets[0].deadline > event.focus_deadline_monotonic
    record = session._tab_registry.get(focus.target_id)
    assert record is not None and record.state is TabState.CLOSED
    assert session.agent_focus is replacement
    assert session.focus_snapshot().target_id == replacement.target_id


async def test_unknown_close_clears_focus_within_cleanup_budget_under_lock_contention(monkeypatch):
    from openbrowser.browser import session as session_module

    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session._tab_registry.mark_closing(focus.target_id)
    monkeypatch.setattr(session_module, "CLOSE_RECONCILIATION_SECONDS", 0.1)

    async def slow_unknown_liveness(target_id, budget):
        await asyncio.sleep(0.06)
        return None

    await session._focus_lock.acquire()
    started = asyncio.get_running_loop().time()
    with patch.object(session, "_target_liveness", side_effect=slow_unknown_liveness):
        session._schedule_close_reconciliation(focus.target_id)
        await asyncio.sleep(0.04)
        session._focus_lock.release()
        await _drain_focus_tasks(session)

    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.5
    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.LOST
    assert snapshot.previous_target_id == focus.target_id
    record = session._tab_registry.get(focus.target_id)
    assert record is not None and record.state is TabState.CLOSING


async def test_queued_close_reconciliation_does_not_clear_a_reopened_target():
    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)

    with patch.object(session, "_target_liveness", AsyncMock(return_value=None)):
        session._schedule_close_reconciliation(focus.target_id)
        await _drain_focus_tasks(session)

    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.FOCUSED
    assert snapshot.target_id == focus.target_id
    assert session.agent_focus is focus


async def test_cleanup_rechecks_reopened_tab_after_waiting_for_focus_lock():
    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session._tab_registry.mark_closing(focus.target_id)
    await session._focus_lock.acquire()
    with patch.object(session, "_target_liveness", AsyncMock(return_value=None)):
        session._schedule_close_reconciliation(focus.target_id)
        for _attempt in range(100):
            if getattr(session._focus_lock, "_waiters", ()):
                break
            await asyncio.sleep(0)
        else:
            session._focus_lock.release()
            raise AssertionError("Cleanup never waited for the focus lock")
        session._tab_registry.update(focus.target_id, state=TabState.OPEN, allow_reopen=True)
        session._focus_lock.release()
        await _drain_focus_tasks(session)

    snapshot = session.focus_snapshot()
    assert snapshot.phase is FocusPhase.FOCUSED
    assert snapshot.target_id == focus.target_id
    assert session.agent_focus is focus


async def test_stale_close_reconciliation_generation_cannot_clear_new_focus():
    session, _client, focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session._tab_registry.mark_closing(focus.target_id)
    before = session.focus_snapshot()

    with patch.object(session, "_target_liveness", AsyncMock(return_value=None)):
        session._schedule_close_reconciliation(
            focus.target_id,
            expected_generation=before.generation - 1,
        )
        await _drain_focus_tasks(session)

    assert session.focus_snapshot() == before
    assert session.agent_focus is focus


async def test_cancelled_close_returns_before_cleanup_and_uses_a_separate_budget():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    manager.get_live_session_for_target = AsyncMock(
        side_effect=lambda target_id, *, budget=None: replacement if target_id == replacement.target_id else None
    )
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]
    )
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    cleanup_budgets: list[FocusTransitionBudget] = []
    repair_called = asyncio.Event()
    reconciliation_started = asyncio.Event()
    release_reconciliation = asyncio.Event()

    async def liveness(target_id, budget):
        reconciliation_started.set()
        await release_reconciliation.wait()
        cleanup_budgets.append(budget)
        return False

    async def slow_repair(target_id, budget):
        repair_called.set()
        return session.focus_snapshot()

    client.send.Target.closeTarget = AsyncMock(side_effect=asyncio.CancelledError)
    event = CloseTabEvent(
        target_id=focus.target_id,
        focus_deadline_monotonic=asyncio.get_running_loop().time() + 1.0,
    )
    with (
        patch.object(session, "_target_liveness", side_effect=liveness),
        patch.object(session, "_repair_focus_after_closed_tab", side_effect=slow_repair),
    ):
        try:
            await session.on_CloseTabEvent(event)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Close cancellation did not propagate")
        await asyncio.wait_for(reconciliation_started.wait(), timeout=1.0)
        assert release_reconciliation.is_set() is False
        release_reconciliation.set()
        await _drain_focus_tasks(session)
    assert cleanup_budgets
    assert repair_called.is_set()
    assert any(budget.deadline > event.focus_deadline_monotonic for budget in cleanup_budgets)
    assert session.focus_snapshot().phase is FocusPhase.LOST
    record = session._tab_registry.get(focus.target_id)
    assert record is not None and record.state is TabState.CLOSED


async def test_action_tab_resolution_consumes_the_focus_budget(monkeypatch):
    from openbrowser.browser import focus as focus_module

    session, _client, _focus, _manager, _replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    monkeypatch.setattr(focus_module, "FOCUS_TRANSITION_BUDGET_SECONDS", 0.02)

    async def slow_resolution(tab_id):
        await asyncio.sleep(0.05)
        return "PAGE-TARGET-000000"

    object.__setattr__(session, "get_target_id_from_tab_id", AsyncMock(side_effect=slow_resolution))
    tools = Tools()
    tools.browser_session = session
    for action in ("switch", "close"):
        result = await tools.registry.execute_action(
            action,
            {"tab_id": "0000"},
            browser_session=session,
        )
        assert result.success is False
        assert "timed out" in (result.error or "").lower()


async def test_close_action_reports_post_close_observer_failure_truthfully():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    client.send.Target.closeTarget = AsyncMock(return_value={})
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]
    )
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()

    observer_called = asyncio.Event()

    async def failing_observer(event: TabClosedEvent):
        observer_called.set()
        raise RuntimeError("post-close observer failed")

    session.event_bus.on(TabClosedEvent, failing_observer)
    object.__setattr__(
        session,
        "get_target_id_from_tab_id",
        AsyncMock(return_value=focus.target_id),
    )
    try:
        result = await Tools().registry.execute_action(
            "close",
            {"tab_id": focus.target_id[-4:]},
            browser_session=session,
        )

        assert result.success is False
        error = (result.error or "").lower()
        assert "closed" in error
        assert "post-close" in error
        assert "post-close observer failed" in error
        assert "failed to close tab" not in error
        assert observer_called.is_set()
        record = session._tab_registry.get(focus.target_id)
        assert record is not None and record.state is TabState.CLOSED
        assert session.agent_focus is replacement
    finally:
        await _stop_bus(session)


async def test_close_timeout_with_confirmed_absence_repairs_focus():
    session, client, focus, manager, replacement = _focused_session("PAGE-TARGET-000000", "page-session")
    client.send.Target.closeTarget = AsyncMock(side_effect=TimeoutError)
    manager.get_live_session_for_target = AsyncMock(return_value=replacement)
    session._cdp_get_all_pages = AsyncMock(
        return_value=[{"targetId": replacement.target_id, "type": "page", "url": replacement.url}]
    )
    session._tab_registry.register(focus.target_id, state=TabState.OPEN)
    session.event_bus = EventBus()
    session._core_handlers_attached = False
    session._attach_core_handlers()
    object.__setattr__(
        session,
        "get_target_id_from_tab_id",
        AsyncMock(return_value=focus.target_id),
    )
    try:
        with patch.object(session, "_target_liveness", AsyncMock(return_value=False)):
            result = await Tools().registry.execute_action(
                "close",
                {"tab_id": focus.target_id[-4:]},
                browser_session=session,
            )
        await _drain_focus_tasks(session)

        assert result.success is False
        error = (result.error or "").lower()
        assert "timed out" in error
        assert "closed" in error
        assert "failed to close tab" not in error
        record = session._tab_registry.get(focus.target_id)
        assert record is not None and record.state is TabState.CLOSED
        assert session.agent_focus is replacement
    finally:
        await _stop_bus(session)
