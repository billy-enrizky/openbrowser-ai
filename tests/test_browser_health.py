"""Behavior tests for explicit browser health state."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from openbrowser.browser.focus import FocusPhase
from openbrowser.browser.health import (
    BrowserCrashToken,
    BrowserHealthMachine,
    BrowserHealthPhase,
    BrowserHealthReason,
    FocusRecoveryOutcome,
    OwnedRootBinding,
    canonical_cdp_endpoint,
)
from openbrowser.browser.lifecycle import LifecyclePhase
from openbrowser.browser.profile import BrowserProfile
from openbrowser.browser.session import BrowserSession
from openbrowser.browser.watchdogs.profile_lease import ProfileLease


def test_initial_snapshot_is_complete() -> None:
    snapshot = BrowserHealthMachine().snapshot().to_dict()

    assert snapshot == {
        "phase": "unknown",
        "healthy": None,
        "reason": "initial",
        "connection_generation": 0,
        "process_present": None,
        "process_pid": None,
        "target_crash_count": 0,
        "last_target_crash_at": None,
        "last_target_crash_target_id": None,
        "last_target_crash_detail": None,
        "last_target_crash_generation": None,
        "last_change_at": None,
        "last_focus_recovery_outcome": "not_attempted",
        "last_focus_recovery_target_id": None,
        "last_focus_recovery_at": None,
        "process_checked_at": None,
    }


def test_connect_success_records_connected_generation() -> None:
    health = BrowserHealthMachine()

    health.record_connected(7)
    snapshot = health.snapshot().to_dict()

    assert snapshot["phase"] == "healthy"
    assert snapshot["healthy"] is True
    assert snapshot["reason"] == "connected"
    assert snapshot["connection_generation"] == 7


def test_same_generation_crash_during_setup_survives_connect_success() -> None:
    health = BrowserHealthMachine()

    token = health.record_target_crash(4, "target-a", "status=crashed")
    health.record_connected(4)
    snapshot = health.snapshot().to_dict()

    assert isinstance(token, BrowserCrashToken)
    assert snapshot["phase"] == "degraded"
    assert snapshot["reason"] == "target_crash"
    assert snapshot["target_crash_count"] == 1


def test_immediate_duplicate_crash_has_no_second_health_effect_and_aba_counts_three() -> None:
    health = BrowserHealthMachine()

    first = health.record_target_crash(1, "target-a", "first")
    duplicate = health.record_target_crash(1, "target-a", "duplicate")
    second = health.record_target_crash(1, "target-b", "second")
    third = health.record_target_crash(1, "target-a", "third")

    assert first is not None
    assert duplicate is None
    assert second is not None
    assert third is not None
    assert health.snapshot().target_crash_count == 3


def test_older_connection_generation_crash_is_ignored_by_the_machine() -> None:
    health = BrowserHealthMachine()
    health.record_connected(2)

    assert health.record_target_crash(1, "target-old", "stale") is None
    assert health.snapshot().target_crash_count == 0


def test_failed_phase_preserves_failure_while_recording_crash_history() -> None:
    health = BrowserHealthMachine()

    health.record_cleanup_failed(2)
    token = health.record_target_crash(2, "target-a", "status=crashed")
    snapshot = health.snapshot().to_dict()

    assert token is not None
    assert snapshot["phase"] == "failed"
    assert snapshot["reason"] == "cleanup_failed"
    assert snapshot["target_crash_count"] == 1
    assert snapshot["last_focus_recovery_outcome"] == "not_attempted"


def test_recovery_outcome_lifecycle_and_stale_completion() -> None:
    health = BrowserHealthMachine()
    health.record_connected(1)
    token = health.record_target_crash(1, "target-a", "first")
    assert token is not None

    assert health.record_focus_recovery_started(token, 1) is True
    pending = health.snapshot().to_dict()
    assert pending["last_focus_recovery_outcome"] == "pending"
    assert pending["last_focus_recovery_at"] is not None

    newer = health.record_target_crash(1, "target-b", "second")
    assert newer is not None
    stale_time = health.snapshot().last_focus_recovery_at
    assert health.record_focus_recovery_success(token, 1, "replacement-a") is False
    after_stale = health.snapshot().to_dict()
    assert after_stale["phase"] == "degraded"
    assert after_stale["last_target_crash_target_id"] == "target-b"
    assert after_stale["last_focus_recovery_outcome"] == "not_attempted"
    assert after_stale["last_focus_recovery_target_id"] is None
    assert after_stale["last_focus_recovery_at"] == stale_time

    assert health.record_focus_recovery_started(newer, 1) is True
    assert health.record_focus_recovery_success(newer, 1, "replacement-b") is True
    recovered = health.snapshot().to_dict()
    assert recovered["phase"] == "healthy"
    assert recovered["reason"] == "focus_recovered"
    assert recovered["last_focus_recovery_outcome"] == "succeeded"
    assert recovered["last_focus_recovery_target_id"] == "replacement-b"
    assert recovered["target_crash_count"] == 2


def test_failed_phase_records_recovery_outcome_without_hiding_failure() -> None:
    health = BrowserHealthMachine()
    health.record_connect_failed(3)
    token = health.record_target_crash(3, "target-a", "status=crashed")
    assert token is not None

    assert health.record_focus_recovery_started(token, 3) is True
    assert health.record_focus_recovery_failure(token, 3) is True
    snapshot = health.snapshot().to_dict()

    assert snapshot["phase"] == "failed"
    assert snapshot["reason"] == "connect_failed"
    assert snapshot["last_focus_recovery_outcome"] == "failed"


def test_newer_connection_generation_clears_pending_recovery() -> None:
    health = BrowserHealthMachine()
    health.record_connected(1)
    token = health.record_target_crash(1, "target-a", "status=crashed")
    assert token is not None
    assert health.record_focus_recovery_started(token, 1) is True

    health.record_connect_failed(2)
    snapshot = health.snapshot()

    assert snapshot.phase is BrowserHealthPhase.FAILED
    assert snapshot.connection_generation == 2
    assert snapshot.last_focus_recovery_outcome is FocusRecoveryOutcome.NOT_ATTEMPTED
    assert snapshot.last_focus_recovery_target_id is None
    assert snapshot.last_focus_recovery_at is None


def test_target_crash_detail_is_sanitized_and_bounded_in_public_snapshot() -> None:
    health = BrowserHealthMachine()
    health.record_connected(1)
    health.record_target_crash(
        1,
        "target-a",
        "Authorization: Bearer crash-secret https://example.test/private " + "x" * 400,
    )

    snapshot = health.snapshot().to_dict()

    assert snapshot["last_target_crash_detail"] is not None
    assert len(snapshot["last_target_crash_detail"]) <= 240
    assert "crash-secret" not in snapshot["last_target_crash_detail"]
    assert "https://" not in snapshot["last_target_crash_detail"]


def test_startup_failure_does_not_replace_connect_failure_or_stopped() -> None:
    health = BrowserHealthMachine()
    health.record_connect_failed(5)

    assert health.record_startup_failed(5) is False
    assert health.snapshot().reason == BrowserHealthReason.CONNECT_FAILED

    health.record_stopped(BrowserHealthReason.STOP, 5)
    assert health.record_startup_failed(5) is False
    assert health.snapshot().phase is BrowserHealthPhase.STOPPED


def test_startup_failure_replaces_a_previous_cleanup_failure() -> None:
    health = BrowserHealthMachine()
    health.record_cleanup_failed(4)

    assert health.record_startup_failed(4) is True
    snapshot = health.snapshot()
    assert snapshot.phase is BrowserHealthPhase.FAILED
    assert snapshot.reason is BrowserHealthReason.STARTUP_FAILED


def test_reset_and_stop_clear_recovery_fields_but_keep_crash_history() -> None:
    health = BrowserHealthMachine()
    health.record_connected(2)
    token = health.record_target_crash(2, "target-a", "status=crashed")
    assert token is not None
    assert health.record_focus_recovery_started(token, 2) is True
    assert health.record_focus_recovery_success(token, 2, "replacement") is True

    health.record_stopped(BrowserHealthReason.RESET, 2)
    snapshot = health.snapshot().to_dict()

    assert snapshot["phase"] == "stopped"
    assert snapshot["healthy"] is None
    assert snapshot["reason"] == "reset"
    assert snapshot["target_crash_count"] == 1
    assert snapshot["last_target_crash_target_id"] == "target-a"
    assert snapshot["last_focus_recovery_outcome"] == "not_attempted"
    assert snapshot["last_focus_recovery_target_id"] is None
    assert snapshot["last_focus_recovery_at"] is None


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("WS://Example.COM:80/devtools/browser/id/", "ws://example.com/devtools/browser/id"),
        ("wss://Example.COM:443/devtools/browser/id", "wss://example.com/devtools/browser/id"),
        ("ws://[0:0:0:0:0:0:0:1]:9222/devtools/browser/id", "ws://[::1]:9222/devtools/browser/id"),
        ("ws://host:9222/devtools/browser/id#one", "ws://host:9222/devtools/browser/id#two"),
    ],
)
def test_canonical_endpoint_equivalence(left: str, right: str) -> None:
    assert canonical_cdp_endpoint(left) == canonical_cdp_endpoint(right)


def test_canonical_endpoint_preserves_query_and_rejects_user_info() -> None:
    assert canonical_cdp_endpoint("ws://host:9222/path?token=a") != canonical_cdp_endpoint(
        "ws://host:9222/path?token=b"
    )
    assert canonical_cdp_endpoint("ws://user:pass@host:9222/path") is None


def _bound_session(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    endpoint = "ws://127.0.0.1:9222/devtools/browser/fixture"
    session = BrowserSession(
        browser_profile=BrowserProfile(cdp_url=endpoint, is_local=True, user_data_dir=str(profile))
    )
    root = MagicMock()
    websocket = SimpleNamespace()
    transport = SimpleNamespace()
    websocket.transport = transport
    root.ws = websocket
    session._cdp_client_root = root
    generation = session._cdp_health.begin_connect()
    session._cdp_health.mark_connected(generation)
    record = {
        "pid": 4321,
        "start_time": 123.0,
        "profile_dir": str(profile.resolve()),
        "instance_id": "instance-1",
        "ownership_marker": "--openbrowser-instance-id=instance-1",
        "executable": "/Applications/Chrome",
        "cdp_port": 9222,
    }
    process = SimpleNamespace(pid=4321, create_time=lambda: 123.0, net_connections=MagicMock(return_value=[]))
    lease = SimpleNamespace(profile_dir=profile.resolve(), instance_id="instance-1")
    watchdog = SimpleNamespace(
        _owns_browser_resources=True,
        _browser_record=record,
        _profile_lease=lease,
        _instance_id="instance-1",
        _subprocess=process,
    )
    session._local_browser_watchdog = watchdog
    session._health_owned_root_binding = OwnedRootBinding(
        connection_generation=generation,
        endpoint=canonical_cdp_endpoint(endpoint) or endpoint,
        pid=4321,
        start_time=123.0,
        profile_dir=str(profile.resolve()),
        instance_id="instance-1",
        ownership_marker="--openbrowser-instance-id=instance-1",
        executable="/Applications/Chrome",
        cdp_port=9222,
        root=root,
        websocket=websocket,
        transport=transport,
    )
    session._health.record_connected(generation)
    return session, root, websocket, transport, record, process, lease, watchdog


def test_bound_process_match_preserves_phase(tmp_path) -> None:
    session, *_ = _bound_session(tmp_path)

    with patch.object(ProfileLease, "process_identity_status", return_value="match"):
        health = session.browser_health_snapshot()

    assert health["phase"] == "healthy"
    assert health["process_present"] is True
    assert health["process_pid"] == 4321
    assert health["process_checked_at"] is not None


@pytest.mark.parametrize("identity", ["missing", "mismatch"])
def test_missing_or_reused_bound_process_derives_crashed(tmp_path, identity) -> None:
    session, *_ = _bound_session(tmp_path)
    recorded_change = session.browser_health_snapshot()["last_change_at"]

    with patch.object(ProfileLease, "process_identity_status", return_value=identity):
        health = session.browser_health_snapshot()

    assert health["phase"] == "crashed"
    assert health["healthy"] is False
    assert health["reason"] == "process_exit"
    assert health["process_present"] is False
    assert health["process_checked_at"] is not None
    assert health["last_change_at"] == recorded_change


def test_unknown_process_identity_is_not_a_crash(tmp_path) -> None:
    session, *_ = _bound_session(tmp_path)

    with patch.object(ProfileLease, "process_identity_status", return_value="unknown"):
        health = session.browser_health_snapshot()

    assert health["phase"] == "healthy"
    assert health["process_present"] is None
    assert health["reason"] == "connected"


def test_process_identity_inspection_error_is_unknown_not_a_crash(tmp_path) -> None:
    session, *_ = _bound_session(tmp_path)

    with patch.object(ProfileLease, "process_identity_status", side_effect=psutil.AccessDenied(4321)):
        health = session.browser_health_snapshot()

    assert health["phase"] == "healthy"
    assert health["process_present"] is None
    assert health["reason"] == "connected"


def test_closed_socket_after_owner_exit_still_derives_crashed(tmp_path) -> None:
    session, _root, _websocket, transport, *_ = _bound_session(tmp_path)
    transport.get_extra_info = MagicMock(side_effect=ConnectionError("closed"))

    with patch.object(ProfileLease, "process_identity_status", return_value="missing"):
        health = session.browser_health_snapshot()

    assert health["phase"] == "crashed"
    assert health["process_present"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "root",
        "websocket",
        "transport",
        "endpoint",
        "record",
        "pid",
        "start_time",
        "generation",
        "marker",
        "executable",
        "profile",
        "instance",
        "lease_profile",
        "lease",
    ],
)
def test_stale_binding_returns_unknown_without_deriving_crash(tmp_path, mutation) -> None:
    session, root, websocket, transport, record, _process, lease, _watchdog = _bound_session(tmp_path)
    if mutation == "root":
        session._cdp_client_root = MagicMock()
    elif mutation == "websocket":
        root.ws = SimpleNamespace(transport=transport)
    elif mutation == "transport":
        websocket.transport = SimpleNamespace()
    elif mutation == "endpoint":
        session.browser_profile.cdp_url = "ws://127.0.0.1:9222/devtools/browser/other"
    elif mutation == "record":
        record["cdp_port"] = 9333
    elif mutation == "pid":
        record["pid"] = 5432
    elif mutation == "start_time":
        record["start_time"] = 124.0
    elif mutation == "generation":
        session._cdp_health.begin_connect()
    elif mutation == "marker":
        record["ownership_marker"] = "--openbrowser-instance-id=other"
    elif mutation == "executable":
        record["executable"] = "/Applications/Other"
    elif mutation == "profile":
        record["profile_dir"] = str((tmp_path / "other-profile").resolve())
    elif mutation == "instance":
        record["instance_id"] = "instance-2"
    elif mutation == "lease_profile":
        lease.profile_dir = (tmp_path / "other-profile").resolve()
    elif mutation == "lease":
        lease.instance_id = "instance-2"

    with patch.object(ProfileLease, "process_identity_status", return_value="missing"):
        health = session.browser_health_snapshot()

    assert health["phase"] == "healthy"
    assert health["reason"] == "connected"
    assert health["process_present"] is None
    assert health["process_pid"] is None
    assert health["process_checked_at"] is None


def test_same_port_different_host_is_not_bound(tmp_path) -> None:
    session, *_ = _bound_session(tmp_path)
    session.browser_profile.cdp_url = "ws://remote.invalid:9222/devtools/browser/fixture"

    with patch.object(ProfileLease, "process_identity_status", return_value="missing") as identity:
        health = session.browser_health_snapshot()

    identity.assert_not_called()
    assert health["process_present"] is None
    assert health["phase"] == "healthy"


def test_keepalive_stopped_root_retirement_has_no_binding(tmp_path) -> None:
    session, *_ = _bound_session(tmp_path)
    generation = session._cdp_health.connection_generation
    session._cdp_client_root = None
    session._health.record_stopped(BrowserHealthReason.RESET, generation)

    health = session.browser_health_snapshot()

    assert health["phase"] == "stopped"
    assert health["process_present"] is None
    assert health["process_pid"] is None
    assert health["process_checked_at"] is None


def test_stopping_lifecycle_reports_unknown_process_without_crash(tmp_path) -> None:
    session, *_ = _bound_session(tmp_path)
    session._lifecycle.transition(LifecyclePhase.STOPPING, reason="test stop")

    with patch.object(ProfileLease, "process_identity_status", return_value="missing") as identity:
        health = session.browser_health_snapshot()

    identity.assert_not_called()
    assert health["phase"] == "healthy"
    assert health["process_present"] is None
    assert health["reason"] == "connected"


def test_direct_connect_at_stopped_and_retained_root_at_failed_derive_crashed(tmp_path) -> None:
    direct_root = tmp_path / "direct"
    direct_root.mkdir()
    direct, *_ = _bound_session(direct_root)
    with patch.object(ProfileLease, "process_identity_status", return_value="missing"):
        direct_health = direct.browser_health_snapshot()
    assert direct.lifecycle.phase is LifecyclePhase.STOPPED
    assert direct_health["phase"] == "crashed"

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed, *_ = _bound_session(failed_root)
    failed._lifecycle.transition(LifecyclePhase.STARTING)
    failed._lifecycle.transition(LifecyclePhase.READY)
    failed._lifecycle.fail(RuntimeError("cleanup failed"), reason="cleanup")
    with patch.object(ProfileLease, "process_identity_status", return_value="missing"):
        failed_health = failed.browser_health_snapshot()
    assert failed_health["phase"] == "crashed"


def test_establish_binding_uses_one_synchronous_process_socket_proof(tmp_path) -> None:
    session, root, _websocket, _transport, record, process, *_ = _bound_session(tmp_path)
    session._health_owned_root_binding = None
    connection_socket = SimpleNamespace(
        family=socket.AF_INET,
        getpeername=lambda: ("127.0.0.1", 9222),
        getsockname=lambda: ("127.0.0.1", 5555),
    )
    transport = SimpleNamespace(get_extra_info=lambda name: connection_socket if name == "socket" else None)
    websocket = SimpleNamespace(transport=transport)
    root.ws = websocket
    process.net_connections = MagicMock(
        return_value=[
            SimpleNamespace(family=socket.AF_INET, status=psutil.CONN_LISTEN, laddr=("127.0.0.1", 9222), raddr=()),
            SimpleNamespace(
                family=socket.AF_INET,
                status=psutil.CONN_ESTABLISHED,
                laddr=("127.0.0.1", 9222),
                raddr=("127.0.0.1", 5555),
            ),
        ]
    )

    with patch.object(ProfileLease, "process_matches_browser", return_value=True):
        binding = session._establish_owned_root_binding(session._cdp_health.connection_generation)

    assert binding is not None
    assert binding.root is root
    assert binding.websocket is websocket
    assert binding.transport is transport
    assert binding.endpoint == canonical_cdp_endpoint(session.cdp_url)
    assert binding.ownership_marker == record["ownership_marker"]
    process.net_connections.assert_called_once_with(kind="tcp")


@pytest.mark.parametrize("mode", ["remote_peer", "missing_accept", "access_denied"])
def test_establish_binding_rejects_unproven_socket_ownership(tmp_path, mode) -> None:
    session, root, _websocket, _transport, _record, process, *_ = _bound_session(tmp_path)
    session._health_owned_root_binding = None
    peer = ("203.0.113.5", 9222) if mode == "remote_peer" else ("127.0.0.1", 9222)
    connection_socket = SimpleNamespace(
        family=socket.AF_INET,
        getpeername=lambda: peer,
        getsockname=lambda: ("127.0.0.1", 5555),
    )
    transport = SimpleNamespace(get_extra_info=lambda name: connection_socket if name == "socket" else None)
    root.ws = SimpleNamespace(transport=transport)
    connections = [
        SimpleNamespace(family=socket.AF_INET, status=psutil.CONN_LISTEN, laddr=("127.0.0.1", 9222), raddr=())
    ]
    if mode != "missing_accept":
        connections.append(
            SimpleNamespace(
                family=socket.AF_INET,
                status=psutil.CONN_ESTABLISHED,
                laddr=("127.0.0.1", 9222),
                raddr=("127.0.0.1", 5555),
            )
        )
    process.net_connections = MagicMock(return_value=connections)
    if mode == "access_denied":
        process.net_connections.side_effect = psutil.AccessDenied(4321)

    with patch.object(ProfileLease, "process_matches_browser", return_value=True):
        binding = session._establish_owned_root_binding(session._cdp_health.connection_generation)

    assert binding is None


@pytest.mark.asyncio
async def test_connect_preflight_rejection_leaves_health_unknown() -> None:
    session = BrowserSession()
    session._cdp_client_root = MagicMock()

    with pytest.raises(RuntimeError, match="already exists"):
        await session.connect()

    health = session.browser_health_snapshot()
    assert health["phase"] == "unknown"
    assert health["reason"] == "initial"


@pytest.mark.asyncio
async def test_start_preflight_rejection_leaves_health_unknown() -> None:
    session = BrowserSession()
    session._cdp_health.mark_cleanup_failure(RuntimeError("retained cleanup"))

    with pytest.raises(RuntimeError, match="cleanup"):
        await session.start()

    health = session.browser_health_snapshot()
    assert health["phase"] == "unknown"
    assert health["reason"] == "initial"


@pytest.mark.asyncio
async def test_connect_failure_records_connect_failed() -> None:
    session = BrowserSession(browser_profile=BrowserProfile(cdp_url="ws://127.0.0.1:9222", is_local=False))
    root = MagicMock()
    root.start = AsyncMock(side_effect=RuntimeError("connect failed"))
    root.stop = AsyncMock()

    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        pytest.raises(RuntimeError, match="Failed to establish CDP connection"),
    ):
        await session.connect()

    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "connect_failed"


@pytest.mark.asyncio
async def test_start_cancellation_before_connect_records_startup_failed() -> None:
    session = BrowserSession()

    async def hanging() -> None:
        await asyncio.Event().wait()

    session.event_bus.dispatch = MagicMock(return_value=hanging())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(session.start(), timeout=0.05)

    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "startup_failed"


def _failing_start_event():
    class FailingStartEvent:
        def __await__(self):
            async def done():
                return None

            return done().__await__()

        async def event_result(self, **_kwargs):
            raise RuntimeError("start publication failed")

    return FailingStartEvent()


@pytest.mark.asyncio
async def test_start_wrapped_connect_failure_keeps_connect_failed() -> None:
    session = BrowserSession()
    generation = session._cdp_health.begin_connect()
    session._health.record_connect_failed(generation)
    session.event_bus.dispatch = MagicMock(return_value=_failing_start_event())

    with pytest.raises(RuntimeError, match="start publication failed"):
        await session.start()

    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "connect_failed"


@pytest.mark.asyncio
async def test_failure_after_connect_before_ready_records_startup_failed() -> None:
    session = BrowserSession()
    session._cdp_client_root = MagicMock()
    generation = session._cdp_health.begin_connect()
    session._cdp_health.mark_connected(generation)
    session._health.record_connected(generation)
    session.event_bus.dispatch = MagicMock(return_value=_failing_start_event())

    with pytest.raises(RuntimeError, match="start publication failed"):
        await session.start()

    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "startup_failed"


@pytest.mark.asyncio
async def test_retry_start_failure_replaces_previous_cleanup_failure() -> None:
    session = BrowserSession()
    session._health.record_cleanup_failed(0)
    session.event_bus.dispatch = MagicMock(return_value=_failing_start_event())

    with pytest.raises(RuntimeError, match="start publication failed"):
        await session.start()

    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "startup_failed"


@pytest.mark.asyncio
async def test_reset_records_reset_and_repeated_reset_updates_reason() -> None:
    session = BrowserSession()

    await session.reset()
    first = session.browser_health_snapshot()
    assert first["phase"] == "stopped"
    assert first["reason"] == "reset"

    await session.reset()
    second = session.browser_health_snapshot()
    assert second["phase"] == "stopped"
    assert second["reason"] == "reset"


@pytest.mark.asyncio
async def test_stop_event_bus_retirement_failure_records_cleanup_failed() -> None:
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)

    class ReadyEvent:
        def __await__(self):
            async def ready():
                return None

            return ready().__await__()

    class StopEvent(ReadyEvent):
        def event_result(self, **_kwargs):
            async def advance_generation():
                session._lifecycle.seed_generation(session._lifecycle.generation + 1)

            return advance_generation()

    session.event_bus.dispatch = MagicMock(side_effect=[ReadyEvent(), StopEvent()])
    with (
        patch.object(
            session,
            "_retire_event_bus",
            AsyncMock(side_effect=RuntimeError("bus retirement failed")),
        ),
        pytest.raises(RuntimeError, match="bus retirement failed"),
    ):
        await session._shutdown_locked(force=False)

    assert session.lifecycle.phase is LifecyclePhase.FAILED
    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "cleanup_failed"


@pytest.mark.asyncio
async def test_noop_stop_and_kill_leave_health_unchanged_after_reset() -> None:
    session = BrowserSession()
    await session.reset()
    before = session.browser_health_snapshot()

    await session.stop()
    after_stop = session.browser_health_snapshot()
    await session.kill()
    after_kill = session.browser_health_snapshot()

    assert after_stop == before
    assert after_kill == before


@pytest.mark.asyncio
async def test_kill_retries_pending_cleanup_failure_into_health() -> None:
    session = BrowserSession()
    session._cdp_client_root = MagicMock()
    session._cdp_health.mark_cleanup_failure(RuntimeError("retained cleanup"))

    with (
        patch.object(
            session,
            "_retry_pending_cdp_cleanup",
            AsyncMock(return_value=RuntimeError("cleanup retry failed")),
        ),
        pytest.raises(RuntimeError, match="retained CDP resources"),
    ):
        await session.kill()

    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "cleanup_failed"
    assert session.cdp_connection_health().cleanup_pending is True
    assert session.lifecycle.phase is LifecyclePhase.FAILED
    with pytest.raises(RuntimeError, match="cleanup"):
        await session.start()


@pytest.mark.asyncio
async def test_kill_outer_force_failure_records_cleanup_failed() -> None:
    session = BrowserSession()
    session._lifecycle.transition(LifecyclePhase.STARTING)
    session._lifecycle.transition(LifecyclePhase.READY)
    session._local_browser_watchdog = SimpleNamespace(
        _subprocess=MagicMock(),
        _spawned_subprocess=None,
        _profile_lease=MagicMock(),
    )

    with (
        patch.object(session, "_shutdown_locked", AsyncMock()),
        patch.object(
            session,
            "_force_stop_retained_browser",
            AsyncMock(return_value=RuntimeError("force failed")),
        ),
        pytest.raises(RuntimeError, match="force failed"),
    ):
        await session.kill()

    health = session.browser_health_snapshot()
    assert health["phase"] == "failed"
    assert health["reason"] == "cleanup_failed"
    assert session.cdp_connection_health().cleanup_pending is True
    assert session.lifecycle.phase is LifecyclePhase.FAILED
    with pytest.raises(RuntimeError, match="cleanup"):
        await session.start()


def test_lifecycle_snapshot_carries_health_without_private_endpoint(tmp_path) -> None:
    session, *_ = _bound_session(tmp_path)
    with patch.object(ProfileLease, "process_identity_status", return_value="match"):
        snapshot = session.lifecycle_snapshot()

    assert snapshot["health"]["phase"] == "healthy"
    assert snapshot["health"]["process_present"] is True
    assert "ws://" not in json.dumps(snapshot["health"])


def test_focus_phase_import_is_available_for_manager_tests() -> None:
    assert FocusPhase.UNINITIALIZED.value == "uninitialized"


def test_recovery_outcome_vocabulary_includes_pending() -> None:
    assert {item.value for item in FocusRecoveryOutcome} == {
        "not_attempted",
        "pending",
        "succeeded",
        "failed",
    }


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "plugin/README.md",
        "docs/customize/integrations/mcp-server.mdx",
        "docs/customize/integrations/cli-daemon.mdx",
        "src/openbrowser/code_use/descriptions.py",
    ],
)
def test_public_guidance_describes_health_without_page_readiness(path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / path).read_text(encoding="utf-8")

    assert "browser.health" in text or "`health`" in text or "health`" in text
    assert "process_present" in text or "daemon status" in text
    assert "page" in text.lower() and ("not" in text.lower() or "does not" in text.lower())


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "plugin/README.md",
        "docs/customize/integrations/mcp-server.mdx",
        "docs/customize/integrations/cli-daemon.mdx",
        "src/openbrowser/code_use/descriptions.py",
    ],
)
def test_public_guidance_has_state_specific_recovery_actions(path: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / path).read_text(encoding="utf-8").lower()

    assert "pending" in text
    assert "usable target" in text
    assert "reset" in text


def test_compact_description_names_both_status_commands() -> None:
    from openbrowser.code_use.descriptions import EXECUTE_CODE_DESCRIPTION_COMPACT

    assert "get_status" in EXECUTE_CODE_DESCRIPTION_COMPACT
    assert "openbrowser-ai daemon status" in EXECUTE_CODE_DESCRIPTION_COMPACT
    assert "health` is not page readiness" in EXECUTE_CODE_DESCRIPTION_COMPACT
