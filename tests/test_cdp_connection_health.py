"""Tests for explicit CDP root connection health."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from openbrowser.browser.cdp_connection import (
    CdpConnectionErrorCode,
    CdpConnectionHealthTracker,
    CdpConnectionPhase,
    sanitize_error_detail,
)


def test_initial_snapshot_is_disconnected_without_a_health_claim():
    tracker = CdpConnectionHealthTracker()

    health = tracker.snapshot(client_present=False, connection_expected=False)

    assert health.phase is CdpConnectionPhase.DISCONNECTED
    assert health.client_present is False
    assert health.healthy is None
    assert health.connection_generation == 0
    assert health.connect_attempts == 0
    assert health.reconnect_count == 0
    assert health.probe_attempts == 0
    assert health.last_error_code is None


def test_connect_attempt_counts_and_phase_progression():
    tracker = CdpConnectionHealthTracker()

    first = tracker.begin_connect()
    tracker.mark_connecting(first)
    connecting = tracker.snapshot(client_present=True, connection_expected=True)
    tracker.mark_connected(first)
    connected = tracker.snapshot(client_present=True, connection_expected=True)
    second = tracker.begin_connect()

    assert first == 1
    assert connecting.phase is CdpConnectionPhase.CONNECTING
    assert connected.phase is CdpConnectionPhase.CONNECTED
    assert connected.healthy is True
    assert connected.connect_attempts == 1
    assert connected.reconnect_count == 0
    assert connected.last_connected_at is not None
    assert connected.last_success_at is not None
    assert second == 2
    assert tracker.snapshot(client_present=True, connection_expected=True).reconnect_count == 1


def test_connect_failure_records_a_sanitized_error():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()

    tracker.mark_connect_failed(
        generation,
        RuntimeError("connection closed ws://user:secret@example.test/devtools?token=abc"),
        client_present=False,
    )

    health = tracker.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.CONNECTION_CLOSED
    assert health.last_error_type == "RuntimeError"
    assert "secret" not in (health.last_error_detail or "")
    assert "token=abc" not in (health.last_error_detail or "")


def test_cancelled_connect_keeps_a_terminal_failed_state():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()

    tracker.mark_connect_failed(generation, asyncio.CancelledError(), client_present=False, cancelled=True)

    health = tracker.snapshot(client_present=False, connection_expected=False)
    assert health.phase is CdpConnectionPhase.FAILED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.CANCELLED


def test_probe_success_updates_latency_and_clears_previous_error():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connected(generation)
    probe = tracker.begin_probe(generation)

    tracker.complete_probe_success(probe, latency_ms=12.5)

    health = tracker.snapshot(client_present=True, connection_expected=True)
    assert health.phase is CdpConnectionPhase.CONNECTED
    assert health.healthy is True
    assert health.probe_attempts == 1
    assert health.last_probe_at is not None
    assert health.last_latency_ms == 12.5
    assert health.last_error_code is None
    assert health.last_error_detail is None


def test_probe_failure_with_a_root_is_degraded():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connected(generation)
    probe = tracker.begin_probe(generation)

    tracker.complete_probe_failure(
        probe,
        TimeoutError("Browser.getVersion timed out at ws://proxy.test/secret"),
        latency_ms=2000.0,
        client_present=True,
    )

    health = tracker.snapshot(client_present=True, connection_expected=True)
    assert health.phase is CdpConnectionPhase.DEGRADED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.TIMEOUT
    assert health.last_latency_ms == 2000.0
    assert "proxy.test" not in (health.last_error_detail or "")


def test_no_root_probe_is_disconnected_and_not_connected():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connected(generation)
    probe = tracker.begin_probe(generation)

    tracker.complete_probe_failure(
        probe,
        RuntimeError("CDP client is not connected"),
        latency_ms=0.0,
        client_present=False,
    )

    health = tracker.snapshot(client_present=False, connection_expected=True)
    assert health.phase is CdpConnectionPhase.DISCONNECTED
    assert health.healthy is False
    assert health.last_error_code is CdpConnectionErrorCode.NOT_CONNECTED


def test_snapshot_derives_disconnection_after_root_removal():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connected(generation)

    health = tracker.snapshot(client_present=False, connection_expected=False)

    assert health.phase is CdpConnectionPhase.DISCONNECTED
    assert health.client_present is False
    assert health.healthy is None
    assert health.last_success_at is not None


def test_snapshot_preserves_connecting_while_connection_is_expected():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connecting(generation)

    expected = tracker.snapshot(client_present=False, connection_expected=True)
    stopped = tracker.snapshot(client_present=False, connection_expected=False)

    assert expected.phase is CdpConnectionPhase.CONNECTING
    assert stopped.phase is CdpConnectionPhase.DISCONNECTED
    assert stopped.healthy is None


def test_older_probe_completion_cannot_overwrite_newer_result():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connected(generation)
    older = tracker.begin_probe(generation)
    newer = tracker.begin_probe(generation)

    tracker.complete_probe_success(older, latency_ms=1.0)
    tracker.complete_probe_failure(
        newer,
        TimeoutError("timed out"),
        latency_ms=2.0,
        client_present=True,
    )

    health = tracker.snapshot(client_present=True, connection_expected=True)
    assert health.phase is CdpConnectionPhase.DEGRADED
    assert health.last_error_code is CdpConnectionErrorCode.TIMEOUT
    assert health.probe_attempts == 2


def test_stale_generation_and_discarded_probe_are_ignored():
    tracker = CdpConnectionHealthTracker()
    old_generation = tracker.begin_connect()
    tracker.mark_connected(old_generation)
    probe = tracker.begin_probe(old_generation)
    new_generation = tracker.begin_connect()

    tracker.complete_probe_success(probe, latency_ms=1.0)
    tracker.mark_connected(old_generation)
    tracker.discard_probe(probe)

    health = tracker.snapshot(client_present=True, connection_expected=True)
    assert health.connection_generation == new_generation
    assert health.phase is CdpConnectionPhase.RESOLVING
    assert health.healthy is None


def test_cleanup_failure_is_visible_and_success_clears_it():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connect_failed(generation, RuntimeError("connect failed"), client_present=True)
    tracker.mark_cleanup_failure(RuntimeError("root stop timed out"))

    pending = tracker.snapshot(client_present=True, connection_expected=False)
    assert pending.phase is CdpConnectionPhase.FAILED
    assert pending.cleanup_pending is True
    assert "root stop" in (pending.last_cleanup_error or "")

    next_generation = tracker.begin_connect()
    tracker.mark_connected(next_generation)
    cleared = tracker.snapshot(client_present=True, connection_expected=True)
    assert cleared.cleanup_pending is False
    assert cleared.last_cleanup_error is None


def test_cleanup_error_is_sanitized():
    tracker = CdpConnectionHealthTracker()

    tracker.mark_cleanup_failure(RuntimeError("root stop failed at ws://user:pass@example.test token=secret-token"))

    health = tracker.snapshot(client_present=True, connection_expected=True)
    detail = health.last_cleanup_error or ""
    assert "<url>" in detail
    assert "token=<redacted>" in detail
    assert "user:pass" not in detail
    assert "secret-token" not in detail


def _sanitized_error_fields(text: str) -> tuple[str, str]:
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connect_failed(generation, RuntimeError(text), client_present=False)
    tracker.mark_cleanup_failure(RuntimeError(text))
    health = tracker.snapshot(client_present=False, connection_expected=False)
    return health.last_error_detail or "", health.last_cleanup_error or ""


def test_error_sanitizer_redacts_full_bearer_authorization_value():
    detail, cleanup = _sanitized_error_fields("Authorization: Bearer abc123")

    for text in (detail, cleanup):
        assert "abc123" not in text
        assert "authorization=<redacted>" in text.lower()


def test_error_sanitizer_redacts_every_cookie_pair():
    detail, cleanup = _sanitized_error_fields("Cookie: session=one; csrf=two")

    for text in (detail, cleanup):
        assert "session=one" not in text
        assert "csrf=two" not in text
        assert "cookie=<redacted>" in text.lower()


def test_error_sanitizer_redacts_proxy_authorization_header():
    detail, cleanup = _sanitized_error_fields("Proxy-Authorization: Basic abc123")

    for text in (detail, cleanup):
        assert "abc123" not in text
        assert "proxy-authorization=<redacted>" in text.lower()


def test_error_sanitizer_redacts_quoted_header_mappings():
    detail, cleanup = _sanitized_error_fields(
        "headers={'Authorization': 'Bearer abc123', "
        "'Proxy-Authorization': 'Basic proxy456', "
        "'Cookie': 'session=one; csrf=two'}"
    )

    for text in (detail, cleanup):
        for secret in ("abc123", "proxy456", "session=one", "csrf=two"):
            assert secret not in text
        assert "<redacted>" in text


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer\r\n folded-secret",
        "Proxy-Authorization: Basic\r\n proxy-secret",
        "Cookie: session=one;\r\n csrf=two",
        "headers={'Authorization': 'Bearer\r\n folded-secret'}",
        "headers={'Proxy-Authorization': 'Basic\r\n proxy-secret'}",
        "headers={'Cookie': 'session=one;\r\n csrf=two'}",
    ],
)
def test_error_sanitizer_redacts_multiline_header_values(text):
    detail = sanitize_error_detail(text)

    assert "folded-secret" not in detail
    assert "proxy-secret" not in detail
    assert "csrf=two" not in detail
    assert "<redacted>" in detail


def test_error_sanitizer_removes_non_whitespace_control_characters():
    detail = sanitize_error_detail("error\x1b[31mred\x07done")

    assert detail == "error [31mred done"


@pytest.mark.parametrize(
    "value",
    [
        "http://example.test/secret",
        "https://example.test/secret",
        "ws://example.test/secret",
        "wss://example.test/secret",
    ],
)
def test_error_sanitizer_replaces_every_supported_url_scheme(value):
    detail, cleanup = _sanitized_error_fields(f"connect {value} failed")

    for text in (detail, cleanup):
        assert value not in text
        assert "<url>" in text


@pytest.mark.parametrize("key", ["password", "passwd", "pwd", "token", "secret", "api_key", "username"])
def test_error_sanitizer_redacts_every_supported_assignment_key(key):
    sentinel = f"sentinel-{key}"
    detail, cleanup = _sanitized_error_fields(f"{key}={sentinel}")

    for text in (detail, cleanup):
        assert sentinel not in text
        assert f"{key}=<redacted>" in text


def test_error_sanitizer_redacts_quoted_assignments_with_spaces():
    detail, cleanup = _sanitized_error_fields("password='two word value' token=\"three word value\"")

    for text in (detail, cleanup):
        assert "two word value" not in text
        assert "three word value" not in text
        assert text.count("<redacted>") == 2


def test_error_detail_is_bounded_after_redaction():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()

    tracker.mark_connect_failed(
        generation,
        RuntimeError(
            "password=secret-password ws://user:pass@example.test/devtools?token=secret-token\n"
            "authorization: bearer-secret\n" + "x" * 500
        ),
        client_present=False,
    )

    health = tracker.snapshot(client_present=False, connection_expected=False)
    detail = health.last_error_detail or ""
    assert len(detail) <= 240
    for secret in ("secret-password", "bearer-secret", "user:pass", "secret-token"):
        assert secret not in detail
    assert "<url>" in detail
    assert "<redacted>" in detail


def test_cleanup_error_detail_is_bounded_after_redaction():
    tracker = CdpConnectionHealthTracker()

    tracker.mark_cleanup_failure(RuntimeError("password=secret-password\n" + "x" * 500))

    health = tracker.snapshot(client_present=False, connection_expected=False)
    detail = health.last_cleanup_error or ""
    assert len(detail) <= 240
    assert "secret-password" not in detail
    assert "<redacted>" in detail


def test_no_root_probe_health_becomes_unknown_after_reset():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connected(generation)
    probe = tracker.begin_probe(generation)
    tracker.complete_probe_failure(
        probe,
        RuntimeError("CDP client is not connected"),
        latency_ms=0.0,
        client_present=False,
    )

    health = tracker.snapshot(client_present=False, connection_expected=False)

    assert health.phase is CdpConnectionPhase.DISCONNECTED
    assert health.healthy is None
    assert health.last_error_code is CdpConnectionErrorCode.NOT_CONNECTED
    assert health.last_failure_at is not None


def test_older_probe_failure_finishing_after_newer_success_is_ignored():
    tracker = CdpConnectionHealthTracker()
    generation = tracker.begin_connect()
    tracker.mark_connected(generation)
    older = tracker.begin_probe(generation)
    newer = tracker.begin_probe(generation)

    tracker.complete_probe_success(newer, latency_ms=2.0)
    tracker.complete_probe_failure(
        older,
        TimeoutError("older probe timed out"),
        latency_ms=10.0,
        client_present=True,
    )

    health = tracker.snapshot(client_present=True, connection_expected=True)
    assert health.phase is CdpConnectionPhase.CONNECTED
    assert health.healthy is True
    assert health.last_latency_ms == 2.0
    assert health.last_error_code is None
    assert health.last_failure_at is None


def test_connect_and_probe_failures_record_ordered_timestamps():
    tracker = CdpConnectionHealthTracker()

    with patch("openbrowser.browser.cdp_connection.time.time", side_effect=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]):
        first = tracker.begin_connect()
        tracker.mark_connect_failed(first, RuntimeError("connect failed"), client_present=False)
        failed_connect = tracker.snapshot(client_present=False, connection_expected=False)

        second = tracker.begin_connect()
        tracker.mark_connected(second)
        probe = tracker.begin_probe(second)
        tracker.complete_probe_failure(
            probe,
            TimeoutError("probe failed"),
            latency_ms=1.0,
            client_present=True,
        )
        failed_probe = tracker.snapshot(client_present=True, connection_expected=True)

    assert failed_connect.last_failure_at is not None
    assert failed_connect.last_connect_started_at is not None
    assert failed_connect.last_failure_at >= failed_connect.last_connect_started_at
    assert failed_probe.last_probe_at is not None
    assert failed_probe.last_failure_at is not None
    assert failed_probe.last_failure_at >= failed_probe.last_probe_at


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4])
def test_sanitizer_never_exceeds_small_caps(limit):
    detail = sanitize_error_detail("x" * 1000, max_chars=limit)

    assert len(detail) <= limit


def test_preserve_lines_redacts_multitoken_folded_values():
    detail = sanitize_error_detail(
        'Authorization:\n Bearer abc123 def456\n  File "<execute_code>", line 3\n  captured diagnostic = expected',
        preserve_lines=True,
    )

    assert "abc123" not in detail
    assert "def456" not in detail
    assert '  File "<execute_code>", line 3' in detail
    assert "  captured diagnostic = expected" in detail


@pytest.mark.parametrize("header", ["Authorization", "Proxy-Authorization"])
@pytest.mark.parametrize("value", ["q", "q7", "z9Q", "abcd"])
def test_preserve_lines_redacts_short_folded_auth_values(header, value):
    detail = sanitize_error_detail(
        f'{header}: Bearer\n  {value}\n  File "<execute_code>", line 3\n  result=1',
        preserve_lines=True,
    )

    assert f"  {value}" not in detail.splitlines()
    assert '  File "<execute_code>", line 3' in detail
    assert "  result=1" in detail


def test_preserve_lines_redacts_cookie_value_split_after_equals():
    detail = sanitize_error_detail(
        'Cookie: session=\n  q7\n  a=b\n  csrf=xyz\n  File "<execute_code>", line 3\n  captured diagnostic = expected',
        preserve_lines=True,
    )

    assert "  q7" not in detail.splitlines()
    assert "  a=b" not in detail.splitlines()
    assert "  csrf=xyz" not in detail.splitlines()
    assert '  File "<execute_code>", line 3' in detail
    assert "  captured diagnostic = expected" in detail


def test_preserve_lines_redacts_short_folded_cookie_value():
    detail = sanitize_error_detail(
        'Cookie: session=one;\n  q7=z\n  File "<execute_code>", line 3',
        preserve_lines=True,
    )

    assert "q7=z" not in detail
    assert '  File "<execute_code>", line 3' in detail


@pytest.mark.parametrize("diagnostic", ["result=1", "request_id=123"])
def test_preserve_lines_keeps_compact_assignment_after_complete_header(diagnostic):
    detail = sanitize_error_detail(
        f"Authorization: Bearer long-secret-token\n  {diagnostic}",
        preserve_lines=True,
    )

    assert "long-secret-token" not in detail
    assert diagnostic in detail


def test_error_code_classification_precedence():
    tracker = CdpConnectionHealthTracker()

    assert tracker.classify_error(asyncio.CancelledError(), connect=True) is CdpConnectionErrorCode.CANCELLED
    assert tracker.classify_error(TimeoutError("timed out"), connect=True) is CdpConnectionErrorCode.TIMEOUT
    assert (
        tracker.classify_error(RuntimeError("-32001 Session with given id not found"), connect=True)
        is CdpConnectionErrorCode.SESSION_NOT_FOUND
    )
    assert (
        tracker.classify_error(RuntimeError("ConnectionClosedError: websocket closed"), connect=False)
        is CdpConnectionErrorCode.CONNECTION_CLOSED
    )
    assert tracker.classify_error(RuntimeError("boom"), connect=False) is CdpConnectionErrorCode.PROBE_FAILED
    assert tracker.classify_error(RuntimeError("boom"), connect=True) is CdpConnectionErrorCode.CONNECT_FAILED


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("-32001 Session with given id not found websocket"), CdpConnectionErrorCode.TIMEOUT),
        (RuntimeError("-32001 Session with given id not found websocket"), CdpConnectionErrorCode.SESSION_NOT_FOUND),
        (RuntimeError("ConnectionClosedError websocket -32001"), CdpConnectionErrorCode.SESSION_NOT_FOUND),
    ],
)
def test_competing_error_markers_follow_the_fixed_precedence(error, expected):
    tracker = CdpConnectionHealthTracker()

    assert tracker.classify_error(error, connect=False) is expected


def test_connection_closed_classifier_uses_only_the_approved_markers():
    tracker = CdpConnectionHealthTracker()

    for marker in ("connectionclosed", "connection closed", "no close frame", "websocket"):
        assert tracker.classify_error(RuntimeError(marker), connect=False) is CdpConnectionErrorCode.CONNECTION_CLOSED
    for marker in ("browser closed", "target closed"):
        assert tracker.classify_error(RuntimeError(marker), connect=False) is CdpConnectionErrorCode.PROBE_FAILED
