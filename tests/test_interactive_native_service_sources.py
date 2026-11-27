"""Pure native ownership candidates, with all host access forbidden."""

from __future__ import annotations

import json
import socket
import subprocess
from dataclasses import replace

import psutil
import pytest

from benchmarks.interactive.adapters import native_service_sources as sources


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Native service source tests permit no host I/O")

    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.mark.parametrize("source", list(sources.PidSource))
@pytest.mark.parametrize("payload", [b"22160", b"22160\n", b" 22160\r\n"])
def test_exact_decimal_pid_is_a_candidate_only(source, payload):
    result = sources.parse_native_pid_record(payload, source=source)
    assert result == sources.PidRecord(pid=22160, record_format="decimal", started=None)
    assert not hasattr(result, "verified")


@pytest.mark.parametrize("started", ["Fri Nov 27 12:01:27 2026", "84919", 134087838870000000])
def test_browser_use_pending_json_preserves_start_fingerprint(started):
    result = sources.parse_native_pid_record(
        json.dumps({"pid": 22160, "started": started}).encode(),
        source=sources.PidSource.BROWSER_USE,
    )
    assert result.pid == 22160
    assert result.record_format == "json"
    assert result.started == started
    assert type(result.started) is type(started)


def test_pending_json_then_ready_decimal_are_separate_native_records():
    first = sources.parse_native_pid_record(
        b'{"pid":22160,"started":"native-start"}', source=sources.PidSource.BROWSER_USE
    )
    ready = sources.parse_native_pid_record(b"22160", source=sources.PidSource.BROWSER_USE)
    assert first.pid == ready.pid
    assert first.started == "native-start"
    assert ready.started is None


@pytest.mark.parametrize("source", [sources.PidSource.OPENBROWSER, sources.PidSource.AGENT_BROWSER])
def test_json_is_not_an_observed_format_for_other_pid_sources(source):
    with pytest.raises(sources.SourceError):
        sources.parse_native_pid_record(b'{"pid":12,"started":"start"}', source=source)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b" \n",
        b"0",
        b"-1",
        b"+1",
        b"01",
        b"1.0",
        b"1 2",
        b"true",
        b"null",
        b"\xff",
        b"12\x00",
        b"[12]",
        b'{"pid":true,"started":"s"}',
        b'{"pid":1.5,"started":"s"}',
        b'{"pid":"12","started":"s"}',
        b'{"pid":0,"started":"s"}',
        b'{"pid":-1,"started":"s"}',
        b'{"pid":12}',
        b'{"started":"s"}',
        b'{"pid":12,"started":null}',
        b'{"pid":12,"started":true}',
        b'{"pid":12,"started":1.5}',
        b'{"pid":12,"started":""}',
        b'{"pid":12,"started":" "}',
        b'{"pid":12,"started":"s","other":1}',
        b'{"pid":12,"pid":13,"started":"s"}',
        b'{"pid":12,"started":NaN}',
        b'{"pid":12,"started":Infinity}',
        b'{"pid":12,"started":-Infinity}',
    ],
)
def test_invalid_records_fail_closed(payload):
    with pytest.raises(sources.SourceError):
        sources.parse_native_pid_record(payload, source=sources.PidSource.BROWSER_USE)


@pytest.mark.parametrize("payload", [None, "12", 12, True, [12], bytearray(b"12")])
def test_payload_requires_bytes(payload):
    with pytest.raises(sources.SourceError):
        sources.parse_native_pid_record(payload, source=sources.PidSource.BROWSER_USE)


def test_payload_bound_is_exact_and_precedes_json_parsing():
    payload = b" " * (sources.MAX_PID_RECORD_BYTES - 2) + b"12"
    assert sources.parse_native_pid_record(payload, source=sources.PidSource.BROWSER_USE).pid == 12
    with pytest.raises(sources.SourceError, match="record_too_large"):
        sources.parse_native_pid_record(payload + b" ", source=sources.PidSource.BROWSER_USE)


@pytest.mark.parametrize("source", [None, "browser-use", object(), True])
def test_source_requires_the_closed_enum(source):
    with pytest.raises(sources.SourceError, match="invalid_source"):
        sources.parse_native_pid_record(b"12", source=source)


def expected():
    return {
        "executable": "/opt/node/bin/node",
        "daemon_entry": "/opt/playwright/lib/entry/cliDaemon.js",
        "session": "b20ed560908",
        "profile": "/private/tmp/owned/profile with spaces",
        "config": "/private/tmp/owned/config/playwright-config.json",
    }


def command(pid=73, *, executable=None, argv=None):
    e = expected()
    return sources.ProcessCommand(
        pid=pid,
        executable=e["executable"] if executable is None else executable,
        argv=(
            "/opt/node/bin/node",
            e["daemon_entry"],
            e["session"],
            "--profile=" + e["profile"],
            "--config=" + e["config"],
        )
        if argv is None
        else argv,
    )


def test_exact_playwright_daemon_is_selected_without_shell_splitting():
    wanted = command()
    other = command(74, argv=("node", "other.js"))
    assert sources.select_playwright_daemon([other, wanted], **expected()) is wanted


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_each_native_scope_token_must_match_exactly(index):
    argv = list(command().argv)
    argv[index] += "-decoy"
    with pytest.raises(sources.SourceError, match="process_missing"):
        sources.select_playwright_daemon([command(argv=tuple(argv))], **expected())


def test_executable_must_match_the_expected_native_node_image():
    with pytest.raises(sources.SourceError, match="process_missing"):
        sources.select_playwright_daemon([command(executable="/other/node")], **expected())


def test_argv_zero_may_use_a_launcher_alias_while_executable_is_exact():
    argv = ("/usr/local/bin/node", *command().argv[1:])
    wanted = command(argv=argv)
    assert sources.select_playwright_daemon([wanted], **expected()) is wanted


def test_missing_or_duplicate_native_owner_fails_closed():
    with pytest.raises(sources.SourceError, match="process_missing"):
        sources.select_playwright_daemon([], **expected())
    with pytest.raises(sources.SourceError, match="ambiguous_process"):
        sources.select_playwright_daemon([command(73), command(74)], **expected())


@pytest.mark.parametrize("argv", ["node cliDaemon.js", None, ("node", 12), ("node", "bad\x00token")])
def test_malformed_process_argv_is_an_unknown_candidate(argv):
    with pytest.raises(sources.SourceError, match="invalid_process"):
        sources.select_playwright_daemon(
            [replace(command(), argv=argv)],
            **expected(),
        )


@pytest.mark.parametrize("pid", [True, 0, -1, 1.5, "73"])
def test_malformed_candidate_pid_is_rejected(pid):
    with pytest.raises(sources.SourceError, match="invalid_process"):
        sources.select_playwright_daemon([command(pid)], **expected())


@pytest.mark.parametrize("field", ["executable", "daemon_entry", "profile", "config"])
def test_relative_expected_paths_are_rejected(field):
    e = expected()
    e[field] = "relative/path"
    with pytest.raises(sources.SourceError, match="invalid_expectation"):
        sources.select_playwright_daemon([command()], **e)


@pytest.mark.parametrize("field", ["executable", "daemon_entry", "profile", "config"])
@pytest.mark.parametrize("value", [None, True, 12, [], {}, "", "/bad\x00path"])
def test_malformed_expected_paths_fail_with_closed_reason(field, value):
    e = expected()
    e[field] = value
    with pytest.raises(sources.SourceError, match="invalid_expectation"):
        sources.select_playwright_daemon([command()], **e)


@pytest.mark.parametrize("started", [0, -1, "\x00", "s" * 513])
def test_invalid_native_fingerprint_is_rejected(started):
    with pytest.raises(sources.SourceError, match="invalid_fingerprint"):
        sources.parse_native_pid_record(
            json.dumps({"pid": 12, "started": started}).encode(), source=sources.PidSource.BROWSER_USE
        )


def test_large_integer_fingerprint_keeps_raw_precision():
    started = 2**54 + 1
    result = sources.parse_native_pid_record(
        json.dumps({"pid": 12, "started": started}).encode(), source=sources.PidSource.BROWSER_USE
    )
    assert type(result.started) is int
    assert result.started == started
    assert result.started != float(started)


def test_fake_io_guard_cannot_be_swallowed():
    with pytest.raises(pytest.fail.Exception):
        psutil.Process(73)
