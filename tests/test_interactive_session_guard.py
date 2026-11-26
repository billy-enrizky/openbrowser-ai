"""Browser-free tests for the shared matched-engine session guard."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import psutil
import pytest

from benchmarks.interactive import engine_binding, session_guard

SHELL_RELATIVE = Path("chromium_headless_shell-1255/chrome-headless-shell-mac-arm64/chrome-headless-shell")
VERSION_OUTPUT = "Google Chrome for Testing 159.0.8154.6 "
LAUNCHER_ARGV0 = "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    import socket
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("native I/O is forbidden in fake tests")

    monkeypatch.setattr(psutil, "Process", forbidden)
    monkeypatch.setattr(psutil, "process_iter", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def make_pin(tmp_path: Path):
    launcher = tmp_path / "cache" / SHELL_RELATIVE
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_bytes(b"shell")
    launcher.chmod(0o755)
    return engine_binding.capture_engine_pin(
        engine_binding.LANE_MATCHED_SHELL_1255,
        tmp_path / "cache",
        version_runner=lambda path: VERSION_OUTPUT,
        platform_name="darwin",
        machine="arm64",
    )


def profile_marker(tmp_path: Path) -> Path:
    return tmp_path / "Session Guard" / "config" / "profiles" / "daemon"


def argv(marker: str, *, keychain: bool = True, two_token: bool = False) -> list[str]:
    row = [LAUNCHER_ARGV0]
    if two_token:
        row.extend(["--user-data-dir", marker])
    else:
        row.append(f"--user-data-dir={marker}")
    if keychain:
        row.append("--use-mock-keychain")
    return row


class FakeProcess:
    def __init__(
        self,
        exe: str,
        *,
        created_at: float = 1000.0,
        created_sequence: list[float] | None = None,
        running: bool = True,
        status: str = psutil.STATUS_RUNNING,
        exe_error: Exception | None = None,
    ):
        self._exe = exe
        self._created_at = created_at
        self._created_sequence = list(created_sequence or [])
        self._running = running
        self._status = status
        self._exe_error = exe_error

    def create_time(self):
        if self._created_sequence:
            return self._created_sequence.pop(0)
        return self._created_at

    def exe(self):
        if self._exe_error is not None:
            raise self._exe_error
        return self._exe

    def is_running(self):
        return self._running

    def status(self):
        return self._status


def factory_for(process):
    return lambda pid: process


def table(*rows):
    return lambda: list(rows)


class SequenceTable:
    def __init__(self, *snapshots):
        self._snapshots = [list(snapshot) for snapshot in snapshots]
        self.calls = 0

    def __call__(self):
        index = min(self.calls, len(self._snapshots) - 1)
        self.calls += 1
        return list(self._snapshots[index])


def valid_rows(tmp_path: Path, *, pid: int = 4242, keychain: bool = True, two_token: bool = False):
    marker = str(profile_marker(tmp_path))
    return [(pid, argv(marker, keychain=keychain, two_token=two_token))]


def observe(
    pin,
    tmp_path: Path,
    *,
    rows=None,
    process=None,
    process_factory=None,
    table_fn=None,
    stat_reader=None,
):
    marker = profile_marker(tmp_path)
    process = process or FakeProcess(str(pin.launcher))
    process_table = table_fn or table(*(rows if rows is not None else valid_rows(tmp_path)))
    return session_guard.observe_session(
        pin,
        marker,
        process_table=process_table,
        process_factory=process_factory or factory_for(process),
        stat_reader=stat_reader,
    )


def test_observe_accepts_exact_marker_with_spaces(tmp_path):
    pin = make_pin(tmp_path)
    observation = observe(pin, tmp_path)
    assert observation.status == "valid"
    assert observation.eligible is True
    assert observation.proof is not None
    assert observation.pid == 4242
    assert observation.keychain_verified is True
    assert observation.public_proof()["executable_basename"] == "chrome-headless-shell"


def test_observe_accepts_two_token_user_data_dir(tmp_path):
    pin = make_pin(tmp_path)
    observation = observe(pin, tmp_path, rows=valid_rows(tmp_path, two_token=True))
    assert observation.status == "valid"
    assert observation.eligible is True


def test_observe_accepts_canonical_profile_alias(tmp_path):
    pin = make_pin(tmp_path)
    real = tmp_path / "real profile"
    alias = tmp_path / "alias profile"
    real.mkdir()
    alias.symlink_to(real, target_is_directory=True)
    rows = [(4242, argv(str(real)))]
    process = FakeProcess(str(pin.launcher))
    observation = session_guard.observe_session(
        pin,
        alias,
        process_table=table(*rows),
        process_factory=factory_for(process),
    )
    assert observation.status == "valid"


def test_observe_ignores_substring_marker_decoy(tmp_path):
    pin = make_pin(tmp_path)
    marker = profile_marker(tmp_path)
    decoy = str(marker) + "-decoy"
    observation = observe(pin, tmp_path, rows=[(4242, argv(decoy))])
    assert observation.status == "process_missing"
    assert observation.eligible is False
    assert observation.proof is None


def test_observe_ignores_helper_processes(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [
        (4242, argv(marker)),
        (4243, [*argv(marker), "--type=renderer"]),
        (
            4244,
            [
                "/Applications/Google Chrome Helper.app/Contents/MacOS/Google Chrome Helper",
                f"--user-data-dir={marker}",
            ],
        ),
    ]
    observation = observe(pin, tmp_path, rows=rows)
    assert observation.status == "valid"
    assert observation.pid == 4242


def test_observe_ignores_node_launcher_with_same_marker(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [
        (4243, ["/opt/homebrew/bin/node", f"--user-data-dir={marker}", "--use-mock-keychain"]),
        (4242, argv(marker)),
    ]
    observation = observe(pin, tmp_path, rows=rows)
    assert observation.status == "valid"
    assert observation.pid == 4242


def test_observe_ignores_python_and_agent_launchers_with_same_marker(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [
        (4243, ["/usr/bin/python3.12", f"--user-data-dir={marker}", "--use-mock-keychain"]),
        (4244, ["/usr/local/bin/agent-browser", f"--user-data-dir={marker}", "--use-mock-keychain"]),
        (4245, ["/usr/local/bin/chrome-devtools-mcp", f"--user-data-dir={marker}", "--use-mock-keychain"]),
        (4242, argv(marker)),
    ]
    observation = observe(pin, tmp_path, rows=rows)
    assert observation.status == "valid"
    assert observation.pid == 4242


def test_observe_launchers_only_are_process_missing(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [
        (4243, ["/opt/homebrew/bin/node", f"--user-data-dir={marker}", "--use-mock-keychain"]),
        (4244, ["/usr/bin/python3.12", f"--user-data-dir={marker}", "--use-mock-keychain"]),
    ]
    observation = observe(pin, tmp_path, rows=rows)
    assert observation.status == "process_missing"


def test_observe_rejects_true_duplicate_browser_with_launchers(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [
        (4243, ["/opt/homebrew/bin/node", f"--user-data-dir={marker}", "--use-mock-keychain"]),
        (4242, argv(marker)),
        (4245, argv(marker)),
    ]
    observation = observe(pin, tmp_path, rows=rows)
    assert observation.status == "duplicate_main"


def test_observe_rejects_wrong_engine_browser_family(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [(4242, ["/usr/bin/chromium", f"--user-data-dir={marker}", "--use-mock-keychain"])]
    observation = observe(pin, tmp_path, rows=rows, process=FakeProcess("/usr/bin/true"))
    assert observation.status == "executable_changed"
    assert observation.proof is None


def test_observe_rejects_malformed_pin_without_exception(tmp_path):
    observation = session_guard.observe_session(object(), profile_marker(tmp_path))  # pyright: ignore[reportArgumentType]
    assert observation.status == "malformed"
    assert observation.eligible is False
    assert observation.proof is None


def test_observe_closes_malformed_pin_fence_errors(tmp_path):
    pin = make_pin(tmp_path)
    observation = observe(
        replace(pin, launcher=None),
        tmp_path,
        process=FakeProcess(str(pin.launcher)),
    )
    assert observation.status == "malformed"
    assert observation.proof is None


def test_observe_rejects_malformed_marker_forms(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    for argv_row in (
        [LAUNCHER_ARGV0, "--user-data-dir"],
        [LAUNCHER_ARGV0, "--user-data-dir="],
        [LAUNCHER_ARGV0, f"--user-data-dir={marker}", f"--user-data-dir={marker}-other"],
    ):
        observation = observe(pin, tmp_path, rows=[(4242, argv_row)])
        assert observation.status == "malformed"
        assert observation.proof is None


def test_observe_ignores_crashpad_helper(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [
        (
            4242,
            [
                "/Applications/Google Chrome.app/Contents/Frameworks/chrome_crashpad_handler",
                f"--user-data-dir={marker}",
                "--use-mock-keychain",
            ],
        )
    ]
    observation = observe(pin, tmp_path, rows=rows)
    assert observation.status == "process_missing"


def test_observe_rejects_duplicate_main_processes(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    rows = [(4242, argv(marker)), (4243, argv(marker))]
    observation = observe(pin, tmp_path, rows=rows)
    assert observation.status == "duplicate_main"
    assert observation.eligible is False
    assert observation.proof is None


def test_observe_requires_exact_mock_keychain_token(tmp_path):
    pin = make_pin(tmp_path)
    missing = observe(pin, tmp_path, rows=valid_rows(tmp_path, keychain=False))
    assert missing.status == "keychain_missing"
    marker = str(profile_marker(tmp_path))
    substring = [(4242, [LAUNCHER_ARGV0, f"--user-data-dir={marker}", "--use-mock-keychain=false"])]
    rejected = observe(pin, tmp_path, rows=substring)
    assert rejected.status == "keychain_missing"


@pytest.mark.parametrize(
    ("process", "status"),
    [
        (FakeProcess("/usr/bin/true", exe_error=psutil.NoSuchProcess(4242)), "process_missing"),
        (FakeProcess("/usr/bin/true", exe_error=psutil.AccessDenied(4242)), "process_denied"),
        (FakeProcess("/usr/bin/true", running=False), "process_missing"),
        (FakeProcess("/usr/bin/true", status=psutil.STATUS_ZOMBIE), "process_missing"),
        (FakeProcess("/usr/bin/true", status="unknown"), "unknown"),
        (FakeProcess("/usr/bin/true", created_at=math.nan), "unknown"),
        (FakeProcess("/usr/bin/true", created_at=10**400), "unknown"),
        (FakeProcess("/usr/bin/true", created_sequence=[1000.0, 1001.0]), "process_replaced"),
    ],
)
def test_observe_fails_closed_on_bad_process(tmp_path, process, status):
    pin = make_pin(tmp_path)
    observation = observe(pin, tmp_path, process=process)
    assert observation.status == status
    assert observation.proof is None
    assert observation.eligible is False


def test_observe_does_not_accept_foreign_executable(tmp_path):
    pin = make_pin(tmp_path)
    observation = observe(pin, tmp_path, process=FakeProcess("/usr/bin/true"))
    assert observation.status == "executable_changed"
    assert observation.proof is None


def test_observe_fences_keychain_change_between_scans(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    process_table = SequenceTable(
        [(4242, argv(marker))],
        [(4242, argv(marker, keychain=False))],
    )
    observation = observe(pin, tmp_path, table_fn=process_table)
    assert observation.status == "keychain_missing"
    assert observation.proof is None


def test_observe_fences_marker_change_between_scans(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    other = str(tmp_path / "Other Profile" / "config" / "profiles" / "daemon")
    process_table = SequenceTable(
        [(4242, argv(marker))],
        [(4242, argv(other))],
    )
    observation = observe(pin, tmp_path, table_fn=process_table)
    assert observation.status == "marker_changed"
    assert observation.proof is None


def test_observe_fences_duplicate_main_between_scans(tmp_path):
    pin = make_pin(tmp_path)
    marker = str(profile_marker(tmp_path))
    process_table = SequenceTable(
        [(4242, argv(marker))],
        [(4242, argv(marker)), (4243, argv(marker))],
    )
    observation = observe(pin, tmp_path, table_fn=process_table)
    assert observation.status == "duplicate_main"
    assert observation.proof is None


@pytest.mark.parametrize(
    "rows",
    [
        "not-a-table",
        [(True, ["/bin/chrome"])],
        [("4242", ["/bin/chrome"])],
        [(4242, "not-an-argv-list")],
        [(4242, [1, 2, 3])],
        [(4242, [])],
    ],
)
def test_observe_rejects_malformed_process_table(tmp_path, rows):
    pin = make_pin(tmp_path)
    observation = observe(pin, tmp_path, table_fn=lambda: rows)
    assert observation.status == "malformed"
    assert observation.proof is None
    assert observation.eligible is False


def test_verify_accepts_same_live_session(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None
    second = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(*valid_rows(tmp_path)),
        process_factory=factory_for(FakeProcess(str(pin.launcher))),
    )
    assert second.status == "valid"
    assert second.eligible is True
    assert second.proof is first.proof


def test_verify_known_proof_fails_when_process_missing_or_denied(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None

    missing = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(*valid_rows(tmp_path)),
        process_factory=lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)),
    )
    assert missing.status == "process_missing"
    assert missing.proof is None

    denied = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(*valid_rows(tmp_path)),
        process_factory=lambda pid: (_ for _ in ()).throw(psutil.AccessDenied(pid)),
    )
    assert denied.status == "process_denied"
    assert denied.proof is None


def test_verify_detects_replaced_browser_and_foreign_executable(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None

    replaced = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(*valid_rows(tmp_path)),
        process_factory=factory_for(FakeProcess(str(pin.launcher), created_at=2000.0)),
    )
    assert replaced.status == "process_replaced"

    foreign = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(*valid_rows(tmp_path)),
        process_factory=factory_for(FakeProcess("/usr/bin/true")),
    )
    assert foreign.status == "executable_changed"


def test_verify_detects_keychain_and_marker_loss(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None

    no_keychain = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(*valid_rows(tmp_path, keychain=False)),
        process_factory=factory_for(FakeProcess(str(pin.launcher))),
    )
    assert no_keychain.status == "keychain_missing"

    other = str(tmp_path / "Other Profile" / "config" / "profiles" / "daemon")
    other_marker = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table((4242, argv(other))),
        process_factory=factory_for(FakeProcess(str(pin.launcher))),
    )
    assert other_marker.status == "marker_changed"


def test_verify_detects_pin_change(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None
    (tmp_path / "cache" / SHELL_RELATIVE).write_bytes(b"changed")
    observation = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(*valid_rows(tmp_path)),
        process_factory=factory_for(FakeProcess(str(pin.launcher))),
    )
    assert observation.status == "pin_changed"
    assert observation.proof is None


def test_verify_rejects_malformed_proof_without_recache(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None
    before = first.proof.public_proof()
    observation = session_guard.verify_session(object(), pin)  # pyright: ignore[reportArgumentType]
    assert observation.status == "malformed"
    assert observation.proof is None
    assert first.proof.public_proof() == before


def test_observe_rejects_identity_change_after_second_argv_scan(tmp_path):
    pin = make_pin(tmp_path)
    process_table = SequenceTable(valid_rows(tmp_path), valid_rows(tmp_path))
    processes = [
        FakeProcess(str(pin.launcher), created_at=1000.0),
        FakeProcess(str(pin.launcher), created_at=2000.0),
    ]

    def factory(pid):
        return processes.pop(0)

    observation = observe(pin, tmp_path, table_fn=process_table, process_factory=factory)
    assert observation.status == "process_replaced"
    assert observation.proof is None


def test_observe_rejects_exec_change_after_second_argv_scan(tmp_path):
    pin = make_pin(tmp_path)
    process_table = SequenceTable(valid_rows(tmp_path), valid_rows(tmp_path))
    processes = [
        FakeProcess(str(pin.launcher), created_at=1000.0),
        FakeProcess("/usr/bin/true", created_at=1000.0),
    ]

    def factory(pid):
        return processes.pop(0)

    observation = observe(pin, tmp_path, table_fn=process_table, process_factory=factory)
    assert observation.status == "executable_changed"
    assert observation.proof is None


def test_default_process_table_keeps_argv_tokens(monkeypatch):
    argv = [
        "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "--user-data-dir=/tmp/Session Guard/config/profiles/daemon",
        "--use-mock-keychain",
    ]

    class Process:
        pid = 7

        def cmdline(self):
            return argv

    def process_iter(attrs):
        assert attrs == ["pid"]
        return [Process()]

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    assert session_guard._default_process_table() == [(7, argv)]


def test_default_process_table_skips_unreadable_processes(tmp_path, monkeypatch):
    marker = str(profile_marker(tmp_path))
    argv = [
        "/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        f"--user-data-dir={marker}",
        "--use-mock-keychain",
    ]

    class Denied:
        pid = 1

        def cmdline(self):
            raise psutil.AccessDenied(1)

    class Gone:
        pid = 2

        def cmdline(self):
            raise psutil.NoSuchProcess(2)

    class Good:
        pid = 7

        def cmdline(self):
            return argv

    def process_iter(attrs):
        assert attrs == ["pid"]
        return [Denied(), Gone(), Good()]

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    assert session_guard._default_process_table() == [(7, argv)]


def test_default_process_table_iterator_failure_is_unknown(tmp_path, monkeypatch):
    pin = make_pin(tmp_path)
    marker = profile_marker(tmp_path)

    def process_iter(attrs):
        assert attrs == ["pid"]
        raise psutil.Error("iterator failed")

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    observation = session_guard.observe_session(pin, marker)
    assert observation.status == "unknown"
    assert observation.proof is None


def test_verify_known_proof_rejects_duplicate_main(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None
    marker = str(profile_marker(tmp_path))
    observation = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table((4242, argv(marker)), (4243, argv(marker))),
        process_factory=factory_for(FakeProcess(str(pin.launcher))),
    )
    assert observation.status == "duplicate_main"
    assert observation.proof is None


def test_verify_known_proof_fails_when_table_is_empty(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None
    observation = session_guard.verify_session(
        first.proof,
        pin,
        process_table=table(),
        process_factory=factory_for(FakeProcess(str(pin.launcher))),
    )
    assert observation.status == "process_missing"
    assert observation.proof is None


@pytest.mark.parametrize(
    "field",
    ["lane", "revision", "version", "launcher_sha256", "engine_image_sha256"],
)
def test_verify_rejects_malformed_proof_fields(tmp_path, field):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None
    bad = replace(first.proof, **{field: None})
    observation = session_guard.verify_session(bad, pin)
    assert observation.status == "malformed"
    assert observation.proof is None


def test_verify_rejects_noncanonical_proof_marker(tmp_path):
    pin = make_pin(tmp_path)
    first = observe(pin, tmp_path)
    assert first.proof is not None
    bad = replace(first.proof, profile_marker="relative-marker")
    observation = session_guard.verify_session(bad, pin)
    assert observation.status == "malformed"
    assert observation.proof is None


def test_public_diagnostics_hide_paths_argv_and_exceptions(tmp_path):
    pin = make_pin(tmp_path)
    marker = profile_marker(tmp_path)
    observation = observe(pin, tmp_path)
    public = observation.public_proof()
    text = json.dumps(public, sort_keys=True)
    assert str(tmp_path) not in text
    assert str(marker) not in text
    assert "--user-data-dir" not in text
    assert "Traceback" not in text
    assert public["profile_digest"] is not None
    assert public["status"] == "valid"

    unknown = observe(pin, tmp_path, process=FakeProcess("/usr/bin/true", status="unknown"))
    unknown_public = unknown.public_proof()
    assert unknown_public["status"] == "unknown"
    assert unknown_public["pid"] is None
    assert unknown_public["keychain_verified"] is None
    assert unknown_public["profile_digest"] == public["profile_digest"] or unknown_public["profile_digest"] is None


def test_session_proof_public_has_no_paths(tmp_path):
    pin = make_pin(tmp_path)
    observation = observe(pin, tmp_path)
    assert observation.proof is not None
    text = json.dumps(observation.proof.public_proof(), sort_keys=True)
    assert str(tmp_path) not in text
    assert str(profile_marker(tmp_path)) not in text
    assert "--user-data-dir" not in text
