"""Keep fresh-profile startup separate from persistent Chrome startup choices."""

from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from openbrowser.browser.watchdogs import local_browser_watchdog as module
from openbrowser.browser.watchdogs.profile_lease import ProfileLease


@pytest.fixture(autouse=True)
def block_native_io(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Native I/O is outside this fake test")

    monkeypatch.setattr(psutil, "Process", denied)
    monkeypatch.setattr(psutil, "process_iter", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)
    monkeypatch.setattr(socket.socket, "bind", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture
def launch_case(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile"
    profile = SimpleNamespace(
        user_data_dir=str(profile_dir),
        profile_directory="Default",
        executable_path="/native/chrome",
        headless=True,
        extra=[],
    )
    profile.get_args = lambda: ["--no-first-run", f"--user-data-dir={profile.user_data_dir}", *profile.extra]
    session = SimpleNamespace(
        browser_profile=profile,
        id="fake-browser",
        logger=logging.getLogger("pristine-startup-test"),
    )
    watchdog = module.LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())
    leases = []

    async def ensure(path, marker):
        if watchdog._profile_lease is not None:
            return
        lease = ProfileLease(path, instance_id=watchdog._instance_id, owner_pid=4321, owner_start_time=9.0)
        lease.acquire()
        lease.write_metadata()
        leases.append(lease)
        watchdog._profile_lease = lease

    process = SimpleNamespace(pid=1234, create_time=lambda: 10.0)
    spawn = AsyncMock(return_value=SimpleNamespace(pid=1234))
    monkeypatch.setattr(watchdog, "_ensure_profile_lease", ensure)
    monkeypatch.setattr(watchdog, "_find_free_port", lambda: 9223)
    monkeypatch.setattr(watchdog, "_cleanup_profile_cache", lambda *args: 0)
    monkeypatch.setattr(
        watchdog, "_wait_for_cdp_url", AsyncMock(return_value="ws://localhost:9223/devtools/browser/fake")
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(psutil, "Process", lambda pid: process if pid == 1234 else pytest.fail("Unexpected process"))
    case = SimpleNamespace(profile=profile, directory=profile_dir, watchdog=watchdog, spawn=spawn, process=process)
    try:
        yield case
    finally:
        for lease in reversed(leases):
            lease.release()


@pytest.mark.parametrize("existing_empty", [False, True])
async def test_fresh_profile_launch_opens_one_blank_page(launch_case, existing_empty):
    case = launch_case
    if existing_empty:
        case.directory.mkdir()
    process, endpoint = await case.watchdog._launch_browser(max_retries=1)
    assert process is case.process
    assert endpoint == "ws://localhost:9223/devtools/browser/fake"
    args = case.spawn.await_args.args
    assert args.count("about:blank") == 1
    assert f"--user-data-dir={case.directory}" in args
    assert "--remote-debugging-port=9223" in args
    assert any(arg.startswith("--openbrowser-instance-id=") for arg in args)


async def test_existing_profile_preserves_startup_preferences(launch_case):
    case = launch_case
    default = case.directory / "Default"
    default.mkdir(parents=True)
    preferences = default / "Preferences"
    content = '{"session":{"restore_on_startup":4,"startup_urls":["https://example.test/home"]}}'
    preferences.write_text(content)
    await case.watchdog._launch_browser(max_retries=1)
    assert "about:blank" not in case.spawn.await_args.args
    assert preferences.read_text() == content


async def test_seeded_storage_state_keeps_original_startup(launch_case):
    case = launch_case
    case.directory.mkdir()
    state = case.directory / "storage_state.json"
    content = '{"cookies":[],"origins":[]}'
    state.write_text(content)
    await case.watchdog._launch_browser(max_retries=1)
    assert "about:blank" not in case.spawn.await_args.args
    assert state.read_text() == content


@pytest.mark.parametrize("artifact", ["Default/Preferences", "storage_state.json", "foreign.lock"])
async def test_profile_data_added_after_lease_preserves_startup(launch_case, monkeypatch, artifact):
    case = launch_case

    def add_late_profile_data(*args):
        path = case.directory / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing profile data")
        return 0

    monkeypatch.setattr(case.watchdog, "_cleanup_profile_cache", add_late_profile_data)
    await case.watchdog._launch_browser(max_retries=1)
    assert "about:blank" not in case.spawn.await_args.args


async def test_final_directory_read_uncertainty_preserves_startup(launch_case, monkeypatch):
    case = launch_case
    original = module.os.scandir

    def deny_final_profile_read(*args):
        def scan(path):
            if Path(path) == case.directory.resolve():
                raise PermissionError("unavailable profile directory")
            return original(path)

        monkeypatch.setattr(module.os, "scandir", scan)
        return 0

    monkeypatch.setattr(case.watchdog, "_cleanup_profile_cache", deny_final_profile_read)
    await case.watchdog._launch_browser(max_retries=1)
    assert "about:blank" not in case.spawn.await_args.args


@pytest.mark.parametrize("mutation", ["directory", "lock", "closing_read"])
async def test_final_listing_closes_with_an_identity_fence(launch_case, monkeypatch, mutation):
    case = launch_case
    original_scandir = module.os.scandir
    original_stat = Path.stat
    events = []

    def install_mutation(*args):
        def scan(path):
            if Path(path) == case.directory.resolve():
                events.append("listing")
                if mutation == "directory":
                    moved = case.directory.with_name("moved-profile")
                    case.directory.rename(moved)
                    case.directory.mkdir()
                    for name in (ProfileLease.LOCK_FILENAME, ProfileLease.METADATA_FILENAME):
                        module.shutil.copyfile(moved / name, case.directory / name)
                elif mutation == "lock":
                    lock = case.directory / ProfileLease.LOCK_FILENAME
                    lock.unlink()
                    lock.write_bytes(b"")
                else:

                    def deny_closing_stat(target, *, follow_symlinks=True):
                        if target == case.directory and follow_symlinks is False:
                            events.append("closing_stat_denied")
                            raise PermissionError("closing profile stat unavailable")
                        return original_stat(target, follow_symlinks=follow_symlinks)

                    monkeypatch.setattr(Path, "stat", deny_closing_stat)
            return original_scandir(path)

        monkeypatch.setattr(module.os, "scandir", scan)
        return 0

    monkeypatch.setattr(case.watchdog, "_cleanup_profile_cache", install_mutation)
    await case.watchdog._launch_browser(max_retries=1)
    assert "about:blank" not in case.spawn.await_args.args
    if mutation == "closing_read":
        assert events == ["listing", "closing_stat_denied"]


@pytest.mark.parametrize(
    "extra",
    [
        ["https://example.test/start"],
        ["about:blank"],
        ["--app=https://example.test/app"],
        ["--app-id=fake-app"],
        ["--restore-last-session"],
        ["--no-startup-window"],
        ["--kiosk"],
    ],
)
async def test_explicit_startup_arguments_are_preserved(launch_case, extra):
    case = launch_case
    case.profile.extra = extra
    await case.watchdog._launch_browser(max_retries=1)
    args = case.spawn.await_args.args
    assert all(arg in args for arg in extra)
    assert args.count("about:blank") == extra.count("about:blank")


async def test_unknown_lease_adds_no_startup_page(launch_case, monkeypatch):
    case = launch_case
    monkeypatch.setattr(case.watchdog, "_ensure_profile_lease", AsyncMock())
    await case.watchdog._launch_browser(max_retries=1)
    assert "about:blank" not in case.spawn.await_args.args


async def test_proof_for_a_different_profile_keeps_original_startup(launch_case, monkeypatch, tmp_path):
    case = launch_case
    lease = ProfileLease(
        tmp_path / "different-profile", instance_id=case.watchdog._instance_id, owner_pid=4321, owner_start_time=9.0
    )
    lease.acquire()
    lease.write_metadata()
    monkeypatch.setattr(case.watchdog, "_ensure_profile_lease", AsyncMock())
    case.watchdog._profile_lease = lease
    try:
        await case.watchdog._launch_browser(max_retries=1)
        assert "about:blank" not in case.spawn.await_args.args
    finally:
        lease.release()


async def test_a_released_profile_lease_cannot_enable_blank_startup(launch_case, monkeypatch):
    case = launch_case
    lease = SimpleNamespace(
        profile_dir=case.directory,
        acquired=False,
        profile_pristine=True,
        record_browser=MagicMock(return_value={}),
    )
    monkeypatch.setattr(case.watchdog, "_ensure_profile_lease", AsyncMock())
    case.watchdog._profile_lease = lease
    await case.watchdog._launch_browser(max_retries=1)
    assert "about:blank" not in case.spawn.await_args.args


async def test_profile_error_retry_uses_its_own_blank_page_proof(launch_case, monkeypatch, tmp_path):
    case = launch_case
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    case.spawn.side_effect = [RuntimeError("user data directory in use"), SimpleNamespace(pid=1234)]
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda **kwargs: str(fallback))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    await case.watchdog._launch_browser(max_retries=2)
    assert len(case.spawn.await_args_list) == 2
    first, second = (call.args for call in case.spawn.await_args_list)
    assert first.count("about:blank") == 1
    assert second.count("about:blank") == 1
    assert f"--user-data-dir={fallback}" in second
