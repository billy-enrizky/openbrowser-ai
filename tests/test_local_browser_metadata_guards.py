"""Missing ownership metadata is rejected before typed conversions or lookup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from openbrowser.browser.watchdogs import local_browser_watchdog as module


def test_missing_profile_directory_fails_before_path_resolution(monkeypatch):
    watchdog: Any = SimpleNamespace(
        _profile_lease=None,
        browser_session=SimpleNamespace(browser_profile=SimpleNamespace(user_data_dir=None)),
    )

    def forbidden(*args):
        raise AssertionError("invalid directory reached path resolution")

    monkeypatch.setattr(module, "Path", forbidden)
    with pytest.raises(TypeError, match="user_data_dir"):
        module.LocalBrowserWatchdog._profile_lease_path(watchdog)


@pytest.mark.parametrize("record", [{}, {"pid": None, "start_time": 2.0}, {"pid": 42, "start_time": None}])
def test_missing_record_fields_return_unknown_before_conversion(monkeypatch, record):
    process: Any = SimpleNamespace(pid=42, create_time=lambda: 2.0)

    def parse_int(value):
        assert value is not None, "missing PID reached integer conversion"
        return int(value)

    def parse_float(value):
        assert value is not None, "missing time reached float conversion"
        return float(value)

    monkeypatch.setattr(module, "int", parse_int, raising=False)
    monkeypatch.setattr(module, "float", parse_float, raising=False)
    assert module.LocalBrowserWatchdog._recorded_process_state(process, record) == "unknown"


def test_gone_process_stays_gone_even_with_missing_record_fields():
    def gone():
        raise psutil.NoSuchProcess(42)

    process: Any = SimpleNamespace(pid=42, create_time=gone)
    assert module.LocalBrowserWatchdog._recorded_process_state(process, {}) == "gone"


async def test_missing_browser_pid_fails_before_conversion_and_lookup(monkeypatch, tmp_path):
    directory = str(tmp_path.resolve())
    metadata = {
        "profile_dir": directory,
        "instance_id": "owned",
        "owner_pid": 999,
        "owner_start_time": 1.0,
        "browser": {"profile_dir": directory, "instance_id": "owned", "pid": None},
    }

    def parse_int(value):
        assert value is not None, "missing PID reached integer conversion"
        return int(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("missing ownership PID reached process lookup")

    monkeypatch.setattr(module, "int", parse_int, raising=False)
    monkeypatch.setattr(module.psutil, "Process", forbidden)
    monkeypatch.setattr(module.ProfileLease, "process_identity_status", lambda *args: "missing")
    assert (
        await module.LocalBrowserWatchdog._kill_stale_chrome_for_profile(
            directory, metadata=metadata, instance_id="owned"
        )
        is False
    )
