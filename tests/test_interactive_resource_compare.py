"""Browser-free accounting tests for the interactive resource lane."""

from __future__ import annotations

import importlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def resource_module():
    try:
        return importlib.import_module("benchmarks.interactive.resource_compare")
    except ModuleNotFoundError as error:
        if error.name == "benchmarks.interactive.resource_compare":
            pytest.fail("The standalone interactive resource lane has not been implemented")
        raise


@pytest.fixture(autouse=True)
def _no_native_io(monkeypatch):
    import socket

    module = resource_module()

    def forbidden(*args, **kwargs):
        pytest.fail("fake test attempted native I/O")

    monkeypatch.setattr(module.psutil, "Process", forbidden)
    monkeypatch.setattr(module.psutil, "process_iter", forbidden)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden)
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_native_io_guard_is_not_swallowed_by_identity_probe():
    module = resource_module()
    sampler = object.__new__(module.ProcessTreeSampler)
    sampler._process = module.psutil.Process

    with pytest.raises(pytest.fail.Exception):
        sampler._probe_identity(90)


def test_accounting_keeps_simultaneous_peaks_and_exited_child_cpu():
    module = resource_module()
    ledger = module.ResourceLedger((10, 1.0), footprint_supported=True)
    ledger.observe(
        [
            module.ProcessReading(10, 1.0, 100, 0.1, 90),
            module.ProcessReading(20, 2.0, 50, 0.4, 30),
        ]
    )
    ledger.observe([module.ProcessReading(10, 1.0, 120, 0.3, 95)])

    result = ledger.result()

    assert result["peak_tree_rss_bytes"] == 150
    assert result["peak_worker_rss_bytes"] == 120
    assert result["peak_child_tree_rss_bytes"] == 50
    assert result["peak_child_footprint_bytes"] == 30
    assert result["observed_tree_cpu_seconds"] == pytest.approx(0.7)
    assert result["observed_child_cpu_seconds"] == pytest.approx(0.4)
    assert result["observed_processes"] == 2
    assert result["cpu_accounting"] == "sampled_lower_bound"


def test_unknown_footprint_is_not_zero_or_a_complete_peak():
    module = resource_module()
    ledger = module.ResourceLedger((10, 1.0), footprint_supported=True)
    ledger.observe(
        [
            module.ProcessReading(10, 1.0, 100, 0.1, 90),
            module.ProcessReading(20, 2.0, 50, 0.4, None),
        ]
    )

    result = ledger.result()

    assert result["peak_child_footprint_bytes"] is None
    assert result["footprint_status"] == "unavailable"
    assert result["peak_child_tree_rss_bytes"] == 50


def test_empty_sampling_does_not_report_a_zero_cost_win():
    module = resource_module()
    ledger = module.ResourceLedger((10, 1.0), footprint_supported=False)
    ledger.observe([])

    result = ledger.result()

    assert result["peak_tree_rss_bytes"] is None
    assert result["peak_child_tree_rss_bytes"] is None
    assert result["observed_tree_cpu_seconds"] is None
    assert result["peak_child_footprint_bytes"] is None
    assert result["footprint_status"] == "unsupported"


def test_duplicate_process_readings_are_not_double_counted():
    module = resource_module()
    ledger = module.ResourceLedger((10, 1.0), footprint_supported=True)
    reading = module.ProcessReading(10, 1.0, 100, 0.1, 90)
    ledger.observe([reading, reading])

    assert ledger.result()["peak_tree_rss_bytes"] == 100
    assert ledger.result()["observed_processes"] == 1


class FakeProcess:
    def __init__(self, pid, created_at, rss=100, cpu=0.1):
        self.pid = pid
        self.created_at = created_at
        self.rss = rss
        self.cpu = cpu
        self.descendants = []
        self.state = "running"
        self.executable = "python"
        self.name_value: str | None = None
        self.terminate: Callable[[], object] = lambda: pytest.fail("Unexpected fake terminate")
        self.kill: Callable[[], object] = lambda: pytest.fail("Unexpected fake kill")

    def create_time(self):
        return self.created_at

    def is_running(self) -> Any:
        return True

    def status(self):
        return self.state

    def name(self):
        return self.name_value or self.executable

    def exe(self):
        return self.executable

    def cmdline(self):
        return [self.executable]

    def children(self, recursive=True):
        assert recursive
        return self.descendants

    def memory_info(self):
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self):
        return SimpleNamespace(user=self.cpu, system=0.0)

    def uids(self):
        return SimpleNamespace(effective=getattr(os, "geteuid", lambda: 0)())


class DiscoveryProcess:
    def __init__(self, pid, command, created_at: Any = 1.0, uids_value: Any = None):
        self.pid = pid
        self._command = command
        self._created_at = created_at
        self._uids_value = uids_value

    def cmdline(self):
        if callable(self._command):
            return self._command()
        return list(self._command)

    def create_time(self):
        if callable(self._created_at):
            return self._created_at()
        return self._created_at

    def uids(self):
        value = self._uids_value
        if value is None:
            return SimpleNamespace(effective=getattr(os, "geteuid", lambda: 0)())
        if callable(value):
            return value()
        if isinstance(value, int):
            return SimpleNamespace(effective=value)
        return value


def fake_sampler(module, processes, root=10, marker=None):
    def lookup(pid):
        if pid not in processes:
            raise module.psutil.NoSuchProcess(pid)
        return processes[pid]

    return module.ProcessTreeSampler(
        root,
        process_factory=lookup,
        footprint_reader=lambda pid: processes[pid].rss // 2,
        footprint_supported=True,
        ownership_marker=marker,
    )


def test_sampler_keeps_reparented_children_and_discovers_their_new_children():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0, rss=50, cpu=0.2)
    processes = {10: root, 20: child}
    root.descendants = [child]
    sampler = fake_sampler(module, processes)
    sampler.sample()

    grandchild = FakeProcess(30, 3.0, rss=80, cpu=0.3)
    processes.pop(10)
    processes[30] = grandchild
    child.cpu = 0.4
    child.descendants = [grandchild]
    sampler.sample()

    result = sampler.result()
    assert result["peak_tree_rss_bytes"] == 150
    assert result["peak_child_tree_rss_bytes"] == 130
    assert result["observed_tree_cpu_seconds"] == pytest.approx(0.8)
    assert result["observed_processes"] == 3


def test_sampler_does_not_attribute_an_unrelated_reused_pid():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0, rss=50, cpu=0.2)
    processes = {10: root, 20: child}
    root.descendants = [child]
    sampler = fake_sampler(module, processes)
    sampler.sample()

    root.descendants = []
    processes[20] = FakeProcess(20, 99.0, rss=999999, cpu=99.0)
    sampler.sample()

    result = sampler.result()
    assert result["peak_tree_rss_bytes"] == 150
    assert result["observed_tree_cpu_seconds"] == pytest.approx(0.3)
    assert result["observed_processes"] == 2


def test_sampler_reports_access_denied_instead_of_zero_memory():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    processes = {10: root}
    sampler = fake_sampler(module, processes)

    def denied():
        raise module.psutil.AccessDenied(10)

    root.memory_info = denied
    sampler.sample()

    result = sampler.result()
    assert result["peak_tree_rss_bytes"] is None
    assert result["read_errors"] > 0
    assert result["fatal_read_errors"] > 0
    assert result["read_error_types"]["AccessDenied"] > 0


def test_footprint_reader_is_explicitly_unsupported_off_macos():
    module = resource_module()
    assert module.make_footprint_reader(platform_name="linux") is None


def test_macos_footprint_uses_the_documented_v0_structure(monkeypatch):
    module = resource_module()
    calls = []

    def pid_rusage(pid, flavor, buffer):
        calls.append((pid, flavor))
        view = module.ctypes.cast(buffer, module.ctypes.POINTER(module.RUsageInfoV0)).contents
        view.ri_phys_footprint = 4096
        return 0

    monkeypatch.setattr(module.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(proc_pid_rusage=pid_rusage))
    reader = module.make_footprint_reader(platform_name="darwin")

    assert module.ctypes.sizeof(module.RUsageInfoV0) == 96
    assert reader(42) == 4096
    assert calls == [(42, 0)]


def test_plan_rotates_every_arm_through_each_position():
    module = resource_module()
    plan = list(module.sample_plan(["a", "b", "c"], ["tabs"], 3))

    assert [(item[0], item[2]) for item in plan] == [
        (0, "a"),
        (0, "b"),
        (0, "c"),
        (1, "b"),
        (1, "c"),
        (1, "a"),
        (2, "c"),
        (2, "a"),
        (2, "b"),
    ]
    assert [item[3] for item in plan] == [0, 1, 2] * 3


def test_summary_keeps_failures_and_unavailable_resource_metrics():
    module = resource_module()
    records = [
        {
            "task": "tabs",
            "arm": "a",
            "ok": True,
            "resources": {
                "peak_tree_rss_bytes": 100,
                "observed_tree_cpu_seconds": 0.1,
                "peak_child_footprint_bytes": None,
            },
        },
        {
            "task": "tabs",
            "arm": "a",
            "ok": False,
            "error": "oracle failed",
            "resources": {
                "peak_tree_rss_bytes": 900,
                "observed_tree_cpu_seconds": 0.9,
                "peak_child_footprint_bytes": None,
            },
        },
    ]

    result = module.summarize_resources(records, provenance_verified=True)[0]

    assert result["runs"] == 2
    assert result["successes"] == 1
    assert result["failures"] == 1
    assert result["all_correct"] is False
    assert result["metrics"]["peak_tree_rss_bytes"]["median"] == 100
    assert result["metrics"]["peak_child_footprint_bytes"] == {"count": 0, "median": None, "max": None}


def test_worker_uses_the_existing_native_runner_without_changing_its_result(monkeypatch, tmp_path):
    module = resource_module()
    from benchmarks.interactive import run_benchmark

    calls = []
    native = {"task": "tabs", "arm": "browser-use", "ok": False, "oracle_ok": False, "error": "wrong title"}

    def run_one(*args, **kwargs):
        calls.append((args, kwargs))
        return native

    monkeypatch.setattr(run_benchmark, "run_one", run_one)
    destination = tmp_path / "worker.json"

    module.run_worker("tabs", "browser-use", 2, tmp_path / "runs", destination)

    assert json.loads(destination.read_text()) == native
    assert calls == [(("tabs", "browser-use", 2, tmp_path / "runs", False), {"record_binary_version": False})]


def test_lane_writes_every_attempt_and_never_overwrites_evidence(monkeypatch, tmp_path):
    module = resource_module()
    output = tmp_path / "results.jsonl"
    calls = []

    def run_sample(task, arm, sample, **kwargs):
        calls.append((task, arm, sample))
        return {"task": task, "arm": arm, "sample": sample, "ok": arm == "a", "resources": {}}

    metadata = {"source_commit": "abc", "source_digest": "def"}
    monkeypatch.setattr(module, "source_manifest", lambda: metadata)
    monkeypatch.setattr(module, "run_sample", run_sample)
    module.run_lane(["a", "b"], ["tabs"], 2, output, workdir=tmp_path / "runs", interval=0.02, worker_timeout=10)
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert records[0]["record_type"] == "run"
    assert records[0]["source_digest"] == "def"
    assert records[-1]["record_type"] == "end"
    assert records[-1]["observations"] == 4
    assert records[-1]["failures"] == 2
    assert records[-1]["provenance_verified"] is True
    assert calls == [("tabs", "a", 0), ("tabs", "b", 0), ("tabs", "b", 1), ("tabs", "a", 1)]
    assert len([item for item in records if item.get("record_type") == "sample"]) == 4

    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.run_lane(["a"], ["tabs"], 1, output, workdir=tmp_path / "runs", interval=0.02, worker_timeout=10)
    assert output.read_bytes() == before


@pytest.mark.parametrize("native_ok, child_count, browser_count", [(False, 1, 1), (True, 0, 0), (True, 1, 0)])
def test_parent_preserves_oracles_and_rejects_unobserved_product_work(
    monkeypatch, tmp_path, native_ok, child_count, browser_count
):
    module = resource_module()

    class Worker:
        pid = 123
        returncode = 0

        def __init__(self, command, **kwargs):
            destination = Path(command[command.index("--worker-output") + 1])
            destination.write_text(
                json.dumps(
                    {
                        "task": "tabs",
                        "arm": "a",
                        "ok": native_ok,
                        "oracle_ok": native_ok,
                        "error": "wrong title" if not native_ok else "",
                    }
                )
            )

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    class Sampler:
        def __init__(self, pid, *, ownership_marker=None):
            assert pid == 123

        def sample(self, *, phase="native_run"):
            return None

        def live_processes(self, *, phase="native_run"):
            return {}

        def discover_orphans(self):
            return None

        def result(self):
            return {
                "peak_tree_rss_bytes": 100,
                "observed_processes": 1 + child_count,
                "observed_child_processes": child_count,
                "observed_browser_processes": browser_count,
                "observed_browser_during_native_run": browser_count,
                "read_errors": 0,
                "fatal_read_errors": 0,
            }

    monkeypatch.setattr(module.subprocess, "Popen", Worker)
    monkeypatch.setattr(module, "ProcessTreeSampler", Sampler)

    result = module.run_sample("tabs", "a", 0, workdir=tmp_path, interval=0.02, worker_timeout=10)

    assert result["ok"] is native_ok
    assert result["oracle_ok"] is native_ok
    assert result["resource_ok"] is bool(browser_count)
    assert result["resources"]["peak_tree_rss_bytes"] == 100


def test_zombie_is_not_a_live_product_or_a_cleanup_leak():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0, rss=999)
    child.state = module.psutil.STATUS_ZOMBIE
    root.descendants = [child]
    sampler = fake_sampler(module, {10: root, 20: child})

    sampler.sample()

    assert sampler.result()["peak_tree_rss_bytes"] == 100
    assert sampler.result()["observed_child_processes"] == 0


@pytest.mark.parametrize("interval, timeout", [(0.02, 0.1), (10.0, 1.0)])
def test_worker_deadline_is_a_failed_observation_and_kills_only_its_worker(monkeypatch, tmp_path, interval, timeout):
    module = resource_module()
    workers = []

    class Worker:
        pid = 123
        returncode = None

        def __init__(self, *args, **kwargs):
            self.killed = False
            workers.append(self)

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            assert timeout == 5
            return self.returncode

    class Sampler:
        def __init__(self, pid, *, ownership_marker=None):
            assert pid == 123

        def sample(self, *, phase="native_run"):
            return None

        def live_processes(self, *, phase="native_run"):
            return {}

        def discover_orphans(self):
            return None

        def result(self):
            return {
                "peak_tree_rss_bytes": None,
                "read_errors": 0,
                "fatal_read_errors": 0,
                "observed_child_processes": 0,
            }

    clock = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    monkeypatch.setattr(module.subprocess, "Popen", Worker)
    monkeypatch.setattr(module, "ProcessTreeSampler", Sampler)

    record = module.run_sample("tabs", "a", 0, workdir=tmp_path, interval=interval, worker_timeout=timeout)

    assert record["ok"] is False
    assert record["worker_timed_out"] is True
    assert record["resource_ok"] is False
    assert record["error"] == "Worker exceeded its deadline"
    assert workers[0].killed is True
    assert sum(sleeps) == pytest.approx(timeout)


def test_identity_evidence_distinguishes_browser_work_from_an_incidental_helper():
    module = resource_module()
    ledger = module.ResourceLedger((10, 1.0), footprint_supported=True)
    ledger.observe(
        [
            module.ProcessReading(10, 1.0, 100, 0.1, 90, name="python"),
            module.ProcessReading(20, 2.0, 50, 0.4, 30, name="ps"),
        ]
    )

    result = ledger.result()

    assert result["observed_browser_processes"] == 0
    assert result["processes"][1]["pid"] == 20
    assert result["processes"][1]["created_at"] == 2.0
    assert result["processes"][1]["name"] == "ps"
    assert result["processes"][1]["observed_cpu_seconds"] == 0.4


def test_sampler_records_a_main_browser_but_does_not_mislabel_crashpad():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"
    crashpad = FakeProcess(30, 3.0)
    crashpad.executable = "chrome_crashpad_handler"
    root.descendants = [browser, crashpad]
    sampler = fake_sampler(module, {10: root, 20: browser, 30: crashpad})

    sampler.sample()

    assert sampler.result()["observed_browser_processes"] == 1


@pytest.mark.parametrize("option,value", [("--interval", "nan"), ("--worker-timeout", "inf"), ("--interval", "-inf")])
def test_cli_rejects_nonfinite_limits_before_starting_any_worker(monkeypatch, tmp_path, capsys, option, value):
    module = resource_module()
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["resources", "--arms", "browser-use", "--output", str(tmp_path / "raw.jsonl"), option + "=" + value],
    )
    monkeypatch.setattr(
        module, "run_lane", lambda *args, **kwargs: pytest.fail("Invalid limits reached the worker lane")
    )

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert "finite" in capsys.readouterr().err


def fake_clock(monkeypatch, module):
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_cleanup_reports_a_child_that_survives_term_and_kill(monkeypatch):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    root.descendants = [child]
    processes = {10: root, 20: child}
    sampler = fake_sampler(module, processes)
    sampler.sample()
    processes.pop(10)
    signals = []
    child.terminate = lambda: signals.append("term")
    child.kill = lambda: signals.append("kill")
    fake_clock(monkeypatch, module)

    result = module.cleanup_owned(sampler, interval=0.1, timeout=1.0)

    assert result["verified"] is False
    assert result["remaining"] == [{"pid": 20, "created_at": 2.0}]
    assert signals == ["term", "kill"]


def test_cleanup_samples_termination_work_and_new_descendants(monkeypatch):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0, rss=50, cpu=0.1)
    root.descendants = [child]
    processes = {10: root, 20: child}
    sampler = fake_sampler(module, processes)
    sampler.sample()
    processes.pop(10)
    grandchild = FakeProcess(30, 3.0, rss=200, cpu=0.2)

    def terminate():
        child.rss = 300
        child.cpu = 0.9
        child.descendants = [grandchild]
        processes[30] = grandchild

    child.terminate = terminate
    child.kill = lambda: processes.pop(20, None)
    grandchild.terminate = lambda: None
    grandchild.kill = lambda: processes.pop(30, None)
    fake_clock(monkeypatch, module)

    result = module.cleanup_owned(sampler, interval=0.1, timeout=1.0)

    assert result["verified"] is True
    assert result["remaining"] == []
    assert sampler.result()["peak_child_tree_rss_bytes"] == 500
    assert sampler.result()["observed_child_cpu_seconds"] == pytest.approx(1.1)
    assert all(
        item["observed_during_native_run"] is False for item in sampler.result()["processes"] if item["pid"] == 30
    )


def test_lane_persists_a_parent_failure_and_stops_if_cleanup_is_unverified(monkeypatch, tmp_path):
    module = resource_module()
    output = tmp_path / "failed.jsonl"
    calls = []

    def crash(task, arm, sample, **kwargs):
        calls.append(arm)
        raise RuntimeError("supervision failed")

    monkeypatch.setattr(module, "source_manifest", lambda: {})
    monkeypatch.setattr(module, "arm_versions", lambda arms: {})
    monkeypatch.setattr(module, "run_sample", crash)

    records = module.run_lane(
        ["a", "b"], ["tabs"], 1, output, workdir=tmp_path / "runs", interval=0.02, worker_timeout=1
    )
    raw = [json.loads(line) for line in output.read_text().splitlines()]

    assert calls == ["a"]
    assert records[0]["ok"] is False
    assert records[0]["cleanup"]["verified"] is False
    assert "RuntimeError" in records[0]["error"]
    assert raw[-1]["record_type"] == "end"
    assert raw[-1]["aborted"] is True
    assert raw[-1]["abort_reason"] == "cleanup_unverified"
    assert raw[0]["expected_observations"] == 2
    assert raw[-1]["expected_observations"] == 2
    assert raw[-1]["valid"] is False
    assert raw[-1]["terminal"] is True
    assert raw[-1]["cleanup_verified"] is False
    assert raw[-1]["fixture_errors"] == []


def test_spawn_failure_is_a_durable_failed_attempt_without_live_measurements(monkeypatch, tmp_path):
    module = resource_module()

    def cannot_spawn(*args, **kwargs):
        raise FileNotFoundError("worker interpreter missing")

    monkeypatch.setattr(module.subprocess, "Popen", cannot_spawn)

    record = module.run_sample("tabs", "a", 0, workdir=tmp_path, interval=0.02, worker_timeout=1)

    assert record["ok"] is False
    assert record["cleanup"]["verified"] is True
    assert record["resource_ok"] is False
    assert "FileNotFoundError" in record["error"]


def test_unverified_provenance_cannot_produce_a_comparative_summary():
    module = resource_module()
    records = [{"task": "tabs", "arm": "a", "ok": True, "resource_ok": True, "resources": {"peak_tree_rss_bytes": 100}}]

    result = module.summarize_resources(records)[0]

    assert result["successes"] == 1
    assert result["resource_eligible"] == 0
    assert result["provenance_verified"] is False
    assert result["metrics"]["peak_tree_rss_bytes"]["count"] == 0


def test_lane_rechecks_actual_binary_binding_and_excludes_changed_source(monkeypatch, tmp_path):
    module = resource_module()
    output = tmp_path / "changed.jsonl"
    bindings = iter(
        [
            {"verified": True, "entry_sha256": "entry", "actual_app_digest": "before"},
            {"verified": True, "entry_sha256": "entry", "actual_app_digest": "after"},
        ]
    )
    monkeypatch.setattr(module, "source_manifest", lambda: {"source_digest": "stable"})
    monkeypatch.setattr(module, "binding_snapshot", lambda arms: next(bindings))
    monkeypatch.setattr(module, "arm_versions", lambda arms: {})
    monkeypatch.setattr(
        module,
        "run_sample",
        lambda task, arm, sample, **kw: {
            "task": task,
            "arm": arm,
            "ok": True,
            "resource_ok": True,
            "cleanup": {"verified": True},
            "resources": {"peak_tree_rss_bytes": 100},
        },
    )

    records = module.run_lane(["a"], ["tabs"], 1, output, workdir=tmp_path / "runs", interval=0.02, worker_timeout=1)
    raw = [json.loads(line) for line in output.read_text().splitlines()]

    assert records[0]["ok"] is True
    assert records[0]["provenance_verified"] is False
    assert raw[1]["provisional"] is True
    assert raw[-1]["provenance_verified"] is False
    assert raw[-1]["binding_unchanged"] is False


def test_binding_uses_the_adapter_environment_without_starting_application(monkeypatch, tmp_path):
    module = resource_module()
    from benchmarks.interactive import resource_provenance

    observed = []

    def bind(binary, expected_package, *, env, cwd):
        observed.append((binary, expected_package, env, cwd))
        return {"verified": True}

    monkeypatch.setenv("OPENBROWSER_BIN", "/fake/openbrowser-ai")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source")
    monkeypatch.setattr(resource_provenance, "bind_console_entry", bind)

    def legacy_must_not_run(*args, **kwargs):
        raise AssertionError("Resource binding must dispatch through the native-aware facade")

    monkeypatch.setattr(resource_provenance, "bind_openbrowser", legacy_must_not_run)

    assert module.binding_snapshot(["openbrowser-fast"])["verified"] is True
    assert observed[0][0] == "/fake/openbrowser-ai"
    assert observed[0][1] == module.ROOT / "src" / "openbrowser"
    assert "PYTHONPATH" not in observed[0][2]
    assert observed[0][2]["HOME"] != str(Path.home())


def test_source_manifest_separates_harness_and_application_content(monkeypatch, tmp_path):
    module = resource_module()
    package = tmp_path / "src" / "openbrowser"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("value = 1\n")
    harness = tmp_path / "benchmarks" / "interactive"
    harness.mkdir(parents=True)
    (harness / "runner.py").write_text("oracle = 1\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="commit"))
    first = module.source_manifest()
    (package / "__init__.py").write_text("value = 2\n")
    second = module.source_manifest()

    assert first["harness_digest"] == second["harness_digest"]
    assert first["expected_app_digest"] != second["expected_app_digest"]
    assert first["source_digest"] != second["source_digest"]
    assert first["sampler_method_version"] == module.SAMPLER_METHOD_VERSION == 6
    assert first["configuration_digest"] == second["configuration_digest"]
    assert any("effective UID" in item and "Known ancestry" in item for item in first["limitations"])


def test_cleanup_finds_a_marked_orphan_created_as_its_parent_exits(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    orphan = FakeProcess(30, 3.0)
    orphan.executable = "Google Chrome"
    unrelated = FakeProcess(40, 4.0)
    root.descendants = [child]
    processes = {10: root, 20: child, 40: unrelated}
    sampler = fake_sampler(module, processes, marker=marker)
    sampler.sample()
    processes.pop(10)

    def terminate():
        processes.pop(20)
        processes[30] = orphan

    child.terminate = terminate
    orphan.terminate = lambda: None
    orphan.kill = lambda: processes.pop(30, None)

    def process_iter(attrs):
        assert attrs == ["pid"]
        return [
            DiscoveryProcess(
                pid,
                [process.executable, "--user-data-dir=" + (marker if pid == 30 else "/other")],
                process.created_at,
            )
            for pid, process in processes.items()
        ]

    monkeypatch.setattr(module.psutil, "process_iter", process_iter)
    fake_clock(monkeypatch, module)

    result = module.cleanup_owned(sampler, interval=0.1, timeout=1.0)

    assert result["verified"] is True
    assert set(processes) == {40}
    evidence = next(item for item in sampler.result()["processes"] if item["pid"] == 30)
    assert evidence["observed_during_native_run"] is False


def test_discover_orphans_latches_iterator_oserror_as_unknown(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)

    def fail_iter(attrs):
        assert attrs == ["pid"]
        if False:
            yield None
        raise OSError(0, "Undefined error: 0 (originated from sysctl(KERN_PROCARGS2))")

    monkeypatch.setattr(module.psutil, "process_iter", fail_iter)
    sampler.discover_orphans()
    first = sampler.result()

    assert first["fatal_read_errors"] == 1
    assert first["unverified_processes"] == 1
    assert first["cleanup_unverified_processes"] == 1
    assert first["fatal_read_identities"] == [{"pid": None, "created_at": None}]
    detail = first["read_error_details"][0]
    assert detail["pid"] is None
    assert detail["created_at"] is None
    assert detail["operation"] == "process_iter"
    assert detail["phase"] == "orphan_discovery"
    assert detail["error_type"] == "OSError"
    assert detail["errno"] == 0
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "discovery_scan"
    assert "error" not in detail

    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [])
    sampler.discover_orphans()
    second = sampler.result()
    assert second["fatal_read_errors"] == 1
    assert second["unverified_processes"] == 1
    assert second["cleanup_unverified_processes"] == 0


def test_discover_orphans_records_cmdline_oserror_with_known_pid(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)

    def fail_cmdline():
        raise OSError(0, "Undefined error: 0 (originated from sysctl(KERN_PROCARGS2))")

    process = DiscoveryProcess(20, fail_cmdline, lambda: pytest.fail("create_time must not run"))
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])
    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    assert result["cleanup_unverified_processes"] == 1
    assert result["fatal_read_identities"] == [{"pid": 20, "created_at": None}]
    assert 20 in sampler._marker_unverified
    assert all(pid != 20 for pid, _ in sampler._known)
    detail = result["read_error_details"][0]
    assert detail["pid"] == 20
    assert detail["created_at"] is None
    assert detail["operation"] == "cmdline"
    assert detail["effective_uid"] == os.geteuid()
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "discovery_scan"


def test_discover_orphans_records_owned_create_time_failure(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)

    def fail_time():
        raise OSError(0, "create time unavailable")

    process = DiscoveryProcess(30, ["Chrome", f"--user-data-dir={marker}"], fail_time)
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])
    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    assert result["cleanup_unverified_processes"] == 1
    assert 30 in sampler._marker_unverified
    assert all(pid != 30 for pid, _ in sampler._known)
    detail = result["read_error_details"][0]
    assert detail["pid"] == 30
    assert detail["created_at"] is None
    assert detail["operation"] == "create_time"
    assert detail["effective_uid"] == os.geteuid()
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "discovery_scan"


def test_discover_orphans_accepts_only_marker_matches(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)
    matched = DiscoveryProcess(40, ["Chrome", f"--user-data-dir={marker}"], 4.5)
    unrelated = DiscoveryProcess(41, ["Chrome", "--user-data-dir=/other"], 5.0)
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [matched, unrelated])

    sampler.discover_orphans()

    result = sampler.result()
    assert (40, 4.5) in sampler._known
    assert (41, 5.0) not in sampler._known
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["unverified_processes"] == 0
    assert result["cleanup_unverified_processes"] == 0
    assert result["discovery_uid_stats"]["same_uid"] == 2


def test_discover_orphans_skips_exact_integer_pid_zero(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)
    process = DiscoveryProcess(
        0,
        lambda: pytest.fail("pid zero must not read cmdline"),
        lambda: pytest.fail("pid zero must not read create_time"),
        lambda: pytest.fail("pid zero must not read uids"),
    )
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])

    sampler.discover_orphans()

    result = sampler.result()
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["discovery_uid_stats"]["pid_zero_skipped"] == 1
    assert all(pid != 0 for pid, _ in sampler._known)


@pytest.mark.parametrize("pid", [False, -1, None, 0.0])
def test_discover_orphans_rejects_malformed_pid(monkeypatch, pid):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)
    process = DiscoveryProcess(
        pid,
        lambda: pytest.fail("malformed pid must not read cmdline"),
        lambda: pytest.fail("malformed pid must not read create_time"),
        lambda: pytest.fail("malformed pid must not read uids"),
    )
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])

    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    assert result["discovery_uid_stats"]["invalid_pid"] == 1
    detail = result["read_error_details"][0]
    assert detail["operation"] == "pid"
    assert detail["pid"] is None
    assert detail["created_at"] is None
    assert detail["effective_uid"] is None


def test_discover_orphans_skips_foreign_uid_before_cmdline(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)
    process = DiscoveryProcess(
        60,
        lambda: pytest.fail("foreign uid must not read cmdline"),
        lambda: pytest.fail("foreign uid must not read create_time"),
        os.geteuid() + 1,
    )
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])

    sampler.discover_orphans()

    result = sampler.result()
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["discovery_uid_stats"]["foreign_uid"] == 1
    assert all(pid != 60 for pid, _ in sampler._known)


@pytest.mark.parametrize(
    ("uids_value", "error_type"),
    [
        (lambda: (_ for _ in ()).throw(OSError(0, "uid unavailable")), "OSError"),
        (SimpleNamespace(effective=None), "ValueError"),
        (SimpleNamespace(effective="501"), "ValueError"),
    ],
)
def test_discover_orphans_unknown_uid_is_fatal(monkeypatch, uids_value, error_type):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)
    process = DiscoveryProcess(
        70,
        lambda: pytest.fail("unknown uid must not read cmdline"),
        lambda: pytest.fail("unknown uid must not read create_time"),
        uids_value,
    )
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])

    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["discovery_uid_stats"]["unknown_uid"] == 1
    detail = result["read_error_details"][0]
    assert detail["operation"] == "uids"
    assert detail["pid"] == 70
    assert detail["created_at"] is None
    assert detail["effective_uid"] is None
    assert detail["error_type"] == error_type


def test_discover_orphans_keeps_conservative_policy_off_posix(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)
    process = DiscoveryProcess(
        80,
        ["Chrome", f"--user-data-dir={marker}"],
        8.5,
        lambda: pytest.fail("non-POSIX discovery must not read uids"),
    )
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])
    monkeypatch.setattr(module.os, "name", "nt")

    sampler.discover_orphans()

    result = sampler.result()
    assert (80, 8.5) in sampler._known
    assert result["read_errors"] == 0
    assert result["discovery_uid_stats"]["unfiltered"] == 1


def test_foreign_uid_does_not_remove_known_descendant():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0, rss=50, cpu=0.2)
    child.uids = lambda: SimpleNamespace(effective=os.geteuid() + 1)
    root.descendants = [child]
    sampler = fake_sampler(module, {10: root, 20: child})

    sampler.sample()

    result = sampler.result()
    assert result["peak_tree_rss_bytes"] == 150
    assert result["observed_child_processes"] == 1
    assert result["fatal_read_errors"] == 0


def test_discover_orphans_same_uid_cmdline_denied_is_fatal(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)

    def fail_cmdline():
        raise module.psutil.AccessDenied(50)

    process = DiscoveryProcess(50, fail_cmdline, lambda: pytest.fail("create_time must not run"))
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])
    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    assert result["cleanup_unverified_processes"] == 1
    assert result["discovery_uid_stats"]["same_uid"] == 1
    assert 50 in sampler._marker_unverified
    detail = result["read_error_details"][0]
    assert detail["operation"] == "cmdline"
    assert detail["error_type"] == "AccessDenied"
    assert detail["effective_uid"] == os.geteuid()
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "discovery_scan"


@pytest.mark.parametrize("exception_name", ["NoSuchProcess", "ZombieProcess"])
def test_discover_orphans_preserves_authoritative_exit_policy(monkeypatch, exception_name):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)

    def fail_cmdline():
        raise getattr(module.psutil, exception_name)(50)

    process = DiscoveryProcess(50, fail_cmdline, lambda: pytest.fail("create_time must not run"))
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [process])
    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 0
    assert 50 not in sampler._marker_unverified
    assert result["cleanup_unverified_processes"] == 0


def discovery_probe_sampler(module, monkeypatch, fresh_sequence, *, pids=(90,), cmdline_error=None):
    marker = "/tmp/resource-owned-abcdef/runs"
    processes = []
    for pid in pids:

        def fail_cmdline(pid=pid):
            if cmdline_error is not None:
                raise cmdline_error
            raise module.psutil.AccessDenied(pid)

        processes.append(
            DiscoveryProcess(
                pid,
                fail_cmdline,
                lambda: pytest.fail("iterator create_time must not run"),
                os.geteuid(),
            )
        )
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: list(processes))
    factory_calls = []

    def process_factory(pid):
        if pid == 10:
            return FakeProcess(10, 1.0)
        factory_calls.append(pid)
        if fresh_sequence:
            value = fresh_sequence.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        return FakeProcess(pid, 2.0)

    sampler = module.ProcessTreeSampler(
        10,
        process_factory=process_factory,
        footprint_reader=None,
        footprint_supported=False,
        ownership_marker=marker,
    )
    return sampler, factory_calls


def test_discovery_probe_uses_fresh_identity_and_keeps_fatal(monkeypatch):
    module = resource_module()
    sampler, calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), FakeProcess(90, 2.0)],
    )
    sampler.set_capture_phase("baseline")

    sampler.discover_orphans()

    result = sampler.result()
    assert calls == [90, 90]
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    assert result["cleanup_unverified_processes"] == 1
    detail = result["read_error_details"][0]
    assert detail["created_at"] is None
    probe = detail["discovery_probe"]
    assert probe["cause_tag"] == "access_denied_underlying_cause_unavailable"
    assert probe["capture_phase"] == "baseline"
    assert probe["before"]["ctime"] == 2.0
    assert probe["before"]["status"] == "running"
    assert probe["immediate"]["ctime"] == 2.0
    assert probe["immediate"]["same_identity"] is True
    assert probe["immediate"]["state"] == "same_alive"
    payload = json.dumps(result)
    assert "EINVAL" not in payload
    assert "EIO" not in payload
    assert "force permission denied" not in payload
    with pytest.raises(ValueError):
        sampler.set_capture_phase("bogus")


@pytest.mark.parametrize(
    ("after_value", "expected_state", "expected_same_identity"),
    [
        ("gone", "gone", None),
        ("reused", "reused", False),
        ("malformed", "unknown", None),
    ],
)
def test_discovery_immediate_fence_distinguishes_identity_states(
    monkeypatch, after_value, expected_state, expected_same_identity
):
    module = resource_module()
    if after_value == "gone":
        after = module.psutil.NoSuchProcess(90)
    elif after_value == "reused":
        after = FakeProcess(90, 99.0)
    else:
        after = FakeProcess(90, None)
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), after],
    )

    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    immediate = result["read_error_details"][0]["discovery_probe"]["immediate"]
    assert immediate["state"] == expected_state
    assert immediate["same_identity"] is expected_same_identity
    if after_value == "gone":
        assert immediate["ctime"] is None
        assert immediate["status"] is None
        assert immediate["running"] is False
    if after_value == "malformed":
        assert immediate["ctime"] is None
        assert immediate["error_type"] == "ValueError"


def test_discovery_malformed_running_is_unknown(monkeypatch):
    module = resource_module()
    after = FakeProcess(90, 2.0)
    after.is_running = lambda: 1
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), after],
    )

    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    immediate = result["read_error_details"][0]["discovery_probe"]["immediate"]
    assert immediate["running"] is None
    assert immediate["error_type"] == "ValueError"
    assert immediate["same_identity"] is True
    assert immediate["state"] == "unknown"


def test_discovery_wrong_factory_pid_is_unknown(monkeypatch):
    module = resource_module()
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(91, 2.0), FakeProcess(90, 2.0)],
    )

    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    probe = result["read_error_details"][0]["discovery_probe"]
    assert probe["before"]["ctime"] is None
    assert probe["before"]["status"] is None
    assert probe["before"]["error_type"] == "ValueError"
    assert probe["immediate"]["same_identity"] is None
    assert probe["immediate"]["state"] == "unknown"


def test_discovery_mismatched_exit_pid_is_not_authoritative(monkeypatch):
    module = resource_module()
    after = FakeProcess(90, 2.0)

    def wrong_pid():
        raise module.psutil.NoSuchProcess(91)

    after.create_time = wrong_pid
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), after],
    )

    sampler.discover_orphans()

    result = sampler.result()
    immediate = result["read_error_details"][0]["discovery_probe"]["immediate"]
    assert immediate["ctime"] is None
    assert immediate["error_type"] == "NoSuchProcess"
    assert immediate["running"] is True
    assert immediate["same_identity"] is None
    assert immediate["state"] == "unknown"
    assert result["fatal_read_errors"] == 1


def test_discovery_authoritative_exit_stops_later_reads(monkeypatch):
    module = resource_module()
    after = FakeProcess(90, 2.0)
    after.create_time = lambda: (_ for _ in ()).throw(module.psutil.NoSuchProcess(90))
    after.status = lambda: pytest.fail("status must not run after authoritative exit")
    after.is_running = lambda: pytest.fail("running must not run after authoritative exit")
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), after],
    )

    sampler.discover_orphans()

    result = sampler.result()
    immediate = result["read_error_details"][0]["discovery_probe"]["immediate"]
    assert immediate["ctime"] is None
    assert immediate["status"] is None
    assert immediate["running"] is False
    assert immediate["error_type"] == "NoSuchProcess"
    assert immediate["state"] == "gone"
    assert result["fatal_read_errors"] == 1


@pytest.mark.parametrize(
    ("after_ctime", "expected_state", "expected_same_identity"),
    [(2.0, "unknown", True), (99.0, "reused", False)],
)
def test_discovery_status_denied_uses_only_complete_identity_proof(
    monkeypatch, after_ctime, expected_state, expected_same_identity
):
    module = resource_module()
    after = FakeProcess(90, after_ctime)

    def denied_status():
        raise module.psutil.AccessDenied(90)

    after.status = denied_status
    after.is_running = lambda: True
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), after],
    )

    sampler.discover_orphans()

    result = sampler.result()
    immediate = result["read_error_details"][0]["discovery_probe"]["immediate"]
    assert immediate["running"] is True
    assert immediate["status"] is None
    assert immediate["error_type"] == "AccessDenied"
    assert immediate["same_identity"] is expected_same_identity
    assert immediate["state"] == expected_state
    assert result["fatal_read_errors"] == 1


def test_discovery_before_identity_denied_keeps_primary_fatal(monkeypatch):
    module = resource_module()
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [module.psutil.AccessDenied(90), FakeProcess(90, 2.0)],
    )

    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    probe = result["read_error_details"][0]["discovery_probe"]
    assert probe["before"]["ctime"] is None
    assert probe["before"]["status"] is None
    assert probe["before"]["error_type"] == "AccessDenied"
    assert probe["immediate"]["same_identity"] is None
    assert probe["immediate"]["state"] == "unknown"


def test_discovery_oserror_cause_tag_and_errno_are_preserved(monkeypatch):
    module = resource_module()
    sampler, _calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), FakeProcess(90, 2.0)],
        cmdline_error=OSError(0, "sysctl(KERN_PROCARGS2)"),
    )

    sampler.discover_orphans()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    detail = result["read_error_details"][0]
    assert detail["errno"] == 0
    assert detail["discovery_probe"]["cause_tag"] == "oserror"
    payload = json.dumps(result)
    assert "EINVAL" not in payload
    assert "EIO" not in payload


def test_diagnostic_probe_denied_does_not_change_primary_success(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    iterator_process = DiscoveryProcess(
        90,
        ["Chrome", f"--user-data-dir={marker}"],
        3.0,
        os.geteuid(),
    )
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [iterator_process])

    def process_factory(pid):
        if pid == 10:
            return FakeProcess(10, 1.0)
        raise module.psutil.AccessDenied(pid)

    sampler = module.ProcessTreeSampler(
        10,
        process_factory=process_factory,
        footprint_reader=None,
        footprint_supported=False,
        ownership_marker=marker,
    )

    sampler.discover_orphans()

    result = sampler.result()
    assert (90, 3.0) in sampler._known
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["unverified_processes"] == 0
    assert result["cleanup_unverified_processes"] == 0
    assert result["discovery_end_fences"] == []


def test_discovery_end_fence_is_linked_and_can_show_gone(monkeypatch):
    module = resource_module()
    sampler, calls = discovery_probe_sampler(
        module,
        monkeypatch,
        [FakeProcess(90, 2.0), FakeProcess(90, 2.0), module.psutil.NoSuchProcess(90)],
    )
    sampler.set_capture_phase("final")
    clock = iter([100.123456789, 100.987654321])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    sampler.discover_orphans()
    probe = sampler.result()["read_error_details"][0]["discovery_probe"]
    baseline_probe = json.loads(json.dumps(probe))
    sampler.set_capture_phase("native_run")

    sampler.record_discovery_end_fences()

    result = sampler.result()
    assert calls == [90, 90, 90]
    assert len(result["discovery_end_fences"]) == 1
    fence = result["discovery_end_fences"][0]
    assert fence["probe_id"] == probe["probe_id"]
    assert fence["pid"] == 90
    assert fence["capture_phase"] == "final"
    assert fence["state"] == "gone"
    assert fence["same_identity"] is None
    assert fence["fence"]["ctime"] is None
    assert fence["fence"]["status"] is None
    assert fence["fence"]["running"] is False
    assert fence["fence"]["error_type"] == "NoSuchProcess"
    expected_elapsed = 100.987654321 - 100.123456789
    assert fence["elapsed_s"] == pytest.approx(expected_elapsed, abs=1e-12)
    assert fence["elapsed_s"] != round(expected_elapsed, 6)
    assert result["fatal_read_errors"] == 1
    assert result["discovery_end_fences_dropped"] == 0
    assert result["read_error_details"][0]["discovery_probe"] == baseline_probe

    sampler.record_discovery_end_fences()
    assert len(sampler.result()["discovery_end_fences"]) == 1


def test_discovery_end_fences_are_bounded_and_results_are_immutable(monkeypatch):
    module = resource_module()
    monkeypatch.setattr(module, "MAX_DISCOVERY_END_FENCES", 2)
    sampler, _calls = discovery_probe_sampler(module, monkeypatch, [], pids=(90, 91, 92))

    sampler.discover_orphans()
    sampler.record_discovery_end_fences()
    first = sampler.result()

    assert len(first["discovery_end_fences"]) == 2
    assert first["discovery_end_fences_dropped"] == 1
    detail_probe = first["read_error_details"][0]["discovery_probe"]
    end_state = first["discovery_end_fences"][0]["state"]
    assert detail_probe["immediate"]["same_identity"] is True
    assert end_state == "same_alive"
    first["read_error_details"][0]["discovery_probe"]["immediate"]["ctime"] = 999.0
    first["discovery_end_fences"][0]["state"] = "tampered"
    second = sampler.result()
    assert second["read_error_details"][0]["discovery_probe"]["immediate"]["ctime"] == 2.0
    assert second["discovery_end_fences"][0]["state"] == end_state


def test_cleanup_reports_discovery_oserror_without_escaping(monkeypatch):
    module = resource_module()
    marker = "/tmp/resource-owned-abcdef/runs"
    sampler = fake_sampler(module, {}, root=10, marker=marker)

    def fail_iter(attrs):
        assert attrs == ["pid"]
        if False:
            yield None
        raise OSError(0, "Undefined error: 0 (originated from sysctl(KERN_PROCARGS2))")

    monkeypatch.setattr(module.psutil, "process_iter", fail_iter)
    fake_clock(monkeypatch, module)

    result = module.cleanup_owned(sampler, interval=0.05, timeout=0.1)

    assert result["verified"] is False
    assert result["unverified_processes"] == 1
    assert sampler.result()["fatal_read_errors"] >= 1


@pytest.mark.parametrize(
    ("exception_name", "expected_resolution"),
    [("NoSuchProcess", "gone"), ("ZombieProcess", "zombie")],
)
def test_authoritative_process_exit_is_expected(exception_name, expected_resolution):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0, rss=200, cpu=0.2)
    child.executable = "Google Chrome"
    root.descendants = [child]
    sampler = fake_sampler(module, {10: root, 20: child})
    sampler.sample()

    def exited():
        raise getattr(module.psutil, exception_name)(20)

    child.memory_info = exited
    sampler.sample()
    result = sampler.result()

    assert result["read_errors"] == 1
    assert result["expected_exit_races"] == 1
    assert result["fatal_read_errors"] == 0
    assert result["read_error_types"] == {exception_name: 1}
    detail = result["read_error_details"][0]
    assert detail["resolution"] == expected_resolution
    assert detail["resolution_source"] == "read_exception"
    assert detail["resolution_error_type"] == exception_name
    assert result["peak_tree_rss_bytes"] == 300
    assert result["observed_browser_during_native_run"] == 1
    assert result["cpu_accounting"] == "sampled_lower_bound"


@pytest.mark.parametrize("exception_name", ["NoSuchProcess", "ZombieProcess"])
def test_access_denied_after_process_exit_is_expected_by_identity_fence(exception_name):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    child.executable = "Google Chrome"
    root.descendants = [child]
    processes = {10: root, 20: child}
    sampler = fake_sampler(module, processes)
    sampler.sample()

    def denied():
        if exception_name == "NoSuchProcess":
            processes.pop(20)
        else:
            child.state = module.psutil.STATUS_ZOMBIE
        raise module.psutil.AccessDenied(20)

    child.memory_info = denied
    sampler.sample()
    result = sampler.result()

    assert result["expected_exit_races"] == 1
    assert result["fatal_read_errors"] == 0
    detail = result["read_error_details"][0]
    assert detail["error_type"] == "AccessDenied"
    assert detail["resolution"] in {"gone", "zombie"}
    assert detail["resolution_source"] == "identity_fence"


def test_no_such_process_from_cmdline_is_authoritative_exit():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"

    def missing():
        raise module.psutil.NoSuchProcess(20)

    browser.cmdline = missing
    root.descendants = [browser]
    sampler = fake_sampler(module, {10: root, 20: browser})
    sampler.sample()

    result = sampler.result()
    assert result["expected_exit_races"] == 1
    assert result["fatal_read_errors"] == 0
    detail = result["read_error_details"][0]
    assert detail["operation"] == "cmdline"
    assert detail["resolution"] == "gone"
    assert detail["resolution_source"] == "read_exception"


def test_authoritative_exit_with_malformed_identity_stays_fatal():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    sampler = fake_sampler(module, {10: root})

    sampler._note_read_error(
        module.psutil.NoSuchProcess(20),
        identity=(20, math.nan),
        operation="memory_info",
        phase="native_run",
        sample_index=0,
    )

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "identity_fence"


@pytest.mark.parametrize("error_name", ["NoSuchProcess", "ZombieProcess"])
def test_authoritative_exit_does_not_call_identity_fence(error_name):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    sampler = fake_sampler(module, {10: root})

    def fail(_identity):
        pytest.fail("authoritative exit must not call the identity fence")

    sampler._identity_resolution = fail  # type: ignore[method-assign]
    sampler._note_read_error(
        getattr(module.psutil, error_name)(20),
        identity=(20, 2.0),
        operation="memory_info",
        phase="native_run",
        sample_index=0,
    )

    result = sampler.result()
    assert result["expected_exit_races"] == 1
    assert result["fatal_read_errors"] == 0
    assert result["read_error_details"][0]["resolution_source"] == "read_exception"


@pytest.mark.parametrize(
    ("error_name", "operation"),
    [("AccessDenied", "cmdline"), ("OSError", "memory_info")],
)
def test_ambiguous_read_error_uses_identity_fence(error_name, operation):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    sampler = fake_sampler(module, {10: root})
    calls = []

    def fence(identity):
        calls.append(identity)
        return "alive", None

    sampler._identity_resolution = fence  # type: ignore[method-assign]
    error = module.psutil.AccessDenied(20) if error_name == "AccessDenied" else OSError(0, "ambiguous")
    sampler._note_read_error(
        error,
        identity=(20, 2.0),
        operation=operation,
        phase="native_run",
        sample_index=0,
    )

    result = sampler.result()
    assert calls == [(20, 2.0)]
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["memory_task_verification"] == "not_run"


def memory_task_sampler(module, second):
    root = FakeProcess(10, 1.0, rss=100)
    child = FakeProcess(20, 2.0, rss=200)
    root.descendants = [child]
    calls = []

    def memory_info():
        calls.append(None)
        if len(calls) == 1:
            raise module.psutil.AccessDenied(20)
        return second()

    child.memory_info = memory_info
    sampler = fake_sampler(module, {10: root, 20: child})
    return sampler, calls


def raise_error(error):
    def second():
        raise error

    return second


@pytest.mark.parametrize(
    ("error_name", "resolution", "verification"),
    [("NoSuchProcess", "gone", "no_such_process"), ("ZombieProcess", "zombie", "zombie_process")],
)
def test_memory_task_proof_accepts_authoritative_exit(error_name, resolution, verification):
    module = resource_module()
    sampler, calls = memory_task_sampler(module, raise_error(getattr(module.psutil, error_name)(20)))
    sampler.sample()

    result = sampler.result()
    assert len(calls) == 2
    assert result["expected_exit_races"] == 1
    assert result["fatal_read_errors"] == 0
    assert result["peak_tree_rss_bytes"] == 100
    detail = result["read_error_details"][0]
    assert detail["error_type"] == "AccessDenied"
    assert detail["resolution"] == resolution
    assert detail["resolution_source"] == "memory_task_fence"
    assert detail["memory_task_verification"] == verification


def test_memory_task_proof_success_stays_fatal():
    module = resource_module()
    sampler, calls = memory_task_sampler(module, lambda: SimpleNamespace(rss=999))
    sampler.sample()

    result = sampler.result()
    assert len(calls) == 2
    assert result["expected_exit_races"] == 0
    assert result["fatal_read_errors"] == 1
    assert result["peak_tree_rss_bytes"] == 100
    detail = result["read_error_details"][0]
    assert detail["error_type"] == "AccessDenied"
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "memory_task_fence"
    assert detail["memory_task_verification"] == "success"


def test_memory_task_proof_access_denied_stays_fatal():
    module = resource_module()
    sampler, calls = memory_task_sampler(module, raise_error(module.psutil.AccessDenied(20)))
    sampler.sample()

    result = sampler.result()
    assert len(calls) == 2
    assert result["expected_exit_races"] == 0
    assert result["fatal_read_errors"] == 1
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "memory_task_fence"
    assert detail["memory_task_verification"] == "access_denied"


def test_memory_task_proof_oserror_stays_fatal():
    module = resource_module()
    sampler, calls = memory_task_sampler(module, raise_error(OSError(0, "ambiguous")))
    sampler.sample()

    result = sampler.result()
    assert len(calls) == 2
    assert result["expected_exit_races"] == 0
    assert result["fatal_read_errors"] == 1
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "memory_task_fence"
    assert detail["memory_task_verification"] == "oserror"


def test_memory_task_proof_unknown_error_stays_fatal():
    module = resource_module()
    sampler, calls = memory_task_sampler(module, raise_error(RuntimeError("ambiguous")))
    sampler.sample()

    result = sampler.result()
    assert len(calls) == 2
    assert result["expected_exit_races"] == 0
    assert result["fatal_read_errors"] == 1
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "memory_task_fence"
    assert detail["memory_task_verification"] == "unknown"


@pytest.mark.parametrize("error_name", ["NoSuchProcess", "ZombieProcess"])
def test_memory_task_proof_mismatched_pid_stays_fatal(error_name):
    module = resource_module()
    sampler, calls = memory_task_sampler(module, raise_error(getattr(module.psutil, error_name)(21)))
    sampler.sample()

    result = sampler.result()
    assert len(calls) == 2
    assert result["expected_exit_races"] == 0
    assert result["fatal_read_errors"] == 1
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "memory_task_fence"
    assert detail["memory_task_verification"] == "pid_mismatch"


def test_memory_task_proof_pid_reuse_is_replaced():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    replacement = FakeProcess(20, 99.0)
    processes = {10: root, 20: child}
    sampler = fake_sampler(module, processes)
    sampler._identity_resolution = lambda identity: ("alive", None)  # type: ignore[method-assign]
    processes[20] = replacement

    sampler._note_read_error(
        module.psutil.AccessDenied(20),
        identity=(20, 2.0),
        operation="memory_info",
        phase="native_run",
        sample_index=0,
    )

    result = sampler.result()
    assert result["expected_exit_races"] == 1
    assert result["fatal_read_errors"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "replaced"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["memory_task_verification"] == "pid_reused"


def test_memory_task_proof_malformed_time_is_fatal():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    replacement = FakeProcess(20, 2.0)
    replacement.create_time = lambda: math.nan
    processes = {10: root, 20: child}
    sampler = fake_sampler(module, processes)
    sampler._identity_resolution = lambda identity: ("alive", None)  # type: ignore[method-assign]
    processes[20] = replacement

    sampler._note_read_error(
        module.psutil.AccessDenied(20),
        identity=(20, 2.0),
        operation="memory_info",
        phase="native_run",
        sample_index=0,
    )

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["memory_task_verification"] == "malformed_identity"


def test_memory_task_proof_skips_other_operations():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    processes = {10: root, 20: child}
    sampler = fake_sampler(module, processes)
    sampler._identity_resolution = lambda identity: ("alive", None)  # type: ignore[method-assign]
    child.memory_info = lambda: pytest.fail("memory_info verification must not run for cmdline")
    sampler._note_read_error(
        module.psutil.AccessDenied(20),
        identity=(20, 2.0),
        operation="cmdline",
        phase="native_run",
        sample_index=0,
    )

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["memory_task_verification"] == "not_run"


def test_memory_task_proof_initial_pid_mismatch_skips_verification():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    processes = {10: root, 20: child}
    sampler = fake_sampler(module, processes)
    sampler._identity_resolution = lambda identity: ("alive", None)  # type: ignore[method-assign]
    child.memory_info = lambda: pytest.fail("memory_info verification must not run for a mismatched PID")
    sampler._note_read_error(
        module.psutil.AccessDenied(21),
        identity=(20, 2.0),
        operation="memory_info",
        phase="native_run",
        sample_index=0,
    )

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["memory_task_verification"] == "not_run"


@pytest.mark.parametrize("error_name", ["NoSuchProcess", "ZombieProcess"])
@pytest.mark.parametrize("bad_pid", [21, None])
def test_authoritative_exit_requires_matching_pid(error_name, bad_pid):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    root.descendants = [child]
    sampler = fake_sampler(module, {10: root, 20: child})
    sampler.sample()

    error = getattr(module.psutil, error_name)(bad_pid)
    child.memory_info = raise_error(error)
    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["memory_task_verification"] == "not_run"


def test_authoritative_exit_does_not_retry_memory_info():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    root.descendants = [child]
    calls = []

    def memory_info():
        calls.append(None)
        raise module.psutil.NoSuchProcess(20)

    child.memory_info = memory_info
    sampler = fake_sampler(module, {10: root, 20: child})
    sampler.sample()

    result = sampler.result()
    assert len(calls) == 1
    assert result["expected_exit_races"] == 1
    assert result["fatal_read_errors"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution_source"] == "read_exception"
    assert detail["memory_task_verification"] == "not_run"


@pytest.mark.parametrize("kwargs", [{"resolution_source": "bad"}, {"memory_task_verification": "bad"}])
def test_note_read_error_rejects_unknown_provenance(kwargs):
    module = resource_module()
    ledger = module.ResourceLedger((10, 1.0), footprint_supported=False)

    with pytest.raises(ValueError):
        ledger.note_read_error(
            module.psutil.AccessDenied(10),
            identity=(10, 1.0),
            operation="cmdline",
            phase="native_run",
            sample_index=0,
            **kwargs,
        )


def test_short_worker_layout_stays_inside_macos_socket_budget_for_all_pilot_tasks():
    module = resource_module()
    assert (Path("/tmp") if module.os.name != "nt" else Path(module.tempfile.gettempdir())) == module.DEFAULT_WORKDIR
    root = Path("/private/tmp") / (module.SAMPLE_PREFIX + "abcdefgh")
    for task in ("wizard", "modal", "download", "tabs", "auth"):
        for arm in ("openbrowser-fast", "browser-use"):
            command = module.worker_command(root, task, arm, 0)
            workdir = Path(command[command.index("--workdir") + 1])
            assert workdir == root
            context_root = workdir / "20261125T055841Z" / f"{task}-{arm}-s0"
            socket = (
                context_root / "bu-runtime" / "bu-default.sock"
                if arm == "browser-use"
                else (context_root / "home" / ".openbrowser" / "daemon.sock")
            )
            assert len(str(socket).encode()) <= 103, (task, arm, socket)


def test_sampler_skips_cmdline_for_non_browser_processes():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"
    calls = []

    def denied():
        calls.append(True)
        raise module.psutil.AccessDenied(10)

    root.cmdline = denied
    sampler = fake_sampler(module, {10: root})

    sampler.sample()

    result = sampler.result()
    assert calls == []
    assert result["fatal_read_errors"] == 0
    assert result["sampler_method_version"] == module.SAMPLER_METHOD_VERSION == 6


def test_sampler_skips_cmdline_for_helper_and_crashpad():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"
    helper = FakeProcess(20, 2.0)
    helper.executable = "Google Chrome Helper (Aperitif)"
    crashpad = FakeProcess(30, 3.0)
    crashpad.executable = "chrome_crashpad_handler"
    helper.cmdline = lambda: pytest.fail("helper cmdline must not be read")
    crashpad.cmdline = lambda: pytest.fail("crashpad cmdline must not be read")
    root.descendants = [helper, crashpad]
    sampler = fake_sampler(module, {10: root, 20: helper, 30: crashpad})

    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 0
    assert result["observed_browser_processes"] == 0


def test_sampler_caches_helper_role_across_transient_name_flip():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    helper = FakeProcess(20, 2.0)
    helper.name_value = "Google Chrome Helper (Aperitif)"
    helper.cmdline = lambda: pytest.fail("cached helper cmdline must not be read")
    root.descendants = [helper]
    sampler = fake_sampler(module, {10: root, 20: helper})

    sampler.sample()
    helper.name_value = "Google Chrome"
    sampler.sample()

    result = sampler.result()
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["observed_browser_processes"] == 0


def test_sampler_caches_main_role_across_transient_cmdline_denial():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"
    browser.cmdline = lambda: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    root.descendants = [browser]
    sampler = fake_sampler(module, {10: root, 20: browser})

    sampler.sample()
    browser.cmdline = lambda: (_ for _ in ()).throw(module.psutil.AccessDenied(20))
    sampler.sample()

    result = sampler.result()
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["observed_browser_during_native_run"] == 1


def test_sampler_classifies_truncated_helper_with_exe_before_cmdline():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    helper = FakeProcess(20, 2.0)
    helper.name_value = "Google Chrome He"
    helper.executable = "/Applications/Google Chrome Helper.app/Contents/MacOS/Google Chrome Helper"
    helper.cmdline = lambda: pytest.fail("truncated helper cmdline must not be read")
    root.descendants = [helper]
    sampler = fake_sampler(module, {10: root, 20: helper})

    sampler.sample()

    result = sampler.result()
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["observed_browser_processes"] == 0


def test_sampler_classifies_truncated_main_with_exe_then_cmdline():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    browser = FakeProcess(20, 2.0)
    browser.name_value = "Google Chrome Be"
    browser.executable = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    browser.cmdline = lambda: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    root.descendants = [browser]
    sampler = fake_sampler(module, {10: root, 20: browser})

    sampler.sample()

    result = sampler.result()
    assert result["read_errors"] == 0
    assert result["fatal_read_errors"] == 0
    assert result["observed_browser_during_native_run"] == 1


def test_sampler_exe_access_denied_on_live_identity_is_fatal():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    helper = FakeProcess(20, 2.0)
    helper.name_value = "Google Chrome He"
    helper.executable = "/Applications/Google Chrome Helper.app/Contents/MacOS/Google Chrome Helper"
    helper.exe = lambda: (_ for _ in ()).throw(module.psutil.AccessDenied(20))
    helper.cmdline = lambda: pytest.fail("cmdline must not run after exe denial")
    root.descendants = [helper]
    sampler = fake_sampler(module, {10: root, 20: helper})

    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["operation"] == "exe"
    assert detail["error_type"] == "AccessDenied"
    assert detail["resolution"] == "alive"


def test_sampler_does_not_reuse_role_after_pid_reuse():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"
    browser.cmdline = lambda: ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    processes = {10: root, 20: browser}
    root.descendants = [browser]
    sampler = fake_sampler(module, processes)
    sampler.sample()

    replacement = FakeProcess(20, 99.0)
    replacement.executable = "Google Chrome"
    replacement.cmdline = lambda: (_ for _ in ()).throw(module.psutil.AccessDenied(20))
    processes[20] = replacement
    root.descendants = [replacement]
    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    detail = result["read_error_details"][0]
    assert detail["pid"] == 20
    assert detail["created_at"] == 99.0
    assert detail["resolution"] == "alive"


def test_sampler_does_not_cache_generic_non_browser_name():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"
    root.cmdline = lambda: (_ for _ in ()).throw(module.psutil.AccessDenied(10))
    sampler = fake_sampler(module, {10: root})
    sampler.sample()

    root.name_value = "Google Chrome"
    root.executable = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["read_error_details"][0]["resolution"] == "alive"


def test_sampler_replaced_pid_after_access_denied_is_expected():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"
    processes = {10: root, 20: browser}
    root.descendants = [browser]
    sampler = fake_sampler(module, processes)

    replacement = FakeProcess(20, 99.0)
    replacement.executable = "Google Chrome"

    def denied():
        processes[20] = replacement
        raise module.psutil.AccessDenied(20)

    browser.cmdline = denied
    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 0
    assert result["expected_exit_races"] == 1
    detail = result["read_error_details"][0]
    assert detail["error_type"] == "AccessDenied"
    assert detail["resolution"] == "replaced"
    assert detail["resolution_source"] == "identity_fence"


def test_sampler_unknown_fence_resolution_is_fatal():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"

    def denied_cmdline():
        browser.create_time = lambda: (_ for _ in ()).throw(module.psutil.AccessDenied(20))
        raise module.psutil.AccessDenied(20)

    browser.cmdline = denied_cmdline
    root.descendants = [browser]
    sampler = fake_sampler(module, {10: root, 20: browser})

    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    detail = result["read_error_details"][0]
    assert detail["operation"] == "cmdline"
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "identity_fence"


@pytest.mark.parametrize("pid", [True, False, 0, -1, 20.5, "20", None])
def test_identity_resolution_rejects_malformed_pid(pid: Any):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    sampler = fake_sampler(module, {10: root})

    resolution, error = sampler._identity_resolution((pid, 2.0))  # type: ignore[arg-type]

    assert resolution == "unknown"
    assert isinstance(error, ValueError)


@pytest.mark.parametrize(
    "created",
    [0.0, 0, -1.0, math.nan, math.inf, -math.inf, True, False, "2.0", None],
)
def test_identity_resolution_rejects_malformed_expected_create_time(created: Any):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    sampler = fake_sampler(module, {10: root, 20: child})

    resolution, error = sampler._identity_resolution((20, created))  # type: ignore[arg-type]

    assert resolution == "unknown"
    assert isinstance(error, ValueError)


@pytest.mark.parametrize(
    "created",
    [0.0, 0, -1.0, math.nan, math.inf, -math.inf, True, False, "2.0", None],
)
def test_identity_resolution_rejects_malformed_current_create_time(created: Any):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    child.create_time = lambda: created
    sampler = fake_sampler(module, {10: root, 20: child})

    resolution, error = sampler._identity_resolution((20, 2.0))

    assert resolution == "unknown"
    assert isinstance(error, ValueError)


@pytest.mark.parametrize(
    "created",
    [0.0, 0, -1.0, math.nan, math.inf, -math.inf, True, False, "2.0", None],
)
def test_sampler_malformed_current_create_time_is_fatal(created: Any):
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    root.descendants = [child]

    def denied():
        child.create_time = lambda: created
        raise module.psutil.AccessDenied(20)

    child.memory_info = denied
    sampler = fake_sampler(module, {10: root, 20: child})

    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["resolution_error_type"] == "ValueError"


def test_sampler_child_identity_failure_keeps_sentinel_unknown():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    child = FakeProcess(20, 2.0)
    child.create_time = lambda: (_ for _ in ()).throw(module.psutil.AccessDenied(20))
    root.descendants = [child]
    sampler = fake_sampler(module, {10: root, 20: child})

    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    detail = result["read_error_details"][0]
    assert detail["operation"] == "child_identity"
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "identity_fence"
    assert detail["resolution_error_type"] == "ValueError"


def test_browser_cmdline_failure_is_fatal_and_unverified():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"

    def denied():
        raise module.psutil.AccessDenied(20)

    browser.cmdline = denied
    root.descendants = [browser]
    sampler = fake_sampler(module, {10: root, 20: browser})

    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["expected_exit_races"] == 0
    assert result["unverified_processes"] == 1
    detail = result["read_error_details"][0]
    assert detail["pid"] == 20
    assert detail["created_at"] == 2.0
    assert detail["operation"] == "cmdline"
    assert detail["phase"] == "native_run"
    assert detail["error_type"] == "AccessDenied"
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "identity_fence"
    assert "error" not in detail
    assert detail["errno"] is None
    assert result["cleanup_unverified_processes"] == 0


def test_required_read_failure_records_operation_and_unverified():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"

    def denied():
        raise module.psutil.AccessDenied(10)

    root.memory_info = denied
    sampler = fake_sampler(module, {10: root})

    sampler.sample()

    result = sampler.result()
    assert result["peak_tree_rss_bytes"] is None
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    detail = result["read_error_details"][0]
    assert detail["operation"] == "memory_info"
    assert detail["pid"] == 10
    assert detail["created_at"] == 1.0
    assert detail["resolution"] == "alive"
    assert detail["resolution_source"] == "memory_task_fence"
    assert detail["memory_task_verification"] == "access_denied"


def test_live_process_lookup_failure_records_operation():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"

    def denied():
        raise module.psutil.AccessDenied(10)

    root.is_running = denied
    sampler = fake_sampler(module, {10: root})

    sampler.sample()

    result = sampler.result()
    assert result["fatal_read_errors"] == 1
    assert result["unverified_processes"] == 1
    detail = result["read_error_details"][0]
    assert detail["operation"] == "is_running"
    assert detail["pid"] == 10
    assert detail["created_at"] == 1.0
    assert detail["resolution"] == "unknown"
    assert detail["resolution_source"] == "identity_fence"


def test_read_error_details_are_bounded():
    module = resource_module()
    ledger = module.ResourceLedger((10, 1.0), footprint_supported=False)

    for index in range(module.MAX_READ_ERROR_DETAILS + 5):
        ledger.note_read_error(
            module.psutil.AccessDenied(10),
            identity=(10, 1.0),
            operation="cmdline",
            phase="native_run",
            sample_index=index,
        )

    result = ledger.result()
    assert result["read_error_details"][0]["resolution_source"] == "identity_fence"
    assert len(result["read_error_details"]) == module.MAX_READ_ERROR_DETAILS
    assert result["read_error_details_truncated"] == 5
    assert result["fatal_read_errors"] == module.MAX_READ_ERROR_DETAILS + 5


def test_fatal_access_denied_is_not_reclassified_after_exit():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"

    def denied():
        raise module.psutil.AccessDenied(20)

    browser.cmdline = denied
    root.descendants = [browser]
    processes = {10: root, 20: browser}
    sampler = fake_sampler(module, processes)

    sampler.sample()
    first = sampler.result()
    processes.pop(10)
    processes.pop(20)
    second = sampler.result()

    assert first["fatal_read_errors"] == second["fatal_read_errors"] == 1
    assert second["cleanup_unverified_processes"] == 0
    assert second["expected_exit_races"] == 0
    assert second["unverified_processes"] == 1
    assert second["read_error_details"][0]["operation"] == "cmdline"
    assert second["read_error_details"][0]["resolution"] == "alive"
    assert second["read_error_details"][0]["resolution_source"] == "identity_fence"


def test_lane_writes_strict_terminal_metadata(monkeypatch, tmp_path):
    module = resource_module()
    output = tmp_path / "results.jsonl"

    monkeypatch.setattr(module, "source_manifest", lambda: {"source_digest": "stable"})
    monkeypatch.setattr(
        module,
        "binding_snapshot",
        lambda arms: {
            "verified": True,
            "entry_sha256": "entry",
            "actual_app_digest": "app",
            "expected_app_digest": "app",
            "dist_version": "v",
        },
    )

    def run_sample(task, arm, sample, **kwargs):
        return {
            "task": task,
            "arm": arm,
            "sample": sample,
            "ok": True,
            "resource_ok": True,
            "resources": {"peak_tree_rss_bytes": 100},
            "cleanup": {
                "verified": True,
                "remaining": [],
                "unverified_processes": 0,
                "errors": [],
            },
            "worker_timed_out": False,
        }

    monkeypatch.setattr(module, "run_sample", run_sample)

    module.run_lane(["a", "b"], ["tabs"], 2, output, workdir=tmp_path / "runs", interval=0.02, worker_timeout=10)
    raw = [json.loads(line) for line in output.read_text().splitlines()]

    assert raw[0]["expected_observations"] == 4
    assert raw[-1]["expected_observations"] == 4
    assert raw[-1]["observations"] == 4
    assert raw[-1]["valid"] is True
    assert raw[-1]["terminal"] is True
    assert raw[-1]["cleanup_verified"] is True
    assert raw[-1]["fixture_errors"] == []
    assert raw[-1]["failures"] == 0


def test_oserror_errno_is_closed_and_recorded():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"

    def denied():
        raise OSError(0, "Undefined error: 0 (originated from sysctl(KERN_PROCARGS2))")

    root.memory_info = denied
    sampler = fake_sampler(module, {10: root})

    sampler.sample()

    detail = sampler.result()["read_error_details"][0]
    assert detail["error_type"] == "OSError"
    assert detail["errno"] == 0
    assert "error" not in detail


def test_cleanup_verifies_after_fatal_read_identity_is_gone():
    module = resource_module()
    root = FakeProcess(10, 1.0)
    root.executable = "python3.12"
    browser = FakeProcess(20, 2.0)
    browser.executable = "Google Chrome"

    def denied():
        raise module.psutil.AccessDenied(20)

    browser.cmdline = denied
    root.descendants = [browser]
    processes = {10: root, 20: browser}
    sampler = fake_sampler(module, processes)

    sampler.sample()
    processes.pop(10)
    processes.pop(20)
    cleanup = module.cleanup_owned(sampler, interval=0.1, timeout=1.0)
    resources = sampler.result()

    assert resources["fatal_read_errors"] == 1
    assert resources["unverified_processes"] == 1
    assert resources["cleanup_unverified_processes"] == 0
    assert cleanup["verified"] is True
    assert cleanup["unverified_processes"] == 0
    assert module.observation_ok(resources, cleanup) is False


def test_observation_ok_rejects_fatal_read_even_with_verified_cleanup():
    module = resource_module()
    resources = {
        "peak_tree_rss_bytes": 100,
        "fatal_read_errors": 1,
        "observed_child_processes": 1,
        "observed_browser_during_native_run": 1,
    }

    assert module.observation_ok(resources, {"verified": True}) is False
    assert module.observation_ok({**resources, "fatal_read_errors": 0}, {"verified": True}) is True


@pytest.mark.parametrize("changed_key", ["backend_entry_sha", "native_manifest_sha", "native_source_digest", "target"])
def test_lane_rejects_any_native_binding_change(monkeypatch, tmp_path, changed_key):
    module = resource_module()
    before = {
        "verified": True,
        "entry_sha256": "entry",
        "actual_app_digest": "app",
        "expected_app_digest": "app",
        "dist_version": "v",
        changed_key: "before",
    }
    after = {**before, changed_key: "after"}
    bindings = iter([before, after])
    monkeypatch.setattr(module, "source_manifest", lambda: {"source_digest": "stable"})
    monkeypatch.setattr(module, "binding_snapshot", lambda arms: next(bindings))
    monkeypatch.setattr(module, "arm_versions", lambda arms: {})
    monkeypatch.setattr(
        module,
        "run_sample",
        lambda task, arm, sample, **kw: {
            "task": task,
            "arm": arm,
            "sample": sample,
            "ok": True,
            "resource_ok": True,
            "cleanup": {"verified": True, "remaining": [], "unverified_processes": 0, "errors": []},
            "resources": {"peak_tree_rss_bytes": 100},
        },
    )
    output = tmp_path / "native-drift.jsonl"
    rows = module.run_lane(["a"], ["tabs"], 1, output, workdir=tmp_path / "runs", interval=0.02, worker_timeout=1)
    end = json.loads(output.read_text().splitlines()[-1])
    assert end["binding_unchanged"] is False
    assert end["provenance_verified"] is False
    assert rows[0]["provenance_verified"] is False
