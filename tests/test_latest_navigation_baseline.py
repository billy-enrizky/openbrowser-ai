import asyncio
import importlib.util
import json
import shutil
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "latest_navigation_baseline.py"
)
sys.path.insert(0, str(SCRIPT.parent))


@pytest.fixture
def private_result_root():
    root = (
        Path("/tmp").resolve(strict=True)
        / f"openbrowser-latest-navigation-test-{uuid.uuid4().hex}"
    )
    root.mkdir(mode=0o700)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def runner():
    assert SCRIPT.is_file(), "latest navigation runner is missing"
    spec = importlib.util.spec_from_file_location(
        "latest_navigation_baseline_tests", SCRIPT
    )
    assert spec is not None and spec.loader is not None, (
        "latest navigation runner cannot be loaded"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_reverses_pair_order(runner):
    pair_order = getattr(runner, "pair_order", None)
    assert callable(pair_order), "runner has no pair schedule"
    assert pair_order(1) == ("raw-cdp", "lightpanda", "agent-browser", "browser-use")
    assert pair_order(2) == ("browser-use", "agent-browser", "lightpanda", "raw-cdp")
    with pytest.raises(ValueError, match="round"):
        runner.pair_order(3)


def test_experiment_reverses_source_order(runner):
    source_order = getattr(runner, "experiment_source_order", None)
    assert callable(source_order), "runner has no experiment source schedule"
    assert source_order(1, 1) == ("baseline", "treatment")
    assert source_order(1, 2) == ("treatment", "baseline")
    assert source_order(2, 1) == ("treatment", "baseline")
    assert source_order(2, 2) == ("baseline", "treatment")


def test_sample_order_alternates_both_arms(runner):
    sample_arm_order = getattr(runner, "sample_arm_order", None)
    assert callable(sample_arm_order), "runner has no alternating sample schedule"
    assert sample_arm_order(0, "raw-cdp") == ("openbrowser", "raw-cdp")
    assert sample_arm_order(1, "raw-cdp") == ("raw-cdp", "openbrowser")
    assert sample_arm_order(2, "raw-cdp") == ("openbrowser", "raw-cdp")


def test_pair_rejects_missing_navigation_ns(runner):
    sample = {
        "outcome": "success",
        "start_ns": 10,
        "navigation_return_ns": 20,
        "end_ns": 30,
        "elapsed_ns": 20,
        "navigation_ns": 10,
        "post_navigation_ns": 10,
    }
    phase_errors = getattr(runner, "navigation_phase_errors", None)
    assert callable(phase_errors), "runner has no navigation phase verifier"
    assert phase_errors(sample, "sample") == []

    del sample["navigation_ns"]
    assert any("navigation_ns" in error for error in phase_errors(sample, "sample"))


@pytest.mark.asyncio
async def test_runner_retries_only_one_contaminated_attempt(runner):
    calls = []
    run_pair_attempts = getattr(runner, "run_pair_attempts", None)
    assert callable(run_pair_attempts), "runner has no pair attempt scheduler"

    async def contaminated_attempt(attempt_number):
        calls.append(attempt_number)
        return {
            "state": "contaminated",
            "cleanup_ok": True,
            "errors": ["host load gate"],
        }

    async def quiet_gate():
        return True

    pair = await run_pair_attempts("raw-cdp", contaminated_attempt, quiet_gate)

    assert calls == [1, 2]
    assert pair["state"] == "contaminated"
    assert [attempt["state"] for attempt in pair["attempts"]] == [
        "contaminated",
        "contaminated",
    ]


@pytest.mark.asyncio
async def test_runner_does_not_retry_failed_pair_attempt(runner):
    calls = []
    run_pair_attempts = getattr(runner, "run_pair_attempts", None)
    assert callable(run_pair_attempts), "runner has no pair attempt scheduler"

    async def failed_attempt(attempt_number):
        calls.append(attempt_number)
        return {"state": "failed", "cleanup_ok": True, "errors": ["task proof"]}

    async def quiet_gate():
        return True

    pair = await run_pair_attempts("raw-cdp", failed_attempt, quiet_gate)

    assert calls == [1]
    assert pair["state"] == "failed"


@pytest.mark.asyncio
async def test_runner_stops_after_uncertain_worker_cleanup(runner):
    calls = []
    quiet_checks = []

    async def contaminated_attempt(attempt_number):
        calls.append(attempt_number)
        return {
            "state": "contaminated",
            "cleanup_ok": False,
            "errors": ["worker is alive"],
        }

    async def quiet_gate():
        quiet_checks.append(True)
        return True

    pair = await runner.run_pair_attempts("raw-cdp", contaminated_attempt, quiet_gate)

    assert calls == [1]
    assert quiet_checks == []
    assert pair["state"] == "failed"


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_hashed_attempt_evidence(runner):
    evidence = {
        "state": "failed",
        "cleanup_ok": False,
        "errors": ["worker cleanup incomplete", "runtime root retained"],
        "work_counts": {
            "attempted_work_count": 24,
            "successful_work_count": 24,
            "error_count": 2,
        },
        "error_count": 2,
    }
    before = json.dumps(evidence, sort_keys=True)

    async def execute(_number):
        return evidence

    async def quiet():
        pytest.fail("uncertain cleanup must stop without retry")

    pair = await runner.run_pair_attempts("lightpanda", execute, quiet)

    assert json.dumps(evidence, sort_keys=True) == before
    assert pair["state"] == "failed"
    assert pair["attempts"][0]["error_count"] == 2


def test_incomplete_pair_cannot_rank(runner):
    run_can_rank = getattr(runner, "run_can_rank", None)
    competitors = getattr(runner, "COMPETITORS", None)
    assert callable(run_can_rank) and isinstance(competitors, tuple), (
        "runner has no ranking gate"
    )
    complete = {
        "state": "complete",
        "cleanup": {"ok": True},
        "host_preflight": {"state": "clear"},
        "errors": [],
        "pairs": [
            {"round": round_number, "competitor": competitor, "state": "verified"}
            for round_number in (1, 2)
            for competitor in competitors
        ],
    }
    incomplete = {**complete, "pairs": list(complete["pairs"])}
    incomplete["pairs"][-1] = {**incomplete["pairs"][-1], "state": "contaminated"}

    assert run_can_rank(complete) is True
    assert run_can_rank(incomplete) is False


def test_host_gate_treats_load_limit_and_unknown_processes_as_contaminated(runner):
    classify_host_sample = getattr(runner, "classify_host_sample", None)
    assert callable(classify_host_sample), "runner has no host contamination classifier"
    overloaded = classify_host_sample(
        {
            "load_1m": 8.0,
            "logical_cpu_count": 8,
            "processes": [],
            "classification_errors": [],
        }
    )
    unknown = classify_host_sample(
        {
            "load_1m": 1.0,
            "logical_cpu_count": 8,
            "processes": [],
            "classification_errors": ["process CPU was unavailable"],
        }
    )

    assert overloaded["state"] == "contaminated"
    assert unknown["state"] == "unknown"


def test_host_gate_rejects_boolean_load_and_detects_unrelated_cpu(runner):
    invalid = runner.classify_host_sample(
        {
            "load_1m": True,
            "logical_cpu_count": 8,
            "processes": [],
            "classification_errors": [],
        }
    )
    busy_process = runner.classify_host_sample(
        {
            "load_1m": 1.0,
            "logical_cpu_count": 8,
            "processes": [{"pid": 42, "cpu_percent": 100.0, "excluded": False}],
            "classification_errors": [],
        }
    )
    owned_process = runner.classify_host_sample(
        {
            "load_1m": 1.0,
            "logical_cpu_count": 8,
            "processes": [{"pid": 43, "cpu_percent": 175.0, "excluded": True}],
            "classification_errors": [],
        }
    )

    assert invalid["state"] == "unknown"
    assert busy_process["state"] == "contaminated"
    assert owned_process["state"] == "clear"


def test_host_sample_computes_cpu_delta_and_excludes_benchmark_processes(
    runner, monkeypatch
):
    before = {
        "captured_monotonic": 10.0,
        "processes": {
            (42, 1.0): {
                "pid": 42,
                "name": "unrelated",
                "cpu_seconds": 2.0,
                "excluded": False,
            },
            (43, 2.0): {
                "pid": 43,
                "name": "worker",
                "cpu_seconds": 3.0,
                "excluded": True,
            },
        },
        "classification_errors": [],
    }
    after = {
        "captured_monotonic": 11.0,
        "load_1m": 1.0,
        "processes": {
            (42, 1.0): {
                "pid": 42,
                "name": "unrelated",
                "cpu_seconds": 2.5,
                "excluded": False,
            },
            (43, 2.0): {
                "pid": 43,
                "name": "worker",
                "cpu_seconds": 4.0,
                "excluded": True,
            },
        },
        "classification_errors": [],
    }
    monkeypatch.setattr(runner, "_process_cpu_snapshot", lambda: after)

    sample, current = runner._host_sample_since(before, logical_cpus=8)

    assert current is after
    assert sample["processes"] == [
        {"pid": 42, "name": "unrelated", "cpu_percent": 50.0, "excluded": False},
        {"pid": 43, "name": "worker", "cpu_percent": 0.0, "excluded": True},
    ]


def test_macos_cpu_table_parses_cumulative_time_without_task_port_access(
    runner, monkeypatch
):
    import subprocess

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, " 1 0:58.26\n 42 2:03:04.50\n", ""
        ),
    )
    assert runner.process_cpu_seconds() == {1: 58.26, 42: 7384.5}


def test_host_sample_classifies_a_process_born_after_previous_capture(
    runner, monkeypatch
):
    previous = {
        "captured_monotonic": 10.0,
        "captured_wall_time": 100.0,
        "processes": {},
        "classification_errors": [],
    }
    current = {
        "captured_monotonic": 20.0,
        "captured_wall_time": 110.0,
        "load_1m": 1.0,
        "processes": {
            (42, 105.0): {
                "pid": 42,
                "name": "new process",
                "cpu_seconds": 2.0,
                "excluded": False,
            }
        },
        "classification_errors": [],
    }
    monkeypatch.setattr(runner, "_process_cpu_snapshot", lambda: current)

    sample, _current = runner._host_sample_since(previous, 8)

    assert sample["classification_errors"] == []
    assert sample["processes"][0]["cpu_percent"] == 20.0


@pytest.mark.parametrize(
    "output", ["", "42 invalid\n", "42 -1:00\n", "42 1:00\n42 2:00\n"]
)
def test_macos_cpu_table_rejects_unverifiable_counters(runner, monkeypatch, output):
    import subprocess

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""),
    )
    with pytest.raises(ValueError):
        runner.process_cpu_seconds()


def test_navigation_run_pair_and_attempt_state_machines_reject_shortcuts(runner):
    assert runner.transition_run("preparing", "preflight_passed") == "preflighted"
    with pytest.raises(ValueError, match="illegal navigation run transition"):
        runner.transition_run("preparing", "start")

    assert runner.transition_pair("pending", "start") == "running"
    assert runner.transition_attempt("running", "contaminated") == "contaminated"
    with pytest.raises(ValueError, match="illegal navigation pair transition"):
        runner.transition_pair("verified", "failed")


def test_smoke_cli_has_fixed_control_arms(runner):
    parser = getattr(runner, "parser", None)
    arms_for_mode = getattr(runner, "arms_for_mode", None)
    assert callable(parser) and callable(arms_for_mode), (
        "runner has no fixed smoke contract"
    )
    args = parser().parse_args(
        [
            "smoke",
            "--candidate-identity",
            "/tmp/identity.json",
            "--preflight",
            "/tmp/preflight.json",
            "--version-manifest",
            "/tmp/version.json",
            "--source",
            "/tmp/source",
            "--python",
            "/tmp/python",
            "--chrome",
            "/tmp/chrome",
            "--output",
            "/tmp/result/smoke/navigation",
        ]
    )
    assert args.command == "smoke"
    assert arms_for_mode("smoke") == ("openbrowser", "raw-cdp")
    assert not hasattr(args, "arms")


def test_smoke_exit_requires_cleanup(runner):
    exit_code = getattr(runner, "navigation_run_exit_code", None)
    assert callable(exit_code), "runner has no explicit exit gate"
    manifest = {
        "schema": runner.SMOKE_SCHEMA,
        "mode": "smoke",
        "state": "incomplete",
        "cleanup": {"ok": False},
        "fixture": {"stopped": False},
        "pairs": [],
        "errors": ["worker cleanup uncertain"],
    }
    assert exit_code(manifest) == 1


@pytest.mark.asyncio
async def test_pair_attempt_runs_both_arms_and_proves_each_workload(
    runner,
    private_result_root,
    monkeypatch,
):
    class Ledger:
        server_address = ("127.0.0.1", 12345)

        def snapshot(self, nonce):
            requests = [
                {
                    "path": "/fixture",
                    "method": "GET",
                    "nonce": nonce,
                    "status": 200,
                    "received_ns": 110,
                    "completed_ns": 120,
                },
                {
                    "path": "/effect",
                    "method": "POST",
                    "nonce": nonce,
                    "status": 200,
                    "received_ns": 160,
                    "completed_ns": 170,
                },
            ]
            if "-delayed-" in nonce:
                requests.insert(
                    1,
                    {
                        "path": "/slow.js",
                        "method": "GET",
                        "nonce": nonce,
                        "status": 200,
                        "received_ns": 130,
                        "completed_ns": 140,
                    },
                )
            return {"requests": requests, "effects": 1}

    class Handle:
        def __init__(self, launch):
            self.launch = launch
            self.state = runner.navigation.WorkerState.CREATED
            self.worker_identity = {
                "pid": 1000 + len(handles),
                "created": float(1000 + len(handles)),
                "cmdline": [str(launch.interpreter), "worker"],
                "owner_token": launch.owner_token,
                "owner_root": str(launch.worker_root),
            }

    def successful_sample(request):
        nonce = request["nonce"]
        url = request["url"]
        observation = {
            "url": url,
            "title": f"fixture-{nonce}",
            "readyState": "complete",
            "marker": nonce,
            "loaded": nonce,
            "scriptDone": nonce,
        }
        return {
            **request,
            "outcome": "success",
            "start_ns": 100,
            "navigation_return_ns": 150,
            "end_ns": 200,
            "elapsed_ns": 100,
            "navigation_ns": 50,
            "post_navigation_ns": 50,
            "checked_ns": 190,
            "navigation_observation": dict(observation),
            "effect_observation": {
                "field": nonce,
                "response": {"count": 1, "nonce": nonce, "payload": nonce},
            },
            "final_observation": {**observation, "field": nonce},
        }

    handles = {}
    order = []

    async def launch_worker(launch):
        handle = Handle(launch)
        handles[launch.arm] = handle
        return handle

    async def start_worker(handle):
        handle.state = runner.navigation.WorkerState.READY
        return {"ok": True, "result": {"owned": []}}, []

    async def send_operation(handle, request, *, timeout):
        order.append((request["kind"], request["index"], request["arm"]))
        if len(order) > 2:
            manifest_path = (
                run_root / "round-1/pair-raw-cdp/attempt-1/attempt-manifest.json"
            )
            persisted = json.loads(manifest_path.read_text())
            measured_count = len(order) - 2
            assert len(persisted["execution_order"]) == measured_count
            assert persisted["execution_order"][-1] == {
                "workload": request["kind"],
                "index": request["index"],
                "arm": request["arm"],
            }
            assert len(persisted["samples"]) == measured_count - 1
        return {"ok": True, "result": successful_sample(request)}

    async def cleanup_worker(handle):
        return {
            "ok": True,
            "worker_identity": {**handle.worker_identity, "state": "absent"},
            "post_worker_exit_identities": [],
        }

    async def monitor(stop, samples):
        samples.append(
            {
                "load_1m": 1.0,
                "logical_cpu_count": 8,
                "processes": [],
                "classification_errors": [],
            }
        )
        await stop.wait()

    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "send_operation", send_operation)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    monkeypatch.setattr(runner, "monitor_pair_host", monitor)
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    args = SimpleNamespace(
        source=private_result_root,
        python=sys.executable,
        browser_use_python=sys.executable,
        chrome="/tmp/chrome",
        lightpanda="/tmp/lightpanda",
        agent_browser="/tmp/agent-browser",
        node=None,
    )
    published_attempts = []

    async def save_attempt_reference(reference):
        manifest_path = run_root / reference["directory"] / "attempt-manifest.json"
        saved = json.loads(manifest_path.read_text())
        assert (
            runner.preflight.sha256_file(manifest_path) == reference["manifest_sha256"]
        )
        assert saved["state"] == reference["state"]
        assert saved["work_counts"] == reference["work_counts"]
        published_attempts.append(reference)

    attempt = await runner.run_pair_attempt(
        args,
        result_root=private_result_root,
        run_root=run_root,
        fixture=Ledger(),
        run_id="c" * 32,
        round_number=1,
        competitor="raw-cdp",
        attempt_number=1,
        samples_per_workload=1,
        on_update=save_attempt_reference,
    )

    saved = json.loads(
        (run_root / attempt["directory"] / "attempt-manifest.json").read_text()
    )
    assert attempt["state"] == "verified", attempt
    assert attempt["cleanup_ok"] is True
    assert attempt["runtime_root_removed"] is True
    assert Path(saved["cleanup"]["runtime_root"]).parent == Path("/tmp").resolve()
    assert len(published_attempts) >= 10
    assert published_attempts[0]["state"] == "pending"
    assert published_attempts[1]["state"] == "running"
    assert published_attempts[-1]["state"] == "verified"
    assert saved["work_counts"] == {
        "attempted_work_count": 4,
        "successful_work_count": 4,
        "error_count": 0,
    }
    assert order == [
        ("fast", 0, "openbrowser"),
        ("fast", 0, "raw-cdp"),
        ("fast", 0, "openbrowser"),
        ("fast", 0, "raw-cdp"),
        ("delayed", 0, "openbrowser"),
        ("delayed", 0, "raw-cdp"),
    ]


@pytest.mark.asyncio
async def test_attempt_error_count_does_not_count_warmup_errors_twice(
    runner,
    private_result_root,
    monkeypatch,
):
    class Handle:
        def __init__(self, launch):
            self.launch = launch
            self.state = runner.navigation.WorkerState.CREATED
            self.worker_identity = {
                "pid": 2000 + len(handles),
                "created": float(2000 + len(handles)),
                "cmdline": [str(launch.interpreter), "worker"],
                "owner_token": launch.owner_token,
                "owner_root": str(launch.worker_root),
            }

    handles = {}

    async def launch_worker(launch):
        handle = Handle(launch)
        handles[launch.arm] = handle
        return handle

    async def start_worker(handle):
        handle.state = runner.navigation.WorkerState.READY
        return {"ok": True, "result": {"owned": []}}, []

    async def failed_trial(_handle, fixture, request, *, is_warmup):
        return {
            "arm": request["arm"],
            "workload": request["kind"],
            "index": request["index"],
            "nonce": request["nonce"],
            "url": request["url"],
            "warmup": is_warmup,
            "state": "failed",
            "sample": None,
            "request_proof": {"requests": [], "effects": 0},
            "errors": ["warmup task failed"],
        }

    async def cleanup_worker(handle):
        return {
            "ok": True,
            "worker_identity": {**handle.worker_identity, "state": "absent"},
            "post_worker_exit_identities": [],
        }

    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    monkeypatch.setattr(runner, "execute_trial", failed_trial)
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    attempt = await runner.run_pair_attempt(
        SimpleNamespace(
            source=private_result_root,
            python=sys.executable,
            browser_use_python=sys.executable,
            chrome="/tmp/chrome",
            lightpanda="/tmp/lightpanda",
            agent_browser="/tmp/agent-browser",
            node=None,
        ),
        result_root=private_result_root,
        run_root=run_root,
        fixture=SimpleNamespace(server_address=("127.0.0.1", 12345)),
        run_id="d" * 32,
        round_number=1,
        competitor="raw-cdp",
        attempt_number=1,
        samples_per_workload=1,
    )

    saved = json.loads(
        (run_root / attempt["directory"] / "attempt-manifest.json").read_text()
    )
    assert attempt["state"] == "failed"
    assert attempt["error_count"] == 2
    assert saved["work_counts"] == {
        "attempted_work_count": 0,
        "successful_work_count": 0,
        "error_count": 2,
    }


def test_output_path_accepts_only_the_documented_run_locations(
    runner, private_result_root
):
    result_root_for_output = getattr(runner, "result_root_for_output", None)
    assert callable(result_root_for_output), "runner has no result-root path validator"

    root, output = result_root_for_output(private_result_root / "navigation", "run")
    smoke_root, smoke_output = result_root_for_output(
        private_result_root / "smoke" / "navigation",
        "smoke",
    )

    assert root == private_result_root
    assert output == private_result_root / "navigation"
    assert smoke_root == private_result_root
    assert smoke_output == private_result_root / "smoke" / "navigation"
    with pytest.raises(ValueError, match="output"):
        result_root_for_output(private_result_root / "wrong-name", "run")
    with pytest.raises(ValueError, match="result-root"):
        result_root_for_output(
            private_result_root.parent / "outside" / "navigation", "run"
        )


@pytest.mark.asyncio
async def test_run_navigation_stops_before_fixture_when_host_preflight_fails(
    runner,
    private_result_root,
    monkeypatch,
):
    output = private_result_root / "navigation"
    monkeypatch.setattr(
        runner,
        "validate_navigation_inputs",
        lambda *_args: {
            "identity": {"candidate_commit": "a" * 40, "candidate_tree": "b" * 40},
            "identity_sha256": "c" * 64,
            "version_sha256": "d" * 64,
            "preflight_sha256": "e" * 64,
            "frozen_harness_sha256": "f" * 64,
            "source_root": private_result_root,
            "source_snapshot": {},
            "harness_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        runner,
        "collect_host_preflight",
        lambda **_kwargs: asyncio.sleep(
            0,
            result={
                "state": "contaminated",
                "reasons": ["one-minute load reached logical CPU count"],
                "logical_cpu_count": 8,
                "uptime": "test",
                "samples": [],
            },
        ),
    )

    exit_code = await runner.run_navigation(
        SimpleNamespace(
            command="run",
            output=output,
            rounds=2,
            samples=6,
            source=private_result_root,
        )
    )

    manifest = json.loads((output / "run-manifest.json").read_text())
    assert exit_code == 1
    assert manifest["state"] == "incomplete"
    assert manifest["failed_phase"] == "host_preflight"
    assert manifest["pairs"] == []
    assert manifest["fixture"]["started"] is False


@pytest.mark.asyncio
async def test_run_navigation_records_counterbalanced_pair_blocks(
    runner,
    private_result_root,
    monkeypatch,
):
    class FakeFixture:
        def __init__(self, _delay):
            self.server_address = ("127.0.0.1", 12345)
            self.stopped = threading.Event()

        def serve_forever(self):
            self.stopped.wait()

        def shutdown(self):
            self.stopped.set()

        def server_close(self):
            return None

    output = private_result_root / "navigation"
    identity = {
        "candidate_commit": "a" * 40,
        "candidate_tree": "b" * 40,
        "executed_files": {},
    }
    inputs = {
        "identity": identity,
        "identity_sha256": "c" * 64,
        "version_sha256": "d" * 64,
        "preflight_sha256": "e" * 64,
        "frozen_harness_sha256": "f" * 64,
        "source_root": private_result_root,
        "source_snapshot": {},
        "harness_sha256": "1" * 64,
    }
    monkeypatch.setattr(runner, "validate_navigation_inputs", lambda *_args: inputs)
    monkeypatch.setattr(
        runner,
        "collect_host_preflight",
        lambda **_kwargs: asyncio.sleep(
            0,
            result={
                "state": "clear",
                "reasons": [],
                "logical_cpu_count": 8,
                "uptime": "test",
                "samples": [],
            },
        ),
    )
    monkeypatch.setattr(runner.navigation, "Fixture", FakeFixture)
    monkeypatch.setattr(runner.preflight, "_git_snapshot", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner.preflight, "_hash_executed_files", lambda *_args: {})

    async def fake_pair_attempts(
        competitor, _execute_attempt, _wait_for_quiet, *, pair_record, on_update
    ):
        pair_record["state"] = "running"
        await on_update(pair_record)
        pair_record["attempts"].append(
            {
                "attempt_number": 1,
                "state": "verified",
                "directory": f"round-{pair_record['round']}/pair-{competitor}/attempt-1",
                "manifest_sha256": "2" * 64,
                "cleanup_ok": True,
                "runtime_root_removed": True,
                "host_gate": {"state": "clear", "reasons": []},
                "work_counts": {
                    "attempted_work_count": 24,
                    "successful_work_count": 24,
                    "error_count": 0,
                },
                "error_count": 0,
                "errors": [],
            }
        )
        pair_record["state"] = "verified"
        await on_update(pair_record)
        return pair_record

    monkeypatch.setattr(runner, "run_pair_attempts", fake_pair_attempts)
    exit_code = await runner.run_navigation(
        SimpleNamespace(
            command="run",
            output=output,
            rounds=2,
            samples=6,
            source=private_result_root,
        )
    )
    manifest = json.loads((output / "run-manifest.json").read_text())

    assert exit_code == 0
    assert [pair["competitor"] for pair in manifest["pairs"][:4]] == list(
        runner.COMPETITORS
    )
    assert [pair["competitor"] for pair in manifest["pairs"][4:]] == list(
        reversed(runner.COMPETITORS)
    )
    assert all(pair["state"] == "verified" for pair in manifest["pairs"])


@pytest.mark.asyncio
async def test_run_manifest_links_pending_attempt_before_attempt_work_starts(
    runner,
    private_result_root,
    monkeypatch,
):
    class FakeFixture:
        def __init__(self, _delay):
            self.server_address = ("127.0.0.1", 12345)
            self.stopped = threading.Event()

        def serve_forever(self):
            self.stopped.wait()

        def shutdown(self):
            self.stopped.set()

        def server_close(self):
            return None

    output = private_result_root / "smoke" / "navigation"
    identity = {
        "candidate_commit": "a" * 40,
        "candidate_tree": "b" * 40,
        "executed_files": {},
    }
    inputs = {
        "identity": identity,
        "identity_sha256": "c" * 64,
        "version_sha256": "d" * 64,
        "preflight_sha256": "e" * 64,
        "frozen_harness_sha256": None,
        "source_root": private_result_root,
        "source_snapshot": {},
        "harness_sha256": "1" * 64,
    }
    monkeypatch.setattr(runner, "validate_navigation_inputs", lambda *_args: inputs)
    monkeypatch.setattr(
        runner,
        "collect_host_preflight",
        lambda **_kwargs: asyncio.sleep(
            0,
            result={
                "state": "clear",
                "reasons": [],
                "logical_cpu_count": 8,
                "uptime": "test",
                "samples": [],
            },
        ),
    )
    monkeypatch.setattr(runner.navigation, "Fixture", FakeFixture)
    monkeypatch.setattr(runner.preflight, "_git_snapshot", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner.preflight, "_hash_executed_files", lambda *_args: {})
    pending_observations = []

    async def fake_pair_attempt(
        _args,
        *,
        result_root,
        run_root,
        fixture,
        run_id,
        round_number,
        competitor,
        attempt_number,
        samples_per_workload,
        on_update=None,
    ):
        run_manifest = json.loads((output / "run-manifest.json").read_text())
        pair = run_manifest["pairs"][0]
        pending = pair["attempts"][0] if pair["attempts"] else None
        pending_observations.append(
            isinstance(pending, dict)
            and pending.get("state") == "pending"
            and pending.get("attempt_number") == attempt_number
            and pending.get("directory") == "round-1/pair-raw-cdp/attempt-1"
            and pending.get("manifest_sha256") is None
        )
        return {
            "attempt_number": attempt_number,
            "state": "failed",
            "directory": "round-1/pair-raw-cdp/attempt-1",
            "manifest_sha256": "2" * 64,
            "cleanup_ok": True,
            "runtime_root_removed": True,
            "host_gate": {"state": "clear", "reasons": []},
            "work_counts": {
                "attempted_work_count": 0,
                "successful_work_count": 0,
                "error_count": 1,
            },
            "error_count": 1,
            "errors": ["simulated task failure"],
        }

    monkeypatch.setattr(runner, "run_pair_attempt", fake_pair_attempt)
    exit_code = await runner.run_navigation(
        SimpleNamespace(
            command="smoke",
            output=output,
            source=private_result_root,
        )
    )
    manifest = json.loads((output / "run-manifest.json").read_text())

    assert pending_observations == [True]
    assert exit_code == 1
    assert manifest["state"] == "incomplete"
    assert len(manifest["pairs"][0]["attempts"]) == 1
    assert manifest["pairs"][0]["attempts"][0]["state"] == "failed"


def test_navigation_phase_errors_reject_boolean_and_out_of_order_times(runner):
    sample = {
        "outcome": "success",
        "start_ns": True,
        "navigation_return_ns": 20,
        "end_ns": 30,
        "elapsed_ns": 29,
        "navigation_ns": 19,
        "post_navigation_ns": 10,
    }
    errors = runner.navigation_phase_errors(sample, "sample")
    assert any("start_ns" in error for error in errors)

    sample.update(start_ns=30, navigation_return_ns=20)
    assert any(
        "out of order" in error
        for error in runner.navigation_phase_errors(sample, "sample")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_point", ["initial_manifest", "running_manifest", "source", "cancel"]
)
async def test_attempt_owns_resources_before_setup_and_through_cancellation(
    runner, private_result_root, monkeypatch, failure_point
):
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    runtime_roots = []
    references = []
    handles = []
    cleaned = []
    started = asyncio.Event()
    allocate = runner.new_runtime_root

    def allocate_runtime():
        runtime_root, token = allocate()
        runtime_roots.append(runtime_root)
        return runtime_root, token

    async def launch_worker(launch):
        handle = SimpleNamespace(
            launch=launch,
            state=runner.navigation.WorkerState.CREATED,
            worker_identity={
                "pid": 2000,
                "created": 2000.0,
                "cmdline": [str(launch.interpreter), "worker"],
                "owner_token": launch.owner_token,
                "owner_root": str(launch.worker_root),
            },
        )
        handles.append(handle)
        return handle

    async def start_worker(handle):
        handle.state = runner.navigation.WorkerState.READY
        started.set()
        await asyncio.Event().wait()

    async def cleanup_worker(handle):
        cleaned.append(handle)
        return {
            "ok": True,
            "worker_identity": {**handle.worker_identity, "state": "absent"},
            "post_worker_exit_identities": [],
        }

    if failure_point in {"initial_manifest", "running_manifest"}:
        writer_name = (
            "write_new_json" if failure_point == "initial_manifest" else "atomic_json"
        )
        writer = getattr(runner, writer_name)
        failed = False

        def fail_once(path, value):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected manifest write failure")
            return writer(path, value)

        monkeypatch.setattr(runner, writer_name, fail_once)

    monkeypatch.setattr(runner, "new_runtime_root", allocate_runtime)
    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    args = SimpleNamespace(
        source=(
            private_result_root / "missing"
            if failure_point == "source"
            else private_result_root
        ),
        python=sys.executable,
        chrome="/tmp/chrome",
    )
    task = asyncio.create_task(
        runner.run_pair_attempt(
            args,
            result_root=private_result_root,
            run_root=run_root,
            fixture=SimpleNamespace(server_address=("127.0.0.1", 12345)),
            run_id="e" * 32,
            round_number=1,
            competitor="raw-cdp",
            attempt_number=1,
            samples_per_workload=1,
            on_update=references.append,
        )
    )
    try:
        if failure_point == "cancel":
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
        caught = outcome if isinstance(outcome, BaseException) else None
        assert runtime_roots and not runtime_roots[0].exists()
        assert cleaned == handles
        manifest_path = (
            run_root / "round-1/pair-raw-cdp/attempt-1/attempt-manifest.json"
        )
        saved = json.loads(manifest_path.read_text())
        assert saved["state"] == "failed"
        assert saved["cleanup"]["ok"] is False
        assert saved["cleanup"]["runtime_root_removed"] is True
        assert saved["work_counts"]["error_count"] == 1
        assert references[-1]["state"] == "failed"
        assert references[-1]["cleanup_ok"] is False
        assert references[-1]["manifest_sha256"] == runner.preflight.sha256_file(
            manifest_path
        )
        if failure_point == "cancel":
            assert isinstance(caught, asyncio.CancelledError)
            assert any("CancelledError" in error for error in saved["errors"])
        else:
            assert caught is None
            assert not handles
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for runtime_root in runtime_roots:
            shutil.rmtree(runtime_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_cancelled_attempt_retains_runtime_when_owned_worker_is_unproven(
    runner, private_result_root, monkeypatch
):
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    references = []
    runtime_roots = []
    allocate = runner.new_runtime_root

    def allocate_runtime():
        root, token = allocate()
        runtime_roots.append(root)
        return root, token

    async def launch_worker(launch):
        return SimpleNamespace(launch=launch, worker_identity={"pid": 2000})

    async def start_worker(_handle):
        raise asyncio.CancelledError("injected cancellation")

    async def cleanup_worker(handle):
        return {
            "ok": False,
            "worker_identity": {**handle.worker_identity, "state": "unknown"},
            "post_worker_exit_identities": [],
        }

    monkeypatch.setattr(runner, "new_runtime_root", allocate_runtime)
    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    try:
        with pytest.raises(asyncio.CancelledError):
            await runner.run_pair_attempt(
                SimpleNamespace(
                    source=private_result_root,
                    python=sys.executable,
                    chrome="/tmp/chrome",
                ),
                result_root=private_result_root,
                run_root=run_root,
                fixture=SimpleNamespace(server_address=("127.0.0.1", 12345)),
                run_id="f" * 32,
                round_number=1,
                competitor="raw-cdp",
                attempt_number=1,
                samples_per_workload=1,
                on_update=references.append,
            )
        assert runtime_roots[0].is_dir()
        manifest_path = (
            run_root / "round-1/pair-raw-cdp/attempt-1/attempt-manifest.json"
        )
        saved = json.loads(manifest_path.read_text())
        assert saved["state"] == "failed"
        assert saved["cleanup"]["runtime_root_removed"] is False
        assert (
            saved["cleanup"]["workers"]["openbrowser"]["worker_identity"]["state"]
            == "unknown"
        )
        assert references[-1]["state"] == "failed"
        assert references[-1]["cleanup_ok"] is False
        assert references[-1]["manifest_sha256"] == runner.preflight.sha256_file(
            manifest_path
        )
    finally:
        for runtime_root in runtime_roots:
            shutil.rmtree(runtime_root, ignore_errors=True)


@pytest.mark.asyncio
async def test_cancellation_during_worker_cleanup_waits_for_owned_resource_proof(
    runner, private_result_root, monkeypatch
):
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    references = []
    handles = []
    cleaned = []
    roots = []
    allocate = runner.new_runtime_root

    def allocate_runtime():
        root, token = allocate()
        roots.append(root)
        return root, token

    async def launch_worker(launch):
        handle = SimpleNamespace(
            launch=launch, worker_identity={"pid": 2000 + len(handles)}
        )
        handles.append(handle)
        return handle

    async def start_worker(_handle):
        return {"ok": True}, []

    async def failed_trial(_handle, _fixture, _request, *, is_warmup):
        assert is_warmup
        return {"state": "failed", "errors": ["injected warmup failure"]}

    async def cleanup_worker(handle):
        cleanup_started.set()
        await cleanup_release.wait()
        cleaned.append(handle)
        return {
            "ok": True,
            "worker_identity": {**handle.worker_identity, "state": "absent"},
            "post_worker_exit_identities": [],
        }

    monkeypatch.setattr(runner, "new_runtime_root", allocate_runtime)
    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    monkeypatch.setattr(runner, "execute_trial", failed_trial)
    task = asyncio.create_task(
        runner.run_pair_attempt(
            SimpleNamespace(
                source=private_result_root, python=sys.executable, chrome="/tmp/chrome"
            ),
            result_root=private_result_root,
            run_root=run_root,
            fixture=SimpleNamespace(server_address=("127.0.0.1", 12345)),
            run_id="a" * 32,
            round_number=1,
            competitor="raw-cdp",
            attempt_number=1,
            samples_per_workload=1,
            on_update=references.append,
        )
    )
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned == list(reversed(handles))
        assert roots and not roots[0].exists()
        saved = json.loads(
            (
                run_root / "round-1/pair-raw-cdp/attempt-1/attempt-manifest.json"
            ).read_text()
        )
        assert saved["state"] == "failed"
        assert saved["cleanup"]["ok"] is True
        assert any("CancelledError" in error for error in saved["errors"])
        assert references[-1]["cleanup_ok"] is True
        assert references[-1]["state"] == "failed"
    finally:
        cleanup_release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for root in roots:
            shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def offline_navigation_verifier():
    path = SCRIPT.with_name("summarize_latest_navigation.py")
    spec = importlib.util.spec_from_file_location(
        "resource_cleanup_offline_verifier", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("started_workers", [0, 1, 2])
async def test_partial_setup_cleanup_preserves_public_two_arm_proof_contract(
    runner, offline_navigation_verifier, monkeypatch, started_workers
):
    root, token = runner.new_runtime_root()
    arms = ("openbrowser", "raw-cdp")
    handles = {
        arm: SimpleNamespace(
            worker_identity={
                "pid": 2000 + index,
                "created": 2000.0 + index,
                "cmdline": ["/private/tmp/fake-worker", arm],
                "owner_token": token,
                "owner_root": str(root / arm),
            }
        )
        for index, arm in enumerate(arms[:started_workers])
    }

    async def cleanup_worker(handle):
        return {
            "ok": True,
            "fallback_used": False,
            "result": {"identities": [], "errors": []},
            "worker_identity": {**handle.worker_identity, "state": "absent"},
            "post_worker_exit_identities": [],
        }

    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    try:
        cleanup = await runner.cleanup_pair_workers(handles, arms, root, token)
        offline_navigation_verifier._validate_attempt_cleanup(
            cleanup, arms, "partial setup", terminal_state="failed"
        )
        assert cleanup["workers_ok"] is (started_workers == 2)
        assert cleanup["ok"] is (started_workers == 2)
        assert cleanup["runtime_root_removed"] is True
        assert not root.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", ["first_warmup", "first_measured"])
async def test_cancelled_trial_artifacts_are_accepted_by_actual_offline_verifier(
    runner, offline_navigation_verifier, private_result_root, monkeypatch, cancel_phase
):
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    started = asyncio.Event()
    references = []
    requests = []
    handles = []

    class Ledger:
        server_address = ("127.0.0.1", 12345)

        def snapshot(self, nonce):
            return {
                "requests": [
                    {
                        "path": path,
                        "method": method,
                        "nonce": nonce,
                        "status": 200,
                        "received_ns": start,
                        "completed_ns": end,
                        "bytes": 64,
                    }
                    for path, method, start, end in (
                        ("/fixture", "GET", 110, 120),
                        ("/effect", "POST", 160, 170),
                    )
                ],
                "effects": 1,
            }

    async def launch_worker(launch):
        handle = SimpleNamespace(
            launch=launch,
            state=runner.navigation.WorkerState.READY,
            worker_identity={
                "pid": 2000 + len(handles),
                "created": 2000.0 + len(handles),
                "cmdline": ["/private/tmp/fake-worker", launch.arm],
                "owner_token": launch.owner_token,
                "owner_root": str(launch.worker_root),
            },
        )
        handles.append(handle)
        return handle

    async def start_worker(_handle):
        return {"ok": True}, []

    async def send_operation(_handle, request, *, timeout):
        requests.append(request)
        if len(requests) == (1 if cancel_phase == "first_warmup" else 3):
            started.set()
            await asyncio.Event().wait()
        nonce, url = request["nonce"], request["url"]
        observation = {
            "url": url,
            "title": f"fixture-{nonce}",
            "readyState": "complete",
            "marker": nonce,
            "loaded": nonce,
            "scriptDone": nonce,
        }
        return {
            "ok": True,
            "result": {
                **request,
                "outcome": "success",
                "start_ns": 100,
                "navigation_return_ns": 150,
                "end_ns": 200,
                "elapsed_ns": 100,
                "navigation_ns": 50,
                "post_navigation_ns": 50,
                "checked_ns": 190,
                "navigation_observation": {**observation, "field": ""},
                "final_observation": {**observation, "field": nonce},
                "effect_observation": {
                    "field": nonce,
                    "response": {"count": 1, "nonce": nonce, "payload": nonce},
                },
            },
        }

    async def cleanup_worker(handle):
        return {
            "ok": True,
            "fallback_used": False,
            "result": {"identities": [], "errors": []},
            "worker_identity": {**handle.worker_identity, "state": "absent"},
            "post_worker_exit_identities": [],
        }

    async def monitor(stop, _samples):
        await stop.wait()

    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "send_operation", send_operation)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    monkeypatch.setattr(runner, "monitor_pair_host", monitor)
    task = asyncio.create_task(
        runner.run_pair_attempt(
            SimpleNamespace(
                source=private_result_root, python=sys.executable, chrome="/tmp/chrome"
            ),
            result_root=private_result_root,
            run_root=run_root,
            fixture=Ledger(),
            run_id="b" * 32,
            round_number=1,
            competitor="raw-cdp",
            attempt_number=1,
            samples_per_workload=1,
            on_update=references.append,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        attempt, totals = offline_navigation_verifier._verify_attempt(
            run_root,
            {"run_id": "b" * 32, "samples_per_workload": 1},
            {"round": 1, "competitor": "raw-cdp"},
            references[-1],
            set(),
        )
        collection = "warmups" if cancel_phase == "first_warmup" else "samples"
        assert attempt[collection][-1]["state"] == "failed"
        assert attempt[collection][-1]["sample"] is None
        assert attempt[collection][-1]["nonce"] == requests[-1]["nonce"]
        assert any(
            "CancelledError" in error for error in attempt[collection][-1]["errors"]
        )
        assert totals["attempted_work_count"] == (cancel_phase == "first_measured")
        assert totals["successful_work_count"] == 0
        assert attempt["cleanup"]["ok"] is True
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("acquired_workers", [0, 1])
async def test_failed_setup_manifest_roundtrips_actual_offline_attempt_verifier(
    runner,
    offline_navigation_verifier,
    private_result_root,
    monkeypatch,
    acquired_workers,
):
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    references = []

    async def launch_worker(launch):
        return SimpleNamespace(
            worker_identity={
                "pid": 2000,
                "created": 2000.0,
                "cmdline": ["/private/tmp/fake-worker", launch.arm],
                "owner_token": launch.owner_token,
                "owner_root": str(launch.worker_root),
            }
        )

    async def start_worker(_handle):
        raise RuntimeError("injected setup failure")

    async def cleanup_worker(handle):
        return {
            "ok": True,
            "fallback_used": False,
            "result": {"identities": [], "errors": []},
            "worker_identity": {**handle.worker_identity, "state": "absent"},
            "post_worker_exit_identities": [],
        }

    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    reference = await runner.run_pair_attempt(
        SimpleNamespace(
            source=private_result_root
            if acquired_workers
            else private_result_root / "missing",
            python=sys.executable,
            chrome="/tmp/chrome",
        ),
        result_root=private_result_root,
        run_root=run_root,
        fixture=SimpleNamespace(server_address=("127.0.0.1", 12345)),
        run_id="c" * 32,
        round_number=1,
        competitor="raw-cdp",
        attempt_number=1,
        samples_per_workload=1,
        on_update=references.append,
    )
    attempt, totals = offline_navigation_verifier._verify_attempt(
        run_root,
        {"run_id": "c" * 32, "samples_per_workload": 1},
        {"round": 1, "competitor": "raw-cdp"},
        reference,
        set(),
    )
    assert len(attempt["cleanup"]["workers"]) == acquired_workers
    assert attempt["cleanup"]["workers_ok"] is False
    assert attempt["cleanup"]["ok"] is False
    assert attempt["cleanup"]["runtime_root_removed"] is True
    assert attempt["state"] == references[-1]["state"] == "failed"
    assert totals["attempted_work_count"] == totals["successful_work_count"] == 0


@pytest.mark.asyncio
async def test_cleanup_exception_manifest_roundtrips_actual_offline_verifier(
    runner, offline_navigation_verifier, private_result_root, monkeypatch
):
    run_root = private_result_root / "navigation"
    run_root.mkdir(mode=0o700)
    runtime_roots = []

    async def launch_worker(launch):
        runtime_roots.append(launch.worker_root.parent)
        return SimpleNamespace(
            worker_identity={
                "pid": 2000,
                "created": 2000.0,
                "cmdline": ["/private/tmp/fake-worker", launch.arm],
                "owner_token": launch.owner_token,
                "owner_root": str(launch.worker_root),
            }
        )

    async def start_worker(_handle):
        raise RuntimeError("injected setup failure")

    async def cleanup_worker(_handle):
        raise OSError("injected cleanup coordinator failure")

    monkeypatch.setattr(runner.navigation, "launch_worker", launch_worker)
    monkeypatch.setattr(runner.navigation, "start_worker", start_worker)
    monkeypatch.setattr(runner.navigation, "cleanup_worker", cleanup_worker)
    try:
        reference = await runner.run_pair_attempt(
            SimpleNamespace(
                source=private_result_root,
                python=sys.executable,
                chrome="/tmp/chrome",
            ),
            result_root=private_result_root,
            run_root=run_root,
            fixture=SimpleNamespace(server_address=("127.0.0.1", 12345)),
            run_id="d" * 32,
            round_number=1,
            competitor="raw-cdp",
            attempt_number=1,
            samples_per_workload=1,
        )
        attempt, totals = offline_navigation_verifier._verify_attempt(
            run_root,
            {"run_id": "d" * 32, "samples_per_workload": 1},
            {"round": 1, "competitor": "raw-cdp"},
            reference,
            set(),
        )
        cleanup = attempt["cleanup"]
        worker = cleanup["workers"]["openbrowser"]
        assert worker["ok"] is False
        assert worker["fallback_used"] is None
        assert worker["worker_identity"]["state"] == "unknown"
        assert worker["error"] == "OSError: injected cleanup coordinator failure"
        assert worker["post_worker_exit_identities"] == []
        assert cleanup["workers_ok"] is cleanup["ok"] is False
        assert cleanup["runtime_root_removed"] is False
        assert runtime_roots and runtime_roots[0].is_dir()
        assert reference["state"] == attempt["state"] == "failed"
        assert reference["cleanup_ok"] is reference["runtime_root_removed"] is False
        assert totals["attempted_work_count"] == totals["successful_work_count"] == 0
        assert totals["sample_results"] == []
        pair_result, _validated, _declared = offline_navigation_verifier._verify_pair(
            run_root,
            {"run_id": "d" * 32, "samples_per_workload": 1, "state": "incomplete"},
            {
                "round": 1,
                "competitor": "raw-cdp",
                "arms": ["openbrowser", "raw-cdp"],
                "state": "failed",
                "attempts": [reference],
                "retry_host_checks": [],
                "errors": attempt["errors"],
            },
            set(),
        )
        assert pair_result["state"] == "failed"
        assert pair_result["metrics"] is None
    finally:
        for root in runtime_roots:
            shutil.rmtree(root, ignore_errors=True)
