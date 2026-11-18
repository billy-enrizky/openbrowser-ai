import hashlib
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "summarize_latest_navigation.py"
SUMMARY_SPEC_ROOT = "local_docs/specs/2026-11-17-latest-navigation-benchmark"
RUN_ID = "a" * 32
COMPETITORS = ("raw-cdp", "lightpanda", "agent-browser", "browser-use")
WORKLOADS = ("fast", "delayed")
EXECUTED_FILES = (
    "benchmarks/navigation_compare.py",
    "benchmarks/latest_competitor_metadata.py",
    "benchmarks/latest_benchmark_preflight.py",
    "benchmarks/latest_navigation_baseline.py",
    "benchmarks/summarize_latest_navigation.py",
    "benchmarks/screenshot_lightpanda.js",
    "benchmarks/screenshot_capability.py",
    "benchmarks/summarize_screenshot_capability.py",
)
LATEST_VERSIONS = {
    "lightpanda": "1.0.4",
    "agent-browser": "0.43.0",
    "browser-use": "1.0.6",
    "browser-harness": "0.1.13",
    "chrome_for_testing": "158.0.8127.36",
}
INSTALLED_TOOL_VERSIONS = {
    name: LATEST_VERSIONS[name]
    for name in ("lightpanda", "agent-browser", "chrome_for_testing")
}
INSTALLED_PACKAGE_VERSIONS = {
    **INSTALLED_TOOL_VERSIONS,
    "browser-use": LATEST_VERSIONS["browser-use"],
    "browser-harness": LATEST_VERSIONS["browser-harness"],
    "candidate:psutil": "7.1.0",
    "browser_use:psutil": "7.1.0",
    "candidate:websockets": "15.0.1",
    "browser_use:websockets": "15.0.1",
}


@pytest.fixture(autouse=True)
def clear_aws_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def capture_summary_logs(monkeypatch):
    # MCP initialization redirects existing loggers away from pytest's root handler.
    logger = logging.getLogger("summarize-latest-navigation")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)


@pytest.fixture
def private_result_root():
    root = (
        Path("/tmp").resolve(strict=True)
        / f"openbrowser-navigation-summary-test-{uuid.uuid4().hex}"
    )
    root.mkdir(mode=0o700)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_summary_environment_excludes_inherited_aws_variables():
    assert not [name for name in os.environ if name.startswith("AWS_")]


def test_clear_aws_environment_restores_parent_environment(monkeypatch):
    inherited = {
        "AWS_PROFILE": "summary-test-profile",
        "AWS_CONFIG_FILE": "/summary-test-config",
        "AWS_SUMMARY_TEST_UNKNOWN": "summary-test-value",
    }
    with monkeypatch.context() as parent:
        for name, value in inherited.items():
            parent.setenv(name, value)
        with monkeypatch.context() as isolated:
            clear_aws_environment.__wrapped__(isolated)
            assert not [name for name in os.environ if name.startswith("AWS_")]
        assert {name: os.environ[name] for name in inherited} == inherited
    assert not [name for name in os.environ if name.startswith("AWS_")]


def test_summary_log_capture_restores_parent_configuration(
    private_result_root, monkeypatch, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    parent_handlers = [logging.NullHandler()]
    with monkeypatch.context() as parent:
        parent.setattr(summarizer.LOG, "handlers", parent_handlers)
        parent.setattr(summarizer.LOG, "propagate", False)
        with monkeypatch.context() as isolated:
            capture_summary_logs.__wrapped__(isolated)
            args, _run_root, _project_root = _build_run(
                private_result_root, boolean_start=True, run_state="incomplete"
            )
            assert summarizer.dispatch(args) == 1
            assert "sample.start_ns must be an exact integer" in caplog.text
        assert summarizer.LOG.handlers is parent_handlers
        assert summarizer.LOG.propagate is False


def _load_summarizer():
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location(
        "summarize_latest_navigation_under_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("offline navigation summary verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_bytes(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )


def _write_json(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    path.chmod(0o600)
    return path.read_bytes()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _identity():
    runtime_tree = "c" * 40
    return {
        "schema": "openbrowser-candidate-identity/1",
        "candidate_commit": "a1" * 20,
        "candidate_tree": "b2" * 20,
        "runtime_tree": runtime_tree,
        "release_runtime_tree": runtime_tree,
        "origin_main_commit": "d3" * 20,
        "origin_main_runtime_tree": runtime_tree,
        "executed_files": {
            name: _sha((ROOT / name).read_bytes()) for name in EXECUTED_FILES
        },
        "candidate_python": {
            "supplied_path": "/private/tmp/candidate-env/bin/python",
            "resolved_path": "/private/tmp/candidate-env/bin/python",
            "version": "3.12.12",
            "executable_sha256": "e" * 64,
        },
        "browser_use_python": {
            "supplied_path": "/private/tmp/browser-use-env/bin/python",
            "resolved_path": "/private/tmp/browser-use-env/bin/python",
            "version": "3.12.12",
            "executable_sha256": "f" * 64,
        },
    }


def _host_sample():
    return {
        "captured_at_utc": "2026-11-17T00:00:00Z",
        "load_1m": 0.25,
        "logical_cpu_count": 8,
        "processes": [],
        "classification_errors": [],
    }


def _trial_record(
    run_id,
    round_number,
    competitor,
    attempt_number,
    arm,
    workload,
    index,
    *,
    warmup=False,
    boolean_start=False,
):
    nonce = f"{run_id}-r{round_number}-{competitor}-a{attempt_number}-{arm}-{workload}-{index}-{uuid.uuid4().hex}"
    url = f"http://127.0.0.1:49152/fixture?{urlencode((('nonce', nonce), ('delayed', int(workload == 'delayed'))))}"
    start = (
        1_000_000_000 + index * 10_000_000 + (0 if arm == "openbrowser" else 500_000)
    )
    navigation_ns = (
        20_000_000 if workload == "fast" else 160_000_000
    ) + index * 100_000
    navigation_return = start + navigation_ns
    elapsed = navigation_ns + 5_000_000
    end = start + elapsed
    recorded_start = True if boolean_start else start
    requests = [
        {
            "path": "/fixture",
            "nonce": nonce,
            "method": "GET",
            "received_ns": start + 1_000,
            "status": 200,
            "bytes": 320,
            "completed_ns": start + 2_000,
        },
    ]
    if workload == "delayed":
        requests.append(
            {
                "path": "/slow.js",
                "nonce": nonce,
                "method": "GET",
                "received_ns": start + 3_000,
                "status": 200,
                "bytes": 58,
                "completed_ns": navigation_return - 1_000,
            }
        )
    requests.append(
        {
            "path": "/effect",
            "nonce": nonce,
            "method": "POST",
            "received_ns": navigation_return + 1_000,
            "status": 200,
            "bytes": 64,
            "completed_ns": navigation_return + 2_000,
        }
    )
    ledger = {"requests": requests, "effects": 1}
    observation = {
        "url": url,
        "title": f"fixture-{nonce}",
        "readyState": "complete",
        "marker": nonce,
        "loaded": nonce,
        "scriptDone": nonce,
        "field": "",
    }
    worker_sample = {
        "op": "trial",
        "arm": arm,
        "kind": workload,
        "index": index,
        "nonce": nonce,
        "url": url,
        "start_ns": recorded_start,
        "navigation_return_ns": navigation_return,
        "end_ns": end,
        "elapsed_ns": elapsed,
        "navigation_ns": navigation_ns,
        "post_navigation_ns": elapsed - navigation_ns,
        "checked_ns": end - 100,
        "navigation_observation": dict(observation),
        "effect_observation": {
            "field": nonce,
            "response": {"count": 1, "nonce": nonce, "payload": nonce},
        },
        "final_observation": {**observation, "field": nonce},
        "outcome": "success",
        "server": ledger,
    }
    return {
        "arm": arm,
        "workload": workload,
        "index": index,
        "nonce": nonce,
        "url": url,
        "warmup": warmup,
        "state": "verified",
        "sample": worker_sample,
        "request_proof": ledger,
        "errors": [],
    }


def _arm_order(index, competitor):
    pair = ("openbrowser", competitor)
    return pair if index % 2 == 0 else tuple(reversed(pair))


def _make_attempt(
    run_root,
    run_id,
    round_number,
    competitor,
    attempt_number=1,
    *,
    samples=6,
    arms=None,
    boolean_start=False,
):
    arms = arms or ("openbrowser", competitor)
    directory = f"round-{round_number}/pair-{competitor}/attempt-{attempt_number}"
    attempt_dir = run_root / directory
    attempt_dir.mkdir(mode=0o700, parents=True)
    for parent in (attempt_dir, attempt_dir.parent, attempt_dir.parent.parent):
        parent.chmod(0o700)
    execution_order = []
    measured = []
    for workload in WORKLOADS:
        for index in range(samples):
            for arm in _arm_order(index, competitor):
                execution_order.append(
                    {"workload": workload, "index": index, "arm": arm}
                )
                measured.append(
                    _trial_record(
                        run_id,
                        round_number,
                        competitor,
                        attempt_number,
                        arm,
                        workload,
                        index,
                        boolean_start=boolean_start and not measured,
                    )
                )
    warmups = [
        _trial_record(
            run_id,
            round_number,
            competitor,
            attempt_number,
            arm,
            "fast",
            0,
            warmup=True,
        )
        for arm in arms
    ]
    host_samples = [_host_sample()]
    host_gate = {"state": "clear", "reasons": []}
    work_counts = {
        "attempted_work_count": len(measured),
        "successful_work_count": len(measured),
        "error_count": 0,
    }
    cleanup = {
        "ok": True,
        "workers_ok": True,
        "runtime_root": str(
            Path("/tmp").resolve(strict=True)
            / f"nb-navigation-{round_number}-{competitor}-{attempt_number}"
        ),
        "runtime_root_removed": True,
        "workers": {},
        "errors": [],
    }
    owner_token = uuid.uuid4().hex
    runtime_root = cleanup["runtime_root"]
    for arm in arms:
        cleanup["workers"][arm] = {
            "ok": True,
            "fallback_used": False,
            "result": {"identities": [], "errors": []},
            "worker_identity": {
                "pid": 1000 + len(cleanup["workers"]),
                "created": float(1000 + len(cleanup["workers"])),
                "cmdline": ["/private/tmp/worker", arm],
                "owner_token": owner_token,
                "owner_root": f"{runtime_root}/{arm}",
                "state": "absent",
            },
            "post_worker_exit_identities": [],
        }
        cleanup["workers"][arm]["result"]["identities"] = [
            dict(cleanup["workers"][arm]["worker_identity"])
        ]
        cleanup["workers"][arm]["post_worker_exit_identities"] = [
            dict(cleanup["workers"][arm]["worker_identity"])
        ]
    attempt = {
        "schema": "openbrowser-navigation-pair-attempt/1",
        "run_id": run_id,
        "round": round_number,
        "competitor": competitor,
        "attempt_number": attempt_number,
        "state": "verified",
        "execution_order": execution_order,
        "warmups": warmups,
        "samples": measured,
        "host_samples": host_samples,
        "host_gate": host_gate,
        "work_counts": work_counts,
        "cleanup": cleanup,
        "errors": [],
    }
    raw = _write_json(attempt_dir / "attempt-manifest.json", attempt)
    reference = {
        "attempt_number": attempt_number,
        "state": "verified",
        "directory": directory,
        "manifest_sha256": _sha(raw),
        "cleanup_ok": True,
        "runtime_root_removed": True,
        "host_gate": host_gate,
        "work_counts": work_counts,
        "error_count": 0,
        "errors": [],
    }
    return reference, attempt


def _build_run(
    result_root,
    *,
    mode="run",
    only_competitor="raw-cdp",
    boolean_start=False,
    all_pairs=False,
    run_state=None,
):
    run_id = RUN_ID
    smoke = mode == "smoke"
    run_root = (
        result_root / "smoke" / "navigation" if smoke else result_root / "navigation"
    )
    run_root.mkdir(mode=0o700, parents=True)
    run_root.chmod(0o700)
    if smoke:
        run_root.parent.chmod(0o700)

    identity = _identity()
    identity_raw = _write_json(result_root / "candidate-identity.json", identity)
    identity_sha = _sha(identity_raw)
    version_sha = "1" * 64
    phase = "navigation-smoke" if smoke else "navigation-run"
    preflight = {
        "schema": "openbrowser-latest-preflight/1",
        "phase": phase,
        "candidate_identity_sha256": identity_sha,
        "version_manifest_sha256": version_sha,
        "checks": [
            {
                "name": "latest_versions",
                "expected": LATEST_VERSIONS,
                "actual": LATEST_VERSIONS,
                "state": "passed",
            },
            {
                "name": "installed_tool_versions",
                "expected": INSTALLED_TOOL_VERSIONS,
                "actual": INSTALLED_TOOL_VERSIONS,
                "state": "passed",
            },
            {
                "name": "installed_package_versions",
                "expected": INSTALLED_PACKAGE_VERSIONS,
                "actual": INSTALLED_PACKAGE_VERSIONS,
                "state": "passed",
            },
        ],
        "outcome": "passed",
        "errors": [],
    }
    preflight_path = result_root / f"preflight-{phase}.json"
    preflight_raw = _write_json(preflight_path, preflight)
    frozen = None
    frozen_hash = None
    smoke_preflight_hashes = {}
    if not smoke:
        for name in ("navigation", "screenshot"):
            smoke_phase = f"{name}-smoke"
            smoke_preflight_path = result_root / f"preflight-{smoke_phase}.json"
            smoke_preflight_raw = _write_json(
                smoke_preflight_path,
                {
                    "schema": "openbrowser-latest-preflight/1",
                    "phase": smoke_phase,
                    "candidate_identity_sha256": identity_sha,
                    "version_manifest_sha256": version_sha,
                    "checks": [
                        {
                            "name": "fixture",
                            "expected": True,
                            "actual": True,
                            "state": "passed",
                        }
                    ],
                    "outcome": "passed",
                    "errors": [],
                },
            )
            smoke_preflight_hashes[name] = _sha(smoke_preflight_raw)
        frozen = {
            "schema": "openbrowser-frozen-harness/1",
            "candidate_identity_sha256": identity_sha,
            "version_manifest_sha256": version_sha,
            "tool_manifest_sha256": "2" * 64,
            "candidate_freeze_sha256": "3" * 64,
            "browser_use_freeze_sha256": "4" * 64,
            "smoke_preflight_sha256": smoke_preflight_hashes,
            "smoke_manifest_sha256": {"navigation": "7" * 64, "screenshot": "8" * 64},
            "executed_file_sha256": identity["executed_files"],
        }
        frozen_raw = _write_json(result_root / "frozen-harness.json", frozen)
        frozen_hash = _sha(frozen_raw)

    competitors = (
        ("raw-cdp",)
        if smoke
        else ((only_competitor,) if not all_pairs else COMPETITORS)
    )
    pairs = []
    total = {
        "attempted_work_count": 0,
        "successful_work_count": 0,
        "error_count": 0,
        "host_contaminated_attempt_count": 0,
    }
    rounds = (1,) if smoke or not all_pairs else (1, 2)
    for round_number in rounds:
        round_competitors = competitors
        if all_pairs and round_number == 2:
            round_competitors = tuple(reversed(competitors))
        for competitor in round_competitors:
            reference, attempt = _make_attempt(
                run_root,
                run_id,
                round_number,
                competitor,
                samples=1 if smoke else 6,
                boolean_start=boolean_start and not pairs,
            )
            pairs.append(
                {
                    "round": round_number,
                    "competitor": competitor,
                    "arms": ["openbrowser", competitor],
                    "state": "verified",
                    "attempts": [reference],
                    "retry_host_checks": [],
                    "errors": [],
                }
            )
            for field in (
                "attempted_work_count",
                "successful_work_count",
                "error_count",
            ):
                total[field] += attempt["work_counts"][field]

    if run_state is None:
        expected_pair_count = 1 if smoke else 8
        run_state = "complete" if len(pairs) == expected_pair_count else "incomplete"
    manifest = {
        "schema": "openbrowser-latest-navigation-smoke/1"
        if smoke
        else "openbrowser-latest-navigation-run/1",
        "mode": mode,
        "run_id": run_id,
        "state": run_state,
        "created_at_utc": "2026-11-17T00:00:00Z",
        "candidate_identity_sha256": identity_sha,
        "version_manifest_sha256": version_sha,
        "preflight_sha256": _sha(preflight_raw),
        "frozen_harness_sha256": frozen_hash,
        "harness_sha256": identity["executed_files"][
            "benchmarks/latest_navigation_baseline.py"
        ],
        "source_commit": identity["candidate_commit"],
        "source_tree": identity["candidate_tree"],
        "arms": ["openbrowser", "raw-cdp"]
        if smoke
        else [
            "openbrowser",
            "raw-cdp",
            "lightpanda",
            "agent-browser",
            "browser-use",
        ],
        "workloads": list(WORKLOADS),
        "rounds": 1 if smoke else 2,
        "samples_per_workload": 1 if smoke else 6,
        "pair_order_by_round": {"1": ["raw-cdp"]}
        if smoke
        else {
            "1": list(COMPETITORS),
            "2": list(reversed(COMPETITORS)),
        },
        "pairs": pairs,
        "host_preflight": {
            "state": "clear",
            "reasons": [],
            "logical_cpu_count": 8,
            "uptime": "test fixture",
            "samples": [_host_sample() for _ in range(6)],
        },
        "work_counts": total,
        "fixture": {
            "started": True,
            "stopped": True,
            "socket_closed": True,
            "thread_stopped": True,
        },
        "cleanup": {
            "ok": True,
            "workers_ok": True,
            "runtime_roots_removed": True,
            "fixture_stopped": True,
        },
        "failed_phase": None,
        "errors": [],
    }
    _write_json(run_root / "run-manifest.json", manifest)
    project_root = result_root / "project"
    project_root.mkdir(mode=0o700)
    args = [
        "verify-smoke" if smoke else "verify",
        "--run-root",
        str(run_root),
        "--candidate-identity",
        str(result_root / "candidate-identity.json"),
        "--preflight",
        str(preflight_path),
    ]
    if not smoke:
        args.extend(
            [
                "--frozen-harness",
                str(result_root / "frozen-harness.json"),
                "--output",
                str(run_root / "summary.json"),
                "--project-root",
                str(project_root),
            ]
        )
    return args, run_root, project_root


def _rewrite_first_attempt(run_root, mutation):
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    reference = run_manifest["pairs"][0]["attempts"][0]
    attempt_path = run_root / reference["directory"] / "attempt-manifest.json"
    attempt = json.loads(attempt_path.read_text())
    mutation(attempt)
    raw = _write_json(attempt_path, attempt)
    reference["manifest_sha256"] = _sha(raw)
    _write_json(run_manifest_path, run_manifest)


def _rewrite_run_manifest(run_root, mutation):
    path = run_root / "run-manifest.json"
    manifest = json.loads(path.read_text())
    mutation(manifest)
    _write_json(path, manifest)


def test_summary_rejects_boolean_phase_time(private_result_root, caplog):
    assert SCRIPT.is_file(), "offline navigation summary verifier is missing"
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, _run_root, _project_root = _build_run(
        private_result_root,
        boolean_start=True,
        run_state="incomplete",
    )

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "start_ns" in caplog.text


def test_summary_reports_a_verified_partial_pair_without_ranking(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    exit_code = summarizer.dispatch(args)
    assert exit_code == 0, caplog.text
    summary = json.loads((run_root / "summary.json").read_text())

    assert exit_code == 0
    assert "ranking" not in summary
    assert summary["versions"] == {
        "openbrowser": "0.1.56",
        "raw-cdp": "158.0.8127.36",
        "lightpanda": "1.0.4",
        "agent-browser": "0.43.0",
        "browser-use": "1.0.6",
    }
    assert summary["dependencies"] == {"browser-harness": "0.1.13"}
    assert list(summary["pair_results"]) == ["round-1/raw-cdp"]
    metrics = summary["pair_results"]["round-1/raw-cdp"]["metrics"]
    assert metrics["openbrowser"]["fast"]["navigation_ns"]["p50"] == 20_250_000
    assert metrics["openbrowser"]["fast"]["navigation_ns"]["min"] == 20_000_000
    assert metrics["openbrowser"]["fast"]["navigation_ns"]["max"] == 20_500_000
    assert metrics["raw-cdp"]["delayed"]["elapsed_ns"]["p50"] == 165_250_000
    assert metrics["raw-cdp"]["delayed"]["samples"] == 6
    assert metrics["raw-cdp"]["delayed"]["errors"] == 0


def test_summary_rejects_runner_files_changed_after_candidate_identity(
    private_result_root,
    monkeypatch,
):
    summarizer = _load_summarizer()
    args, run_root, project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    original_hashes = summarizer.preflight._hash_executed_files

    def changed_runner_hashes(source):
        hashes = original_hashes(source)
        hashes["benchmarks/navigation_compare.py"] = "0" * 64
        return hashes

    monkeypatch.setattr(
        summarizer.preflight, "_hash_executed_files", changed_runner_hashes
    )
    verification_args = SimpleNamespace(
        run_root=str(run_root),
        candidate_identity=args[args.index("--candidate-identity") + 1],
        preflight=args[args.index("--preflight") + 1],
        frozen_harness=args[args.index("--frozen-harness") + 1],
        output=str(run_root / "summary.json"),
        project_root=str(project_root),
        sanitized_output=None,
    )

    with pytest.raises(
        summarizer.VerificationError, match="runner or verifier files changed"
    ):
        summarizer.verify_run(verification_args)


def test_summary_rejects_a_changed_frozen_smoke_preflight(private_result_root, caplog):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    result_root = run_root.parent
    smoke_preflight = result_root / "preflight-navigation-smoke.json"
    smoke_preflight.write_text(smoke_preflight.read_text() + " ")
    smoke_preflight.chmod(0o600)

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "smoke preflight hash differs" in caplog.text
    assert not (run_root / "summary.json").exists()


def test_summary_accepts_pre_retry_and_terminal_host_gates_for_contaminated_pair(
    monkeypatch,
):
    summarizer = _load_summarizer()
    attempt_references = []
    for attempt_number in (1, 2):
        attempt_references.append(
            {
                "attempt_number": attempt_number,
                "state": "contaminated",
                "directory": f"round-1/pair-raw-cdp/attempt-{attempt_number}",
                "manifest_sha256": str(attempt_number) * 64,
                "cleanup_ok": True,
                "runtime_root_removed": True,
                "host_gate": {"state": "contaminated", "reasons": ["busy host"]},
                "work_counts": {
                    "attempted_work_count": 24,
                    "successful_work_count": 24,
                    "error_count": 0,
                },
                "error_count": 0,
                "errors": [],
            }
        )
    pair = {
        "round": 1,
        "competitor": "raw-cdp",
        "arms": ["openbrowser", "raw-cdp"],
        "state": "contaminated",
        "attempts": attempt_references,
        "retry_host_checks": [
            {"state": "clear", "reasons": []},
            {"state": "clear", "reasons": []},
        ],
        "errors": [],
    }

    def verified_attempt(_run_root, _run_manifest, _pair, reference, _nonce_set):
        return (
            {"state": reference["state"], "errors": []},
            {
                "attempted_work_count": 24,
                "successful_work_count": 24,
                "error_count": 0,
                "host_contaminated_attempt_count": 1,
                "sample_results": [],
                "cleanup": {"ok": True, "runtime_root_removed": True},
                "host_gate": reference["host_gate"],
            },
        )

    monkeypatch.setattr(summarizer, "_verify_attempt", verified_attempt)
    result, _counts, _declared = summarizer._verify_pair(
        Path("/tmp"),
        {"run_id": "f" * 32, "samples_per_workload": 6, "state": "incomplete"},
        pair,
        set(),
    )

    assert result["state"] == "contaminated"
    assert result["verification_state"] == "incomplete"
    assert result["metrics"] is None


def test_summary_counts_a_sample_error_once_across_nested_and_attempt_records(
    private_result_root,
    caplog,
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root,
        all_pairs=True,
        run_state="incomplete",
    )
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    pair = run_manifest["pairs"][0]
    reference = pair["attempts"][0]
    attempt_path = run_root / reference["directory"] / "attempt-manifest.json"
    attempt = json.loads(attempt_path.read_text())
    sample = attempt["samples"][0]
    sample["state"] = "failed"
    sample["errors"] = ["task failed"]
    attempt_errors = ["r1/raw-cdp/a1/openbrowser/fast/0: task failed"]
    attempt["state"] = "failed"
    attempt["errors"] = attempt_errors
    attempt["work_counts"] = {
        "attempted_work_count": 24,
        "successful_work_count": 23,
        "error_count": 1,
    }
    attempt_raw = _write_json(attempt_path, attempt)
    reference.update(
        state="failed",
        manifest_sha256=_sha(attempt_raw),
        work_counts=attempt["work_counts"],
        error_count=1,
        errors=attempt_errors,
    )
    pair.update(state="failed", errors=attempt_errors)
    run_manifest["work_counts"] = {
        "attempted_work_count": 192,
        "successful_work_count": 191,
        "error_count": 1,
        "host_contaminated_attempt_count": 0,
    }
    _write_json(run_manifest_path, run_manifest)

    exit_code = summarizer.dispatch(args)
    summary = json.loads((run_root / "summary.json").read_text())

    assert exit_code == 0, caplog.text
    assert "ranking" not in summary
    assert summary["work_counts"]["error_count"] == 1
    assert summary["pair_results"]["round-1/raw-cdp"]["error_count"] == 1


def test_summary_emits_pairwise_ranking_only_after_all_eight_verified_pairs(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(private_result_root, all_pairs=True)

    exit_code = summarizer.dispatch(args)
    assert exit_code == 0, caplog.text
    summary = json.loads((run_root / "summary.json").read_text())

    assert summary["state"] == "complete"
    assert len(summary["pair_results"]) == 8
    assert "ranking" in summary
    assert summary["versions"]["openbrowser"] == "0.1.56"
    assert summary["versions"]["lightpanda"] == "1.0.4"
    assert (
        summary["ranking"]["scope"]
        == "pairwise adapter navigation on the named local workloads only"
    )
    assert summary["ranking"]["common_navigation_win"] is False
    assert summary["ranking"]["comparisons"]["raw-cdp"]["fast"] == "INCONCLUSIVE"
    assert summary["pair_results"]["round-1/raw-cdp"]["metrics"]["openbrowser"]["fast"][
        "navigation_ns"
    ] == {
        "p50": 20_250_000,
        "min": 20_000_000,
        "max": 20_500_000,
    }


def test_verify_smoke_checks_only_openbrowser_and_raw_cdp_controls(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(private_result_root, mode="smoke")

    exit_code = summarizer.dispatch(args)

    assert exit_code == 0, caplog.text
    assert not (run_root / "summary.json").exists()


def test_verify_smoke_rejects_a_control_with_an_insufficient_delayed_gap(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(private_result_root, mode="smoke")

    def shrink_gap(attempt):
        record = next(
            item
            for item in attempt["samples"]
            if item["arm"] == "openbrowser" and item["workload"] == "delayed"
        )
        sample = record["sample"]
        start = sample["start_ns"]
        navigation_return = start + 90_000_000
        end = navigation_return + 5_000_000
        sample.update(
            navigation_return_ns=navigation_return,
            navigation_ns=90_000_000,
            post_navigation_ns=5_000_000,
            elapsed_ns=95_000_000,
            end_ns=end,
            checked_ns=end - 100,
        )
        ledger = sample["server"]
        for event in ledger["requests"]:
            if event["path"] == "/slow.js":
                event["completed_ns"] = navigation_return - 1_000
            elif event["path"] == "/effect":
                event["received_ns"] = navigation_return + 1_000
                event["completed_ns"] = navigation_return + 2_000
        record["request_proof"] = ledger

    _rewrite_first_attempt(run_root, shrink_gap)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "100000000" in caplog.text


def test_summary_reads_only_attempt_paths_named_in_the_run_manifest(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    orphan = (
        run_root / "round-2" / "pair-lightpanda" / "attempt-1" / "attempt-manifest.json"
    )
    orphan.parent.mkdir(mode=0o700, parents=True)
    orphan.write_text("{ not valid JSON }")

    exit_code = summarizer.dispatch(args)

    assert exit_code == 0, caplog.text
    summary = json.loads((run_root / "summary.json").read_text())
    assert list(summary["pair_results"]) == ["round-1/raw-cdp"]


def test_summary_marks_an_attempt_with_a_hash_mismatch_unverified(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["pairs"][0]["attempts"][0]["manifest_sha256"] = "0" * 64
    _write_json(run_manifest_path, run_manifest)

    exit_code = summarizer.dispatch(args)
    summary = json.loads((run_root / "summary.json").read_text())
    pair_result = summary["pair_results"]["round-1/raw-cdp"]

    assert exit_code == 0, caplog.text
    assert summary["verification_state"] == "partial"
    assert summary["unverified_attempt_count"] == 1
    assert "ranking" not in summary
    assert pair_result["state"] == "unverified"
    assert pair_result["manifest_state"] == "verified"
    assert pair_result["metrics"] is None
    assert pair_result["attempted_work_count"] == 0
    assert pair_result["successful_work_count"] == 0


def test_hash_mismatch_in_a_complete_run_suppresses_all_ranking(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(private_result_root, all_pairs=True)
    run_manifest = json.loads((run_root / "run-manifest.json").read_text())
    reference = run_manifest["pairs"][0]["attempts"][0]
    attempt_path = run_root / reference["directory"] / "attempt-manifest.json"
    attempt_path.write_bytes(attempt_path.read_bytes() + b" ")

    exit_code = summarizer.dispatch(args)
    summary = json.loads((run_root / "summary.json").read_text())

    assert exit_code == 0, caplog.text
    assert summary["state"] == "complete"
    assert summary["verification_state"] == "partial"
    assert summary["unverified_attempt_count"] == 1
    assert "ranking" not in summary
    assert summary["pair_results"]["round-1/raw-cdp"]["state"] == "unverified"
    assert summary["pair_results"]["round-1/raw-cdp"]["metrics"] is None
    assert summary["pair_results"]["round-1/lightpanda"]["metrics"] is not None


def test_summary_rejects_a_failed_dom_side_effect_proof(private_result_root, caplog):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def alter_response(attempt):
        attempt["samples"][0]["sample"]["effect_observation"]["response"]["count"] = 2

    _rewrite_first_attempt(run_root, alter_response)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "side-effect response proof" in caplog.text


def test_summary_rejects_work_counts_that_disagree_with_attempt_samples(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def lower_success_count(attempt):
        attempt["work_counts"]["successful_work_count"] -= 1

    _rewrite_first_attempt(run_root, lower_success_count)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "work_counts" in caplog.text


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("navigation_ns", 20.0, "navigation_ns"),
        ("post_navigation_ns", -1, "post_navigation_ns"),
        ("navigation_return_ns", "missing", "unknown or missing keys"),
        ("navigation_return_ns", "past-end", "out of interval"),
    ],
)
def test_summary_rejects_malformed_phase_timing(
    private_result_root, caplog, field, replacement, expected_error
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def change_timing(attempt):
        sample = attempt["samples"][0]["sample"]
        if replacement == "missing":
            sample.pop(field)
        elif replacement == "past-end":
            sample[field] = sample["end_ns"] + 1
        else:
            sample[field] = replacement

    _rewrite_first_attempt(run_root, change_timing)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert expected_error in caplog.text


def test_summary_rejects_a_fixture_response_outside_the_timer(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def move_request(attempt):
        record = attempt["samples"][0]
        request = next(
            event
            for event in record["request_proof"]["requests"]
            if event["path"] == "/fixture"
        )
        worker_request = next(
            event
            for event in record["sample"]["server"]["requests"]
            if event["path"] == "/fixture"
        )
        request["received_ns"] = record["sample"]["end_ns"] + 1
        worker_request["received_ns"] = request["received_ns"]

    _rewrite_first_attempt(run_root, move_request)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "outside the timer" in caplog.text


def test_summary_rejects_a_missing_delayed_script_request(private_result_root, caplog):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def remove_slow_request(attempt):
        record = next(
            item for item in attempt["samples"] if item["workload"] == "delayed"
        )
        record["request_proof"]["requests"] = [
            event
            for event in record["request_proof"]["requests"]
            if event["path"] != "/slow.js"
        ]
        record["sample"]["server"]["requests"] = [
            event
            for event in record["sample"]["server"]["requests"]
            if event["path"] != "/slow.js"
        ]

    _rewrite_first_attempt(run_root, remove_slow_request)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "unexpected work" in caplog.text


def test_summary_rejects_a_candidate_source_revision_mismatch(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    _rewrite_run_manifest(
        run_root, lambda manifest: manifest.update(source_commit="0" * 40)
    )

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "source revision" in caplog.text


def test_summary_rejects_a_non_string_run_state_without_raising(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    _rewrite_run_manifest(run_root, lambda manifest: manifest.update(state=[]))

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "run manifest.state" in caplog.text


def test_summary_rejects_a_preflight_for_a_different_candidate(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    preflight_path = private_result_root / "preflight-navigation-run.json"
    preflight_value = json.loads(preflight_path.read_text())
    preflight_value["candidate_identity_sha256"] = "0" * 64
    preflight_raw = _write_json(preflight_path, preflight_value)
    _rewrite_run_manifest(
        run_root, lambda manifest: manifest.update(preflight_sha256=_sha(preflight_raw))
    )

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "preflight does not match candidate identity" in caplog.text


def test_summary_rejects_a_frozen_harness_for_a_different_candidate(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    frozen_path = private_result_root / "frozen-harness.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["candidate_identity_sha256"] = "0" * 64
    frozen_raw = _write_json(frozen_path, frozen)
    _rewrite_run_manifest(
        run_root,
        lambda manifest: manifest.update(frozen_harness_sha256=_sha(frozen_raw)),
    )

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "frozen harness does not match the candidate identity" in caplog.text


def test_summary_rejects_non_counterbalanced_attempt_order(private_result_root, caplog):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def change_order(attempt):
        attempt["execution_order"][0]["arm"] = "raw-cdp"

    _rewrite_first_attempt(run_root, change_order)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "counterbalanced schedule" in caplog.text


def test_summary_rejects_cleanup_that_claims_a_live_worker_is_retired(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def mark_worker_alive(attempt):
        attempt["cleanup"]["workers"]["openbrowser"]["worker_identity"]["state"] = (
            "alive"
        )

    _rewrite_first_attempt(run_root, mark_worker_alive)
    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "retirement proof" in caplog.text


def test_summary_rejects_cleanup_without_post_exit_evidence_for_a_child(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def omit_child_evidence(attempt):
        worker = attempt["cleanup"]["workers"]["openbrowser"]
        child = {**worker["worker_identity"], "pid": 2000, "cmdline": ["chrome"]}
        worker["result"]["identities"].append(child)

    _rewrite_first_attempt(run_root, omit_child_evidence)

    assert summarizer.dispatch(args) == 1
    assert "retirement proof" in caplog.text


@pytest.mark.parametrize("field", ["pid", "created", "owner_token", "cmdline"])
def test_summary_rejects_cleanup_with_mismatched_post_exit_identity(
    private_result_root, caplog, field
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def mismatch_identity(attempt):
        worker = attempt["cleanup"]["workers"]["openbrowser"]
        terminal = worker["post_worker_exit_identities"][0]
        terminal[field] = {
            "pid": terminal["pid"] + 1,
            "created": terminal["created"] + 1,
            "owner_token": "f" * 32,
            "cmdline": ["different-worker"],
        }[field]

    _rewrite_first_attempt(run_root, mismatch_identity)

    assert summarizer.dispatch(args) == 1
    expected_error = (
        "owner_token differs" if field == "owner_token" else "retirement proof"
    )
    assert expected_error in caplog.text


@pytest.mark.parametrize("state", ["alive", "zombie", "unknown"])
def test_summary_rejects_cleanup_with_a_nonretired_post_exit_identity(
    private_result_root, caplog, state
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def mark_child_not_retired(attempt):
        worker = attempt["cleanup"]["workers"]["openbrowser"]
        worker["post_worker_exit_identities"][0]["state"] = state

    _rewrite_first_attempt(run_root, mark_child_not_retired)

    assert summarizer.dispatch(args) == 1
    assert "retirement proof" in caplog.text


@pytest.mark.parametrize("stop_state", ["absent", "reused"])
@pytest.mark.parametrize("terminal_state", ["absent", "reused"])
def test_summary_accepts_cleanup_rechecked_after_fallback(
    private_result_root, stop_state, terminal_state
):
    summarizer = _load_summarizer()
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def recheck_child_after_fallback(attempt):
        worker = attempt["cleanup"]["workers"]["openbrowser"]
        worker["fallback_used"] = True
        worker["result"]["identities"][0]["state"] = stop_state
        worker["post_worker_exit_identities"][0]["state"] = terminal_state

    _rewrite_first_attempt(run_root, recheck_child_after_fallback)

    assert summarizer.dispatch(args) == 0


@pytest.mark.parametrize(
    "stop_state", ["absent", "reused", "alive", "zombie", "unknown"]
)
@pytest.mark.parametrize("terminal_state", ["absent", "reused"])
async def test_worker_stop_cleanup_round_trip_preserves_stop_outcome(
    private_result_root, monkeypatch, stop_state, terminal_state
):
    summarizer = _load_summarizer()
    import navigation_compare as navigation

    runtime_root = private_result_root / "runtime"
    worker_identity = {
        "pid": 1000,
        "created": 1000.0,
        "cmdline": ["worker"],
        "owner_token": "a" * 32,
        "owner_root": str(runtime_root / "openbrowser"),
    }
    child = {**worker_identity, "pid": 1001, "cmdline": ["chrome"]}
    progress_child = {**child, "pid": 1002}
    stop_result = {"identities": [{**child, "state": stop_state}], "errors": []}

    class FakeArm:
        def __init__(self, _args):
            pass

        async def stop(self):
            return stop_result

    stdout = io.StringIO()
    with monkeypatch.context() as protocol:
        protocol.setattr(navigation, "Arm", FakeArm)
        protocol.setattr(navigation.sys, "stdin", io.StringIO('{"op":"stop"}\n'))
        protocol.setattr(navigation.sys, "stdout", stdout)
        await navigation.worker(SimpleNamespace(worker_mode="navigation"))
    reply = json.loads(stdout.getvalue().removeprefix(navigation.PREFIX))
    stop_ok = stop_state in {"absent", "reused"}
    assert reply["ok"] is stop_ok
    assert reply["result"] == stop_result

    class Process:
        stdin = io.StringIO()
        returncode = 0

        async def wait(self):
            return self.returncode

    async def send(_process, request, _log, timeout):
        assert request == {"op": "stop"}
        assert timeout == 60
        return reply

    terminated = False

    def identity_state(identity):
        # Earlier progress may still need fallback even after a successful stop.
        if identity["pid"] == progress_child["pid"] and not terminated:
            return "alive"
        return terminal_state

    async def terminate(identities, token):
        nonlocal terminated
        assert identities == [worker_identity, progress_child, child]
        assert token == worker_identity["owner_token"]
        terminated = True

    monkeypatch.setattr(navigation, "send", send)
    monkeypatch.setattr(navigation, "identity_state", identity_state)
    monkeypatch.setattr(navigation, "descendant_identities", lambda *_args: [])
    monkeypatch.setattr(navigation, "terminate_owned", terminate)
    handle = navigation.WorkerHandle(
        process=Process(),
        stdout_log=io.StringIO(),
        stderr_log=io.StringIO(),
        worker_identity=worker_identity,
        initial_owned=(progress_child,),
        owner_token=worker_identity["owner_token"],
        mode=navigation.WorkerMode.NAVIGATION,
        state=navigation.WorkerState.READY,
    )
    cleanup = await navigation.cleanup_worker(handle)
    assert terminated
    assert cleanup["fallback_used"] is True
    assert cleanup["ok"] is stop_ok
    assert all(
        identity["state"] == terminal_state
        for identity in cleanup["post_worker_exit_identities"]
    )
    assert (
        summarizer._validate_worker_cleanup(
            cleanup, "openbrowser", runtime_root, "worker", set(), require_ok=stop_ok
        )
        is stop_ok
    )

    if not stop_ok:
        assert cleanup["error"] == "worker process cleanup is incomplete"
        with pytest.raises(
            summarizer.VerificationError, match="cleanup is not verified"
        ):
            summarizer._validate_worker_cleanup(
                cleanup, "openbrowser", runtime_root, "worker", set(), require_ok=True
            )
        missing_stop_error = dict(cleanup)
        del missing_stop_error["error"]
        with pytest.raises(summarizer.VerificationError, match="retirement proof"):
            summarizer._validate_worker_cleanup(
                missing_stop_error,
                "openbrowser",
                runtime_root,
                "worker",
                set(),
                require_ok=False,
            )
        forged_success = {**missing_stop_error, "ok": True}
        with pytest.raises(
            summarizer.VerificationError,
            match="successful cleanup has nonretired worker stop identities",
        ):
            summarizer._validate_worker_cleanup(
                forged_success,
                "openbrowser",
                runtime_root,
                "worker",
                set(),
                require_ok=False,
            )


def test_summary_rejects_successful_cleanup_with_worker_stop_errors(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )

    def add_worker_stop_error(attempt):
        worker = attempt["cleanup"]["workers"]["openbrowser"]
        worker["result"]["errors"] = ["worker stop verification failed"]

    _rewrite_first_attempt(run_root, add_worker_stop_error)

    assert summarizer.dispatch(args) == 1
    assert "successful cleanup has worker stop errors" in caplog.text


@pytest.mark.parametrize("error_source", ["worker_stop", "coordinator"])
def test_cleanup_preserves_retired_processes_with_failed_stop_evidence(
    private_result_root, error_source
):
    summarizer = _load_summarizer()
    runtime_root = private_result_root / "runtime"
    identity = {
        "pid": 1000,
        "created": 1000.0,
        "cmdline": ["worker"],
        "owner_token": "a" * 32,
        "owner_root": str(runtime_root / "openbrowser"),
        "state": "absent",
    }
    worker = {
        "ok": False,
        "fallback_used": False,
        "worker_identity": dict(identity),
        "post_worker_exit_identities": [dict(identity)],
        "result": {
            "identities": [dict(identity)],
            "errors": ["stop failed"] if error_source == "worker_stop" else [],
        },
        "error": "worker stop: stop failed"
        if error_source == "worker_stop"
        else "cannot close worker input",
    }

    assert (
        summarizer._validate_worker_cleanup(
            worker, "openbrowser", runtime_root, "worker", set(), require_ok=False
        )
        is False
    )
    with pytest.raises(summarizer.VerificationError, match="cleanup is not verified"):
        summarizer._validate_worker_cleanup(
            worker, "openbrowser", runtime_root, "worker", set(), require_ok=True
        )


@pytest.mark.parametrize(
    ("ok", "error"),
    [
        (True, None),
        (True, "coordinator failed"),
        (False, None),
        (False, ""),
        (False, "   "),
    ],
)
def test_cleanup_rejects_unknown_fallback_without_failed_coordinator_evidence(
    private_result_root, ok, error
):
    summarizer = _load_summarizer()
    runtime_root = private_result_root / "runtime"
    worker = {
        "ok": ok,
        "fallback_used": None,
        "worker_identity": {
            "pid": 1000,
            "created": 1000.0,
            "cmdline": ["worker"],
            "owner_token": "a" * 32,
            "owner_root": str(runtime_root / "openbrowser"),
            "state": "absent",
        },
        "post_worker_exit_identities": [],
        "result": {"identities": [], "errors": []},
    }
    if error is not None:
        worker["error"] = error

    with pytest.raises(
        summarizer.VerificationError,
        match="fallback_used may be unknown only for failed cleanup",
    ):
        summarizer._validate_worker_cleanup(
            worker, "openbrowser", runtime_root, "worker", set(), require_ok=False
        )


def test_summary_preserves_but_never_ranks_a_crashed_in_flight_sample(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    pair = run_manifest["pairs"][0]
    reference = pair["attempts"][0]
    attempt_path = run_root / reference["directory"] / "attempt-manifest.json"
    attempt = json.loads(attempt_path.read_text())
    attempt["state"] = "running"
    attempt["samples"].pop()
    attempt["work_counts"] = {
        "attempted_work_count": 24,
        "successful_work_count": 23,
        "error_count": 0,
    }
    attempt["cleanup"].update(
        ok=False,
        workers_ok=False,
        runtime_root_removed=False,
        workers={},
    )
    raw = _write_json(attempt_path, attempt)
    reference.update(
        state="running",
        manifest_sha256=_sha(raw),
        cleanup_ok=False,
        runtime_root_removed=False,
        work_counts=attempt["work_counts"],
        error_count=0,
        errors=[],
    )
    pair["state"] = "running"
    run_manifest.update(
        state="running",
        work_counts={
            "attempted_work_count": 24,
            "successful_work_count": 23,
            "error_count": 0,
            "host_contaminated_attempt_count": 0,
        },
        fixture={
            "started": True,
            "stopped": False,
            "socket_closed": False,
            "thread_stopped": False,
        },
        cleanup={
            "ok": False,
            "workers_ok": False,
            "runtime_roots_removed": False,
            "fixture_stopped": False,
        },
    )
    _write_json(run_manifest_path, run_manifest)

    exit_code = summarizer.dispatch(args)
    summary = json.loads((run_root / "summary.json").read_text())
    pair_result = summary["pair_results"]["round-1/raw-cdp"]

    assert exit_code == 0, caplog.text
    assert "ranking" not in summary
    assert pair_result["state"] == "unverified"
    assert pair_result["manifest_state"] == "running"
    assert pair_result["attempted_work_count"] == 24
    assert pair_result["successful_work_count"] == 23
    assert pair_result["metrics"] is None


def test_summary_does_not_read_an_attempt_manifest_with_a_null_pending_hash(
    private_result_root,
    caplog,
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    pair = run_manifest["pairs"][0]
    reference = pair["attempts"][0]
    attempt_path = run_root / reference["directory"] / "attempt-manifest.json"
    attempt_path.write_text("{ malformed child manifest }")
    reference.update(
        state="pending",
        manifest_sha256=None,
        cleanup_ok=False,
        runtime_root_removed=False,
        host_gate={"state": "unknown", "reasons": ["not started"]},
        work_counts={
            "attempted_work_count": 0,
            "successful_work_count": 0,
            "error_count": 0,
        },
        error_count=0,
        errors=[],
    )
    pair["state"] = "running"
    run_manifest.update(
        state="running",
        work_counts={
            "attempted_work_count": 0,
            "successful_work_count": 0,
            "error_count": 0,
            "host_contaminated_attempt_count": 1,
        },
        fixture={
            "started": True,
            "stopped": False,
            "socket_closed": False,
            "thread_stopped": False,
        },
        cleanup={
            "ok": False,
            "workers_ok": False,
            "runtime_roots_removed": False,
            "fixture_stopped": False,
        },
    )
    _write_json(run_manifest_path, run_manifest)

    exit_code = summarizer.dispatch(args)
    summary = json.loads((run_root / "summary.json").read_text())
    pair_result = summary["pair_results"]["round-1/raw-cdp"]

    assert exit_code == 0, caplog.text
    assert "ranking" not in summary
    assert pair_result["state"] == "unverified"
    assert pair_result["manifest_state"] == "running"
    assert pair_result["attempt_count"] == 1
    assert pair_result["attempted_work_count"] == 0
    assert pair_result["metrics"] is None


def test_summary_verifies_a_pending_reference_when_its_child_manifest_is_hashed(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    pair = run_manifest["pairs"][0]
    reference = pair["attempts"][0]
    attempt_path = run_root / reference["directory"] / "attempt-manifest.json"
    attempt = json.loads(attempt_path.read_text())
    attempt.update(
        state="pending",
        execution_order=[],
        warmups=[],
        samples=[],
        host_samples=[],
        host_gate={"state": "unknown", "reasons": ["not started"]},
        work_counts={
            "attempted_work_count": 0,
            "successful_work_count": 0,
            "error_count": 0,
        },
        errors=[],
    )
    attempt["cleanup"].update(
        ok=False, workers_ok=False, runtime_root_removed=False, workers={}
    )
    raw = _write_json(attempt_path, attempt)
    reference.update(
        state="pending",
        manifest_sha256=_sha(raw),
        cleanup_ok=False,
        runtime_root_removed=False,
        host_gate=attempt["host_gate"],
        work_counts=attempt["work_counts"],
        error_count=0,
        errors=[],
    )
    pair["state"] = "running"
    run_manifest.update(
        state="running",
        work_counts={
            "attempted_work_count": 0,
            "successful_work_count": 0,
            "error_count": 0,
            "host_contaminated_attempt_count": 1,
        },
        fixture={
            "started": True,
            "stopped": False,
            "socket_closed": False,
            "thread_stopped": False,
        },
        cleanup={
            "ok": False,
            "workers_ok": False,
            "runtime_roots_removed": False,
            "fixture_stopped": False,
        },
    )
    _write_json(run_manifest_path, run_manifest)

    exit_code = summarizer.dispatch(args)
    summary = json.loads((run_root / "summary.json").read_text())

    assert exit_code == 0, caplog.text
    assert "ranking" not in summary
    assert summary["pair_results"]["round-1/raw-cdp"]["state"] == "unverified"
    assert summary["pair_results"]["round-1/raw-cdp"]["manifest_state"] == "running"
    assert summary["pair_results"]["round-1/raw-cdp"]["metrics"] is None


def test_summary_rejects_a_null_manifest_hash_after_attempt_start(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["pairs"][0]["attempts"][0]["state"] = "running"
    run_manifest["pairs"][0]["attempts"][0]["manifest_sha256"] = None
    run_manifest["pairs"][0]["state"] = "running"
    run_manifest["state"] = "running"
    _write_json(run_manifest_path, run_manifest)

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert "pending or failed" in caplog.text


def test_summary_preserves_a_failed_reference_without_a_child_manifest(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, _project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    run_manifest_path = run_root / "run-manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    pair = run_manifest["pairs"][0]
    reference = pair["attempts"][0]
    (run_root / reference["directory"] / "attempt-manifest.json").unlink()
    failure = "attempt failed before writing its manifest"
    reference.update(
        state="failed",
        manifest_sha256=None,
        cleanup_ok=False,
        runtime_root_removed=False,
        host_gate={"state": "unknown", "reasons": ["not started"]},
        work_counts={
            "attempted_work_count": 0,
            "successful_work_count": 0,
            "error_count": 1,
        },
        error_count=1,
        errors=[failure],
    )
    pair.update(state="failed", errors=[failure])
    run_manifest.update(
        state="incomplete",
        work_counts={
            "attempted_work_count": 0,
            "successful_work_count": 0,
            "error_count": 1,
            "host_contaminated_attempt_count": 1,
        },
        cleanup={
            "ok": False,
            "workers_ok": False,
            "runtime_roots_removed": False,
            "fixture_stopped": True,
        },
    )
    _write_json(run_manifest_path, run_manifest)

    exit_code = summarizer.dispatch(args)
    summary = json.loads((run_root / "summary.json").read_text())
    pair_result = summary["pair_results"]["round-1/raw-cdp"]

    assert exit_code == 0, caplog.text
    assert "ranking" not in summary
    assert pair_result["state"] == "unverified"
    assert pair_result["manifest_state"] == "failed"
    assert pair_result["attempted_work_count"] == 0
    assert pair_result["error_count"] == 1
    assert pair_result["metrics"] is None


def _sanitized_directory(project_root):
    directory = project_root / SUMMARY_SPEC_ROOT
    directory.mkdir(mode=0o700, parents=True)
    directory.chmod(0o700)
    return directory


def test_summary_does_not_export_a_partial_run(private_result_root, caplog):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, project_root = _build_run(
        private_result_root, run_state="incomplete"
    )
    sanitized = _sanitized_directory(project_root) / "candidate-baseline-summary.md"
    args.extend(["--sanitized-output", str(sanitized)])

    exit_code = summarizer.dispatch(args)

    assert exit_code == 1
    assert not sanitized.exists()
    assert (run_root / "summary.json").is_file()
    assert "complete verified run" in caplog.text


def test_summary_sanitizes_a_complete_run_without_private_artifacts(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, _run_root, project_root = _build_run(private_result_root, all_pairs=True)
    sanitized = _sanitized_directory(project_root) / "current-benchmark-summary.md"
    sanitized.write_text("old summary")
    args.extend(["--sanitized-output", str(sanitized)])

    exit_code = summarizer.dispatch(args)
    report = sanitized.read_text()

    assert exit_code == 0, caplog.text
    assert "Local navigation benchmark" in report
    assert "0.1.56" in report
    assert "158.0.8127.36" in report
    assert "1.0.4" in report
    assert "0.43.0" in report
    assert "1.0.6" in report
    assert "0.1.13" in report
    assert "round-1" not in report
    assert "/private/tmp" not in report
    assert "owner_token" not in report
    assert "screenshot" not in report.casefold()
    assert "private-owner-token" not in report
    assert "a1a1a1a1" not in report
    assert sanitized.stat().st_mode & 0o777 == 0o600


def test_summary_rejects_an_unapproved_sanitized_destination(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, project_root = _build_run(private_result_root, all_pairs=True)
    wrong_path = project_root / "summary.md"
    args.extend(["--sanitized-output", str(wrong_path)])

    exit_code = summarizer.dispatch(args)

    assert exit_code == 2
    assert not wrong_path.exists()
    assert not (run_root / "summary.json").exists()


def test_candidate_baseline_sanitized_summary_cannot_be_replaced(
    private_result_root, caplog
):
    summarizer = _load_summarizer()
    caplog.set_level(logging.ERROR, logger="summarize-latest-navigation")
    args, run_root, project_root = _build_run(private_result_root, all_pairs=True)
    baseline = _sanitized_directory(project_root) / "candidate-baseline-summary.md"
    baseline.write_text("immutable")
    args.extend(["--sanitized-output", str(baseline)])

    exit_code = summarizer.dispatch(args)

    assert exit_code == 2
    assert baseline.read_text() == "immutable"
    assert not (run_root / "summary.json").exists()
