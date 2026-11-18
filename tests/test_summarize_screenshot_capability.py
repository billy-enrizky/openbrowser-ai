import hashlib
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

PROJECT_SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_SOURCE / "benchmarks" / "summarize_screenshot_capability.py"
sys.path.insert(0, str(PROJECT_SOURCE / "benchmarks"))

import latest_benchmark_preflight as preflight
import screenshot_capability as capability

INSTALLED_TOOL_KEYS = {"lightpanda", "agent-browser", "chrome_for_testing"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_json(path: Path, value: dict) -> None:
    capability.atomic_json(path, value)


def _make_pixel_contract_png(index: int) -> bytes:
    canvas = Image.new("RGB", (128, 128), (255, 255, 255))
    fixture = Image.open(BytesIO(capability.png_for_index(index))).convert("RGB")
    canvas.paste(fixture, (0, 0))
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def _make_attempt(run_root: Path, run_id: str, arm: str, index: int) -> dict:
    attempt_dir = run_root / "arms" / arm / f"attempt-{index + 1}"
    for directory in (run_root / "arms", run_root / "arms" / arm, attempt_dir):
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
    content = _make_pixel_contract_png(index)
    png_path = attempt_dir / "result.png"
    png_path.write_bytes(content)
    png_path.chmod(0o600)
    pixel = capability.inspect_png(content, index)
    request = capability.screenshot_request(run_id, arm, index, 12345)
    nonce = request["nonce"]
    runtime_root = Path("/tmp").resolve(strict=True) / (
        f"nb-screenshot-summary-{run_id}-{arm}-{index}"
    )
    worker_identity = {
        "pid": 41000 + len(arm) * 10 + index,
        "created": 1000.0 + index,
        "cmdline": [f"/tmp/{arm}-worker"],
        "owner_token": f"private-owner-token-{arm}-{index}",
        "owner_root": str(runtime_root / arm),
    }
    worker_cleanup = {
        "ok": True,
        "worker_identity": {**worker_identity, "state": "absent"},
        "post_worker_exit_identities": [],
        "errors": [],
    }
    if arm == "lightpanda":
        worker_cleanup.update(
            {"process_state": "verified", "process_group_id": worker_identity["pid"]}
        )
        logs = attempt_dir / "logs"
        logs.mkdir(mode=0o700)
        script = logs / "lightpanda.input.js"
        script.write_bytes(capability.render_lightpanda_script(request))
        script.chmod(0o600)

    attempt = {
        "attempt_id": f"{run_id}:{arm}:{index}",
        "arm": arm,
        "index": index,
        "nonce": nonce,
        "request": request,
        "directory": f"arms/{arm}/attempt-{index + 1}",
        "state": "passed",
        "parseable_png": True,
        "image_request_ok": True,
        "cleanup_ok": True,
        "runtime_root_removed": True,
        "runtime_root": str(runtime_root),
        "worker_identity": worker_identity,
        "worker_cleanup": worker_cleanup,
        "request_proof": {
            "nonce": nonce,
            "count": 1,
            "successful_count": 1,
            "ok": True,
            "image_requests": [
                {
                    "method": "GET",
                    "path": "/pixels.png",
                    "nonce": nonce,
                    "status": 200,
                }
            ],
        },
        "png": {
            "file": "result.png",
            "byte_count": len(content),
            "sha256": _sha256(png_path),
            "pixel": pixel,
        },
        "pixel": pixel,
        "mismatch_signature": None,
        "errors": [],
    }
    if arm in ("openbrowser", "raw-cdp"):
        attempt["worker_reply"] = {
            "schema": "openbrowser-screenshot-reply/1",
            "attempt_id": request["attempt_id"],
            "arm": arm,
            "nonce": nonce,
            "ready_marker": nonce,
            "output": "result.png",
            "byte_count": len(content),
            "sha256": _sha256(png_path),
        }
    return attempt


def _write_valid_run(
    project_root: Path, result_root: Path, *, run_name: str = "screenshot"
) -> dict:
    result_root.mkdir(mode=0o700, exist_ok=True)
    result_root.chmod(0o700)
    run_root = result_root / run_name
    run_root.mkdir(mode=0o700)
    run_root.chmod(0o700)
    run_id = "c" * 32

    executed_files = {}
    for relative in preflight.EXECUTED_FILE_PATHS:
        source_path = PROJECT_SOURCE / relative
        executed_files[relative] = (
            _sha256(source_path)
            if source_path.is_file()
            else hashlib.sha256(relative.encode()).hexdigest()
        )
    identity = {
        "schema": preflight.IDENTITY_SCHEMA,
        "candidate_commit": "a" * 40,
        "candidate_tree": "b" * 40,
        "runtime_tree": "c" * 40,
        "release_runtime_tree": "c" * 40,
        "origin_main_commit": "d" * 40,
        "origin_main_runtime_tree": "c" * 40,
        "executed_files": executed_files,
        "candidate_python": {
            "supplied_path": "/candidate/bin/python",
            "resolved_path": "/candidate/bin/python",
            "version": preflight.PYTHON_VERSION_REQUIRED,
            "executable_sha256": "e" * 64,
        },
        "browser_use_python": {
            "supplied_path": "/browser-use/bin/python",
            "resolved_path": "/browser-use/bin/python",
            "version": preflight.PYTHON_VERSION_REQUIRED,
            "executable_sha256": "f" * 64,
        },
    }
    identity_path = result_root / "candidate-identity.json"
    _private_json(identity_path, identity)
    identity_hash = _sha256(identity_path)
    candidate_freeze_path = result_root / "candidate-freeze.txt"
    candidate_freeze_path.write_text("pillow==12.0.0\n")
    candidate_freeze_path.chmod(0o600)
    browser_use_freeze_path = result_root / "browser-use-freeze.txt"
    browser_use_freeze_path.write_text("browser-use==1.0.6\n")
    browser_use_freeze_path.chmod(0o600)
    tool_manifest_path = result_root / "tool-manifest.json"
    _private_json(tool_manifest_path, {})
    tool_manifest_hash = _sha256(tool_manifest_path)
    versions = {
        "lightpanda": "1.0.4",
        "agent-browser": "0.43.0",
        "browser-use": "1.0.6",
        "browser-harness": "0.1.13",
        "chrome_for_testing": "157.0.8103.78",
    }
    version_manifest = {
        "schema": preflight.VERSION_MANIFEST_SCHEMA,
        "fetched_at_utc": "2026-11-17T00:00:00Z",
        "metadata_sources": {
            name: {"url": url, "response_sha256": "a" * 64}
            for name, url in preflight._metadata.SOURCE_URLS.items()
        },
        "latest_versions": versions,
        "installed_versions": {
            "lightpanda": versions["lightpanda"],
            "agent-browser": versions["agent-browser"],
            "chrome_for_testing": versions["chrome_for_testing"],
            "browser-use": versions["browser-use"],
            "browser-harness": versions["browser-harness"],
            "candidate:psutil": "7.1.3",
            "browser_use:psutil": "7.1.3",
            "candidate:websockets": "16.0",
            "browser_use:websockets": "16.0",
        },
        "interpreters": {
            "candidate": identity["candidate_python"],
            "browser_use": identity["browser_use_python"],
        },
        "dependency_freezes": {
            "candidate": {
                "path": str(candidate_freeze_path),
                "sha256": _sha256(candidate_freeze_path),
            },
            "browser_use": {
                "path": str(browser_use_freeze_path),
                "sha256": _sha256(browser_use_freeze_path),
            },
        },
        "tool_manifest_sha256": tool_manifest_hash,
        "candidate_identity_sha256": identity_hash,
        "executed_file_sha256": executed_files,
    }
    version_path = result_root / "version-manifest.json"
    _private_json(version_path, version_manifest)
    version_hash = _sha256(version_path)
    preflight_record = {
        "schema": preflight.PREFLIGHT_SCHEMA,
        "phase": "screenshot-final"
        if run_name == "screenshot-final"
        else "screenshot-run",
        "candidate_identity_sha256": identity_hash,
        "version_manifest_sha256": version_hash,
        "checks": [
            {
                "name": "latest_versions",
                "expected": versions,
                "actual": versions,
                "state": "passed",
            },
            {
                "name": "installed_tool_versions",
                "expected": {name: versions[name] for name in INSTALLED_TOOL_KEYS},
                "actual": {name: versions[name] for name in INSTALLED_TOOL_KEYS},
                "state": "passed",
            },
        ],
        "outcome": "passed",
        "errors": [],
    }
    preflight_path = result_root / "preflight-screenshot.json"
    _private_json(preflight_path, preflight_record)

    smoke_preflight_hashes = {}
    for phase in ("navigation-smoke", "screenshot-smoke"):
        smoke_preflight_path = result_root / f"preflight-{phase}.json"
        _private_json(
            smoke_preflight_path,
            {
                "schema": preflight.PREFLIGHT_SCHEMA,
                "phase": phase,
                "candidate_identity_sha256": identity_hash,
                "version_manifest_sha256": version_hash,
                "checks": [
                    {
                        "name": "smoke_fixture_check",
                        "expected": "passed",
                        "actual": "passed",
                        "state": "passed",
                    }
                ],
                "outcome": "passed",
                "errors": [],
            },
        )
        smoke_preflight_hashes[
            "navigation" if phase == "navigation-smoke" else "screenshot"
        ] = _sha256(smoke_preflight_path)

    frozen_record = {
        "schema": preflight.FROZEN_HARNESS_SCHEMA,
        "candidate_identity_sha256": identity_hash,
        "version_manifest_sha256": version_hash,
        "tool_manifest_sha256": tool_manifest_hash,
        "candidate_freeze_sha256": _sha256(candidate_freeze_path),
        "browser_use_freeze_sha256": _sha256(browser_use_freeze_path),
        "smoke_preflight_sha256": smoke_preflight_hashes,
        "smoke_manifest_sha256": {"navigation": "5" * 64, "screenshot": "6" * 64},
        "executed_file_sha256": executed_files,
    }
    frozen_path = result_root / "frozen-harness.json"
    _private_json(frozen_path, frozen_record)

    attempts = [
        _make_attempt(run_root, run_id, arm, index)
        for arm in capability.SCREENSHOT_ARMS
        for index in range(6)
    ]
    manifest = {
        "schema": capability.RUN_SCHEMA,
        "mode": "run",
        "run_id": run_id,
        "state": "complete",
        "candidate_identity_sha256": identity_hash,
        "version_manifest_sha256": version_hash,
        "preflight_sha256": _sha256(preflight_path),
        "frozen_harness_sha256": _sha256(frozen_path),
        "tool_manifest_sha256": tool_manifest_hash,
        "harness_sha256": capability.navigation.sha256(capability.__file__),
        "lightpanda_script_sha256": capability.navigation.sha256(
            Path(capability.__file__).with_name("screenshot_lightpanda.js")
        ),
        "arms": list(capability.SCREENSHOT_ARMS),
        "attempts": attempts,
        "arm_status": {arm: "passed" for arm in capability.SCREENSHOT_ARMS},
        "work_counts": capability.screenshot_work_counts(attempts),
        "fixture": {
            "started": True,
            "stopped": True,
            "socket_closed": True,
            "thread_stopped": True,
        },
        "cleanup": {"ok": True, "attempts_ok": True, "runtime_roots_removed": True},
        "errors": [],
    }
    _private_json(run_root / "run-manifest.json", manifest)
    capability.write_screenshot_summary(run_root, manifest)

    report_directory = (
        project_root / "local_docs" / "specs" / "2026-11-17-latest-navigation-benchmark"
    )
    report_directory.mkdir(mode=0o755, parents=True)
    report_directory.chmod(0o755)
    return {
        "run_root": run_root,
        "identity_path": identity_path,
        "preflight_path": preflight_path,
        "frozen_path": frozen_path,
        "report_directory": report_directory,
        "manifest": manifest,
        "identity": identity,
        "preflight": preflight_record,
        "frozen": frozen_record,
    }


def _verify_command(
    project_root: Path, artifacts: dict, report_path: Path
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "verify",
        "--run-root",
        str(artifacts["run_root"]),
        "--candidate-identity",
        str(artifacts["identity_path"]),
        "--preflight",
        str(artifacts["preflight_path"]),
        "--frozen-harness",
        str(artifacts["frozen_path"]),
        "--project-root",
        str(project_root),
        "--sanitized-output",
        str(report_path),
    ]


def _refresh_manifest(run_root: Path, manifest: dict) -> None:
    controls = {}
    for arm in ("openbrowser", "raw-cdp"):
        controls[arm] = capability.classify_arm_status(
            arm,
            [attempt for attempt in manifest["attempts"] if attempt["arm"] == arm],
            controls,
            fixture_stopped=True,
        )
    controls["lightpanda"] = capability.classify_arm_status(
        "lightpanda",
        [attempt for attempt in manifest["attempts"] if attempt["arm"] == "lightpanda"],
        controls,
        fixture_stopped=True,
    )
    manifest["arm_status"] = controls
    manifest["work_counts"] = capability.screenshot_work_counts(manifest["attempts"])
    _private_json(run_root / "run-manifest.json", manifest)
    capability.write_screenshot_summary(run_root, manifest)


def _set_lightpanda_stable_dimension_mismatch(run_root: Path, manifest: dict) -> None:
    for attempt in manifest["attempts"]:
        if attempt["arm"] != "lightpanda":
            continue
        content = capability.png_for_index(attempt["index"])
        png_path = run_root / attempt["directory"] / "result.png"
        png_path.write_bytes(content)
        png_path.chmod(0o600)
        pixel = capability.inspect_png(content, attempt["index"])
        attempt["state"] = "pixel_mismatch"
        attempt["parseable_png"] = True
        attempt["pixel"] = pixel
        attempt["mismatch_signature"] = pixel["mismatch_signature"]
        attempt["png"] = {
            "file": "result.png",
            "byte_count": len(content),
            "sha256": _sha256(png_path),
            "pixel": pixel,
        }


def _restore_lightpanda_passed_attempt(run_root: Path, attempt: dict) -> None:
    content = _make_pixel_contract_png(attempt["index"])
    png_path = run_root / attempt["directory"] / "result.png"
    png_path.write_bytes(content)
    png_path.chmod(0o600)
    pixel = capability.inspect_png(content, attempt["index"])
    attempt["state"] = "passed"
    attempt["parseable_png"] = True
    attempt["pixel"] = pixel
    attempt["mismatch_signature"] = None
    attempt["png"] = {
        "file": "result.png",
        "byte_count": len(content),
        "sha256": _sha256(png_path),
        "pixel": pixel,
    }


def _set_attempt_failed(
    attempt: dict, *, remove_png: bool = False, run_root: Path | None = None
) -> None:
    attempt["state"] = "failed"
    attempt["errors"] = ["simulated adapter failure after verified cleanup"]
    attempt["worker_reply"] = None
    if remove_png:
        assert run_root is not None
        (run_root / attempt["directory"] / "result.png").unlink()
        attempt["parseable_png"] = False
        attempt["png"] = None
        attempt["pixel"] = {
            "parseable": False,
            "probe_count": 0,
            "mismatch_count": 0,
            "mismatch_signature": None,
        }
        attempt["mismatch_signature"] = None


def test_verified_screenshot_run_exports_a_sanitized_capability_report(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        report = report_path.read_text()
        assert "Lightpanda" in report
        assert "OpenBrowser runtime: `0.1.56`" in report
        assert "passed" in report
        assert str(result_root) not in report
        assert "private-owner-token" not in report
        assert "owner_token" not in report
        for private_detail in (
            "worker_identity",
            "pid",
            "cmdline",
            "runtime_root",
            "AWS_",
            "NAVIGATION_BENCHMARK_OWNER",
            "worker.stdout.log",
            "worker.stderr.log",
        ):
            assert private_detail not in report
        assert "elapsed_ns" not in report
        assert "navigation_ns" not in report
        assert "duration_ns" not in report


def test_screenshot_report_rejects_changed_native_script_input(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, Path(temp))
        script = (
            artifacts["run_root"] / "arms/lightpanda/attempt-1/logs/lightpanda.input.js"
        )
        script.write_text('throw new Error("different program");\n')
        report = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report.exists()


def test_screenshot_report_requires_a_complete_verified_run(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )
        manifest = artifacts["manifest"]
        manifest["attempts"].pop()
        _refresh_manifest(artifacts["run_root"], manifest)

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_a_private_output_path(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        report_path = result_root / "private-report.md"
        command = _verify_command(project_root, artifacts, report_path)

        result = subprocess.run(command, capture_output=True, text=True, check=False)

        assert result.returncode == 2
        assert not report_path.exists()


def test_screenshot_report_rejects_parent_traversal_in_the_export_path(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        traversal = (
            artifacts["report_directory"]
            / "existing"
            / ".."
            / ("candidate-baseline-screenshot-capability.md")
        )
        (artifacts["report_directory"] / "existing").mkdir()

        result = subprocess.run(
            _verify_command(project_root, artifacts, traversal),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert not (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        ).exists()


def test_candidate_baseline_report_is_never_replaced(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )
        report_path.write_text("preserve baseline\n")

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert report_path.read_text() == "preserve baseline\n"


def test_screenshot_report_contains_no_latency_fields(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        report = report_path.read_text()
        assert "latency" not in report.lower()
        for field in (
            "elapsed_ns",
            "navigation_ns",
            "post_navigation_ns",
            "duration_ns",
        ):
            assert field not in report


def test_valid_unsupported_screenshot_capability_remains_reportable(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        _set_lightpanda_stable_dimension_mismatch(artifacts["run_root"], manifest)
        _refresh_manifest(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        report = report_path.read_text()
        assert "| lightpanda | unsupported | 6 | 0 | 6 | 0 | 24 | 0 |" in report
        assert (
            "does not establish that Lightpanda cannot take screenshots generally"
            in report
        )


@pytest.mark.parametrize(
    ("outcome", "expected_statuses"),
    [
        (
            "inconclusive",
            {
                "openbrowser": "failed",
                "raw-cdp": "passed",
                "lightpanda": "inconclusive",
            },
        ),
        (
            "failed",
            {"openbrowser": "passed", "raw-cdp": "passed", "lightpanda": "failed"},
        ),
    ],
)
def test_valid_nonpassing_capability_statuses_remain_reportable(
    tmp_path,
    outcome,
    expected_statuses,
):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / f"project-{outcome}"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        target = next(
            attempt
            for attempt in manifest["attempts"]
            if (
                outcome == "inconclusive"
                and attempt["arm"] == "openbrowser"
                and attempt["index"] == 0
            )
            or (
                outcome == "failed"
                and attempt["arm"] == "lightpanda"
                and attempt["index"] == 0
            )
        )
        _set_attempt_failed(
            target,
            remove_png=(outcome == "failed"),
            run_root=artifacts["run_root"],
        )
        _refresh_manifest(artifacts["run_root"], manifest)
        assert manifest["arm_status"] == expected_statuses
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        report = report_path.read_text()
        assert f"| lightpanda | {outcome} |" in report


def test_screenshot_report_rejects_status_precedence_tampering(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        _set_lightpanda_stable_dimension_mismatch(artifacts["run_root"], manifest)
        manifest["arm_status"]["lightpanda"] = "passed"
        manifest["work_counts"] = capability.screenshot_work_counts(
            manifest["attempts"]
        )
        _private_json(artifacts["run_root"] / "run-manifest.json", manifest)
        capability.write_screenshot_summary(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_valid_mixed_lightpanda_pixel_results_are_inconclusive(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        _set_lightpanda_stable_dimension_mismatch(artifacts["run_root"], manifest)
        mixed_attempt = next(
            attempt
            for attempt in manifest["attempts"]
            if attempt["arm"] == "lightpanda" and attempt["index"] == 0
        )
        _restore_lightpanda_passed_attempt(artifacts["run_root"], mixed_attempt)
        _refresh_manifest(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert (
            "| lightpanda | inconclusive | 6 | 1 | 5 | 0 | 24 | 0 |"
            in report_path.read_text()
        )


def test_screenshot_report_rejects_boolean_work_counts(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        manifest["work_counts"]["error_count"] = False
        _private_json(artifacts["run_root"] / "run-manifest.json", manifest)
        capability.write_screenshot_summary(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


@pytest.mark.parametrize("changed_artifact", ("candidate", "preflight"))
def test_screenshot_report_rejects_candidate_or_preflight_identity_mismatch(
    tmp_path,
    changed_artifact,
):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        if changed_artifact == "candidate":
            identity = artifacts["identity"]
            identity["candidate_commit"] = "f" * 40
            _private_json(artifacts["identity_path"], identity)
        else:
            preflight_record = artifacts["preflight"]
            preflight_record["version_manifest_sha256"] = "9" * 64
            _private_json(artifacts["preflight_path"], preflight_record)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_malformed_request_proof_events(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        manifest["attempts"][0]["request_proof"]["image_requests"][0]["extra"] = (
            "unrecognized"
        )
        _private_json(artifacts["run_root"] / "run-manifest.json", manifest)
        capability.write_screenshot_summary(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_tampered_pixel_probe_evidence(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        manifest["attempts"][0]["pixel"]["probes"][0]["actual_rgb"] = [1, 2, 3]
        _private_json(artifacts["run_root"] / "run-manifest.json", manifest)
        capability.write_screenshot_summary(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_an_unrecorded_attempt_image(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        attempt = next(
            record
            for record in manifest["attempts"]
            if record["arm"] == "lightpanda" and record["index"] == 0
        )
        attempt["state"] = "failed"
        attempt["errors"] = ["simulated output verification failure"]
        attempt["parseable_png"] = False
        attempt["png"] = None
        attempt["pixel"] = {
            "parseable": False,
            "probe_count": 0,
            "mismatch_count": 0,
            "mismatch_signature": None,
        }
        attempt["mismatch_signature"] = None
        _refresh_manifest(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_float_pixel_probe_counts(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        manifest["attempts"][0]["pixel"]["probe_count"] = 4.0
        _private_json(artifacts["run_root"] / "run-manifest.json", manifest)
        capability.write_screenshot_summary(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_a_tampered_png_hash(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        png_path = (
            artifacts["run_root"]
            / artifacts["manifest"]["attempts"][0]["directory"]
            / "result.png"
        )
        png_path.write_bytes(png_path.read_bytes() + b"tampered")
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_unretired_worker_cleanup(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        manifest = artifacts["manifest"]
        worker_cleanup = manifest["attempts"][0]["worker_cleanup"]
        worker_cleanup["worker_identity"]["state"] = "alive"
        _private_json(artifacts["run_root"] / "run-manifest.json", manifest)
        capability.write_screenshot_summary(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_screenshot_report_rejects_frozen_harness_identity_mismatch(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(project_root, result_root)
        frozen = artifacts["frozen"]
        frozen["tool_manifest_sha256"] = "9" * 64
        _private_json(artifacts["frozen_path"], frozen)
        manifest = artifacts["manifest"]
        manifest["frozen_harness_sha256"] = _sha256(artifacts["frozen_path"])
        _private_json(artifacts["run_root"] / "run-manifest.json", manifest)
        capability.write_screenshot_summary(artifacts["run_root"], manifest)
        report_path = (
            artifacts["report_directory"]
            / "candidate-baseline-screenshot-capability.md"
        )

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert not report_path.exists()


def test_current_report_is_unchanged_when_final_run_verification_fails(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(
            project_root, result_root, run_name="screenshot-final"
        )
        report_path = artifacts["report_directory"] / "current-screenshot-capability.md"
        report_path.write_text("existing verified report\n")
        manifest = artifacts["manifest"]
        manifest["attempts"].pop()
        _refresh_manifest(artifacts["run_root"], manifest)

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert report_path.read_text() == "existing verified report\n"


def test_current_report_is_replaced_only_after_a_valid_final_run(tmp_path):
    with tempfile.TemporaryDirectory(dir=Path("/tmp").resolve()) as temp:
        result_root = Path(temp)
        project_root = tmp_path / "project"
        project_root.mkdir()
        artifacts = _write_valid_run(
            project_root, result_root, run_name="screenshot-final"
        )
        report_path = artifacts["report_directory"] / "current-screenshot-capability.md"
        report_path.write_text("old report\n")

        result = subprocess.run(
            _verify_command(project_root, artifacts, report_path),
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert report_path.read_text() != "old report\n"
        assert report_path.read_text().startswith("# Screenshot capability report\n")
