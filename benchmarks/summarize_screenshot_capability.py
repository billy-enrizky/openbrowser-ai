from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import uuid


CLI_DESCRIPTION = 'Verify a completed screenshot run and export only its sanitized capability result.'
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import latest_benchmark_preflight as preflight  # noqa: E402
import screenshot_capability as capability  # noqa: E402


LOG = logging.getLogger("summarize-screenshot-capability")
REPORT_DIRECTORY = Path("local_docs/specs/2026-11-17-latest-navigation-benchmark")
BASELINE_REPORT = "candidate-baseline-screenshot-capability.md"
CURRENT_REPORT = "current-screenshot-capability.md"
WORK_COUNT_FIELDS = {
    "attempted_work_count",
    "completed_work_count",
    "successful_work_count",
    "pixel_mismatch_attempt_count",
    "pixel_probe_count",
    "pixel_probe_mismatch_count",
    "error_count",
}
PROOF_KEYS = {"nonce", "image_requests", "count", "successful_count", "ok"}
REQUEST_EVENT_KEYS = {"method", "path", "nonce", "status"}
LATEST_VERSION_KEYS = {
    "lightpanda", "agent-browser", "browser-use", "browser-harness", "chrome_for_testing",
}
INSTALLED_TOOL_KEYS = {"lightpanda", "agent-browser", "chrome_for_testing"}


class InputPathError(ValueError):
    pass


class ArtifactError(ValueError):
    pass


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    commands = root.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="verify and export a completed screenshot run")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--candidate-identity", required=True)
    verify.add_argument("--preflight", required=True)
    verify.add_argument("--frozen-harness", required=True)
    verify.add_argument("--project-root", required=True)
    verify.add_argument("--sanitized-output", required=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_artifact_path(value: str, result_root: Path, name: str) -> Path:
    try:
        physical_root, path = capability.navigation.physical_profile_path(value)
    except (OSError, ValueError) as error:
        raise InputPathError(f"{name} path is invalid: {error}") from error
    if physical_root != result_root.parent or path == result_root \
            or not path.is_relative_to(result_root):
        raise InputPathError(f"{name} must be below RESULT_ROOT")
    return path


def _private_json(result_root: Path, value: str | Path, name: str) -> tuple[Path, dict]:
    try:
        return capability.read_json_artifact(result_root, value, name)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"{name} is invalid: {error}") from error


def _project_directory(value: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise InputPathError("--project-root must be an absolute directory")
    try:
        root = raw.resolve(strict=True)
    except OSError as error:
        raise InputPathError(f"--project-root does not exist: {error}") from error
    if not root.is_dir():
        raise InputPathError("--project-root must be a directory")
    return root


def _require_real_directory(root: Path, relative: Path) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise InputPathError(f"sanitized report directory is unavailable: {error}") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise InputPathError("sanitized report directory must contain only real directories")
    return current


def _export_target(project_value: str, output_value: str, run_root: Path) -> tuple[Path, bool]:
    project_root = _project_directory(project_value)
    report_directory = _require_real_directory(project_root, REPORT_DIRECTORY)
    expected_name = CURRENT_REPORT if run_root.name == "screenshot-final" else BASELINE_REPORT
    expected = report_directory / expected_name

    raw = Path(output_value).expanduser()
    if not raw.is_absolute() or ".." in raw.parts:
        raise InputPathError("--sanitized-output must be absolute and contain no parent traversal")
    if raw.is_symlink():
        raise InputPathError("sanitized output must not be a symlink")
    try:
        resolved_parent = raw.parent.resolve(strict=True)
        resolved_output = raw.resolve(strict=False)
    except OSError as error:
        raise InputPathError(f"sanitized output path is invalid: {error}") from error
    if Path(os.path.abspath(raw.parent)) != report_directory \
            or resolved_parent != report_directory or resolved_output != expected \
            or raw.name != expected_name:
        raise InputPathError("sanitized output must use the approved screenshot report path")

    if expected.exists() or expected.is_symlink():
        try:
            info = expected.lstat()
        except OSError as error:
            raise InputPathError(f"sanitized output cannot be inspected: {error}") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise InputPathError("sanitized output must be a regular file")
        if expected_name == BASELINE_REPORT:
            raise InputPathError("refusing to replace an existing candidate baseline report")
    return expected, expected_name == CURRENT_REPORT


def _check_version_record(preflight_record: dict, name: str, keys: set[str]) -> dict[str, str]:
    matches = [
        check for check in preflight_record.get("checks", [])
        if isinstance(check, dict) and check.get("name") == name
    ]
    if len(matches) != 1:
        raise ArtifactError(f"preflight must contain exactly one {name} check")
    record = matches[0]
    expected = record.get("expected")
    actual = record.get("actual")
    if record.get("state") != "passed" or not isinstance(expected, dict) \
            or not isinstance(actual, dict) or expected != actual or set(actual) != keys:
        raise ArtifactError(f"preflight {name} check is incomplete or inconsistent")
    versions: dict[str, str] = {}
    for tool, version in actual.items():
        pattern = preflight.CHROME_VERSION_RE if tool == "chrome_for_testing" else preflight.VERSION_VALUE_RE
        if not isinstance(version, str) or not pattern.fullmatch(version):
            raise ArtifactError(f"preflight {name} contains an invalid {tool} version")
        versions[tool] = version
    return versions


def _json_exact_equal(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_exact_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _validate_attempt_proofs(manifest: dict) -> list[str]:
    errors: list[str] = []
    run_id = manifest.get("run_id")
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != len(capability.SCREENSHOT_ARMS) * 6:
        return ["complete screenshot run does not contain 18 attempts"]

    expected_keys = [
        (arm, index)
        for arm in capability.SCREENSHOT_ARMS
        for index in range(6)
    ]
    actual_keys = [
        (attempt.get("arm"), attempt.get("index"))
        for attempt in attempts if isinstance(attempt, dict)
    ]
    if actual_keys != expected_keys:
        errors.append("screenshot attempts are not the complete ordered arm/index set")

    for attempt in attempts:
        if not isinstance(attempt, dict):
            errors.append("screenshot attempt is not an object")
            continue
        arm = attempt.get("arm")
        index = attempt.get("index")
        if arm not in capability.SCREENSHOT_ARMS or type(index) is not int or not 0 <= index < 6:
            errors.append("screenshot attempt arm or index is invalid")
            continue
        nonce = f"{run_id}-{arm}-{index}"
        if attempt.get("attempt_id") != f"{run_id}:{arm}:{index}" \
                or attempt.get("nonce") != nonce:
            errors.append(f"{arm} attempt {index} identity does not match the run")

        for field in ("parseable_png", "image_request_ok", "cleanup_ok", "runtime_root_removed"):
            if type(attempt.get(field)) is not bool:
                errors.append(f"{arm} attempt {index} has an invalid {field} flag")

        attempt_errors = attempt.get("errors")
        if not isinstance(attempt_errors, list) or any(not isinstance(item, str) for item in attempt_errors):
            errors.append(f"{arm} attempt {index} errors must be a string list")
        elif attempt.get("state") in ("passed", "pixel_mismatch") and attempt_errors:
            errors.append(f"{arm} attempt {index} has errors despite a completed pixel result")
        elif attempt.get("state") == "failed" and not attempt_errors:
            errors.append(f"{arm} failed attempt {index} has no execution error")

        proof = attempt.get("request_proof")
        if not isinstance(proof, dict) or set(proof) != PROOF_KEYS:
            errors.append(f"{arm} attempt {index} request proof has an invalid schema")
            continue
        image_requests = proof.get("image_requests")
        if not isinstance(image_requests, list) or proof.get("nonce") != nonce \
                or type(proof.get("count")) is not int or proof["count"] != len(image_requests) \
                or type(proof.get("successful_count")) is not int \
                or type(proof.get("ok")) is not bool:
            errors.append(f"{arm} attempt {index} request proof counts or identity are invalid")
            continue
        if any(not isinstance(event, dict) or set(event) != REQUEST_EVENT_KEYS
               for event in image_requests):
            errors.append(f"{arm} attempt {index} request events have an invalid schema")
            continue
        for event in image_requests:
            if not isinstance(event.get("method"), str) or not isinstance(event.get("path"), str) \
                    or not isinstance(event.get("nonce"), str) or type(event.get("status")) is not int:
                errors.append(f"{arm} attempt {index} has a malformed image request event")
                break
        else:
            successful_count = sum(
                event["method"] == "GET" and event["path"] == "/pixels.png"
                and event["nonce"] == nonce and event["status"] == 200
                for event in image_requests
            )
            proof_ok = len(image_requests) == 1 and successful_count == 1
            if proof["successful_count"] != successful_count or proof["ok"] is not proof_ok \
                    or attempt.get("image_request_ok") is not proof_ok:
                errors.append(f"{arm} attempt {index} request proof does not match its events")
            if attempt.get("state") in ("passed", "pixel_mismatch") and not proof_ok:
                errors.append(f"{arm} attempt {index} completed without one successful image request")

    return errors


def _validate_attempt_pixels(run_root: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list):
        return ["screenshot attempts are not a list"]
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        directory = attempt.get("directory")
        index = attempt.get("index")
        if not isinstance(directory, str) or type(index) is not int:
            continue
        attempt_dir = run_root / directory
        output = attempt_dir / "result.png"
        try:
            unexpected_files = {
                entry.name for entry in attempt_dir.iterdir()
                if entry.name not in {"result.png", "logs"}
            }
        except OSError as error:
            errors.append(f"screenshot attempt directory cannot be listed: {error}")
            continue
        if unexpected_files:
            errors.append("screenshot attempt directory contains unrecognized files")

        if attempt.get("png") is None:
            default_pixel = {
                "parseable": False,
                "probe_count": 0,
                "mismatch_count": 0,
                "mismatch_signature": None,
            }
            if output.exists() or output.is_symlink():
                errors.append("unverified screenshot output exists without a PNG record")
            if attempt.get("parseable_png") is not False \
                    or not _json_exact_equal(attempt.get("pixel"), default_pixel):
                errors.append("attempt without a PNG has inconsistent pixel evidence")
            continue
        try:
            content = output.read_bytes()
            pixel = capability.inspect_png(content, index)
            expected_record = {
                "file": "result.png",
                "byte_count": len(content),
                "sha256": _sha256(output),
                "pixel": pixel,
            }
        except (OSError, ValueError) as error:
            errors.append(f"screenshot PNG cannot be re-read: {error}")
            continue
        if not _json_exact_equal(attempt.get("png"), expected_record):
            errors.append("screenshot PNG byte count, hash, or pixel record has an invalid type or value")
        if not _json_exact_equal(attempt.get("pixel"), pixel):
            errors.append("screenshot pixel dimensions or four-probe evidence has an invalid type or value")
    return errors


def _validate_manifest_counts_and_status(manifest: dict) -> list[str]:
    errors: list[str] = []
    counts = manifest.get("work_counts")
    if not isinstance(counts, dict) or set(counts) != WORK_COUNT_FIELDS \
            or any(type(value) is not int or value < 0 for value in counts.values()):
        return ["screenshot work counts have an invalid schema or value"]
    expected_counts = capability.screenshot_work_counts(manifest.get("attempts"))
    if counts != expected_counts:
        errors.append("screenshot work counts do not match the attempt evidence")

    fixture = manifest.get("fixture")
    fixture_stopped = isinstance(fixture, dict) and fixture.get("stopped") is True \
        and fixture.get("socket_closed") is True and fixture.get("thread_stopped") is True
    controls: dict[str, str] = {}
    attempts = manifest.get("attempts", [])
    for arm in ("openbrowser", "raw-cdp"):
        arm_attempts = [attempt for attempt in attempts if isinstance(attempt, dict) and attempt.get("arm") == arm]
        controls[arm] = capability.classify_arm_status(
            arm, arm_attempts, controls,
            fixture_stopped=fixture_stopped, expected_attempts=6,
        )
    lightpanda_attempts = [
        attempt for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("arm") == "lightpanda"
    ]
    controls["lightpanda"] = capability.classify_arm_status(
        "lightpanda", lightpanda_attempts, controls,
        fixture_stopped=fixture_stopped, expected_attempts=6,
    )
    if manifest.get("arm_status") != controls:
        errors.append("screenshot arm status does not follow the attempt and control precedence")
    return errors


def _verify_provenance(
    result_root: Path,
    run_root: Path,
    manifest: dict,
    identity_path: Path,
    preflight_path: Path,
    frozen_path: Path,
) -> tuple[dict | None, dict[str, str] | None, list[str]]:
    errors: list[str] = []
    try:
        identity = preflight.load_identity(identity_path)
        identity_hash = _sha256(identity_path)
        _preflight_path, preflight_record = _private_json(result_root, preflight_path, "screenshot preflight")
        preflight_record = preflight._validate_preflight_result(preflight_record)
        version_path, version_raw = _private_json(
            result_root, result_root / "version-manifest.json", "version manifest",
        )
        version = preflight._validate_version_manifest(version_raw)
        _frozen_path, frozen_raw = _private_json(result_root, frozen_path, "frozen harness")
        frozen = preflight._validate_frozen_harness(
            frozen_raw,
            identity_sha=identity_hash,
            candidate_freeze_sha=frozen_raw.get("candidate_freeze_sha256"),
            browser_use_freeze_sha=frozen_raw.get("browser_use_freeze_sha256"),
            tool_manifest_sha=frozen_raw.get("tool_manifest_sha256"),
        )

        expected_phase = "screenshot-final" if run_root.name == "screenshot-final" else "screenshot-run"
        preflight_hash = _sha256(preflight_path)
        version_hash = _sha256(version_path)
        frozen_hash = _sha256(frozen_path)
        if manifest.get("candidate_identity_sha256") != identity_hash \
                or preflight_record.get("candidate_identity_sha256") != identity_hash \
                or frozen.get("candidate_identity_sha256") != identity_hash:
            errors.append("candidate identity hash is not shared by the run and sealed artifacts")
        if manifest.get("preflight_sha256") != preflight_hash:
            errors.append("screenshot preflight hash does not match the run manifest")
        if preflight_record.get("phase") != expected_phase \
                or preflight_record.get("outcome") != "passed" \
                or preflight_record.get("errors") != []:
            errors.append("screenshot preflight is not a passing preflight for this run phase")
        if preflight_record.get("version_manifest_sha256") != manifest.get("version_manifest_sha256") \
                or version_hash != manifest.get("version_manifest_sha256") \
                or frozen.get("version_manifest_sha256") != manifest.get("version_manifest_sha256"):
            errors.append("version-manifest identity does not match the screenshot run")
        if version.get("candidate_identity_sha256") != identity_hash \
                or version.get("tool_manifest_sha256") != manifest.get("tool_manifest_sha256"):
            errors.append("version manifest does not match the candidate or tool identity")
        if version.get("executed_file_sha256") != identity.get("executed_files"):
            errors.append("version manifest executed-file hashes differ from the candidate identity")
        if manifest.get("frozen_harness_sha256") != frozen_hash:
            errors.append("frozen-harness hash does not match the screenshot run")
        if frozen.get("tool_manifest_sha256") != manifest.get("tool_manifest_sha256"):
            errors.append("tool-manifest identity does not match the screenshot run")
        if identity.get("executed_files") != frozen.get("executed_file_sha256"):
            errors.append("candidate and frozen-harness executed-file hashes differ")

        _tool_path, _tool_manifest = _private_json(
            result_root, result_root / "tool-manifest.json", "tool manifest",
        )
        if _sha256(_tool_path) != version.get("tool_manifest_sha256"):
            errors.append("tool-manifest hash does not match the version manifest")
        for freeze_name, filename in (
            ("candidate", "candidate-freeze.txt"),
            ("browser_use", "browser-use-freeze.txt"),
        ):
            freeze_record = version["dependency_freezes"][freeze_name]
            freeze_path = result_root / filename
            if Path(freeze_record["path"]) != freeze_path or _sha256(freeze_path) != freeze_record["sha256"]:
                errors.append(f"{freeze_name} dependency-freeze hash does not match the version manifest")
        candidate_packages = preflight._parse_freeze(
            result_root / "candidate-freeze.txt", expected_under=result_root,
        )
        openbrowser_version = candidate_packages.get("openbrowser-ai", "0.1.56")
        if openbrowser_version != "0.1.56":
            raise ArtifactError("installed OpenBrowser package differs from the validated v0.1.56 runtime")

        executed_files = identity["executed_files"]
        checked_files = {
            "benchmarks/screenshot_capability.py": Path(capability.__file__),
            "benchmarks/screenshot_lightpanda.js": Path(capability.__file__).with_name("screenshot_lightpanda.js"),
            "benchmarks/summarize_screenshot_capability.py": Path(__file__),
        }
        for relative, path in checked_files.items():
            expected_hash = executed_files.get(relative)
            if not isinstance(expected_hash, str) or not path.is_file() \
                    or _sha256(path) != expected_hash:
                errors.append(f"candidate executed-file hash does not match {relative}")
        if manifest.get("harness_sha256") != executed_files.get("benchmarks/screenshot_capability.py"):
            errors.append("screenshot runner hash does not match the candidate identity")
        if manifest.get("lightpanda_script_sha256") != executed_files.get("benchmarks/screenshot_lightpanda.js"):
            errors.append("Lightpanda script hash does not match the candidate identity")

        latest_versions = _check_version_record(preflight_record, "latest_versions", LATEST_VERSION_KEYS)
        installed_versions = _check_version_record(preflight_record, "installed_tool_versions", INSTALLED_TOOL_KEYS)
        if latest_versions != version["latest_versions"]:
            errors.append("preflight latest versions differ from the version manifest")
        expected_installed = {
            name: version["installed_versions"][name] for name in INSTALLED_TOOL_KEYS
        }
        if installed_versions != expected_installed:
            errors.append("preflight installed tool versions differ from the version manifest")
        for tool in INSTALLED_TOOL_KEYS:
            if latest_versions[tool] != installed_versions[tool]:
                errors.append(f"installed {tool} version does not match sealed latest metadata")
        return identity, {**installed_versions, "openbrowser-ai": openbrowser_version}, errors
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"candidate, preflight, or frozen-harness evidence is invalid: {error}")
        return None, None, errors


def verify_run(args) -> tuple[dict | None, dict[str, str] | None, list[str]]:
    try:
        result_root, run_root = capability.result_root_for_run(args.run_root, "verify")
    except (OSError, ValueError) as error:
        raise InputPathError(f"--run-root is invalid: {error}") from error
    identity_path = _input_artifact_path(args.candidate_identity, result_root, "candidate identity")
    preflight_path = _input_artifact_path(args.preflight, result_root, "screenshot preflight")
    frozen_path = _input_artifact_path(args.frozen_harness, result_root, "frozen harness")

    _manifest_path, manifest = _private_json(
        result_root, run_root / "run-manifest.json", "screenshot run manifest",
    )
    verification_args = SimpleNamespace(
        command="verify",
        candidate_identity=str(identity_path),
        preflight=str(preflight_path),
        frozen_harness=str(frozen_path),
    )
    try:
        runner_verification = capability.verify_run_directory(run_root, result_root, verification_args)
    except Exception as error:
        return None, None, [f"screenshot artifact verifier rejected malformed evidence: {type(error).__name__}"]
    if runner_verification != 0:
        return None, None, ["screenshot capability run or attempt artifacts failed offline verification"]

    identity, versions, provenance_errors = _verify_provenance(
        result_root, run_root, manifest, identity_path, preflight_path, frozen_path,
    )
    errors = [*provenance_errors, *_validate_attempt_proofs(manifest),
              *_validate_attempt_pixels(run_root, manifest),
              *_validate_manifest_counts_and_status(manifest)]
    if manifest.get("state") != "complete" or manifest.get("mode") != "run" \
            or manifest.get("schema") != capability.RUN_SCHEMA:
        errors.append("screenshot manifest is not a complete capability run")
    if manifest.get("errors") != []:
        errors.append("complete screenshot run contains run-level errors")
    if not isinstance(manifest.get("cleanup"), dict) \
            or any(manifest["cleanup"].get(key) is not True for key in ("ok", "attempts_ok", "runtime_roots_removed")):
        errors.append("screenshot run cleanup proof is incomplete")
    if not isinstance(manifest.get("fixture"), dict) \
            or any(manifest["fixture"].get(key) is not True for key in ("started", "stopped", "socket_closed", "thread_stopped")):
        errors.append("screenshot fixture cleanup proof is incomplete")
    return (manifest if not errors else None), versions, errors


def render_report(manifest: dict, identity: dict, versions: dict[str, str]) -> str:
    attempts = manifest["attempts"]
    lines = [
        "# Screenshot capability report",
        "",
        "## Tested scope",
        "",
        f"- OpenBrowser runtime: `{versions['openbrowser-ai']}`",
        f"- Candidate commit: `{identity['candidate_commit']}`",
        f"- OpenBrowser runtime Git tree: `{identity['runtime_tree']}`",
        f"- Lightpanda: `{versions['lightpanda']}`",
        f"- Chrome for Testing: `{versions['chrome_for_testing']}`",
        "- Fixture: a local loopback page with one nonce-bound image request per completed capture.",
        "- Contract: capture `#proof` as a 128 by 128 PNG and check four fixed RGB probes.",
        "- Probe points: `(16,16)`, `(48,16)`, `(16,48)`, and `(48,48)`.",
        "- Attempts: six per arm, with one successful nonce-bound image request required per completed capture.",
        "- This report describes screenshot capability only and is separate from navigation timing.",
        "- It is not a universal browser-capability claim.",
        "",
        "## Results",
        "",
        "| Arm | Status | Attempts | Passed | Pixel mismatches | Failed | Pixel probes | Probe mismatches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in capability.SCREENSHOT_ARMS:
        arm_attempts = [attempt for attempt in attempts if attempt["arm"] == arm]
        probe_count = sum(
            attempt["pixel"].get("probe_count", 0)
            for attempt in arm_attempts if isinstance(attempt.get("pixel"), dict)
        )
        mismatch_count = sum(
            attempt["pixel"].get("mismatch_count", 0)
            for attempt in arm_attempts if isinstance(attempt.get("pixel"), dict)
        )
        passed = sum(attempt["state"] == "passed" for attempt in arm_attempts)
        pixel_mismatches = sum(attempt["state"] == "pixel_mismatch" for attempt in arm_attempts)
        failed = sum(attempt["state"] == "failed" for attempt in arm_attempts)
        lines.append(
            f"| {arm} | {manifest['arm_status'][arm]} | {len(arm_attempts)} | {passed} "
            f"| {pixel_mismatches} | {failed} | {probe_count} | {mismatch_count} |"
        )

    counts = manifest["work_counts"]
    lines.extend([
        "",
        "## Work counts",
        "",
        f"- Attempts launched: {counts['attempted_work_count']}",
        f"- Completed pixel checks: {counts['completed_work_count']}",
        f"- Full-contract successes: {counts['successful_work_count']}",
        f"- Pixel-mismatch attempts: {counts['pixel_mismatch_attempt_count']}",
        f"- Pixel probes evaluated: {counts['pixel_probe_count']}",
        f"- Pixel-probe mismatches: {counts['pixel_probe_mismatch_count']}",
        f"- Execution and cleanup errors: {counts['error_count']}",
        "- A pixel mismatch is a completed contract check, not an execution error.",
        "",
        "An `unsupported` result means the tested Lightpanda version failed this exact pixel contract "
        "consistently while both controls passed. It does not establish that Lightpanda cannot take screenshots generally.",
        "",
    ])
    return "\n".join(lines)


def _atomic_write(path: Path, content: str, *, replace: bool) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def dispatch(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        try:
            _result_root, run_root = capability.result_root_for_run(args.run_root, "verify")
        except (OSError, ValueError) as error:
            raise InputPathError(f"--run-root is invalid: {error}") from error
        output, replace = _export_target(args.project_root, args.sanitized_output, run_root)
        manifest, versions, errors = verify_run(args)
        for error in errors:
            LOG.error("%s", error)
        if errors or manifest is None or versions is None:
            return 1

        result_root, _run_root = capability.result_root_for_run(args.run_root, "verify")
        identity_path = _input_artifact_path(args.candidate_identity, result_root, "candidate identity")
        identity = preflight.load_identity(identity_path)
        content = render_report(manifest, identity, versions)
        forbidden = (
            "elapsed_ns", "latency_ns", "navigation_ns", "post_navigation_ns",
            "duration_ns", "start_ns", "end_ns",
        )
        if any(field in content for field in forbidden) or capability.contains_latency_fields({"report": content}):
            LOG.error("sanitized screenshot report contains a latency field")
            return 1
        _atomic_write(output, content, replace=replace)
        LOG.info("wrote verified screenshot capability report")
        return 0
    except InputPathError as error:
        LOG.error("%s", error)
        return 2
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        LOG.error("screenshot capability verification failed: %s", error)
        return 1


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return dispatch(argv)


if __name__ == "__main__":
    raise SystemExit(main())
