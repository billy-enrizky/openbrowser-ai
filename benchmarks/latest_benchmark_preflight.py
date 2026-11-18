#!/usr/bin/env python3
from __future__ import annotations

CLI_DESCRIPTION = "Seal and verify isolated provenance for a current-version local benchmark."

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)

IDENTITY_SCHEMA = "openbrowser-candidate-identity/1"
SOURCE_IMPORT_SCHEMA = "openbrowser-source-import-proof/1"
VERSION_MANIFEST_SCHEMA = "openbrowser-latest-version-manifest/1"
PREFLIGHT_SCHEMA = "openbrowser-latest-preflight/1"
RUNTIME_VARIANT_SCHEMA = "openbrowser-runtime-variant/1"
FROZEN_HARNESS_SCHEMA = "openbrowser-frozen-harness/1"

IDENTITY_KEYS = {
    "schema",
    "candidate_commit",
    "candidate_tree",
    "runtime_tree",
    "release_runtime_tree",
    "origin_main_commit",
    "origin_main_runtime_tree",
    "executed_files",
    "candidate_python",
    "browser_use_python",
}
INTERPRETER_KEYS = {"supplied_path", "resolved_path", "version", "executable_sha256"}
SOURCE_IMPORT_KEYS = {
    "schema",
    "candidate_identity_sha256",
    "source_commit",
    "source_tree",
    "runtime_tree",
    "source_root",
    "supplied_python",
    "resolved_python",
    "python_sha256",
    "working_directory",
    "module_file",
    "module_sha256",
    "config_file",
    "config_sha256",
    "config_debug_log_file",
    "dotenv_disabled",
    "source_sentinel_in_environment",
    "source_sentinel_in_config",
    "unexpected_aws_environment_names",
    "outcome",
}
TOOL_MANIFEST_KEYS = {"schema", "metadata_sha256", "tools"}
TOOL_KEYS = {"lightpanda", "agent-browser", "chrome_for_testing"}
OFFICIAL_TOOL_KEYS = {"path", "installed_version", "download_url", "local_sha256", "official_sha256"}
CHROME_TOOL_KEYS = {"path", "installed_version", "download_url", "local_sha256", "archive_path", "archive_sha256"}
VERSION_MANIFEST_KEYS = {
    "schema",
    "fetched_at_utc",
    "metadata_sources",
    "latest_versions",
    "installed_versions",
    "interpreters",
    "dependency_freezes",
    "tool_manifest_sha256",
    "candidate_identity_sha256",
    "executed_file_sha256",
}
PREFLIGHT_KEYS = {
    "schema",
    "phase",
    "candidate_identity_sha256",
    "version_manifest_sha256",
    "checks",
    "outcome",
    "errors",
}
PREFLIGHT_CHECK_KEYS = {"name", "expected", "actual", "state"}
RUNTIME_VARIANT_KEYS = {
    "schema",
    "baseline",
    "treatment",
    "changed_paths",
    "shared_harness_sha256",
    "candidate_freeze_sha256",
    "browser_use_freeze_sha256",
    "tool_manifest_sha256",
}
SOURCE_REVISION_KEYS = {"commit", "tree", "runtime_tree"}
FROZEN_HARNESS_KEYS = {
    "schema",
    "candidate_identity_sha256",
    "version_manifest_sha256",
    "tool_manifest_sha256",
    "candidate_freeze_sha256",
    "browser_use_freeze_sha256",
    "smoke_preflight_sha256",
    "smoke_manifest_sha256",
    "executed_file_sha256",
}
FREEZE_RECORD_KEYS = {"path", "sha256"}

EXECUTED_FILE_PATHS = (
    "benchmarks/navigation_compare.py",
    "benchmarks/latest_competitor_metadata.py",
    "benchmarks/latest_benchmark_preflight.py",
    "benchmarks/latest_navigation_baseline.py",
    "benchmarks/summarize_latest_navigation.py",
    "benchmarks/screenshot_lightpanda.js",
    "benchmarks/screenshot_capability.py",
    "benchmarks/summarize_screenshot_capability.py",
)

AWS_SAFETY_ENV = {
    "AWS_EC2_METADATA_DISABLED": "true",
    "AWS_CONFIG_FILE": "/dev/null",
    "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
}
PYTHON_VERSION_REQUIRED = "3.12.12"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
FREEZE_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")
VERSION_VALUE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}[A-Za-z0-9._+-]*$")
CHROME_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
PHASE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SOURCE_SENTINEL = "__OPENBROWSER_BENCH_SOURCE_ENV_SENTINEL_9C66A9__"


def _load_metadata_module() -> Any:
    source = Path(__file__).resolve().with_name("latest_competitor_metadata.py")
    spec = importlib.util.spec_from_file_location("_latest_competitor_metadata_preflight", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the standard-library metadata verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_metadata = _load_metadata_module()


class PreflightError(ValueError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    return _metadata.sha256_file(path)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PreflightError(f"{where} must be a JSON object with string keys")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        detail = []
        if missing:
            detail.append(f"missing keys {missing}")
        if unknown:
            detail.append(f"unknown keys {unknown}")
        raise PreflightError(f"{where} has invalid keys: {'; '.join(detail)}")


def _require_nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightError(f"{where} must be a non-empty string")
    return value


def _require_sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PreflightError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _require_git_object(value: Any, where: str) -> str:
    if not isinstance(value, str) or not GIT_OBJECT_RE.fullmatch(value):
        raise PreflightError(f"{where} must be a full lowercase Git object ID")
    return value


def _strict_json(path: str | os.PathLike[str], where: str) -> tuple[dict[str, Any], bytes]:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise PreflightError(f"{where} is unavailable: {file_path}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{where} is not valid UTF-8 JSON") from exc
    return _require_object(value, where), raw


def _path_string(path: Any, where: str) -> Path:
    value = _require_nonempty_string(path, where)
    result = Path(value).expanduser()
    if not result.is_absolute():
        raise PreflightError(f"{where} must be an absolute path")
    return result


def _canonical_existing(path: str | os.PathLike[str], where: str, *, directory: bool | None = None) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise PreflightError(f"{where} must be an absolute path")
    if raw.is_symlink():
        raise PreflightError(f"{where} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise PreflightError(f"{where} does not exist: {raw}") from exc
    if Path(os.path.abspath(raw)) != resolved:
        raise PreflightError(f"{where} must use its physical path")
    if directory is True and not resolved.is_dir():
        raise PreflightError(f"{where} must be a directory")
    if directory is False and not resolved.is_file():
        raise PreflightError(f"{where} must be a regular file")
    return resolved


def _canonical_output(path: str | os.PathLike[str], where: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise PreflightError(f"{where} must be an absolute path")
    if raw.exists() or raw.is_symlink():
        raise PreflightError(f"refusing to overwrite existing {where}: {raw}")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise PreflightError(f"{where} parent directory does not exist: {raw.parent}") from exc
    if Path(os.path.abspath(raw.parent)) != parent:
        raise PreflightError(f"{where} parent directory must use its physical path")
    return parent / raw.name


def _private_result_root(path: str | os.PathLike[str]) -> Path:
    root = _canonical_existing(path, "RESULT_ROOT", directory=True)
    physical_tmp = Path("/tmp").resolve(strict=True)
    if root.parent != physical_tmp:
        raise PreflightError("RESULT_ROOT must be a direct child of physical /tmp")
    status = root.stat()
    if stat.S_IMODE(status.st_mode) != 0o700:
        raise PreflightError("RESULT_ROOT must have mode 0700")
    if status.st_uid != os.getuid():
        raise PreflightError("RESULT_ROOT must be owned by the current user UID")
    return root


def _require_under(path: Path, root: Path, where: str, *, allow_root: bool = False) -> Path:
    resolved = path.resolve(strict=True)
    if resolved == root:
        if allow_root:
            return resolved
        raise PreflightError(f"{where} must be below RESULT_ROOT")
    if not resolved.is_relative_to(root):
        raise PreflightError(f"{where} must be below RESULT_ROOT")
    if Path(os.path.abspath(path)) != resolved:
        raise PreflightError(f"{where} must use its physical path")
    return resolved


def _artifact_file(path: str | os.PathLike[str], root: Path, where: str) -> Path:
    resolved = _canonical_existing(path, where, directory=False)
    return _require_under(resolved, root, where)


def _artifact_directory(path: str | os.PathLike[str], root: Path, where: str) -> Path:
    resolved = _canonical_existing(path, where, directory=True)
    return _require_under(resolved, root, where)


def _private_artifact_directory(path: str | os.PathLike[str], root: Path, where: str) -> Path:
    directory = _artifact_directory(path, root, where)
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise PreflightError(f"{where} must have mode 0700")
    return directory


def _write_new_json(path: str | os.PathLike[str], value: dict[str, Any], where: str) -> Path:
    output = _canonical_output(path, where)
    body = _json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError as exc:
        raise PreflightError(f"refusing to overwrite existing {where}: {output}") from exc
    except OSError as exc:
        raise PreflightError(f"cannot create {where}: {output}") from exc
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(body)
            target.flush()
            os.fsync(target.fileno())
    except OSError as exc:
        try:
            output.unlink()
        except OSError:
            pass
        raise PreflightError(f"cannot write {where}: {output}") from exc
    return output


def _normal_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _check_aws_environment(environment: dict[str, str] | os._Environ[str] | None = None) -> dict[str, list[str]]:
    current = dict(os.environ if environment is None else environment)
    unexpected = sorted(name for name in current if name.startswith("AWS_") and name not in AWS_SAFETY_ENV)
    incorrect = sorted(name for name, expected in AWS_SAFETY_ENV.items() if name in current and current[name] != expected)
    return {"unexpected_names": unexpected, "incorrect_safety_names": incorrect}


def _require_no_aws(environment: dict[str, str] | os._Environ[str] | None = None) -> None:
    state = _check_aws_environment(environment)
    if state["unexpected_names"] or state["incorrect_safety_names"]:
        raise PreflightError(
            "no-AWS boundary failed: "
            f"unexpected AWS variable names={state['unexpected_names']}, "
            f"incorrect safety variable names={state['incorrect_safety_names']}"
        )


def _safe_environment(root: Path, *, python_path: str | None = None) -> dict[str, str]:
    system_path = "/usr/bin:/bin:/usr/sbin:/sbin"
    env = {
        "HOME": str(root),
        "PATH": system_path,
        "TMPDIR": str(root),
        "LANG": "C",
        **AWS_SAFETY_ENV,
    }
    if python_path is not None:
        env["PYTHONPATH"] = python_path
        env["PYTHON_DOTENV_DISABLED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _git(source: Path, *args: str, env_root: Path | None = None, check: bool = True) -> str:
    git_path = shutil.which("git")
    if git_path is None or not Path(git_path).is_absolute():
        raise PreflightError("an absolute git executable is required")
    env = {
        "PATH": f"{Path(git_path).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(env_root or source),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        result = subprocess.run(
            [git_path, "-C", str(source), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f"could not inspect Git checkout {source}") from exc
    if check and result.returncode != 0:
        raise PreflightError(f"Git command failed for {source}: {' '.join(args)}: {result.stderr.strip()}")
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _source_root(path: str | os.PathLike[str], result_root: Path | None = None) -> Path:
    source = _canonical_existing(path, "candidate source", directory=True)
    if result_root is not None and source.is_relative_to(result_root):
        raise PreflightError("candidate source must be outside RESULT_ROOT")
    git_root = _git(source, "rev-parse", "--show-toplevel", env_root=result_root)
    if Path(git_root).resolve(strict=True) != source:
        raise PreflightError("candidate source must be the Git repository root")
    return source


def _git_snapshot(source: Path, *, result_root: Path | None = None) -> dict[str, str]:
    status = _git(source, "status", "--porcelain", "--untracked-files=all", env_root=result_root)
    if status:
        raise PreflightError(f"candidate source checkout is dirty: {source}")
    commit = _git(source, "rev-parse", "HEAD", env_root=result_root)
    tree = _git(source, "rev-parse", "HEAD^{tree}", env_root=result_root)
    runtime_tree = _git(source, "rev-parse", "HEAD:src/openbrowser", env_root=result_root)
    release_runtime_tree = _git(source, "rev-parse", "v0.1.56:src/openbrowser", env_root=result_root)
    origin_main_commit = _git(source, "rev-parse", "origin/main", env_root=result_root)
    origin_main_runtime_tree = _git(source, "rev-parse", "origin/main:src/openbrowser", env_root=result_root)
    for name, value in (
        ("candidate_commit", commit),
        ("candidate_tree", tree),
        ("runtime_tree", runtime_tree),
        ("release_runtime_tree", release_runtime_tree),
        ("origin_main_commit", origin_main_commit),
        ("origin_main_runtime_tree", origin_main_runtime_tree),
    ):
        _require_git_object(value, name)
    if runtime_tree != release_runtime_tree:
        raise PreflightError("candidate src/openbrowser tree does not match the v0.1.56 release tree")
    if runtime_tree != origin_main_runtime_tree:
        raise PreflightError("candidate src/openbrowser tree does not match fetched origin/main")
    return {
        "candidate_commit": commit,
        "candidate_tree": tree,
        "runtime_tree": runtime_tree,
        "release_runtime_tree": release_runtime_tree,
        "origin_main_commit": origin_main_commit,
        "origin_main_runtime_tree": origin_main_runtime_tree,
    }


def _git_head_snapshot(source: Path, *, result_root: Path | None = None) -> dict[str, str]:
    status = _git(source, "status", "--porcelain", "--untracked-files=all", env_root=result_root)
    if status:
        raise PreflightError(f"source checkout is dirty: {source}")
    return {
        "commit": _require_git_object(_git(source, "rev-parse", "HEAD", env_root=result_root), "commit"),
        "tree": _require_git_object(_git(source, "rev-parse", "HEAD^{tree}", env_root=result_root), "tree"),
        "runtime_tree": _require_git_object(_git(source, "rev-parse", "HEAD:src/openbrowser", env_root=result_root), "runtime_tree"),
    }


def _git_file_map(source: Path, ref: str, *, result_root: Path | None = None) -> dict[str, str]:
    git_path = shutil.which("git")
    if git_path is None:
        raise PreflightError("git executable is unavailable")
    env = {
        "PATH": f"{Path(git_path).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(result_root or source),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        listing = subprocess.run(
            [git_path, "-C", str(source), "ls-tree", "-r", "-z", "--name-only", ref],
            check=True,
            capture_output=True,
            timeout=30,
            env=env,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f"could not list tracked files for {source}@{ref}") from exc
    result: dict[str, str] = {}
    for raw_name in listing.split(b"\0"):
        if not raw_name:
            continue
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PreflightError("tracked path is not UTF-8") from exc
        if name.startswith("/") or ".." in Path(name).parts:
            raise PreflightError(f"tracked path is unsafe: {name}")
        try:
            blob = subprocess.run(
                [git_path, "-C", str(source), "show", f"{ref}:{name}"],
                check=True,
                capture_output=True,
                timeout=30,
                env=env,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise PreflightError(f"could not read tracked file {name} at {source}@{ref}") from exc
        result[name] = sha256_bytes(blob)
    return result


def _inspect_python(path: str | os.PathLike[str], *, root: Path, require_current: bool = False) -> dict[str, str]:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise PreflightError("interpreter path must be absolute")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise PreflightError(f"interpreter does not exist: {supplied}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PreflightError(f"interpreter is not an executable file: {resolved}")
    current = Path(sys.executable).resolve(strict=True)
    if require_current and current != resolved:
        raise PreflightError(f"supplied candidate interpreter is not the running executable: {resolved}")
    probe = "import json,sys; sys.stdout.write(json.dumps({'executable':sys.executable,'version':'.'.join(map(str,sys.version_info[:3]))})+'\\n')"
    env = _safe_environment(root)
    try:
        result = subprocess.run(
            [str(resolved), "-I", "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=root,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f"could not inspect interpreter: {resolved}") from exc
    if result.returncode != 0:
        raise PreflightError(f"interpreter version probe failed: {resolved}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"interpreter probe returned invalid JSON: {resolved}") from exc
    report = _require_object(report, "interpreter probe")
    if set(report) != {"executable", "version"}:
        raise PreflightError("interpreter probe returned unknown or missing keys")
    reported_executable = Path(_require_nonempty_string(report["executable"], "interpreter executable")).resolve(strict=True)
    if reported_executable != resolved:
        raise PreflightError("interpreter probe resolved to a different executable")
    version = _require_nonempty_string(report["version"], "interpreter version")
    if version != PYTHON_VERSION_REQUIRED:
        raise PreflightError(f"interpreter version must be {PYTHON_VERSION_REQUIRED}, got {version}")
    return {
        "supplied_path": str(supplied),
        "resolved_path": str(resolved),
        "version": version,
        "executable_sha256": sha256_file(resolved),
    }


def _validate_interpreter(value: Any, where: str) -> dict[str, str]:
    record = _require_object(value, where)
    _require_exact_keys(record, INTERPRETER_KEYS, where)
    supplied = _path_string(record["supplied_path"], f"{where}.supplied_path")
    resolved = _path_string(record["resolved_path"], f"{where}.resolved_path")
    version = _require_nonempty_string(record["version"], f"{where}.version")
    if version != PYTHON_VERSION_REQUIRED:
        raise PreflightError(f"{where}.version must be {PYTHON_VERSION_REQUIRED}")
    digest = _require_sha256(record["executable_sha256"], f"{where}.executable_sha256")
    return {"supplied_path": str(supplied), "resolved_path": str(resolved), "version": version, "executable_sha256": digest}


def _validate_identity(value: Any) -> dict[str, Any]:
    identity = _require_object(value, "candidate identity")
    _require_exact_keys(identity, IDENTITY_KEYS, "candidate identity")
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise PreflightError("candidate identity schema is unsupported")
    for name in ("candidate_commit", "candidate_tree", "runtime_tree", "release_runtime_tree", "origin_main_commit", "origin_main_runtime_tree"):
        _require_git_object(identity.get(name), f"candidate identity.{name}")
    if identity["runtime_tree"] != identity["release_runtime_tree"] or identity["runtime_tree"] != identity["origin_main_runtime_tree"]:
        raise PreflightError("candidate identity runtime tree does not match v0.1.56 and origin/main")
    files = _require_object(identity.get("executed_files"), "candidate identity.executed_files")
    if set(files) != set(EXECUTED_FILE_PATHS):
        raise PreflightError("candidate identity.executed_files has unknown or missing runner/verifier paths")
    for file_path, digest in files.items():
        if Path(file_path).is_absolute() or ".." in Path(file_path).parts:
            raise PreflightError(f"candidate identity path is not repository-relative: {file_path}")
        _require_sha256(digest, f"candidate identity.executed_files.{file_path}")
    _validate_interpreter(identity.get("candidate_python"), "candidate identity.candidate_python")
    _validate_interpreter(identity.get("browser_use_python"), "candidate identity.browser_use_python")
    return identity


def _hash_executed_files(source: Path) -> dict[str, str]:
    executed_files: dict[str, str] = {}
    for relative_path in EXECUTED_FILE_PATHS:
        file_path = source / relative_path
        if file_path.is_symlink() or not file_path.is_file():
            raise PreflightError(f"required benchmark runner or verifier is missing or unsafe: {relative_path}")
        executed_files[relative_path] = sha256_file(file_path)
    return executed_files


def load_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    value, _raw = _strict_json(path, "candidate identity")
    return _validate_identity(value)


def _expected_tool_path(name: str, tools_root: Path) -> Path:
    if name == "lightpanda":
        return tools_root / "lightpanda"
    if name == "agent-browser":
        return tools_root / "agent-browser"
    if name == "chrome_for_testing":
        return tools_root / "chrome-mac-arm64" / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
    raise PreflightError(f"unknown tool name: {name}")


def _validate_tool_manifest(value: Any, *, result_root: Path, tools_root: Path, verify_installation: bool) -> dict[str, Any]:
    manifest = _require_object(value, "tool manifest")
    _require_exact_keys(manifest, TOOL_MANIFEST_KEYS, "tool manifest")
    if manifest.get("schema") != _metadata.TOOL_MANIFEST_SCHEMA:
        raise PreflightError("tool manifest schema is unsupported")
    _require_sha256(manifest.get("metadata_sha256"), "tool manifest.metadata_sha256")
    tools = _require_object(manifest.get("tools"), "tool manifest.tools")
    if set(tools) != TOOL_KEYS:
        raise PreflightError("tool manifest.tools has unknown or missing tool names")
    for name in ("lightpanda", "agent-browser", "chrome_for_testing"):
        record = _require_object(tools[name], f"tool manifest.tools.{name}")
        expected_keys = CHROME_TOOL_KEYS if name == "chrome_for_testing" else OFFICIAL_TOOL_KEYS
        _require_exact_keys(record, expected_keys, f"tool manifest.tools.{name}")
        path = _path_string(record.get("path"), f"tool manifest.tools.{name}.path")
        expected_path = _expected_tool_path(name, tools_root).resolve(strict=False)
        if path.resolve(strict=False) != expected_path:
            raise PreflightError(f"tool manifest path does not match the expected {name} installation")
        _require_nonempty_string(record.get("installed_version"), f"tool manifest.tools.{name}.installed_version")
        _require_nonempty_string(record.get("download_url"), f"tool manifest.tools.{name}.download_url")
        _require_sha256(record.get("local_sha256"), f"tool manifest.tools.{name}.local_sha256")
        if name == "chrome_for_testing":
            archive_path = _path_string(record.get("archive_path"), "tool manifest.tools.chrome_for_testing.archive_path")
            expected_archive = (tools_root / "chrome-mac-arm64.zip").resolve(strict=False)
            if archive_path.resolve(strict=False) != expected_archive:
                raise PreflightError("Chrome archive path is not the expected local archive")
            _require_sha256(record.get("archive_sha256"), "tool manifest.tools.chrome_for_testing.archive_sha256")
        else:
            _require_sha256(record.get("official_sha256"), f"tool manifest.tools.{name}.official_sha256")
        if verify_installation:
            resolved = _canonical_existing(path, f"installed {name}", directory=False)
            _require_under(resolved, tools_root, f"installed {name}")
            if not os.access(resolved, os.X_OK):
                raise PreflightError(f"installed {name} is not executable")
            if sha256_file(resolved) != record["local_sha256"]:
                raise PreflightError(f"installed {name} file hash differs from tool manifest")
            if name in {"lightpanda", "agent-browser"} and record["local_sha256"] != record["official_sha256"]:
                raise PreflightError(f"installed {name} official digest does not match its local hash")
            if name == "chrome_for_testing":
                archive = _canonical_existing(record["archive_path"], "Chrome archive", directory=False)
                _require_under(archive, tools_root, "Chrome archive")
                if sha256_file(archive) != record["archive_sha256"]:
                    raise PreflightError("Chrome archive hash differs from tool manifest")
            try:
                actual_version = _metadata._isolated_version(resolved)
            except _metadata.MetadataError as exc:
                raise PreflightError(f"could not verify installed {name} version") from exc
            if actual_version != record["installed_version"]:
                raise PreflightError(f"installed {name} version differs from tool manifest")
    return manifest


def load_tool_manifest(
    path: str | os.PathLike[str],
    *,
    result_root: str | os.PathLike[str] | None = None,
    verify_installation: bool = True,
) -> dict[str, Any]:
    tool_manifest_path = Path(path)
    root = _private_result_root(result_root or tool_manifest_path.parent)
    tool_manifest_path = _artifact_file(tool_manifest_path, root, "tool manifest")
    tools_root = _private_artifact_directory(root / "tools", root, "tools directory")
    value, _raw = _strict_json(tool_manifest_path, "tool manifest")
    return _validate_tool_manifest(value, result_root=root, tools_root=tools_root, verify_installation=verify_installation)


def _parse_freeze(path: Path, *, expected_under: Path) -> dict[str, str]:
    _artifact_file(path, expected_under, "dependency freeze")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PreflightError(f"dependency freeze is not valid UTF-8 text: {path}") from exc
    if not lines:
        raise PreflightError(f"dependency freeze is empty: {path}")
    packages: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        if not line or line.lstrip().startswith("#"):
            continue
        match = FREEZE_LINE_RE.fullmatch(line.strip())
        if match is None:
            raise PreflightError(f"dependency freeze has an unsupported row at line {number}: {path}")
        name, version = match.groups()
        normalized = _normal_package_name(name)
        if normalized in packages:
            raise PreflightError(f"dependency freeze contains a duplicate package: {name}")
        packages[normalized] = version
    if not packages:
        raise PreflightError(f"dependency freeze contains no pinned packages: {path}")
    return packages


def _validate_metadata(parsed: dict[str, Any]) -> dict[str, Any]:
    latest_versions = _require_object(parsed.get("latest_versions"), "latest metadata versions")
    expected_latest = {"lightpanda", "agent-browser", "browser-use", "browser-harness", "chrome_for_testing"}
    if set(latest_versions) != expected_latest:
        raise PreflightError("latest metadata versions have unknown or missing names")
    sources = _require_object(parsed.get("metadata_sources"), "metadata sources")
    if set(sources) != set(_metadata.SOURCE_URLS):
        raise PreflightError("metadata sources have unknown or missing names")
    validated_sources: dict[str, dict[str, str]] = {}
    for name, expected_url in _metadata.SOURCE_URLS.items():
        record = _require_object(sources[name], f"metadata sources.{name}")
        _require_exact_keys(record, {"url", "response_sha256"}, f"metadata sources.{name}")
        if record.get("url") != expected_url:
            raise PreflightError(f"metadata sources.{name} URL differs from the official source")
        validated_sources[name] = {
            "url": expected_url,
            "response_sha256": _require_sha256(record.get("response_sha256"), f"metadata sources.{name}.response_sha256"),
        }
    release_assets = _require_object(parsed.get("release_assets"), "metadata release assets")
    if set(release_assets) != {"lightpanda", "agent-browser"}:
        raise PreflightError("metadata release assets have unknown or missing tool names")
    for name in ("lightpanda", "agent-browser"):
        asset = _require_object(release_assets[name], f"metadata release assets.{name}")
        _require_exact_keys(asset, {"version", "asset_name", "download_url", "official_sha256", "size"}, f"metadata release assets.{name}")
        _require_sha256(asset.get("official_sha256"), f"metadata release assets.{name}.official_sha256")
        if isinstance(asset.get("size"), bool) or not isinstance(asset.get("size"), int) or asset["size"] <= 0:
            raise PreflightError(f"metadata release assets.{name}.size must be a positive integer")
    _require_nonempty_string(parsed.get("chrome_download_url"), "Chrome download URL")
    return {
        "latest_versions": latest_versions,
        "metadata_sources": validated_sources,
        "release_assets": release_assets,
        "chrome_download_url": parsed["chrome_download_url"],
        "metadata_manifest_sha256": _require_sha256(parsed.get("metadata_manifest_sha256"), "metadata manifest SHA-256"),
    }


def _read_metadata(metadata_dir: Path) -> dict[str, Any]:
    try:
        return _validate_metadata(_metadata.parse_metadata_sources(metadata_dir))
    except _metadata.MetadataError as exc:
        raise PreflightError(f"invalid required metadata: {exc}") from exc


def _tool_versions(tool_manifest: dict[str, Any]) -> dict[str, str]:
    return {name: tool_manifest["tools"][name]["installed_version"] for name in ("lightpanda", "agent-browser", "chrome_for_testing")}


def _latest_expected_tool_versions(parsed: dict[str, Any]) -> dict[str, str]:
    versions = parsed["latest_versions"]
    return {
        "lightpanda": versions["lightpanda"],
        "agent-browser": versions["agent-browser"],
        "chrome_for_testing": versions["chrome_for_testing"],
    }


def _package_installation_versions(
    latest_versions: dict[str, str], candidate_packages: dict[str, str], browser_use_packages: dict[str, str]
) -> dict[str, str]:
    required = {"browser-use": "browser-use", "browser-harness": "browser-harness"}
    installed: dict[str, str] = {}
    for package, freeze_key in required.items():
        normalized = _normal_package_name(freeze_key)
        if normalized not in browser_use_packages:
            raise PreflightError(f"browser-use dependency freeze is missing {package}")
        installed[package] = browser_use_packages[normalized]
        if installed[package] != latest_versions[package]:
            raise PreflightError(f"installed {package} version does not match current official metadata")
    for dependency in ("psutil", "websockets"):
        normalized = _normal_package_name(dependency)
        if normalized in candidate_packages:
            installed[f"candidate:{dependency}"] = candidate_packages[normalized]
        if normalized in browser_use_packages:
            installed[f"browser_use:{dependency}"] = browser_use_packages[normalized]
    return installed


def _validate_source_import_proof(value: Any) -> dict[str, Any]:
    proof = _require_object(value, "source import proof")
    _require_exact_keys(proof, SOURCE_IMPORT_KEYS, "source import proof")
    if proof.get("schema") != SOURCE_IMPORT_SCHEMA:
        raise PreflightError("source import proof schema is unsupported")
    _require_sha256(proof.get("candidate_identity_sha256"), "source import proof.candidate_identity_sha256")
    for field in ("source_commit", "source_tree", "runtime_tree"):
        _require_git_object(proof.get(field), f"source import proof.{field}")
    for field in ("source_root", "supplied_python", "resolved_python", "working_directory", "module_file", "config_file"):
        _path_string(proof.get(field), f"source import proof.{field}")
    for field in ("python_sha256", "module_sha256", "config_sha256"):
        _require_sha256(proof.get(field), f"source import proof.{field}")
    if proof.get("config_debug_log_file") is not None and not isinstance(proof.get("config_debug_log_file"), str):
        raise PreflightError("source import proof.config_debug_log_file must be a string or null")
    if proof.get("dotenv_disabled") is not True:
        raise PreflightError("source import proof does not establish PYTHON_DOTENV_DISABLED=1")
    if proof.get("source_sentinel_in_environment") is not False or proof.get("source_sentinel_in_config") is not False:
        raise PreflightError("source import proof found a source .env sentinel")
    aws_names = proof.get("unexpected_aws_environment_names")
    if aws_names != []:
        raise PreflightError("source import proof contains unexpected AWS variables")
    if proof.get("outcome") != "passed":
        raise PreflightError("source import proof did not pass")
    return proof


def _source_import_report(source: Path, python: dict[str, str], cwd: Path, identity_sha256: str, *, result_root: Path) -> dict[str, Any]:
    if cwd.is_symlink() or not cwd.is_dir():
        raise PreflightError("source import working directory must be a real directory")
    resolved_cwd = _canonical_existing(cwd, "source import working directory", directory=True)
    _require_under(resolved_cwd, result_root, "source import working directory")
    if resolved_cwd.is_relative_to(source):
        raise PreflightError("source import working directory must be outside the source checkout")
    if stat.S_IMODE(resolved_cwd.stat().st_mode) != 0o700:
        raise PreflightError("source import working directory must have mode 0700")
    try:
        entries = list(resolved_cwd.iterdir())
    except OSError as exc:
        raise PreflightError("source import working directory cannot be inspected") from exc
    if entries:
        raise PreflightError("source import working directory must be empty, not a decoy or inherited CWD")
    _require_no_aws()
    snapshot = _git_head_snapshot(source, result_root=result_root)
    source_path = source / "src"
    if not source_path.is_dir():
        raise PreflightError("candidate source is missing src/")
    program = (
        "import importlib, json, os, pathlib, sys; "
        "import openbrowser; import openbrowser.config as config_module; "
        "config = config_module.CONFIG; "
        "module_file = pathlib.Path(openbrowser.__file__).resolve(); "
        "config_file = pathlib.Path(config_module.__file__).resolve(); "
        "debug_value = getattr(config, 'OPENBROWSER_DEBUG_LOG_FILE', None); "
        "sentinel = '__OPENBROWSER_BENCH_SOURCE_ENV_SENTINEL_9C66A9__'; "
        "sys.stdout.write(json.dumps({'executable': str(pathlib.Path(sys.executable).resolve()), "
        "'module_file': str(module_file), 'config_file': str(config_file), "
        "'config_debug_log_file': None if debug_value is None else str(debug_value), "
        "'dotenv_disabled': os.environ.get('PYTHON_DOTENV_DISABLED') == '1', "
        "'source_sentinel_in_environment': any(sentinel in k or sentinel in v for k, v in os.environ.items()), "
        "'source_sentinel_in_config': sentinel in str(debug_value), "
        "'unexpected_aws_environment_names': sorted(k for k in os.environ if k.startswith('AWS_') "
        "and k not in {'AWS_EC2_METADATA_DISABLED','AWS_CONFIG_FILE','AWS_SHARED_CREDENTIALS_FILE'})})+'\\n')"
    )
    env = _safe_environment(resolved_cwd, python_path=str(source_path.resolve(strict=True)))
    try:
        result = subprocess.run(
            [python["supplied_path"], "-c", program],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            cwd=resolved_cwd,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError("candidate source import probe could not run") from exc
    if result.returncode != 0:
        raise PreflightError(f"candidate source import failed: {result.stderr.strip()[-1000:]}")
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise PreflightError("candidate source import probe returned invalid JSON") from exc
    report = _require_object(report, "source import process report")
    report_keys = {
        "executable",
        "module_file",
        "config_file",
        "config_debug_log_file",
        "dotenv_disabled",
        "source_sentinel_in_environment",
        "source_sentinel_in_config",
        "unexpected_aws_environment_names",
    }
    _require_exact_keys(report, report_keys, "source import process report")
    if Path(report["executable"]).resolve(strict=True) != Path(python["resolved_path"]):
        raise PreflightError("source import probe used the wrong Python executable")
    module_file = _canonical_existing(report["module_file"], "imported openbrowser package", directory=False)
    config_file = _canonical_existing(report["config_file"], "imported openbrowser.config module", directory=False)
    package_root = (source / "src" / "openbrowser").resolve(strict=True)
    if not module_file.is_relative_to(package_root) or not config_file.is_relative_to(package_root):
        raise PreflightError("source import probe resolved openbrowser outside the selected source tree")
    if report["dotenv_disabled"] is not True:
        raise PreflightError("source import probe did not disable python-dotenv")
    if report["source_sentinel_in_environment"] is not False or report["source_sentinel_in_config"] is not False:
        raise PreflightError("source import probe loaded a source .env sentinel")
    unexpected_aws = report["unexpected_aws_environment_names"]
    if unexpected_aws != []:
        raise PreflightError("source import probe received AWS credential or override variables")
    return {
        "schema": SOURCE_IMPORT_SCHEMA,
        "candidate_identity_sha256": identity_sha256,
        "source_commit": snapshot["commit"],
        "source_tree": snapshot["tree"],
        "runtime_tree": snapshot["runtime_tree"],
        "source_root": str(source),
        "supplied_python": python["supplied_path"],
        "resolved_python": python["resolved_path"],
        "python_sha256": python["executable_sha256"],
        "working_directory": str(resolved_cwd),
        "module_file": str(module_file),
        "module_sha256": sha256_file(module_file),
        "config_file": str(config_file),
        "config_sha256": sha256_file(config_file),
        "config_debug_log_file": report["config_debug_log_file"],
        "dotenv_disabled": report["dotenv_disabled"],
        "source_sentinel_in_environment": report["source_sentinel_in_environment"],
        "source_sentinel_in_config": report["source_sentinel_in_config"],
        "unexpected_aws_environment_names": unexpected_aws,
        "outcome": "passed",
    }


def create_identity(
    source: str | os.PathLike[str],
    python: str | os.PathLike[str],
    browser_use_python: str | os.PathLike[str],
    candidate_freeze: str | os.PathLike[str],
    browser_use_freeze: str | os.PathLike[str],
    tool_manifest: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    _require_no_aws()
    output_path = _canonical_output(output, "candidate identity")
    result_root = _private_result_root(output_path.parent)
    source_root = _source_root(source, result_root)
    freeze_candidate = _artifact_file(candidate_freeze, result_root, "candidate dependency freeze")
    freeze_browser = _artifact_file(browser_use_freeze, result_root, "browser-use dependency freeze")
    tool_manifest_path = _artifact_file(tool_manifest, result_root, "tool manifest")
    if any(path.parent != result_root for path in (freeze_candidate, freeze_browser, tool_manifest_path)):
        raise PreflightError("dependency freezes and tool manifest must be direct children of RESULT_ROOT")
    tools_root = _private_artifact_directory(result_root / "tools", result_root, "tools directory")
    _parse_freeze(freeze_candidate, expected_under=result_root)
    _parse_freeze(freeze_browser, expected_under=result_root)
    tool_manifest_value, _tool_raw = _strict_json(tool_manifest_path, "tool manifest")
    _validate_tool_manifest(tool_manifest_value, result_root=result_root, tools_root=tools_root, verify_installation=True)
    candidate_interpreter = _inspect_python(python, root=result_root, require_current=True)
    browser_use_interpreter = _inspect_python(browser_use_python, root=result_root)
    source_snapshot = _git_snapshot(source_root, result_root=result_root)
    executed_files = _hash_executed_files(source_root)
    identity = {
        "schema": IDENTITY_SCHEMA,
        **source_snapshot,
        "executed_files": executed_files,
        "candidate_python": candidate_interpreter,
        "browser_use_python": browser_use_interpreter,
    }
    _validate_identity(identity)
    _write_new_json(output_path, identity, "candidate identity")
    LOGGER.info("wrote candidate identity for a clean runtime-matched checkout")
    return identity


def create_source_import_proof(
    source: str | os.PathLike[str],
    python: str | os.PathLike[str],
    cwd: str | os.PathLike[str],
    identity: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    _require_no_aws()
    output_path = _canonical_output(output, "source import proof")
    result_root = _private_result_root(output_path.parent)
    source_root = _source_root(source, result_root)
    identity_path = _artifact_file(identity, result_root, "candidate identity")
    identity_value, identity_raw = _strict_json(identity_path, "candidate identity")
    identity_value = _validate_identity(identity_value)
    current_python = _inspect_python(python, root=result_root, require_current=True)
    if current_python != identity_value["candidate_python"]:
        raise PreflightError("source import proof interpreter differs from candidate identity")
    current = _git_snapshot(source_root, result_root=result_root)
    if current["candidate_commit"] != identity_value["candidate_commit"] or current["candidate_tree"] != identity_value["candidate_tree"]:
        raise PreflightError("source import proof source differs from candidate identity")
    cwd_path = _canonical_existing(cwd, "source import working directory", directory=True)
    proof = _source_import_report(source_root, current_python, cwd_path, sha256_bytes(identity_raw), result_root=result_root)
    _validate_source_import_proof(proof)
    _write_new_json(output_path, proof, "source import proof")
    LOGGER.info("wrote a source-import proof from an empty private working directory")
    return proof


def _validate_freeze_record(value: Any, where: str) -> dict[str, str]:
    record = _require_object(value, where)
    _require_exact_keys(record, FREEZE_RECORD_KEYS, where)
    return {
        "path": str(_path_string(record["path"], f"{where}.path")),
        "sha256": _require_sha256(record["sha256"], f"{where}.sha256"),
    }


def _validate_version_manifest(value: Any) -> dict[str, Any]:
    manifest = _require_object(value, "version manifest")
    _require_exact_keys(manifest, VERSION_MANIFEST_KEYS, "version manifest")
    if manifest.get("schema") != VERSION_MANIFEST_SCHEMA:
        raise PreflightError("version manifest schema is unsupported")
    timestamp = _require_nonempty_string(manifest.get("fetched_at_utc"), "version manifest.fetched_at_utc")
    try:
        _metadata._validate_timestamp(timestamp, "version manifest.fetched_at_utc")
    except _metadata.MetadataError as exc:
        raise PreflightError(f"version manifest.fetched_at_utc is invalid: {exc}") from exc
    sources = _require_object(manifest.get("metadata_sources"), "version manifest.metadata_sources")
    if set(sources) != set(_metadata.SOURCE_URLS):
        raise PreflightError("version manifest.metadata_sources has unknown or missing names")
    for name, expected_url in _metadata.SOURCE_URLS.items():
        record = _require_object(sources[name], f"version manifest.metadata_sources.{name}")
        _require_exact_keys(record, {"url", "response_sha256"}, f"version manifest.metadata_sources.{name}")
        if record.get("url") != expected_url:
            raise PreflightError(f"version manifest metadata source URL is invalid: {name}")
        _require_sha256(record.get("response_sha256"), f"version manifest.metadata_sources.{name}.response_sha256")
    latest_versions = _require_object(manifest.get("latest_versions"), "version manifest.latest_versions")
    if set(latest_versions) != {"lightpanda", "agent-browser", "browser-use", "browser-harness", "chrome_for_testing"}:
        raise PreflightError("version manifest.latest_versions has unknown or missing names")
    for name, version in latest_versions.items():
        version = _require_nonempty_string(version, f"version manifest.latest_versions.{name}")
        pattern = CHROME_VERSION_RE if name == "chrome_for_testing" else VERSION_VALUE_RE
        if not pattern.fullmatch(version):
            raise PreflightError(f"version manifest.latest_versions.{name} is not a valid version")
    installed_versions = _require_object(manifest.get("installed_versions"), "version manifest.installed_versions")
    if set(installed_versions) != {
        "lightpanda",
        "agent-browser",
        "chrome_for_testing",
        "browser-use",
        "browser-harness",
        "candidate:psutil",
        "browser_use:psutil",
        "candidate:websockets",
        "browser_use:websockets",
    }:
        raise PreflightError("version manifest.installed_versions has unknown or missing names")
    for name, version in installed_versions.items():
        version = _require_nonempty_string(version, f"version manifest.installed_versions.{name}")
        if not VERSION_VALUE_RE.fullmatch(version):
            raise PreflightError(f"version manifest.installed_versions.{name} is not a valid version")
    interpreters = _require_object(manifest.get("interpreters"), "version manifest.interpreters")
    _require_exact_keys(interpreters, {"candidate", "browser_use"}, "version manifest.interpreters")
    _validate_interpreter(interpreters["candidate"], "version manifest.interpreters.candidate")
    _validate_interpreter(interpreters["browser_use"], "version manifest.interpreters.browser_use")
    freezes = _require_object(manifest.get("dependency_freezes"), "version manifest.dependency_freezes")
    _require_exact_keys(freezes, {"candidate", "browser_use"}, "version manifest.dependency_freezes")
    _validate_freeze_record(freezes["candidate"], "version manifest.dependency_freezes.candidate")
    _validate_freeze_record(freezes["browser_use"], "version manifest.dependency_freezes.browser_use")
    _require_sha256(manifest.get("tool_manifest_sha256"), "version manifest.tool_manifest_sha256")
    _require_sha256(manifest.get("candidate_identity_sha256"), "version manifest.candidate_identity_sha256")
    executed = _require_object(manifest.get("executed_file_sha256"), "version manifest.executed_file_sha256")
    if set(executed) != set(EXECUTED_FILE_PATHS):
        raise PreflightError("version manifest.executed_file_sha256 has unknown or missing paths")
    for relative_path, digest in executed.items():
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise PreflightError("version manifest executed-file path is unsafe")
        _require_sha256(digest, f"version manifest.executed_file_sha256.{relative_path}")
    return manifest


def seal_version_manifest(
    source: str | os.PathLike[str],
    python: str | os.PathLike[str],
    browser_use_python: str | os.PathLike[str],
    metadata: str | os.PathLike[str],
    tools: str | os.PathLike[str],
    candidate_freeze: str | os.PathLike[str],
    browser_use_freeze: str | os.PathLike[str],
    tool_manifest: str | os.PathLike[str],
    identity: str | os.PathLike[str],
    source_import_proof: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    _require_no_aws()
    output_path = _canonical_output(output, "version manifest")
    result_root = _private_result_root(output_path.parent)
    source_root = _source_root(source, result_root)
    metadata_root = _private_artifact_directory(metadata, result_root, "metadata directory")
    tools_root = _private_artifact_directory(tools, result_root, "tools directory")
    freeze_candidate = _artifact_file(candidate_freeze, result_root, "candidate dependency freeze")
    freeze_browser = _artifact_file(browser_use_freeze, result_root, "browser-use dependency freeze")
    tool_manifest_path = _artifact_file(tool_manifest, result_root, "tool manifest")
    identity_path = _artifact_file(identity, result_root, "candidate identity")
    proof_path = _artifact_file(source_import_proof, result_root, "source import proof")
    if any(path.parent != result_root for path in (freeze_candidate, freeze_browser, tool_manifest_path, identity_path, proof_path)):
        raise PreflightError("sealed metadata artifacts must be direct children of RESULT_ROOT")
    if tools_root != result_root / "tools" or metadata_root != result_root / "metadata":
        raise PreflightError("metadata and tools must use the documented RESULT_ROOT subdirectories")
    identity_value, identity_raw = _strict_json(identity_path, "candidate identity")
    identity_value = _validate_identity(identity_value)
    source_snapshot = _git_snapshot(source_root, result_root=result_root)
    for field, actual in source_snapshot.items():
        if identity_value[field] != actual:
            raise PreflightError(f"candidate source no longer matches identity field {field}")
    if identity_value["executed_files"] != _hash_executed_files(source_root):
        raise PreflightError("candidate runner or verifier files changed after identity creation")
    candidate_python = _inspect_python(python, root=result_root, require_current=True)
    browser_python = _inspect_python(browser_use_python, root=result_root)
    if candidate_python != identity_value["candidate_python"] or browser_python != identity_value["browser_use_python"]:
        raise PreflightError("interpreter identities differ from candidate identity")
    parsed = _read_metadata(metadata_root)
    metadata_manifest = _metadata._read_metadata_manifest(metadata_root)[0]
    tool_manifest_value, tool_manifest_raw = _strict_json(tool_manifest_path, "tool manifest")
    tool_manifest_value = _validate_tool_manifest(
        tool_manifest_value,
        result_root=result_root,
        tools_root=tools_root,
        verify_installation=True,
    )
    if tool_manifest_value["metadata_sha256"] != parsed.get("metadata_manifest_sha256"):
        raise PreflightError("tool manifest is not bound to the saved metadata response set")
    candidate_packages = _parse_freeze(freeze_candidate, expected_under=result_root)
    browser_packages = _parse_freeze(freeze_browser, expected_under=result_root)
    installed_tool_versions = _tool_versions(tool_manifest_value)
    if installed_tool_versions != _latest_expected_tool_versions(parsed):
        raise PreflightError("installed tool versions do not match current official metadata")
    installed_versions = {
        **installed_tool_versions,
        **_package_installation_versions(parsed["latest_versions"], candidate_packages, browser_packages),
    }
    proof_value, _proof_raw = _strict_json(proof_path, "source import proof")
    proof_value = _validate_source_import_proof(proof_value)
    if proof_value["candidate_identity_sha256"] != sha256_bytes(identity_raw):
        raise PreflightError("source import proof does not match candidate identity")
    if (
        proof_value["source_root"] != str(source_root)
        or proof_value["source_commit"] != source_snapshot["candidate_commit"]
        or proof_value["source_tree"] != source_snapshot["candidate_tree"]
        or proof_value["runtime_tree"] != source_snapshot["runtime_tree"]
    ):
        raise PreflightError("source import proof source differs from candidate checkout")
    if proof_value["resolved_python"] != candidate_python["resolved_path"] or proof_value["python_sha256"] != candidate_python["executable_sha256"]:
        raise PreflightError("source import proof interpreter differs from the candidate interpreter")
    cwd = _canonical_existing(proof_value["working_directory"], "source import working directory", directory=True)
    _require_under(cwd, result_root, "source import working directory")
    if cwd.is_relative_to(source_root):
        raise PreflightError("source import proof working directory must be outside the candidate source")
    cwd_status = cwd.stat()
    if (
        stat.S_IMODE(cwd_status.st_mode) != 0o700
        or cwd_status.st_uid != os.getuid()
        or list(cwd.iterdir())
    ):
        raise PreflightError("source import proof working directory is not empty and private")
    module_file = _canonical_existing(proof_value["module_file"], "imported OpenBrowser package", directory=False)
    config_file = _canonical_existing(proof_value["config_file"], "imported OpenBrowser config", directory=False)
    if module_file.is_relative_to(source_root / ".git") or not module_file.is_relative_to(source_root / "src" / "openbrowser"):
        raise PreflightError("source import proof package path escaped the candidate source")
    if not config_file.is_relative_to(source_root / "src" / "openbrowser"):
        raise PreflightError("source import proof config path escaped the candidate source")
    if sha256_file(module_file) != proof_value["module_sha256"] or sha256_file(config_file) != proof_value["config_sha256"]:
        raise PreflightError("source import proof files changed after the probe")
    version_manifest = {
        "schema": VERSION_MANIFEST_SCHEMA,
        "fetched_at_utc": metadata_manifest["fetched_at_utc"],
        "metadata_sources": parsed["metadata_sources"],
        "latest_versions": parsed["latest_versions"],
        "installed_versions": installed_versions,
        "interpreters": {"candidate": candidate_python, "browser_use": browser_python},
        "dependency_freezes": {
            "candidate": {"path": str(freeze_candidate), "sha256": sha256_file(freeze_candidate)},
            "browser_use": {"path": str(freeze_browser), "sha256": sha256_file(freeze_browser)},
        },
        "tool_manifest_sha256": sha256_bytes(tool_manifest_raw),
        "candidate_identity_sha256": sha256_bytes(identity_raw),
        "executed_file_sha256": identity_value["executed_files"],
    }
    _validate_version_manifest(version_manifest)
    _write_new_json(output_path, version_manifest, "version manifest")
    LOGGER.info("sealed immutable current-version, interpreter, tool, and dependency identities")
    return version_manifest


def _record_check(checks: list[dict[str, Any]], errors: list[str], name: str, expected: Any, actual: Any, passed: bool) -> None:
    state = "passed" if passed else "failed"
    checks.append({"name": name, "expected": expected, "actual": actual, "state": state})
    if not passed:
        errors.append(f"{name} did not match the sealed value")


def _check_equal(checks: list[dict[str, Any]], errors: list[str], name: str, expected: Any, actual: Any) -> None:
    _record_check(checks, errors, name, expected, actual, expected == actual)


def _inspect_python_for_check(
    checks: list[dict[str, Any]], errors: list[str], name: str, path: str | os.PathLike[str], *, root: Path, require_current: bool
) -> dict[str, str] | None:
    try:
        result = _inspect_python(path, root=root, require_current=require_current)
    except PreflightError as exc:
        _record_check(checks, errors, name, PYTHON_VERSION_REQUIRED, str(exc), False)
        return None
    _record_check(checks, errors, name, PYTHON_VERSION_REQUIRED, result["version"], True)
    return result


def _tool_installation_for_check(
    checks: list[dict[str, Any]], errors: list[str], tool_manifest_value: dict[str, Any] | None, root: Path
) -> None:
    if tool_manifest_value is None:
        _record_check(checks, errors, "tool_installation", "all frozen tools and hashes valid", "tool manifest unavailable", False)
        return
    try:
        tools_root = _private_artifact_directory(root / "tools", root, "tools directory")
        _validate_tool_manifest(tool_manifest_value, result_root=root, tools_root=tools_root, verify_installation=True)
    except PreflightError as exc:
        _record_check(checks, errors, "tool_installation", "all frozen tools and hashes valid", str(exc), False)
    else:
        _record_check(checks, errors, "tool_installation", "all frozen tools and hashes valid", "all tools verified", True)


def _source_record(source: Path, *, result_root: Path) -> dict[str, str]:
    return _git_head_snapshot(source, result_root=result_root)


def _verify_runtime_variant(
    variant_path: Path,
    identity: dict[str, Any],
    identity_raw: bytes,
    source: Path,
    root: Path,
    current_python: dict[str, str] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        value, _raw = _strict_json(variant_path, "runtime-variant identity")
        variant = _validate_runtime_variant(value)
        current = _source_record(source, result_root=root)
        if variant["baseline"] != {"commit": identity["candidate_commit"], "tree": identity["candidate_tree"], "runtime_tree": identity["runtime_tree"]}:
            raise PreflightError("runtime-variant baseline differs from candidate identity")
        if variant["treatment"] != current:
            raise PreflightError("runtime-variant treatment differs from the selected source")
        if current_python is not None and current_python["executable_sha256"] != identity["candidate_python"]["executable_sha256"]:
            raise PreflightError("runtime-variant uses a different candidate interpreter")
        if not variant["changed_paths"] or any(not path.startswith("src/openbrowser/") for path in variant["changed_paths"]):
            raise PreflightError("runtime-variant changes are not limited to src/openbrowser/")
        actual_map = _git_file_map(source, identity["candidate_commit"], result_root=root)
        current_map = _git_file_map(source, "HEAD", result_root=root)
        differences = sorted(path for path in set(actual_map) | set(current_map) if actual_map.get(path) != current_map.get(path))
        if differences != variant["changed_paths"]:
            raise PreflightError("runtime-variant changed_paths do not match the committed source difference")
        if any(not path.startswith("src/openbrowser/") for path in differences):
            raise PreflightError("runtime-variant contains a non-runtime source change")
        return variant, errors
    except PreflightError as exc:
        errors.append(str(exc))
        return None, errors


def _validate_runtime_variant(value: Any) -> dict[str, Any]:
    variant = _require_object(value, "runtime-variant identity")
    _require_exact_keys(variant, RUNTIME_VARIANT_KEYS, "runtime-variant identity")
    if variant.get("schema") != RUNTIME_VARIANT_SCHEMA:
        raise PreflightError("runtime-variant identity schema is unsupported")
    for name in ("baseline", "treatment"):
        source = _require_object(variant.get(name), f"runtime-variant.{name}")
        _require_exact_keys(source, SOURCE_REVISION_KEYS, f"runtime-variant.{name}")
        for field, value in source.items():
            _require_git_object(value, f"runtime-variant.{name}.{field}")
    paths = variant.get("changed_paths")
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise PreflightError("runtime-variant.changed_paths must be a string list")
    if paths != sorted(set(paths)):
        raise PreflightError("runtime-variant.changed_paths must be unique and sorted")
    for path in paths:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise PreflightError(f"runtime-variant path is unsafe: {path}")
    for field in ("shared_harness_sha256", "candidate_freeze_sha256", "browser_use_freeze_sha256", "tool_manifest_sha256"):
        _require_sha256(variant.get(field), f"runtime-variant.{field}")
    return variant


def _validate_frozen_harness(value: Any, *, identity_sha: str, candidate_freeze_sha: str, browser_use_freeze_sha: str, tool_manifest_sha: str) -> dict[str, Any]:
    harness = _require_object(value, "frozen harness")
    _require_exact_keys(harness, FROZEN_HARNESS_KEYS, "frozen harness")
    if harness.get("schema") != FROZEN_HARNESS_SCHEMA:
        raise PreflightError("frozen harness schema is unsupported")
    for field, expected in (
        ("candidate_identity_sha256", identity_sha),
        ("candidate_freeze_sha256", candidate_freeze_sha),
        ("browser_use_freeze_sha256", browser_use_freeze_sha),
        ("tool_manifest_sha256", tool_manifest_sha),
    ):
        if _require_sha256(harness.get(field), f"frozen harness.{field}") != expected:
            raise PreflightError(f"frozen harness {field} differs from current sealed artifacts")
    _require_sha256(harness.get("version_manifest_sha256"), "frozen harness.version_manifest_sha256")
    smoke_preflights = _require_object(
        harness.get("smoke_preflight_sha256"), "frozen harness.smoke_preflight_sha256",
    )
    if set(smoke_preflights) != {"navigation", "screenshot"}:
        raise PreflightError("frozen harness must include both control-smoke preflight hashes")
    for name, digest in smoke_preflights.items():
        _require_sha256(digest, f"frozen harness.smoke_preflight_sha256.{name}")
    smoke_manifests = _require_object(harness.get("smoke_manifest_sha256"), "frozen harness.smoke_manifest_sha256")
    if set(smoke_manifests) != {"navigation", "screenshot"}:
        raise PreflightError("frozen harness must include both control-smoke manifest hashes")
    for name, digest in smoke_manifests.items():
        _require_sha256(digest, f"frozen harness.smoke_manifest_sha256.{name}")
    executed = _require_object(harness.get("executed_file_sha256"), "frozen harness.executed_file_sha256")
    if set(executed) != set(EXECUTED_FILE_PATHS):
        raise PreflightError("frozen harness executed-file map has unknown or missing paths")
    for name, digest in executed.items():
        _require_sha256(digest, f"frozen harness.executed_file_sha256.{name}")
    return harness


def validate_frozen_smoke_preflight_hashes(result_root: str | os.PathLike[str], harness: dict[str, Any]) -> None:
    root = _private_result_root(result_root)
    hashes = _require_object(harness.get("smoke_preflight_sha256"), "frozen harness.smoke_preflight_sha256")
    if set(hashes) != {"navigation", "screenshot"}:
        raise PreflightError("frozen harness must include both control-smoke preflight hashes")
    for name in ("navigation", "screenshot"):
        path = _artifact_file(root / f"preflight-{name}-smoke.json", root, f"{name} smoke preflight")
        if path.parent != root:
            raise PreflightError(f"{name} smoke preflight must be a direct child of RESULT_ROOT")
        record, _raw = _read_private_json_artifact(path, root, f"{name} smoke preflight")
        record = _validate_preflight_result(record)
        if record.get("phase") != f"{name}-smoke" or record.get("outcome") != "passed" \
                or record.get("errors") != [] \
                or not record.get("checks") \
                or any(check.get("state") != "passed" for check in record["checks"]):
            raise PreflightError(f"{name} smoke preflight does not contain only passed checks")
        if sha256_file(path) != hashes[name]:
            raise PreflightError(f"frozen harness {name} smoke preflight hash differs from the saved artifact")


def _read_private_json_artifact(path: Path, root: Path, where: str) -> tuple[dict[str, Any], bytes]:
    artifact = _artifact_file(path, root, where)
    status = artifact.lstat()
    if not stat.S_ISREG(status.st_mode) or stat.S_IMODE(status.st_mode) != 0o600 \
            or status.st_uid != os.getuid():
        raise PreflightError(f"{where} must be a current-user-owned regular file with mode 0600")
    value, raw = _strict_json(artifact, where)
    return _require_object(value, where), raw


def _validate_smoke_preflight(
    supplied_path: str | os.PathLike[str],
    root: Path,
    *,
    name: str,
    identity_sha: str,
    version_sha: str,
) -> tuple[Path, dict[str, Any]]:
    phase = f"{name}-smoke"
    raw_path = Path(supplied_path).expanduser()
    expected_name = f"preflight-{phase}.json"
    if ".." in raw_path.parts or raw_path.name != expected_name:
        raise PreflightError(f"{phase} preflight must be named {expected_name} without traversal")
    path = _artifact_file(raw_path, root, f"{phase} preflight")
    if path.parent != root:
        raise PreflightError(f"{phase} preflight must be a direct child of RESULT_ROOT")
    result, _raw = _read_private_json_artifact(path, root, f"{phase} preflight")
    result = _validate_preflight_result(result)
    if result.get("phase") != phase or result.get("outcome") != "passed" or result.get("errors") != []:
        raise PreflightError(f"{phase} preflight did not pass")
    if not result.get("checks") or any(check.get("state") != "passed" for check in result["checks"]):
        raise PreflightError(f"{phase} preflight does not contain only passed checks")
    if result.get("candidate_identity_sha256") != identity_sha \
            or result.get("version_manifest_sha256") != version_sha:
        raise PreflightError(f"{phase} preflight does not match the sealed identity and version")
    return path, result


def _validate_smoke_manifest(
    supplied_root: str | os.PathLike[str],
    root: Path,
    *,
    name: str,
    preflight_path: Path,
    identity: dict[str, Any],
    identity_sha: str,
    version_sha: str,
    tool_manifest_sha: str,
) -> tuple[Path, dict[str, Any]]:
    run_root = _private_artifact_directory(supplied_root, root, f"{name} smoke run")
    expected_root = root / "smoke" / name
    if run_root != expected_root:
        raise PreflightError(f"{name} smoke output must be RESULT_ROOT/smoke/{name}")
    manifest_path = _artifact_file(run_root / "run-manifest.json", root, f"{name} smoke manifest")
    if manifest_path != run_root / "run-manifest.json":
        raise PreflightError(f"{name} smoke manifest has an unexpected path")
    manifest, _raw = _read_private_json_artifact(manifest_path, root, f"{name} smoke manifest")
    expected_schema = (
        "openbrowser-latest-navigation-smoke/1"
        if name == "navigation" else "openbrowser-screenshot-smoke/1"
    )
    expected_harness = identity["executed_files"][
        "benchmarks/latest_navigation_baseline.py"
        if name == "navigation" else "benchmarks/screenshot_capability.py"
    ]
    if manifest.get("schema") != expected_schema or manifest.get("mode") != "smoke" \
            or manifest.get("state") != "complete":
        raise PreflightError(f"{name} smoke run is not a complete control-only manifest")
    if manifest.get("candidate_identity_sha256") != identity_sha \
            or manifest.get("version_manifest_sha256") != version_sha:
        raise PreflightError(f"{name} smoke manifest does not match the sealed candidate and version")
    if manifest.get("preflight_sha256") != sha256_file(preflight_path):
        raise PreflightError(f"{name} smoke manifest does not match its smoke preflight")
    if manifest.get("frozen_harness_sha256") is not None:
        raise PreflightError(f"{name} smoke must run before the frozen harness is created")
    if manifest.get("harness_sha256") != expected_harness:
        raise PreflightError(f"{name} smoke runner changed after candidate identity creation")
    cleanup = manifest.get("cleanup")
    fixture = manifest.get("fixture")
    if not isinstance(cleanup, dict) or cleanup.get("ok") is not True \
            or not isinstance(fixture, dict) or fixture.get("stopped") is not True \
            or manifest.get("errors") != []:
        raise PreflightError(f"{name} smoke cleanup or fixture proof is incomplete")
    if name == "navigation":
        host_preflight = manifest.get("host_preflight")
        if not isinstance(host_preflight, dict) or host_preflight.get("state") != "clear":
            raise PreflightError("navigation smoke host preflight is not clear")
    if name == "screenshot":
        if manifest.get("tool_manifest_sha256") != tool_manifest_sha:
            raise PreflightError("screenshot smoke tool manifest does not match the sealed tools")
        if manifest.get("arm_status") != {}:
            raise PreflightError("screenshot smoke must not classify screenshot capability")
    return manifest_path, manifest


def _run_smoke_verifier(
    source: Path,
    python: Path,
    root: Path,
    *,
    name: str,
    run_root: Path,
    identity_path: Path,
    preflight_path: Path,
) -> None:
    script = source / (
        "benchmarks/summarize_latest_navigation.py"
        if name == "navigation" else "benchmarks/screenshot_capability.py"
    )
    command = [
        str(python), str(script), "verify-smoke",
        "--run-root", str(run_root),
        "--candidate-identity", str(identity_path),
        "--preflight", str(preflight_path),
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=120,
            cwd=root, env=_safe_environment(root, python_path=str(source / "src")),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreflightError(f"{name} smoke verifier could not run: {type(error).__name__}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PreflightError(f"{name} smoke verifier failed with exit {result.returncode}: {detail}")


def _write_or_reuse_frozen_harness(
    output: Path,
    root: Path,
    value: dict[str, Any],
    *,
    identity_sha: str,
    candidate_freeze_sha: str,
    browser_use_freeze_sha: str,
    tool_manifest_sha: str,
) -> None:
    expected_bytes = _json_bytes(value)
    if output.is_symlink():
        raise PreflightError("frozen harness must not be a symlink")
    if output.exists():
        existing, existing_raw = _read_private_json_artifact(output, root, "frozen harness")
        _validate_frozen_harness(
            existing,
            identity_sha=identity_sha,
            candidate_freeze_sha=candidate_freeze_sha,
            browser_use_freeze_sha=browser_use_freeze_sha,
            tool_manifest_sha=tool_manifest_sha,
        )
        if existing_raw == expected_bytes:
            return
        raise PreflightError("refusing to replace frozen harness with different inputs")
    _write_new_json(output, value, "frozen harness")


def create_frozen_harness(
    source: str | os.PathLike[str],
    candidate_identity: str | os.PathLike[str],
    version_manifest: str | os.PathLike[str],
    tool_manifest: str | os.PathLike[str],
    candidate_freeze: str | os.PathLike[str],
    browser_use_freeze: str | os.PathLike[str],
    navigation_preflight: str | os.PathLike[str],
    screenshot_preflight: str | os.PathLike[str],
    navigation_smoke: str | os.PathLike[str],
    screenshot_smoke: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    _require_no_aws()
    identity_parent = Path(candidate_identity).expanduser().parent
    root = _private_result_root(identity_parent)
    source_root = _source_root(source, result_root=root)
    identity_path = _artifact_file(candidate_identity, root, "candidate identity")
    version_path = _artifact_file(version_manifest, root, "version manifest")
    tool_manifest_path = _artifact_file(tool_manifest, root, "tool manifest")
    candidate_freeze_path = _artifact_file(candidate_freeze, root, "candidate dependency freeze")
    browser_use_freeze_path = _artifact_file(browser_use_freeze, root, "browser-use dependency freeze")
    if any(path.parent != root for path in (
        identity_path, version_path, tool_manifest_path, candidate_freeze_path, browser_use_freeze_path,
    )):
        raise PreflightError("identity, version, tool, freeze, and frozen-harness files must be direct children of RESULT_ROOT")
    if identity_path.name != "candidate-identity.json" or version_path.name != "version-manifest.json" \
            or tool_manifest_path.name != "tool-manifest.json" \
            or candidate_freeze_path.name != "candidate-freeze.txt" \
            or browser_use_freeze_path.name != "browser-use-freeze.txt":
        raise PreflightError("freeze inputs do not use their exact documented filenames")
    raw_output = Path(output).expanduser()
    if ".." in raw_output.parts or raw_output.name != "frozen-harness.json" \
            or raw_output.parent != root:
        raise PreflightError("frozen harness output must be RESULT_ROOT/frozen-harness.json without traversal")
    output_path = root / "frozen-harness.json"

    identity_value, identity_raw = _read_private_json_artifact(identity_path, root, "candidate identity")
    identity = _validate_identity(identity_value)
    identity_sha = sha256_bytes(identity_raw)
    source_snapshot = _git_snapshot(source_root, result_root=root)
    if any(identity.get(field) != value for field, value in source_snapshot.items()):
        raise PreflightError("candidate source differs from its sealed identity")
    if _hash_executed_files(source_root) != identity["executed_files"]:
        raise PreflightError("benchmark runner or verifier files changed after candidate identity creation")
    candidate_python = _inspect_python(
        identity["candidate_python"]["supplied_path"], root=root, require_current=True,
    )
    browser_python = _inspect_python(identity["browser_use_python"]["supplied_path"], root=root)
    if candidate_python != identity["candidate_python"] or browser_python != identity["browser_use_python"]:
        raise PreflightError("freeze interpreter identity differs from the candidate identity")

    version_value, version_raw = _read_private_json_artifact(version_path, root, "version manifest")
    version = _validate_version_manifest(version_value)
    version_sha = sha256_bytes(version_raw)
    if version["candidate_identity_sha256"] != identity_sha \
            or version["executed_file_sha256"] != identity["executed_files"]:
        raise PreflightError("version manifest does not match the candidate identity or runner files")

    metadata_root = _private_artifact_directory(root / "metadata", root, "metadata directory")
    parsed_metadata = _read_metadata(metadata_root)
    tool_value, tool_raw = _read_private_json_artifact(tool_manifest_path, root, "tool manifest")
    tools_root = _private_artifact_directory(root / "tools", root, "tools directory")
    tools = _validate_tool_manifest(
        tool_value, result_root=root, tools_root=tools_root, verify_installation=True,
    )
    tool_sha = sha256_bytes(tool_raw)
    if tools["metadata_sha256"] != parsed_metadata["metadata_manifest_sha256"] \
            or version["tool_manifest_sha256"] != tool_sha:
        raise PreflightError("tool manifest does not match current saved metadata or version manifest")
    if version["latest_versions"] != parsed_metadata["latest_versions"]:
        raise PreflightError("version manifest does not match the saved official metadata")

    candidate_freeze_sha = sha256_file(candidate_freeze_path)
    browser_use_freeze_sha = sha256_file(browser_use_freeze_path)
    candidate_packages = _parse_freeze(candidate_freeze_path, expected_under=root)
    browser_use_packages = _parse_freeze(browser_use_freeze_path, expected_under=root)
    for name, path, digest in (
        ("candidate", candidate_freeze_path, candidate_freeze_sha),
        ("browser_use", browser_use_freeze_path, browser_use_freeze_sha),
    ):
        record = version["dependency_freezes"][name]
        if record["path"] != str(path) or record["sha256"] != digest:
            raise PreflightError(f"{name} dependency freeze differs from the version manifest")
    installed_versions = {
        **_tool_versions(tools),
        **_package_installation_versions(
            parsed_metadata["latest_versions"], candidate_packages, browser_use_packages,
        ),
    }
    if installed_versions != version["installed_versions"]:
        raise PreflightError("installed tool or package versions differ from the version manifest")

    navigation_preflight_path, _navigation_preflight = _validate_smoke_preflight(
        navigation_preflight, root, name="navigation",
        identity_sha=identity_sha, version_sha=version_sha,
    )
    screenshot_preflight_path, _screenshot_preflight = _validate_smoke_preflight(
        screenshot_preflight, root, name="screenshot",
        identity_sha=identity_sha, version_sha=version_sha,
    )
    smoke_roots = {
        "navigation": _private_artifact_directory(navigation_smoke, root, "navigation smoke run"),
        "screenshot": _private_artifact_directory(screenshot_smoke, root, "screenshot smoke run"),
    }
    expected_roots = {name: root / "smoke" / name for name in ("navigation", "screenshot")}
    if smoke_roots != expected_roots:
        raise PreflightError("smoke runs must use RESULT_ROOT/smoke/navigation and RESULT_ROOT/smoke/screenshot")

    smoke_preflight_paths = {
        "navigation": navigation_preflight_path,
        "screenshot": screenshot_preflight_path,
    }
    smoke_preflight_hashes = {
        name: sha256_file(path) for name, path in smoke_preflight_paths.items()
    }
    smoke_manifest_paths = {}
    for name in ("navigation", "screenshot"):
        _run_smoke_verifier(
            source_root, Path(identity["candidate_python"]["supplied_path"]), root,
            name=name, run_root=smoke_roots[name], identity_path=identity_path,
            preflight_path=smoke_preflight_paths[name],
        )
        manifest_path, _manifest = _validate_smoke_manifest(
            smoke_roots[name], root, name=name,
            preflight_path=smoke_preflight_paths[name], identity=identity,
            identity_sha=identity_sha, version_sha=version_sha,
            tool_manifest_sha=tool_sha,
        )
        smoke_manifest_paths[name] = manifest_path

    value = {
        "schema": FROZEN_HARNESS_SCHEMA,
        "candidate_identity_sha256": identity_sha,
        "version_manifest_sha256": version_sha,
        "tool_manifest_sha256": tool_sha,
        "candidate_freeze_sha256": candidate_freeze_sha,
        "browser_use_freeze_sha256": browser_use_freeze_sha,
        "smoke_preflight_sha256": smoke_preflight_hashes,
        "smoke_manifest_sha256": {
            name: sha256_file(path) for name, path in smoke_manifest_paths.items()
        },
        "executed_file_sha256": identity["executed_files"],
    }
    _validate_frozen_harness(
        value,
        identity_sha=identity_sha,
        candidate_freeze_sha=candidate_freeze_sha,
        browser_use_freeze_sha=browser_use_freeze_sha,
        tool_manifest_sha=tool_sha,
    )
    _write_or_reuse_frozen_harness(
        output_path, root, value,
        identity_sha=identity_sha,
        candidate_freeze_sha=candidate_freeze_sha,
        browser_use_freeze_sha=browser_use_freeze_sha,
        tool_manifest_sha=tool_sha,
    )
    LOGGER.info("wrote or verified immutable frozen-harness.json")
    return value


def create_runtime_variant_identity(
    baseline_source: str | os.PathLike[str],
    treatment_source: str | os.PathLike[str],
    candidate_identity: str | os.PathLike[str],
    frozen_harness: str | os.PathLike[str],
    candidate_freeze: str | os.PathLike[str],
    browser_use_freeze: str | os.PathLike[str],
    tool_manifest: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    _require_no_aws()
    output_path = _canonical_output(output, "runtime-variant identity")
    result_root = _private_result_root(Path(candidate_freeze).expanduser().parent)
    if output_path.parent != result_root and not output_path.parent.is_relative_to(result_root):
        raise PreflightError("runtime-variant identity must be below RESULT_ROOT")
    baseline = _source_root(baseline_source, result_root)
    treatment = _source_root(treatment_source, result_root)
    identity_path = _artifact_file(candidate_identity, result_root, "candidate identity")
    harness_path = _artifact_file(frozen_harness, result_root, "frozen harness")
    candidate_freeze_path = _artifact_file(candidate_freeze, result_root, "candidate dependency freeze")
    browser_freeze_path = _artifact_file(browser_use_freeze, result_root, "browser-use dependency freeze")
    tool_manifest_path = _artifact_file(tool_manifest, result_root, "tool manifest")
    if any(
        path.parent != result_root
        for path in (identity_path, harness_path, candidate_freeze_path, browser_freeze_path, tool_manifest_path)
    ):
        raise PreflightError("runtime-variant inputs must be direct children of RESULT_ROOT")
    identity, identity_raw = _strict_json(identity_path, "candidate identity")
    identity = _validate_identity(identity)
    baseline_snapshot = _git_snapshot(baseline, result_root=result_root)
    for field in ("candidate_commit", "candidate_tree", "runtime_tree"):
        identity_field = "candidate_commit" if field == "candidate_commit" else field
        if baseline_snapshot[identity_field] != identity[identity_field]:
            raise PreflightError(f"baseline source differs from candidate identity at {field}")
    treatment_snapshot = _git_head_snapshot(treatment, result_root=result_root)
    baseline_files = _git_file_map(baseline, "HEAD", result_root=result_root)
    treatment_files = _git_file_map(treatment, "HEAD", result_root=result_root)
    changed_paths = sorted(path for path in set(baseline_files) | set(treatment_files) if baseline_files.get(path) != treatment_files.get(path))
    if not changed_paths:
        raise PreflightError("runtime treatment has no source changes")
    non_runtime_changes = [path for path in changed_paths if not path.startswith("src/openbrowser/")]
    if non_runtime_changes:
        raise PreflightError(f"runtime treatment changes paths outside src/openbrowser/: {non_runtime_changes}")
    if treatment_snapshot["runtime_tree"] == baseline_snapshot["runtime_tree"]:
        raise PreflightError("runtime treatment did not change the src/openbrowser tree")
    candidate_freeze_sha = sha256_file(candidate_freeze_path)
    browser_freeze_sha = sha256_file(browser_freeze_path)
    tool_manifest_value, tool_manifest_raw = _strict_json(tool_manifest_path, "tool manifest")
    tools_root = _private_artifact_directory(result_root / "tools", result_root, "tools directory")
    _validate_tool_manifest(tool_manifest_value, result_root=result_root, tools_root=tools_root, verify_installation=True)
    tool_manifest_sha = sha256_bytes(tool_manifest_raw)
    harness_value, _harness_raw = _strict_json(harness_path, "frozen harness")
    harness = _validate_frozen_harness(
        harness_value,
        identity_sha=sha256_bytes(identity_raw),
        candidate_freeze_sha=candidate_freeze_sha,
        browser_use_freeze_sha=browser_freeze_sha,
        tool_manifest_sha=tool_manifest_sha,
    )
    validate_frozen_smoke_preflight_hashes(result_root, harness)
    version_path = _artifact_file(result_root / "version-manifest.json", result_root, "version manifest")
    version_value, version_raw = _strict_json(version_path, "version manifest")
    _validate_version_manifest(version_value)
    if harness["version_manifest_sha256"] != sha256_bytes(version_raw):
        raise PreflightError("frozen harness version-manifest hash differs from the sealed version manifest")
    if harness["executed_file_sha256"] != identity["executed_files"]:
        raise PreflightError("frozen harness runner/verifier hashes differ from candidate identity")
    for relative_path, expected in identity["executed_files"].items():
        if treatment_files.get(relative_path) != expected:
            raise PreflightError(f"runtime treatment changed benchmark runner/verifier {relative_path}")
    variant = {
        "schema": RUNTIME_VARIANT_SCHEMA,
        "baseline": {
            "commit": baseline_snapshot["candidate_commit"],
            "tree": baseline_snapshot["candidate_tree"],
            "runtime_tree": baseline_snapshot["runtime_tree"],
        },
        "treatment": treatment_snapshot,
        "changed_paths": changed_paths,
        "shared_harness_sha256": sha256_file(harness_path),
        "candidate_freeze_sha256": candidate_freeze_sha,
        "browser_use_freeze_sha256": browser_freeze_sha,
        "tool_manifest_sha256": tool_manifest_sha,
    }
    _validate_runtime_variant(variant)
    _write_new_json(output_path, variant, "runtime-variant identity")
    LOGGER.info("wrote runtime-variant identity with only OpenBrowser runtime changes")
    return variant


def _metadata_freshness(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_versions": parsed["latest_versions"],
        "release_digests": {
            "lightpanda": parsed["release_assets"]["lightpanda"]["official_sha256"],
            "agent-browser": parsed["release_assets"]["agent-browser"]["official_sha256"],
        },
        "chrome_download_url": parsed["chrome_download_url"],
    }


def _frozen_metadata_state(version_manifest: dict[str, Any], tool_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_versions": version_manifest["latest_versions"],
        "release_digests": {
            "lightpanda": tool_manifest["tools"]["lightpanda"]["official_sha256"],
            "agent-browser": tool_manifest["tools"]["agent-browser"]["official_sha256"],
        },
        "chrome_download_url": tool_manifest["tools"]["chrome_for_testing"]["download_url"],
    }


def _check_freeze(
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    name: str,
    supplied_path: str | os.PathLike[str],
    record: dict[str, str],
    root: Path,
) -> dict[str, str] | None:
    try:
        path = _artifact_file(supplied_path, root, f"{name} dependency freeze")
        actual_hash = sha256_file(path)
        expected_path = Path(record["path"])
        _check_equal(checks, errors, f"{name}_freeze_path", str(expected_path), str(path))
        _check_equal(checks, errors, f"{name}_freeze_sha256", record["sha256"], actual_hash)
        packages = _parse_freeze(path, expected_under=root)
        return packages
    except PreflightError as exc:
        _record_check(checks, errors, f"{name}_freeze", "valid frozen dependency file", str(exc), False)
        return None


def _phase_output(output: str | os.PathLike[str], root: Path, phase: str) -> Path:
    raw = Path(output).expanduser()
    if ".." in raw.parts:
        raise PreflightError("preflight artifact path must not contain parent traversal")
    expected_name = f"preflight-{phase}.json"
    if raw.name != expected_name:
        raise PreflightError(f"preflight artifact for phase {phase} must be named {expected_name}")
    output_path = _canonical_output(raw, "preflight artifact")
    parent = output_path.parent
    if parent != root and not parent.is_relative_to(root):
        raise PreflightError("preflight artifact must be below RESULT_ROOT")
    if parent != root:
        status = parent.stat()
        if stat.S_IMODE(status.st_mode) != 0o700 or status.st_uid != os.getuid():
            raise PreflightError("preflight artifact parent must be owned by the current user with mode 0700")
    return output_path


def _run_verify_source_import(source: Path, python: dict[str, str], root: Path, identity_sha: str) -> dict[str, Any]:
    import_root = root / "import-probe"
    if import_root.is_symlink() or not import_root.is_dir():
        raise PreflightError("RESULT_ROOT/import-probe must exist as a private directory")
    _private_artifact_directory(import_root, root, "import-probe directory")
    if stat.S_IMODE(import_root.stat().st_mode) != 0o700:
        raise PreflightError("RESULT_ROOT/import-probe must have mode 0700")
    temporary = Path(tempfile.mkdtemp(prefix="verify-", dir=import_root))
    try:
        temporary.chmod(0o700)
        report = _source_import_report(source, python, temporary, identity_sha, result_root=root)
        _validate_source_import_proof(report)
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def verify_preflight(
    *,
    source: str | os.PathLike[str],
    python: str | os.PathLike[str],
    browser_use_python: str | os.PathLike[str],
    metadata: str | os.PathLike[str],
    tools: str | os.PathLike[str],
    candidate_freeze: str | os.PathLike[str],
    browser_use_freeze: str | os.PathLike[str],
    tool_manifest: str | os.PathLike[str],
    identity: str | os.PathLike[str],
    version_manifest: str | os.PathLike[str],
    phase: str,
    output: str | os.PathLike[str],
    metadata_check: str | os.PathLike[str] | None = None,
    runtime_variant_identity: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(phase, str) or not PHASE_RE.fullmatch(phase):
        raise PreflightError("phase must be a lowercase slug without path separators")
    root = _private_result_root(Path(candidate_freeze).expanduser().parent)
    output_path = _phase_output(output, root, phase)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    _require_no_aws_value = _check_aws_environment()
    _record_check(
        checks,
        errors,
        "no_aws_environment",
        {"unexpected_names": [], "incorrect_safety_names": []},
        _require_no_aws_value,
        not _require_no_aws_value["unexpected_names"] and not _require_no_aws_value["incorrect_safety_names"],
    )

    identity_path: Path | None = None
    version_path: Path | None = None
    identity_raw = b""
    version_raw = b""
    identity_value: dict[str, Any] | None = None
    version_value: dict[str, Any] | None = None
    try:
        identity_path = _artifact_file(identity, root, "candidate identity")
        identity_obj, identity_raw = _strict_json(identity_path, "candidate identity")
        identity_value = _validate_identity(identity_obj)
        _record_check(checks, errors, "candidate_identity_manifest", "exact known schema", "valid", True)
    except PreflightError as exc:
        _record_check(checks, errors, "candidate_identity_manifest", "exact known schema", str(exc), False)
    try:
        version_path = _artifact_file(version_manifest, root, "version manifest")
        version_obj, version_raw = _strict_json(version_path, "version manifest")
        version_value = _validate_version_manifest(version_obj)
        _record_check(checks, errors, "version_manifest", "exact known schema", "valid", True)
    except PreflightError as exc:
        _record_check(checks, errors, "version_manifest", "exact known schema", str(exc), False)

    if identity_value is not None and identity_raw:
        identity_sha = sha256_bytes(identity_raw)
        if version_value is not None:
            _check_equal(checks, errors, "candidate_identity_sha256", version_value["candidate_identity_sha256"], identity_sha)
            _check_equal(checks, errors, "version_executed_file_sha256", identity_value["executed_files"], version_value["executed_file_sha256"])
    else:
        identity_sha = sha256_bytes(identity_raw) if identity_raw else None
    version_sha = sha256_bytes(version_raw) if version_raw else None

    source_root: Path | None = None
    source_snapshot: dict[str, str] | None = None
    runtime_variant_value: dict[str, Any] | None = None
    try:
        source_root = _source_root(source, root)
        if identity_value is None:
            raise PreflightError("candidate identity is invalid")
        if runtime_variant_identity is None:
            source_snapshot = _git_snapshot(source_root, result_root=root)
            expected_source = {field: identity_value[field] for field in (
                "candidate_commit",
                "candidate_tree",
                "runtime_tree",
                "release_runtime_tree",
                "origin_main_commit",
                "origin_main_runtime_tree",
            )}
            _check_equal(checks, errors, "candidate_source_identity", expected_source, source_snapshot)
            if expected_source != source_snapshot:
                raise PreflightError("candidate source differs from sealed candidate identity")
        else:
            source_snapshot = _git_head_snapshot(source_root, result_root=root)
            variant_path = _artifact_file(runtime_variant_identity, root, "runtime-variant identity")
            variant, variant_errors = _verify_runtime_variant(variant_path, identity_value, identity_raw, source_root, root, None)
            runtime_variant_value = variant
            _record_check(
                checks,
                errors,
                "runtime_variant_identity",
                "clean src/openbrowser-only treatment matching the baseline identity",
                "valid" if variant is not None else variant_errors,
                variant is not None,
            )
        actual_executed_files = _hash_executed_files(source_root)
        _check_equal(
            checks,
            errors,
            "executed_file_sha256",
            identity_value["executed_files"],
            actual_executed_files,
        )
        source_commit = source_snapshot.get("candidate_commit", source_snapshot.get("commit", "unknown"))
        _record_check(checks, errors, "candidate_source_clean", "clean Git checkout", source_commit, True)
    except PreflightError as exc:
        _record_check(checks, errors, "candidate_source_identity", "clean source matching sealed identity", str(exc), False)

    candidate_python: dict[str, str] | None = None
    browser_python: dict[str, str] | None = None
    candidate_python = _inspect_python_for_check(
        checks, errors, "candidate_python", python, root=root, require_current=True
    )
    browser_python = _inspect_python_for_check(
        checks, errors, "browser_use_python", browser_use_python, root=root, require_current=False
    )
    if identity_value is not None and candidate_python is not None:
        expected = identity_value["candidate_python"]
        _check_equal(checks, errors, "candidate_python_identity", expected, candidate_python)
    if identity_value is not None and browser_python is not None:
        _check_equal(checks, errors, "browser_use_python_identity", identity_value["browser_use_python"], browser_python)

    tool_manifest_value: dict[str, Any] | None = None
    tool_manifest_raw = b""
    try:
        tools_root = _private_artifact_directory(tools, root, "tools directory")
        if tools_root != root / "tools":
            raise PreflightError("tools path must be RESULT_ROOT/tools")
        tool_manifest_path = _artifact_file(tool_manifest, root, "tool manifest")
        tool_manifest_obj, tool_manifest_raw = _strict_json(tool_manifest_path, "tool manifest")
        tool_manifest_value = _validate_tool_manifest(tool_manifest_obj, result_root=root, tools_root=tools_root, verify_installation=False)
        if version_value is not None:
            _check_equal(
                checks,
                errors,
                "tool_manifest_sha256",
                version_value["tool_manifest_sha256"],
                sha256_bytes(tool_manifest_raw),
            )
        _record_check(checks, errors, "tool_manifest_schema", "exact known schema", "valid", True)
    except PreflightError as exc:
        _record_check(checks, errors, "tool_manifest_schema", "exact known schema", str(exc), False)
    _tool_installation_for_check(checks, errors, tool_manifest_value, root)

    if runtime_variant_identity is not None:
        try:
            if runtime_variant_value is None:
                raise PreflightError("runtime-variant identity is invalid")
            candidate_freeze_path = _artifact_file(candidate_freeze, root, "candidate dependency freeze")
            browser_use_freeze_path = _artifact_file(browser_use_freeze, root, "browser-use dependency freeze")
            tool_manifest_path = _artifact_file(tool_manifest, root, "tool manifest")
            harness_path = _artifact_file(root / "frozen-harness.json", root, "frozen harness")
            candidate_hash = sha256_file(candidate_freeze_path)
            browser_hash = sha256_file(browser_use_freeze_path)
            tool_hash = sha256_file(tool_manifest_path)
            if runtime_variant_value["candidate_freeze_sha256"] != candidate_hash:
                raise PreflightError("runtime-variant candidate freeze hash changed")
            if runtime_variant_value["browser_use_freeze_sha256"] != browser_hash:
                raise PreflightError("runtime-variant browser-use freeze hash changed")
            if runtime_variant_value["tool_manifest_sha256"] != tool_hash:
                raise PreflightError("runtime-variant tool manifest hash changed")
            harness_obj, harness_raw = _strict_json(harness_path, "frozen harness")
            harness = _validate_frozen_harness(
                harness_obj,
                identity_sha=sha256_bytes(identity_raw),
                candidate_freeze_sha=candidate_hash,
                browser_use_freeze_sha=browser_hash,
                tool_manifest_sha=tool_hash,
            )
            version_path = _artifact_file(root / "version-manifest.json", root, "version manifest")
            version_obj, version_raw = _strict_json(version_path, "version manifest")
            _validate_version_manifest(version_obj)
            if harness["version_manifest_sha256"] != sha256_bytes(version_raw):
                raise PreflightError("frozen harness version-manifest hash differs from the sealed version manifest")
            if runtime_variant_value["shared_harness_sha256"] != sha256_bytes(harness_raw):
                raise PreflightError("runtime-variant shared harness hash changed")
            _record_check(checks, errors, "runtime_variant_shared_inputs", "frozen harness, dependency freezes, and tools unchanged", "valid", True)
        except PreflightError as exc:
            _record_check(checks, errors, "runtime_variant_shared_inputs", "frozen harness, dependency freezes, and tools unchanged", str(exc), False)

    parsed_metadata: dict[str, Any] | None = None
    try:
        metadata_root = _private_artifact_directory(metadata, root, "metadata directory")
        if metadata_root != root / "metadata":
            raise PreflightError("metadata path must be RESULT_ROOT/metadata")
        parsed_metadata = _read_metadata(metadata_root)
        _record_check(checks, errors, "saved_metadata", "six validated official source responses", "valid", True)
    except PreflightError as exc:
        _record_check(checks, errors, "saved_metadata", "six validated official source responses", str(exc), False)
    if parsed_metadata is not None and version_value is not None:
        _check_equal(checks, errors, "latest_versions", version_value["latest_versions"], parsed_metadata["latest_versions"])
        _check_equal(checks, errors, "metadata_sources", version_value["metadata_sources"], parsed_metadata["metadata_sources"])
    if tool_manifest_value is not None and parsed_metadata is not None:
        try:
            _check_equal(
                checks,
                errors,
                "tool_metadata_manifest_sha256",
                parsed_metadata["metadata_manifest_sha256"],
                tool_manifest_value["metadata_sha256"],
            )
            _check_equal(
                checks,
                errors,
                "installed_tool_versions",
                _latest_expected_tool_versions(parsed_metadata),
                _tool_versions(tool_manifest_value),
            )
            for name in ("lightpanda", "agent-browser"):
                _check_equal(
                    checks,
                    errors,
                    f"{name}_official_digest",
                    parsed_metadata["release_assets"][name]["official_sha256"],
                    tool_manifest_value["tools"][name]["official_sha256"],
                )
            _check_equal(
                checks,
                errors,
                "chrome_stable_download_url",
                parsed_metadata["chrome_download_url"],
                tool_manifest_value["tools"]["chrome_for_testing"]["download_url"],
            )
        except (KeyError, PreflightError) as exc:
            _record_check(checks, errors, "tool_metadata_binding", "tool versions and official digests match", str(exc), False)

    candidate_packages = None
    browser_packages = None
    if version_value is not None:
        candidate_packages = _check_freeze(
            checks,
            errors,
            name="candidate",
            supplied_path=candidate_freeze,
            record=version_value["dependency_freezes"]["candidate"],
            root=root,
        )
        browser_packages = _check_freeze(
            checks,
            errors,
            name="browser_use",
            supplied_path=browser_use_freeze,
            record=version_value["dependency_freezes"]["browser_use"],
            root=root,
        )
        if candidate_packages is not None and browser_packages is not None:
            try:
                installed = _package_installation_versions(
                    version_value["latest_versions"], candidate_packages, browser_packages
                )
                _check_equal(checks, errors, "installed_package_versions", version_value["installed_versions"], {
                    **(_tool_versions(tool_manifest_value) if tool_manifest_value is not None else {}),
                    **installed,
                })
            except PreflightError as exc:
                _record_check(checks, errors, "installed_package_versions", "latest browser-use packages", str(exc), False)

    if metadata_check is not None and version_value is not None and tool_manifest_value is not None:
        try:
            metadata_check_root = _private_artifact_directory(metadata_check, root, "metadata-check directory")
            parsed_check = _read_metadata(metadata_check_root)
            expected_state = _frozen_metadata_state(version_value, tool_manifest_value)
            actual_state = _metadata_freshness(parsed_check)
            _check_equal(checks, errors, "latest_metadata_freshness", expected_state, actual_state)
        except PreflightError as exc:
            _record_check(checks, errors, "latest_metadata_freshness", "versions, official release digests, and Chrome URL unchanged", str(exc), False)
    elif metadata_check is not None:
        _record_check(checks, errors, "latest_metadata_freshness", "sealed versions available", "version manifest unavailable", False)

    if source_root is not None and candidate_python is not None and identity_sha is not None:
        try:
            report = _run_verify_source_import(source_root, candidate_python, root, identity_sha)
            _record_check(checks, errors, "source_import", "selected source imported from isolated private CWD", "passed", True)
        except PreflightError as exc:
            _record_check(checks, errors, "source_import", "selected source imported from isolated private CWD", str(exc), False)

    result = {
        "schema": PREFLIGHT_SCHEMA,
        "phase": phase,
        "candidate_identity_sha256": identity_sha,
        "version_manifest_sha256": version_sha,
        "checks": checks,
        "outcome": "passed" if not errors else "failed",
        "errors": errors,
    }
    _validate_preflight_result(result)
    _write_new_json(output_path, result, "preflight artifact")
    LOGGER.info("wrote %s preflight evidence for phase %s", result["outcome"], phase)
    return result


def _validate_preflight_result(value: Any) -> dict[str, Any]:
    result = _require_object(value, "preflight artifact")
    _require_exact_keys(result, PREFLIGHT_KEYS, "preflight artifact")
    if result.get("schema") != PREFLIGHT_SCHEMA:
        raise PreflightError("preflight artifact schema is unsupported")
    phase = _require_nonempty_string(result.get("phase"), "preflight artifact.phase")
    if not PHASE_RE.fullmatch(phase):
        raise PreflightError("preflight artifact.phase is unsafe")
    for field in ("candidate_identity_sha256", "version_manifest_sha256"):
        value_hash = result.get(field)
        if value_hash is not None:
            _require_sha256(value_hash, f"preflight artifact.{field}")
    checks = result.get("checks")
    if not isinstance(checks, list):
        raise PreflightError("preflight artifact.checks must be a list")
    check_names = set()
    for index, raw_check in enumerate(checks):
        check = _require_object(raw_check, f"preflight artifact.checks[{index}]")
        _require_exact_keys(check, PREFLIGHT_CHECK_KEYS, f"preflight artifact.checks[{index}]")
        name = _require_nonempty_string(check.get("name"), f"preflight artifact.checks[{index}].name")
        if name in check_names:
            raise PreflightError(f'preflight artifact contains duplicate check name: {name}')
        check_names.add(name)
        if check.get("state") not in {"passed", "failed"}:
            raise PreflightError("preflight check state must be passed or failed")
    errors = result.get("errors")
    if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
        raise PreflightError("preflight artifact.errors must be a string list")
    expected_outcome = "passed" if not errors else "failed"
    if result.get("outcome") != expected_outcome:
        raise PreflightError("preflight artifact outcome does not agree with its errors")
    return result


def _phase_identity_check(source: Path, identity_path: Path, root: Path) -> tuple[dict[str, Any], bytes]:
    identity, identity_raw = _strict_json(identity_path, "candidate identity")
    identity = _validate_identity(identity)
    snapshot = _git_snapshot(source, result_root=root)
    for field, actual in snapshot.items():
        if identity[field] != actual:
            raise PreflightError(f"candidate source differs from identity field {field}")
    return identity, identity_raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    commands = parser.add_subparsers(dest="command", required=True)

    identity = commands.add_parser("identity", help="create candidate and interpreter identity")
    identity.add_argument("--source", required=True, type=Path)
    identity.add_argument("--python", required=True, type=Path)
    identity.add_argument("--browser-use-python", required=True, type=Path)
    identity.add_argument("--candidate-freeze", required=True, type=Path)
    identity.add_argument("--browser-use-freeze", required=True, type=Path)
    identity.add_argument("--tool-manifest", required=True, type=Path)
    identity.add_argument("--output", required=True, type=Path)

    import_proof = commands.add_parser("source-import-proof", help="prove the selected source imports from a private empty CWD")
    import_proof.add_argument("--source", required=True, type=Path)
    import_proof.add_argument("--python", required=True, type=Path)
    import_proof.add_argument("--cwd", required=True, type=Path)
    import_proof.add_argument("--identity", required=True, type=Path)
    import_proof.add_argument("--output", required=True, type=Path)

    seal = commands.add_parser("seal", help="write immutable current version metadata")
    seal.add_argument("--source", required=True, type=Path)
    seal.add_argument("--python", required=True, type=Path)
    seal.add_argument("--browser-use-python", required=True, type=Path)
    seal.add_argument("--metadata", required=True, type=Path)
    seal.add_argument("--tools", required=True, type=Path)
    seal.add_argument("--candidate-freeze", required=True, type=Path)
    seal.add_argument("--browser-use-freeze", required=True, type=Path)
    seal.add_argument("--tool-manifest", required=True, type=Path)
    seal.add_argument("--identity", required=True, type=Path)
    seal.add_argument("--source-import-proof", required=True, type=Path)
    seal.add_argument("--version-manifest", "--output", dest="output", required=True, type=Path)

    verify = commands.add_parser("verify", help="verify frozen identities before a run phase")
    verify.add_argument("--source", required=True, type=Path)
    verify.add_argument("--python", required=True, type=Path)
    verify.add_argument("--browser-use-python", required=True, type=Path)
    verify.add_argument("--metadata", required=True, type=Path)
    verify.add_argument("--metadata-check", type=Path)
    verify.add_argument("--tools", required=True, type=Path)
    verify.add_argument("--candidate-freeze", required=True, type=Path)
    verify.add_argument("--browser-use-freeze", required=True, type=Path)
    verify.add_argument("--tool-manifest", required=True, type=Path)
    verify.add_argument("--identity", required=True, type=Path)
    verify.add_argument("--runtime-variant-identity", type=Path)
    verify.add_argument("--version-manifest", required=True, type=Path)
    verify.add_argument("--phase", required=True)
    verify.add_argument("--output", required=True, type=Path)

    freeze = commands.add_parser("freeze-harness", help="verify both control smokes and freeze their sealed inputs")
    freeze.add_argument("--source", required=True, type=Path)
    freeze.add_argument("--identity", required=True, type=Path)
    freeze.add_argument("--version-manifest", required=True, type=Path)
    freeze.add_argument("--tool-manifest", required=True, type=Path)
    freeze.add_argument("--candidate-freeze", required=True, type=Path)
    freeze.add_argument("--browser-use-freeze", required=True, type=Path)
    freeze.add_argument("--navigation-preflight", required=True, type=Path)
    freeze.add_argument("--screenshot-preflight", required=True, type=Path)
    freeze.add_argument("--navigation-smoke", required=True, type=Path)
    freeze.add_argument("--screenshot-smoke", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)

    variant = commands.add_parser("runtime-variant", help="seal a treatment limited to OpenBrowser runtime files")
    variant.add_argument("--baseline-source", required=True, type=Path)
    variant.add_argument("--treatment-source", required=True, type=Path)
    variant.add_argument("--candidate-identity", required=True, type=Path)
    variant.add_argument("--frozen-harness", required=True, type=Path)
    variant.add_argument("--candidate-freeze", required=True, type=Path)
    variant.add_argument("--browser-use-freeze", required=True, type=Path)
    variant.add_argument("--tool-manifest", required=True, type=Path)
    variant.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "identity":
            create_identity(
                args.source,
                args.python,
                args.browser_use_python,
                args.candidate_freeze,
                args.browser_use_freeze,
                args.tool_manifest,
                args.output,
            )
            return 0
        if args.command == "source-import-proof":
            create_source_import_proof(args.source, args.python, args.cwd, args.identity, args.output)
            return 0
        if args.command == "seal":
            seal_version_manifest(
                args.source,
                args.python,
                args.browser_use_python,
                args.metadata,
                args.tools,
                args.candidate_freeze,
                args.browser_use_freeze,
                args.tool_manifest,
                args.identity,
                args.source_import_proof,
                args.output,
            )
            return 0
        if args.command == "verify":
            result = verify_preflight(
                source=args.source,
                python=args.python,
                browser_use_python=args.browser_use_python,
                metadata=args.metadata,
                metadata_check=args.metadata_check,
                tools=args.tools,
                candidate_freeze=args.candidate_freeze,
                browser_use_freeze=args.browser_use_freeze,
                tool_manifest=args.tool_manifest,
                identity=args.identity,
                runtime_variant_identity=args.runtime_variant_identity,
                version_manifest=args.version_manifest,
                phase=args.phase,
                output=args.output,
            )
            return 0 if result["outcome"] == "passed" else 1
        if args.command == "freeze-harness":
            create_frozen_harness(
                args.source,
                args.identity,
                args.version_manifest,
                args.tool_manifest,
                args.candidate_freeze,
                args.browser_use_freeze,
                args.navigation_preflight,
                args.screenshot_preflight,
                args.navigation_smoke,
                args.screenshot_smoke,
                args.output,
            )
            return 0
        if args.command == "runtime-variant":
            create_runtime_variant_identity(
                args.baseline_source,
                args.treatment_source,
                args.candidate_identity,
                args.frozen_harness,
                args.candidate_freeze,
                args.browser_use_freeze,
                args.tool_manifest,
                args.output,
            )
            return 0
    except (PreflightError, _metadata.MetadataError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.error("unsupported command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
