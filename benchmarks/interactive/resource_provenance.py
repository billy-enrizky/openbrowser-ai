"""Bind an installed console launcher to the application source being compared.

The probe reads import specifications and distribution metadata. It never imports
OpenBrowser or invokes the console entry point. Public results contain digests and
closed reason codes, not local installation or checkout paths.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path, PurePosixPath

_PROBE = """
import sys
sys.path[0] = sys.argv[1]
import importlib.metadata
import importlib.util
import json
try:
    if any(name == "openbrowser" or name.startswith("openbrowser.") for name in sys.modules):
        result = {"error": "application_already_imported"}
    else:
        spec = importlib.util.find_spec("openbrowser")
        dist = importlib.metadata.distribution("openbrowser-ai")
        result = {
            "origin": spec.origin if spec is not None else None,
            "locations": list(spec.submodule_search_locations or []) if spec is not None else [],
            "dist_name": dist.metadata.get("Name"),
            "dist_version": dist.version,
            "dist_location": str(dist.locate_file("")),
        }
except Exception:
    result = {"error": "probe_failed"}
print(json.dumps(result))
"""

_NATIVE_PROBE = r"""
import importlib.metadata
import importlib.util
import json
import sys
sys.path[0] = sys.argv[1]
try:
    if any(name == "openbrowser" or name.startswith("openbrowser.") for name in sys.modules):
        result = {"error": "application_already_imported"}
    else:
        spec = importlib.util.find_spec("openbrowser")
        dist = importlib.metadata.distribution("openbrowser-ai")
        manifest = None
        manifest_record_hash = None
        manifest_record_size = None
        for path in dist.files or []:
            text = str(path).replace("\\", "/")
            if text.endswith("extra_metadata/openbrowser-native.json"):
                manifest = dist.read_text("extra_metadata/openbrowser-native.json")
                record = dist.read_text("RECORD") or ""
                for line in record.splitlines():
                    parts = line.split(",")
                    if (
                        len(parts) >= 3
                        and parts[0].replace("\\", "/").endswith(
                            "extra_metadata/openbrowser-native.json"
                        )
                    ):
                        manifest_record_hash = parts[1]
                        manifest_record_size = parts[2]
                        break
                break
        result = {
            "origin": spec.origin if spec is not None else None,
            "locations": list(spec.submodule_search_locations or []) if spec is not None else [],
            "dist_name": dist.metadata.get("Name"),
            "dist_version": dist.version,
            "dist_location": str(dist.locate_file("")),
            "manifest": manifest,
            "manifest_record_hash": manifest_record_hash,
            "manifest_record_size": manifest_record_size,
        }
except Exception:
    result = {"error": "probe_failed"}
print(json.dumps(result))
"""

_SIMPLE_ENTRY = """
import sys
from openbrowser.cli_entry import cli
if __name__ == "__main__":
    sys.exit(cli())
"""
_UV_ENTRY = """
import sys
from openbrowser.cli_entry import cli
if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(cli())
"""
_PIP_ENTRY = """
import re
import sys
from openbrowser.cli_entry import cli
if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\\.pyw|\\.exe)?$", "", sys.argv[0])
    sys.exit(cli())
"""
_ENTRY_TREES = frozenset(ast.dump(ast.parse(body)) for body in (_SIMPLE_ENTRY, _UV_ENTRY, _PIP_ENTRY))


def application_digest(package_dir: Path) -> str:
    """Hash package files, including runtime data, independently of their root.

    Python bytecode caches are ignored. The caller supplies the application
    package, not the repository, so harnesses and benchmark results are outside
    this digest. Symlinked content is unsupported rather than silently omitted.
    """
    package_dir = package_dir.resolve(strict=True)
    if not package_dir.is_dir() or not (package_dir / "__init__.py").is_file():
        raise ValueError("application_package_required")
    digest = hashlib.sha256(b"openbrowser-application-v1\0")
    for path in sorted(package_dir.rglob("*"), key=lambda item: item.relative_to(package_dir).as_posix()):
        relative = path.relative_to(package_dir)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("symlinked_package_content")
        if not path.is_file():
            continue
        name = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _launcher_interpreter(content: bytes) -> str | None:
    try:
        source = content.decode("utf-8")
        first_line = source.splitlines()[0]
        if not first_line.startswith("#!") or ast.dump(ast.parse(source)) not in _ENTRY_TREES:
            return None
        interpreter = first_line[2:].strip()
        path = Path(interpreter)
        if (
            not path.is_absolute()
            or re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", path.name) is None
            or any(character.isspace() for character in interpreter)
            or not path.is_file()
            or not os.access(path, os.X_OK)
        ):
            return None
        # Do not resolve a venv's interpreter symlink: its invocation path selects the venv.
        return interpreter
    except (IndexError, UnicodeError, SyntaxError, OSError, ValueError):
        return None


def bind_openbrowser(binary: str, expected_package: Path, *, env: dict[str, str], cwd: Path) -> dict[str, object]:
    """Fail closed unless the launcher resolves to the exact or identical app.

    ``env`` is the adapter's environment. PYTHONPATH is removed and the probe's
    initial import directory is the resolved launcher directory, as for executing
    that script, not the benchmark's current directory. This binds app content,
    not dependencies, interpreter contents, or a future race-free execution.
    """
    result: dict[str, object] = {
        "verified": False,
        "reason": "binding_failed",
        "entry_sha256": None,
        "expected_app_digest": None,
        "actual_app_digest": None,
        "dist_version": None,
        "origin_matches_expected": False,
    }
    try:
        expected_package = expected_package.resolve(strict=True)
        result["expected_app_digest"] = application_digest(expected_package)
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "expected_package_unreadable"
        return result

    try:
        if os.path.dirname(binary):
            entry = Path(binary)
            if not entry.is_absolute():
                entry = cwd / entry
        else:
            search_path = os.pathsep.join(
                str(Path(part) if Path(part).is_absolute() else cwd / part)
                for part in env.get("PATH", os.defpath).split(os.pathsep)
            )
            found = shutil.which(binary, path=search_path)
            if found is None:
                result["reason"] = "entry_not_found"
                return result
            entry = Path(found)
            if not entry.is_absolute():
                entry = cwd / entry
        entry = entry.resolve(strict=True)
        content = entry.read_bytes()
        result["entry_sha256"] = hashlib.sha256(content).hexdigest()
        interpreter = _launcher_interpreter(content)
        if interpreter is None or not os.access(entry, os.X_OK):
            result["reason"] = "unsupported_launcher"
            return result
    except (OSError, RuntimeError, ValueError):
        result["reason"] = "entry_unreadable"
        return result

    probe_env = {key: value for key, value in env.items() if key != "PYTHONPATH"}
    try:
        probe = subprocess.run(
            [interpreter, "-c", _PROBE, str(entry.parent)],
            env=probe_env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode != 0:
            result["reason"] = "probe_failed"
            return result
        payload = json.loads(probe.stdout)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        result["reason"] = "probe_failed"
        return result
    if not isinstance(payload, dict) or "error" in payload:
        result["reason"] = "probe_invalid"
        return result

    version = payload.get("dist_version")
    if (
        payload.get("dist_name") != "openbrowser-ai"
        or not isinstance(version, str)
        or re.fullmatch(r"[0-9][A-Za-z0-9.+!_-]{0,127}", version) is None
        or not isinstance(payload.get("dist_location"), str)
    ):
        result["reason"] = "distribution_metadata_invalid"
        return result
    result["dist_version"] = version
    origin, locations = payload.get("origin"), payload.get("locations")
    if (
        not isinstance(origin, str)
        or not isinstance(locations, list)
        or len(locations) != 1
        or not isinstance(locations[0], str)
    ):
        result["reason"] = "ambiguous_package"
        return result
    try:
        origin_path = Path(origin).resolve(strict=True)
        actual_package = Path(locations[0]).resolve(strict=True)
        if origin_path.name != "__init__.py" or origin_path.parent != actual_package:
            result["reason"] = "inconsistent_package_spec"
            return result
        result["origin_matches_expected"] = actual_package == expected_package
        result["actual_app_digest"] = application_digest(actual_package)
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "actual_package_unreadable"
        return result
    if result["actual_app_digest"] != result["expected_app_digest"]:
        result["reason"] = "application_digest_mismatch"
        return result
    result["verified"] = True
    result["reason"] = "exact_package" if result["origin_matches_expected"] else "byte_identical_package"
    return result


_NATIVE_ENTRY = "openbrowser-ai"
_NATIVE_BACKEND = "openbrowser-ai-py"
_NATIVE_MANIFEST_KEYS = {
    "schema_version",
    "frontend",
    "entry_name",
    "backend_name",
    "package_version",
    "target",
    "compiler",
    "source_files",
    "entry_sha256",
}
_SUPPORTED_NATIVE_TARGETS = {
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "aarch64-unknown-linux-musl",
    "x86_64-unknown-linux-musl",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9][A-Za-z0-9.+!_-]{0,127}\Z")


def _host_target() -> str | None:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine in {"arm64", "aarch64"}:
            return "aarch64-apple-darwin"
        if machine in {"x86_64", "amd64"}:
            return "x86_64-apple-darwin"
    elif system == "Linux":
        if machine in {"x86_64", "amd64"}:
            return "x86_64-unknown-linux-musl"
        if machine in {"arm64", "aarch64"}:
            return "aarch64-unknown-linux-musl"
    return None


def _native_binary_format(data: bytes) -> str | None:
    if data[:4] == b"\xcf\xfa\xed\xfe":
        return "macho"
    if data[:4] == b"\x7fELF":
        return "elf"
    return None


def _native_binary_arch(data: bytes) -> str | None:
    if data[:4] == b"\xcf\xfa\xed\xfe":
        if len(data) < 16:
            return None
        cpu, _, file_type = struct.unpack_from("<III", data, 4)
        if file_type != 2:
            return None
        if cpu == 0x0100000C:
            return "aarch64"
        if cpu == 0x01000007:
            return "x86_64"
        return None
    if data[:4] == b"\x7fELF":
        if len(data) < 20 or data[4:7] != b"\x02\x01\x01":
            return None
        file_type, machine = struct.unpack_from("<HH", data, 16)
        if file_type not in {2, 3}:
            return None
        if machine == 183:
            return "aarch64"
        if machine == 62:
            return "x86_64"
    return None


def _safe_source_key(key: str) -> bool:
    if not isinstance(key, str) or not key or "\\" in key:
        return False
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    if path.as_posix() != key:
        return False
    return key in {"Cargo.toml", "Cargo.lock"} or key.endswith(".rs")


def _native_source_hashes(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("native_source_required")
    sources: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "target" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("native_source_symlink")
        if path.is_file() and (path.name in {"Cargo.toml", "Cargo.lock"} or path.suffix == ".rs"):
            sources[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not {"Cargo.toml", "Cargo.lock", "src/main.rs"}.issubset(sources):
        raise ValueError("native_source_incomplete")
    return sources


def _source_map_digest(sources: dict[str, str]) -> str:
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _native_failure(reason: str) -> dict[str, object]:
    return {
        "verified": False,
        "reason": reason,
        "entry_sha256": None,
        "backend_entry_sha": None,
        "native_manifest_sha": None,
        "native_source_digest": None,
        "expected_app_digest": None,
        "actual_app_digest": None,
        "dist_version": None,
        "origin_matches_expected": False,
        "target": None,
    }


def _valid_compiler_identity(compiler: object) -> bool:
    if not isinstance(compiler, str) or not compiler or len(compiler) > 4096:
        return False
    lines = compiler.splitlines()
    if not lines or not lines[0].startswith("rustc "):
        return False
    hosts = [line.partition(":")[2].strip() for line in lines if line.startswith("host:")]
    return len(hosts) == 1 and re.fullmatch(r"[A-Za-z0-9_.-]+", hosts[0]) is not None


def bind_native_openbrowser(
    native_entry: str,
    expected_package: Path,
    *,
    env: dict[str, str],
    cwd: Path,
) -> dict[str, object]:
    """Bind a native entry to its exact installed manifest, sources, and backend."""
    result = _native_failure("binding_failed")
    try:
        expected_package = Path(expected_package).resolve(strict=True)
        result["expected_app_digest"] = application_digest(expected_package)
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "expected_package_unreadable"
        return result
    try:
        expected_sources = _native_source_hashes(expected_package / "native_client" / "rust")
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "expected_native_source_unreadable"
        return result

    if not os.path.dirname(str(native_entry)):
        result["reason"] = "native_entry_path_required"
        return result
    try:
        entry = Path(native_entry)
        if not entry.is_absolute():
            entry = cwd / entry
        entry = entry.resolve(strict=True)
        if not entry.is_file() or not os.access(entry, os.X_OK):
            result["reason"] = "native_entry_unsupported"
            return result
        entry_content = entry.read_bytes()
        result["entry_sha256"] = hashlib.sha256(entry_content).hexdigest()
    except (OSError, RuntimeError, ValueError):
        result["reason"] = "native_entry_unreadable"
        return result

    target = _host_target()
    if target is None:
        result["reason"] = "native_target_unsupported"
        return result
    result["target"] = target
    expected_format = "macho" if target.endswith("-apple-darwin") else "elf"
    if (
        _native_binary_format(entry_content) != expected_format
        or _native_binary_arch(entry_content) != target.split("-", 1)[0]
    ):
        result["reason"] = "native_entry_unsupported"
        return result

    backup_candidate = entry.parent / _NATIVE_BACKEND
    try:
        backup = backup_candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        result["reason"] = "backend_missing"
        return result
    if backup.parent != entry.parent:
        result["reason"] = "backend_not_sibling"
        return result
    try:
        if not backup.is_file() or not os.access(backup, os.X_OK):
            result["reason"] = "backend_launcher_invalid"
            return result
        backup_content = backup.read_bytes()
        result["backend_entry_sha"] = hashlib.sha256(backup_content).hexdigest()
        interpreter = _launcher_interpreter(backup_content)
        if interpreter is None:
            result["reason"] = "backend_launcher_invalid"
            return result
    except (OSError, RuntimeError, ValueError):
        result["reason"] = "backend_launcher_invalid"
        return result

    probe_env = {key: value for key, value in env.items() if key != "PYTHONPATH"}
    try:
        probe = subprocess.run(
            [interpreter, "-c", _NATIVE_PROBE, str(entry.parent)],
            env=probe_env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if probe.returncode != 0:
            result["reason"] = "probe_failed"
            return result
        payload = json.loads(probe.stdout)
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        result["reason"] = "probe_failed"
        return result
    if not isinstance(payload, dict) or "error" in payload:
        result["reason"] = "probe_invalid"
        return result

    version = payload.get("dist_version")
    if (
        payload.get("dist_name") != "openbrowser-ai"
        or not isinstance(version, str)
        or _VERSION.fullmatch(version) is None
        or not isinstance(payload.get("dist_location"), str)
    ):
        result["reason"] = "distribution_metadata_invalid"
        return result
    result["dist_version"] = version

    origin = payload.get("origin")
    locations = payload.get("locations")
    if (
        not isinstance(origin, str)
        or not isinstance(locations, list)
        or len(locations) != 1
        or not isinstance(locations[0], str)
    ):
        result["reason"] = "ambiguous_package"
        return result
    try:
        origin_path = Path(origin).resolve(strict=True)
        actual_package = Path(locations[0]).resolve(strict=True)
        if origin_path.name != "__init__.py" or origin_path.parent != actual_package:
            result["reason"] = "inconsistent_package_spec"
            return result
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "actual_package_unreadable"
        return result

    manifest_text = payload.get("manifest")
    if manifest_text is None:
        result["reason"] = "manifest_missing"
        return result
    if not isinstance(manifest_text, str):
        result["reason"] = "manifest_malformed"
        return result
    manifest_bytes = manifest_text.encode("utf-8")
    result["native_manifest_sha"] = hashlib.sha256(manifest_bytes).hexdigest()
    record_hash = payload.get("manifest_record_hash")
    record_size = payload.get("manifest_record_size")
    if (
        not isinstance(record_hash, str)
        or record_size is None
        or isinstance(record_size, bool)
        or not isinstance(record_size, (int, str))
    ):
        result["reason"] = "manifest_record_missing"
        return result
    try:
        record_size_value = int(record_size)
    except (TypeError, ValueError):
        result["reason"] = "manifest_record_missing"
        return result
    expected_record_hash = (
        "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(manifest_bytes).digest()).rstrip(b"=").decode()
    )
    if record_hash != expected_record_hash or record_size_value != len(manifest_bytes):
        result["reason"] = "manifest_record_mismatch"
        return result
    try:
        manifest = json.loads(manifest_text)
    except (TypeError, ValueError):
        result["reason"] = "manifest_malformed"
        return result
    if not isinstance(manifest, dict):
        result["reason"] = "manifest_malformed"
        return result
    if set(manifest) != _NATIVE_MANIFEST_KEYS:
        result["reason"] = "manifest_schema_invalid"
        return result
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        result["reason"] = "manifest_schema_invalid"
        return result
    if manifest.get("frontend") != "openbrowser-native-v1" or not _valid_compiler_identity(manifest.get("compiler")):
        result["reason"] = "manifest_schema_invalid"
        return result
    if manifest.get("entry_name") != _NATIVE_ENTRY:
        result["reason"] = "manifest_entry_mismatch"
        return result
    if manifest.get("backend_name") != _NATIVE_BACKEND:
        result["reason"] = "manifest_backend_mismatch"
        return result
    manifest_target = manifest.get("target")
    if (
        not isinstance(manifest_target, str)
        or manifest_target not in _SUPPORTED_NATIVE_TARGETS
        or manifest_target != target
    ):
        result["reason"] = "target_mismatch"
        return result
    manifest_version = manifest.get("package_version")
    if not isinstance(manifest_version, str) or _VERSION.fullmatch(manifest_version) is None:
        result["reason"] = "manifest_schema_invalid"
        return result
    if manifest_version != version:
        result["reason"] = "version_mismatch"
        return result
    entry_sha = manifest.get("entry_sha256")
    if not isinstance(entry_sha, str) or _HEX64.fullmatch(entry_sha) is None:
        result["reason"] = "manifest_schema_invalid"
        return result
    if entry_sha != result["entry_sha256"]:
        result["reason"] = "entry_hash_mismatch"
        return result
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict):
        result["reason"] = "manifest_schema_invalid"
        return result
    manifest_sources: dict[str, str] = {}
    for key, value in source_files.items():
        if not isinstance(key, str) or not isinstance(value, str):
            result["reason"] = "manifest_schema_invalid"
            return result
        if not _safe_source_key(key):
            result["reason"] = "manifest_path_escape"
            return result
        if _HEX64.fullmatch(value) is None:
            result["reason"] = "manifest_schema_invalid"
            return result
        manifest_sources[key] = value
    if manifest_sources != expected_sources:
        result["reason"] = "manifest_source_mismatch"
        return result
    try:
        actual_sources = _native_source_hashes(actual_package / "native_client" / "rust")
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "native_source_mismatch"
        return result
    if actual_sources != expected_sources:
        result["reason"] = "native_source_mismatch"
        return result
    result["native_source_digest"] = _source_map_digest(expected_sources)

    result["origin_matches_expected"] = actual_package == expected_package
    try:
        result["actual_app_digest"] = application_digest(actual_package)
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "actual_package_unreadable"
        return result
    if result["actual_app_digest"] != result["expected_app_digest"]:
        result["reason"] = "application_digest_mismatch"
        return result
    result["verified"] = True
    result["reason"] = "exact_package" if result["origin_matches_expected"] else "byte_identical_package"
    return result


def _resolve_console_entry(binary: str, env: dict[str, str], cwd: Path) -> Path | None:
    if os.path.dirname(str(binary)):
        entry = Path(binary)
        if not entry.is_absolute():
            entry = cwd / entry
        return entry.resolve(strict=True)
    search_path = os.pathsep.join(
        str(Path(part) if Path(part).is_absolute() else cwd / part)
        for part in env.get("PATH", os.defpath).split(os.pathsep)
    )
    found = shutil.which(binary, path=search_path)
    if found is None:
        return None
    entry = Path(found)
    if not entry.is_absolute():
        entry = cwd / entry
    return entry.resolve(strict=True)


def bind_console_entry(
    binary: str,
    expected_package: Path,
    *,
    env: dict[str, str],
    cwd: Path,
) -> dict[str, object]:
    """Dispatch native entries to the native binder and everything else to legacy."""
    try:
        entry = _resolve_console_entry(binary, env, cwd)
        if entry is not None and _native_binary_arch(entry.read_bytes()) is not None:
            return bind_native_openbrowser(str(entry), expected_package, env=env, cwd=cwd)
    except (OSError, RuntimeError, ValueError):
        pass
    return bind_openbrowser(binary, expected_package, env=env, cwd=cwd)
