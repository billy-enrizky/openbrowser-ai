"""Verify a native OpenBrowser wheel offline and bind it to expected source.

The verifier inspects the wheel before installing it, installs it into a
temporary venv with no dependency resolution, and runs the native entry and the
Python backup entry with ``--version``. It never starts a daemon or browser and
never contacts an external package index.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from benchmarks.interactive.resource_provenance import bind_console_entry

NATIVE_ENTRY = "openbrowser-ai"
BACKEND_ENTRY = "openbrowser-ai-py"
MANIFEST_SUFFIX = "extra_metadata/openbrowser-native.json"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_TOKEN = re.compile(r"[0-9A-Za-z.+!_-]{1,128}\Z")
_TARGET_TAGS = {
    "aarch64-apple-darwin": "py3-none-macosx_11_0_arm64",
    "x86_64-apple-darwin": "py3-none-macosx_10_13_x86_64",
    "aarch64-unknown-linux-musl": "py3-none-manylinux_2_17_aarch64",
    "x86_64-unknown-linux-musl": "py3-none-manylinux_2_17_x86_64",
}
_MANIFEST_KEYS = {
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


class VerifierError(RuntimeError):
    """A closed verification failure with a stable reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _source_hashes(expected_package: Path) -> dict[str, str]:
    crate = expected_package / "native_client" / "rust"
    result: dict[str, str] = {}
    for path in sorted(crate.rglob("*")):
        relative = path.relative_to(crate)
        if "target" in relative.parts or path.is_symlink() or not path.is_file():
            continue
        if path.name in {"Cargo.toml", "Cargo.lock"} or path.suffix == ".rs":
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not {"Cargo.toml", "Cargo.lock", "src/main.rs"}.issubset(result):
        raise VerifierError("expected_package_invalid")
    return result


def _parse_metadata(files: dict[str, bytes]) -> tuple[str, str]:
    names = [name for name in files if name.endswith(".dist-info/METADATA")]
    if len(names) != 1:
        raise VerifierError("wheel_metadata_invalid")
    metadata = files[names[0]].decode("utf-8", errors="strict")
    package_name = None
    version = None
    in_headers = True
    for line in metadata.splitlines():
        if not line:
            in_headers = False
            continue
        if not in_headers or line[:1].isspace():
            continue
        if ":" not in line:
            raise VerifierError("wheel_metadata_invalid")
        if line.startswith("Name:"):
            package_name = line.partition(":")[2].strip()
        elif line.startswith("Version:"):
            version = line.partition(":")[2].strip()
    if package_name != "openbrowser-ai" or not isinstance(version, str) or _VERSION_TOKEN.fullmatch(version) is None:
        raise VerifierError("wheel_metadata_invalid")
    return package_name, version


def _parse_wheel(files: dict[str, bytes]) -> tuple[str, bool]:
    names = [name for name in files if name.endswith(".dist-info/WHEEL")]
    if len(names) != 1:
        raise VerifierError("wheel_metadata_invalid")
    tag = None
    purelib = None
    for line in files[names[0]].decode("utf-8", errors="strict").splitlines():
        if line.startswith("Tag:"):
            tag = line.partition(":")[2].strip()
        elif line.startswith("Root-Is-Purelib:"):
            value = line.partition(":")[2].strip().lower()
            if value not in {"true", "false"}:
                raise VerifierError("wheel_metadata_invalid")
            purelib = value == "true"
    if tag is None or purelib is None:
        raise VerifierError("wheel_metadata_invalid")
    return tag, purelib


def _parse_entry_points(files: dict[str, bytes]) -> dict[str, str]:
    names = [name for name in files if name.endswith(".dist-info/entry_points.txt")]
    if len(names) != 1:
        raise VerifierError("entry_points_invalid")
    name = names[0]
    parser = configparser.ConfigParser()
    try:
        parser.read_string(files[name].decode("utf-8", errors="strict"))
    except (UnicodeError, configparser.Error):
        raise VerifierError("entry_points_invalid") from None
    if not parser.has_section("console_scripts"):
        raise VerifierError("entry_points_invalid")
    return dict(parser.items("console_scripts"))


def _parse_record(files: dict[str, bytes]) -> dict[str, tuple[str, str]]:
    names = [name for name in files if name.endswith(".dist-info/RECORD")]
    if len(names) != 1:
        raise VerifierError("manifest_record_mismatch")
    name = names[0]
    rows: dict[str, tuple[str, str]] = {}
    for line in files[name].decode("utf-8", errors="strict").splitlines():
        parts = line.split(",")
        if len(parts) >= 3:
            rows[parts[0]] = (parts[1], parts[2])
    return rows


def _require_hex(value: object) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise VerifierError("manifest_schema_invalid")
    return value


def _inspect_wheel(wheel: Path, expected_package: Path) -> dict[str, object]:
    if not wheel.is_file():
        raise VerifierError("wheel_missing")
    try:
        with zipfile.ZipFile(wheel) as archive:
            member_names = archive.namelist()
            if len(member_names) != len(set(member_names)):
                raise VerifierError("wheel_member_duplicate")
            infos = {name: archive.getinfo(name) for name in member_names}
            files = {name: archive.read(name) for name in member_names}
    except (OSError, zipfile.BadZipFile, KeyError):
        raise VerifierError("wheel_unreadable") from None
    _name, version = _parse_metadata(files)
    wheel_tag, purelib = _parse_wheel(files)
    entry_points = _parse_entry_points(files)
    data_names = [name for name in files if name.endswith(".data/scripts/openbrowser-ai")]
    manifest_names = [name for name in files if name.endswith(MANIFEST_SUFFIX)]
    if not data_names and not manifest_names:
        raise VerifierError("wheel_not_native")
    if len(data_names) != 1:
        raise VerifierError("native_entry_missing" if not data_names else "native_entry_duplicate")
    if len(manifest_names) != 1:
        raise VerifierError("manifest_missing" if not manifest_names else "manifest_duplicate")
    if purelib:
        raise VerifierError("wheel_metadata_invalid")
    if NATIVE_ENTRY in entry_points or BACKEND_ENTRY not in entry_points:
        raise VerifierError("entry_points_invalid")
    data_name = data_names[0]
    manifest_name = manifest_names[0]
    if infos[data_name].external_attr & (0o111 << 16) == 0:
        raise VerifierError("native_entry_not_executable")
    data = files[data_name]
    try:
        manifest = json.loads(files[manifest_name])
    except (UnicodeError, ValueError):
        raise VerifierError("manifest_invalid") from None
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise VerifierError("manifest_schema_invalid")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise VerifierError("manifest_schema_invalid")
    if manifest.get("frontend") != "openbrowser-native-v1":
        raise VerifierError("manifest_schema_invalid")
    if manifest.get("entry_name") != NATIVE_ENTRY:
        raise VerifierError("manifest_entry_mismatch")
    if manifest.get("backend_name") != BACKEND_ENTRY:
        raise VerifierError("manifest_backend_mismatch")
    target = manifest.get("target")
    if not isinstance(target, str) or target not in _TARGET_TAGS:
        raise VerifierError("manifest_target_invalid")
    expected_tag = _TARGET_TAGS[target]
    if wheel_tag != expected_tag or not wheel.name.endswith(f"-{expected_tag}.whl"):
        raise VerifierError("wheel_tag_invalid")
    if manifest.get("package_version") != version:
        raise VerifierError("version_mismatch")
    _require_hex(manifest.get("entry_sha256"))
    if manifest["entry_sha256"] != hashlib.sha256(data).hexdigest():
        raise VerifierError("binary_hash_mismatch")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict):
        raise VerifierError("manifest_schema_invalid")
    expected_sources = _source_hashes(expected_package)
    if source_files != expected_sources:
        raise VerifierError("source_hash_mismatch")
    source_digest = hashlib.sha256(
        json.dumps(expected_sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    records = _parse_record(files)
    for name, data_value in ((data_name, data), (manifest_name, files[manifest_name])):
        row = records.get(name)
        if row is None or row[0] != _record_hash(data_value) or row[1] != str(len(data_value)):
            raise VerifierError("manifest_record_mismatch")
    return {
        "version": version,
        "target": target,
        "entry_sha256": hashlib.sha256(data).hexdigest(),
        "backend_entry_sha": hashlib.sha256(entry_points[BACKEND_ENTRY].encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(files[manifest_name]).hexdigest(),
        "source_digest": source_digest,
        "manifest": manifest,
        "wheel_tag": wheel_tag,
    }


def _run(command: Sequence[str], *, env: dict[str, str], cwd: Path, timeout: float = 120.0):
    return subprocess.run(
        list(command),
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _prepare_environment(wheel: Path, expected_package: Path):
    root = Path(tempfile.mkdtemp(prefix="openbrowser-wheel-verify-"))
    venv = root / "venv"
    try:
        result = _run([sys.executable, "-m", "venv", str(venv)], env=dict(os.environ), cwd=root)
        if result.returncode != 0:
            raise VerifierError("venv_failed")
        python = venv / "bin" / "python"
        entry = venv / "bin" / NATIVE_ENTRY
        backend = venv / "bin" / BACKEND_ENTRY
        env = {
            "PATH": str(venv / "bin") + os.pathsep + os.environ.get("PATH", os.defpath),
            "HOME": str(root),
            "TMPDIR": str(root),
            "PYTHONNOUSERSITE": "1",
            "OPENBROWSER_DISABLE_TELEMETRY": "true",
        }
        result = _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--disable-pip-version-check",
                "--no-cache-dir",
                str(wheel),
            ],
            env=env,
            cwd=root,
        )
        if result.returncode != 0:
            raise VerifierError("install_failed")
        if not entry.is_file() or not backend.is_file():
            raise VerifierError("entry_points_invalid")
        return SimpleNamespace(root=root, entry=entry, backend=backend, env=env)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _run_version(executable: Path, env: dict[str, str]) -> str:
    result = _run([str(executable), "--version"], env=env, cwd=Path.cwd(), timeout=60)
    if result.returncode != 0:
        raise VerifierError("version_command_failed")
    value = (result.stdout or "").strip()
    if not value:
        raise VerifierError("version_command_failed")
    return value


def _receipt(wheel: Path) -> dict[str, object]:
    wheel_sha256 = None
    try:
        if wheel.is_file():
            wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    except OSError:
        wheel_sha256 = None
    return {
        "verified": False,
        "reason": "verification_failed",
        "wheel": wheel.name,
        "wheel_sha256": wheel_sha256,
        "package_version": None,
        "entry_version": None,
        "backend_version": None,
        "binding": None,
        "binding_before_after_equal": False,
    }


def _binding_matches(info: dict[str, object], result: object) -> bool:
    if not isinstance(result, dict) or result.get("verified") is not True:
        return False
    for key in ("entry_sha256", "backend_entry_sha", "native_manifest_sha", "native_source_digest"):
        value = result.get(key)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            return False
    return (
        result.get("entry_sha256") == info["entry_sha256"]
        and result.get("native_manifest_sha") == info["manifest_sha256"]
        and result.get("native_source_digest") == info["source_digest"]
        and result.get("target") == info["target"]
        and result.get("dist_version") == info["version"]
    )


def verify_native_wheel(wheel: Path, expected_package: Path) -> dict[str, object]:
    """Verify one built native wheel without network access or a browser."""
    wheel = Path(wheel).expanduser().resolve(strict=False)
    receipt = _receipt(wheel)
    environment = None
    try:
        expected_package = Path(expected_package).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        receipt["reason"] = "expected_package_invalid"
        return receipt
    try:
        info = _inspect_wheel(wheel, expected_package)
        receipt["package_version"] = info["version"]
        environment = _prepare_environment(wheel, expected_package)
        before = bind_console_entry(
            str(environment.entry),
            expected_package,
            env=environment.env,
            cwd=environment.root,
        )
        receipt["binding"] = before
        if not isinstance(before, dict) or before.get("verified") is not True:
            raise VerifierError("binding_failed")
        if not _binding_matches(info, before):
            raise VerifierError("binding_invalid")
        entry_version = _run_version(environment.entry, environment.env)
        backend_version = _run_version(environment.backend, environment.env)
        if _VERSION_TOKEN.fullmatch(entry_version) is None or _VERSION_TOKEN.fullmatch(backend_version) is None:
            raise VerifierError("version_invalid")
        receipt["entry_version"] = entry_version
        receipt["backend_version"] = backend_version
        after = bind_console_entry(
            str(environment.entry),
            expected_package,
            env=environment.env,
            cwd=environment.root,
        )
        receipt["binding_before_after_equal"] = before == after
        if before != after:
            raise VerifierError("binding_changed")
        if entry_version != info["version"] or backend_version != info["version"]:
            raise VerifierError("version_mismatch")
        receipt["verified"] = True
        receipt["reason"] = "verified"
    except VerifierError as error:
        receipt["reason"] = error.reason
    except (OSError, ValueError, RuntimeError):
        receipt["reason"] = "verification_failed"
    except Exception:
        receipt["reason"] = "verification_failed"
    finally:
        root = getattr(environment, "root", None)
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expected-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = args.output.open("x", encoding="utf-8")
    except (FileExistsError, OSError):
        return 2
    try:
        receipt = verify_native_wheel(args.wheel, args.expected_package)
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    finally:
        handle.close()
    return 0 if receipt["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
