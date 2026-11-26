"""Fake-only contracts for the offline native wheel verifier."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import subprocess
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import verify_native_wheel as verifier

TARGET = "aarch64-apple-darwin"
TAG = "py3-none-macosx_11_0_arm64"
VERSION = "0.1.0"


@pytest.fixture(autouse=True)
def guard_host_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("verifier fake tests must not start processes or sockets")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def expected_package(tmp_path: Path) -> Path:
    package = tmp_path / "expected" / "openbrowser"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (package / "cli_entry.py").write_text("def cli(): pass\n")
    crate = package / "native_client" / "rust"
    files = {
        "Cargo.toml": b'[package]\nname="openbrowser-native-client"\n',
        "Cargo.lock": b"version = 4\n",
        "src/main.rs": b"fn main() {}\n",
        "tests/transport.rs": b"#[test]\nfn transport() {}\n",
    }
    for relative, content in files.items():
        path = crate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return package


def source_map(expected_package: Path) -> dict[str, str]:
    crate = expected_package / "native_client" / "rust"
    return {
        relative: hashlib.sha256((crate / relative).read_bytes()).hexdigest()
        for relative in ("Cargo.toml", "Cargo.lock", "src/main.rs", "tests/transport.rs")
    }


def binary_bytes() -> bytes:
    data = bytearray(32)
    data[:4] = b"\xcf\xfa\xed\xfe"
    struct.pack_into("<IIIII", data, 4, 0x0100000C, 0, 2, 0, 0)
    return bytes(data)


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def append_zip_member(wheel: Path, name: str, data: bytes) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr(name, data)


def wheel_record(entries: dict[str, bytes]) -> bytes:
    lines = []
    for name, data in entries.items():
        lines.append(f"{name},{record_hash(data)},{len(data)}")
    lines.append(f"{next(iter(entries)).split('/')[0]}/RECORD,,")
    return ("\n".join(lines) + "\n").encode()


def make_wheel(
    tmp_path: Path,
    expected_package: Path,
    *,
    native: bool = True,
    include_binary: bool = True,
    include_manifest: bool = True,
    entry_points: str | None = None,
    root_is_purelib: bool | None = None,
    metadata_version: str = VERSION,
    manifest_overrides: dict[str, object] | None = None,
    record_override: tuple[str, str] | None = None,
    data_bytes: bytes | None = None,
    data_mode: int = 0o755,
    tag: str | None = None,
) -> Path:
    data_dir = "openbrowser_ai-0.1.0.data"
    dist_info = "openbrowser_ai-0.1.0.dist-info"
    binary = binary_bytes() if native else b""
    if data_bytes is not None:
        binary = data_bytes
    manifest: dict[str, object] = {
        "schema_version": 1,
        "frontend": "openbrowser-native-v1",
        "entry_name": "openbrowser-ai",
        "backend_name": "openbrowser-ai-py",
        "package_version": VERSION,
        "target": TARGET,
        "compiler": "rustc 1.100.0\nhost: aarch64-apple-darwin\n",
        "source_files": source_map(expected_package),
        "entry_sha256": hashlib.sha256(binary).hexdigest(),
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if entry_points is None:
        entry_points = (
            "[console_scripts]\nopenbrowser-ai-py = openbrowser.cli_entry:cli\n"
            if native
            else "[console_scripts]\nopenbrowser-ai = openbrowser.cli_entry:cli\n"
            "openbrowser-ai-py = openbrowser.cli_entry:cli\n"
        )
    purelib = (not native) if root_is_purelib is None else root_is_purelib
    wheel_bytes = (
        "Wheel-Version: 1.0\n"
        f"Root-Is-Purelib: {'true' if purelib else 'false'}\n"
        f"Tag: {tag or (TAG if native else 'py3-none-any')}\n"
    ).encode()
    files: dict[str, bytes] = {
        f"{dist_info}/METADATA": f"Metadata-Version: 2.3\nName: openbrowser-ai\nVersion: {metadata_version}\n".encode(),
        f"{dist_info}/WHEEL": wheel_bytes,
        f"{dist_info}/entry_points.txt": entry_points.encode(),
    }
    if native:
        if include_binary:
            files[f"{data_dir}/scripts/openbrowser-ai"] = binary
        if include_manifest:
            files[f"{dist_info}/extra_metadata/openbrowser-native.json"] = manifest_bytes
    files[f"{dist_info}/RECORD"] = wheel_record(files)
    if record_override:
        bad_name, bad_hash = record_override
        old_row = f"{bad_name},{record_hash(files[bad_name])},{len(files[bad_name])}".encode()
        new_row = f"{bad_name},{bad_hash},{len(files[bad_name])}".encode()
        files[f"{dist_info}/RECORD"] = files[f"{dist_info}/RECORD"].replace(old_row, new_row)
    wheel_name = f"openbrowser_ai-0.1.0-{tag or (TAG if native else 'py3-none-any')}.whl"
    wheel_path = tmp_path / wheel_name
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            if name.endswith("scripts/openbrowser-ai"):
                info.external_attr = data_mode << 16
            archive.writestr(info, data)
    return wheel_path


def fake_environment(tmp_path: Path):
    root = tmp_path / "environment"
    binary = root / "venv" / "bin"
    binary.mkdir(parents=True)
    entry = binary / "openbrowser-ai"
    backend = binary / "openbrowser-ai-py"
    entry.write_text("native fixture")
    backend.write_text("python fixture")
    entry.chmod(0o755)
    backend.chmod(0o755)
    return SimpleNamespace(
        root=root,
        entry=entry,
        backend=backend,
        env={"PATH": str(binary)},
        cwd=root,
    )


def binding(
    version: str = VERSION,
    *,
    verified: bool = True,
    reason: str = "byte_identical_package",
    info: dict[str, object] | None = None,
):
    source = info or {
        "entry_sha256": hashlib.sha256(binary_bytes()).hexdigest(),
        "manifest_sha256": "c" * 64,
        "source_digest": "d" * 64,
        "target": TARGET,
        "version": VERSION,
    }
    return {
        "verified": verified,
        "reason": reason,
        "entry_sha256": source["entry_sha256"],
        "backend_entry_sha": "b" * 64,
        "native_manifest_sha": source["manifest_sha256"],
        "native_source_digest": source["source_digest"],
        "expected_app_digest": "e" * 64,
        "actual_app_digest": "e" * 64,
        "dist_version": version,
        "origin_matches_expected": False,
        "target": source["target"],
    }


def verified_binding(wheel: Path, expected_package: Path, **overrides):
    result = binding(info=verifier._inspect_wheel(wheel, expected_package))
    result.update(overrides)
    return result


def fake_runtime(
    monkeypatch,
    tmp_path: Path,
    *,
    wheel: Path | None = None,
    expected_package: Path | None = None,
    entry_version=VERSION,
    backend_version=VERSION,
    bindings=None,
):
    environment = fake_environment(tmp_path)
    monkeypatch.setattr(verifier, "_prepare_environment", lambda wheel, package: environment)
    versions = iter([entry_version, backend_version])
    monkeypatch.setattr(verifier, "_run_version", lambda executable, env: next(versions))
    calls = []
    if bindings is None:
        info = (
            verifier._inspect_wheel(wheel, expected_package)
            if wheel is not None and expected_package is not None
            else None
        )
        sequence = [binding(info=info), binding(info=info)]
    else:
        sequence = bindings
    iterator = iter(sequence)

    def fake_bind(native_entry, expected_package, *, env, cwd):
        calls.append((native_entry, expected_package, env, cwd))
        return next(iterator)

    monkeypatch.setattr(verifier, "bind_console_entry", fake_bind)
    return calls


def test_valid_native_wheel_verifies_versions_and_binding(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package)
    calls = fake_runtime(monkeypatch, tmp_path, wheel=wheel, expected_package=expected_package)
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is True, result
    assert result["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert result["package_version"] == VERSION
    assert result["entry_version"] == VERSION
    assert result["backend_version"] == VERSION
    assert result["binding_before_after_equal"] is True
    binding_result = result["binding"]
    assert isinstance(binding_result, dict)
    assert binding_result["backend_entry_sha"] == "b" * 64
    assert len(calls) == 2
    assert str(tmp_path) not in json.dumps(result)


def test_pure_wheel_never_false_passes(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package, native=False)
    prepared = []
    monkeypatch.setattr(
        verifier,
        "_prepare_environment",
        lambda wheel, package: prepared.append(wheel) or pytest.fail("pure wheel was installed"),
    )
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "wheel_not_native"
    assert prepared == []


@pytest.mark.parametrize(
    "wheel_kwargs,reason",
    [
        ({"native": True, "include_binary": False}, "native_entry_missing"),
        ({"native": True, "data_mode": 0o644}, "native_entry_not_executable"),
        ({"native": True, "include_manifest": False}, "manifest_missing"),
        (
            {
                "native": True,
                "entry_points": (
                    "[console_scripts]\n"
                    "openbrowser-ai = openbrowser.cli_entry:cli\n"
                    "openbrowser-ai-py = openbrowser.cli_entry:cli\n"
                ),
            },
            "entry_points_invalid",
        ),
        ({"native": True, "root_is_purelib": True}, "wheel_metadata_invalid"),
        ({"native": True, "manifest_overrides": {"entry_sha256": "0" * 64}}, "binary_hash_mismatch"),
        ({"native": True, "manifest_overrides": {"source_files": {"src/main.rs": "0" * 64}}}, "source_hash_mismatch"),
        ({"native": True, "manifest_overrides": {"package_version": "9.9.9"}}, "version_mismatch"),
        ({"native": True, "metadata_version": f"/tmp/{VERSION}"}, "wheel_metadata_invalid"),
        ({"native": True, "metadata_version": f"{VERSION}\nextra"}, "wheel_metadata_invalid"),
    ],
)
def test_wheel_mutations_fail_closed(tmp_path, expected_package, monkeypatch, wheel_kwargs, reason):
    wheel = make_wheel(tmp_path, expected_package, **wheel_kwargs)
    fake_runtime(monkeypatch, tmp_path)
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == reason
    assert str(tmp_path) not in json.dumps(result)


def test_manifest_record_mismatch_fails_closed(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(
        tmp_path,
        expected_package,
        record_override=("openbrowser_ai-0.1.0.dist-info/extra_metadata/openbrowser-native.json", "sha256=wrong"),
    )
    fake_runtime(monkeypatch, tmp_path)
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "manifest_record_mismatch"


def test_binding_failure_and_changed_binding_fail_closed(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package)
    fake_runtime(
        monkeypatch,
        tmp_path,
        bindings=[binding(verified=False, reason="application_digest_mismatch")],
    )
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "binding_failed"
    binding_result = result["binding"]
    assert isinstance(binding_result, dict)
    assert binding_result["reason"] == "application_digest_mismatch"

    stable = verified_binding(wheel, expected_package)
    fake_runtime(monkeypatch, tmp_path, bindings=[stable, stable | {"dist_version": "9.9.9"}])
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "binding_changed"


def test_version_gates_and_install_failures_are_closed(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package)
    fake_runtime(monkeypatch, tmp_path, wheel=wheel, expected_package=expected_package, entry_version="9.9.9")
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "version_mismatch"

    def fail(wheel, package):
        raise verifier.VerifierError("install_failed")

    monkeypatch.setattr(verifier, "_prepare_environment", fail)
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "install_failed"


def test_cli_writes_sanitized_receipt(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package)
    fake_runtime(monkeypatch, tmp_path, wheel=wheel, expected_package=expected_package)
    output = tmp_path / "receipt.json"
    status = verifier.main(
        [
            "--wheel",
            str(wheel),
            "--expected-package",
            str(expected_package),
            "--output",
            str(output),
        ]
    )
    assert status == 0
    receipt = json.loads(output.read_text())
    assert receipt["verified"] is True
    assert str(tmp_path) not in output.read_text()


def test_missing_and_unreadable_wheels_return_closed_receipts(tmp_path, expected_package):
    missing = tmp_path / "missing.whl"
    result = verifier.verify_native_wheel(missing, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "wheel_missing"
    assert result["wheel_sha256"] is None

    unreadable = tmp_path / "unreadable.whl"
    unreadable.write_bytes(b"not a wheel")
    result = verifier.verify_native_wheel(unreadable, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "wheel_unreadable"
    assert result["wheel_sha256"] == hashlib.sha256(b"not a wheel").hexdigest()


def test_unexpected_inspection_failure_returns_closed_receipt(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package)

    def fail(wheel, package):
        raise AttributeError("unexpected")

    monkeypatch.setattr(verifier, "_inspect_wheel", fail)
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "verification_failed"


def test_relative_wheel_and_expected_package_become_absolute(tmp_path, expected_package, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fail(wheel, package):
        seen["wheel"] = wheel
        seen["package"] = package
        raise verifier.VerifierError("inspection_failed")

    monkeypatch.setattr(verifier, "_inspect_wheel", fail)
    result = verifier.verify_native_wheel(Path("relative.whl"), Path("expected/openbrowser"))
    assert result["reason"] == "inspection_failed"
    assert seen["wheel"].is_absolute()
    assert seen["wheel"] == tmp_path / "relative.whl"
    assert seen["package"].is_absolute()
    assert seen["package"] == expected_package


def test_existing_output_file_is_rejected_before_verification(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package)
    output = tmp_path / "receipt.json"
    output.write_text("keep\n")
    monkeypatch.setattr(verifier, "verify_native_wheel", lambda *args, **kwargs: pytest.fail("verification ran"))
    status = verifier.main(
        [
            "--wheel",
            str(wheel),
            "--expected-package",
            str(expected_package),
            "--output",
            str(output),
        ]
    )
    assert status == 2
    assert output.read_text() == "keep\n"


def test_legacy_positive_binding_shape_is_rejected(tmp_path, expected_package, monkeypatch):
    wheel = make_wheel(tmp_path, expected_package)
    info = verifier._inspect_wheel(wheel, expected_package)
    legacy = {
        "verified": True,
        "reason": "byte_identical_package",
        "entry_sha256": info["entry_sha256"],
        "backend_entry_sha": "b" * 64,
        "dist_version": VERSION,
        "target": TARGET,
    }
    fake_runtime(monkeypatch, tmp_path, bindings=[legacy, legacy])
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "binding_invalid"


@pytest.mark.parametrize("field", ["entry_sha256", "native_manifest_sha", "native_source_digest"])
def test_binding_wheel_identity_mismatch_is_rejected(tmp_path, expected_package, monkeypatch, field):
    wheel = make_wheel(tmp_path, expected_package)
    changed = verified_binding(wheel, expected_package, **{field: "0" * 64})
    fake_runtime(monkeypatch, tmp_path, bindings=[changed, changed])
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "binding_invalid"


@pytest.mark.parametrize(
    "entry_version,backend_version",
    [
        (f"{VERSION}\nextra", VERSION),
        (f"/tmp/{VERSION}", VERSION),
        (VERSION, f"{VERSION} extra"),
    ],
)
def test_version_output_must_be_one_bounded_token(
    tmp_path, expected_package, monkeypatch, entry_version, backend_version
):
    wheel = make_wheel(tmp_path, expected_package)
    fake_runtime(
        monkeypatch,
        tmp_path,
        wheel=wheel,
        expected_package=expected_package,
        entry_version=entry_version,
        backend_version=backend_version,
    )
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "version_invalid"


@pytest.mark.parametrize(
    "member_name",
    [
        "openbrowser_ai-0.1.0.dist-info/METADATA",
        "openbrowser_ai-0.1.0.dist-info/WHEEL",
        "openbrowser_ai-0.1.0.dist-info/entry_points.txt",
        "openbrowser_ai-0.1.0.dist-info/RECORD",
        "openbrowser_ai-0.1.0.data/scripts/openbrowser-ai",
        "openbrowser_ai-0.1.0.dist-info/extra_metadata/openbrowser-native.json",
    ],
)
def test_duplicate_zip_members_are_rejected(tmp_path, expected_package, member_name):
    wheel = make_wheel(tmp_path, expected_package)
    append_zip_member(wheel, member_name, b"duplicate")
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "wheel_member_duplicate"


@pytest.mark.parametrize(
    "member_name,reason",
    [
        ("other.dist-info/METADATA", "wheel_metadata_invalid"),
        ("other.dist-info/WHEEL", "wheel_metadata_invalid"),
        ("other.dist-info/entry_points.txt", "entry_points_invalid"),
        ("other.dist-info/RECORD", "manifest_record_mismatch"),
        ("other.data/scripts/openbrowser-ai", "native_entry_duplicate"),
        ("other.dist-info/extra_metadata/openbrowser-native.json", "manifest_duplicate"),
    ],
)
def test_duplicate_wheel_component_classes_are_rejected(tmp_path, expected_package, member_name, reason):
    wheel = make_wheel(tmp_path, expected_package)
    append_zip_member(wheel, member_name, b"duplicate")
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == reason


def test_backend_entry_point_must_target_the_python_cli(tmp_path, expected_package):
    wheel = make_wheel(
        tmp_path,
        expected_package,
        entry_points="[console_scripts]\nopenbrowser-ai-py = attacker.module:main\n",
    )
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "entry_points_invalid"


def test_native_wheel_exposes_only_the_python_backend_console_script(tmp_path, expected_package):
    wheel = make_wheel(
        tmp_path,
        expected_package,
        entry_points=(
            "[console_scripts]\nopenbrowser-ai-py = openbrowser.cli_entry:cli\nunexpected = openbrowser.cli_entry:cli\n"
        ),
    )
    result = verifier.verify_native_wheel(wheel, expected_package)
    assert result["verified"] is False
    assert result["reason"] == "entry_points_invalid"
