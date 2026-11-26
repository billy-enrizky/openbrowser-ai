"""Application binding tests using only fixture files and fake probe processes."""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def provenance():
    try:
        return importlib.import_module("benchmarks.interactive.resource_provenance")
    except ModuleNotFoundError as error:
        if error.name == "benchmarks.interactive.resource_provenance":
            pytest.fail("The independent application provenance module is not implemented")
        raise


@pytest.fixture
def package(tmp_path: Path):
    path = tmp_path / "expected" / "openbrowser"
    path.mkdir(parents=True)
    (path / "__init__.py").write_text("VALUE = 1\n")
    (path / "cli_entry.py").write_text("def cli(): pass\n")
    (path / "instructions.md").write_text("Runtime instructions.\n")
    return path


@pytest.fixture
def launcher(tmp_path):
    bin_dir = tmp_path / "installed" / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = bin_dir / "python3"
    interpreter.write_text("fixture interpreter, never executed\n")
    interpreter.chmod(0o755)
    entry = bin_dir / "openbrowser-ai"
    entry.write_text(
        f"#!{interpreter}\n"
        "import sys\n"
        "from openbrowser.cli_entry import cli\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(cli())\n"
    )
    entry.chmod(0o755)
    return entry


def fake_probe(monkeypatch, provenance, package, **overrides):
    payload = {
        "origin": str(package / "__init__.py"),
        "locations": [str(package)],
        "dist_name": "openbrowser-ai",
        "dist_version": "0.1.58",
        "dist_location": str(package.parent),
    }
    payload.update(overrides)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    return calls


def bind(provenance, launcher, package, tmp_path, **kwargs):
    return provenance.bind_openbrowser(str(launcher), package, env=kwargs.pop("env", {}), cwd=tmp_path, **kwargs)


def test_application_digest_is_path_independent_and_includes_runtime_resources(provenance, package, tmp_path):
    import shutil

    copied = tmp_path / "copy" / "openbrowser"
    shutil.copytree(package, copied)
    assert provenance.application_digest(package) == provenance.application_digest(copied)
    (copied / "instructions.md").write_text("Changed runtime instructions.\n")
    assert provenance.application_digest(package) != provenance.application_digest(copied)


def test_application_digest_ignores_caches_and_external_benchmark_results(provenance, package, tmp_path):
    before = provenance.application_digest(package)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "cached.pyc").write_bytes(b"cache")
    (package / "generated.pyc").write_bytes(b"cache")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "results.json").write_text("private benchmark results")
    assert provenance.application_digest(package) == before


def test_application_digest_rejects_symlinked_runtime_files(provenance, package, tmp_path):
    target = tmp_path / "external.py"
    target.write_text("VALUE = 3\n")
    (package / "external.py").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        provenance.application_digest(package)


def test_application_digest_rejects_non_package(provenance, tmp_path):
    with pytest.raises((ValueError, OSError)):
        provenance.application_digest(tmp_path)


def test_exact_source_binding_returns_hashed_sanitized_metadata(provenance, package, launcher, tmp_path, monkeypatch):
    fake_probe(monkeypatch, provenance, package)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is True
    assert result["origin_matches_expected"] is True
    assert result["reason"] == "exact_package"
    assert result["expected_app_digest"] == result["actual_app_digest"]
    assert len(result["entry_sha256"]) == 64
    assert result["dist_version"] == "0.1.58"
    assert str(tmp_path) not in json.dumps(result)


def test_byte_identical_installation_is_accepted_without_claiming_same_origin(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
):
    import shutil

    installed = tmp_path / "different-install" / "openbrowser"
    shutil.copytree(package, installed)
    fake_probe(monkeypatch, provenance, installed)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is True
    assert result["origin_matches_expected"] is False
    assert result["reason"] == "byte_identical_package"


def test_same_version_different_source_fails_closed(provenance, package, launcher, tmp_path, monkeypatch):
    installed = tmp_path / "wrong-install" / "openbrowser"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text('VALUE = "other-source"\n')
    fake_probe(monkeypatch, provenance, installed)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "application_digest_mismatch"
    assert result["expected_app_digest"] != result["actual_app_digest"]


def test_probe_uses_shebang_venv_path_and_entry_directory_not_cwd(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
):
    calls = fake_probe(monkeypatch, provenance, package)
    env = {"PYTHONPATH": str(package.parent), "PATH": str(launcher.parent), "KEEP": "value"}
    result = bind(provenance, launcher, package, tmp_path, env=env)
    assert result["verified"] is True
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == str(launcher.parent / "python3")
    assert command[1] == "-c"
    assert command[3] == str(launcher.parent)
    assert "sys.path[0]" in command[2]
    assert "find_spec" in command[2]
    assert "import openbrowser" not in command[2]
    assert options["env"] == {"PATH": str(launcher.parent), "KEEP": "value"}
    assert env["PYTHONPATH"] == str(package.parent)
    assert options["cwd"] == tmp_path
    assert 0 < options["timeout"] <= 30
    assert options["capture_output"] is True


def test_symlinked_entry_resolves_to_real_launcher(provenance, package, launcher, tmp_path, monkeypatch):
    calls = fake_probe(monkeypatch, provenance, package)
    link = tmp_path / "openbrowser-ai"
    link.symlink_to(launcher)
    assert bind(provenance, link, package, tmp_path)["verified"] is True
    assert calls[0][0][3] == str(launcher.parent)


def test_path_binary_resolution_uses_supplied_environment(provenance, package, launcher, tmp_path, monkeypatch):
    fake_probe(monkeypatch, provenance, package)
    result = provenance.bind_openbrowser("openbrowser-ai", package, env={"PATH": str(launcher.parent)}, cwd=tmp_path)
    assert result["verified"] is True


@pytest.mark.parametrize(
    "contents",
    [
        '#!/bin/sh\nexec elsewhere "$@"\n',
        "#!/usr/bin/env python3\nfrom openbrowser.cli_entry import cli\ncli()\n",
        '#!/usr/bin/python3\nimport os\nos.execv("elsewhere", [])\n',
        "#!/usr/bin/python3\nimport sys\nfrom openbrowser.cli_entry import cli\ncli = lambda: 0\nsys.exit(cli())\n",
    ],
)
def test_unsupported_shims_are_rejected_without_probe(provenance, package, launcher, tmp_path, monkeypatch, contents):
    launcher.write_text(contents)

    def forbidden(*args, **kwargs):
        pytest.fail("An unsupported launcher must not start a probe")

    monkeypatch.setattr(provenance.subprocess, "run", forbidden)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "unsupported_launcher"
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize(
    "override",
    [
        {"origin": None},
        {"locations": []},
        {"locations": ["/first", "/second"]},
        {"origin": "/different/__init__.py"},
    ],
)
def test_ambiguous_or_inconsistent_package_spec_is_rejected(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
    override,
):
    fake_probe(monkeypatch, provenance, package, **override)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is False
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize(
    "override",
    [
        {"dist_name": "different-app"},
        {"dist_version": None},
        {"dist_version": "/private/checkout/version"},
        {"error": "application_already_imported"},
    ],
)
def test_missing_or_unsafe_probe_metadata_is_rejected(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
    override,
):
    fake_probe(monkeypatch, provenance, package, **override)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is False
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("behavior", ["timeout", "exception", "nonzero", "invalid_json", "wrong_shape"])
def test_probe_failures_do_not_leak_private_paths(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
    behavior,
):
    def run(command, **kwargs):
        if behavior == "timeout":
            raise subprocess.TimeoutExpired(str(tmp_path), 10)
        if behavior == "exception":
            raise OSError(str(tmp_path))
        if behavior == "nonzero":
            return subprocess.CompletedProcess(command, 1, "", str(tmp_path))
        output = "[" if behavior == "invalid_json" else "[]"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is False
    assert str(tmp_path) not in json.dumps(result)


def test_source_and_launcher_changes_change_binding_fingerprints(provenance, package, launcher, tmp_path, monkeypatch):
    fake_probe(monkeypatch, provenance, package)
    before = bind(provenance, launcher, package, tmp_path)
    launcher.write_text(launcher.read_text() + "\n# changed launcher\n")
    (package / "__init__.py").write_text("VALUE = 2\n")
    after = bind(provenance, launcher, package, tmp_path)
    assert before["entry_sha256"] != after["entry_sha256"]
    assert before["expected_app_digest"] != after["expected_app_digest"]
    assert after["verified"] is True


def test_uv_launcher_with_windows_name_normalization_is_supported(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
):
    launcher.write_text(
        f"#!{launcher.parent / 'python3'}\n"
        "import sys\nfrom openbrowser.cli_entry import cli\n"
        'if __name__ == "__main__":\n'
        '    if sys.argv[0].endswith("-script.pyw"):\n'
        "        sys.argv[0] = sys.argv[0][:-11]\n"
        '    elif sys.argv[0].endswith(".exe"):\n'
        "        sys.argv[0] = sys.argv[0][:-4]\n"
        "    sys.exit(cli())\n"
    )
    fake_probe(monkeypatch, provenance, package)
    assert bind(provenance, launcher, package, tmp_path)["verified"] is True


def test_relative_path_lookup_is_relative_to_adapter_cwd(provenance, package, launcher, tmp_path, monkeypatch):
    fake_probe(monkeypatch, provenance, package)
    result = provenance.bind_openbrowser(
        "openbrowser-ai",
        package,
        env={"PATH": "installed/bin"},
        cwd=tmp_path,
    )
    assert result["verified"] is True


def test_probe_runs_spec_and_metadata_lookup_without_importing_the_application(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
    capsys,
):
    import builtins
    import importlib.metadata
    import importlib.util
    import sys

    original_import = builtins.__import__
    observed = []

    def guarded_import(name, *args, **kwargs):
        if name == "openbrowser" or name.startswith("openbrowser."):
            pytest.fail("The probe must not import the application")
        return original_import(name, *args, **kwargs)

    def find_spec(name):
        observed.append((name, sys.path[0]))
        return SimpleNamespace(origin=str(package / "__init__.py"), submodule_search_locations=[str(package)])

    distribution = SimpleNamespace(
        metadata={"Name": "openbrowser-ai"},
        version="0.1.58",
        locate_file=lambda _: package.parent,
    )
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in list(sys.modules):
        if name == "openbrowser" or name.startswith("openbrowser."):
            monkeypatch.delitem(sys.modules, name)

    def run(command, **kwargs):
        monkeypatch.setattr(sys, "argv", ["-c", command[3]])
        exec(command[2], {})
        return subprocess.CompletedProcess(command, 0, capsys.readouterr().out, "")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    result = bind(provenance, launcher, package, tmp_path)
    assert result["verified"] is True
    assert observed == [("openbrowser", str(launcher.parent))]


def test_symlinked_venv_interpreter_is_not_resolved(provenance, package, launcher, tmp_path, monkeypatch):
    interpreter = launcher.parent / "python3"
    interpreter.unlink()
    real_python = tmp_path / "native-python3"
    real_python.write_text("fixture interpreter, never executed\n")
    real_python.chmod(0o755)
    interpreter.symlink_to(real_python)
    calls = fake_probe(monkeypatch, provenance, package)
    assert bind(provenance, launcher, package, tmp_path)["verified"] is True
    assert calls[0][0][0] == str(interpreter)


def test_pip_launcher_is_supported(provenance, package, launcher, tmp_path, monkeypatch):
    launcher.write_text(f"#!{launcher.parent / 'python3'}\n" + provenance._PIP_ENTRY)
    fake_probe(monkeypatch, provenance, package)
    assert bind(provenance, launcher, package, tmp_path)["verified"] is True


def test_unreadable_expected_package_fails_before_probe(provenance, launcher, tmp_path, monkeypatch):
    calls = fake_probe(monkeypatch, provenance, tmp_path)
    result = bind(provenance, launcher, tmp_path / "missing", tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "expected_package_unreadable"
    assert calls == []


def test_missing_launcher_fails_closed(provenance, package, tmp_path):
    result = provenance.bind_openbrowser("absent-console-entry", package, env={"PATH": ""}, cwd=tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "entry_not_found"
    assert str(tmp_path) not in json.dumps(result)


def _native_source_files(root: Path) -> dict[str, bytes]:
    files = {
        "Cargo.toml": b'[package]\nname="openbrowser-native-client"\n',
        "Cargo.lock": b"version = 4\n",
        "src/main.rs": b"fn main() {}\n",
        "tests/transport.rs": b"#[test]\nfn transport() {}\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files


@pytest.fixture
def native_package(tmp_path: Path):
    expected = tmp_path / "expected" / "openbrowser"
    expected.mkdir(parents=True)
    (expected / "__init__.py").write_text("VALUE = 1\n")
    (expected / "cli_entry.py").write_text("def cli(): pass\n")
    _native_source_files(expected / "native_client" / "rust")
    installed = tmp_path / "installed" / "openbrowser"
    import shutil

    shutil.copytree(expected, installed)
    return expected, installed


def _target(provenance) -> str:
    target = provenance._host_target()
    if target is None:
        pytest.skip("native binder tests require a supported host")
    assert target is not None
    return target


def _native_entry_bytes(target: str) -> bytes:

    if target.endswith("-apple-darwin"):
        cpu = 0x0100000C if target.startswith("aarch64-") else 0x01000007
        data = bytearray(32)
        data[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into("<IIIII", data, 4, cpu, 0, 2, 0, 0)
        return bytes(data)
    machine = 183 if target.startswith("aarch64-") else 62
    data = bytearray(64)
    data[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", data, 16, 2)
    struct.pack_into("<H", data, 18, machine)
    return bytes(data)


def _source_map(root: Path) -> dict[str, str]:
    crate = root / "native_client" / "rust"
    result = {}
    for relative in ("Cargo.toml", "Cargo.lock", "src/main.rs", "tests/transport.rs"):
        result[relative] = hashlib.sha256((crate / relative).read_bytes()).hexdigest()
    return result


def _manifest(
    expected: Path,
    native_entry: Path,
    target: str,
    *,
    version: str = "0.1.61",
    overrides: dict[str, object] | None = None,
) -> tuple[dict[str, object], str, str, int]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "frontend": "openbrowser-native-v1",
        "entry_name": "openbrowser-ai",
        "backend_name": "openbrowser-ai-py",
        "package_version": version,
        "target": target,
        "compiler": "rustc 1.100.0\nhost: test\n",
        "source_files": _source_map(expected),
        "entry_sha256": hashlib.sha256(native_entry.read_bytes()).hexdigest(),
    }
    if overrides:
        manifest.update(overrides)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(text.encode()).digest()
    import base64

    record_hash = "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return manifest, text, record_hash, len(text.encode())


def _native_probe_payload(
    installed: Path,
    manifest_text: str,
    record_hash: str,
    record_size: int,
    *,
    version: str = "0.1.61",
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "origin": str(installed / "__init__.py"),
        "locations": [str(installed)],
        "dist_name": "openbrowser-ai",
        "dist_version": version,
        "dist_location": str(installed.parent),
        "manifest": manifest_text,
        "manifest_record_hash": record_hash,
        "manifest_record_size": record_size,
    }
    if overrides:
        payload.update(overrides)
    return payload


def fake_native_probe(monkeypatch, provenance, payload, **overrides):
    payload = dict(payload)
    payload.update(overrides)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    return calls


def native_fixture(tmp_path: Path, native_package):
    expected, installed = native_package
    target = _target(importlib.import_module("benchmarks.interactive.resource_provenance"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = bin_dir / "python3"
    interpreter.write_text("fixture interpreter, never executed\n")
    interpreter.chmod(0o755)
    native_entry = bin_dir / "openbrowser-ai"
    native_entry.write_bytes(_native_entry_bytes(target))
    native_entry.chmod(0o755)
    backup = bin_dir / "openbrowser-ai-py"
    backup.write_text(
        f"#!{interpreter}\n"
        "import sys\n"
        "from openbrowser.cli_entry import cli\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(cli())\n"
    )
    backup.chmod(0o755)
    return expected, installed, target, bin_dir, native_entry, backup


def bind_native(provenance, native_entry, expected, tmp_path, **kwargs):
    return provenance.bind_native_openbrowser(
        str(native_entry),
        expected,
        env=kwargs.pop("env", {}),
        cwd=tmp_path,
        **kwargs,
    )


def test_native_binding_verifies_manifest_binary_backend_sources_and_app_digest(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, _bin_dir, native_entry, backup = native_fixture(tmp_path, native_package)
    manifest, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is True
    assert result["reason"] in {"exact_package", "byte_identical_package"}
    assert result["origin_matches_expected"] is False
    assert result["entry_sha256"] == manifest["entry_sha256"]
    assert result["backend_entry_sha"] == hashlib.sha256(backup.read_bytes()).hexdigest()
    assert len(result["native_manifest_sha"]) == 64
    assert len(result["native_source_digest"]) == 64
    assert result["dist_version"] == "0.1.61"
    assert result["target"] == target
    assert result["actual_app_digest"] == result["expected_app_digest"]
    assert str(tmp_path) not in json.dumps(result)


def test_native_binding_probe_uses_backend_interpreter_and_never_runs_entry(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, bin_dir, native_entry, _backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    calls = fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    result = bind_native(
        provenance, native_entry, expected, tmp_path, env={"PYTHONPATH": "/private/source", "KEEP": "1"}
    )
    assert result["verified"] is True
    command, options = calls[0]
    assert command[0] == str(bin_dir / "python3")
    assert command[1] == "-c"
    assert command[3] == str(native_entry.parent)
    assert "find_spec" in command[2]
    assert "extra_metadata" in command[2]
    assert "import openbrowser" not in command[2]
    assert str(native_entry) not in command
    assert "PYTHONPATH" not in options["env"]
    assert options["env"]["KEEP"] == "1"
    assert options["cwd"] == tmp_path
    assert 0 < options["timeout"] <= 30


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"schema_version": True}, "manifest_schema_invalid"),
        ({"frontend": "other"}, "manifest_schema_invalid"),
        ({"entry_name": "other"}, "manifest_entry_mismatch"),
        ({"backend_name": "other"}, "manifest_backend_mismatch"),
        ({"target": "unknown-target"}, "target_mismatch"),
        ({"package_version": "0.0.0"}, "version_mismatch"),
        ({"entry_sha256": "0" * 64}, "entry_hash_mismatch"),
        ({"source_files": {"../escape.rs": "0" * 64}}, "manifest_path_escape"),
    ],
)
def test_native_manifest_mutations_fail_closed(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
    overrides,
    reason,
):
    expected, installed, target, _bin_dir, native_entry, _backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(
        expected,
        native_entry,
        target,
        overrides=overrides,
    )
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == reason
    assert str(tmp_path) not in json.dumps(result)


def test_native_missing_manifest_and_bad_record_fail_closed(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, _bin_dir, native_entry, _backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size, overrides={"manifest": None}),
    )
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "manifest_missing"

    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, "sha256=wrong", record_size),
    )
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "manifest_record_mismatch"


def test_native_stale_source_hash_and_missing_installed_source_fail_closed(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, _bin_dir, native_entry, _backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(
        expected,
        native_entry,
        target,
        overrides={"source_files": {**_source_map(expected), "src/main.rs": "0" * 64}},
    )
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "manifest_source_mismatch"

    (installed / "native_client" / "rust" / "tests" / "transport.rs").unlink()
    _manifest_data, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "native_source_mismatch"


def test_native_unknown_launcher_and_bad_backend_ast_fail_closed(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, _bin_dir, native_entry, backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    native_entry.write_bytes(b"#!/bin/sh\nexit 0\n")
    assert bind_native(provenance, native_entry, expected, tmp_path)["reason"] == "native_entry_unsupported"

    native_entry.write_bytes(_native_entry_bytes(target))
    backup.write_text("#!/bin/sh\nexit 0\n")
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "backend_launcher_invalid"


def test_native_application_digest_mismatch_and_wrong_sibling_fail_closed(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, _bin_dir, native_entry, backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    (installed / "__init__.py").write_text('VALUE = "other"\n')
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "application_digest_mismatch"

    (installed / "__init__.py").write_text("VALUE = 1\n")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    backup.rename(foreign / "openbrowser-ai-py")
    backup.symlink_to(foreign / "openbrowser-ai-py")
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "backend_not_sibling"


def test_console_entry_dispatches_native_format_to_native_binder(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, _bin_dir, native_entry, _backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Native dispatch must not run the Python binder")

    monkeypatch.setattr(provenance, "bind_openbrowser", forbidden)
    result = provenance.bind_console_entry(
        str(native_entry),
        expected,
        env={},
        cwd=tmp_path,
    )
    assert result["verified"] is True
    assert "backend_entry_sha" in result
    assert "native_manifest_sha" in result
    assert result["target"] == target


def test_console_entry_delegates_python_launcher_with_unchanged_result(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
):
    fake_probe(monkeypatch, provenance, package)

    def forbidden(*args, **kwargs):
        pytest.fail("Python dispatch must not run the native binder")

    monkeypatch.setattr(provenance, "bind_native_openbrowser", forbidden)
    expected = provenance.bind_openbrowser(
        str(launcher),
        package,
        env={},
        cwd=tmp_path,
    )
    actual = provenance.bind_console_entry(
        str(launcher),
        package,
        env={},
        cwd=tmp_path,
    )
    assert actual == expected
    assert actual["reason"] == "exact_package"


def test_console_entry_delegates_unknown_format_to_legacy(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
):
    launcher.write_text("#!/bin/sh\nexit 0\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Unknown format must not run the native binder")

    monkeypatch.setattr(provenance, "bind_native_openbrowser", forbidden)
    result = provenance.bind_console_entry(
        str(launcher),
        package,
        env={},
        cwd=tmp_path,
    )
    assert result["verified"] is False
    assert result["reason"] == "unsupported_launcher"


def test_console_entry_bare_name_and_unreadable_path_keep_legacy_reasons(
    provenance,
    package,
    launcher,
    tmp_path,
    monkeypatch,
):
    fake_probe(monkeypatch, provenance, package)

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy paths must not run the native binder")

    monkeypatch.setattr(provenance, "bind_native_openbrowser", forbidden)
    result = provenance.bind_console_entry(
        "openbrowser-ai",
        package,
        env={"PATH": str(launcher.parent)},
        cwd=tmp_path,
    )
    assert result["verified"] is True

    missing = provenance.bind_console_entry(
        str(tmp_path / "missing"),
        package,
        env={},
        cwd=tmp_path,
    )
    assert missing["verified"] is False
    assert missing["reason"] == "entry_unreadable"


def test_console_entry_dispatches_bare_native_name_through_path(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
):
    expected, installed, target, bin_dir, native_entry, _backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(expected, native_entry, target)
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Bare native dispatch must not run the Python binder")

    monkeypatch.setattr(provenance, "bind_openbrowser", forbidden)
    result = provenance.bind_console_entry(
        "openbrowser-ai",
        expected,
        env={"PATH": str(bin_dir)},
        cwd=tmp_path,
    )
    assert result["verified"] is True
    assert result["target"] == target


@pytest.mark.parametrize(
    "compiler",
    [
        "",
        42,
        "gcc 14.0\nhost: test\n",
        "rustc 1.100.0\n",
        "rustc 1.100.0\nhost:\n",
    ],
)
def test_native_compiler_identity_must_be_bounded_rustc_vv(
    provenance,
    native_package,
    tmp_path,
    monkeypatch,
    compiler,
):
    expected, installed, target, _bin_dir, native_entry, _backup = native_fixture(tmp_path, native_package)
    _manifest_data, manifest_text, record_hash, record_size = _manifest(
        expected,
        native_entry,
        target,
        overrides={"compiler": compiler},
    )
    fake_native_probe(
        monkeypatch,
        provenance,
        _native_probe_payload(installed, manifest_text, record_hash, record_size),
    )
    result = bind_native(provenance, native_entry, expected, tmp_path)
    assert result["verified"] is False
    assert result["reason"] == "manifest_schema_invalid"
