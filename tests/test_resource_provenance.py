"""Application binding tests using only fixture files and fake probe processes."""

from __future__ import annotations

import importlib
import json
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
