"""Pure fake contracts for optional native CLI wheel assembly."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hooks(monkeypatch):
    class Interface:
        def __init__(self, root=None):
            self.root = str(root) if root is not None else str(ROOT)
            self.metadata = SimpleNamespace(version="0.1.61")

    for name in (
        "hatchling",
        "hatchling.builders",
        "hatchling.builders.hooks",
        "hatchling.metadata",
        "hatchling.metadata.plugin",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    build = ModuleType("hatchling.builders.hooks.plugin.interface")
    build.__dict__["BuildHookInterface"] = Interface
    metadata = ModuleType("hatchling.metadata.plugin.interface")
    metadata.__dict__["MetadataHookInterface"] = Interface
    monkeypatch.setitem(sys.modules, "hatchling.builders.hooks.plugin", ModuleType("hatchling.builders.hooks.plugin"))
    monkeypatch.setitem(sys.modules, build.__name__, build)
    monkeypatch.setitem(sys.modules, metadata.__name__, metadata)
    path = ROOT / "hatch_native.py"
    assert path.is_file(), "Native wheel hook module is required"
    spec = importlib.util.spec_from_file_location("_native_hooks_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.delenv("OPENBROWSER_BUILD_NATIVE", raising=False)
    monkeypatch.delenv("OPENBROWSER_NATIVE_TARGET", raising=False)
    monkeypatch.delenv("CARGO_BUILD_TARGET", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure hook test attempted a real build or metadata process")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    return module


@pytest.fixture
def crate(tmp_path):
    root = tmp_path / "project"
    rust = root / "src/openbrowser/native_client/rust"
    (rust / "src").mkdir(parents=True)
    (rust / "Cargo.toml").write_text('[package]\nname="openbrowser-ai"\nversion="0.1.0"\n')
    (rust / "Cargo.lock").write_text("version = 4\n")
    (rust / "src/main.rs").write_text("fn main() {}\n")
    return root, rust


def test_pure_metadata_keeps_both_python_entries(hooks):
    metadata = {}
    hooks.CustomMetadataHook().update(metadata)
    assert metadata["scripts"] == {
        "openbrowser-ai": "openbrowser.cli_entry:cli",
        "openbrowser-ai-py": "openbrowser.cli_entry:cli",
    }


def test_native_metadata_leaves_primary_to_data_scripts(hooks, monkeypatch):
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", "1")
    metadata = {}
    hooks.CustomMetadataHook().update(metadata)
    assert metadata["scripts"] == {"openbrowser-ai-py": "openbrowser.cli_entry:cli"}


@pytest.mark.parametrize("value", ["yes", "true", "2", "-1"])
def test_unknown_native_build_flag_fails_closed(hooks, monkeypatch, value):
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", value)
    with pytest.raises(ValueError, match="OPENBROWSER_BUILD_NATIVE"):
        hooks.CustomMetadataHook().update({})


def test_pure_build_spawns_nothing_and_preserves_build_data(hooks, crate):
    root, _ = crate
    data = {"pure_python": True, "tag": "py3-none-any", "shared_scripts": {}}
    hooks.CustomBuildHook(root).initialize("standard", data)
    assert data == {"pure_python": True, "tag": "py3-none-any", "shared_scripts": {}}


def fake_cargo(hooks, monkeypatch, *, fail=False, binary=True, host="aarch64-apple-darwin"):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "rustc":
            return SimpleNamespace(stdout=f"rustc 1.100.0\nhost: {host}\n", stderr="", returncode=0)
        assert command[0] == "cargo"
        assert "--locked" in command and "--release" in command
        if fail:
            raise subprocess.CalledProcessError(1, command, stderr="test compiler failure")
        target = command[command.index("--target") + 1]
        if binary:
            output = Path(kwargs["env"]["CARGO_TARGET_DIR"]) / target / "release/openbrowser-ai"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"\xcf\xfa\xed\xfeNATIVE-FIXTURE")
            output.chmod(0o755)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(hooks.subprocess, "run", run)
    return calls


def test_native_build_pins_artifacts_sources_and_platform(hooks, crate, monkeypatch):
    root, _rust = crate
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", "1")
    calls = fake_cargo(hooks, monkeypatch)
    hook = hooks.CustomBuildHook(root)
    data = {"pure_python": True, "shared_scripts": {}}
    hook.initialize("standard", data)
    assert data["pure_python"] is False
    assert data["tag"] == "py3-none-macosx_11_0_arm64"
    scripts = {destination: Path(source) for source, destination in data["shared_scripts"].items()}
    assert set(scripts) == {"openbrowser-ai", "openbrowser-ai.native.json"}
    manifest = json.loads(scripts["openbrowser-ai.native.json"].read_text())
    assert manifest["schema_version"] == 1
    assert manifest["frontend"] == "openbrowser-native-v1"
    assert manifest["backend_name"] == "openbrowser-ai-py"
    assert manifest["package_version"] == "0.1.61"
    assert manifest["target"] == "aarch64-apple-darwin"
    assert set(manifest["source_files"]) == {"Cargo.toml", "Cargo.lock", "src/main.rs"}
    assert len(manifest["entry_sha256"]) == 64
    assert str(root) not in json.dumps(manifest)
    cargo = next(kwargs for command, kwargs in calls if command[0] == "cargo")
    assert not Path(cargo["env"]["CARGO_TARGET_DIR"]).is_relative_to(root)
    hook.finalize("standard", data, str(root / "fixture.whl"))
    assert all(not source.exists() for source in scripts.values())


@pytest.mark.parametrize("fail,binary", [(True, True), (False, False)])
def test_native_build_failure_never_silently_becomes_pure(hooks, crate, monkeypatch, fail, binary):
    root, _ = crate
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", "1")
    fake_cargo(hooks, monkeypatch, fail=fail, binary=binary)
    with pytest.raises(RuntimeError):
        hooks.CustomBuildHook(root).initialize("standard", {"pure_python": True, "shared_scripts": {}})


def test_unsupported_target_fails_before_cargo(hooks, crate, monkeypatch):
    root, _ = crate
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", "1")
    calls = fake_cargo(hooks, monkeypatch, host="x86_64-pc-windows-msvc")
    with pytest.raises(ValueError, match="target"):
        hooks.CustomBuildHook(root).initialize("standard", {})
    assert all(command[0] != "cargo" for command, _ in calls)


@pytest.mark.parametrize(
    "host,target,tag",
    [
        ("x86_64-unknown-linux-gnu", "x86_64-unknown-linux-musl", "manylinux_2_17_x86_64"),
        ("aarch64-unknown-linux-gnu", "aarch64-unknown-linux-musl", "manylinux_2_17_aarch64"),
        ("x86_64-apple-darwin", "x86_64-apple-darwin", "macosx_10_13_x86_64"),
    ],
)
def test_target_mapping_is_platform_specific(hooks, host, target, tag):
    assert hooks.select_target(host) == (target, "py3-none-" + tag)


def test_native_editable_build_fails_before_compiler(hooks, crate, monkeypatch):
    root, _ = crate
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", "1")
    with pytest.raises(ValueError, match="wheel"):
        hooks.CustomBuildHook(root).initialize("editable", {})


def test_native_sdist_contains_sources_without_running_compiler(hooks, crate, monkeypatch):
    root, _ = crate
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", "1")
    hook = hooks.CustomBuildHook(root)
    hook.target_name = "sdist"
    data = {}
    hook.initialize("standard", data)
    assert data == {}


def test_source_mutation_during_compile_rejects_manifest(hooks, crate, monkeypatch):
    root, rust = crate
    monkeypatch.setenv("OPENBROWSER_BUILD_NATIVE", "1")
    fake_cargo(hooks, monkeypatch)
    original = hooks.subprocess.run

    def mutate(command, **kwargs):
        result = original(command, **kwargs)
        if command[0] == "cargo":
            (rust / "src/main.rs").write_text('fn main() { panic!("drift") }\n')
        return result

    monkeypatch.setattr(hooks.subprocess, "run", mutate)
    with pytest.raises(RuntimeError, match="Native CLI build failed"):
        hooks.CustomBuildHook(root).initialize("standard", {})
