"""Release-independent contracts for the native wheel build matrix."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/native-wheels.yml"


def workflow():
    assert WORKFLOW.is_file(), "Native wheel CI must verify every supported platform"
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def step_named(name):
    return next(step for step in workflow()["jobs"]["build-native"]["steps"] if step.get("name") == name)


def test_native_ci_runs_on_changes_and_is_reusable():
    data = workflow()
    assert {"workflow_call", "workflow_dispatch", "pull_request", "push"} <= data["on"].keys()
    paths = data["on"]["pull_request"]["paths"]
    assert {"hatch_native.py", "src/openbrowser/native_client/rust/**", ".github/workflows/publish.yml"} <= set(paths)
    assert data["permissions"] == {"contents": "read"}


def test_every_supported_native_architecture_has_a_matching_runner():
    job = workflow()["jobs"]["build-native"]
    assert job["strategy"]["fail-fast"] == "false"
    matrix = job["strategy"]["matrix"]["include"]
    assert {(row["runner"], row["target"]) for row in matrix} == {
        ("macos-15", "aarch64-apple-darwin"),
        ("macos-15-intel", "x86_64-apple-darwin"),
        ("ubuntu-24.04", "x86_64-unknown-linux-musl"),
        ("ubuntu-24.04-arm", "aarch64-unknown-linux-musl"),
    }
    assert len(matrix) == 4
    assert "continue-on-error" not in job


def test_native_toolchain_and_build_dependencies_are_pinned():
    data = workflow()
    assert data["env"]["RUST_VERSION"] == "1.100.0"
    toolchain = step_named("Install pinned toolchain")["run"]
    assert "rustup toolchain install" in toolchain and '"$RUST_VERSION"' in toolchain
    assert "clippy,rustfmt" in toolchain
    deps = step_named("Install build dependencies")["run"]
    for pin in ["build==1.3.0", "hatchling==1.33.0", "hatch-vcs==0.5.0"]:
        assert pin in deps


def test_rust_protocol_tests_and_static_checks_precede_native_build():
    steps = workflow()["jobs"]["build-native"]["steps"]
    check = step_named("Verify Rust core")["run"]
    assert "cargo test --locked --release" in check
    assert "cargo clippy --locked" in check and "-D warnings" in check
    assert "cargo fmt" in check and "--check" in check
    names = [step.get("name") for step in steps]
    assert names.index("Verify Rust core") < names.index("Build platform wheel")
    build = step_named("Build platform wheel")
    assert build["env"]["OPENBROWSER_BUILD_NATIVE"] == "1"
    assert "--no-isolation --wheel" in build["run"]


def test_actual_install_smoke_and_equal_version_selection_are_required():
    smoke = step_named("Verify installed native wheel")["run"]
    assert "scripts.ci.verify_native_wheel" in smoke
    assert "--expected-package src/openbrowser" in smoke
    assert "--output native-wheel-receipt.json" in smoke
    selection = step_named("Verify same-version wheel selection")["run"]
    assert "OPENBROWSER_BUILD_NATIVE=0" in selection
    assert "--no-index" in selection and "--no-deps" in selection
    assert "--find-links dist" in selection and "--find-links pure-dist" in selection
    assert "parse_wheel_filename" in selection and "sha256" in selection
    assert "uv pip install" in selection and "uv venv" in selection
    assert "zipfile.ZipFile" in selection


def test_linux_uses_oldest_claimed_glibc_baseline_and_retains_image_identity():
    rows = workflow()["jobs"]["build-native"]["strategy"]["matrix"]["include"]
    linux = [row for row in rows if "linux" in row["target"]]
    assert {row["baseline_image"] for row in linux} == {
        "quay.io/pypa/manylinux2014_x86_64",
        "quay.io/pypa/manylinux2014_aarch64",
    }
    baseline = step_named("Verify Linux compatibility baseline")
    assert baseline["if"] == "matrix.baseline_image != ''"
    for required in ["docker pull", "docker inspect", "RepoDigests", "docker run --rm", ":/project:ro", "cp312-cp312"]:
        assert required in baseline["run"]
    assert "scripts.ci.verify_native_wheel" in baseline["run"]


def test_native_artifact_is_unique_and_missing_files_fail():
    upload = step_named("Store verified native wheel")
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["name"] == "native-wheel-${{ matrix.target }}"
    assert upload["with"]["if-no-files-found"] == "error"
    assert "dist/*.whl" in upload["with"]["path"]
    assert upload["with"]["path"] == "dist/*.whl"
    evidence = step_named("Store native build evidence")
    assert evidence["with"]["name"] == "native-evidence-${{ matrix.target }}"
    assert "native-wheel-receipt.json" in evidence["with"]["path"]
    assert "dist/" not in evidence["with"]["path"]


@pytest.mark.parametrize("target,minimum", [("aarch64-apple-darwin", "11.0"), ("x86_64-apple-darwin", "10.13")])
def test_macos_deployment_minimum_is_explicit(target, minimum):
    rows = workflow()["jobs"]["build-native"]["strategy"]["matrix"]["include"]
    assert next(row for row in rows if row["target"] == target)["deployment_target"] == minimum
    assert workflow()["jobs"]["build-native"]["env"]["MACOSX_DEPLOYMENT_TARGET"] == "${{ matrix.deployment_target }}"


def test_native_wheel_upload_follows_all_checks_without_failure_overrides():
    steps = workflow()["jobs"]["build-native"]["steps"]
    names = [step.get("name") for step in steps]
    assert names.index("Verify Linux compatibility baseline") < names.index("Store verified native wheel")
    for step in steps:
        assert "continue-on-error" not in step
        if step.get("name") not in {"Verify Linux compatibility baseline", "Store native build evidence"}:
            assert "if" not in step
