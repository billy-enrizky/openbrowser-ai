"""Fail-closed release assembly contracts for pure and native distributions."""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TAGS = {
    "py3-none-any",
    "py3-none-macosx_11_0_arm64",
    "py3-none-macosx_10_13_x86_64",
    "py3-none-manylinux_2_17_aarch64",
    "py3-none-manylinux_2_17_x86_64",
}


def workflow():
    return yaml.load((ROOT / ".github/workflows/publish.yml").read_text(), Loader=yaml.BaseLoader)


def assembly():
    jobs = workflow()["jobs"]
    assert "assemble" in jobs, "Release upload requires complete pure/native assembly"
    return jobs["assemble"]


def assembly_program():
    step = next(step for step in assembly()["steps"] if step.get("name") == "Verify complete distribution cohort")
    command = step["run"]
    assert command.startswith("python - <<'PY'\n") and command.endswith("PY\n")
    return command.split("\n", 1)[1].rsplit("\nPY", 1)[0]


def make_distributions(tmp_path, *, wheel_version="0.1.61", sdist_version="0.1.61"):
    dist = tmp_path / "dist"
    dist.mkdir()
    for tag in TAGS:
        with zipfile.ZipFile(dist / f"openbrowser_ai-0.1.61-{tag}.whl", "w") as archive:
            archive.writestr(
                "openbrowser_ai-0.1.61.dist-info/METADATA", f"Name: openbrowser-ai\nVersion: {wheel_version}\n"
            )
    data = f"Name: openbrowser-ai\nVersion: {sdist_version}\n".encode()
    with tarfile.open(dist / "openbrowser_ai-0.1.61.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("openbrowser_ai-0.1.61/PKG-INFO")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return dist


def test_native_matrix_is_reused_and_blocks_assembly_on_failure():
    jobs = workflow()["jobs"]
    assert jobs.get("build-native", {}).get("uses") == "./.github/workflows/native-wheels.yml"
    assert set(assembly()["needs"]) == {"build", "build-native"}
    assert "continue-on-error" not in assembly() and "if" not in assembly()


def test_distribution_and_evidence_artifacts_stay_separate():
    downloads = [step["with"] for step in assembly()["steps"] if step.get("uses") == "actions/download-artifact@v4"]
    assert {item.get("name") for item in downloads if "name" in item} == {"python-package-distributions"}
    (native,) = [item for item in downloads if "pattern" in item]
    assert native == {"pattern": "native-wheel-*", "path": "dist/", "merge-multiple": "true"}


@pytest.mark.parametrize("name", ["publish-to-pypi", "publish-to-testpypi"])
def test_both_indexes_upload_only_after_verified_assembly(name):
    job = workflow()["jobs"][name]
    assert job["needs"] == "assemble"
    downloads = [step for step in job["steps"] if step.get("uses") == "actions/download-artifact@v4"]
    assert len(downloads) == 1
    assert downloads[0]["with"] == {"name": "python-verified-distributions", "path": "dist/"}


def test_complete_equal_version_cohort_passes_without_execution(tmp_path, monkeypatch):
    make_distributions(tmp_path)
    monkeypatch.chdir(tmp_path)
    exec(compile(assembly_program(), "<release assembly>", "exec"), {})


@pytest.mark.parametrize(
    "mutation", ["missing_native", "extra_json", "wrong_tag", "duplicate_metadata", "wheel_version", "sdist_version"]
)
@pytest.mark.parametrize("optimization", [0, 2])
def test_distribution_mutations_fail_before_publish(tmp_path, monkeypatch, mutation, optimization):
    program = assembly_program()
    dist = make_distributions(
        tmp_path,
        wheel_version="0.1.62" if mutation == "wheel_version" else "0.1.61",
        sdist_version="0.1.62" if mutation == "sdist_version" else "0.1.61",
    )
    native = dist / "openbrowser_ai-0.1.61-py3-none-manylinux_2_17_x86_64.whl"
    if mutation == "missing_native":
        native.unlink()
    elif mutation == "extra_json":
        (dist / "receipt.json").write_text("{}")
    elif mutation == "wrong_tag":
        native.rename(dist / "openbrowser_ai-0.1.61-py3-none-linux_x86_64.whl")
    elif mutation == "duplicate_metadata":
        with zipfile.ZipFile(native, "a") as archive:
            archive.writestr("second.dist-info/METADATA", "Name: openbrowser-ai\nVersion: 0.1.61\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises((AssertionError, ValueError)):
        exec(compile(program, "<release assembly>", "exec", optimize=optimization), {})


def test_pure_build_uses_same_pinned_metadata_backend_and_disables_native_flag():
    steps = workflow()["jobs"]["build"]["steps"]
    deps = next(step for step in steps if step.get("name") == "Install build dependencies")["run"]
    assert "hatchling==1.33.0" in deps and "hatch-vcs==0.5.0" in deps
    build = next(step for step in steps if step.get("name") == "Build package")
    assert build["env"]["OPENBROWSER_BUILD_NATIVE"] == "0"
    assert build["run"] == "python -m build --no-isolation"


def test_release_installer_assets_wait_for_verified_assembly():
    assert workflow()["jobs"]["upload-install-script"]["needs"] == "assemble"


@pytest.mark.parametrize(
    "event,tag,valid",
    [
        ("release", "v0.1.61", True),
        ("release", "v0.1.60", False),
        ("release", "", False),
        ("workflow_dispatch", "feature-branch", True),
    ],
)
def test_release_artifacts_match_release_tag(tmp_path, monkeypatch, event, tag, valid):
    program = assembly_program()
    make_distributions(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("GITHUB_REF_NAME", tag)
    if valid:
        exec(compile(program, "<release assembly>", "exec", optimize=2), {})
    else:
        with pytest.raises(ValueError, match="incomplete_or_inconsistent_release_distributions"):
            exec(compile(program, "<release assembly>", "exec", optimize=2), {})
