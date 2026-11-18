import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
METADATA_SCRIPT = ROOT / "benchmarks" / "latest_competitor_metadata.py"
PREFLIGHT_SCRIPT = ROOT / "benchmarks" / "latest_benchmark_preflight.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metadata = load_module("latest_competitor_metadata_for_preflight_tests", METADATA_SCRIPT)
preflight = load_module("latest_benchmark_preflight_under_test", PREFLIGHT_SCRIPT)


@pytest.fixture(autouse=True)
def clear_aws_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name, raising=False)


LIGHTPANDA_BINARY = b"#!/bin/sh\ntest \"$1\" = version || exit 2\necho 'Lightpanda 1.0.4'\n"
AGENT_BROWSER_BINARY = b"#!/bin/sh\necho 'agent-browser 0.43.0'\n"
CHROME_BINARY = b"#!/bin/sh\necho 'Google Chrome for Testing 158.0.8127.36'\n"
CHROME_ARCHIVE_URL = "https://storage.googleapis.com/chrome-for-testing-public/158.0.8127.36/mac-arm64/chrome-mac-arm64.zip"


class FixtureResponse(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def chrome_archive():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        entry = zipfile.ZipInfo(
            "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
        )
        entry.external_attr = 0o100755 << 16
        bundle.writestr(entry, CHROME_BINARY)
    return archive.getvalue()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def saved_responses(chrome_version="158.0.8127.36"):
    light_digest = hashlib.sha256(LIGHTPANDA_BINARY).hexdigest()
    agent_digest = hashlib.sha256(AGENT_BROWSER_BINARY).hexdigest()
    chrome_url = f"https://storage.googleapis.com/chrome-for-testing-public/{chrome_version}/mac-arm64/chrome-mac-arm64.zip"
    sources = {
        "lightpanda_github_latest": {
            "tag_name": "1.0.4",
            "draft": False,
            "prerelease": False,
            "assets": [
                {
                    "name": "lightpanda-aarch64-macos",
                    "browser_download_url": "https://github.com/lightpanda-io/browser/releases/download/1.0.4/lightpanda-aarch64-macos",
                    "digest": f"sha256:{light_digest}",
                    "size": len(LIGHTPANDA_BINARY),
                }
            ],
        },
        "agent_browser_npm_latest": {
            "name": "agent-browser",
            "version": "0.43.0",
            "dist": {
                "tarball": "https://registry.npmjs.org/agent-browser/-/agent-browser-0.43.0.tgz",
                "integrity": "sha512-TExwnZAW39fUBDJhUPrw8KKGmcjzBo7FxNqoP2/usHG1ODsjGI2jAWq3+Q6I5JYyBMjePFEsj8KGB68mALtglA==",
            },
            "engines": {"node": ">=22.0.0"},
            "repository": {"type": "git", "url": "git+https://github.com/vercel-labs/agent-browser.git"},
        },
        "agent_browser_github_latest": {
            "tag_name": "v0.43.0",
            "draft": False,
            "prerelease": False,
            "assets": [
                {
                    "name": "agent-browser-darwin-arm64",
                    "browser_download_url": "https://github.com/vercel-labs/agent-browser/releases/download/v0.43.0/agent-browser-darwin-arm64",
                    "digest": f"sha256:{agent_digest}",
                    "size": len(AGENT_BROWSER_BINARY),
                }
            ],
        },
        "browser_use_pypi": {"info": {"name": "browser-use", "version": "1.0.6"}, "urls": [], "releases": {}},
        "browser_harness_pypi": {"info": {"name": "browser-harness", "version": "0.1.13"}, "urls": [], "releases": {}},
        "chrome_for_testing_stable": {
            "timestamp": "2026-11-17T00:00:00Z",
            "channels": {
                "Stable": {
                    "channel": "Stable",
                    "version": chrome_version,
                    "revision": "1512345",
                    "downloads": {
                        "chrome": [
                            {"platform": "mac-arm64", "url": chrome_url},
                            {"platform": "mac-x64", "url": "https://example.invalid/chrome-x64.zip"},
                        ]
                    },
                }
            },
        },
    }
    return {name: json_bytes(payload) for name, payload in sources.items()}


def offline_opener(bodies):
    def open_url(request, timeout=30):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        for source_name, source_url in metadata.SOURCE_URLS.items():
            if url == source_url:
                return FixtureResponse(bodies[source_name])
        try:
            return FixtureResponse(bodies[url])
        except KeyError as exc:
            raise AssertionError(f"unexpected offline URL: {url}") from exc

    return open_url


def make_private_roots():
    parent = Path("/tmp").resolve(strict=True)
    suffix = uuid.uuid4().hex
    result_root = parent / f"openbrowser-preflight-result-{suffix}"
    source_root = parent / f"openbrowser-preflight-source-{suffix}"
    result_root.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    return result_root, source_root


def git(source, *args):
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(source),
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    return result.stdout.strip()


def create_candidate_repo(source):
    git(source, "init", "-q", "--template=/dev/null")
    git(source, "config", "core.hooksPath", "/dev/null")
    git(source, "config", "user.email", "benchmark-test@example.invalid")
    git(source, "config", "user.name", "Benchmark Test")
    (source / "src" / "openbrowser").mkdir(parents=True)
    (source / "src" / "openbrowser" / "__init__.py").write_text(
        "from dotenv import load_dotenv\nload_dotenv()\n"
    )
    (source / "src" / "dotenv.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def load_dotenv():\n"
        "    if os.environ.get('PYTHON_DOTENV_DISABLED') == '1':\n"
        "        return False\n"
        "    env_file = Path('.env')\n"
        "    if env_file.is_file():\n"
        "        for line in env_file.read_text().splitlines():\n"
        "            if '=' in line:\n"
        "                key, value = line.split('=', 1)\n"
        "                os.environ.setdefault(key, value)\n"
        "    return True\n"
    )
    (source / "src" / "openbrowser" / "config.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def _cwd_value():\n"
        "    env_file = Path('.env')\n"
        "    if env_file.is_file():\n"
        "        for line in env_file.read_text().splitlines():\n"
        "            if line.startswith('OPENBROWSER_DEBUG_LOG_FILE='):\n"
        "                return line.split('=', 1)[1]\n"
        "    return None\n"
        "class Config:\n"
        "    OPENBROWSER_DEBUG_LOG_FILE = os.environ.get('OPENBROWSER_DEBUG_LOG_FILE') or _cwd_value()\n"
        "CONFIG = Config()\n"
    )
    (source / ".env").write_text(
        f"OPENBROWSER_DEBUG_LOG_FILE={preflight.SOURCE_SENTINEL}\nAWS_ACCESS_KEY_ID=FAKE_SOURCE_KEY\n"
    )
    for relative in preflight.EXECUTED_FILE_PATHS:
        file_path = source / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"# frozen fixture for {relative}\n")
    (source / "tests" / "test_dummy.py").parent.mkdir(parents=True, exist_ok=True)
    (source / "tests" / "test_dummy.py").write_text("def test_dummy():\n    assert True\n")
    (source / "uv.lock").write_text("version = 1\n")
    git(source, "add", "--all")
    git(source, "commit", "-qm", "candidate")
    git(source, "tag", "v0.1.56", "HEAD")
    git(source, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert git(source, "status", "--porcelain", "--untracked-files=all") == ""


def prepare_scenario():
    result_root, source_root = make_private_roots()
    try:
        create_candidate_repo(source_root)
        metadata_root = result_root / "metadata"
        tools_root = result_root / "tools"
        tools_root.mkdir(mode=0o700)
        bodies = saved_responses()
        metadata.fetch_metadata(
            metadata_root,
            result_root=result_root,
            open_url=offline_opener(bodies),
        )
        downloads = {
            "https://github.com/lightpanda-io/browser/releases/download/1.0.4/lightpanda-aarch64-macos": LIGHTPANDA_BINARY,
            "https://github.com/vercel-labs/agent-browser/releases/download/v0.43.0/agent-browser-darwin-arm64": AGENT_BROWSER_BINARY,
            CHROME_ARCHIVE_URL: chrome_archive(),
        }
        tool_manifest_path = result_root / "tool-manifest.json"
        metadata.download_tools(
            metadata_root,
            tools_root,
            tool_manifest_path,
            result_root=result_root,
            open_url=offline_opener(downloads),
        )
        candidate_freeze = result_root / "candidate-freeze.txt"
        candidate_freeze.write_text(
            "openbrowser-ai==0.1.56\nPillow==12.0.0\npsutil==7.2.2\nwebsockets==15.0.1\n"
        )
        browser_use_freeze = result_root / "browser-use-freeze.txt"
        browser_use_freeze.write_text(
            "browser-harness==0.1.13\nbrowser-use==1.0.6\npsutil==7.2.2\nwebsockets==15.0.1\n"
        )
        identity_path = result_root / "candidate-identity.json"
        identity = preflight.create_identity(
            source_root,
            sys.executable,
            sys.executable,
            candidate_freeze,
            browser_use_freeze,
            tool_manifest_path,
            identity_path,
        )
        import_cwd = result_root / "import-probe"
        import_cwd.mkdir(mode=0o700)
        proof_path = result_root / "source-import-proof.json"
        proof = preflight.create_source_import_proof(
            source_root,
            sys.executable,
            import_cwd,
            identity_path,
            proof_path,
        )
        version_path = result_root / "version-manifest.json"
        version = preflight.seal_version_manifest(
            source_root,
            sys.executable,
            sys.executable,
            metadata_root,
            tools_root,
            candidate_freeze,
            browser_use_freeze,
            tool_manifest_path,
            identity_path,
            proof_path,
            version_path,
        )
        frozen_harness = {
            "schema": "openbrowser-frozen-harness/1",
            "candidate_identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
            "version_manifest_sha256": hashlib.sha256(version_path.read_bytes()).hexdigest(),
            "tool_manifest_sha256": hashlib.sha256(tool_manifest_path.read_bytes()).hexdigest(),
            "candidate_freeze_sha256": hashlib.sha256(candidate_freeze.read_bytes()).hexdigest(),
            "browser_use_freeze_sha256": hashlib.sha256(browser_use_freeze.read_bytes()).hexdigest(),
            "smoke_preflight_sha256": {"navigation": "3" * 64, "screenshot": "4" * 64},
            "smoke_manifest_sha256": {"navigation": "1" * 64, "screenshot": "2" * 64},
            "executed_file_sha256": identity["executed_files"],
        }
        frozen_harness_path = result_root / "frozen-harness.json"
        frozen_harness_path.write_text(json.dumps(frozen_harness, sort_keys=True, separators=(",", ":")) + "\n")
        return {
            "result_root": result_root,
            "source_root": source_root,
            "metadata_root": metadata_root,
            "tools_root": tools_root,
            "candidate_freeze": candidate_freeze,
            "browser_use_freeze": browser_use_freeze,
            "tool_manifest_path": tool_manifest_path,
            "identity_path": identity_path,
            "identity": identity,
            "proof_path": proof_path,
            "proof": proof,
            "version_path": version_path,
            "version": version,
            "frozen_harness_path": frozen_harness_path,
        }
    except Exception:
        shutil.rmtree(result_root, ignore_errors=True)
        shutil.rmtree(source_root, ignore_errors=True)
        raise


@pytest.fixture
def scenario():
    prepared = prepare_scenario()
    yield prepared
    shutil.rmtree(prepared["result_root"], ignore_errors=True)
    shutil.rmtree(prepared["source_root"], ignore_errors=True)


def test_source_import_proof_uses_selected_source_and_ignores_source_dotenv(scenario):
    proof = scenario["proof"]
    assert proof["outcome"] == "passed"
    assert Path(proof["module_file"]).resolve().is_relative_to(
        (scenario["source_root"] / "src" / "openbrowser").resolve()
    )
    assert proof["dotenv_disabled"] is True
    assert proof["source_sentinel_in_environment"] is False
    assert proof["source_sentinel_in_config"] is False
    assert proof["config_debug_log_file"] is None


def test_import_probe_preserves_virtual_environment_dependencies(scenario):
    package = scenario["source_root"] / "src/openbrowser/__init__.py"
    package.write_text(package.read_text() + "\nfrom PIL import Image\n")
    git(scenario["source_root"], "add", "src/openbrowser/__init__.py")
    git(scenario["source_root"], "commit", "-qm", "exercise virtual environment dependency")
    cwd = scenario["result_root"] / "venv-import-probe"
    cwd.mkdir(mode=0o700)

    report = preflight._source_import_report(
        scenario["source_root"], scenario["identity"]["candidate_python"], cwd,
        preflight.sha256_file(scenario["identity_path"]), result_root=scenario["result_root"],
    )

    assert report["outcome"] == "passed"


def test_preflight_rejects_a_result_root_with_the_wrong_owner_uid(monkeypatch, scenario):
    actual_owner = scenario["result_root"].stat().st_uid
    monkeypatch.setattr(preflight.os, "getuid", lambda: actual_owner + 1)

    with pytest.raises(preflight.PreflightError, match="owned by the current user UID"):
        preflight._private_result_root(scenario["result_root"])


def test_preflight_rejects_a_decoy_nonempty_import_working_directory(scenario):
    decoy = scenario["result_root"] / "decoy-cwd"
    decoy.mkdir(mode=0o700)
    (decoy / ".env").write_text("OPENBROWSER_DEBUG_LOG_FILE=__DECOY_SENTINEL__\n")

    with pytest.raises(preflight.PreflightError, match="empty|decoy|working directory"):
        preflight.create_source_import_proof(
            scenario["source_root"],
            sys.executable,
            decoy,
            scenario["identity_path"],
            scenario["result_root"] / "decoy-proof.json",
        )


def test_preflight_verify_rejects_wrong_candidate_interpreter(scenario):
    output = scenario["result_root"] / "preflight-navigation-smoke.json"

    result = preflight.verify_preflight(
        source=scenario["source_root"],
        python="/bin/sh",
        browser_use_python=sys.executable,
        metadata=scenario["metadata_root"],
        tools=scenario["tools_root"],
        candidate_freeze=scenario["candidate_freeze"],
        browser_use_freeze=scenario["browser_use_freeze"],
        tool_manifest=scenario["tool_manifest_path"],
        identity=scenario["identity_path"],
        version_manifest=scenario["version_path"],
        phase="navigation-smoke",
        output=output,
    )

    assert result["outcome"] == "failed"
    assert output.is_file()
    assert result["errors"]
    assert any(check["state"] == "failed" for check in result["checks"])


def test_preflight_success_writes_phase_evidence_without_mutating_sealed_manifests(scenario):
    version_before = scenario["version_path"].read_bytes()
    identity_before = scenario["identity_path"].read_bytes()
    output = scenario["result_root"] / "preflight-navigation-smoke.json"

    result = preflight.verify_preflight(
        source=scenario["source_root"],
        python=sys.executable,
        browser_use_python=sys.executable,
        metadata=scenario["metadata_root"],
        tools=scenario["tools_root"],
        candidate_freeze=scenario["candidate_freeze"],
        browser_use_freeze=scenario["browser_use_freeze"],
        tool_manifest=scenario["tool_manifest_path"],
        identity=scenario["identity_path"],
        version_manifest=scenario["version_path"],
        phase="navigation-smoke",
        output=output,
    )

    assert result["schema"] == "openbrowser-latest-preflight/1"
    assert result["outcome"] == "passed"
    assert result["errors"] == []
    assert result["checks"]
    names = [check['name'] for check in result['checks']]
    assert len(names) == len(set(names)), 'preflight checks must have unique identities'
    assert all(check["state"] == "passed" for check in result["checks"])
    assert scenario["version_path"].read_bytes() == version_before
    assert scenario["identity_path"].read_bytes() == identity_before
    assert json.loads(output.read_text()) == result


def test_preflight_rejects_output_filename_that_does_not_match_phase(scenario):
    output = scenario["result_root"] / "preflight-unexpected.json"

    with pytest.raises(preflight.PreflightError, match="preflight-navigation-smoke.json"):
        preflight.verify_preflight(
            source=scenario["source_root"],
            python=sys.executable,
            browser_use_python=sys.executable,
            metadata=scenario["metadata_root"],
            tools=scenario["tools_root"],
            candidate_freeze=scenario["candidate_freeze"],
            browser_use_freeze=scenario["browser_use_freeze"],
            tool_manifest=scenario["tool_manifest_path"],
            identity=scenario["identity_path"],
            version_manifest=scenario["version_path"],
            phase="navigation-smoke",
            output=output,
        )


    assert not output.exists()


def test_preflight_artifact_rejects_duplicate_check_identities():
    check = {'name': 'executed_file_sha256', 'expected': True, 'actual': True, 'state': 'passed'}
    result = {
        'schema': preflight.PREFLIGHT_SCHEMA, 'phase': 'navigation-smoke',
        'candidate_identity_sha256': 'a' * 64, 'version_manifest_sha256': 'b' * 64,
        'checks': [check, dict(check)], 'outcome': 'passed', 'errors': [],
    }
    with pytest.raises(preflight.PreflightError, match='duplicate'):
        preflight._validate_preflight_result(result)


def test_preflight_rejects_raw_parent_traversal_in_output_path(scenario):
    nested = scenario["result_root"] / "nested"
    nested.mkdir(mode=0o700)
    output = nested / ".." / "preflight-navigation-smoke.json"

    with pytest.raises(preflight.PreflightError, match="parent traversal"):
        preflight.verify_preflight(
            source=scenario["source_root"],
            python=sys.executable,
            browser_use_python=sys.executable,
            metadata=scenario["metadata_root"],
            tools=scenario["tools_root"],
            candidate_freeze=scenario["candidate_freeze"],
            browser_use_freeze=scenario["browser_use_freeze"],
            tool_manifest=scenario["tool_manifest_path"],
            identity=scenario["identity_path"],
            version_manifest=scenario["version_path"],
            phase="navigation-smoke",
            output=output,
        )

    assert not (scenario["result_root"] / "preflight-navigation-smoke.json").exists()


def test_preflight_verify_rejects_changed_dependency_freeze_without_resealing(scenario):
    before = scenario["version_path"].read_bytes()
    scenario["candidate_freeze"].write_text(scenario["candidate_freeze"].read_text() + "requests==2.32.0\n")
    output = scenario["result_root"] / "preflight-navigation-run.json"

    result = preflight.verify_preflight(
        source=scenario["source_root"],
        python=sys.executable,
        browser_use_python=sys.executable,
        metadata=scenario["metadata_root"],
        tools=scenario["tools_root"],
        candidate_freeze=scenario["candidate_freeze"],
        browser_use_freeze=scenario["browser_use_freeze"],
        tool_manifest=scenario["tool_manifest_path"],
        identity=scenario["identity_path"],
        version_manifest=scenario["version_path"],
        phase="navigation-run",
        output=output,
    )

    assert result["outcome"] == "failed"
    assert any("freeze" in error for error in result["errors"])
    assert scenario["version_path"].read_bytes() == before


def test_seal_rejects_source_import_working_directory_outside_result_root(scenario):
    outside = scenario["result_root"].parent / f"openbrowser-import-decoy-{uuid.uuid4().hex}"
    outside.mkdir(mode=0o700)
    proof = json.loads(scenario["proof_path"].read_text())
    proof["working_directory"] = str(outside)
    scenario["proof_path"].write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
    output = scenario["result_root"] / "second-version-manifest.json"

    try:
        with pytest.raises(preflight.PreflightError, match="below RESULT_ROOT"):
            preflight.seal_version_manifest(
                scenario["source_root"],
                sys.executable,
                sys.executable,
                scenario["metadata_root"],
                scenario["tools_root"],
                scenario["candidate_freeze"],
                scenario["browser_use_freeze"],
                scenario["tool_manifest_path"],
                scenario["identity_path"],
                scenario["proof_path"],
                output,
            )
    finally:
        shutil.rmtree(outside, ignore_errors=True)

    assert not output.exists()


def test_preflight_rejects_latest_metadata_advance(scenario):
    metadata_check = scenario["result_root"] / "metadata-check" / "navigation"
    metadata.fetch_metadata(
        metadata_check,
        result_root=scenario["result_root"],
        open_url=offline_opener(saved_responses(chrome_version="158.0.8127.37")),
    )
    output = scenario["result_root"] / "preflight-navigation-run.json"

    result = preflight.verify_preflight(
        source=scenario["source_root"],
        python=sys.executable,
        browser_use_python=sys.executable,
        metadata=scenario["metadata_root"],
        metadata_check=metadata_check,
        tools=scenario["tools_root"],
        candidate_freeze=scenario["candidate_freeze"],
        browser_use_freeze=scenario["browser_use_freeze"],
        tool_manifest=scenario["tool_manifest_path"],
        identity=scenario["identity_path"],
        version_manifest=scenario["version_path"],
        phase="navigation-run",
        output=output,
    )

    freshness = next(check for check in result["checks"] if check["name"] == "latest_metadata_freshness")
    assert result["outcome"] == "failed"
    assert freshness["state"] == "failed"


def test_preflight_rejects_inherited_aws_credentials(scenario, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "untrusted-parent-profile")
    output = scenario["result_root"] / "preflight-screenshot-smoke.json"

    result = preflight.verify_preflight(
        source=scenario["source_root"],
        python=sys.executable,
        browser_use_python=sys.executable,
        metadata=scenario["metadata_root"],
        tools=scenario["tools_root"],
        candidate_freeze=scenario["candidate_freeze"],
        browser_use_freeze=scenario["browser_use_freeze"],
        tool_manifest=scenario["tool_manifest_path"],
        identity=scenario["identity_path"],
        version_manifest=scenario["version_path"],
        phase="screenshot-smoke",
        output=output,
    )

    no_aws = next(check for check in result["checks"] if check["name"] == "no_aws_environment")
    assert result["outcome"] == "failed"
    assert no_aws["state"] == "failed"
    assert "AWS_PROFILE" in no_aws["actual"]["unexpected_names"]


def test_preflight_rejects_unknown_identity_fields(scenario):
    identity = json.loads(scenario["identity_path"].read_text())
    identity["unexpected"] = "not part of the schema"
    scenario["identity_path"].write_text(json.dumps(identity))

    with pytest.raises(preflight.PreflightError, match="unknown|keys"):
        preflight.load_identity(scenario["identity_path"])


def test_identity_rejects_dirty_candidate_checkout(scenario):
    (scenario["source_root"] / "untracked.txt").write_text("dirty")

    with pytest.raises(preflight.PreflightError, match="clean|dirty"):
        preflight.create_identity(
            scenario["source_root"],
            sys.executable,
            sys.executable,
            scenario["candidate_freeze"],
            scenario["browser_use_freeze"],
            scenario["tool_manifest_path"],
            scenario["result_root"] / "dirty-identity.json",
        )


def test_runtime_variant_accepts_only_src_openbrowser_differences(scenario):
    prepare_smoke_preflight_files(scenario)
    treatment = scenario["result_root"].parent / f"openbrowser-treatment-{uuid.uuid4().hex}"
    subprocess.run(
        ["git", "clone", "-q", str(scenario["source_root"]), str(treatment)],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        (treatment / "src" / "openbrowser" / "config.py").write_text(
            (treatment / "src" / "openbrowser" / "config.py").read_text() + "\n# runtime treatment\n"
        )
        git(treatment, "add", "src/openbrowser/config.py")
        git(treatment, "commit", "-qm", "runtime-only treatment")
        output = scenario["result_root"] / "runtime-variant-identity.json"

        result = preflight.create_runtime_variant_identity(
            scenario["source_root"],
            treatment,
            scenario["identity_path"],
            scenario["frozen_harness_path"],
            scenario["candidate_freeze"],
            scenario["browser_use_freeze"],
            scenario["tool_manifest_path"],
            output,
        )

        assert result["schema"] == "openbrowser-runtime-variant/1"
        assert result["changed_paths"] == ["src/openbrowser/config.py"]
        assert all(path.startswith("src/openbrowser/") for path in result["changed_paths"])
        assert output.is_file()
    finally:
        shutil.rmtree(treatment, ignore_errors=True)


def test_runtime_variant_rejects_benchmark_harness_changes(scenario):
    prepare_smoke_preflight_files(scenario)
    treatment = scenario["result_root"].parent / f"openbrowser-treatment-{uuid.uuid4().hex}"
    subprocess.run(
        ["git", "clone", "-q", str(scenario["source_root"]), str(treatment)],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        (treatment / "benchmarks" / "latest_navigation_baseline.py").write_text("# changed harness\n")
        git(treatment, "add", "benchmarks/latest_navigation_baseline.py")
        git(treatment, "commit", "-qm", "invalid harness treatment")

        with pytest.raises(preflight.PreflightError, match="src/openbrowser|runtime"):
            preflight.create_runtime_variant_identity(
                scenario["source_root"],
                treatment,
                scenario["identity_path"],
                scenario["frozen_harness_path"],
                scenario["candidate_freeze"],
                scenario["browser_use_freeze"],
                scenario["tool_manifest_path"],
                scenario["result_root"] / "invalid-runtime-variant.json",
            )
    finally:
        shutil.rmtree(treatment, ignore_errors=True)


def test_preflight_accepts_a_valid_runtime_variant_and_binds_its_shared_harness(scenario):
    prepare_smoke_preflight_files(scenario)
    treatment = scenario["result_root"].parent / f"openbrowser-treatment-{uuid.uuid4().hex}"
    subprocess.run(
        ["git", "clone", "-q", str(scenario["source_root"]), str(treatment)],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        git(treatment, "config", "core.hooksPath", "/dev/null")
        runtime_file = treatment / "src" / "openbrowser" / "config.py"
        runtime_file.write_text(runtime_file.read_text() + "\n# preflight treatment\n")
        git(treatment, "add", "src/openbrowser/config.py")
        git(treatment, "commit", "-qm", "runtime-only treatment")
        variant_path = scenario["result_root"] / "runtime-variant-identity.json"
        preflight.create_runtime_variant_identity(
            scenario["source_root"],
            treatment,
            scenario["identity_path"],
            scenario["frozen_harness_path"],
            scenario["candidate_freeze"],
            scenario["browser_use_freeze"],
            scenario["tool_manifest_path"],
            variant_path,
        )
        experiment_root = scenario["result_root"] / "experiments" / "runtime-only" / "replicate-1"
        experiment_root.mkdir(mode=0o700, parents=True)
        output = experiment_root / "preflight-experiment-1.json"

        result = preflight.verify_preflight(
            source=treatment,
            python=sys.executable,
            browser_use_python=sys.executable,
            metadata=scenario["metadata_root"],
            tools=scenario["tools_root"],
            candidate_freeze=scenario["candidate_freeze"],
            browser_use_freeze=scenario["browser_use_freeze"],
            tool_manifest=scenario["tool_manifest_path"],
            identity=scenario["identity_path"],
            runtime_variant_identity=variant_path,
            version_manifest=scenario["version_path"],
            phase="experiment-1",
            output=output,
        )

        assert result["outcome"] == "passed", result["errors"]
        assert next(check for check in result["checks"] if check["name"] == "runtime_variant_shared_inputs")["state"] == "passed"
    finally:
        shutil.rmtree(treatment, ignore_errors=True)


def prepare_smoke_preflight_files(scenario):
    result_root = scenario["result_root"]
    identity_sha = preflight.sha256_file(scenario["identity_path"])
    version_sha = preflight.sha256_file(scenario["version_path"])
    paths = {}
    hashes = {}
    for name in ("navigation", "screenshot"):
        phase = f"{name}-smoke"
        path = result_root / f"preflight-{phase}.json"
        record = {
            "schema": preflight.PREFLIGHT_SCHEMA,
            "phase": phase,
            "candidate_identity_sha256": identity_sha,
            "version_manifest_sha256": version_sha,
            "checks": [{"name": "fixture", "expected": True, "actual": True, "state": "passed"}],
            "outcome": "passed",
            "errors": [],
        }
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
        path.chmod(0o600)
        paths[name] = path
        hashes[name] = preflight.sha256_file(path)
    frozen_path = scenario["frozen_harness_path"]
    frozen = json.loads(frozen_path.read_text())
    frozen["smoke_preflight_sha256"] = hashes
    frozen_path.write_text(json.dumps(frozen, sort_keys=True, separators=(",", ":")) + "\n")
    frozen_path.chmod(0o600)
    return paths


def prepare_smoke_freeze_inputs(scenario):
    result_root = scenario["result_root"]
    identity_sha = preflight.sha256_file(scenario["identity_path"])
    smoke_preflights = prepare_smoke_preflight_files(scenario)
    version_sha = preflight.sha256_file(scenario["version_path"])
    tool_manifest_sha = preflight.sha256_file(scenario["tool_manifest_path"])
    inputs = {}
    for name in ("navigation", "screenshot"):
        preflight_path = smoke_preflights[name]
        smoke_root = result_root / "smoke" / name
        smoke_root.mkdir(mode=0o700, parents=True)
        manifest_path = smoke_root / "run-manifest.json"
        manifest = {
            "schema": (
                "openbrowser-latest-navigation-smoke/1"
                if name == "navigation" else "openbrowser-screenshot-smoke/1"
            ),
            "mode": "smoke",
            "run_id": "a" * 32,
            "state": "complete",
            "candidate_identity_sha256": identity_sha,
            "version_manifest_sha256": version_sha,
            "preflight_sha256": preflight.sha256_file(preflight_path),
            "harness_sha256": scenario["identity"]["executed_files"][
                "benchmarks/latest_navigation_baseline.py"
                if name == "navigation" else "benchmarks/screenshot_capability.py"
            ],
            "cleanup": {"ok": True},
            "fixture": {"stopped": True},
            "errors": [],
        }
        if name == "navigation":
            manifest["host_preflight"] = {"state": "clear"}
        else:
            manifest["tool_manifest_sha256"] = tool_manifest_sha
            manifest["arm_status"] = {}
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        manifest_path.chmod(0o600)
        inputs[name] = {
            "preflight": preflight_path,
            "smoke_root": smoke_root,
            "manifest": manifest_path,
        }
    return inputs


def test_freeze_harness_binds_both_smoke_preflights_and_manifests(scenario, monkeypatch):
    smoke_inputs = prepare_smoke_freeze_inputs(scenario)
    verifier_calls = []
    logged_errors = []
    original_run = subprocess.run
    monkeypatch.setattr(preflight.LOGGER, "error", lambda *args: logged_errors.append(args))

    def run_with_offline_verifiers(command, *args, **kwargs):
        command_text = " ".join(str(value) for value in command)
        if "verify-smoke" in command_text:
            assert command[0] == scenario["identity"]["candidate_python"]["supplied_path"]
            verifier_calls.append(command_text)
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_offline_verifiers)
    result_root = scenario["result_root"]
    output = result_root / "frozen-harness.json"
    output.unlink()
    command = [
        "freeze-harness",
        "--source", str(scenario["source_root"]),
        "--identity", str(scenario["identity_path"]),
        "--version-manifest", str(scenario["version_path"]),
        "--tool-manifest", str(scenario["tool_manifest_path"]),
        "--candidate-freeze", str(scenario["candidate_freeze"]),
        "--browser-use-freeze", str(scenario["browser_use_freeze"]),
        "--navigation-preflight", str(smoke_inputs["navigation"]["preflight"]),
        "--screenshot-preflight", str(smoke_inputs["screenshot"]["preflight"]),
        "--navigation-smoke", str(smoke_inputs["navigation"]["smoke_root"]),
        "--screenshot-smoke", str(smoke_inputs["screenshot"]["smoke_root"]),
        "--output", str(output),
    ]
    exit_code = preflight.main(command)

    assert exit_code == 0, logged_errors
    frozen = json.loads(output.read_text())
    assert len(verifier_calls) == 2
    assert any("summarize_latest_navigation.py" in call for call in verifier_calls)
    assert any("screenshot_capability.py" in call for call in verifier_calls)
    assert frozen["smoke_preflight_sha256"] == {
        name: preflight.sha256_file(smoke_inputs[name]["preflight"])
        for name in ("navigation", "screenshot")
    }
    assert frozen["smoke_manifest_sha256"] == {
        name: preflight.sha256_file(smoke_inputs[name]["manifest"])
        for name in ("navigation", "screenshot")
    }
    assert frozen["executed_file_sha256"] == scenario["identity"]["executed_files"]
    frozen_bytes = output.read_bytes()

    assert preflight.main(command) == 0
    assert output.read_bytes() == frozen_bytes
    assert len(verifier_calls) == 4


def test_freeze_harness_does_not_write_when_a_smoke_verifier_fails(scenario, monkeypatch):
    smoke_inputs = prepare_smoke_freeze_inputs(scenario)
    result_root = scenario["result_root"]
    output = result_root / "frozen-harness.json"
    output.unlink()
    original_run = subprocess.run

    def fail_screenshot_verifier(command, *args, **kwargs):
        command_text = " ".join(str(value) for value in command)
        if "screenshot_capability.py" in command_text and "verify-smoke" in command_text:
            return subprocess.CompletedProcess(command, 1, "", "screenshot control failed")
        if "verify-smoke" in command_text:
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_screenshot_verifier)
    monkeypatch.setattr(preflight.LOGGER, "error", lambda *_args: None)
    exit_code = preflight.main([
        "freeze-harness",
        "--source", str(scenario["source_root"]),
        "--identity", str(scenario["identity_path"]),
        "--version-manifest", str(scenario["version_path"]),
        "--tool-manifest", str(scenario["tool_manifest_path"]),
        "--candidate-freeze", str(scenario["candidate_freeze"]),
        "--browser-use-freeze", str(scenario["browser_use_freeze"]),
        "--navigation-preflight", str(smoke_inputs["navigation"]["preflight"]),
        "--screenshot-preflight", str(smoke_inputs["screenshot"]["preflight"]),
        "--navigation-smoke", str(smoke_inputs["navigation"]["smoke_root"]),
        "--screenshot-smoke", str(smoke_inputs["screenshot"]["smoke_root"]),
        "--output", str(output),
    ])

    assert exit_code == 1
    assert not output.exists()


def test_frozen_harness_rejects_changed_smoke_preflight(scenario):
    prepare_smoke_preflight_files(scenario)
    preflight_path = scenario["result_root"] / "preflight-navigation-smoke.json"
    preflight_path.write_text(preflight_path.read_text() + " ")
    preflight_path.chmod(0o600)

    with pytest.raises(preflight.PreflightError, match="smoke preflight hash"):
        preflight.validate_frozen_smoke_preflight_hashes(
            scenario["result_root"],
            json.loads(scenario["frozen_harness_path"].read_text()),
        )
