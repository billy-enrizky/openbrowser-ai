import hashlib
import importlib.util
import io
import json
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "latest_competitor_metadata.py"
)
spec = importlib.util.spec_from_file_location(
    "latest_competitor_metadata_under_test", SCRIPT
)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


LIGHTPANDA_BINARY = (
    b"#!/bin/sh\ntest \"$1\" = version || exit 2\necho 'Lightpanda 1.0.4'\n"
)
AGENT_BROWSER_BINARY = b"#!/bin/sh\necho 'agent-browser 0.43.0'\n"
CHROME_BINARY = b"#!/bin/sh\necho 'Google Chrome for Testing 158.0.8127.36'\n"


@pytest.mark.parametrize("mutation", ["none", "replacement", "foreign_parent"])
@pytest.mark.parametrize("write_error", [OSError, KeyboardInterrupt])
def test_failed_manifest_write_only_removes_its_owned_path(
    tmp_path, monkeypatch, mutation, write_error
):
    parent = tmp_path / "artifacts"
    parent.mkdir()
    path = parent / "manifest.json"

    def fail_sync(_descriptor):
        if mutation == "replacement":
            path.unlink()
            path.write_bytes(b"foreign replacement")
        elif mutation == "foreign_parent":
            moved = tmp_path / "moved-artifacts"
            parent.rename(moved)
            parent.mkdir()
            metadata.os.link(moved / path.name, path)
        raise write_error("injected sync failure")

    monkeypatch.setattr(metadata.os, "fsync", fail_sync)
    expected_error = metadata.MetadataError if write_error is OSError else write_error
    with pytest.raises(expected_error):
        metadata._write_new_file(path, b"our manifest")

    if mutation == "none":
        assert not path.exists()
    else:
        assert path.read_bytes() == (
            b"foreign replacement" if mutation == "replacement" else b"our manifest"
        )


class FixtureResponse(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@pytest.fixture
def private_result_root():
    parent = Path("/tmp").resolve(strict=True)
    root = parent / f"openbrowser-metadata-test-{uuid.uuid4().hex}"
    root.mkdir(mode=0o700)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def run_metadata_cli(args):
    try:
        return metadata.main(args)
    except SystemExit as exc:
        return int(exc.code)


def chrome_archive(binary=CHROME_BINARY):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        entry = zipfile.ZipInfo(
            "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
        )
        entry.external_attr = 0o100755 << 16
        bundle.writestr(entry, binary)
    return archive.getvalue()


def test_chrome_archive_preserves_internal_framework_symlink_chain(tmp_path):
    archive = tmp_path / "chrome.zip"
    base = "chrome-mac-arm64/Google Chrome for Testing.app/Contents"
    framework = f"{base}/Frameworks/Test.framework"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{base}/MacOS/Google Chrome for Testing", CHROME_BINARY)
        bundle.writestr(f"{framework}/Versions/158/Resources/proof", b"framework data")
        for path, target in (
            (f"{framework}/Resources", "Versions/Current/Resources"),
            (f"{framework}/Versions/Current", "158"),
        ):
            entry = zipfile.ZipInfo(path)
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(entry, target)

    extraction = tmp_path / "unpacked"
    executable = metadata._extract_chrome_archive(archive, extraction)

    assert executable.is_file()
    resources = (
        extraction
        / "Google Chrome for Testing.app/Contents/Frameworks/Test.framework/Resources"
    )
    assert resources.is_symlink()
    assert (resources / "proof").read_bytes() == b"framework data"


@pytest.mark.parametrize("target", ["/tmp/outside", "../../../../outside", "missing"])
def test_chrome_archive_rejects_escaping_or_dangling_symlinks(tmp_path, target):
    archive = tmp_path / "chrome.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        entry = zipfile.ZipInfo("chrome-mac-arm64/unsafe-link")
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(entry, target)

    with pytest.raises(metadata.MetadataError):
        metadata._extract_chrome_archive(archive, tmp_path / "unpacked")


def response_fixtures(
    *, lightpanda_digest=None, agent_browser_digest=None, chrome_version="158.0.8127.36"
):
    lightpanda_digest = (
        lightpanda_digest or hashlib.sha256(LIGHTPANDA_BINARY).hexdigest()
    )
    agent_browser_digest = (
        agent_browser_digest or hashlib.sha256(AGENT_BROWSER_BINARY).hexdigest()
    )
    release_asset = lambda name, url, digest: {
        "name": name,
        "browser_download_url": url,
        "size": len(
            LIGHTPANDA_BINARY
            if name.startswith("lightpanda-")
            else AGENT_BROWSER_BINARY
        ),
        "digest": None if digest is None else f"sha256:{digest}",
    }
    payloads = {
        "lightpanda_github_latest": {
            "tag_name": "1.0.4",
            "draft": False,
            "prerelease": False,
            "assets": [
                release_asset(
                    "lightpanda-aarch64-macos",
                    "https://github.com/lightpanda-io/browser/releases/download/1.0.4/lightpanda-aarch64-macos",
                    lightpanda_digest,
                ),
                release_asset(
                    "lightpanda-x86_64-macos",
                    "https://example.invalid/lightpanda-x64",
                    "a" * 64,
                ),
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
            "repository": {
                "type": "git",
                "url": "git+https://github.com/vercel-labs/agent-browser.git",
            },
        },
        "agent_browser_github_latest": {
            "tag_name": "v0.43.0",
            "draft": False,
            "prerelease": False,
            "assets": [
                release_asset(
                    "agent-browser-darwin-arm64",
                    "https://github.com/vercel-labs/agent-browser/releases/download/v0.43.0/agent-browser-darwin-arm64",
                    agent_browser_digest,
                ),
                release_asset(
                    "agent-browser-linux-x64",
                    "https://example.invalid/agent-linux",
                    "b" * 64,
                ),
            ],
        },
        "browser_use_pypi": {
            "info": {"name": "browser-use", "version": "1.0.6"},
            "urls": [],
            "releases": {},
        },
        "browser_harness_pypi": {
            "info": {"name": "browser-harness", "version": "0.1.13"},
            "urls": [],
            "releases": {},
        },
        "chrome_for_testing_stable": {
            "timestamp": "2026-11-17T00:00:00Z",
            "channels": {
                "Stable": {
                    "channel": "Stable",
                    "version": chrome_version,
                    "revision": "1512345",
                    "downloads": {
                        "chrome": [
                            {
                                "platform": "mac-arm64",
                                "url": f"https://storage.googleapis.com/chrome-for-testing-public/{chrome_version}/mac-arm64/chrome-mac-arm64.zip",
                            },
                            {
                                "platform": "mac-x64",
                                "url": f"https://storage.googleapis.com/chrome-for-testing-public/{chrome_version}/mac-x64/chrome-mac-x64.zip",
                            },
                        ]
                    },
                }
            },
        },
    }
    return {
        name: json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        for name, payload in payloads.items()
    }


def fixture_opener(bodies, requested=None):
    requested = requested if requested is not None else []

    def open_url(request, timeout=30):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        requested.append((url, timeout))
        for source_name, source_url in metadata.SOURCE_URLS.items():
            if url == source_url:
                return FixtureResponse(bodies[source_name])
        try:
            return FixtureResponse(bodies[url])
        except KeyError as exc:
            raise AssertionError(f"unexpected offline URL: {url}") from exc

    return open_url


def test_fetch_records_official_urls_and_exact_response_hashes_from_saved_fixtures(
    private_result_root,
):
    bodies = response_fixtures()
    requested = []
    output = private_result_root / "metadata"

    manifest = metadata.fetch_metadata(
        output,
        result_root=private_result_root,
        open_url=fixture_opener(bodies, requested),
        fetched_at_utc="2026-11-18T13:00:00Z",
    )

    assert manifest["schema"] == "openbrowser-latest-metadata/1"
    assert manifest["fetched_at_utc"] == "2026-11-18T13:00:00Z"
    assert set(manifest["sources"]) == set(metadata.SOURCE_URLS)
    assert metadata.SOURCE_URLS == {
        "lightpanda_github_latest": "https://api.github.com/repos/lightpanda-io/browser/releases/latest",
        "agent_browser_npm_latest": "https://registry.npmjs.org/agent-browser/latest",
        "agent_browser_github_latest": "https://api.github.com/repos/vercel-labs/agent-browser/releases/latest",
        "browser_use_pypi": "https://pypi.org/pypi/browser-use/json",
        "browser_harness_pypi": "https://pypi.org/pypi/browser-harness/json",
        "chrome_for_testing_stable": "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json",
    }
    assert {url for url, _timeout in requested} == set(metadata.SOURCE_URLS.values())
    for name, url in metadata.SOURCE_URLS.items():
        record = manifest["sources"][name]
        assert record == {
            "url": url,
            "file": metadata.SOURCE_FILES[name],
            "response_sha256": hashlib.sha256(bodies[name]).hexdigest(),
        }
        assert (output / record["file"]).read_bytes() == bodies[name]

    parsed = metadata.parse_metadata_sources(output)
    assert parsed["latest_versions"] == {
        "lightpanda": "1.0.4",
        "agent-browser": "0.43.0",
        "browser-use": "1.0.6",
        "browser-harness": "0.1.13",
        "chrome_for_testing": "158.0.8127.36",
    }
    assert parsed["chrome_download_url"].endswith(
        "/158.0.8127.36/mac-arm64/chrome-mac-arm64.zip"
    )


def test_metadata_files_match_the_documented_output_layout():
    assert metadata.SOURCE_FILES["chrome_for_testing_stable"] == "chrome-stable.json"


def test_fetch_rejects_metadata_output_outside_result_root_before_fetching(
    private_result_root, tmp_path
):
    output = tmp_path / "metadata-outside-result-root"

    status = run_metadata_cli(
        ["fetch", "--result-root", str(private_result_root), "--output", str(output)]
    )

    assert status == 1
    assert not output.exists()


def test_fetch_rejects_a_result_root_with_the_wrong_owner_uid(
    private_result_root, monkeypatch
):
    actual_owner = private_result_root.stat().st_uid
    monkeypatch.setattr(metadata.os, "getuid", lambda: actual_owner + 1)
    output = private_result_root / "metadata"

    status = run_metadata_cli(
        ["fetch", "--result-root", str(private_result_root), "--output", str(output)]
    )

    assert status == 1
    assert not output.exists()


def test_download_rejects_tools_output_outside_result_root_before_downloading(
    private_result_root, tmp_path
):
    metadata_dir = private_result_root / "metadata"
    metadata.fetch_metadata(
        metadata_dir,
        result_root=private_result_root,
        open_url=fixture_opener(response_fixtures()),
    )
    tools_dir = tmp_path / "tools-outside-result-root"
    manifest_path = private_result_root / "tool-manifest.json"

    status = run_metadata_cli(
        [
            "download-tools",
            "--result-root",
            str(private_result_root),
            "--metadata",
            str(metadata_dir),
            "--tools",
            str(tools_dir),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert status == 1
    assert not tools_dir.exists()
    assert not manifest_path.exists()


def test_download_rejects_metadata_input_outside_result_root_before_downloading(
    private_result_root, tmp_path
):
    source_metadata = private_result_root / "metadata"
    metadata.fetch_metadata(
        source_metadata,
        result_root=private_result_root,
        open_url=fixture_opener(response_fixtures()),
    )
    metadata_dir = tmp_path / "metadata-outside-result-root"
    shutil.copytree(source_metadata, metadata_dir)
    metadata_dir.chmod(0o700)
    tools_dir = private_result_root / "tools"
    manifest_path = private_result_root / "tool-manifest.json"

    status = run_metadata_cli(
        [
            "download-tools",
            "--result-root",
            str(private_result_root),
            "--metadata",
            str(metadata_dir),
            "--tools",
            str(tools_dir),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert status == 1
    assert not tools_dir.exists()
    assert not manifest_path.exists()


def test_download_rejects_manifest_outside_result_root_before_creating_tools(
    private_result_root, tmp_path
):
    metadata_dir = private_result_root / "metadata"
    metadata.fetch_metadata(
        metadata_dir,
        result_root=private_result_root,
        open_url=fixture_opener(response_fixtures()),
    )
    tools_dir = private_result_root / "tools"
    tools_dir.mkdir(mode=0o700)
    manifest_path = tmp_path / "tool-manifest-outside-result-root.json"

    with pytest.raises(metadata.MetadataError, match="below RESULT_ROOT"):
        metadata.download_tools(
            metadata_dir,
            tools_dir,
            manifest_path,
            result_root=private_result_root,
            open_url=no_preparation,
        )

    status = run_metadata_cli(
        [
            "download-tools",
            "--result-root",
            str(private_result_root),
            "--metadata",
            str(metadata_dir),
            "--tools",
            str(tools_dir),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert status == 1
    assert list(tools_dir.iterdir()) == []
    assert not manifest_path.exists()


@pytest.mark.parametrize("integrity", ["sha512-c2F2ZGF0YQ==", "sha512-***"])
def test_npm_integrity_must_be_base64_encoded_sha512_digest(integrity):
    payload = json.loads(response_fixtures()["agent_browser_npm_latest"])
    payload["dist"]["integrity"] = integrity

    with pytest.raises(metadata.MetadataError, match="integrity"):
        metadata._parse_npm(payload)


@pytest.mark.parametrize(
    ("source_name", "field"),
    [
        ("lightpanda_github_latest", "lightpanda-aarch64-macos"),
        ("agent_browser_github_latest", "agent-browser-darwin-arm64"),
    ],
)
def test_metadata_rejects_missing_official_release_digest(
    private_result_root, source_name, field
):
    payloads = json.loads(response_fixtures()[source_name])
    next(asset for asset in payloads["assets"] if asset["name"] == field)["digest"] = (
        None
    )
    bodies = response_fixtures()
    bodies[source_name] = json.dumps(payloads).encode()

    with pytest.raises(metadata.MetadataError, match="digest"):
        metadata.fetch_metadata(
            private_result_root / "metadata",
            result_root=private_result_root,
            open_url=fixture_opener(bodies),
        )


def test_metadata_rejects_agent_browser_npm_and_release_version_disagreement(
    private_result_root,
):
    payloads = json.loads(response_fixtures()["agent_browser_github_latest"])
    payloads["tag_name"] = "v0.43.1"
    bodies = response_fixtures()
    bodies["agent_browser_github_latest"] = json.dumps(payloads).encode()

    with pytest.raises(metadata.MetadataError, match="version"):
        metadata.fetch_metadata(
            private_result_root / "metadata",
            result_root=private_result_root,
            open_url=fixture_opener(bodies),
        )


def test_metadata_rejects_malformed_agent_browser_node_engine_metadata(
    private_result_root,
):
    payloads = json.loads(response_fixtures()["agent_browser_npm_latest"])
    payloads["engines"] = "node >=22"
    bodies = response_fixtures()
    bodies["agent_browser_npm_latest"] = json.dumps(payloads).encode()

    with pytest.raises(metadata.MetadataError, match="engines"):
        metadata.fetch_metadata(
            private_result_root / "metadata",
            result_root=private_result_root,
            open_url=fixture_opener(bodies),
        )


def test_metadata_rejects_chrome_manifest_without_mac_arm64_stable_download(
    private_result_root,
):
    payloads = json.loads(response_fixtures()["chrome_for_testing_stable"])
    payloads["channels"]["Stable"]["downloads"]["chrome"] = [
        item
        for item in payloads["channels"]["Stable"]["downloads"]["chrome"]
        if item["platform"] != "mac-arm64"
    ]
    bodies = response_fixtures()
    bodies["chrome_for_testing_stable"] = json.dumps(payloads).encode()

    with pytest.raises(metadata.MetadataError, match="mac-arm64"):
        metadata.fetch_metadata(
            private_result_root / "metadata",
            result_root=private_result_root,
            open_url=fixture_opener(bodies),
        )


def test_fetch_refuses_to_overwrite_an_existing_metadata_snapshot(private_result_root):
    output = private_result_root / "metadata"
    output.mkdir(mode=0o700)
    existing = output / metadata.SOURCE_FILES["lightpanda_github_latest"]
    existing.write_bytes(b"historical snapshot")
    requested = []

    with pytest.raises(metadata.MetadataError, match="overwrite"):
        metadata.fetch_metadata(
            output,
            result_root=private_result_root,
            open_url=fixture_opener(response_fixtures(), requested),
        )

    assert existing.read_bytes() == b"historical snapshot"
    assert requested == []


def test_metadata_parser_rejects_unknown_manifest_keys_and_tampered_source_bytes(
    private_result_root,
):
    output = private_result_root / "metadata"
    bodies = response_fixtures()
    metadata.fetch_metadata(
        output, result_root=private_result_root, open_url=fixture_opener(bodies)
    )
    manifest_path = output / metadata.METADATA_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(metadata.MetadataError, match="unknown or missing keys"):
        metadata.parse_metadata_sources(output)

    manifest.pop("unexpected")
    manifest_path.write_text(json.dumps(manifest))
    (output / metadata.SOURCE_FILES["browser_use_pypi"]).write_bytes(b"{}")
    with pytest.raises(metadata.MetadataError, match="SHA-256 mismatch"):
        metadata.parse_metadata_sources(output)


def test_download_verifies_official_binaries_and_labels_chrome_hashes_as_local(
    private_result_root,
):
    bodies = response_fixtures()
    metadata_dir = private_result_root / "metadata"
    metadata.fetch_metadata(
        metadata_dir, result_root=private_result_root, open_url=fixture_opener(bodies)
    )
    archive_url = "https://storage.googleapis.com/chrome-for-testing-public/158.0.8127.36/mac-arm64/chrome-mac-arm64.zip"
    downloads = {
        "https://github.com/lightpanda-io/browser/releases/download/1.0.4/lightpanda-aarch64-macos": LIGHTPANDA_BINARY,
        "https://github.com/vercel-labs/agent-browser/releases/download/v0.43.0/agent-browser-darwin-arm64": AGENT_BROWSER_BINARY,
        archive_url: chrome_archive(),
    }
    tools_dir = private_result_root / "tools"
    tools_dir.mkdir(mode=0o700)

    manifest = metadata.download_tools(
        metadata_dir,
        tools_dir,
        private_result_root / "tool-manifest.json",
        result_root=private_result_root,
        open_url=fixture_opener(downloads),
    )

    assert manifest["schema"] == "openbrowser-latest-tool-manifest/1"
    assert set(manifest["tools"]) == {
        "lightpanda",
        "agent-browser",
        "chrome_for_testing",
    }
    assert (
        manifest["tools"]["lightpanda"]["official_sha256"]
        == hashlib.sha256(LIGHTPANDA_BINARY).hexdigest()
    )
    assert (
        manifest["tools"]["agent-browser"]["official_sha256"]
        == hashlib.sha256(AGENT_BROWSER_BINARY).hexdigest()
    )
    chrome = manifest["tools"]["chrome_for_testing"]
    assert "official_sha256" not in chrome
    assert chrome["archive_sha256"] == hashlib.sha256(chrome_archive()).hexdigest()
    assert chrome["local_sha256"] == hashlib.sha256(CHROME_BINARY).hexdigest()
    for name in ("lightpanda", "agent-browser", "chrome_for_testing"):
        assert Path(manifest["tools"][name]["path"]).is_file()
        assert Path(manifest["tools"][name]["path"]).stat().st_mode & 0o111


def test_download_rejects_a_binary_that_does_not_match_its_official_release_digest(
    private_result_root,
):
    output = private_result_root / "metadata"
    bodies = response_fixtures()
    metadata.fetch_metadata(
        output, result_root=private_result_root, open_url=fixture_opener(bodies)
    )
    archive_url = "https://storage.googleapis.com/chrome-for-testing-public/158.0.8127.36/mac-arm64/chrome-mac-arm64.zip"
    downloads = {
        "https://github.com/lightpanda-io/browser/releases/download/1.0.4/lightpanda-aarch64-macos": LIGHTPANDA_BINARY[
            :-1
        ]
        + b"X",
        "https://github.com/vercel-labs/agent-browser/releases/download/v0.43.0/agent-browser-darwin-arm64": AGENT_BROWSER_BINARY,
        archive_url: chrome_archive(),
    }
    tools_dir = private_result_root / "tools"
    tools_dir.mkdir(mode=0o700)

    with pytest.raises(metadata.MetadataError, match="official SHA-256 mismatch"):
        metadata.download_tools(
            output,
            tools_dir,
            private_result_root / "tool-manifest.json",
            result_root=private_result_root,
            open_url=fixture_opener(downloads),
        )

    assert not (tools_dir / "lightpanda").exists()
    assert not (private_result_root / "tool-manifest.json").exists()


@pytest.fixture
def tool_transaction_inputs(private_result_root):
    metadata_dir = private_result_root / "metadata"
    metadata.fetch_metadata(
        metadata_dir,
        result_root=private_result_root,
        open_url=fixture_opener(response_fixtures()),
    )
    tools = private_result_root / "tools"
    tools.mkdir(mode=0o700)
    (tools / "preexisting").write_bytes(b"preexisting data")
    archive = io.BytesIO(chrome_archive())
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("chrome-mac-arm64/ABOUT", b"nonexecuted Chrome resource")
        directory = zipfile.ZipInfo("chrome-mac-arm64/Empty/")
        directory.external_attr = (stat.S_IFDIR | 0o700) << 16
        bundle.writestr(directory, b"")
        link = zipfile.ZipInfo("chrome-mac-arm64/Current")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(link, "ABOUT")
    downloads = {
        "https://github.com/lightpanda-io/browser/releases/download/1.0.4/lightpanda-aarch64-macos": LIGHTPANDA_BINARY,
        "https://github.com/vercel-labs/agent-browser/releases/download/v0.43.0/agent-browser-darwin-arm64": AGENT_BROWSER_BINARY,
        "https://storage.googleapis.com/chrome-for-testing-public/158.0.8127.36/mac-arm64/chrome-mac-arm64.zip": archive.getvalue(),
    }
    return metadata_dir, tools, private_result_root / "tool-manifest.json", downloads


def fresh_installer():
    spec = importlib.util.spec_from_file_location("fresh_tool_installer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def no_preparation(*_args, **_kwargs):
    pytest.fail("selected retry performed network or version preparation")


def selected_cache(root):
    receipt = json.loads((root / metadata._INSTALL_RECEIPT).read_bytes())
    return root / f".tool-install-cache-{receipt['cache_id']}"


def pause_selected(root, inputs, monkeypatch, *, moved_count=0):
    metadata_dir, tools, manifest_path, downloads = inputs
    move, link = metadata._rename_noreplace, metadata.os.link
    roles = metadata._INSTALL_ROLES

    def pause_move(source, destination):
        if destination.name == roles[moved_count] and moved_count < len(roles):
            raise OSError("selected pause")
        return move(source, destination)

    def pause_commit(source, destination, **kwargs):
        if destination == manifest_path and moved_count == len(roles):
            raise OSError("selected pause")
        return link(source, destination, **kwargs)

    with monkeypatch.context() as patch:
        if moved_count < len(roles):
            patch.setattr(metadata, "_rename_noreplace", pause_move)
        patch.setattr(metadata.os, "link", pause_commit)
        with pytest.raises(OSError, match="selected pause"):
            metadata.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=root,
                open_url=fixture_opener(downloads),
            )
    return selected_cache(root)


@pytest.mark.parametrize("phase", ["receipt", *metadata._INSTALL_ROLES, "manifest"])
@pytest.mark.parametrize("timing", ["before", "after"])
@pytest.mark.parametrize("fault", [OSError, KeyboardInterrupt])
def test_all_publication_boundaries_resume_from_persisted_positions(
    private_result_root, tool_transaction_inputs, monkeypatch, phase, timing, fault
):
    metadata_dir, tools, manifest_path, downloads = tool_transaction_inputs
    move, link = metadata._rename_noreplace, metadata.os.link
    fired = []

    def boundary(name, operation, *args, **kwargs):
        if name == phase and timing == "before":
            fired.append(True)
            raise fault("injected publication boundary")
        result = operation(*args, **kwargs)
        if name == phase and timing == "after":
            fired.append(True)
            raise fault("injected publication boundary")
        return result

    def move_boundary(source, destination):
        return boundary(destination.name, move, source, destination)

    def link_boundary(source, destination, **kwargs):
        name = (
            "receipt" if destination.name == metadata._INSTALL_RECEIPT else "manifest"
        )
        return boundary(name, link, source, destination, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "_rename_noreplace", move_boundary)
        patch.setattr(metadata.os, "link", link_boundary)
        with pytest.raises(fault, match="injected publication boundary"):
            metadata.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=private_result_root,
                open_url=fixture_opener(downloads),
            )
    assert fired
    module = fresh_installer()
    selected = not (phase == "receipt" and timing == "before")
    with monkeypatch.context() as patch:
        if selected:
            patch.setattr(module, "_isolated_version", no_preparation)
        result = module.download_tools(
            metadata_dir,
            tools,
            manifest_path,
            result_root=private_result_root,
            open_url=no_preparation if selected else fixture_opener(downloads),
        )
    assert result["schema"] == "openbrowser-latest-tool-manifest/1"
    assert (tools / "preexisting").read_bytes() == b"preexisting data"
    assert (tools / "chrome-mac-arm64/Current").is_symlink()
    assert (
        tools / "chrome-mac-arm64/Current"
    ).read_bytes() == b"nonexecuted Chrome resource"
    assert (
        manifest_path.read_bytes()
        == (
            selected_cache(private_result_root) / metadata._INSTALL_MANIFEST
        ).read_bytes()
    )


@pytest.mark.parametrize("moved_count", range(5))
@pytest.mark.parametrize("inspection", ["source", "destination", "receipt", "manifest"])
def test_transient_inspection_failure_is_unknown_and_fresh_retry_resumes(
    private_result_root, tool_transaction_inputs, monkeypatch, moved_count, inspection
):
    cache = pause_selected(
        private_result_root,
        tool_transaction_inputs,
        monkeypatch,
        moved_count=moved_count,
    )
    metadata_dir, tools, manifest_path, _downloads = tool_transaction_inputs
    role = metadata._INSTALL_ROLES[min(moved_count, 3)]
    target = {
        "source": cache / role,
        "destination": tools / role,
        "receipt": private_result_root / metadata._INSTALL_RECEIPT,
        "manifest": manifest_path,
    }[inspection]
    optional = metadata._install_optional_stat

    def cannot_inspect(path):
        if path == target:
            raise PermissionError("transient inspection failure")
        return optional(path)

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "_install_optional_stat", cannot_inspect)
        patch.setattr(metadata, "_rename_noreplace", no_preparation)
        with pytest.raises(PermissionError, match="transient inspection failure"):
            metadata.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=private_result_root,
                open_url=no_preparation,
            )
    module = fresh_installer()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_isolated_version", no_preparation)
        result = module.download_tools(
            metadata_dir,
            tools,
            manifest_path,
            result_root=private_result_root,
            open_url=no_preparation,
        )
    assert all(Path(tool["path"]).is_file() for tool in result["tools"].values())


@pytest.mark.parametrize("stage", ["download", "extract", "manifest", "plan"])
@pytest.mark.parametrize("timing", ["before", "after"])
@pytest.mark.parametrize("fault", [OSError, KeyboardInterrupt])
def test_unselected_preparations_are_retained_and_retry_uses_a_fresh_generation(
    private_result_root, tool_transaction_inputs, monkeypatch, stage, timing, fault
):
    metadata_dir, tools, manifest_path, downloads = tool_transaction_inputs
    write, download, extract = (
        metadata._write_retained_file,
        metadata._download_file,
        metadata._extract_chrome_archive,
    )
    touched = []

    def effect(name, operation, *args, **kwargs):
        if name == stage and not touched:
            touched.append(True)
            if timing == "before":
                raise fault("interrupted preparation")
            operation(*args, **kwargs)
            raise fault("interrupted preparation")
        return operation(*args, **kwargs)

    def write_boundary(path, body, **kwargs):
        name = "plan" if path.name == metadata._INSTALL_PLAN else "manifest"
        return effect(name, write, path, body, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "_write_retained_file", write_boundary)
        patch.setattr(
            metadata,
            "_download_file",
            lambda *args, **kwargs: effect("download", download, *args, **kwargs),
        )
        patch.setattr(
            metadata,
            "_extract_chrome_archive",
            lambda *args, **kwargs: effect("extract", extract, *args, **kwargs),
        )
        with pytest.raises(fault, match="interrupted preparation"):
            metadata.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=private_result_root,
                open_url=fixture_opener(downloads),
            )
    generations = set(private_result_root.glob(".tool-install-cache-*"))
    assert generations
    assert not (private_result_root / metadata._INSTALL_RECEIPT).exists()
    assert list(tools.iterdir()) == [tools / "preexisting"]
    result = fresh_installer().download_tools(
        metadata_dir,
        tools,
        manifest_path,
        result_root=private_result_root,
        open_url=fixture_opener(downloads),
    )
    assert all(path.is_dir() for path in generations)
    assert (
        len(set(private_result_root.glob(".tool-install-cache-*")))
        == len(generations) + 1
    )
    assert Path(result["tools"]["lightpanda"]["path"]).read_bytes() == LIGHTPANDA_BINARY


@pytest.mark.parametrize(
    "mutation",
    [
        "resource",
        "extra",
        "same_bytes_inode",
        "bad_plan",
        "duplicate_plan",
        "boolean_identity",
        "witness",
        "foreign_receipt",
        "cache_parent",
        "tools_parent",
    ],
)
def test_selected_corruption_is_preserved_as_conflict_without_repair(
    private_result_root, tool_transaction_inputs, monkeypatch, mutation
):
    metadata_dir, tools, manifest_path, _downloads = tool_transaction_inputs
    cache = pause_selected(private_result_root, tool_transaction_inputs, monkeypatch)
    plan_path = cache / metadata._INSTALL_PLAN
    if mutation == "resource":
        (cache / "chrome-mac-arm64/ABOUT").write_bytes(b"foreign resource bytes")
    elif mutation == "extra":
        (cache / "chrome-mac-arm64/foreign").write_bytes(b"foreign extra")
    elif mutation == "same_bytes_inode":
        path = cache / "lightpanda"
        body = path.read_bytes()
        path.rename(cache / "old-lightpanda")
        path.write_bytes(body)
    elif mutation == "bad_plan":
        plan_path.write_bytes(b"{")
    elif mutation == "duplicate_plan":
        raw = plan_path.read_bytes()
        plan_path.write_bytes(b'{"schema":"ignored",' + raw[1:])
    elif mutation == "boolean_identity":
        plan = json.loads(plan_path.read_bytes())
        plan["slots"]["lightpanda"][0]["size"] = True
        plan_path.write_text(json.dumps(plan))
    elif mutation == "witness":
        (cache / metadata._INSTALL_MANIFEST).write_bytes(b"{}")
    elif mutation == "foreign_receipt":
        receipt = private_result_root / metadata._INSTALL_RECEIPT
        raw = receipt.read_bytes()
        receipt.unlink()
        receipt.write_bytes(raw)
    elif mutation == "cache_parent":
        cache.rename(private_result_root / "saved-cache")
        cache.mkdir(mode=0o700)
    else:
        tools.rename(private_result_root / "saved-tools")
        tools.mkdir(mode=0o700)
        (tools / "foreign").write_bytes(b"foreign tools")
    before = {
        path: path.read_bytes()
        for parent in (cache, tools)
        for path in parent.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    with pytest.raises((metadata.MetadataError, OSError)):
        metadata.download_tools(
            metadata_dir,
            tools,
            manifest_path,
            result_root=private_result_root,
            open_url=no_preparation,
        )
    assert not manifest_path.exists()
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "kind", ["file", "empty_directory", "populated_directory", "symlink"]
)
def test_real_native_publication_collision_preserves_foreign_roles(
    private_result_root, tool_transaction_inputs, monkeypatch, kind
):
    metadata_dir, tools, manifest_path, _downloads = tool_transaction_inputs
    cache = pause_selected(
        private_result_root, tool_transaction_inputs, monkeypatch, moved_count=3
    )
    destination = tools / "chrome-mac-arm64"
    protected = private_result_root / "foreign-proof"
    protected.write_bytes(b"foreign data")
    move = metadata._rename_noreplace
    identities = []

    def collision(source, target):
        if target == destination:
            if kind == "file":
                target.write_bytes(b"foreign data")
            elif kind == "symlink":
                target.symlink_to(protected)
            else:
                target.mkdir(mode=0o700)
                if kind == "populated_directory":
                    (target / "proof").write_bytes(b"foreign data")
            identities.append(target.lstat())
        move(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "_rename_noreplace", collision)
        with pytest.raises(OSError):
            metadata.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=private_result_root,
                open_url=no_preparation,
            )
    assert metadata.os.path.samestat(destination.lstat(), identities[0])
    assert (cache / "chrome-mac-arm64").is_dir()
    assert protected.read_bytes() == b"foreign data"
    with pytest.raises(metadata.MetadataError, match="exactly one"):
        metadata.download_tools(
            metadata_dir,
            tools,
            manifest_path,
            result_root=private_result_root,
            open_url=no_preparation,
        )
    assert not manifest_path.exists()


def test_committed_plan_retry_forbids_all_installer_mutation_and_probing(
    private_result_root, tool_transaction_inputs, monkeypatch
):
    metadata_dir, tools, manifest_path, downloads = tool_transaction_inputs
    expected = metadata.download_tools(
        metadata_dir,
        tools,
        manifest_path,
        result_root=private_result_root,
        open_url=fixture_opener(downloads),
    )
    module = fresh_installer()
    write_open = module.os.open

    def read_only_open(path, flags, *args, **kwargs):
        assert (
            flags
            & (
                module.os.O_WRONLY
                | module.os.O_RDWR
                | module.os.O_CREAT
                | module.os.O_TRUNC
            )
            == 0
        )
        return write_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        for name in (
            "_isolated_version",
            "_rename_noreplace",
            "_write_retained_file",
            "_download_file",
        ):
            patch.setattr(module, name, no_preparation)
        for name in ("link", "unlink", "rmdir", "chmod", "fchmod", "rename", "replace"):
            patch.setattr(module.os, name, no_preparation)
        patch.setattr(module.os, "open", read_only_open)
        patch.setattr(Path, "mkdir", no_preparation)
        assert (
            module.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=private_result_root,
                open_url=no_preparation,
            )
            == expected
        )


def test_tools_directory_must_preexist_before_any_fetch(private_result_root):
    metadata_dir = private_result_root / "metadata"
    metadata.fetch_metadata(
        metadata_dir,
        result_root=private_result_root,
        open_url=fixture_opener(response_fixtures()),
    )
    with pytest.raises(metadata.MetadataError, match="does not exist"):
        metadata.download_tools(
            metadata_dir,
            private_result_root / "tools",
            private_result_root / "tool-manifest.json",
            result_root=private_result_root,
            open_url=no_preparation,
        )
    assert not (private_result_root / "tools").exists()


def test_version_probe_scratch_is_outside_payload_for_installer_and_legacy_consumer(
    private_result_root, tool_transaction_inputs, monkeypatch
):
    metadata_dir, tools, manifest_path, downloads = tool_transaction_inputs
    run = metadata.subprocess.run
    seen = []

    def inspect_scratch(command, **kwargs):
        executable = Path(command[0])
        scratch = Path(kwargs["cwd"])
        assert scratch.parent == private_result_root
        assert not executable.is_relative_to(scratch) and not scratch.is_relative_to(
            executable.parent
        )
        assert kwargs["env"]["HOME"] == kwargs["env"]["TMPDIR"] == str(scratch)
        seen.append(executable)
        return run(command, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(metadata.subprocess, "run", inspect_scratch)
        manifest = metadata.download_tools(
            metadata_dir,
            tools,
            manifest_path,
            result_root=private_result_root,
            open_url=fixture_opener(downloads),
        )
        spec = importlib.util.spec_from_file_location(
            "installer_consumer_roundtrip",
            SCRIPT.with_name("latest_benchmark_preflight.py"),
        )
        consumer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(consumer)
        assert (
            consumer.load_tool_manifest(
                manifest_path, result_root=private_result_root, verify_installation=True
            )
            == manifest
        )
        (private_result_root / metadata._INSTALL_RECEIPT).unlink()
        assert (
            consumer.load_tool_manifest(
                manifest_path, result_root=private_result_root, verify_installation=True
            )
            == manifest
        )
    assert len(seen) == 9
    with pytest.raises(metadata.MetadataError, match="without an owned receipt"):
        metadata.download_tools(
            metadata_dir,
            tools,
            manifest_path,
            result_root=private_result_root,
            open_url=no_preparation,
        )


@pytest.mark.parametrize(
    "number", [metadata.errno.EXDEV, metadata.errno.EINVAL, metadata.errno.ENOSYS]
)
def test_native_exclusive_move_errno_has_no_unsafe_fallback(
    tmp_path, monkeypatch, number
):
    class Native:
        def __call__(self, *_args):
            metadata.ctypes.set_errno(number)
            return -1

    class Library:
        renamex_np = Native()

    monkeypatch.setattr(metadata.sys, "platform", "darwin")
    monkeypatch.setattr(metadata.ctypes, "CDLL", lambda *_args, **_kwargs: Library())
    monkeypatch.setattr(metadata.os, "rename", no_preparation)
    with pytest.raises(OSError) as failure:
        metadata._rename_noreplace(tmp_path / "source", tmp_path / "destination")
    assert failure.value.errno == number


def test_logging_failure_does_not_replace_publication_interrupt(
    private_result_root, tool_transaction_inputs, monkeypatch
):
    metadata_dir, tools, manifest_path, downloads = tool_transaction_inputs

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt("original publication interruption")

    def log_failed(*_args, **_kwargs):
        raise RuntimeError("secondary log error")

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "_rename_noreplace", interrupted)
        patch.setattr(metadata.LOGGER, "warning", log_failed)
        with pytest.raises(
            KeyboardInterrupt, match="original publication interruption"
        ):
            metadata.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=private_result_root,
                open_url=fixture_opener(downloads),
            )
    result = fresh_installer().download_tools(
        metadata_dir,
        tools,
        manifest_path,
        result_root=private_result_root,
        open_url=no_preparation,
    )
    assert Path(result["tools"]["lightpanda"]["path"]).is_file()


@pytest.mark.parametrize("source_kind", ["file", "directory"])
@pytest.mark.parametrize("destination_kind", ["file", "directory"])
def test_native_exclusive_move_preserves_actual_existing_destinations(
    tmp_path, source_kind, destination_kind
):
    move = getattr(metadata, "_rename_noreplace", None)
    assert callable(move), "native exclusive move is missing"
    source, destination = tmp_path / "source", tmp_path / "destination"
    if source_kind == "directory":
        source.mkdir(mode=0o700)
        (source / "proof").write_bytes(b"owned source")
    else:
        source.write_bytes(b"owned source")
    if destination_kind == "directory":
        destination.mkdir(mode=0o700)
    else:
        destination.write_bytes(b"foreign destination")
    source_identity, destination_identity = source.lstat(), destination.lstat()
    with pytest.raises(OSError):
        move(source, destination)
    assert metadata.os.path.samestat(source.lstat(), source_identity)
    assert metadata.os.path.samestat(destination.lstat(), destination_identity)
    if destination_kind == "file":
        assert destination.read_bytes() == b"foreign destination"
    else:
        assert list(destination.iterdir()) == []


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_native_exclusive_move_keeps_preknown_tree_identities(tmp_path, kind):
    move = getattr(metadata, "_rename_noreplace", None)
    assert callable(move), "native exclusive move is missing"
    source, destination = tmp_path / "source", tmp_path / "destination"
    if kind == "directory":
        source.mkdir(mode=0o700)
        (source / "proof").write_bytes(b"owned payload")
        (source / "alias").symlink_to("proof")
        nodes = [source, source / "proof", source / "alias"]
    else:
        source.write_bytes(b"owned payload")
        nodes = [source]
    records = [(node.relative_to(source), node.lstat()) for node in nodes]
    move(source, destination)
    assert not source.exists()
    for relative, before in records:
        after = (destination / relative).lstat()
        assert (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        ) == (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
    if kind == "directory":
        assert (destination / "alias").is_symlink()
        assert (destination / "alias").read_bytes() == b"owned payload"


@pytest.mark.parametrize("bad_path", [".", "bad\x00path"])
def test_installation_inventory_requires_nonaliased_canonical_paths(
    private_result_root, tool_transaction_inputs, monkeypatch, bad_path
):
    cache = pause_selected(private_result_root, tool_transaction_inputs, monkeypatch)
    plan = json.loads((cache / metadata._INSTALL_PLAN).read_bytes())
    nodes = plan["slots"]["chrome-mac-arm64"]
    nodes.append({**nodes[0], "path": bad_path})
    nodes.sort(key=lambda node: node["path"])
    with pytest.raises(metadata.MetadataError):
        metadata._install_nodes_shape(nodes, directory=True)


def test_retained_json_writer_rejects_oversize_before_creating_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(metadata, "_MAX_INSTALL_PLAN_BYTES", 128)
    path = tmp_path / "oversize.json"
    with pytest.raises(metadata.MetadataError, match="size limit"):
        metadata._write_retained_file(path, b"x" * 129)
    assert not path.exists()


@pytest.mark.parametrize(
    "mutation", ["resource", "extra", "foreign_manifest", "missing_payload"]
)
def test_committed_retry_rejects_corruption_without_repair(
    private_result_root, tool_transaction_inputs, monkeypatch, mutation
):
    metadata_dir, tools, manifest_path, downloads = tool_transaction_inputs
    metadata.download_tools(
        metadata_dir,
        tools,
        manifest_path,
        result_root=private_result_root,
        open_url=fixture_opener(downloads),
    )
    if mutation == "resource":
        (tools / "chrome-mac-arm64/ABOUT").write_bytes(b"foreign resource")
    elif mutation == "extra":
        (tools / "chrome-mac-arm64/foreign").write_bytes(b"foreign extra")
    elif mutation == "foreign_manifest":
        body = manifest_path.read_bytes()
        manifest_path.unlink()
        manifest_path.write_bytes(body)
    else:
        (tools / "lightpanda").rename(private_result_root / "saved-lightpanda")
    before = manifest_path.read_bytes()
    module = fresh_installer()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_isolated_version", no_preparation)
        patch.setattr(module, "_rename_noreplace", no_preparation)
        with pytest.raises(module.MetadataError):
            module.download_tools(
                metadata_dir,
                tools,
                manifest_path,
                result_root=private_result_root,
                open_url=no_preparation,
            )
    assert manifest_path.read_bytes() == before


def test_missing_native_symbol_fails_closed_without_ordinary_rename(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(metadata.ctypes, "CDLL", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(metadata.os, "rename", no_preparation)
    with pytest.raises(metadata.MetadataError, match="symbol is unavailable"):
        metadata._rename_noreplace(tmp_path / "source", tmp_path / "destination")


@pytest.mark.parametrize("kind", ["MetadataError", "OSError"])
def test_metadata_cli_reports_operational_failures_without_traceback(tmp_path, kind):
    program = (
        "import sys; "
        f"sys.path.insert(0, {str(SCRIPT.parent)!r}); "
        "import latest_competitor_metadata as metadata; "
        "kind = sys.argv[1]\n"
        "def fail(*args, **kwargs):\n"
        "    error_type = metadata.MetadataError if kind == 'MetadataError' else OSError\n"
        "    error = error_type('operational failure')\n"
        "    error.add_note('retained installation evidence')\n"
        "    raise error\n"
        "metadata.download_tools = fail\n"
        "raise SystemExit(metadata.main(['download-tools', '--result-root', 'root', "
        "'--metadata', 'metadata', '--tools', 'tools', '--manifest', 'manifest']))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, kind],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "operational failure" in result.stderr
    assert "retained installation evidence" in result.stderr
    assert "Traceback" not in result.stderr


def test_installation_json_rejects_fifo_without_blocking(tmp_path):
    path = tmp_path / "receipt.json"
    metadata.os.mkfifo(path, 0o600)
    original = path.lstat()
    program = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(SCRIPT.parent)!r}); "
        "import latest_competitor_metadata as metadata\n"
        "try:\n"
        "    metadata._install_json(Path(sys.argv[1]))\n"
        "except metadata.MetadataError as error:\n"
        "    sys.stderr.write(str(error))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )

    assert result.returncode == 0
    assert "unsafe installation JSON" in result.stderr
    assert metadata.os.path.samestat(original, path.lstat())
    assert stat.S_ISFIFO(path.lstat().st_mode)
