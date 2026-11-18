import hashlib
import importlib.util
import io
import json
import shutil
import stat
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
    manifest_path = tmp_path / "tool-manifest-outside-result-root.json"

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
