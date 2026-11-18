#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)
CLI_DESCRIPTION = "Fetch and validate current browser benchmark metadata with the standard library."
METADATA_SCHEMA = "openbrowser-latest-metadata/1"
TOOL_MANIFEST_SCHEMA = "openbrowser-latest-tool-manifest/1"
METADATA_MANIFEST_NAME = "metadata-manifest.json"
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 200_000
READ_CHUNK_SIZE = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
CHROME_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")

SOURCE_URLS = {
    "lightpanda_github_latest": "https://api.github.com/repos/lightpanda-io/browser/releases/latest",
    "agent_browser_npm_latest": "https://registry.npmjs.org/agent-browser/latest",
    "agent_browser_github_latest": "https://api.github.com/repos/vercel-labs/agent-browser/releases/latest",
    "browser_use_pypi": "https://pypi.org/pypi/browser-use/json",
    "browser_harness_pypi": "https://pypi.org/pypi/browser-harness/json",
    "chrome_for_testing_stable": "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json",
}

SOURCE_FILES = {
    "lightpanda_github_latest": "lightpanda-github-release.json",
    "agent_browser_npm_latest": "agent-browser-npm.json",
    "agent_browser_github_latest": "agent-browser-github-release.json",
    "browser_use_pypi": "browser-use-pypi.json",
    "browser_harness_pypi": "browser-harness-pypi.json",
    "chrome_for_testing_stable": "chrome-stable.json",
}

LIGHTPANDA_ASSET = "lightpanda-aarch64-macos"
AGENT_BROWSER_ASSET = "agent-browser-darwin-arm64"
CHROME_PLATFORM = "mac-arm64"
CHROME_ZIP_ROOT = "chrome-mac-arm64"
CHROME_EXECUTABLE_RELATIVE = Path(
    "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
)


class MetadataError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise MetadataError(f"{where} must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MetadataError(f"{where} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise MetadataError(f"{where} must use UTC")
    return value


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise MetadataError(f"{where} must be a JSON object with string keys")
    return value


def _require_nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"{where} must be a non-empty string")
    return value


def _require_version(value: Any, where: str) -> str:
    version = _require_nonempty_string(value, where)
    if not VERSION_RE.fullmatch(version):
        raise MetadataError(f"{where} is not a valid dotted release version")
    return version


def _validate_https_url(value: Any, where: str, *, host: str | None = None, prefix: str | None = None) -> str:
    url = _require_nonempty_string(value, where)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise MetadataError(f"{where} must be an absolute HTTPS URL")
    if host is not None and parsed.hostname.lower() != host.lower():
        raise MetadataError(f"{where} must use {host}")
    if prefix is not None and not url.startswith(prefix):
        raise MetadataError(f"{where} must begin with the expected release URL")
    return url


def _sha256_hex(value: Any, where: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise MetadataError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _release_version(tag: Any, where: str) -> str:
    version = _require_nonempty_string(tag, where)
    if version.startswith("v"):
        version = version[1:]
    if not VERSION_RE.fullmatch(version):
        raise MetadataError(f"{where} is not a valid release tag")
    return version


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(READ_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MetadataError(f"cannot read file for hashing: {path}") from exc
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json_payload(body: bytes, where: str) -> Any:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataError(f"{where} did not return valid UTF-8 JSON") from exc


def _read_response(open_url: Callable[..., Any], url: str, *, max_bytes: int = MAX_METADATA_BYTES) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json, application/octet-stream, application/zip", "User-Agent": "OpenBrowser-local-benchmark/1"},
    )
    try:
        response_context = open_url(request, timeout=60 if max_bytes > MAX_METADATA_BYTES else 30)
        with response_context as response:
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                status = response.getcode()
            if status is not None and (not isinstance(status, int) or status < 200 or status >= 300):
                raise MetadataError(f"HTTP request failed with status {status}: {url}")
            body = response.read(max_bytes + 1)
    except MetadataError:
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise MetadataError(f"required metadata or binary source is unavailable: {url}") from exc
    if not isinstance(body, bytes):
        raise MetadataError(f"response body is not bytes: {url}")
    if len(body) > max_bytes:
        raise MetadataError(f"response exceeds the size limit: {url}")
    return body


def _parse_github_release(payload: Any, *, source_name: str, asset_name: str) -> dict[str, Any]:
    release = _require_object(payload, source_name)
    version = _release_version(release.get("tag_name"), f"{source_name}.tag_name")
    if release.get("draft") is not False:
        raise MetadataError(f"{source_name} is not a published release")
    if release.get("prerelease") is not False:
        raise MetadataError(f"{source_name} is a prerelease, not the latest stable release")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise MetadataError(f"{source_name}.assets must be a list")
    selected = []
    for index, raw_asset in enumerate(assets):
        asset = _require_object(raw_asset, f"{source_name}.assets[{index}]")
        if asset.get("name") == asset_name:
            selected.append(asset)
    if len(selected) != 1:
        raise MetadataError(f"{source_name} must contain exactly one {asset_name} asset")
    asset = selected[0]
    url = _require_nonempty_string(asset.get("browser_download_url"), f"{source_name}.{asset_name}.browser_download_url")
    digest_value = asset.get("digest")
    if not isinstance(digest_value, str) or not digest_value.startswith("sha256:"):
        raise MetadataError(f"{source_name}.{asset_name} is missing its official SHA-256 digest")
    digest = _sha256_hex(digest_value.removeprefix("sha256:"), f"{source_name}.{asset_name}.digest")
    size = asset.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise MetadataError(f"{source_name}.{asset_name}.size must be a positive integer")
    if source_name == "lightpanda_github_latest":
        expected_prefix = "https://github.com/lightpanda-io/browser/releases/download/"
    else:
        expected_prefix = "https://github.com/vercel-labs/agent-browser/releases/download/"
    url = _validate_https_url(url, f"{source_name}.{asset_name}.browser_download_url", host="github.com", prefix=expected_prefix)
    return {"version": version, "asset_name": asset_name, "download_url": url, "official_sha256": digest, "size": size}


def _parse_pypi(payload: Any, *, source_name: str, expected_package: str) -> str:
    response = _require_object(payload, source_name)
    info = _require_object(response.get("info"), f"{source_name}.info")
    package = _require_nonempty_string(info.get("name"), f"{source_name}.info.name")
    if package.casefold().replace("_", "-") != expected_package:
        raise MetadataError(f"{source_name} returned package {package!r}, expected {expected_package!r}")
    return _require_version(info.get("version"), f"{source_name}.info.version")


def _parse_npm(payload: Any) -> tuple[str, str]:
    response = _require_object(payload, "agent_browser_npm_latest")
    package = _require_nonempty_string(response.get("name"), "agent_browser_npm_latest.name")
    if package != "agent-browser":
        raise MetadataError("agent_browser_npm_latest returned a different package")
    version = _require_version(response.get("version"), "agent_browser_npm_latest.version")
    distribution = _require_object(response.get("dist"), "agent_browser_npm_latest.dist")
    _validate_https_url(
        distribution.get("tarball"),
        "agent_browser_npm_latest.dist.tarball",
        host="registry.npmjs.org",
        prefix="https://registry.npmjs.org/agent-browser/",
    )
    integrity = _require_nonempty_string(distribution.get("integrity"), "agent_browser_npm_latest.dist.integrity")
    sha512_values = [token.removeprefix("sha512-") for token in integrity.split() if token.startswith("sha512-")]
    if not sha512_values:
        raise MetadataError("agent_browser_npm_latest.dist.integrity must use SHA-512 Subresource Integrity")
    for encoded_digest in sha512_values:
        try:
            digest = base64.b64decode(encoded_digest, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MetadataError("agent_browser_npm_latest.dist.integrity contains invalid SHA-512 Base64") from exc
        if len(digest) != hashlib.sha512().digest_size or base64.b64encode(digest).decode("ascii") != encoded_digest:
            raise MetadataError("agent_browser_npm_latest.dist.integrity must contain a canonical 64-byte SHA-512 digest")
    repository = _require_object(response.get("repository"), "agent_browser_npm_latest.repository")
    repository_url = _require_nonempty_string(repository.get("url"), "agent_browser_npm_latest.repository.url")
    if "github.com/vercel-labs/agent-browser" not in repository_url:
        raise MetadataError("agent_browser_npm_latest repository does not match the official agent-browser project")
    engines = _require_object(response.get("engines"), "agent_browser_npm_latest.engines")
    node_engine = _require_nonempty_string(engines.get("node"), "agent_browser_npm_latest.engines.node")
    return version, node_engine


def _parse_chrome_manifest(payload: Any) -> tuple[str, str]:
    response = _require_object(payload, "chrome_for_testing_stable")
    _require_nonempty_string(response.get("timestamp"), "chrome_for_testing_stable.timestamp")
    channels = _require_object(response.get("channels"), "chrome_for_testing_stable.channels")
    stable = _require_object(channels.get("Stable"), "chrome_for_testing_stable.channels.Stable")
    if stable.get("channel") != "Stable":
        raise MetadataError("Chrome for Testing response does not identify the Stable channel")
    version = _require_nonempty_string(stable.get("version"), "chrome_for_testing_stable.channels.Stable.version")
    if not CHROME_VERSION_RE.fullmatch(version):
        raise MetadataError("Chrome Stable version must contain four numeric components")
    downloads = _require_object(stable.get("downloads"), "chrome_for_testing_stable.channels.Stable.downloads")
    chrome_assets = downloads.get("chrome")
    if not isinstance(chrome_assets, list):
        raise MetadataError("Chrome Stable downloads.chrome must be a list")
    selected = []
    for index, raw_asset in enumerate(chrome_assets):
        asset = _require_object(raw_asset, f"chrome_for_testing_stable.downloads.chrome[{index}]")
        if asset.get("platform") == CHROME_PLATFORM:
            selected.append(asset)
    if len(selected) != 1:
        raise MetadataError("Chrome Stable manifest must contain exactly one mac-arm64 download")
    url = _validate_https_url(
        selected[0].get("url"),
        "chrome_for_testing_stable.downloads.chrome.mac-arm64.url",
        host="storage.googleapis.com",
        prefix=f"https://storage.googleapis.com/chrome-for-testing-public/{version}/mac-arm64/",
    )
    if not url.endswith("/chrome-mac-arm64.zip"):
        raise MetadataError("Chrome Stable mac-arm64 URL is not the expected Chrome archive")
    return version, url


def _parse_payloads(payloads: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    lightpanda = _parse_github_release(
        payloads["lightpanda_github_latest"],
        source_name="lightpanda_github_latest",
        asset_name=LIGHTPANDA_ASSET,
    )
    agent_release = _parse_github_release(
        payloads["agent_browser_github_latest"],
        source_name="agent_browser_github_latest",
        asset_name=AGENT_BROWSER_ASSET,
    )
    agent_npm_version, agent_node_engine = _parse_npm(payloads["agent_browser_npm_latest"])
    if agent_npm_version != agent_release["version"]:
        raise MetadataError(
            "agent-browser NPM latest version does not match its GitHub latest release version"
        )
    browser_use = _parse_pypi(payloads["browser_use_pypi"], source_name="browser_use_pypi", expected_package="browser-use")
    browser_harness = _parse_pypi(
        payloads["browser_harness_pypi"],
        source_name="browser_harness_pypi",
        expected_package="browser-harness",
    )
    chrome_version, chrome_url = _parse_chrome_manifest(payloads["chrome_for_testing_stable"])
    return {
        "latest_versions": {
            "lightpanda": lightpanda["version"],
            "agent-browser": agent_release["version"],
            "browser-use": browser_use,
            "browser-harness": browser_harness,
            "chrome_for_testing": chrome_version,
        },
        "agent_browser_node_engine": agent_node_engine,
        "chrome_download_url": chrome_url,
        "release_assets": {"lightpanda": lightpanda, "agent-browser": agent_release},
        "metadata_sources": sources,
    }


def _read_metadata_manifest(metadata_dir: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(metadata_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise MetadataError(f"metadata path must be a real directory: {directory}")
    manifest_path = directory / METADATA_MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = _require_object(_read_json_payload(manifest_bytes, METADATA_MANIFEST_NAME), METADATA_MANIFEST_NAME)
    except OSError as exc:
        raise MetadataError(f"metadata manifest is unavailable: {manifest_path}") from exc
    expected = {"schema", "fetched_at_utc", "sources"}
    if set(manifest) != expected:
        raise MetadataError(f"{METADATA_MANIFEST_NAME} has unknown or missing keys")
    if manifest.get("schema") != METADATA_SCHEMA:
        raise MetadataError(f"{METADATA_MANIFEST_NAME} has an unsupported schema")
    _validate_timestamp(manifest.get("fetched_at_utc"), f"{METADATA_MANIFEST_NAME}.fetched_at_utc")
    sources = _require_object(manifest.get("sources"), f"{METADATA_MANIFEST_NAME}.sources")
    if set(sources) != set(SOURCE_URLS):
        raise MetadataError(f"{METADATA_MANIFEST_NAME}.sources has unknown or missing sources")
    payloads: dict[str, Any] = {}
    validated_sources: dict[str, dict[str, str]] = {}
    for name, url in SOURCE_URLS.items():
        record = _require_object(sources[name], f"{METADATA_MANIFEST_NAME}.sources.{name}")
        if set(record) != {"url", "file", "response_sha256"}:
            raise MetadataError(f"{METADATA_MANIFEST_NAME}.sources.{name} has unknown or missing keys")
        if record.get("url") != url or record.get("file") != SOURCE_FILES[name]:
            raise MetadataError(f"{METADATA_MANIFEST_NAME}.sources.{name} does not match the official source contract")
        response_sha = _sha256_hex(record.get("response_sha256"), f"{METADATA_MANIFEST_NAME}.sources.{name}.response_sha256")
        source_path = directory / SOURCE_FILES[name]
        if source_path.is_symlink() or not source_path.is_file():
            raise MetadataError(f"required metadata response is missing or not a regular file: {source_path}")
        try:
            body = source_path.read_bytes()
        except OSError as exc:
            raise MetadataError(f"cannot read metadata response: {source_path}") from exc
        if sha256_bytes(body) != response_sha:
            raise MetadataError(f"metadata response SHA-256 mismatch for {name}")
        payloads[name] = _read_json_payload(body, name)
        validated_sources[name] = {"url": url, "response_sha256": response_sha}
    parsed = _parse_payloads(payloads, validated_sources)
    parsed["metadata_manifest_sha256"] = sha256_bytes(manifest_bytes)
    parsed["fetched_at_utc"] = manifest["fetched_at_utc"]
    return manifest, parsed


def parse_metadata_sources(metadata_dir: str | os.PathLike[str]) -> dict[str, Any]:
    _manifest, parsed = _read_metadata_manifest(metadata_dir)
    return parsed


def _open_url(open_url: Callable[..., Any] | None) -> Callable[..., Any]:
    return open_url if open_url is not None else urllib.request.urlopen


def _write_new_file(path: Path, body: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_identity = path.parent.lstat()
        descriptor = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise MetadataError(f"refusing to overwrite existing artifact: {path}") from exc
    except OSError as exc:
        raise MetadataError(f"cannot create artifact: {path}") from exc
    created_identity = None
    try:
        with os.fdopen(descriptor, "wb") as output:
            created_identity = os.fstat(output.fileno())
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
    except BaseException as exc:
        try:
            if (
                created_identity is not None
                and os.path.samestat(path.lstat(), created_identity)
                and os.path.samestat(path.parent.lstat(), parent_identity)
            ):
                path.unlink()
        except OSError:
            pass
        if isinstance(exc, OSError):
            raise MetadataError(f"cannot write artifact: {path}") from exc
        raise


def _require_private_result_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute() or ".." in root.parts or root.is_symlink():
        raise MetadataError("RESULT_ROOT must be an absolute physical directory without parent traversal")
    try:
        resolved = root.resolve(strict=True)
        physical_tmp = Path("/tmp").resolve(strict=True)
    except OSError as exc:
        raise MetadataError("RESULT_ROOT and physical /tmp must exist") from exc
    if Path(os.path.abspath(root)) != resolved or resolved.parent != physical_tmp:
        raise MetadataError("RESULT_ROOT must be a direct child of physical /tmp")
    status = resolved.stat()
    if stat.S_IMODE(status.st_mode) != 0o700:
        raise MetadataError("RESULT_ROOT must have mode 0700")
    if status.st_uid != os.getuid():
        raise MetadataError("RESULT_ROOT must be owned by the current user UID")
    return resolved


def _result_path(path: str | os.PathLike[str], root: Path, where: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise MetadataError(f"{where} path must be absolute")
    if ".." in raw.parts:
        raise MetadataError(f"{where} path must not contain parent traversal")
    if raw.is_symlink():
        raise MetadataError(f"{where} path must not be a symlink")
    resolved = raw.resolve(strict=False)
    if Path(os.path.abspath(raw)) != resolved:
        raise MetadataError(f"{where} path must not contain symlink components")
    if resolved == root or not resolved.is_relative_to(root):
        raise MetadataError(f"{where} must be below RESULT_ROOT")
    return resolved


def _result_directory(
    path: str | os.PathLike[str],
    root: Path,
    where: str,
    *,
    create: bool,
    require_exists: bool,
) -> Path:
    directory = _result_path(path, root, where)
    current = root
    for part in directory.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise MetadataError(f"{where} must not contain symlink components")
        if not current.exists():
            if create:
                try:
                    current.mkdir(mode=0o700)
                except OSError as exc:
                    raise MetadataError(f"cannot create private {where} directory: {current}") from exc
            elif require_exists:
                raise MetadataError(f"{where} does not exist: {current}")
            else:
                break
        if current.exists():
            status = current.stat()
            if not stat.S_ISDIR(status.st_mode):
                raise MetadataError(f"{where} must be a directory: {current}")
            if stat.S_IMODE(status.st_mode) != 0o700:
                raise MetadataError(f"{where} directories must have mode 0700")
            if status.st_uid != os.getuid():
                raise MetadataError(f"{where} directories must be owned by the current user UID")
    if require_exists and not directory.is_dir():
        raise MetadataError(f"{where} must be a directory: {directory}")
    return directory


def fetch_metadata(
    output: str | os.PathLike[str],
    *,
    result_root: str | os.PathLike[str],
    open_url: Callable[..., Any] | None = None,
    fetched_at_utc: str | None = None,
) -> dict[str, Any]:
    root = _require_private_result_root(result_root)
    output_dir = _result_directory(output, root, "metadata output", create=True, require_exists=True)
    targets = [output_dir / METADATA_MANIFEST_NAME] + [output_dir / name for name in SOURCE_FILES.values()]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise MetadataError("refusing to overwrite an existing metadata snapshot")

    fetch = _open_url(open_url)
    payloads: dict[str, Any] = {}
    bodies: dict[str, bytes] = {}
    sources: dict[str, dict[str, str]] = {}
    for name, url in SOURCE_URLS.items():
        body = _read_response(fetch, url)
        payload = _read_json_payload(body, name)
        bodies[name] = body
        payloads[name] = payload
        sources[name] = {
            "url": url,
            "file": SOURCE_FILES[name],
            "response_sha256": sha256_bytes(body),
        }
    _parse_payloads(payloads, {name: {"url": record["url"], "response_sha256": record["response_sha256"]} for name, record in sources.items()})
    timestamp = _validate_timestamp(fetched_at_utc if fetched_at_utc is not None else _utc_now(), "fetched_at_utc")
    manifest = {"schema": METADATA_SCHEMA, "fetched_at_utc": timestamp, "sources": sources}

    for name, file_name in SOURCE_FILES.items():
        _write_new_file(output_dir / file_name, bodies[name])
    _write_new_file(output_dir / METADATA_MANIFEST_NAME, _json_bytes(manifest))
    LOGGER.info("saved validated metadata from %d official sources", len(SOURCE_URLS))
    return manifest


def _download_file(
    open_url: Callable[..., Any],
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    mode: int = 0o600,
) -> tuple[str, int]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream, application/zip", "User-Agent": "OpenBrowser-local-benchmark/1"},
    )
    if destination.exists() or destination.is_symlink():
        raise MetadataError(f"refusing to overwrite existing download: {destination}")
    digest = hashlib.sha256()
    total = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        response_context = open_url(request, timeout=60)
        with response_context as response:
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                status = response.getcode()
            if status is not None and (not isinstance(status, int) or status < 200 or status >= 300):
                raise MetadataError(f"HTTP download failed with status {status}: {url}")
            descriptor = os.open(destination, flags, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                os.fchmod(output.fileno(), mode)
                while True:
                    chunk = response.read(READ_CHUNK_SIZE)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise MetadataError(f"download response was not bytes: {url}")
                    total += len(chunk)
                    if total > max_bytes:
                        raise MetadataError(f"download exceeds size limit: {url}")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
    except MetadataError:
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise MetadataError(f"required binary download is unavailable: {url}") from exc
    checksum = digest.hexdigest()
    if expected_size is not None and total != expected_size:
        raise MetadataError(f"download size mismatch for {url}: expected {expected_size}, got {total}")
    if expected_sha256 is not None and checksum != expected_sha256:
        raise MetadataError(f"official SHA-256 mismatch for downloaded asset: {url}")
    return checksum, total


def _extract_chrome_archive(archive_path: Path, extraction_root: Path) -> Path:
    if extraction_root.exists() or extraction_root.is_symlink():
        raise MetadataError(f"refusing to overwrite existing Chrome extraction: {extraction_root}")
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise MetadataError("Chrome for Testing download is not a valid ZIP archive") from exc
    extraction_root.mkdir(mode=0o700, parents=True)
    physical_root = extraction_root.resolve(strict=True)
    links = []
    aggregate_size = 0
    entries = archive.infolist()
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        archive.close()
        raise MetadataError("Chrome archive contains too many entries")
    try:
        with archive:
            for info in entries:
                if "\\" in info.filename:
                    raise MetadataError("Chrome archive contains a non-POSIX path")
                name = PurePosixPath(info.filename)
                if name.is_absolute() or any(part in {"", ".", ".."} for part in name.parts):
                    raise MetadataError("Chrome archive contains an unsafe path")
                if not name.parts or name.parts[0] != CHROME_ZIP_ROOT:
                    raise MetadataError("Chrome archive contains an unexpected top-level directory")
                relative_parts = name.parts[1:]
                if not relative_parts:
                    continue
                destination = extraction_root.joinpath(*relative_parts)
                if not destination.resolve(strict=False).is_relative_to(extraction_root.resolve(strict=True)):
                    raise MetadataError("Chrome archive path escapes its extraction directory")
                mode = (info.external_attr >> 16) & 0xFFFF
                file_kind = stat.S_IFMT(mode)
                if file_kind == stat.S_IFLNK:
                    if info.file_size > 4096:
                        raise MetadataError('Chrome archive symbolic link target is too large')
                    target = archive.read(info).decode('utf-8')
                    if not target or '\x00' in target or '\\' in target or Path(target).is_absolute():
                        raise MetadataError('Chrome archive symbolic link target is unsafe')
                    target_path = Path(os.path.abspath(destination.parent / target))
                    if not target_path.is_relative_to(physical_root):
                        raise MetadataError('Chrome archive symbolic link escapes its extraction directory')
                    links.append((destination, target))
                    continue
                if info.is_dir():
                    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                    continue
                if file_kind not in {0, stat.S_IFREG}:
                    raise MetadataError("Chrome archive contains a non-regular file")
                aggregate_size += info.file_size
                if aggregate_size > MAX_ARCHIVE_BYTES:
                    raise MetadataError("Chrome archive expands beyond the allowed size")
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(destination, flags, 0o600)
                with archive.open(info, "r") as source, os.fdopen(descriptor, "wb") as output:
                    permissions = 0o755 if Path(*relative_parts) == CHROME_EXECUTABLE_RELATIVE else mode & 0o777 or 0o600
                    os.fchmod(output.fileno(), permissions)
                    shutil.copyfileobj(source, output, READ_CHUNK_SIZE)
            for destination, target in links:
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if destination.parent.resolve(strict=True) != destination.parent:
                    raise MetadataError('Chrome archive symbolic link parent is not a physical directory')
                destination.symlink_to(target)
            for destination, _target in links:
                if not destination.resolve(strict=True).is_relative_to(physical_root):
                    raise MetadataError('Chrome archive symbolic link escapes its extraction directory')
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile, RuntimeError) as exc:
        raise MetadataError("could not safely extract Chrome for Testing archive") from exc
    executable = extraction_root / CHROME_EXECUTABLE_RELATIVE
    if not executable.is_file():
        raise MetadataError("Chrome archive is missing its expected mac-arm64 executable")
    return executable.resolve(strict=True)


def _isolated_version(path: Path, *, result_root: Path) -> str:
    root = _require_private_result_root(result_root)
    scratch = Path(tempfile.mkdtemp(prefix='.tool-version-', dir=root))
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "LANG": "C",
    }
    try:
        result = subprocess.run(
            [str(path), 'version' if path.name == 'lightpanda' else '--version'],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=scratch,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MetadataError(f"could not read installed version from {path}") from exc
    if result.returncode != 0:
        raise MetadataError(f"version command failed for {path}: {result.stderr.strip()}")
    output = f"{result.stdout}\n{result.stderr}"
    matches = re.findall(r"(?<![A-Za-z0-9])v?([0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][0-9A-Za-z.-]+)?)(?![A-Za-z0-9])", output)
    if not matches:
        raise MetadataError(f"version command returned no parseable version: {path}")
    return matches[-1]


def _rename_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        if sys.platform == "darwin":
            native = library.renamex_np
            native.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            arguments = (os.fsencode(source), os.fsencode(destination), 4)
        elif sys.platform == "linux":
            native = library.renameat2
            native.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        else:
            raise MetadataError(
                "native exclusive moves are unsupported on this platform"
            )
    except AttributeError as error:
        raise MetadataError("native exclusive move symbol is unavailable") from error
    native.restype = ctypes.c_int
    ctypes.set_errno(0)
    if native(*arguments) != 0:
        number = ctypes.get_errno() or errno.EIO
        raise OSError(number, os.strerror(number), str(destination))


_INSTALL_SCHEMA = "openbrowser-tool-install-plan/1"
_INSTALL_RECEIPT = ".tool-install-receipt.json"
_INSTALL_PLAN = "install-plan.json"
_INSTALL_MANIFEST = "public-manifest.json"
_INSTALL_ROLES = (
    "lightpanda",
    "agent-browser",
    "chrome-mac-arm64.zip",
    "chrome-mac-arm64",
)
_MAX_INSTALL_PLAN_BYTES = 128 * 1024 * 1024


def _write_retained_file(path: Path, body: bytes, *, mode: int = 0o600) -> None:
    if len(body) > _MAX_INSTALL_PLAN_BYTES:
        raise MetadataError("installation JSON exceeds size limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    with os.fdopen(descriptor, "wb") as output:
        os.fchmod(output.fileno(), mode)
        output.write(body)
        output.flush()
        os.fsync(output.fileno())


def _install_exact(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise MetadataError(f"conflict: {where} has unknown or missing keys")
    return value


def _install_integer(value: Any, where: str) -> None:
    if type(value) is not int or not 0 <= value < 2**64:
        raise MetadataError(f"conflict: {where} must be an unsigned integer")


def _install_json(path: Path) -> tuple[dict[str, Any], os.stat_result, bytes]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MetadataError("conflict: duplicate installation JSON key")
            result[key] = value
        return result

    descriptor = os.open(
        path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(descriptor, "rb") as source:
        identity = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(identity.st_mode)
            or stat.S_IMODE(identity.st_mode) != 0o600
            or identity.st_uid != os.getuid()
            or identity.st_size > _MAX_INSTALL_PLAN_BYTES
        ):
            raise MetadataError(f"conflict: unsafe installation JSON: {path}")
        body = source.read(_MAX_INSTALL_PLAN_BYTES + 1)
    if len(body) > _MAX_INSTALL_PLAN_BYTES:
        raise MetadataError("conflict: installation JSON exceeds size limit")
    try:
        value = json.loads(body, object_pairs_hook=unique_object)
    except (ValueError, UnicodeDecodeError) as error:
        raise MetadataError(f"conflict: invalid installation JSON: {path}") from error
    if not isinstance(value, dict):
        raise MetadataError("conflict: installation JSON must be an object")
    return value, identity, body


def _install_anchor(path: Path, relative: str) -> dict[str, Any]:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.getuid()
    ):
        raise MetadataError(
            f"conflict: installation ancestor is not a private physical directory: {path}"
        )
    return {
        "path": relative,
        "dev": info.st_dev,
        "ino": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
    }


def _install_anchors(root: Path, tools: Path) -> list[dict[str, Any]]:
    records = [_install_anchor(root, "")]
    parent = root
    for part in tools.relative_to(root).parts:
        parent /= part
        records.append(_install_anchor(parent, parent.relative_to(root).as_posix()))
    return records


def _install_node(path: Path, relative: str) -> dict[str, Any]:
    info = path.lstat()
    if stat.S_ISREG(info.st_mode):
        kind = "file"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    elif stat.S_ISLNK(info.st_mode):
        kind = "symlink"
    else:
        raise MetadataError(f"conflict: unsupported payload node: {path}")
    return {
        "path": relative,
        "dev": info.st_dev,
        "ino": info.st_ino,
        "kind": kind,
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def _install_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    pending = [(root, "")]
    while pending:
        path, relative = pending.pop()
        node = _install_node(path, relative)
        if node["kind"] == "file":
            node["sha256"] = sha256_file(path)
        elif node["kind"] == "symlink":
            node["target"] = os.readlink(path)
            if (
                not node["target"]
                or "\\" in node["target"]
                or Path(node["target"]).is_absolute()
                or not path.resolve(strict=True).is_relative_to(root)
            ):
                raise MetadataError(
                    "conflict: payload symbolic link is not internally closed"
                )
        else:
            for child in sorted(path.iterdir(), reverse=True):
                pending.append((child, child.relative_to(root).as_posix()))
        records.append(node)
        if len(records) > MAX_ARCHIVE_ENTRIES + 1:
            raise MetadataError("conflict: payload inventory exceeds entry limit")
    return sorted(records, key=lambda node: node["path"])


def _install_nodes_shape(nodes: Any, *, directory: bool) -> None:
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_ARCHIVE_ENTRIES + 1:
        raise MetadataError("conflict: invalid payload inventory length")
    paths = []
    base = {"path", "dev", "ino", "kind", "mode", "uid", "gid", "size", "mtime_ns"}
    for node in nodes:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("kind"), str)
            or node["kind"] not in {"file", "directory", "symlink"}
        ):
            raise MetadataError("conflict: invalid payload inventory node")
        extra = (
            {"sha256"}
            if node["kind"] == "file"
            else {"target"}
            if node["kind"] == "symlink"
            else set()
        )
        _install_exact(node, base | extra, "inventory node")
        relative = node["path"]
        if (
            not isinstance(relative, str)
            or relative == "."
            or "\x00" in relative
            or (
                relative
                and (
                    PurePosixPath(relative).is_absolute()
                    or ".." in PurePosixPath(relative).parts
                    or PurePosixPath(relative).as_posix() != relative
                    or "\\" in relative
                )
            )
        ):
            raise MetadataError("conflict: noncanonical inventory path")
        for name in ("dev", "ino", "mode", "uid", "gid", "size", "mtime_ns"):
            _install_integer(node[name], f"inventory.{name}")
        if node["mode"] > 0o777 or node["uid"] != os.getuid():
            raise MetadataError("conflict: unsafe payload permissions or owner")
        if extra == {"sha256"}:
            _sha256_hex(node["sha256"], "inventory.sha256")
        elif extra:
            target = node["target"]
            if (
                not isinstance(target, str)
                or not target
                or "\x00" in target
                or "\\" in target
                or Path(target).is_absolute()
            ):
                raise MetadataError("conflict: unsafe inventory symbolic link")
        paths.append(relative)
    if (
        paths != sorted(set(paths))
        or paths[0] != ""
        or nodes[0]["kind"] != ("directory" if directory else "file")
    ):
        raise MetadataError("conflict: incomplete or unsorted payload inventory")
    if not directory and len(nodes) != 1:
        raise MetadataError("conflict: regular slot has descendants")
    kinds = {node["path"]: node["kind"] for node in nodes}
    for relative in paths[1:]:
        parent = PurePosixPath(relative).parent.as_posix()
        if kinds.get("" if parent == "." else parent) != "directory":
            raise MetadataError(
                "conflict: inventory parent is not a physical directory"
            )


def _install_match(root: Path, nodes: list[dict[str, Any]]) -> None:
    expected = {node["path"]: node for node in nodes}
    children: dict[str, list[str]] = {
        name: [] for name, node in expected.items() if node["kind"] == "directory"
    }
    for relative in expected:
        if relative:
            parent = PurePosixPath(relative).parent.as_posix()
            children["" if parent == "." else parent].append(
                PurePosixPath(relative).name
            )
    for relative, node in expected.items():
        path = root / relative
        actual = _install_node(path, relative)
        if actual != {
            key: value for key, value in node.items() if key not in {"sha256", "target"}
        }:
            raise MetadataError(
                f"conflict: payload identity or metadata changed: {path}"
            )
        if node["kind"] == "directory":
            if sorted(child.name for child in path.iterdir()) != sorted(
                children[relative]
            ):
                raise MetadataError(f"conflict: payload membership changed: {path}")
        elif node["kind"] == "file":
            if sha256_file(path) != node["sha256"]:
                raise MetadataError(f"conflict: payload bytes changed: {path}")
        elif os.readlink(path) != node["target"] or not path.resolve(
            strict=True
        ).is_relative_to(root):
            raise MetadataError(f"conflict: payload symbolic link changed: {path}")


def _install_manifest(
    parsed: dict[str, Any], tools: Path, slots: dict[str, Any]
) -> dict[str, Any]:
    records = {}
    for name in ("lightpanda", "agent-browser"):
        release = parsed["release_assets"][name]
        node = slots[name][0]
        if (
            node["sha256"] != release["official_sha256"]
            or node["size"] != release["size"]
            or node["mode"] != 0o755
        ):
            raise MetadataError(
                f"conflict: prepared {name} does not match official metadata"
            )
        records[name] = {
            "path": str(tools / name),
            "installed_version": release["version"],
            "download_url": release["download_url"],
            "local_sha256": node["sha256"],
            "official_sha256": release["official_sha256"],
        }
    chrome = {node["path"]: node for node in slots["chrome-mac-arm64"]}.get(
        CHROME_EXECUTABLE_RELATIVE.as_posix()
    )
    if chrome is None or chrome["kind"] != "file" or chrome["mode"] != 0o755:
        raise MetadataError(
            "conflict: Chrome executable is not a prepared physical executable"
        )
    records["chrome_for_testing"] = {
        "path": str(tools / "chrome-mac-arm64" / CHROME_EXECUTABLE_RELATIVE),
        "installed_version": parsed["latest_versions"]["chrome_for_testing"],
        "download_url": parsed["chrome_download_url"],
        "local_sha256": chrome["sha256"],
        "archive_path": str(tools / "chrome-mac-arm64.zip"),
        "archive_sha256": slots["chrome-mac-arm64.zip"][0]["sha256"],
    }
    return {
        "schema": TOOL_MANIFEST_SCHEMA,
        "metadata_sha256": parsed["metadata_manifest_sha256"],
        "tools": records,
    }


def _install_plan_shape(
    plan: dict[str, Any], root: Path, tools: Path, cache: Path, parsed: dict[str, Any]
) -> None:
    _install_exact(
        plan,
        {
            "schema",
            "cache_id",
            "metadata_sha256",
            "root_path",
            "tools_relative",
            "anchors",
            "cache",
            "slots",
            "manifest",
        },
        "installation plan",
    )
    if (
        plan["schema"] != _INSTALL_SCHEMA
        or not isinstance(plan["cache_id"], str)
        or not re.fullmatch("[0-9a-f]{32}", plan["cache_id"])
        or cache != root / f".tool-install-cache-{plan['cache_id']}"
        or plan["metadata_sha256"] != parsed["metadata_manifest_sha256"]
        or plan["root_path"] != str(root)
        or plan["tools_relative"] != tools.relative_to(root).as_posix()
    ):
        raise MetadataError("conflict: installation plan request binding changed")
    anchors = plan["anchors"]
    if (
        not isinstance(anchors, list)
        or len(anchors) != len(tools.relative_to(root).parts) + 1
    ):
        raise MetadataError("conflict: installation ancestor list is invalid")
    for anchor in [*anchors, plan["cache"]]:
        _install_exact(
            anchor, {"path", "dev", "ino", "mode", "uid", "gid"}, "installation anchor"
        )
        if not isinstance(anchor["path"], str):
            raise MetadataError("conflict: anchor path must be a string")
        for name in ("dev", "ino", "mode", "uid", "gid"):
            _install_integer(anchor[name], f"anchor.{name}")
    if anchors != _install_anchors(root, tools) or plan["cache"] != _install_anchor(
        cache, cache.name
    ):
        raise MetadataError("conflict: installation ancestor identity changed")
    _install_exact(plan["slots"], set(_INSTALL_ROLES), "installation slots")
    for name in _INSTALL_ROLES:
        _install_nodes_shape(plan["slots"][name], directory=name == "chrome-mac-arm64")
    _install_nodes_shape(plan["manifest"], directory=False)
    if plan["manifest"][0]["mode"] != 0o600:
        raise MetadataError("conflict: manifest witness mode changed")


def _install_optional_stat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _inspect_installation(
    plan: dict[str, Any],
    root: Path,
    tools: Path,
    cache: Path,
    metadata_root: Path,
    *,
    selected: bool,
) -> tuple[list[str], dict[str, Any], bool]:
    parsed = parse_metadata_sources(metadata_root)
    _install_plan_shape(plan, root, tools, cache, parsed)
    saved, source_identity, _body = _install_json(cache / _INSTALL_PLAN)
    if saved != plan:
        raise MetadataError("conflict: persisted installation plan changed")
    if selected:
        _receipt, receipt_identity, _receipt_body = _install_json(
            root / _INSTALL_RECEIPT
        )
        if not os.path.samestat(source_identity, receipt_identity):
            raise MetadataError(
                "conflict: root receipt is not the retained plan witness"
            )
    manifest, _identity, _body = _install_json(cache / _INSTALL_MANIFEST)
    _install_match(cache / _INSTALL_MANIFEST, plan["manifest"])
    if manifest != _install_manifest(parsed, tools, plan["slots"]):
        raise MetadataError(
            "conflict: public manifest does not match plan and metadata"
        )
    positions = []
    for name in _INSTALL_ROLES:
        source, destination = cache / name, tools / name
        source_info, destination_info = (
            _install_optional_stat(source),
            _install_optional_stat(destination),
        )
        if (source_info is None) == (destination_info is None):
            raise MetadataError(
                f"conflict: payload {name} must occupy exactly one planned slot"
            )
        location = source if source_info is not None else destination
        _install_match(location, plan["slots"][name])
        positions.append("ready" if source_info is not None else "moved")
    expected_members = {_INSTALL_PLAN, _INSTALL_MANIFEST} | {
        name for name, position in zip(_INSTALL_ROLES, positions) if position == "ready"
    }
    if {path.name for path in cache.iterdir()} != expected_members:
        raise MetadataError("conflict: retained generation has unexpected members")
    marker = _install_optional_stat(root / "tool-manifest.json")
    if marker is not None:
        if not os.path.samestat(marker, (cache / _INSTALL_MANIFEST).lstat()):
            raise MetadataError(
                "conflict: final manifest is not the retained manifest witness"
            )
        _install_match(root / "tool-manifest.json", plan["manifest"])
        if any(position != "moved" for position in positions):
            raise MetadataError(
                "conflict: committed marker has incomplete installation"
            )
    return positions, manifest, marker is not None


def _prepare_installation(
    root: Path,
    tools: Path,
    metadata_root: Path,
    parsed: dict[str, Any],
    fetch: Callable[..., Any],
) -> tuple[Path, dict[str, Any]]:
    cache_id = uuid.uuid4().hex
    cache = root / f".tool-install-cache-{cache_id}"
    try:
        cache.mkdir(mode=0o700)
        anchors = _install_anchors(root, tools)
        for name in ("lightpanda", "agent-browser"):
            release = parsed["release_assets"][name]
            _download_file(
                fetch,
                release["download_url"],
                cache / name,
                max_bytes=256 * 1024 * 1024,
                expected_sha256=release["official_sha256"],
                expected_size=release["size"],
                mode=0o755,
            )
        _download_file(
            fetch,
            parsed["chrome_download_url"],
            cache / "chrome-mac-arm64.zip",
            max_bytes=MAX_ARCHIVE_BYTES,
        )
        chrome = _extract_chrome_archive(
            cache / "chrome-mac-arm64.zip", cache / "chrome-mac-arm64"
        )
        slots = {name: _install_inventory(cache / name) for name in _INSTALL_ROLES}
        for name, path in (
            ("lightpanda", cache / "lightpanda"),
            ("agent-browser", cache / "agent-browser"),
            ("chrome_for_testing", chrome),
        ):
            actual = _isolated_version(path, result_root=root)
            expected = (
                parsed["latest_versions"][name.replace("-", "_")]
                if name == "chrome_for_testing"
                else parsed["release_assets"][name]["version"]
            )
            if actual != expected:
                raise MetadataError(
                    f"installed {name} version mismatch: expected {expected}, got {actual}"
                )
        for name in _INSTALL_ROLES:
            _install_match(cache / name, slots[name])
        manifest = _install_manifest(parsed, tools, slots)
        _write_retained_file(cache / _INSTALL_MANIFEST, _json_bytes(manifest))
        plan = {
            "schema": _INSTALL_SCHEMA,
            "cache_id": cache_id,
            "metadata_sha256": parsed["metadata_manifest_sha256"],
            "root_path": str(root),
            "tools_relative": tools.relative_to(root).as_posix(),
            "anchors": anchors,
            "cache": _install_anchor(cache, cache.name),
            "slots": slots,
            "manifest": _install_inventory(cache / _INSTALL_MANIFEST),
        }
        _write_retained_file(cache / _INSTALL_PLAN, _json_bytes(plan))
        _inspect_installation(plan, root, tools, cache, metadata_root, selected=False)
        return cache, plan
    except BaseException as original:
        BaseException.add_note(original, f"retained_preparation: {cache}")
        raise


def download_tools(
    metadata_dir: str | os.PathLike[str],
    tools_dir: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    result_root: str | os.PathLike[str],
    open_url: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    root = _require_private_result_root(result_root)
    metadata_root = _result_directory(
        metadata_dir, root, "metadata input", create=False, require_exists=True
    )
    tools = _result_directory(
        tools_dir, root, "tools output", create=False, require_exists=True
    )
    output = _result_path(manifest_path, root, "tool manifest")
    if output != root / "tool-manifest.json":
        raise MetadataError("tool manifest must use RESULT_ROOT/tool-manifest.json")
    parsed = parse_metadata_sources(metadata_root)
    receipt = root / _INSTALL_RECEIPT
    if _install_optional_stat(receipt) is None:
        if _install_optional_stat(output) is not None:
            raise MetadataError(
                "refusing existing tool manifest without an owned receipt"
            )
        if any(
            _install_optional_stat(tools / name) is not None for name in _INSTALL_ROLES
        ):
            raise MetadataError("refusing to overwrite an existing tool installation")
        cache, plan = _prepare_installation(
            root, tools, metadata_root, parsed, _open_url(open_url)
        )
        os.link(cache / _INSTALL_PLAN, receipt)
    else:
        plan, _identity, _body = _install_json(receipt)
        cache_id = plan.get("cache_id")
        if not isinstance(cache_id, str) or not re.fullmatch("[0-9a-f]{32}", cache_id):
            raise MetadataError("conflict: invalid installation receipt cache ID")
        cache = root / f".tool-install-cache-{cache_id}"
    try:
        positions, manifest, committed = _inspect_installation(
            plan, root, tools, cache, metadata_root, selected=True
        )
        if committed:
            return manifest
        for index, name in enumerate(_INSTALL_ROLES):
            positions, manifest, _committed = _inspect_installation(
                plan, root, tools, cache, metadata_root, selected=True
            )
            if positions[index] == "ready":
                _rename_noreplace(cache / name, tools / name)
        positions, manifest, _committed = _inspect_installation(
            plan, root, tools, cache, metadata_root, selected=True
        )
        if positions != ["moved"] * len(_INSTALL_ROLES):
            raise MetadataError("conflict: installation is not ready to commit")
        os.link(cache / _INSTALL_MANIFEST, output)
        return manifest
    except BaseException as original:
        BaseException.add_note(original, f"selected installation evidence: {cache}")
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch", help="fetch official latest-version metadata")
    fetch_parser.add_argument("--result-root", required=True, type=Path)
    fetch_parser.add_argument("--output", required=True, type=Path)
    tools_parser = subparsers.add_parser("download-tools", help="download latest macOS arm64 browser binaries")
    tools_parser.add_argument("--result-root", required=True, type=Path)
    tools_parser.add_argument("--metadata", required=True, type=Path)
    tools_parser.add_argument("--tools", required=True, type=Path)
    tools_parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fetch":
            fetch_metadata(args.output, result_root=args.result_root)
            return 0
        if args.command == "download-tools":
            download_tools(args.metadata, args.tools, args.manifest, result_root=args.result_root)
            return 0
    except (MetadataError, OSError) as exc:
        LOGGER.error("%s", "\n".join((str(exc), *getattr(exc, "__notes__", ()))))
        return 1
    LOGGER.error("unsupported command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
