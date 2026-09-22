"""Build the deterministic, allowlisted Context Atlas Chrome Web Store package."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

PACKAGE_FILES = (
    "manifest.json",
    "background.js",
    "jev-client.js",
    "source-snapshot.js",
    "localhost-bridge.js",
    "context-atlas-extension.js",
    "context-atlas-offscreen.js",
    "context-atlas-sandbox.js",
    "offscreen.html",
    "sandbox.html",
    "options.html",
    "options.js",
    "options.css",
    "permissions.html",
    "permissions.js",
    "privacy.html",
    "icons/context-atlas-16.png",
    "icons/context-atlas-48.png",
    "icons/context-atlas-128.png",
    "ort/ort-wasm-simd-threaded.asyncify.mjs",
    "ort/ort-wasm-simd-threaded.asyncify.wasm",
)
EXPECTED_PERMISSIONS = {"activeTab", "scripting", "storage", "offscreen", "unlimitedStorage"}
EXPECTED_OPTIONAL_HOST_PERMISSIONS = {
    "https://api.typesafe.ai/v1/systemone",
    "https://huggingface.co/mizchi/laya-multilingual-onnx/resolve/d9d003d543e63d6d3375c21d44624136bd1e0bad/*",
    "https://us.aws.cdn.hf.co/xet-bridge-us/*",
}
SECRET_PATTERNS = (
    re.compile(rb"Bearer\s+sk-", re.IGNORECASE),
    re.compile(rb"JEV_API_KEY\s*=", re.IGNORECASE),
    re.compile(rb"api[_-]?key\s*[:=]\s*['\"][^'\"]{20,}['\"]", re.IGNORECASE),
)
REMOTE_EXECUTABLE_PATTERNS = (
    re.compile(rb"<script[^>]+src\s*=\s*['\"]https?://", re.IGNORECASE),
    re.compile(rb"import\s*\(\s*['\"]https?://", re.IGNORECASE),
    re.compile(rb"importScripts\s*\(\s*['\"]https?://", re.IGNORECASE),
)
UNSAFE_EVAL_PATTERNS = (
    re.compile(rb"\beval\s*\(", re.IGNORECASE),
    re.compile(rb"new\s+Function\s*\(", re.IGNORECASE),
)
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class PackageError(ValueError):
    """Raised when the extension cannot be safely packaged."""


@dataclass(frozen=True)
class PackageResult:
    output_path: Path
    file_count: int
    byte_size: int
    sha256: str


def _path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def _read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PackageError("required package file could not be read") from error


def _validate_reference(reference: str) -> None:
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts or "\\" in reference or not reference:
        raise PackageError("manifest contains an unsafe file reference")


def _validate_file_references(references: Iterable[object]) -> set[str]:
    validated: set[str] = set()
    for reference in references:
        if not isinstance(reference, str):
            raise PackageError("manifest file reference is invalid")
        _validate_reference(reference)
        validated.add(reference)
    return validated


def _matches_expected_strings(value: object, expected: set[str]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(isinstance(item, str) for item in value)
        and set(value) == expected
    )


def _validate_manifest(manifest: dict[str, object]) -> set[str]:
    if manifest.get("manifest_version") != 3:
        raise PackageError("manifest must use Manifest V3")
    version = manifest.get("version")
    parts = version.split(".") if isinstance(version, str) else []
    if (
        len(parts) not in range(1, 5)
        or any(
            not re.fullmatch(r"[0-9]+", part)
            or (len(part) > 1 and part.startswith("0"))
            or len(part) > 5
            or (len(part) == 5 and int(part) > 65535)
            for part in parts
        )
        or not any(part.lstrip("0") for part in parts)
    ):
        raise PackageError("manifest version is invalid")
    permissions = manifest.get("permissions")
    if not _matches_expected_strings(permissions, EXPECTED_PERMISSIONS):
        raise PackageError("manifest permissions do not match the release policy")
    host_permissions = manifest.get("host_permissions")
    if host_permissions not in (None, []):
        raise PackageError("required host permissions must be empty; use optional host permissions")
    optional_host_permissions = manifest.get("optional_host_permissions")
    if not _matches_expected_strings(optional_host_permissions, EXPECTED_OPTIONAL_HOST_PERMISSIONS):
        raise PackageError("manifest optional host permissions do not match the release policy")
    background = manifest.get("background")
    if not isinstance(background, dict) or not isinstance(background.get("service_worker"), str):
        raise PackageError("manifest service worker is missing")
    references = _validate_file_references((background["service_worker"],))
    options_page = manifest.get("options_page")
    if options_page is not None:
        references.update(_validate_file_references((options_page,)))
    sandbox = manifest.get("sandbox")
    if not isinstance(sandbox, dict) or not isinstance(sandbox.get("pages"), list) or sandbox["pages"] != ["sandbox.html"]:
        raise PackageError("manifest sandbox pages do not match the release policy")
    references.update(sandbox["pages"])
    content_scripts = manifest.get("content_scripts", [])
    if not isinstance(content_scripts, list):
        raise PackageError("manifest content scripts are invalid")
    for content_script in content_scripts:
        if not isinstance(content_script, dict) or not isinstance(content_script.get("js", []), list):
            raise PackageError("manifest content script is invalid")
        references.update(_validate_file_references(content_script["js"]))
    web_accessible_resources = manifest.get("web_accessible_resources", [])
    if not isinstance(web_accessible_resources, list):
        raise PackageError("manifest web-accessible resources are invalid")
    for resource_entry in web_accessible_resources:
        if not isinstance(resource_entry, dict) or not isinstance(resource_entry.get("resources"), list):
            raise PackageError("manifest web-accessible resource entry is invalid")
        references.update(_validate_file_references(resource_entry["resources"]))
    action = manifest.get("action")
    if not isinstance(action, dict):
        raise PackageError("manifest action is missing")
    action_icon = action.get("default_icon")
    if action_icon is not None:
        if not isinstance(action_icon, dict):
            raise PackageError("manifest action icon path is invalid")
        references.update(_validate_file_references(action_icon.values()))
    icons = manifest.get("icons")
    if icons is not None:
        if not isinstance(icons, dict):
            raise PackageError("manifest icon path is invalid")
        references.update(_validate_file_references(icons.values()))
    return references


def _validate_content(files: Iterable[tuple[str, bytes]]) -> None:
    for name, content in files:
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            raise PackageError(f"secret-like content found in {name}")
        if any(pattern.search(content) for pattern in REMOTE_EXECUTABLE_PATTERNS):
            raise PackageError(f"remote executable code found in {name}")
        # ONNX Runtime's WASM bridge emits a small dynamic binding helper in
        # the explicitly sandboxed page. The manifest grants unsafe-eval only
        # to that sandbox; extension pages remain eval-free.
        if name != "context-atlas-sandbox.js" and not name.startswith("ort/") and any(pattern.search(content) for pattern in UNSAFE_EVAL_PATTERNS):
            raise PackageError(f"unsafe code execution found in {name}")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def build_package(output_path: Path | str, extension_root: Path | str, bundle_path: Path | str) -> PackageResult:
    """Validate and write a deterministic extension ZIP."""
    output = _path(output_path)
    root = _path(extension_root)
    bundle = _path(bundle_path)
    if not root.is_dir():
        raise PackageError("extension root does not exist")
    if not bundle.is_file():
        raise PackageError("extension bundle does not exist")
    manifest_path = root / "manifest.json"
    manifest_bytes = _read_file(manifest_path)
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise PackageError("manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise PackageError("manifest must be a JSON object")
    references = _validate_manifest(manifest)

    files: dict[str, bytes] = {"manifest.json": manifest_bytes}
    for name in PACKAGE_FILES:
        if name == "manifest.json":
            continue
        source_path = bundle if name == "context-atlas-extension.js" else root / name
        if not source_path.is_file():
            raise PackageError(f"required package file could not be read: {name}")
        files[name] = _read_file(source_path)
    if not references.issubset(files):
        raise PackageError("manifest references missing package files")
    _validate_content(files.items())

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(files):
            archive.writestr(_zip_info(name), files[name])
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return PackageResult(output_path=output, file_count=len(files), byte_size=output.stat().st_size, sha256=digest)


def _parse_args() -> argparse.Namespace:
    module_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--extension-root", type=Path, default=module_root / "extension")
    parser.add_argument("--build", action="store_true", help="build the local extension bundles first")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = _path(args.extension_root)
    bundle_path = root / "context-atlas-extension.js"
    if args.build:
        ui_root = root.parent / "ui"
        subprocess.run(["npm", "--prefix", str(ui_root), "run", "build"], check=True)
        bundle_path = root / "context-atlas-extension.js"
    if args.output is None:
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            version = manifest["version"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            LOGGER.error("package validation failed: manifest version is unavailable")
            return 1
        args.output = root.parent / "dist" / f"context-atlas-extension-v{version}.zip"
    try:
        result = build_package(args.output, root, bundle_path)
    except PackageError as error:
        LOGGER.error("package validation failed: %s", error)
        return 1
    print(f"package: {result.output_path}")
    print(f"files: {result.file_count}")
    print(f"bytes: {result.byte_size}")
    print(f"sha256: {result.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
