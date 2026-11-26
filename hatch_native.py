"""Optional platform-native CLI assembly for Hatch builds.

Pure builds keep the Python console entry points. Native wheel builds use the
locked Rust source and a sibling Python backend with standard wheel scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface

SCRIPTS = {
    "openbrowser-ai": "openbrowser.cli_entry:cli",
    "openbrowser-ai-py": "openbrowser.cli_entry:cli",
}
TARGET_TAGS = {
    "aarch64-apple-darwin": "macosx_11_0_arm64",
    "x86_64-apple-darwin": "macosx_10_13_x86_64",
    "aarch64-unknown-linux-musl": "manylinux_2_17_aarch64",
    "x86_64-unknown-linux-musl": "manylinux_2_17_x86_64",
}


def native_enabled() -> bool:
    value = os.environ.get("OPENBROWSER_BUILD_NATIVE", "0")
    if value not in {"", "0", "1"}:
        raise ValueError("OPENBROWSER_BUILD_NATIVE accepts 0 or 1")
    return value == "1"


def select_target(host: str) -> tuple[str, str]:
    requested = os.environ.get("OPENBROWSER_NATIVE_TARGET") or os.environ.get("CARGO_BUILD_TARGET")
    target = requested or host.replace("-unknown-linux-gnu", "-unknown-linux-musl")
    if target not in TARGET_TAGS:
        raise ValueError(f"Unsupported native target: {target}")
    return target, f"py3-none-{TARGET_TAGS[target]}"


def source_hashes(root: Path) -> dict[str, str]:
    sources = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "target" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("Native source symlinks are unsupported")
        if path.is_file() and (path.name in {"Cargo.toml", "Cargo.lock"} or path.suffix == ".rs"):
            sources[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not {"Cargo.toml", "Cargo.lock", "src/main.rs"}.issubset(sources):
        raise ValueError("Native crate sources are incomplete")
    return sources


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict[str, Any]) -> None:
        scripts = dict(SCRIPTS)
        if native_enabled():
            del scripts["openbrowser-ai"]
        metadata["scripts"] = scripts


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if not native_enabled() or getattr(self, "target_name", "wheel") != "wheel":
            return
        if version == "editable":
            raise ValueError("Native entry points require a platform wheel build")
        root = Path(self.root).resolve()
        crate = root / "src/openbrowser/native_client/rust"
        sources = source_hashes(crate)
        scratch = tempfile.TemporaryDirectory(prefix="openbrowser-native-wheel-")
        self._native_scratch = scratch
        try:
            compiler = subprocess.run(
                ["rustc", "-vV"], capture_output=True, text=True, check=True, timeout=30
            ).stdout.strip()
            host = next(
                (line.partition(":")[2].strip() for line in compiler.splitlines() if line.startswith("host:")), ""
            )
            target, tag = select_target(host)
            env = dict(os.environ)
            if env.get("RUSTFLAGS") or env.get("CARGO_ENCODED_RUSTFLAGS"):
                raise ValueError("Native builds require an unset Rust flags environment")
            env["CARGO_TARGET_DIR"] = str(Path(scratch.name) / "target")
            registry = Path.home() / ".cargo"
            env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(
                (f"--remap-path-prefix={root}=/openbrowser-source", f"--remap-path-prefix={registry}=/cargo-registry")
            )
            subprocess.run(
                [
                    "cargo",
                    "build",
                    "--release",
                    "--locked",
                    "--manifest-path",
                    str(crate / "Cargo.toml"),
                    "--target",
                    target,
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
            if source_hashes(crate) != sources:
                raise RuntimeError("Native sources changed during compilation")
            binary = Path(env["CARGO_TARGET_DIR"]) / target / "release/openbrowser-ai"
            if not binary.is_file() or not os.access(binary, os.X_OK) or binary.stat().st_size == 0:
                raise RuntimeError("Native compiler produced no executable")
            manifest = Path(scratch.name) / "openbrowser-ai.native.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "frontend": "openbrowser-native-v1",
                        "entry_name": "openbrowser-ai",
                        "backend_name": "openbrowser-ai-py",
                        "package_version": self.metadata.version,
                        "target": target,
                        "compiler": compiler,
                        "source_files": sources,
                        "entry_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            build_data["pure_python"] = False
            build_data["tag"] = tag
            build_data.setdefault("shared_scripts", {}).update(
                {str(binary): "openbrowser-ai", str(manifest): "openbrowser-ai.native.json"}
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            scratch.cleanup()
            self._native_scratch = None
            raise RuntimeError(f"Native CLI build failed: {type(error).__name__}") from error
        except BaseException:
            scratch.cleanup()
            self._native_scratch = None
            raise

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        scratch = getattr(self, "_native_scratch", None)
        if scratch is not None:
            scratch.cleanup()
            self._native_scratch = None
