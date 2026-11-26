"""Optional platform-native CLI assembly for Hatch builds.

Pure builds keep the Python console entry points. Native wheel builds use the
locked Rust source and a sibling Python backend with standard wheel scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
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


def validate_binary_tag(data: bytes, target: str) -> str:
    """Validate executable architecture and ABI constraints before choosing a tag."""
    tag = TARGET_TAGS[target]
    arm = target.startswith("aarch64-")
    if target.endswith("-apple-darwin"):
        if len(data) < 32 or data[:4] != b"\xcf\xfa\xed\xfe":
            raise ValueError("Native target requires a 64-bit Mach-O executable")
        cpu, _, file_type, commands, command_bytes = struct.unpack_from("<IIIII", data, 4)
        if cpu != (0x0100000C if arm else 0x01000007) or file_type != 2:
            raise ValueError("Native executable architecture differs from its target")
        end = 32 + command_bytes
        if end > len(data):
            raise ValueError("Invalid Mach-O command table")
        offset = 32
        minimum = None
        for _ in range(commands):
            if offset + 8 > end:
                raise ValueError("Invalid Mach-O load command")
            command, size = struct.unpack_from("<II", data, offset)
            if size < 8 or offset + size > end:
                raise ValueError("Invalid Mach-O load command size")
            version = None
            if command == 0x32 and size >= 24:
                platform, version = struct.unpack_from("<II", data, offset + 8)
                if platform != 1:
                    raise ValueError("Native executable requires the macOS platform")
            elif command == 0x24 and size >= 16:
                version = struct.unpack_from("<I", data, offset + 8)[0]
            if version is not None:
                parsed = (version >> 16, (version >> 8) & 255, version & 255)
                minimum = max(minimum, parsed) if minimum is not None else parsed
            offset += size
        supported = (11, 0, 0) if arm else (10, 13, 0)
        if offset != end or minimum is None or minimum > supported:
            raise ValueError("Native macOS deployment target exceeds the wheel tag")
    else:
        if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
            raise ValueError("Native target requires a little-endian 64-bit ELF executable")
        file_type, machine = struct.unpack_from("<HH", data, 16)
        if file_type not in {2, 3} or machine != (183 if arm else 62):
            raise ValueError("Native executable architecture differs from its target")
        table = struct.unpack_from("<Q", data, 32)[0]
        size, count = struct.unpack_from("<HH", data, 54)
        if size < 56 or count == 0 or table + size * count > len(data):
            raise ValueError("Invalid ELF program table")
        kinds = {struct.unpack_from("<I", data, table + size * index)[0] for index in range(count)}
        if 3 in kinds:
            raise ValueError("Native musl executable has a dynamic interpreter")
        if 1 not in kinds:
            raise ValueError("Native executable has no loadable segments")
    return f"py3-none-{tag}"


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
            binary_data = binary.read_bytes()
            tag = validate_binary_tag(binary_data, target)
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
                        "entry_sha256": hashlib.sha256(binary_data).hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            build_data["pure_python"] = False
            build_data["tag"] = tag
            build_data.setdefault("shared_scripts", {})[str(binary)] = "openbrowser-ai"
            build_data.setdefault("extra_metadata", {})[str(manifest)] = "openbrowser-native.json"
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
