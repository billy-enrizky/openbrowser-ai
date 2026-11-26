"""Closed engine pinning and process binding for matched interactive lanes.

The module pins one exact browser build before a lane starts. A cheap file-stat
fence runs per row, and an explicit refresh re-hashes the launcher and engine
image before and after a lane. Process bindings prove the observed browser is
the pinned executable through PID, create time, and canonical executable path.

Public proofs contain hashes, versions, and stat identities. They never contain
absolute paths. Internal environment serialization carries the paths needed by
fresh resource workers. Imports are inert: psutil, subprocess, platform, and
socket are loaded only by the functions that need them.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ENGINE_PIN_ENV = "OPENBROWSER_BENCH_ENGINE_PIN"
PIN_SCHEMA = "openbrowser-engine-pin-v2"

LANE_MATCHED_SHELL_1255 = "matched-shell-1255"
LANE_MATCHED_FULL_1255 = "matched-full-1255"
SUPPORTED_LANES = (LANE_MATCHED_SHELL_1255, LANE_MATCHED_FULL_1255)

REVISION = "1255"
EXPECTED_VERSION = "159.0.8154.6"

STATUS_VALID = "valid"
STATUS_MALFORMED_BINDING = "malformed_binding"
STATUS_BINDING_MISMATCH = "binding_mismatch"
STATUS_PIN_CHANGED = "pin_changed"
STATUS_PIN_UNAVAILABLE = "pin_unavailable"
STATUS_PROCESS_MISSING = "process_missing"
STATUS_PROCESS_DENIED = "process_denied"
STATUS_PROCESS_REPLACED = "process_replaced"
STATUS_EXECUTABLE_CHANGED = "executable_changed"
STATUS_PROCESS_UNKNOWN = "process_unknown"

_PIN_KEYS = frozenset(
    {
        "schema",
        "lane",
        "revision",
        "version",
        "platform",
        "machine",
        "launcher",
        "engine_image",
        "launcher_sha256",
        "engine_image_sha256",
        "launcher_stat",
        "engine_image_stat",
        "launcher_kind",
        "engine_image_kind",
    }
)
_STAT_KEYS = frozenset({"device", "inode", "size", "mtime_ns", "ctime_ns"})
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_VERSION_RE = re.compile(r"Google Chrome for Testing ([0-9]{1,5}(?:\.[0-9]{1,5}){3})")
_FULL_LAUNCHER_KIND = "chrome-for-testing"
_SHELL_KIND = "headless-shell"

VersionRunner = Callable[[str], str]
Sha256Reader = Callable[[Path], str]
StatReader = Callable[[Path], Any]
ProcessFactory = Callable[[int], Any]
ExecutableChecker = Callable[[Path], bool]


class EngineBindingError(ValueError):
    """Closed engine-binding failure with a public-safe reason string."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class FileStat:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def public(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_result(cls, result: Any) -> FileStat:
        if not all(hasattr(result, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise EngineBindingError("malformed_stat")
        return cls._validated(
            result.st_dev,
            result.st_ino,
            result.st_size,
            result.st_mtime_ns,
            result.st_ctime_ns,
        )

    @classmethod
    def from_mapping(cls, value: Any) -> FileStat:
        if not isinstance(value, dict) or set(value) != _STAT_KEYS:
            raise EngineBindingError("malformed_pin")
        try:
            return cls._validated(
                value["device"],
                value["inode"],
                value["size"],
                value["mtime_ns"],
                value["ctime_ns"],
            )
        except EngineBindingError as error:
            raise EngineBindingError("malformed_pin") from error

    @classmethod
    def _validated(
        cls,
        device: Any,
        inode: Any,
        size: Any,
        mtime_ns: Any,
        ctime_ns: Any,
    ) -> FileStat:
        values = (device, inode, size, mtime_ns, ctime_ns)
        if any(type(item) is not int for item in values):
            raise EngineBindingError("malformed_stat")
        if device < 0 or inode < 0 or size <= 0 or mtime_ns < 0 or ctime_ns < 0:
            raise EngineBindingError("malformed_stat")
        return cls(
            device=device,
            inode=inode,
            size=size,
            mtime_ns=mtime_ns,
            ctime_ns=ctime_ns,
        )


@dataclass(frozen=True)
class LanePaths:
    lane: str
    platform: str
    machine: str
    launcher: Path
    engine_image: Path
    launcher_kind: str
    engine_image_kind: str


@dataclass(frozen=True)
class EnginePin:
    schema: str
    lane: str
    revision: str
    version: str
    platform: str
    machine: str
    launcher: Path
    engine_image: Path
    launcher_sha256: str
    engine_image_sha256: str
    launcher_stat: FileStat
    engine_image_stat: FileStat
    launcher_kind: str
    engine_image_kind: str

    def public_proof(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lane": self.lane,
            "revision": self.revision,
            "version": self.version,
            "platform": self.platform,
            "machine": self.machine,
            "launcher_sha256": self.launcher_sha256,
            "engine_image_sha256": self.engine_image_sha256,
            "launcher_stat": self.launcher_stat.public(),
            "engine_image_stat": self.engine_image_stat.public(),
            "launcher_kind": self.launcher_kind,
            "engine_image_kind": self.engine_image_kind,
        }

    def to_env(self) -> str:
        payload = self.public_proof()
        payload["launcher"] = str(self.launcher)
        payload["engine_image"] = str(self.engine_image)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_env(cls, value: str) -> EnginePin:
        if not isinstance(value, str):
            raise EngineBindingError("malformed_pin")
        try:
            payload = json.loads(value)
        except (TypeError, ValueError) as error:
            raise EngineBindingError("malformed_pin") from error
        if not isinstance(payload, dict) or set(payload) != _PIN_KEYS:
            raise EngineBindingError("malformed_pin")
        if payload["schema"] != PIN_SCHEMA:
            raise EngineBindingError("pin_schema_unsupported")
        lane = _require_choice(payload["lane"], SUPPORTED_LANES)
        revision = _require_choice(payload["revision"], (REVISION,))
        version = _require_choice(payload["version"], (EXPECTED_VERSION,))
        platform_name = _require_text(payload["platform"])
        machine = _require_text(payload["machine"])
        launcher = _require_absolute_path(payload["launcher"])
        engine_image = _require_absolute_path(payload["engine_image"])
        launcher_sha = _require_sha(payload["launcher_sha256"])
        image_sha = _require_sha(payload["engine_image_sha256"])
        launcher_stat = FileStat.from_mapping(payload["launcher_stat"])
        image_stat = FileStat.from_mapping(payload["engine_image_stat"])
        launcher_kind = _require_text(payload["launcher_kind"])
        image_kind = _require_text(payload["engine_image_kind"])
        _validate_kinds(
            lane=lane,
            platform_name=platform_name,
            launcher=launcher,
            engine_image=engine_image,
            launcher_kind=launcher_kind,
            engine_image_kind=image_kind,
        )
        return cls(
            schema=PIN_SCHEMA,
            lane=lane,
            revision=revision,
            version=version,
            platform=platform_name,
            machine=machine,
            launcher=launcher,
            engine_image=engine_image,
            launcher_sha256=launcher_sha,
            engine_image_sha256=image_sha,
            launcher_stat=launcher_stat,
            engine_image_stat=image_stat,
            launcher_kind=launcher_kind,
            engine_image_kind=image_kind,
        )

    def fence(self, *, stat_reader: StatReader | None = None) -> None:
        reader = stat_reader or _default_stat
        if _read_stat(self.launcher, reader) != self.launcher_stat:
            raise EngineBindingError("pin_changed")
        if _read_stat(self.engine_image, reader) != self.engine_image_stat:
            raise EngineBindingError("pin_changed")

    def refresh(
        self,
        *,
        sha256_reader: Sha256Reader | None = None,
        stat_reader: StatReader | None = None,
    ) -> EnginePin:
        self.fence(stat_reader=stat_reader)
        reader = sha256_reader or _default_sha256
        if _read_sha(self.launcher, reader) != self.launcher_sha256:
            raise EngineBindingError("pin_changed")
        if _read_sha(self.engine_image, reader) != self.engine_image_sha256:
            raise EngineBindingError("pin_changed")
        self.fence(stat_reader=stat_reader)
        return self


@dataclass(frozen=True)
class ProcessBinding:
    pid: int
    created_at: float
    executable_basename: str
    lane: str
    revision: str
    version: str
    launcher_sha256: str
    engine_image_sha256: str

    def public_proof(self) -> dict[str, Any]:
        return asdict(self)


def write_pin_environment(pin: EnginePin, env: MutableMapping[str, str]) -> None:
    """Store the internal pin payload for fresh worker processes."""
    if not isinstance(pin, EnginePin) or not _valid_pin(pin):
        raise EngineBindingError("malformed_pin")
    env[ENGINE_PIN_ENV] = pin.to_env()


def read_pin_environment(env: Mapping[str, str]) -> EnginePin:
    """Read the internal pin payload from one worker environment."""
    if not isinstance(env, Mapping):
        raise EngineBindingError("malformed_pin")
    value = env.get(ENGINE_PIN_ENV)
    if not isinstance(value, str):
        raise EngineBindingError("malformed_pin")
    return EnginePin.from_env(value)


def build_lane_paths(
    lane: str,
    cache_root: str | Path,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> LanePaths:
    if lane not in SUPPORTED_LANES:
        raise EngineBindingError("unsupported_lane")
    root = Path(cache_root)
    if not root.is_absolute():
        raise EngineBindingError("path_not_absolute")
    platform_name = platform_name or sys.platform
    machine = machine or _default_machine()
    pairs = _lane_candidates(lane, root, platform_name, machine)
    existing = [(launcher, image) for launcher, image in pairs if launcher.is_file() and image.is_file()]
    if not existing:
        if not any(launcher.is_file() for launcher, _ in pairs):
            raise EngineBindingError("launcher_missing")
        raise EngineBindingError("engine_image_missing")
    if len(existing) != 1:
        raise EngineBindingError("ambiguous_launcher")
    launcher, image = existing[0]
    launcher_kind = _FULL_LAUNCHER_KIND if lane == LANE_MATCHED_FULL_1255 else _SHELL_KIND
    image_kind = _engine_image_kind(lane, platform_name)
    return LanePaths(
        lane=lane,
        platform=platform_name,
        machine=machine,
        launcher=launcher,
        engine_image=image,
        launcher_kind=launcher_kind,
        engine_image_kind=image_kind,
    )


def capture_engine_pin(
    lane: str,
    cache_root: str | Path,
    *,
    version_runner: VersionRunner | None = None,
    sha256_reader: Sha256Reader | None = None,
    stat_reader: StatReader | None = None,
    executable_checker: ExecutableChecker | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
) -> EnginePin:
    paths = build_lane_paths(lane, cache_root, platform_name=platform_name, machine=machine)
    launcher = _canonical_file(paths.launcher, "launcher_missing")
    checker = executable_checker or _default_executable
    if not checker(launcher):
        raise EngineBindingError("launcher_not_executable")
    engine_image = _canonical_file(paths.engine_image, "engine_image_missing")
    if paths.lane == LANE_MATCHED_SHELL_1255 and engine_image != launcher:
        raise EngineBindingError("engine_image_mismatch")
    stat_reader = stat_reader or _default_stat
    launcher_stat = _read_stat(launcher, stat_reader)
    image_stat = _read_stat(engine_image, stat_reader)
    sha_reader = sha256_reader or _default_sha256
    launcher_sha = _read_sha(launcher, sha_reader)
    image_sha = _read_sha(engine_image, sha_reader)
    version = _read_version(version_runner or _default_version_runner, launcher)
    pin = EnginePin(
        schema=PIN_SCHEMA,
        lane=paths.lane,
        revision=REVISION,
        version=version,
        platform=paths.platform,
        machine=paths.machine,
        launcher=launcher,
        engine_image=engine_image,
        launcher_sha256=launcher_sha,
        engine_image_sha256=image_sha,
        launcher_stat=launcher_stat,
        engine_image_stat=image_stat,
        launcher_kind=paths.launcher_kind,
        engine_image_kind=paths.engine_image_kind,
    )
    pin.fence(stat_reader=stat_reader)
    return pin


def capture_process_binding(
    pin: EnginePin,
    pid: int,
    *,
    process_factory: ProcessFactory | None = None,
    stat_reader: StatReader | None = None,
) -> ProcessBinding:
    if not isinstance(pin, EnginePin) or not _valid_pin(pin):
        raise EngineBindingError("malformed_pin")
    pin.fence(stat_reader=stat_reader)
    created_at, executable = _read_process_identity(pid, process_factory)
    if executable != pin.launcher:
        raise EngineBindingError("executable_mismatch")
    pin.fence(stat_reader=stat_reader)
    return ProcessBinding(
        pid=pid,
        created_at=created_at,
        executable_basename=executable.name,
        lane=pin.lane,
        revision=pin.revision,
        version=pin.version,
        launcher_sha256=pin.launcher_sha256,
        engine_image_sha256=pin.engine_image_sha256,
    )


def verify_process_binding(
    binding: ProcessBinding,
    pin: EnginePin,
    *,
    process_factory: ProcessFactory | None = None,
    stat_reader: StatReader | None = None,
) -> str:
    if not isinstance(binding, ProcessBinding) or not isinstance(pin, EnginePin):
        return STATUS_MALFORMED_BINDING
    if not _valid_pin(pin) or not _valid_binding(binding):
        return STATUS_MALFORMED_BINDING
    if (
        binding.lane != pin.lane
        or binding.revision != pin.revision
        or binding.version != pin.version
        or binding.launcher_sha256 != pin.launcher_sha256
        or binding.engine_image_sha256 != pin.engine_image_sha256
    ):
        return STATUS_BINDING_MISMATCH
    try:
        pin.fence(stat_reader=stat_reader)
    except EngineBindingError as error:
        return STATUS_PIN_CHANGED if error.reason == "pin_changed" else STATUS_PIN_UNAVAILABLE
    try:
        created_at, executable = _read_process_identity(binding.pid, process_factory)
    except EngineBindingError as error:
        return _process_failure_status(error.reason)
    if created_at != binding.created_at:
        return STATUS_PROCESS_REPLACED
    if executable != pin.launcher:
        return STATUS_EXECUTABLE_CHANGED
    try:
        pin.fence(stat_reader=stat_reader)
    except EngineBindingError as error:
        return STATUS_PIN_CHANGED if error.reason == "pin_changed" else STATUS_PIN_UNAVAILABLE
    return STATUS_VALID


def _lane_candidates(
    lane: str,
    root: Path,
    platform_name: str,
    machine: str,
) -> list[tuple[Path, Path]]:
    revision_root = (
        root / f"chromium_headless_shell-{REVISION}"
        if lane == LANE_MATCHED_SHELL_1255
        else root / f"chromium-{REVISION}"
    )
    if lane == LANE_MATCHED_SHELL_1255:
        if platform_name == "darwin":
            arch = _mac_arch(machine)
            launcher = revision_root / f"chrome-headless-shell-mac-{arch}" / "chrome-headless-shell"
            return [(launcher, launcher)]
        if platform_name.startswith("linux"):
            names = (
                Path("chrome-headless-shell-linux64") / "chrome-headless-shell",
                Path("chrome-linux64") / "chrome-headless-shell",
                Path("chrome-linux") / "chrome-headless-shell",
            )
            return [(revision_root / name, revision_root / name) for name in names]
        raise EngineBindingError("unsupported_platform")
    if platform_name == "darwin":
        arch = _mac_arch(machine)
        app = revision_root / f"chrome-mac-{arch}" / "Google Chrome for Testing.app"
        launcher = app / "Contents/MacOS/Google Chrome for Testing"
        image = (
            app
            / "Contents/Frameworks/Google Chrome for Testing Framework.framework"
            / "Versions"
            / EXPECTED_VERSION
            / "Google Chrome for Testing Framework"
        )
        return [(launcher, image)]
    if platform_name.startswith("linux"):
        names = (Path("chrome-linux64") / "chrome", Path("chrome-linux") / "chrome")
        return [(revision_root / name, revision_root / name) for name in names]
    raise EngineBindingError("unsupported_platform")


def _engine_image_kind(lane: str, platform_name: str) -> str:
    if lane == LANE_MATCHED_SHELL_1255:
        return _SHELL_KIND
    return "framework" if platform_name == "darwin" else "binary"


def _mac_arch(machine: str) -> str:
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "x64"
    raise EngineBindingError("unsupported_platform")


def _validate_kinds(
    *,
    lane: str,
    platform_name: str,
    launcher: Path,
    engine_image: Path,
    launcher_kind: str,
    engine_image_kind: str,
) -> None:
    if lane == LANE_MATCHED_SHELL_1255:
        if launcher_kind != _SHELL_KIND or engine_image_kind != _SHELL_KIND:
            raise EngineBindingError("malformed_pin")
        if launcher != engine_image:
            raise EngineBindingError("malformed_pin")
        if launcher.name != "chrome-headless-shell":
            raise EngineBindingError("malformed_pin")
        return
    if launcher_kind != _FULL_LAUNCHER_KIND:
        raise EngineBindingError("malformed_pin")
    if engine_image_kind != _engine_image_kind(lane, platform_name):
        raise EngineBindingError("malformed_pin")
    if platform_name == "darwin":
        if launcher.name != "Google Chrome for Testing":
            raise EngineBindingError("malformed_pin")
        if engine_image.name != "Google Chrome for Testing Framework":
            raise EngineBindingError("malformed_pin")
    elif platform_name.startswith("linux"):
        if launcher.name != "chrome":
            raise EngineBindingError("malformed_pin")
        if engine_image != launcher:
            raise EngineBindingError("malformed_pin")
    else:
        raise EngineBindingError("malformed_pin")


def _read_process_identity(
    pid: Any,
    process_factory: ProcessFactory | None,
) -> tuple[float, Path]:
    if type(pid) is not int or pid <= 0:
        raise EngineBindingError("malformed_pid")
    factory = process_factory or _default_process_factory
    try:
        process = factory(pid)
        created = _require_created(process.create_time())
        if not process.is_running():
            raise EngineBindingError("process_gone")
        status = process.status()
        _require_alive_status(status)
        first_executable = _canonical_executable(process.exe())
        if _require_created(process.create_time()) != created:
            raise EngineBindingError("process_replaced")
        if not process.is_running():
            raise EngineBindingError("process_gone")
        _require_alive_status(process.status())
        second_executable = _canonical_executable(process.exe())
        if second_executable != first_executable:
            raise EngineBindingError("executable_changed")
        return created, second_executable
    except EngineBindingError:
        raise
    except Exception as error:
        raise EngineBindingError(_process_error_reason(error)) from error


def _process_error_reason(error: Exception) -> str:
    try:
        import psutil
    except ImportError:
        return "process_unknown"
    if isinstance(error, (psutil.NoSuchProcess, psutil.ZombieProcess)):
        return "process_gone"
    if isinstance(error, psutil.AccessDenied):
        return "process_denied"
    return "process_unknown"


def _process_failure_status(reason: str) -> str:
    return {
        "process_gone": STATUS_PROCESS_MISSING,
        "process_denied": STATUS_PROCESS_DENIED,
        "process_replaced": STATUS_PROCESS_REPLACED,
        "executable_missing": STATUS_EXECUTABLE_CHANGED,
        "executable_changed": STATUS_EXECUTABLE_CHANGED,
        "malformed_pid": STATUS_MALFORMED_BINDING,
    }.get(reason, STATUS_PROCESS_UNKNOWN)


def _canonical_executable(value: Any) -> Path:
    if not isinstance(value, (str, Path)):
        raise EngineBindingError("executable_missing")
    text = str(value)
    if not text or not Path(text).is_absolute():
        raise EngineBindingError("executable_missing")
    try:
        canonical = Path(text).resolve(strict=False)
    except OSError as error:
        raise EngineBindingError("process_unknown") from error
    if not canonical.is_absolute():
        raise EngineBindingError("executable_missing")
    return canonical


def _require_alive_status(status: Any) -> None:
    import psutil

    if not isinstance(status, str):
        raise EngineBindingError("process_unknown")
    if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
        raise EngineBindingError("process_gone")
    alive = {
        value
        for name in (
            "STATUS_RUNNING",
            "STATUS_SLEEPING",
            "STATUS_DISK_SLEEP",
            "STATUS_IDLE",
            "STATUS_WAKING",
            "STATUS_PARKED",
            "STATUS_LOCKED",
            "STATUS_WAITING",
        )
        if isinstance(value := getattr(psutil, name, None), str)
    }
    if status not in alive:
        raise EngineBindingError("process_unknown")


def _require_created(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EngineBindingError("process_unknown")
    created = float(value)
    if created != created or created in (float("inf"), float("-inf")) or created <= 0:
        raise EngineBindingError("process_unknown")
    return created


def _valid_pin(pin: EnginePin) -> bool:
    if not isinstance(pin.launcher, Path) or not isinstance(pin.engine_image, Path):
        return False
    if not pin.launcher.is_absolute() or not pin.engine_image.is_absolute():
        return False
    if ".." in pin.launcher.parts or ".." in pin.engine_image.parts:
        return False
    if not isinstance(pin.schema, str) or pin.schema != PIN_SCHEMA:
        return False
    if not isinstance(pin.lane, str) or pin.lane not in SUPPORTED_LANES:
        return False
    if not isinstance(pin.revision, str) or pin.revision != REVISION:
        return False
    if not isinstance(pin.version, str) or pin.version != EXPECTED_VERSION:
        return False
    if not isinstance(pin.platform, str):
        return False
    if pin.platform != "darwin" and not pin.platform.startswith("linux"):
        return False
    if not isinstance(pin.machine, str) or not pin.machine:
        return False
    if not _valid_sha(pin.launcher_sha256) or not _valid_sha(pin.engine_image_sha256):
        return False
    if not isinstance(pin.launcher_kind, str) or not isinstance(pin.engine_image_kind, str):
        return False
    if not isinstance(pin.launcher_stat, FileStat) or not isinstance(pin.engine_image_stat, FileStat):
        return False
    try:
        _validate_kinds(
            lane=pin.lane,
            platform_name=pin.platform,
            launcher=pin.launcher,
            engine_image=pin.engine_image,
            launcher_kind=pin.launcher_kind,
            engine_image_kind=pin.engine_image_kind,
        )
    except EngineBindingError:
        return False
    return True


def _valid_binding(binding: ProcessBinding) -> bool:
    if type(binding.pid) is not int or binding.pid <= 0:
        return False
    if isinstance(binding.created_at, bool) or not isinstance(binding.created_at, (int, float)):
        return False
    created = float(binding.created_at)
    if created != created or created in (float("inf"), float("-inf")) or created <= 0:
        return False
    if not isinstance(binding.executable_basename, str) or not binding.executable_basename:
        return False
    if "/" in binding.executable_basename or "\\" in binding.executable_basename:
        return False
    if binding.lane not in SUPPORTED_LANES or binding.revision != REVISION or binding.version != EXPECTED_VERSION:
        return False
    return bool(_SHA_RE.fullmatch(binding.launcher_sha256)) and bool(_SHA_RE.fullmatch(binding.engine_image_sha256))


def _canonical_file(path: Path, reason: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise EngineBindingError(reason) from error
    if not resolved.is_file():
        raise EngineBindingError(reason)
    return resolved


def _read_stat(path: Path, reader: StatReader) -> FileStat:
    try:
        return FileStat.from_result(reader(path))
    except EngineBindingError:
        raise
    except OSError as error:
        raise EngineBindingError("stat_unavailable") from error
    except Exception as error:
        raise EngineBindingError("malformed_stat") from error


def _read_sha(path: Path, reader: Sha256Reader) -> str:
    try:
        value = reader(path)
    except EngineBindingError:
        raise
    except Exception as error:
        raise EngineBindingError("sha_unavailable") from error
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise EngineBindingError("malformed_sha")
    return value


def _read_version(runner: VersionRunner, launcher: Path) -> str:
    try:
        output = runner(str(launcher))
    except EngineBindingError:
        raise
    except Exception as error:
        raise EngineBindingError("version_unavailable") from error
    if not isinstance(output, str):
        raise EngineBindingError("version_malformed")
    match = _VERSION_RE.fullmatch(output.strip())
    if match is None:
        raise EngineBindingError("version_malformed")
    if match.group(1) != EXPECTED_VERSION:
        raise EngineBindingError("version_mismatch")
    return match.group(1)


def _default_version_runner(launcher: str) -> str:
    import subprocess

    try:
        completed = subprocess.run(
            [launcher, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EngineBindingError("version_unavailable") from error
    if completed.returncode != 0:
        raise EngineBindingError("version_unavailable")
    output = completed.stdout if (completed.stdout or "").strip() else completed.stderr
    return output or ""


def _default_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EngineBindingError("sha_unavailable") from error
    return digest.hexdigest()


def _default_stat(path: Path) -> Any:
    return os.stat(path)


def _default_executable(path: Path) -> bool:
    return os.access(path, os.X_OK)


def _default_process_factory(pid: int) -> Any:
    import psutil

    return psutil.Process(pid)


def _default_machine() -> str:
    import platform

    return platform.machine()


def _require_text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise EngineBindingError("malformed_pin")
    return value


def _require_choice(value: Any, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise EngineBindingError("malformed_pin")
    return value


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA_RE.fullmatch(value))


def _require_sha(value: Any) -> str:
    if not _valid_sha(value):
        raise EngineBindingError("malformed_pin")
    return value


def _require_absolute_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise EngineBindingError("malformed_pin")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise EngineBindingError("malformed_pin")
    return path
