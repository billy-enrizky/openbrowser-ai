"""Shared types and process hygiene for interactive benchmark adapters."""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import psutil

_HOST_HOME = Path.home()
DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def matched_chrome() -> str:
    """Return the current declared browser path without a process or file probe."""
    value = os.environ.get("CHROME_BIN")
    if value is None:
        return DEFAULT_CHROME
    if not value.strip():
        raise ValueError("CHROME_BIN must name a browser executable")
    return value


def host_playwright_cache() -> Path | None:
    """Return the host Playwright browser cache without consulting a child HOME.

    Benchmark runs isolate HOME so profiles and downloads cannot escape the run
    directory. The installed browser binary is not run state. Exposing its
    cache lets OpenBrowser use the automatic extension-free headless shell that
    it selects for normal users instead of silently falling back to full Chrome.
    """
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_dir() else None
    if os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", _HOST_HOME)) / "ms-playwright"
    elif sys.platform == "darwin":
        candidate = _HOST_HOME / "Library" / "Caches" / "ms-playwright"
    else:
        candidate = _HOST_HOME / ".cache" / "ms-playwright"
    return candidate if candidate.is_dir() else None


@dataclass
class ArmResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    tool_calls: int = 0
    process_starts: int = 0
    reported: dict = field(default_factory=dict)
    error: str = ""
    browser_version: str = ""
    unavailable: bool = False
    keychain_verified: bool | None = None


@dataclass
class RunContext:
    task: str
    arm: str
    sample: int
    base_url: str
    root: Path
    downloads: Path
    profile: Path
    home: Path
    tmp: Path
    config: Path
    session: str

    @property
    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "HOME": str(self.home),
                "TMPDIR": str(self.tmp),
                "OPENBROWSER_DISABLE_TELEMETRY": "true",
                "ANONYMIZED_TELEMETRY": "false",
                "DO_NOT_TRACK": "1",
                "BH_UPDATE_CHECK": "0",
                "BROWSER_USE_NO_UPDATE_CHECK": "1",
            }
        )
        browser_cache = host_playwright_cache()
        if browser_cache is not None:
            env["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_cache)
        else:
            env.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        env.pop("PYTHONPATH", None)
        return env


def run_process(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    input_text: str | None = None,
    timeout: float = 120.0,
) -> tuple[int, str, str, float]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
        elapsed = time.monotonic() - started
        return process.returncode, stdout or "", stderr or "", elapsed
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        elapsed = time.monotonic() - started
        raise TimeoutError(f"command timed out after {timeout}s: {' '.join(command[:3])}") from None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"{url} did not become ready: {last_error}")


def _process_table() -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for process in psutil.process_iter(["pid"]):
        try:
            command = " ".join(process.cmdline())
        except (OSError, psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        if command:
            rows.append((int(process.pid), command))
    return rows


def process_commands_matching(marker: str) -> list[str]:
    if not marker or len(marker) < 8:
        raise ValueError("refusing to match on a broad marker")
    return [command for _pid, command in _process_table() if marker in command]


def verify_mock_keychain(profile_marker: str, *, timeout: float = 10.0) -> bool | None:
    """Return True when the main browser process carries --use-mock-keychain.

    Helper processes inherit --user-data-dir but not every browser-level flag,
    so only the process without --type= is inspected.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        commands = [
            command
            for command in process_commands_matching(profile_marker)
            if "--user-data-dir=" in command and "--type=" not in command
        ]
        if commands:
            return all("--use-mock-keychain" in command for command in commands)
        time.sleep(0.2)
    return None


def processes_matching(marker: str) -> list[int]:
    if not marker or len(marker) < 8:
        raise ValueError("refusing to match on a broad marker")
    current = os.getpid()
    return [pid for pid, command in _process_table() if marker in command and pid != current]


def terminate_matching(marker: str, *, grace: float = 2.0) -> list[int]:
    """Terminate processes whose command line contains a unique run marker."""
    if not marker or len(marker) < 8:
        raise ValueError("refusing to terminate on a broad marker")
    victims = processes_matching(marker)
    for pid in victims:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while victims and time.monotonic() < deadline:
        alive = []
        for pid in victims:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
        victims = alive
        if victims:
            time.sleep(0.1)
    for pid in victims:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    return victims


def kill_pidfile(pid_file: Path) -> None:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


class ChromeInstance:
    """A dedicated Chrome process with a per-run profile and mock keychain."""

    def __init__(
        self,
        executable: str,
        profile: Path,
        *,
        downloads: Path | None = None,
        headless: bool = True,
    ) -> None:
        self.executable = executable
        self.profile = profile
        self.downloads = downloads or (profile.parent / "downloads")
        self.headless = headless
        self.port = free_port()
        self.process: subprocess.Popen | None = None

    @property
    def cdp_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        profile = self.profile
        profile.mkdir(parents=True, exist_ok=True)
        preferences = profile / "Default" / "Preferences"
        preferences.parent.mkdir(parents=True, exist_ok=True)
        if not preferences.exists():
            preferences.write_text(
                f'{{"download":{{"default_directory":"{self.downloads!s}","prompt_for_download":false,'
                f'"directory_upgrade":true}},"savefile":{{"default_directory":"{self.downloads!s}"}}}}'
            )
        args = [
            self.executable,
            "--headless=new" if self.headless else "--new-window",
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-extensions",
            "--disable-component-update",
            "--disable-default-apps",
            "--mute-audio",
            "--window-size=1280,800",
            "--use-mock-keychain",
            "about:blank",
        ]
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        wait_for_http(f"{self.cdp_url}/json/version", timeout=30)

    def stop(self) -> None:
        if self.process is None:
            return
        with contextlib.suppress(OSError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(self.process.pid, signal.SIGKILL)
        self.process = None
        terminate_matching(str(self.profile))
