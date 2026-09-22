"""Launch Context Atlas in the current Chrome profile or an isolated browser."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_SERVER_URL = "http://127.0.0.1:8765"
DEFAULT_CDP_URL = "http://127.0.0.1:9222"
DEFAULT_APP_URL = f"{DEFAULT_SERVER_URL}/"
SERVER_START_TIMEOUT = 8.0
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
EXTENSION_ROOT = Path(__file__).with_name("extension")
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--mode",
		choices=("current-chrome", "managed"),
		default="current-chrome",
		help="Reuse the current Chrome application or explicitly launch managed Chromium",
	)
	parser.add_argument("--url", default=DEFAULT_APP_URL, help="Context Atlas URL to open")
	parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Loopback server base URL")
	parser.add_argument(
		"--cdp-url",
		default=DEFAULT_CDP_URL,
		help="Optional loopback CDP endpoint to verify before current-Chrome use",
	)
	parser.add_argument("--skip-server", action="store_true", help="Do not start the loopback server if health is unavailable")
	parser.add_argument(
		"--install-extension-guide",
		action="store_true",
		help="Print the one-time Chrome Load unpacked instructions",
	)
	return parser.parse_args(argv)


def verify_cdp_endpoint(endpoint: str, *, timeout: float = 1.5) -> dict[str, object] | None:
	"""Return the CDP version payload only when the standard endpoint is usable."""
	base_url = _loopback_base(endpoint)
	if base_url is None:
		return None
	request = Request(f"{base_url}/json/version", headers={"Accept": "application/json"})
	try:
		with urlopen(request, timeout=timeout) as response:
			payload = json.loads(response.read().decode("utf-8"))
	except (HTTPError, URLError, OSError, UnicodeDecodeError, json.JSONDecodeError):
		return None
	if not isinstance(payload, Mapping):
		return None
	websocket_url = payload.get("webSocketDebuggerUrl")
	if not isinstance(websocket_url, str) or not websocket_url.startswith(("ws://", "wss://")):
		return None
	return dict(payload)


def build_server_command(server_url: str, *, python_executable: str | None = None) -> list[str]:
	parsed = _require_loopback_url(server_url)
	return [
		python_executable or sys.executable,
		"-m",
		"examples.context_atlas.server",
		"--host",
		parsed.hostname or "127.0.0.1",
		"--port",
		str(parsed.port or 8765),
	]


def build_browser_command(url: str, *, platform_name: str | None = None) -> list[str]:
	name = platform_name or platform.system().lower()
	if name == "darwin":
		return ["open", "-a", "Google Chrome", url]
	if name == "windows":
		return ["cmd", "/c", "start", "", url]
	return ["xdg-open", url]


def server_is_healthy(server_url: str, *, timeout: float = 0.8) -> bool:
	base_url = _loopback_base(server_url)
	if base_url is None:
		return False
	try:
		with urlopen(f"{base_url}/api/health", timeout=timeout) as response:
			return response.status == 200
	except (HTTPError, URLError, OSError):
		return False


def start_or_reuse_server(server_url: str) -> subprocess.Popen[bytes] | None:
	"""Reuse a healthy server or start a detached loopback server process."""
	if server_is_healthy(server_url):
		LOGGER.info("Reusing Context Atlas server at %s", _safe_url(server_url))
		return None
	command = build_server_command(server_url)
	process = subprocess.Popen(
		command,
		stdin=subprocess.DEVNULL,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		start_new_session=True,
	)
	deadline = time.monotonic() + SERVER_START_TIMEOUT
	while time.monotonic() < deadline:
		if server_is_healthy(server_url):
			LOGGER.info("Started Context Atlas server at %s", _safe_url(server_url))
			return process
		if process.poll() is not None:
			raise RuntimeError("Context Atlas server exited before its health endpoint became ready")
		time.sleep(0.1)
	process.terminate()
	raise TimeoutError("Context Atlas server did not become ready")


def print_extension_guide() -> None:
	LOGGER.info("Load the extension once in the existing Chrome profile:")
	LOGGER.info("1. Open chrome://extensions and enable Developer mode.")
	LOGGER.info("2. Choose Load unpacked and select %s", EXTENSION_ROOT)
	LOGGER.info("3. Open an HTTP(S) page and click the Context Atlas toolbar action.")


async def run_managed(url: str) -> None:
	from openbrowser.browser import BrowserProfile, BrowserSession

	profile = BrowserProfile(headless=False)
	session = BrowserSession(browser_profile=profile)
	try:
		await session.start()
		await session.navigate_to(url)
		LOGGER.info("Managed browser opened %s; press Ctrl-C to stop", _safe_url(url))
		await asyncio.Event().wait()
	except (KeyboardInterrupt, asyncio.CancelledError):
		return
	finally:
		await session.kill()


def main(argv: list[str] | None = None) -> int:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args = parse_args(argv)
	try:
		if not args.skip_server:
			start_or_reuse_server(args.server_url)
		if args.install_extension_guide:
			print_extension_guide()
		if args.mode == "managed":
			asyncio.run(run_managed(args.url))
			return 0

		cdp_payload = verify_cdp_endpoint(args.cdp_url)
		if cdp_payload is None:
			LOGGER.info("No usable CDP endpoint at %s; opening the app normally without attachment", _safe_url(args.cdp_url))
		else:
			LOGGER.info("Verified current-Chrome CDP endpoint for %s", cdp_payload.get("Browser", "Chrome"))
		subprocess.Popen(build_browser_command(args.url))
		LOGGER.info("Opened Context Atlas in the existing Chrome application at %s", _safe_url(args.url))
		return 0
	except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
		LOGGER.error("Could not launch Context Atlas: %s", exc)
		return 1


def _loopback_base(url: str) -> str | None:
	try:
		parsed = _require_loopback_url(url)
	except ValueError:
		return None
	return parsed._replace(path="", query="", fragment="").geturl().rstrip("/")


def _require_loopback_url(url: str):
	parsed = urlsplit(url)
	if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOOPBACK_HOSTS:
		raise ValueError("URL must use HTTP(S) and a loopback host")
	try:
		parsed.port
	except ValueError as exc:
		raise ValueError("URL port is invalid") from exc
	return parsed


def _safe_url(url: str) -> str:
	try:
		parsed = urlsplit(url)
		return f"{parsed.scheme}://{parsed.hostname or '<unknown>'}{f':{parsed.port}' if parsed.port else ''}{parsed.path}"
	except ValueError:
		return "<invalid-url>"


if __name__ == "__main__":
	raise SystemExit(main())


__all__ = [
	"DEFAULT_APP_URL",
	"DEFAULT_CDP_URL",
	"DEFAULT_SERVER_URL",
	"EXTENSION_ROOT",
	"build_browser_command",
	"build_server_command",
	"main",
	"parse_args",
	"print_extension_guide",
	"run_managed",
	"server_is_healthy",
	"start_or_reuse_server",
	"verify_cdp_endpoint",
]
