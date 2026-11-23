"""chrome-devtools-mcp 1.13.0 adapter driven over stdio JSON-RPC."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from ..fixtures.server import ACCOUNT_SECRET, BENCH_PASSWORD, BENCH_USER
from .base import ArmResult, RunContext, terminate_matching, verify_mock_keychain
from .mcp_stdio import McpError, StdioMcpClient

CHROME = os.environ.get(
    "CHROME_BIN",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
SERVER = os.environ.get(
    "CHROME_DEVTOOLS_MCP_BIN",
    str(
        Path.home()
        / ".cache"
        / "openbrowser-interactive-bench"
        / "npm"
        / "node_modules"
        / ".bin"
        / "chrome-devtools-mcp"
    ),
)

UID_LINE = re.compile(r"uid=(\S+)\s+(?:([A-Za-z][A-Za-z0-9_]*)\s+)?\"([^\"]*)\"")
UID_BARE = re.compile(r"uid=(\S+)\s+([A-Za-z][A-Za-z0-9_]*)")
PAGE_LINE = re.compile(r"^(\d+):", re.MULTILINE)


def _text(result: dict) -> str:
    return "\n".join(
        item.get("text", "")
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _parse_snapshot(snapshot: str) -> list[dict]:
    entries: list[dict] = []
    for line in snapshot.splitlines():
        match = UID_LINE.search(line)
        if match:
            entries.append({"uid": match.group(1), "role": match.group(2), "name": match.group(3)})
            continue
        match = UID_BARE.search(line)
        if match:
            entries.append({"uid": match.group(1), "role": match.group(2), "name": ""})
    return entries


def _find(entries: list[dict], *, role: str | None = None, name: str | None = None) -> str:
    for entry in entries:
        if role is not None and entry["role"] != role:
            continue
        if name is not None and entry["name"] != name:
            continue
        return entry["uid"]
    raise RuntimeError(f"uid not found role={role} name={name}; entries={entries[:30]}")


def _json_value(text: str):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _write_download_preferences(profile: Path, downloads: Path) -> None:
    default = profile / "Default"
    default.mkdir(parents=True, exist_ok=True)
    preferences = default / "Preferences"
    if preferences.exists():
        try:
            data = json.loads(preferences.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    data.setdefault("download", {})
    data["download"].update(
        {
            "default_directory": str(downloads),
            "prompt_for_download": False,
            "directory_upgrade": True,
        }
    )
    data.setdefault("savefile", {})
    data["savefile"]["default_directory"] = str(downloads)
    preferences.write_text(json.dumps(data))


class ChromeDevToolsArm:
    name = "chrome-devtools-mcp"

    def __init__(self) -> None:
        self._client: StdioMcpClient | None = None
        self._page_id: int | None = None
        self._result = ArmResult(ok=False)

    def _start(self, ctx: RunContext) -> None:
        _write_download_preferences(ctx.profile, ctx.downloads)
        command = [
            SERVER,
            "--headless",
            "--executablePath",
            CHROME,
            "--userDataDir",
            str(ctx.profile),
            "--viewport",
            "1280x800",
            "--no-usage-statistics",
            "--no-performance-crux",
            "--chromeArg=--use-mock-keychain",
        ]
        self._client = StdioMcpClient(command, env=ctx.env, timeout=120)
        self._result.process_starts += 1
        init = self._client.initialize()
        self._result.browser_version = init.get("serverInfo", {}).get("version", "")
        pages = self._pages()
        if not pages:
            raise RuntimeError("chrome-devtools-mcp reported no pages")
        self._page_id = pages[0][0]
        verified = verify_mock_keychain(str(ctx.profile))
        self._result.keychain_verified = verified
        if verified is False:
            raise RuntimeError("chrome-devtools-mcp Chrome launched without --use-mock-keychain")

    def _call(self, name: str, arguments: dict, *, timeout: float = 120) -> str:
        assert self._client is not None
        self._result.tool_calls += 1
        try:
            result = self._client.call_tool(name, arguments, timeout=timeout)
        except McpError as exc:
            raise RuntimeError(str(exc)) from exc
        self._result.stdout += _text(result)
        if result.get("isError"):
            raise RuntimeError(f"{name} error: {_text(result)[:800]}")
        return _text(result)

    def _pages(self) -> list[tuple[int, str]]:
        text = self._call("list_pages", {})
        pages = []
        for line in text.splitlines():
            match = PAGE_LINE.match(line.strip())
            if match:
                pages.append((int(match.group(1)), line.strip()))
        return pages

    def _navigate(self, url: str) -> None:
        assert self._page_id is not None
        self._call("navigate_page", {"pageId": self._page_id, "type": "url", "url": url})
        self._wait_js("document.readyState !== 'loading'")

    def _snapshot(self) -> list[dict]:
        assert self._page_id is not None
        text = self._call("take_snapshot", {"pageId": self._page_id})
        return _parse_snapshot(text)

    def _snapshot_page(self, page_id: int) -> list[dict]:
        text = self._call("take_snapshot", {"pageId": page_id})
        return _parse_snapshot(text)

    def _click(self, uid: str, page_id: int | None = None) -> None:
        self._call("click", {"pageId": page_id or self._page_id, "uid": uid})

    def _fill(self, uid: str, value: str, page_id: int | None = None) -> None:
        self._call("fill", {"pageId": page_id or self._page_id, "uid": uid, "value": value})

    def _press(self, key: str, page_id: int | None = None) -> None:
        self._call("press_key", {"pageId": page_id or self._page_id, "key": key})

    def _evaluate(self, expression: str, page_id: int | None = None):
        text = self._call(
            "evaluate_script",
            {"pageId": page_id or self._page_id, "function": f"() => ({expression})"},
        )
        return _json_value(text)

    def _wait_js(self, expression: str, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._evaluate(expression) is True:
                return
            time.sleep(0.15)
        raise RuntimeError(f"timeout waiting for {expression}")

    def _run_wizard(self, ctx: RunContext) -> None:
        self._navigate(ctx.base_url + "/wizard")
        entries = self._snapshot()
        self._fill(_find(entries, role="textbox", name="Full name"), "Alice Bench")
        self._fill(_find(entries, role="textbox", name="Email"), "alice@example.com")
        self._fill(_find(entries, role="radio", name="Canada"), "true")
        self._wait_js("!document.getElementById('province-group').hidden")
        entries = self._snapshot()
        self._fill(_find(entries, role="radio", name="Ontario"), "true")
        self._click(_find(entries, role="button", name="Next"))
        self._wait_js("!!document.getElementById('step2')")
        entries = self._snapshot()
        self._fill(_find(entries, role="radio", name="Pro"), "true")
        self._fill(_find(entries, role="spinbutton", name="Seats"), "5")
        self._click(_find(entries, role="button", name="Next"))
        self._wait_js("!!document.getElementById('step3')")
        entries = self._snapshot()
        self._fill(_find(entries, role="checkbox", name="I accept the terms"), "true")
        self._click(_find(entries, role="button", name="Submit"))
        self._wait_js("!!document.getElementById('quote')")
        self._result.reported = {"quote": self._evaluate("document.getElementById('quote').textContent")}

    def _run_modal(self, ctx: RunContext) -> None:
        self._navigate(ctx.base_url + "/modal")
        entries = self._snapshot()
        self._click(_find(entries, role="button", name="Choose project"))
        self._wait_js("!document.getElementById('modal').hidden")
        self._press("Escape")
        self._wait_js("document.getElementById('modal').hidden")
        entries = self._snapshot()
        self._click(_find(entries, role="button", name="Choose project"))
        self._wait_js("!document.getElementById('modal').hidden")
        entries = self._snapshot()
        self._fill(_find(entries, role="combobox"), "Atlas")
        self._wait_js(
            "[...document.querySelectorAll('#project-list [role=option]')].filter(o => !o.hidden).length === 1"
        )
        # chrome-devtools-mcp 1.13.0 does not expose the listbox options in
        # its accessibility snapshot, so the native fallback is script
        # evaluation. The option click still runs in the page.
        self._call(
            "evaluate_script",
            {
                "pageId": self._page_id,
                "function": (
                    "() => { document.querySelector('[data-value=\"Project Atlas\"]').click(); "
                    "return true; }"
                ),
            },
        )
        self._wait_js("document.getElementById('selected-value').textContent === 'Project Atlas'")
        self._result.reported = {"selected": self._evaluate("document.getElementById('selected-value').textContent")}

    def _run_download(self, ctx: RunContext) -> None:
        self._navigate(ctx.base_url + "/download")
        entries = self._snapshot()
        self._click(_find(entries, role="button", name="Download report"))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            files = [path for path in ctx.downloads.rglob("*") if path.is_file()]
            if files:
                self._result.reported = {"files": [str(path) for path in files]}
                return
            time.sleep(0.2)
        raise RuntimeError(
            "chrome-devtools-mcp exposes no download lifecycle and Chrome saved no file to "
            f"{ctx.downloads}"
        )

    def _run_tabs(self, ctx: RunContext) -> None:
        self._navigate(ctx.base_url + "/tabs")
        assert self._page_id is not None
        original = self._page_id
        entries = self._snapshot()
        self._click(_find(entries, role="link", name="Compare plans"))
        deadline = time.monotonic() + 20
        detail = None
        while time.monotonic() < deadline:
            pages = self._pages()
            others = [page for page in pages if page[0] != original]
            if others:
                detail = others[0][0]
                break
            time.sleep(0.15)
        if detail is None:
            raise RuntimeError("detail page did not open")
        entries = self._snapshot_page(detail)
        self._click(_find(entries, role="button", name="Approve"), page_id=detail)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self._evaluate("document.getElementById('approved')?.textContent === 'approved'", page_id=detail):
                break
            time.sleep(0.15)
        self._call("close_page", {"pageId": detail})
        deadline = time.monotonic() + 20
        status = None
        while time.monotonic() < deadline:
            status = self._evaluate("document.getElementById('status').textContent", page_id=original)
            if status == "approved:42":
                break
            time.sleep(0.15)
        self._result.reported = {
            "status": status,
            "original_title": self._evaluate("document.title", page_id=original),
        }

    def _run_auth(self, ctx: RunContext) -> None:
        self._start(ctx)
        try:
            self._navigate(ctx.base_url + "/login")
            entries = self._snapshot()
            self._fill(_find(entries, role="textbox", name="Username"), BENCH_USER)
            self._fill(_find(entries, role="textbox", name="Password"), BENCH_PASSWORD)
            self._click(_find(entries, role="button", name="Sign in"))
            self._wait_js("!!document.getElementById('secret')")
            phase1 = self._evaluate("document.getElementById('secret').textContent")
        finally:
            self._stop_client()
        self._start(ctx)
        try:
            self._navigate(ctx.base_url + "/account")
            self._wait_js("!!document.getElementById('secret')")
            secret = self._evaluate("document.getElementById('secret').textContent")
        finally:
            self._stop_client()
        self._result.reported = {"secret": secret, "phase1_secret": phase1}
        if secret != ACCOUNT_SECRET:
            raise RuntimeError(f"secret after restart was {secret!r}")

    def _stop_client(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def run(self, ctx: RunContext) -> ArmResult:
        try:
            if ctx.task == "auth":
                self._run_auth(ctx)
            else:
                self._start(ctx)
                if ctx.task == "wizard":
                    self._run_wizard(ctx)
                elif ctx.task == "modal":
                    self._run_modal(ctx)
                elif ctx.task == "download":
                    self._run_download(ctx)
                elif ctx.task == "tabs":
                    self._run_tabs(ctx)
                else:
                    raise ValueError(f"unknown task {ctx.task}")
            self._result.ok = True
        except Exception as exc:
            self._result.ok = False
            self._result.error = str(exc)
        finally:
            self._stop_client()
            self.cleanup(ctx)
        return self._result

    def cleanup(self, ctx: RunContext) -> None:
        self._stop_client()
        terminate_matching(str(ctx.profile))
        terminate_matching(str(ctx.root))
