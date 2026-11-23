"""Real-browser focus mechanics for M04."""

import asyncio
import inspect
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.browser.events import AgentFocusChangedEvent
from openbrowser.browser.focus import FocusChangeReason, FocusPhase, FocusSnapshot, FocusTransitionBudget
from openbrowser.tools.service import Tools

pytestmark = pytest.mark.integration


class _FocusHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        path = urlsplit(self.path)
        nonce = parse_qs(path.query).get("nonce", ["focus"])[0]
        body = (
            f'<!doctype html><html><head><title>{nonce}</title></head><body><h1 id="marker">{nonce}</h1></body></html>'
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def focus_site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FocusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _eventually(predicate, wait_seconds: float = 10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_seconds
    while loop.time() < deadline:
        result = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return result
        await asyncio.sleep(0.05)
    raise AssertionError("condition did not become true before the deadline")


async def test_switch_rebind_and_close_repair_with_real_chrome(focus_site, tmp_path):
    initial_observer_failed = False

    async def fail_initial_observation(event: AgentFocusChangedEvent):
        nonlocal initial_observer_failed
        if event.focus_reason is FocusChangeReason.INITIAL and not initial_observer_failed:
            initial_observer_failed = True
            raise RuntimeError("initial observer failed")

    browser = BrowserSession(
        browser_profile=BrowserProfile(
            headless=True,
            user_data_dir=str(tmp_path / "profile"),
            downloads_path=str(tmp_path / "downloads"),
            enable_default_extensions=False,
            keep_alive=False,
        )
    )
    browser.event_bus.on(AgentFocusChangedEvent, fail_initial_observation)
    try:
        await asyncio.wait_for(browser.start(), timeout=60)
        assert initial_observer_failed
        assert browser.focus_snapshot().phase is FocusPhase.FOCUSED
        health = browser.browser_health_snapshot()
        assert health["phase"] == "healthy"
        assert health["reason"] == "connected"
        assert health["process_present"] is True
        assert health["process_pid"] is not None
        await asyncio.wait_for(browser.navigate_to(f"{focus_site}/first?nonce=first"), timeout=30)
        first = browser.focus_snapshot()
        assert first.phase is FocusPhase.FOCUSED
        assert first.target_id is not None

        second_target = await browser._cdp_create_new_page(f"{focus_site}/second?nonce=second")
        await asyncio.wait_for(
            browser.get_or_create_cdp_session(
                second_target,
                focus=True,
                budget=FocusTransitionBudget.start(),
            ),
            timeout=15,
        )
        switched = browser.focus_snapshot()
        assert switched.phase is FocusPhase.FOCUSED
        assert switched.target_id == second_target
        assert switched.reason is FocusChangeReason.SWITCH

        attached = await browser._cdp_client_root.send.Target.attachToTarget(
            params={"targetId": second_target, "flatten": True}
        )
        attached_session_id = attached["sessionId"]
        manager = browser._session_manager
        assert manager is not None

        async def has_attached_session():
            return attached_session_id in await manager.active_session_ids(second_target)

        await _eventually(has_attached_session)
        rebound = await _eventually(
            lambda: (
                browser.focus_snapshot()
                if browser.focus_snapshot().phase is FocusPhase.FOCUSED
                and browser.focus_snapshot().session_id == attached_session_id
                else None
            )
        )
        assert rebound.target_id == second_target
        assert rebound.reason is FocusChangeReason.SESSION_REBIND
        assert isinstance(rebound, FocusSnapshot)

        tools = Tools()
        closed_result = await asyncio.wait_for(
            tools.registry.execute_action(
                "close",
                {"tab_id": second_target[-4:]},
                browser_session=browser,
            ),
            timeout=30,
        )
        assert closed_result.success is not False, closed_result.error
        repaired = await _eventually(
            lambda: (
                browser.focus_snapshot()
                if browser.focus_snapshot().phase is FocusPhase.FOCUSED
                and browser.focus_snapshot().target_id == first.target_id
                else None
            ),
            wait_seconds=15,
        )

        assert repaired.session_id
        assert repaired.reason is FocusChangeReason.CLOSE_REPAIR
        assert browser.agent_focus is not None
        assert browser.agent_focus.target_id == first.target_id
        closed = browser.tab_snapshot()
        assert next(tab for tab in closed if tab["target_id"] == second_target)["state"] == "closed"
        assert next(tab for tab in closed if tab["target_id"] == first.target_id)["state"] == "open"
    finally:
        await asyncio.wait_for(browser.kill(), timeout=30)
