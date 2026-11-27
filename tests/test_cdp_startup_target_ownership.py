"""Initial target attachment and Fetch policy share one owned event stream."""

from __future__ import annotations

import asyncio
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import psutil
import pytest
from cdp_use.cdp.registration_library import CDPRegistrationLibrary
from cdp_use.cdp.registry import EventRegistry
from test_cdp_connection_session import _manager_for, _page_target, _unconnected_session
from test_no_render_resource_policy import _make_session
from test_session_part1 import _make_mock_cdp_client

from openbrowser.browser.profile import ProxySettings
from openbrowser.browser.session import BrowserSession
from openbrowser.browser.session_manager import SessionManager


@pytest.fixture(autouse=True)
def no_native_io(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("This test uses fake native I/O")

    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "bind", denied)
    monkeypatch.setattr(psutil, "Process", denied)


@pytest.mark.parametrize(
    "ownership",
    ["owned", "pending", "cancelled", "failed", "ack_error", "missing", "foreign", "lookup_error", "global_only"],
)
async def test_startup_uses_only_an_owned_live_session(ownership):
    session, mocks = _unconnected_session()
    root = mocks["cdp_root"]
    target = {**_page_target(), "attached": True}
    if ownership == "missing":
        target.pop("attached")
    root.send.Target.getTargets = AsyncMock(return_value={"targetInfos": [target]})
    manager = _manager_for(mocks)
    owned = ownership in ("owned", "pending", "cancelled", "failed", "ack_error")
    candidate = SimpleNamespace(cdp_client=root if owned else object(), session_id="automatic-session")
    manager.get_live_session_for_target = AsyncMock(return_value=candidate if owned or ownership == "foreign" else None)
    entered, release = asyncio.Event(), asyncio.Event()

    async def pending_ack(session_id):
        assert session_id == "automatic-session"
        entered.set()
        await release.wait()
        return True

    manager.wait_for_auto_attach = AsyncMock(return_value=ownership != "failed")
    if ownership in ("pending", "cancelled"):
        manager.wait_for_auto_attach.side_effect = pending_ack
    elif ownership == "ack_error":
        manager.wait_for_auto_attach.side_effect = RuntimeError("ack unavailable")
    if ownership == "lookup_error":
        manager.get_live_session_for_target.side_effect = RuntimeError("lookup unavailable")
    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        patch("openbrowser.browser.session_manager.SessionManager", return_value=manager),
        patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
        patch.object(BrowserSession, "_read_document_identity", AsyncMock(return_value=None)),
        patch.object(BrowserSession, "_notify_focus_changed", AsyncMock()),
        patch.object(
            BrowserSession,
            "_focus_transition_to_target",
            AsyncMock(return_value=SimpleNamespace(session=mocks["cdp_session"], snapshot=None)),
        ),
    ):
        if ownership in ("pending", "cancelled"):
            task = asyncio.create_task(session.connect())
            try:
                await asyncio.wait_for(entered.wait(), timeout=1.0)
                assert not task.done()
                root.send.Target.attachToTarget.assert_not_awaited()
                if ownership == "cancelled":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    release.set()
                    await task
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        else:
            await session.connect()
    if ownership in ("owned", "pending", "cancelled"):
        root.send.Target.attachToTarget.assert_not_awaited()
        root.send.Target.setAutoAttach.assert_awaited_once()
    else:
        root.send.Target.attachToTarget.assert_awaited_once_with(
            params={"targetId": target["targetId"], "flatten": True}
        )
    if owned:
        manager.wait_for_auto_attach.assert_awaited_once_with("automatic-session")
    else:
        manager.wait_for_auto_attach.assert_not_awaited()


@pytest.mark.parametrize("ack_succeeds", [True, False])
async def test_startup_waits_for_real_manager_acknowledgment(ack_succeeds, monkeypatch):
    session, mocks = _unconnected_session()
    root = _make_mock_cdp_client()
    root.send = mocks["cdp_root"].send
    root.start = AsyncMock()
    root.stop = AsyncMock()
    manual_attach = AsyncMock(return_value={"sessionId": "manual-session"})
    root.send.Target.attachToTarget = manual_attach
    registry = EventRegistry()
    root.register = CDPRegistrationLibrary(registry)
    target = {**_page_target(), "attached": True}
    pooled, listed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def note_pool(*_):
        pooled.set()

    async def auto_attach(*, params, session_id=None):
        if session_id is None:
            await registry.handle_event(
                "Target.attachedToTarget",
                {"sessionId": "automatic-session", "targetInfo": target, "waitingForDebugger": False},
            )
            await pooled.wait()
        elif session_id == "automatic-session":
            await release.wait()
            if not ack_succeeds:
                raise RuntimeError("recursive attachment rejected")
        return {}

    async def list_targets():
        listed.set()
        return {"targetInfos": [target]}

    root.send.Target.setAutoAttach = AsyncMock(side_effect=auto_attach)
    root.send.Target.getTargets = AsyncMock(side_effect=list_targets)
    monkeypatch.setattr(session, "_handle_attached_session", AsyncMock(side_effect=note_pool))
    with (
        patch("openbrowser.browser.session.CDPClient", return_value=root),
        patch.object(BrowserSession, "_setup_proxy_auth", AsyncMock()),
        patch.object(BrowserSession, "_read_document_identity", AsyncMock(return_value=None)),
        patch.object(BrowserSession, "_notify_focus_changed", AsyncMock()),
        patch.object(
            BrowserSession,
            "_focus_transition_to_target",
            AsyncMock(return_value=SimpleNamespace(session=mocks["cdp_session"], snapshot=None)),
        ),
    ):
        task = asyncio.create_task(session.connect())
        try:
            await asyncio.wait_for(listed.wait(), timeout=1.0)
            assert not task.done()
            manual_attach.assert_not_awaited()
            release.set()
            await asyncio.wait_for(task, timeout=1.0)
            if ack_succeeds:
                manual_attach.assert_not_awaited()
            else:
                manual_attach.assert_awaited_once_with(params={"targetId": target["targetId"], "flatten": True})
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if session._session_manager is not None:
                await session._session_manager.clear()


@pytest.mark.parametrize("mode", ["lightweight", "no-render", "proxy"])
@pytest.mark.parametrize("target_type", ["page", "iframe", "worker", "service_worker", "background_page"])
async def test_fetch_keeps_session_pool_and_recursive_attachment(mode, target_type, monkeypatch):
    session = _make_session(None if mode == "proxy" else mode)
    if mode == "proxy":
        session.browser_profile.proxy = ProxySettings(
            server="http://proxy.example.test:8080", username="user", password="pass"
        )
    root = _make_mock_cdp_client()
    root.send.Fetch.enable = AsyncMock()
    root.send.Target.setAutoAttach = AsyncMock()
    session._cdp_client_root = root
    monkeypatch.setattr(session, "_handle_attached_session", AsyncMock())
    registry = EventRegistry()
    root.register = CDPRegistrationLibrary(registry)
    manager = SessionManager(session)
    session._session_manager = manager
    await manager.start_monitoring()
    try:
        await session._setup_proxy_auth()
        event = {
            "sessionId": "new-session",
            "waitingForDebugger": False,
            "targetInfo": {
                "targetId": "new-target",
                "type": target_type,
                "title": "Example",
                "url": "https://example.test",
            },
        }
        assert await registry.handle_event("Target.attachedToTarget", event)
        for _ in range(8):
            await asyncio.sleep(0)
        assert session._cdp_session_pool["new-target"].session_id == "new-session"
        assert manager._target_sessions["new-target"] == {"new-session"}
        root.send.Target.setAutoAttach.assert_awaited_once_with(
            params={"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True}, session_id="new-session"
        )
        assert any(call.kwargs.get("session_id") == "new-session" for call in root.send.Fetch.enable.await_args_list)
    finally:
        await manager.clear()
        tasks = list(session._fetch_interceptor_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
