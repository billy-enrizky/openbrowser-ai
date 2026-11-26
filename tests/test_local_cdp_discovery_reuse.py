"""Use the local readiness response as the final CDP connection endpoint."""

from __future__ import annotations

import asyncio
import socket
import ssl
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from openbrowser.browser.session import BrowserSession
from openbrowser.browser.watchdogs import local_browser_watchdog as module


@pytest.fixture
def discovery(monkeypatch):
    state = SimpleNamespace(now=0.0, responses=[], gets=[], constructors=0, enters=0, exits=0, sleeps=[])

    class Response:
        def __init__(self, payload, status=200):
            self.status_code = status
            self.payload = payload

        def json(self):
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    class Client:
        async def __aenter__(self):
            state.enters += 1
            return self

        async def __aexit__(self, *args):
            state.exits += 1
            return False

        async def get(self, url, **kwargs):
            state.gets.append(url)
            response = state.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response

    def factory(**kwargs):
        state.constructors += 1
        return Client()

    async def sleep(delay):
        state.sleeps.append(delay)
        state.now += 1.0

    def forbidden(*args, **kwargs):
        raise AssertionError("local discovery tests require fake native I/O")

    state.real_client_type = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    monkeypatch.setattr(
        module, "asyncio", SimpleNamespace(get_event_loop=lambda: SimpleNamespace(time=lambda: state.now), sleep=sleep)
    )
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "listen", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    state.response = Response
    return state


@pytest.mark.parametrize(
    "url",
    [
        "ws://localhost:9222/devtools/browser/browser-id",
        "wss://debug.example.test/devtools/browser/browser-id",
    ],
)
async def test_readiness_returns_the_discovered_websocket_url(discovery, url):
    discovery.responses.append(discovery.response({"webSocketDebuggerUrl": url}))
    assert await module.LocalBrowserWatchdog._wait_for_cdp_url(9222) == url
    assert discovery.gets == ["http://localhost:9222/json/version"]
    assert discovery.constructors == discovery.enters == discovery.exits == 1
    assert discovery.sleeps == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        None,
        [],
        "invalid",
        {"webSocketDebuggerUrl": None},
        {"webSocketDebuggerUrl": 1},
        {"webSocketDebuggerUrl": ""},
        {"webSocketDebuggerUrl": "http://localhost:9222/"},
        {"webSocketDebuggerUrl": "ws://"},
        {"webSocketDebuggerUrl": " ws://localhost:9222/browser"},
        {"webSocketDebuggerUrl": "ws://localhost:bad/browser"},
        {"webSocketDebuggerUrl": "ws://localhost:9222/browser#fragment"},
        ValueError("invalid JSON"),
    ],
)
async def test_incomplete_metadata_retries_in_the_existing_client(discovery, payload):
    url = "ws://localhost:9222/devtools/browser/ready"
    discovery.responses.extend([discovery.response(payload), discovery.response({"webSocketDebuggerUrl": url})])
    assert await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=5) == url
    assert discovery.gets == ["http://localhost:9222/json/version"] * 2
    assert discovery.sleeps == [0.02]
    assert discovery.constructors == discovery.exits == 1


async def test_bad_status_and_connection_failure_keep_retrying(discovery):
    url = "ws://localhost:9222/devtools/browser/ready"
    discovery.responses.extend(
        [
            discovery.response({}, status=503),
            httpx.ConnectError("starting"),
            discovery.response({"webSocketDebuggerUrl": url}),
        ]
    )
    assert await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=5) == url
    assert len(discovery.gets) == 3
    assert discovery.sleeps == [0.02, 0.02]
    assert discovery.constructors == discovery.exits == 1


async def test_incomplete_metadata_exhausts_the_existing_deadline(discovery):
    discovery.responses.extend([discovery.response({}), discovery.response({})])
    with pytest.raises(TimeoutError, match="Browser did not start within 2 seconds"):
        await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=2)
    assert len(discovery.gets) == 2
    assert discovery.exits == 1


async def test_cancelled_discovery_closes_client_and_preserves_cancellation(discovery):
    discovery.responses.append(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await module.LocalBrowserWatchdog._wait_for_cdp_url(9222)
    assert discovery.exits == 1
    assert discovery.sleeps == []


async def test_local_discovery_skips_the_second_session_http_client(discovery, monkeypatch):
    class AdmissionProbe(RuntimeError):
        pass

    url = "ws://localhost:9222/devtools/browser/ready"
    discovery.responses.extend([discovery.response({"webSocketDebuggerUrl": url})] * 2)
    resolved = await module.LocalBrowserWatchdog._wait_for_cdp_url(9222)
    session = BrowserSession(cdp_url=resolved)

    admission_checks = []

    def stop_before_websocket(**kwargs):
        admission_checks.append(kwargs)
        if len(admission_checks) == 2:
            raise AdmissionProbe("connection endpoint resolution finished")

    monkeypatch.setattr(session, "_raise_if_connect_admission_closed", stop_before_websocket)
    monkeypatch.setattr(session, "_await_connect_failure", AsyncMock())
    with pytest.raises(AdmissionProbe):
        await session._connect_locked()
    assert len(admission_checks) == 2
    assert session.cdp_url == url
    assert discovery.constructors == 1
    assert discovery.gets == ["http://localhost:9222/json/version"]


async def test_plain_http_discovery_keeps_strict_tls_without_loading_ca_roots(discovery, monkeypatch):
    url = "ws://localhost:9222/devtools/browser/ready"
    configured = []

    def ca_load_forbidden(*args, **kwargs):
        raise AssertionError("plain HTTP discovery loaded a certificate store")

    async def version_response(self, requested_url, **kwargs):
        assert requested_url == "http://localhost:9222/json/version"
        return httpx.Response(200, json={"webSocketDebuggerUrl": url})

    def client_factory(**kwargs):
        configured.append(kwargs)
        return discovery.real_client_type(**kwargs)

    monkeypatch.setattr(ssl, "create_default_context", ca_load_forbidden)
    monkeypatch.setattr(discovery.real_client_type, "get", version_response)
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    assert await module.LocalBrowserWatchdog._wait_for_cdp_url(9222) == url
    context = configured[0]["verify"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert configured[0]["trust_env"] is False
