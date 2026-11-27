"""Use the local readiness response as the final CDP connection endpoint."""

from __future__ import annotations

import asyncio
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from openbrowser.browser import local_discovery
from openbrowser.browser.session import BrowserSession
from openbrowser.browser.watchdogs import local_browser_watchdog as module


@pytest.fixture
def discovery(monkeypatch):
    state = SimpleNamespace(
        now=0.0,
        responses=[],
        calls=[],
        sleeps=[],
        fetch_delay=0.0,
    )

    async def fetch(port, *, timeout):  # noqa: ASYNC109
        state.calls.append((port, timeout))
        state.now += state.fetch_delay
        if not state.responses:
            raise local_discovery.LocalDiscoveryError("incomplete")
        value = state.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def sleep(delay):
        state.sleeps.append(delay)
        state.now += delay

    loop = SimpleNamespace(time=lambda: state.now)

    def forbidden(*args, **kwargs):
        raise AssertionError("local discovery tests require fake native I/O")

    monkeypatch.setattr(module.local_discovery, "fetch_loopback_version", fetch)
    monkeypatch.setattr(
        module,
        "asyncio",
        SimpleNamespace(
            CancelledError=asyncio.CancelledError,
            get_event_loop=lambda: loop,
            get_running_loop=lambda: loop,
            sleep=sleep,
        ),
    )
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    monkeypatch.setattr(socket.socket, "listen", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    return state


@pytest.mark.parametrize(
    "url",
    [
        "ws://localhost:9222/devtools/browser/browser-id",
        "wss://127.0.0.1:9222/devtools/browser/browser-id",
        "ws://[::1]:9222/devtools/browser/browser-id",
    ],
)
async def test_readiness_returns_the_discovered_websocket_url(discovery, url):
    discovery.responses.append({"webSocketDebuggerUrl": url})
    assert await module.LocalBrowserWatchdog._wait_for_cdp_url(9222) == url
    assert discovery.calls == [(9222, 2.0)]
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
        {"webSocketDebuggerUrl": "ws://localhost/devtools/browser/foreign"},
        {"webSocketDebuggerUrl": "ws://localhost:9223/devtools/browser/foreign"},
        {"webSocketDebuggerUrl": "ws://debug.example.test:9222/devtools/browser/foreign"},
        {"webSocketDebuggerUrl": "ws://user@localhost:9222/devtools/browser/foreign"},
        {"webSocketDebuggerUrl": "ws://user:password@localhost:9222/devtools/browser/foreign"},
        {"webSocketDebuggerUrl": "ws://localhost:9222/devtools/browser/foreign?token=test"},
        {"webSocketDebuggerUrl": "ws://localhost:9222/devtools/browser/foreign?"},
        {"webSocketDebuggerUrl": "ws://localhost:9222/devtools/browser/"},
        {"webSocketDebuggerUrl": "ws://localhost:9222/devtools/page/foreign"},
        local_discovery.LocalDiscoveryError("invalid JSON"),
    ],
)
async def test_incomplete_metadata_retries(discovery, payload):
    url = "ws://localhost:9222/devtools/browser/ready"
    discovery.responses.extend([payload, {"webSocketDebuggerUrl": url}])
    assert await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=5) == url
    assert discovery.calls == [(9222, 2.0), (9222, 2.0)]
    assert discovery.sleeps == [0.02]


async def test_helper_failures_keep_retrying(discovery):
    url = "ws://localhost:9222/devtools/browser/ready"
    discovery.responses.extend(
        [
            local_discovery.LocalDiscoveryError("starting"),
            TimeoutError("attempt timeout"),
            {"webSocketDebuggerUrl": url},
        ]
    )
    assert await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=5) == url
    assert len(discovery.calls) == 3
    assert discovery.sleeps == [0.02, 0.02]


async def test_deadline_is_clamped_and_exhausted(discovery):
    with pytest.raises(TimeoutError, match=r"Browser did not start within 0\.02 seconds"):
        await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=0.02)
    assert discovery.calls == [(9222, 0.02)]
    assert discovery.sleeps == [0.02]


async def test_retry_sleep_is_clamped_to_remaining(discovery):
    discovery.fetch_delay = 0.015
    with pytest.raises(TimeoutError, match=r"Browser did not start within 0\.02 seconds"):
        await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=0.02)
    assert discovery.calls == [(9222, 0.02)]
    assert discovery.sleeps == [pytest.approx(0.005)]


async def test_cancelled_discovery_propagates(discovery):
    discovery.responses.append(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await module.LocalBrowserWatchdog._wait_for_cdp_url(9222)
    assert discovery.calls == [(9222, 2.0)]
    assert discovery.sleeps == []


async def test_local_discovery_skips_the_second_session_http_client(discovery, monkeypatch):
    class AdmissionProbe(RuntimeError):
        pass

    url = "ws://localhost:9222/devtools/browser/ready"
    discovery.responses.append({"webSocketDebuggerUrl": url})
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
    assert discovery.calls == [(9222, 2.0)]


async def test_explicit_remote_http_attach_preserves_headers_and_client_policy(monkeypatch):
    class AdmissionProbe(RuntimeError):
        pass

    state = SimpleNamespace(gets=[], get_kwargs=[], constructor_kwargs=[])

    class Response:
        def json(self):
            return {"webSocketDebuggerUrl": "wss://debug.example.test:9443/devtools/browser/remote"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            state.gets.append(url)
            state.get_kwargs.append(kwargs)
            return Response()

    def client_factory(**kwargs):
        state.constructor_kwargs.append(kwargs)
        return Client()

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    session = BrowserSession(cdp_url="https://debug.example.test:9443/", headers={"Authorization": "test-only"})
    checks = []

    def stop_after_resolution(**kwargs):
        checks.append(kwargs)
        if len(checks) == 2:
            raise AdmissionProbe("remote metadata resolved")

    monkeypatch.setattr(session, "_raise_if_connect_admission_closed", stop_after_resolution)
    monkeypatch.setattr(session, "_await_connect_failure", AsyncMock())
    with pytest.raises(AdmissionProbe):
        await session._connect_locked()
    assert session.cdp_url == "wss://debug.example.test:9443/devtools/browser/remote"
    assert state.gets == ["https://debug.example.test:9443/json/version"]
    assert state.get_kwargs == [{"headers": {"Authorization": "test-only"}}]
    assert "verify" not in state.constructor_kwargs[0]
    assert "trust_env" not in state.constructor_kwargs[0]


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), True, None, "bad", 10**1000])
async def test_nonfinite_or_malformed_overall_deadline_fails_before_connect(monkeypatch, timeout):
    async def forbidden(*args, **kwargs):
        pytest.fail("invalid overall deadline must not start a request")

    monkeypatch.setattr(module.local_discovery, "fetch_loopback_version", forbidden)
    with pytest.raises(ValueError, match="finite number"):
        await module.LocalBrowserWatchdog._wait_for_cdp_url(9222, timeout=timeout)
