"""Unit tests for the CDP Fetch render resource policy.

These tests exercise the handler boundary directly with a mocked CDP client.
The real-network proof lives in tests/test_no_render_resource_policy_e2e.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from openbrowser.browser.profile import BrowserProfile, ProxySettings
from openbrowser.browser.session import (
    LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES,
    NO_RENDER_BLOCKED_RESOURCE_TYPES,
    BrowserSession,
    blocked_resource_types_for_render_mode,
)


def _make_session(render_mode: str | None = None) -> BrowserSession:
    profile = BrowserProfile(cdp_url="ws://127.0.0.1:9222", headless=True)
    if render_mode is not None:
        profile.render_mode = render_mode
    return BrowserSession(browser_profile=profile)


def _mock_root() -> MagicMock:
    root = MagicMock()
    root.send.Fetch.enable = AsyncMock()
    root.send.Fetch.failRequest = AsyncMock()
    root.send.Fetch.continueRequest = AsyncMock()
    root.send.Fetch.continueWithAuth = AsyncMock()
    root.register.Fetch.requestPaused = MagicMock()
    root.register.Fetch.authRequired = MagicMock()
    root.register.Target.attachedToTarget = MagicMock()
    return root


async def _drain() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _enabled_params(root: MagicMock, session_id: str | None = None) -> dict:
    for call in root.send.Fetch.enable.call_args_list:
        if call.kwargs.get("session_id") == session_id:
            return call.kwargs["params"]
    raise AssertionError(f"Fetch.enable was not called for session {session_id}")


def test_blocked_resource_types_per_mode():
    assert blocked_resource_types_for_render_mode("full") == ()
    assert blocked_resource_types_for_render_mode(BrowserProfile().effective_render_mode) == ()
    assert blocked_resource_types_for_render_mode("lightweight") == LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES
    assert blocked_resource_types_for_render_mode("no-render") == NO_RENDER_BLOCKED_RESOURCE_TYPES
    assert "Stylesheet" in NO_RENDER_BLOCKED_RESOURCE_TYPES
    assert "Stylesheet" not in LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES
    # Unknown values fail open so a typo cannot silently break rendering.
    assert blocked_resource_types_for_render_mode("bogus") == ()


@pytest.mark.asyncio
async def test_full_mode_without_proxy_skips_fetch_domain():
    session = _make_session()
    root = _mock_root()
    session._cdp_client_root = root

    await session._setup_proxy_auth()

    root.send.Fetch.enable.assert_not_called()
    root.register.Fetch.requestPaused.assert_not_called()
    root.register.Fetch.authRequired.assert_not_called()


@pytest.mark.asyncio
async def test_lightweight_enables_patterns_without_stylesheet():
    session = _make_session("lightweight")
    root = _mock_root()
    session._cdp_client_root = root

    await session._setup_proxy_auth()

    params = _enabled_params(root, None)
    assert params["handleAuthRequests"] is False
    assert [pattern["resourceType"] for pattern in params["patterns"]] == list(LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES)
    assert all(pattern["requestStage"] == "Request" for pattern in params["patterns"])
    root.register.Fetch.requestPaused.assert_called_once()
    root.register.Fetch.authRequired.assert_not_called()
    root.register.Target.attachedToTarget.assert_called_once()


@pytest.mark.asyncio
async def test_no_render_blocks_stylesheet():
    session = _make_session("no-render")
    root = _mock_root()
    session._cdp_client_root = root

    await session._setup_proxy_auth()

    params = _enabled_params(root, None)
    resource_types = [pattern["resourceType"] for pattern in params["patterns"]]
    assert resource_types == list(NO_RENDER_BLOCKED_RESOURCE_TYPES)
    assert "Stylesheet" in resource_types


@pytest.mark.asyncio
async def test_request_paused_blocks_assets_and_continues_document():
    session = _make_session("no-render")
    root = _mock_root()
    session._cdp_client_root = root

    await session._setup_proxy_auth()
    handler = root.register.Fetch.requestPaused.call_args[0][0]

    handler({"requestId": "img-1", "resourceType": "Image"}, "session-1")
    handler({"requestId": "css-1", "resourceType": "Stylesheet"}, "session-1")
    handler({"requestId": "doc-1", "resourceType": "Document"}, "session-1")
    handler({"requestId": "script-1", "resourceType": "Script"}, "session-1")
    handler({"requestId": "xhr-1", "resourceType": "XHR"}, "session-1")
    handler({"requestId": "fetch-1", "resourceType": "Fetch"}, "session-1")
    await _drain()

    blocked = root.send.Fetch.failRequest.call_args_list
    assert [call.kwargs["params"]["requestId"] for call in blocked] == ["img-1", "css-1"]
    assert all(call.kwargs["params"]["errorReason"] == "BlockedByClient" for call in blocked)
    assert all(call.kwargs["session_id"] == "session-1" for call in blocked)

    continued = root.send.Fetch.continueRequest.call_args_list
    assert [call.kwargs["params"]["requestId"] for call in continued] == ["doc-1", "script-1", "xhr-1", "fetch-1"]


@pytest.mark.asyncio
async def test_duplicate_paused_delivery_is_decided_once():
    session = _make_session("lightweight")
    root = _mock_root()
    session._cdp_client_root = root

    await session._setup_proxy_auth()
    handler = root.register.Fetch.requestPaused.call_args[0][0]

    handler({"requestId": "img-1", "resourceType": "Image"}, "session-1")
    handler({"requestId": "img-1", "resourceType": "Image"}, "session-1")
    await _drain()

    root.send.Fetch.failRequest.assert_awaited_once()
    root.send.Fetch.continueRequest.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_auth_and_render_policy_share_one_handler():
    session = _make_session("no-render")
    session.browser_profile.proxy = ProxySettings(server="http://proxy:8080", username="user", password="pass")
    root = _mock_root()
    session._cdp_client_root = root
    focus = MagicMock()
    focus.cdp_client = root
    focus.session_id = "focus-1"
    object.__setattr__(session, "agent_focus", focus)

    await session._setup_proxy_auth()

    # The focused client is the root client in production, so one registration decides every request.
    root.register.Fetch.requestPaused.assert_called_once()
    assert _enabled_params(root, "focus-1")["handleAuthRequests"] is True
    root.register.Fetch.authRequired.assert_called_once()
    params = _enabled_params(root, None)
    assert params["handleAuthRequests"] is True
    assert params["patterns"] == [{"urlPattern": "*", "requestStage": "Request"}]

    handler = root.register.Fetch.requestPaused.call_args[0][0]
    handler({"requestId": "img-1", "resourceType": "Image"}, "session-1")
    handler({"requestId": "doc-1", "resourceType": "Document"}, "session-1")
    await _drain()

    root.send.Fetch.failRequest.assert_awaited_once()
    assert root.send.Fetch.failRequest.call_args.kwargs["params"]["requestId"] == "img-1"
    root.send.Fetch.continueRequest.assert_awaited_once()
    assert root.send.Fetch.continueRequest.call_args.kwargs["params"]["requestId"] == "doc-1"


@pytest.mark.asyncio
async def test_attached_target_receives_the_same_policy():
    session = _make_session("no-render")
    root = _mock_root()
    session._cdp_client_root = root

    await session._setup_proxy_auth()
    attached_handler = root.register.Target.attachedToTarget.call_args[0][0]
    attached_handler({"sessionId": "attached-1"}, None)
    await _drain()

    params = _enabled_params(root, "attached-1")
    assert [pattern["resourceType"] for pattern in params["patterns"]] == list(NO_RENDER_BLOCKED_RESOURCE_TYPES)


@pytest.mark.asyncio
async def test_unsupported_resource_filter_is_dropped_not_fatal():
    """Older Chromium rejects TextTrack as a Fetch filter; the rest must still apply."""
    session = _make_session("no-render")
    root = _mock_root()
    session._cdp_client_root = root
    accepted: list[list[str]] = []

    async def enable(params=None, session_id=None):
        resource_types = [pattern["resourceType"] for pattern in params["patterns"]]
        if "TextTrack" in resource_types:
            raise RuntimeError("{'code': -32602, 'message': \"Unknown resource type in fetch filter: 'TextTrack'\"}")
        accepted.append(resource_types)

    root.send.Fetch.enable = AsyncMock(side_effect=enable)

    await session._setup_proxy_auth()

    assert accepted
    for resource_types in accepted:
        assert "TextTrack" not in resource_types
        assert "Stylesheet" in resource_types
    # The handler still exists and blocks the supported types.
    handler = root.register.Fetch.requestPaused.call_args[0][0]
    handler({"requestId": "media-1", "resourceType": "Media"}, "session-1")
    await _drain()
    root.send.Fetch.failRequest.assert_awaited_once()
