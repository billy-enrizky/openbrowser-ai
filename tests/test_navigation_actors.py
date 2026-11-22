"""Behavior tests for actor navigation readiness.

The session navigation machine owns focus, request correlation, policy, and
network-idle readiness. These tests cover the smaller target-scoped barrier
used by ``Page`` and ``_cdp_navigate``.
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.actor.page import Page
from openbrowser.browser.navigation import (
    NavigationError,
    NavigationTimeoutError,
)


def frame(loader_id: str | None, url: str) -> dict:
    return {
        "frameTree": {
            "frame": {
                "id": "frame-1",
                "loaderId": loader_id or "",
                "url": url,
            }
        }
    }


def _sequence(values):
    iterator = iter(values)
    last = values[-1]

    async def call(**kwargs):
        nonlocal last
        with contextlib.suppress(StopIteration):
            last = next(iterator)
        return last

    return call


def _doubled(values):
    expanded = []
    for value in values:
        expanded.extend([value, value])
    return _sequence(expanded)


def make_page(*, frames, states, navigate_response):
    client = MagicMock()
    client.send.Page.getFrameTree = AsyncMock(side_effect=_doubled(frames))
    client.send.Runtime.evaluate = AsyncMock(side_effect=_sequence([{"result": {"value": state}} for state in states]))
    client.send.Page.navigate = AsyncMock(return_value=navigate_response)
    client.register.Page = MagicMock()
    session = SimpleNamespace(cdp_client=client, invalidate_dom_cache=MagicMock())
    page = Page(session, target_id="tab-a", session_id="sess-a")
    return page, client, session


async def test_goto_waits_for_load():
    page, client, session = make_page(
        frames=[
            frame("loader-old", "about:blank"),
            frame("loader-new", "https://example.com"),
            frame("loader-new", "https://example.com"),
        ],
        states=["complete", "loading", "complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    snapshot = await page.goto("https://example.com")

    assert snapshot is not None
    assert snapshot.loader_id == "loader-new"
    assert snapshot.ready_state == "complete"
    assert client.send.Runtime.evaluate.await_count == 3
    session.invalidate_dom_cache.assert_called_once_with(target_id="tab-a", reason="page_actor_navigation")


async def test_goto_commit_returns_while_the_page_is_still_loading():
    page, client, _ = make_page(
        frames=[
            frame("loader-old", "about:blank"),
            frame("loader-new", "https://example.com"),
            frame("loader-new", "https://example.com"),
        ],
        states=["complete", "loading", "loading"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    snapshot = await page.goto("https://example.com", wait_until="commit")

    assert snapshot is not None
    assert snapshot.loader_id == "loader-new"
    assert snapshot.ready_state == "loading"
    assert client.send.Runtime.evaluate.await_count == 2


async def test_goto_raises_on_a_navigation_error_response():
    page, client, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"errorText": "net::ERR_NAME_NOT_RESOLVED"},
    )

    with pytest.raises(NavigationError, match="ERR_NAME_NOT_RESOLVED"):
        await page.goto("https://missing.test")

    assert client.send.Page.getFrameTree.await_count == 2


async def test_goto_download_returns_without_waiting_for_a_document():
    page, client, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"isDownload": True},
    )

    assert await page.goto("https://example.com/report.pdf") is None
    assert client.send.Runtime.evaluate.await_count == 1


async def test_goto_same_document_commits_on_a_url_change():
    page, _, _ = make_page(
        frames=[
            frame("loader-old", "https://example.com/list"),
            frame("loader-old", "https://example.com/details"),
        ],
        states=["complete", "complete"],
        navigate_response={"frameId": "frame-1"},
    )

    snapshot = await page.goto("https://example.com/details", wait_until="commit")

    assert snapshot is not None
    assert snapshot.url == "https://example.com/details"
    assert snapshot.loader_id == "loader-old"


async def test_goto_to_the_same_url_waits_for_the_new_loader():
    page, client, _ = make_page(
        frames=[
            frame("loader-old", "https://example.com"),
            frame("loader-old", "https://example.com"),
            frame("loader-new", "https://example.com"),
        ],
        states=["complete", "complete", "complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    snapshot = await page.goto("https://example.com")

    assert snapshot is not None
    assert snapshot.loader_id == "loader-new"
    assert client.send.Page.getFrameTree.await_count == 6


async def test_goto_redirect_commits_on_a_changed_loader():
    page, _, _ = make_page(
        frames=[
            frame("loader-old", "about:blank"),
            frame("loader-final", "https://example.com/final"),
            frame("loader-final", "https://example.com/final"),
        ],
        states=["complete", "loading", "complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-intermediate"},
    )

    snapshot = await page.goto("https://example.com/start")

    assert snapshot is not None
    assert snapshot.loader_id == "loader-final"
    assert snapshot.ready_state == "complete"


async def test_goto_timeout_reports_the_committed_phase():
    page, _, _ = make_page(
        frames=[
            frame("loader-old", "about:blank"),
            frame("loader-new", "https://example.com"),
        ],
        states=["complete", "loading"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    with pytest.raises(NavigationTimeoutError) as raised:
        await page.goto("https://example.com", timeout_seconds=0.03)

    message = str(raised.value)
    assert "did not reach load" in message
    assert "phase timed_out" in message
    assert "last observed committed" in message
    assert "loader-new" in message


async def test_goto_waits_for_domcontentloaded():
    page, _, _ = make_page(
        frames=[
            frame("loader-old", "about:blank"),
            frame("loader-new", "https://example.com"),
            frame("loader-new", "https://example.com"),
        ],
        states=["complete", "loading", "interactive"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    snapshot = await page.goto("https://example.com", wait_until="domcontentloaded")

    assert snapshot is not None
    assert snapshot.ready_state == "interactive"


async def test_reload_waits_for_load():
    page, client, _ = make_page(
        frames=[
            frame("loader-old", "https://example.com"),
            frame("loader-new", "https://example.com"),
            frame("loader-new", "https://example.com"),
        ],
        states=["complete", "loading", "complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )
    client.send.Page.reload = AsyncMock(return_value={})

    snapshot = await page.reload()

    assert snapshot is not None
    assert snapshot.ready_state == "complete"
    assert client.send.Runtime.evaluate.await_count == 3


async def test_history_navigation_waits_for_load():
    page, client, _ = make_page(
        frames=[
            frame("loader-old", "https://b.test/"),
            frame("loader-new", "https://a.test/"),
            frame("loader-new", "https://a.test/"),
        ],
        states=["complete", "loading", "complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )
    client.send.Page.getNavigationHistory = AsyncMock(
        return_value={
            "currentIndex": 1,
            "entries": [
                {"id": 1, "url": "https://a.test/"},
                {"id": 2, "url": "https://b.test/"},
            ],
        }
    )
    client.send.Page.navigateToHistoryEntry = AsyncMock(return_value={})

    snapshot = await page.go_back()

    assert snapshot is not None
    assert snapshot.url == "https://a.test/"
    assert client.send.Runtime.evaluate.await_count == 3


async def test_history_with_the_same_url_waits_for_the_new_loader():
    page, client, _ = make_page(
        frames=[
            frame("loader-old", "https://example.com/list"),
            frame("loader-old", "https://example.com/list"),
            frame("loader-new", "https://example.com/list"),
        ],
        states=["complete", "complete", "complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )
    client.send.Page.getNavigationHistory = AsyncMock(
        return_value={
            "currentIndex": 1,
            "entries": [
                {"id": 1, "url": "https://example.com/list"},
                {"id": 2, "url": "https://example.com/list"},
            ],
        }
    )
    client.send.Page.navigateToHistoryEntry = AsyncMock(return_value={})

    snapshot = await page.go_back()

    assert snapshot is not None
    assert snapshot.loader_id == "loader-new"
    assert client.send.Page.getFrameTree.await_count == 6


async def test_history_lookup_is_bounded_by_timeout_seconds():
    page, client, _ = make_page(
        frames=[frame("loader-old", "https://example.com/list")],
        states=["complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    async def hang(**kwargs):
        await asyncio.sleep(1)
        return {"currentIndex": 1, "entries": []}

    client.send.Page.getNavigationHistory = AsyncMock(side_effect=hang)

    with pytest.raises(NavigationTimeoutError, match="did not complete"):
        await page.go_back(timeout_seconds=0.05)


async def test_session_setup_is_bounded_by_timeout_seconds():
    page, _, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    async def hang():
        await asyncio.sleep(1)
        return "sess-a"

    page._ensure_session = hang

    with pytest.raises(NavigationTimeoutError, match="did not complete"):
        await page.goto("https://example.com", timeout_seconds=0.05)


async def test_goto_timeout_bounds_the_navigation_command():
    page, client, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    async def hang(params, session_id=None):
        await asyncio.sleep(1)
        return {"frameId": "frame-1", "loaderId": "loader-new"}

    client.send.Page.navigate = AsyncMock(side_effect=hang)

    with pytest.raises(NavigationTimeoutError, match="did not complete"):
        await page.goto("https://example.com", timeout_seconds=0.05)


async def test_goto_rejects_networkidle_before_issuing_the_command():
    page, client, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    with pytest.raises(NavigationError, match="networkidle"):
        await page.goto("https://example.com", wait_until="networkidle")

    client.send.Page.navigate.assert_not_called()


async def test_barrier_registers_no_page_events():
    page, client, _ = make_page(
        frames=[
            frame("loader-old", "about:blank"),
            frame("loader-new", "https://example.com"),
        ],
        states=["complete", "complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )

    await page.goto("https://example.com")

    assert client.register.Page.mock_calls == []


async def test_goto_fails_fast_when_the_cdp_session_is_gone():
    client = MagicMock()
    calls = 0

    async def get_frame_tree(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("-32001 Session with given id not found")

    client.send.Page.getFrameTree = AsyncMock(side_effect=get_frame_tree)
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": "complete"}})
    client.send.Page.navigate = AsyncMock(return_value={"frameId": "frame-1", "loaderId": "loader-new"})
    session = SimpleNamespace(cdp_client=client, invalidate_dom_cache=MagicMock())
    page = Page(session, target_id="tab-a", session_id="sess-a")

    with pytest.raises(NavigationError, match="lost its CDP session"):
        await page.goto("https://example.com", timeout_seconds=30)

    assert calls == 2


async def test_cdp_navigate_waits_for_the_new_loader():
    from openbrowser.browser.session import BrowserSession, CDPSession

    session = BrowserSession(cdp_url="ws://127.0.0.1:9222")
    client = MagicMock()
    client.send.Page.getFrameTree = AsyncMock(
        side_effect=_sequence(
            [
                frame("loader-old", "about:blank"),
                frame("loader-new", "https://example.com"),
            ]
        )
    )
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": "complete"}})
    client.send.Page.navigate = AsyncMock(return_value={"frameId": "frame-1", "loaderId": "loader-new"})
    client.register.Page = MagicMock()
    session._cdp_client_root = client
    focus = CDPSession.model_construct(
        cdp_client=client,
        target_id="tab-a",
        session_id="sess-a",
        url="about:blank",
        title="Tab A",
    )
    session.agent_focus = focus

    with patch.object(
        BrowserSession,
        "get_or_create_cdp_session",
        AsyncMock(return_value=focus),
    ):
        await session._cdp_navigate("https://example.com")

    assert client.send.Page.getFrameTree.await_count >= 2
    assert client.register.Page.mock_calls == []


async def test_cdp_navigate_propagates_a_navigation_error():
    from openbrowser.browser.session import BrowserSession, CDPSession

    session = BrowserSession(cdp_url="ws://127.0.0.1:9222")
    client = MagicMock()
    client.send.Page.navigate = AsyncMock(return_value={"errorText": "net::ERR_CONNECTION_REFUSED"})
    client.send.Page.getFrameTree = AsyncMock(return_value=frame("loader-old", "about:blank"))
    session._cdp_client_root = client
    focus = CDPSession.model_construct(
        cdp_client=client,
        target_id="tab-a",
        session_id="sess-a",
        url="about:blank",
        title="Tab A",
    )
    session.agent_focus = focus

    with (
        patch.object(
            BrowserSession,
            "get_or_create_cdp_session",
            AsyncMock(return_value=focus),
        ),
        pytest.raises(NavigationError, match="ERR_CONNECTION_REFUSED"),
    ):
        await session._cdp_navigate("https://refused.test")


async def test_cdp_navigate_rejects_networkidle_before_issuing_the_command():
    from openbrowser.browser.session import BrowserSession, CDPSession

    session = BrowserSession(cdp_url="ws://127.0.0.1:9222")
    client = MagicMock()
    client.send.Page.navigate = AsyncMock(return_value={"frameId": "frame-1", "loaderId": "loader-new"})
    session._cdp_client_root = client
    focus = CDPSession.model_construct(
        cdp_client=client,
        target_id="tab-a",
        session_id="sess-a",
        url="about:blank",
        title="Tab A",
    )
    session.agent_focus = focus

    with (
        patch.object(
            BrowserSession,
            "get_or_create_cdp_session",
            AsyncMock(return_value=focus),
        ),
        pytest.raises(NavigationError, match="networkidle"),
    ):
        await session._cdp_navigate("https://example.com", wait_until="networkidle")

    client.send.Page.navigate.assert_not_called()


async def test_goto_without_a_baseline_requires_the_new_loader():
    client = MagicMock()
    calls = 0

    async def get_frame_tree(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("baseline unavailable")
        if calls == 2:
            return frame("loader-old", "https://example.com")
        return frame("loader-new", "https://example.com")

    client.send.Page.getFrameTree = AsyncMock(side_effect=get_frame_tree)
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": "complete"}})
    client.send.Page.navigate = AsyncMock(return_value={"frameId": "frame-1", "loaderId": "loader-new"})
    session = SimpleNamespace(cdp_client=client, invalidate_dom_cache=MagicMock())
    page = Page(session, target_id="tab-a", session_id="sess-a")

    snapshot = await page.goto("https://example.com")

    assert snapshot is not None
    assert snapshot.loader_id == "loader-new"
    assert calls == 5


async def test_actor_barrier_does_not_change_focus():
    client = MagicMock()
    client.send.Page.getFrameTree = AsyncMock(
        side_effect=_sequence(
            [
                frame("loader-old", "about:blank"),
                frame("loader-new", "https://example.com"),
            ]
        )
    )
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": "complete"}})
    client.send.Page.navigate = AsyncMock(return_value={"frameId": "frame-1", "loaderId": "loader-new"})
    focus = object()
    session = SimpleNamespace(
        cdp_client=client,
        invalidate_dom_cache=MagicMock(),
        agent_focus=focus,
    )
    page = Page(session, target_id="tab-background", session_id="sess-background")

    await page.goto("https://example.com", wait_until="commit")

    assert session.agent_focus is focus


async def test_goto_classifies_a_download_before_error_text():
    page, _, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"isDownload": True, "errorText": "net::ERR_ABORTED"},
    )

    assert await page.goto("https://example.com/report.pdf") is None


async def test_cdp_navigate_classifies_a_download_before_error_text():
    from openbrowser.browser.session import BrowserSession, CDPSession

    session = BrowserSession(cdp_url="ws://127.0.0.1:9222")
    client = MagicMock()
    client.send.Page.getFrameTree = AsyncMock(return_value=frame("loader-old", "about:blank"))
    client.send.Page.navigate = AsyncMock(return_value={"isDownload": True, "errorText": "net::ERR_ABORTED"})
    session._cdp_client_root = client
    focus = CDPSession.model_construct(
        cdp_client=client,
        target_id="tab-a",
        session_id="sess-a",
        url="about:blank",
        title="Tab A",
    )
    session.agent_focus = focus

    with patch.object(
        BrowserSession,
        "get_or_create_cdp_session",
        AsyncMock(return_value=focus),
    ):
        await session._cdp_navigate("https://example.com/report.pdf")


async def test_reload_without_a_baseline_fails_before_the_command():
    page, client, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )
    client.send.Page.getFrameTree = AsyncMock(side_effect=RuntimeError("no frame"))
    client.send.Page.reload = AsyncMock(return_value={})

    with pytest.raises(NavigationError, match="Cannot establish the current document before reload"):
        await page.reload()

    client.send.Page.reload.assert_not_called()


async def test_history_without_a_baseline_fails_before_the_command():
    page, client, _ = make_page(
        frames=[frame("loader-old", "about:blank")],
        states=["complete"],
        navigate_response={"frameId": "frame-1", "loaderId": "loader-new"},
    )
    client.send.Page.getNavigationHistory = AsyncMock(
        return_value={
            "currentIndex": 1,
            "entries": [
                {"id": 1, "url": "https://a.test/"},
                {"id": 2, "url": "https://b.test/"},
            ],
        }
    )
    client.send.Page.getFrameTree = AsyncMock(side_effect=RuntimeError("no frame"))
    client.send.Page.navigateToHistoryEntry = AsyncMock(return_value={})

    with pytest.raises(NavigationError, match="Cannot establish the current document before history"):
        await page.go_back()

    client.send.Page.navigateToHistoryEntry.assert_not_called()


async def test_goto_fails_fast_when_the_ready_state_read_loses_the_session():
    client = MagicMock()
    client.send.Page.getFrameTree = AsyncMock(return_value=frame("loader-new", "https://example.com"))
    client.send.Runtime.evaluate = AsyncMock(side_effect=RuntimeError("-32001 Session with given id not found"))
    client.send.Page.navigate = AsyncMock(return_value={"frameId": "frame-1", "loaderId": "loader-new"})
    session = SimpleNamespace(cdp_client=client, invalidate_dom_cache=MagicMock())
    page = Page(session, target_id="tab-a", session_id="sess-a")

    with pytest.raises(NavigationError, match="lost its CDP session"):
        await page.goto("https://example.com", wait_until="commit")


async def test_snapshot_discards_a_read_that_straddles_a_document_replacement():
    from openbrowser.browser.navigation import read_document_snapshot

    client = MagicMock()
    client.send.Page.getFrameTree = AsyncMock(
        side_effect=_sequence(
            [
                frame("loader-old", "https://example.com/old"),
                frame("loader-new", "https://example.com/new"),
            ]
        )
    )
    client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": "complete"}})

    snapshot = await read_document_snapshot(client, "sess-a")

    assert snapshot.loader_id == "loader-new"
    assert snapshot.url == "https://example.com/new"
    assert snapshot.ready_state is None
