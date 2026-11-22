"""Behavior tests for the DOM cache document fence.

These tests cover the contract in
``local_docs/specs/2026-11-22-dom-cache-lifecycle/design.md``:

* a cache entry belongs to one document view of one target;
* a document change makes every older entry unreadable;
* a second watchdog copy cannot resurrect a cleared entry;
* a read after an action revalidates against the live document identity.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser.session import BrowserSession, CDPSession
from openbrowser.tools.action_readiness import (
    ActionReadinessCode,
    ActionReadinessError,
    require_element_index,
)


def _make_client():
    from test_session_part1 import _make_mock_cdp_client

    return _make_mock_cdp_client()


def make_session(
    *,
    target_id: str = "tab-a",
    session_id: str = "sess-a",
    loader_id: str = "loader-a",
    url: str = "https://a.test/",
):
    """Build a real session with a fake CDP root and a controllable frame."""
    session = BrowserSession(cdp_url="ws://127.0.0.1:9222")
    client = _make_client()
    session._cdp_client_root = client
    focus = CDPSession.model_construct(
        cdp_client=client,
        target_id=target_id,
        session_id=session_id,
        url=url,
        title="Tab A",
    )
    session.agent_focus = focus
    frame = {"id": "frame-a", "loaderId": loader_id, "url": url}
    client.send.Page.getFrameTree = AsyncMock(side_effect=lambda **kwargs: {"frameTree": {"frame": dict(frame)}})
    return SimpleNamespace(session=session, client=client, focus=focus, frame=frame)


def add_background_tab(harness, *, target_id: str = "tab-b", session_id: str = "sess-b", url: str = "https://b.test/"):
    other = CDPSession.model_construct(
        cdp_client=harness.client,
        target_id=target_id,
        session_id=session_id,
        url=url,
        title="Tab B",
    )
    harness.session._cdp_session_pool[target_id] = other
    return other


def publish(harness, selector_map=None, *, loader_id: str | None = None):
    selector_map = selector_map if selector_map is not None else {1: object()}
    published = harness.session.update_cached_selector_map(selector_map, loader_id=loader_id)
    assert published is True
    return selector_map


def commit(harness, *, session_id: str | None = None, loader_id: str = "loader-b", url: str = "https://b.test/"):
    if session_id is None or session_id == harness.focus.session_id:
        harness.frame.update(loaderId=loader_id, url=url)
    harness.session.observe_document_event(
        "frame",
        {"frame": {"id": "frame-a", "loaderId": loader_id, "url": url}},
        session_id if session_id is not None else harness.focus.session_id,
    )


# ---------------------------------------------------------------------------
# Publication and status
# ---------------------------------------------------------------------------


def test_publish_reports_ready_without_page_content():
    harness = make_session()
    publish(harness)

    snapshot = harness.session.dom_cache_snapshot()

    assert snapshot["phase"] == "ready"
    assert snapshot["target_id"] == "tab-a"
    assert snapshot["selector_count"] == 1
    assert snapshot["invalidation_reason"] is None
    assert "a.test" not in json.dumps(snapshot)


def test_status_has_no_screenshot_or_selector_payload():
    harness = make_session()
    publish(harness, {1: MagicMock(name="node")})
    snapshot = harness.session.dom_cache_snapshot()

    assert set(snapshot) == {
        "phase",
        "global_generation",
        "target_generation",
        "target_id",
        "selector_count",
        "summary_cached",
        "provisional",
        "invalidation_reason",
        "invalidation_count",
        "invalidated_at",
    }


def test_lifecycle_snapshot_carries_dom_cache_without_page_content():
    harness = make_session()
    publish(harness)

    snapshot = harness.session.lifecycle_snapshot()

    assert snapshot["dom_cache"]["phase"] == "ready"
    assert snapshot["dom_cache"]["selector_count"] == 1
    assert "a.test" not in json.dumps(snapshot["dom_cache"])


# ---------------------------------------------------------------------------
# Document commit and the watchdog fallback
# ---------------------------------------------------------------------------


async def test_document_commit_invalidates_selector_map_and_summary():
    harness = make_session()
    publish(harness)
    harness.session.publish_browser_state_summary(
        SimpleNamespace(dom_state=SimpleNamespace(selector_map={1: object()}))
    )

    commit(harness)

    assert await harness.session.get_selector_map() == {}
    assert harness.session._cached_browser_state_summary is None
    snapshot = harness.session.dom_cache_snapshot()
    assert snapshot["phase"] == "invalidated"
    assert snapshot["invalidation_reason"] == "document_commit"


async def test_get_selector_map_never_reads_the_watchdog_copy():
    harness = make_session()
    publish(harness)
    stale = object()
    watchdog = SimpleNamespace(selector_map={1: stale}, clear_cache=MagicMock())
    harness.session._dom_watchdog = watchdog

    harness.session.invalidate_dom_cache(reason="document_commit")

    assert await harness.session.get_selector_map() == {}
    assert harness.session._cached_selector_map == {}


async def test_background_tab_commit_preserves_the_focused_entry():
    harness = make_session()
    add_background_tab(harness)
    cached = publish(harness)

    commit(harness, session_id="sess-b", loader_id="loader-other", url="https://other.test/")

    assert await harness.session.get_selector_map() is cached
    assert harness.session.dom_cache_snapshot()["phase"] == "ready"


async def test_within_document_navigation_invalidates_the_main_frame():
    harness = make_session()
    publish(harness)
    commit(harness, loader_id="loader-a", url="https://a.test/")
    publish(harness, loader_id="loader-a")

    harness.session.observe_document_event(
        "within",
        {"frameId": "frame-a", "url": "https://a.test/#route"},
        harness.focus.session_id,
    )

    assert await harness.session.get_selector_map() == {}
    assert harness.session.dom_cache_snapshot()["invalidation_reason"] == "same_document_navigation"


async def test_within_document_navigation_in_a_child_frame_is_ignored():
    harness = make_session()
    publish(harness)
    commit(harness, loader_id="loader-a", url="https://a.test/")
    cached = publish(harness, loader_id="loader-a")

    harness.session.observe_document_event(
        "within",
        {"frameId": "iframe-child", "url": "https://ads.test/frame"},
        harness.focus.session_id,
    )

    assert await harness.session.get_selector_map() is cached


# ---------------------------------------------------------------------------
# Read-time revalidation
# ---------------------------------------------------------------------------


async def test_provisional_read_keeps_an_unchanged_entry():
    harness = make_session()
    cached = publish(harness, loader_id="loader-a")
    harness.session.mark_dom_cache_provisional(reason="click")

    assert await harness.session.get_selector_map() is cached
    assert harness.session.dom_cache_snapshot()["phase"] == "ready"
    assert harness.client.send.Page.getFrameTree.await_count == 1


async def test_provisional_read_invalidates_on_a_new_loader():
    harness = make_session()
    publish(harness, loader_id="loader-a")
    harness.session.mark_dom_cache_provisional(reason="click")
    harness.frame.update(loaderId="loader-b", url="https://b.test/")

    assert await harness.session.get_selector_map() == {}
    assert harness.session.dom_cache_snapshot()["phase"] == "invalidated"


async def test_provisional_read_invalidates_on_a_url_only_change():
    harness = make_session()
    commit(harness, loader_id="loader-a", url="https://a.test/")
    publish(harness, loader_id="loader-a")
    harness.session.mark_dom_cache_provisional(reason="click")
    harness.frame.update(url="https://a.test/next-route")

    assert await harness.session.get_selector_map() == {}
    assert harness.session.dom_cache_snapshot()["invalidation_reason"] == "document_changed"


async def test_revalidation_failure_is_treated_as_invalid():
    harness = make_session()
    publish(harness, loader_id="loader-a")
    harness.session.mark_dom_cache_provisional(reason="click")
    harness.client.send.Page.getFrameTree = AsyncMock(side_effect=RuntimeError("session gone"))

    assert await harness.session.get_selector_map() == {}
    assert harness.session.dom_cache_snapshot()["phase"] == "invalidated"


async def test_stale_index_is_reported_after_a_new_document_publishes():
    harness = make_session()
    publish(harness, {1: object()}, loader_id="loader-a")
    commit(harness)
    assert await harness.session.get_selector_map() == {}

    publish(harness, {2: object()}, loader_id="loader-b")
    selector_map = await harness.session.get_selector_map()

    with pytest.raises(ActionReadinessError) as raised:
        require_element_index(1, selector_map=selector_map, action="click")
    assert raised.value.code is ActionReadinessCode.STALE


# ---------------------------------------------------------------------------
# Low-level navigation actors
# ---------------------------------------------------------------------------


async def test_cdp_navigate_invalidates_before_the_command():
    harness = make_session()
    publish(harness, loader_id="loader-a")
    seen = {}

    async def navigate(params, session_id=None):
        seen["map"] = dict(harness.session._cached_selector_map)
        return {}

    harness.client.send.Page.navigate = AsyncMock(side_effect=navigate)

    with patch.object(
        BrowserSession,
        "get_or_create_cdp_session",
        AsyncMock(return_value=harness.focus),
    ):
        await harness.session._cdp_navigate("https://b.test/")

    assert seen["map"] == {}
    assert await harness.session.get_selector_map() == {}
    assert harness.session.dom_cache_snapshot()["invalidation_reason"] == "cdp_navigate"


async def test_page_reload_invalidates_before_the_command():
    from openbrowser.actor.page import Page

    harness = make_session()
    publish(harness, loader_id="loader-a")
    seen = {}

    async def reload(session_id=None):
        seen["map"] = dict(harness.session._cached_selector_map)
        return {}

    harness.client.send.Page.reload = AsyncMock(side_effect=reload)
    page = Page(harness.session, target_id="tab-a", session_id="sess-a")

    await page.reload()

    assert seen["map"] == {}
    assert await harness.session.get_selector_map() == {}
    assert harness.session.dom_cache_snapshot()["invalidation_reason"] == "page_actor_navigation"


async def test_page_goto_invalidates_before_the_command():
    from openbrowser.actor.page import Page

    harness = make_session()
    publish(harness, loader_id="loader-a")
    seen = {}

    async def navigate(params, session_id=None):
        seen["map"] = dict(harness.session._cached_selector_map)
        return {}

    harness.client.send.Page.navigate = AsyncMock(side_effect=navigate)
    page = Page(harness.session, target_id="tab-a", session_id="sess-a")

    await page.goto("https://b.test/")

    assert seen["map"] == {}
    assert await harness.session.get_selector_map() == {}
    assert harness.session.dom_cache_snapshot()["invalidation_reason"] == "page_actor_navigation"


# ---------------------------------------------------------------------------
# Raced builds and lifecycle clearing
# ---------------------------------------------------------------------------


def test_publish_from_an_old_epoch_is_refused():
    harness = make_session()
    captured = harness.session.dom_cache_epoch()
    harness.session.invalidate_dom_cache(reason="document_commit")

    published = harness.session.update_cached_selector_map({1: object()}, epoch=captured)

    assert published is False
    assert harness.session._cached_selector_map == {}


async def test_dom_watchdog_discards_a_build_that_raced_a_commit():
    from bubus import EventBus

    from openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog

    harness = make_session()
    watchdog = DOMWatchdog(event_bus=EventBus(), browser_session=harness.session)

    async def build(previous_cached_state=None):
        harness.session.invalidate_dom_cache(reason="document_commit")
        state = MagicMock()
        state.selector_map = {7: object()}
        return state, MagicMock(), {}

    watchdog._dom_service = SimpleNamespace(get_serialized_dom_tree=build)

    result = await watchdog._build_dom_tree_without_highlights()

    assert result.selector_map == {}
    assert watchdog.selector_map is None
    assert harness.session._cached_selector_map == {}


async def test_reset_clears_the_dom_cache():
    harness = make_session()
    publish(harness)
    harness.session.publish_browser_state_summary(
        SimpleNamespace(dom_state=SimpleNamespace(selector_map={1: object()}))
    )

    await harness.session.reset(stop_event_bus=False)

    assert await harness.session.get_selector_map() == {}
    assert harness.session._cached_browser_state_summary is None
    assert harness.session.dom_cache_snapshot()["phase"] == "empty"


# ---------------------------------------------------------------------------
# Navigation failure integration
# ---------------------------------------------------------------------------


async def test_failed_navigation_leaves_the_cache_invalid():
    from test_navigation import handler_fixture

    from openbrowser.browser.events import NavigateToUrlEvent
    from openbrowser.browser.navigation import NavigationError

    session, client, _focus, _frame, patches = handler_fixture()
    session.update_cached_selector_map({1: object()}, loader_id="old")
    client.send.Page.navigate = AsyncMock(return_value={"frameId": "frame", "errorText": "net::ERR_CONNECTION_REFUSED"})

    with patches[0], patches[1], pytest.raises(NavigationError, match="ERR_CONNECTION_REFUSED"):
        await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url="https://example.com/"))

    assert await session.get_selector_map() == {}
    assert session.dom_cache_snapshot()["phase"] == "invalidated"


async def test_successful_navigation_invalidates_at_request_and_at_commit():
    from test_navigation import handler_fixture

    from openbrowser.browser.events import NavigateToUrlEvent

    session, client, focus, frame, patches = handler_fixture()
    cached = {1: object()}
    session.update_cached_selector_map(cached, loader_id="old")
    seen_at_command = {}

    async def navigate(**kwargs):
        seen_at_command["map"] = dict(session._cached_selector_map)
        frame.update(loaderId="next", url=kwargs["params"]["url"])
        session.observe_document_event(
            "frame",
            {"frame": {"id": "frame", "loaderId": "next", "url": kwargs["params"]["url"]}},
            focus.session_id,
        )
        client.register.Page.lifecycleEvent.call_args.args[0](
            {"frameId": "frame", "loaderId": "next", "name": "load"}, focus.session_id
        )
        return {"frameId": "frame", "loaderId": "next"}

    client.send.Page.navigate = AsyncMock(side_effect=navigate)

    with patches[0], patches[1]:
        await session.on_NavigateToUrlEvent(NavigateToUrlEvent(url="https://example.com/"))

    assert seen_at_command["map"] == {}
    assert await session.get_selector_map() == {}


# ---------------------------------------------------------------------------
# Review follow-ups: unconditional validation and concurrency
# ---------------------------------------------------------------------------


async def test_ready_entry_is_revalidated_on_every_read():
    harness = make_session()
    publish(harness, loader_id="loader-a")
    harness.frame.update(loaderId="loader-b", url="https://b.test/")

    assert await harness.session.get_selector_map() == {}
    assert harness.client.send.Page.getFrameTree.await_count == 1
    assert harness.session.dom_cache_snapshot()["invalidation_reason"] == "document_changed"


async def test_cached_summary_is_revalidated_before_use():
    harness = make_session()
    publish(harness, loader_id="loader-a")
    summary = SimpleNamespace(dom_state=SimpleNamespace(selector_map={1: object()}), screenshot="x")
    assert harness.session.publish_browser_state_summary(summary) is True

    harness.frame.update(loaderId="loader-b", url="https://b.test/")
    fresh = SimpleNamespace(dom_state=SimpleNamespace(selector_map={2: object()}))
    event = MagicMock()
    event.event_result = AsyncMock(return_value=fresh)
    harness.session.event_bus.dispatch = MagicMock(return_value=event)

    result = await harness.session.get_browser_state_summary(cached=True)

    assert result is fresh
    assert harness.session._cached_browser_state_summary is None


def test_an_older_build_cannot_overwrite_a_newer_publish():
    harness = make_session()
    first = harness.session.dom_cache_build_token()
    second = harness.session.dom_cache_build_token()
    epoch = harness.session.dom_cache_epoch()

    assert harness.session.update_cached_selector_map({2: object()}, epoch=epoch, sequence=second) is True
    assert harness.session.update_cached_selector_map({1: object()}, epoch=epoch, sequence=first) is False
    assert set(harness.session._cached_selector_map) == {2}


async def test_forgetting_a_target_clears_its_storage():
    harness = make_session()
    publish(harness)

    harness.session.forget_dom_cache("tab-a")

    assert harness.session._cached_selector_map == {}
    assert await harness.session.get_selector_map() == {}


def test_element_action_marks_its_own_target():
    from openbrowser.actor.element import Element

    fake_session = MagicMock()
    fake_session.cdp_client = MagicMock()
    element = Element(fake_session, 1, "sess-b", target_id="tab-b")

    element._mark_dom_provisional("element_click")

    fake_session.mark_dom_cache_provisional.assert_called_once_with(target_id="tab-b", reason="element_click")


async def test_tool_action_marks_provisional_before_execution():
    from openbrowser.models import ActionResult
    from openbrowser.tools.service import CodeAgentTools
    from openbrowser.tools.views import DoneAction

    tools = CodeAgentTools()
    session = MagicMock()
    order: list[str] = []
    session.mark_dom_cache_provisional.side_effect = lambda **kwargs: order.append("mark")

    async def execute(**kwargs):
        order.append("action")
        return ActionResult(extracted_content="done")

    with patch.object(tools.registry, "execute_action", AsyncMock(side_effect=execute)):
        result = await tools.act(DoneAction(text="ok"), session)

    assert order == ["mark", "action"]
    assert result.extracted_content == "done"


async def test_page_evaluate_marks_provisional_before_the_command():
    from openbrowser.actor.page import Page

    harness = make_session()
    publish(harness, loader_id="loader-a")
    seen = {}

    async def evaluate(params, session_id=None):
        seen["phase"] = harness.session.dom_cache_snapshot()["phase"]
        return {"result": {"value": "ok"}}

    harness.client.send.Runtime.evaluate = AsyncMock(side_effect=evaluate)
    page = Page(harness.session, target_id="tab-a", session_id="sess-a")

    await page.evaluate("() => 1")

    assert seen["phase"] == "provisional"


@pytest.mark.parametrize("direction", ["back", "forward"])
async def test_page_history_navigation_invalidates_before_the_command(direction):
    from openbrowser.actor.page import Page

    harness = make_session()
    publish(harness, loader_id="loader-a")
    current_index = 1 if direction == "back" else 0
    harness.client.send.Page.getNavigationHistory = AsyncMock(
        return_value={"currentIndex": current_index, "entries": [{"id": 1}, {"id": 2}]}
    )
    seen = {}

    async def navigate_to_history(params, session_id=None):
        seen["map"] = dict(harness.session._cached_selector_map)
        return {}

    harness.client.send.Page.navigateToHistoryEntry = AsyncMock(side_effect=navigate_to_history)
    page = Page(harness.session, target_id="tab-a", session_id="sess-a")

    await (page.go_back() if direction == "back" else page.go_forward())

    assert seen["map"] == {}
