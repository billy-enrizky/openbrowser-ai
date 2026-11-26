"""Headless blank-page keepalive is independent from decorative rendering."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openbrowser.browser.events import AboutBlankDVDScreensaverShownEvent, TabClosingEvent, TabCreatedEvent
from openbrowser.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog


def make_watchdog(headless):
    evaluate = AsyncMock(return_value={})
    temporary = SimpleNamespace(
        session_id="session",
        cdp_client=SimpleNamespace(send=SimpleNamespace(Runtime=SimpleNamespace(evaluate=evaluate))),
    )
    session = SimpleNamespace(
        id="browser-1234",
        browser_profile=SimpleNamespace(headless=headless),
        logger=logging.getLogger("headless-loading-test"),
        _cdp_get_all_pages=AsyncMock(return_value=[{"targetId": "blank", "url": "about:blank"}]),
        get_or_create_cdp_session=AsyncMock(return_value=temporary),
    )
    bus = MagicMock()
    watchdog = AboutBlankWatchdog.model_construct(browser_session=session, event_bus=bus)
    return watchdog, session, bus, evaluate


async def test_headless_tab_creation_avoids_cosmetic_target_queries_and_script():
    watchdog, session, bus, evaluate = make_watchdog(True)
    await watchdog.on_TabCreatedEvent(TabCreatedEvent(target_id="blank", url="about:blank"))
    session._cdp_get_all_pages.assert_not_awaited()
    session.get_or_create_cdp_session.assert_not_awaited()
    evaluate.assert_not_awaited()
    bus.dispatch.assert_not_called()


async def test_headless_direct_loading_hook_avoids_cdp_and_shown_event():
    watchdog, session, bus, evaluate = make_watchdog(True)
    await watchdog._show_dvd_screensaver_loading_animation_cdp("blank", "1234")
    session.get_or_create_cdp_session.assert_not_awaited()
    evaluate.assert_not_awaited()
    bus.dispatch.assert_not_called()


@pytest.mark.parametrize("headless", [False, None])
async def test_headed_or_unspecified_profile_preserves_loading_script_and_event(headless):
    watchdog, session, bus, evaluate = make_watchdog(headless)
    await watchdog._show_dvd_screensaver_on_about_blank_tabs()
    session._cdp_get_all_pages.assert_awaited_once()
    session.get_or_create_cdp_session.assert_awaited_once_with("blank", focus=False)
    evaluate.assert_awaited_once()
    assert evaluate.await_args is not None
    script = evaluate.await_args.kwargs["params"]["expression"]
    assert "requestAnimationFrame(animate)" in script
    assert "pretty-loading-animation" in script
    event = bus.dispatch.call_args.args[0]
    assert isinstance(event, AboutBlankDVDScreensaverShownEvent)
    assert event.target_id == "blank"


async def test_headless_last_tab_keeps_the_browser_alive(monkeypatch):
    watchdog, session, bus, evaluate = make_watchdog(True)
    create = AsyncMock()
    monkeypatch.setattr(watchdog, "_create_about_blank_tab", create)
    await watchdog.on_TabClosingEvent(TabClosingEvent(target_id="blank"))
    create.assert_awaited_once()
    session._cdp_get_all_pages.assert_awaited_once()
    evaluate.assert_not_awaited()
    bus.dispatch.assert_not_called()


async def test_stopping_headless_session_preserves_the_stop_guard(monkeypatch):
    watchdog, session, bus, evaluate = make_watchdog(True)
    watchdog._stopping = True
    create = AsyncMock()
    monkeypatch.setattr(watchdog, "_create_about_blank_tab", create)
    await watchdog.on_TabClosingEvent(TabClosingEvent(target_id="blank"))
    create.assert_not_awaited()
    session._cdp_get_all_pages.assert_not_awaited()
    evaluate.assert_not_awaited()
    bus.dispatch.assert_not_called()
