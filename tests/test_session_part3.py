"""Tests for BrowserSession methods in lines ~2000-3131 of session.py.

Covers: highlight_interaction_element, add_highlights, _close_extension_options_pages,
downloaded_files, _cdp_get_all_pages, _cdp_create_new_page, _cdp_close_page,
_cdp_get_cookies, _cdp_set_cookies, _cdp_clear_cookies, _cdp_set_extra_headers,
_cdp_grant_permissions, _cdp_set_geolocation, _cdp_clear_geolocation,
_cdp_add_init_script, _cdp_remove_init_script, _cdp_set_viewport,
_cdp_get_origins, _cdp_get_storage_state, _cdp_navigate,
_is_valid_target, get_all_frames, _populate_frame_metadata,
find_frame_target, cdp_client_for_target, get_target_id_from_session_id,
cdp_client_for_frame, cdp_client_for_node,
take_screenshot, screenshot_element, _get_element_bounds.
"""

import asyncio
import base64
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.browser.session import BrowserSession, CDPSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers to construct a BrowserSession with all internals mocked
# ---------------------------------------------------------------------------


def _make_cdp_session(target_id="target-1", session_id="session-1", url="https://example.com", title="Example"):
    """Create a mock CDPSession-like object."""
    cdp_sess = MagicMock(spec=CDPSession)
    cdp_sess.target_id = target_id
    cdp_sess.session_id = session_id
    cdp_sess.url = url
    cdp_sess.title = title
    # CDP client mock with async send methods
    cdp_client = MagicMock()
    cdp_sess.cdp_client = cdp_client
    return cdp_sess


def _make_browser_session(**overrides):
    """Construct a BrowserSession with event_bus and internals mocked out.

    Uses patch.object on the class to bypass model_post_init handler registration.
    Separates BrowserProfile kwargs from BrowserSession constructor kwargs.
    """
    with patch.object(BrowserSession, "model_post_init", lambda self, ctx: None):
        session = BrowserSession(**overrides)

    # Patch EventBus.dispatch to be a MagicMock (not AsyncMock) on the real EventBus instance
    session.event_bus.dispatch = MagicMock()

    # Set up mocked private attrs
    root_client = MagicMock()
    session._cdp_client_root = root_client  # type: ignore[attr-defined]
    session._cdp_session_pool = {}  # type: ignore[attr-defined]
    session._downloaded_files = []  # type: ignore[attr-defined]

    # Mock agent_focus
    agent_focus = _make_cdp_session()
    session.agent_focus = agent_focus

    # Mock session_manager
    session._session_manager = MagicMock()  # type: ignore[attr-defined]
    session._session_manager.get_session_for_target = AsyncMock(return_value=agent_focus)
    session._session_manager.validate_session = AsyncMock(return_value=True)

    return session


def _make_mock_node(**kwargs):
    """Create a mock EnhancedDOMTreeNode."""
    node = MagicMock()
    node.backend_node_id = kwargs.get("backend_node_id", 42)
    node.node_name = kwargs.get("node_name", "div")
    node.node_id = kwargs.get("node_id", 1)
    node.node_value = kwargs.get("node_value", "")
    node.attributes = kwargs.get("attributes", {})
    node.xpath = kwargs.get("xpath", "/html/body/div")
    node.session_id = kwargs.get("session_id", None)
    node.frame_id = kwargs.get("frame_id", None)
    node.target_id = kwargs.get("target_id", None)
    node.is_scrollable = kwargs.get("is_scrollable", False)
    node.absolute_position = kwargs.get("absolute_position", None)
    node.snapshot_node = kwargs.get("snapshot_node", None)
    node.get_all_children_text = MagicMock(return_value=kwargs.get("text", "hello"))
    return node


# ===========================================================================
# highlight_interaction_element
# ===========================================================================


@pytest.mark.asyncio
class TestHighlightInteractionElement:

    async def test_returns_early_when_highlight_disabled(self):
        session = _make_browser_session(highlight_elements=False)
        node = _make_mock_node()
        # Should not raise, should return early
        await session.highlight_interaction_element(node)

    async def test_returns_early_when_no_rect(self):
        session = _make_browser_session(highlight_elements=True)
        node = _make_mock_node()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock) as mock_get:
            mock_cdp = _make_cdp_session()
            mock_get.return_value = mock_cdp
            with patch.object(BrowserSession, "get_element_coordinates", new_callable=AsyncMock) as mock_coords:
                mock_coords.return_value = None
                await session.highlight_interaction_element(node)
                # Runtime.evaluate should NOT be called since there's no rect
                mock_cdp.cdp_client.send.Runtime.evaluate.assert_not_called()

    async def test_highlight_creates_script_when_rect_found(self):
        session = _make_browser_session(highlight_elements=True)
        node = _make_mock_node()

        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": {"created": True}}})

        from openbrowser.dom.views import DOMRect

        rect = DOMRect(x=10, y=20, width=100, height=50)

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            with patch.object(BrowserSession, "get_element_coordinates", new_callable=AsyncMock, return_value=rect):
                await session.highlight_interaction_element(node)
                mock_cdp.cdp_client.send.Runtime.evaluate.assert_called_once()

    async def test_highlight_catches_exception(self):
        session = _make_browser_session(highlight_elements=True)
        node = _make_mock_node()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            # Should not raise
            await session.highlight_interaction_element(node)


# ===========================================================================
# add_highlights
# ===========================================================================


@pytest.mark.asyncio
class TestAddHighlights:

    async def test_returns_early_when_disabled(self):
        session = _make_browser_session(dom_highlight_elements=False)
        selector_map = {1: _make_mock_node()}
        await session.add_highlights(selector_map)

    async def test_returns_early_when_empty_selector_map(self):
        session = _make_browser_session(dom_highlight_elements=True)
        await session.add_highlights({})

    async def test_returns_early_when_no_valid_elements(self):
        session = _make_browser_session(dom_highlight_elements=True)
        # Node with no absolute_position
        node = _make_mock_node(absolute_position=None)
        await session.add_highlights({1: node})

    async def test_adds_highlights_with_valid_elements(self):
        session = _make_browser_session(dom_highlight_elements=True)

        from openbrowser.dom.views import DOMRect

        rect = DOMRect(x=10, y=20, width=100, height=50)
        node = _make_mock_node(absolute_position=rect)
        node.snapshot_node = MagicMock()
        node.snapshot_node.is_clickable = True

        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": {"added": 1}}}
        )

        with patch.object(BrowserSession, "remove_highlights", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
                    await session.add_highlights({1: node})
                    mock_cdp.cdp_client.send.Runtime.evaluate.assert_called_once()

    async def test_add_highlights_logs_when_result_has_no_value(self):
        session = _make_browser_session(dom_highlight_elements=True)

        from openbrowser.dom.views import DOMRect

        rect = DOMRect(x=10, y=20, width=100, height=50)
        node = _make_mock_node(absolute_position=rect)
        node.snapshot_node = MagicMock()
        node.snapshot_node.is_clickable = True

        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {}})

        with patch.object(BrowserSession, "remove_highlights", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
                    await session.add_highlights({1: node})

    async def test_add_highlights_catches_exception(self):
        session = _make_browser_session(dom_highlight_elements=True)

        from openbrowser.dom.views import DOMRect

        rect = DOMRect(x=10, y=20, width=100, height=50)
        node = _make_mock_node(absolute_position=rect)
        node.snapshot_node = MagicMock()
        node.snapshot_node.is_clickable = True

        with patch.object(BrowserSession, "remove_highlights", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            # Should not raise
            await session.add_highlights({1: node})


# ===========================================================================
# _close_extension_options_pages
# ===========================================================================


@pytest.mark.asyncio
class TestCloseExtensionOptionsPages:

    async def test_closes_extension_options_page(self):
        session = _make_browser_session()
        targets = [
            {"url": "chrome-extension://abc/options.html", "targetId": "t1"},
            {"url": "https://example.com", "targetId": "t2"},
        ]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "_cdp_close_page", new_callable=AsyncMock) as mock_close:
                await session._close_extension_options_pages()
                mock_close.assert_called_once_with("t1")

    async def test_closes_welcome_page(self):
        session = _make_browser_session()
        targets = [{"url": "chrome-extension://xyz/welcome.html", "targetId": "t3"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "_cdp_close_page", new_callable=AsyncMock) as mock_close:
                await session._close_extension_options_pages()
                mock_close.assert_called_once_with("t3")

    async def test_closes_onboarding_page(self):
        session = _make_browser_session()
        targets = [{"url": "chrome-extension://xyz/onboarding.html", "targetId": "t4"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "_cdp_close_page", new_callable=AsyncMock) as mock_close:
                await session._close_extension_options_pages()
                mock_close.assert_called_once_with("t4")

    async def test_handles_close_failure_gracefully(self):
        session = _make_browser_session()
        targets = [{"url": "chrome-extension://abc/options.html", "targetId": "t1"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "_cdp_close_page", new_callable=AsyncMock, side_effect=RuntimeError("close failed")):
                # Should not raise
                await session._close_extension_options_pages()

    async def test_handles_get_pages_failure_gracefully(self):
        session = _make_browser_session()

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            # Should not raise
            await session._close_extension_options_pages()

    async def test_skips_non_extension_pages(self):
        session = _make_browser_session()
        targets = [
            {"url": "https://example.com", "targetId": "t2"},
            {"url": "about:blank", "targetId": "t3"},
        ]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "_cdp_close_page", new_callable=AsyncMock) as mock_close:
                await session._close_extension_options_pages()
                mock_close.assert_not_called()


# ===========================================================================
# downloaded_files
# ===========================================================================


class TestDownloadedFiles:

    def test_returns_copy_of_list(self):
        session = _make_browser_session()
        session._downloaded_files = ["/tmp/file1.pdf", "/tmp/file2.zip"]  # type: ignore[attr-defined]
        result = session.downloaded_files
        assert result == ["/tmp/file1.pdf", "/tmp/file2.zip"]
        # Must be a copy, not the same list
        assert result is not session._downloaded_files

    def test_returns_empty_list_when_no_downloads(self):
        session = _make_browser_session()
        assert session.downloaded_files == []


# ===========================================================================
# _cdp_get_all_pages
# ===========================================================================


@pytest.mark.asyncio
class TestCdpGetAllPages:

    async def test_returns_empty_when_no_root_client(self):
        session = _make_browser_session()
        session._cdp_client_root = None  # type: ignore[attr-defined]
        result = await session._cdp_get_all_pages()
        assert result == []

    async def test_filters_valid_targets(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(  # type: ignore[union-attr]
            return_value={
                "targetInfos": [
                    {"type": "page", "url": "https://example.com", "targetId": "t1"},
                    {"type": "service_worker", "url": "https://example.com/sw.js", "targetId": "t2"},
                ]
            }
        )
        result = await session._cdp_get_all_pages()
        # Only the page target should be included (workers excluded by default)
        assert len(result) == 1
        assert result[0]["targetId"] == "t1"


# ===========================================================================
# _cdp_create_new_page
# ===========================================================================


@pytest.mark.asyncio
class TestCdpCreateNewPage:

    async def test_creates_page_with_root_client(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.createTarget = AsyncMock(  # type: ignore[union-attr]
            return_value={"targetId": "new-target"}
        )
        result = await session._cdp_create_new_page(url="https://test.com")
        assert result == "new-target"
        session._cdp_client_root.send.Target.createTarget.assert_called_once_with(  # type: ignore[union-attr]
            params={"url": "https://test.com", "newWindow": False, "background": False}
        )

    async def test_creates_page_without_root_client(self):
        session = _make_browser_session()
        session._cdp_client_root = None  # type: ignore[attr-defined]
        # When root is None, it should fall back to cdp_client property
        # But cdp_client asserts _cdp_client_root is not None, so we mock it
        mock_cdp = MagicMock()
        mock_cdp.send.Target.createTarget = AsyncMock(return_value={"targetId": "fallback-target"})

        with patch.object(BrowserSession, "cdp_client", new_callable=lambda: property(lambda self: mock_cdp)):
            result = await session._cdp_create_new_page()
            assert result == "fallback-target"

    async def test_creates_page_with_background_and_new_window(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.createTarget = AsyncMock(  # type: ignore[union-attr]
            return_value={"targetId": "bg-target"}
        )
        result = await session._cdp_create_new_page(url="about:blank", background=True, new_window=True)
        assert result == "bg-target"
        session._cdp_client_root.send.Target.createTarget.assert_called_once_with(  # type: ignore[union-attr]
            params={"url": "about:blank", "newWindow": True, "background": True}
        )


# ===========================================================================
# _cdp_close_page
# ===========================================================================


@pytest.mark.asyncio
class TestCdpClosePage:

    async def test_closes_page(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.closeTarget = AsyncMock()  # type: ignore[union-attr]
        await session._cdp_close_page("target-to-close")
        session._cdp_client_root.send.Target.closeTarget.assert_called_once_with(  # type: ignore[union-attr]
            params={"targetId": "target-to-close"}
        )


# ===========================================================================
# _cdp_get_cookies
# ===========================================================================


@pytest.mark.asyncio
class TestCdpGetCookies:

    async def test_returns_cookies(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Storage.getCookies = AsyncMock(
            return_value={"cookies": [{"name": "session", "value": "abc"}]}
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_cookies()
            assert len(result) == 1
            assert result[0]["name"] == "session"

    async def test_returns_empty_list_when_no_cookies(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Storage.getCookies = AsyncMock(return_value={})

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_cookies()
            assert result == []


# ===========================================================================
# _cdp_set_cookies
# ===========================================================================


@pytest.mark.asyncio
class TestCdpSetCookies:

    async def test_sets_cookies(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Storage.setCookies = AsyncMock()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            cookies = [{"name": "session", "value": "abc", "domain": ".example.com"}]
            await session._cdp_set_cookies(cookies)
            mock_cdp.cdp_client.send.Storage.setCookies.assert_called_once()

    async def test_returns_early_when_no_agent_focus(self):
        session = _make_browser_session()
        session.agent_focus = None
        # Should return early without calling anything
        await session._cdp_set_cookies([{"name": "x", "value": "y"}])

    async def test_returns_early_when_empty_cookies(self):
        session = _make_browser_session()
        await session._cdp_set_cookies([])


# ===========================================================================
# _cdp_clear_cookies
# ===========================================================================


@pytest.mark.asyncio
class TestCdpClearCookies:

    async def test_clears_cookies(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Storage.clearCookies = AsyncMock()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            await session._cdp_clear_cookies()
            mock_cdp.cdp_client.send.Storage.clearCookies.assert_called_once()


# ===========================================================================
# _cdp_set_extra_headers
# ===========================================================================


@pytest.mark.asyncio
class TestCdpSetExtraHeaders:

    async def test_returns_early_when_no_agent_focus(self):
        session = _make_browser_session()
        session.agent_focus = None
        # Should return without raising
        await session._cdp_set_extra_headers({"X-Custom": "value"})

    async def test_raises_not_implemented(self):
        session = _make_browser_session()
        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=_make_cdp_session()):
            with pytest.raises(NotImplementedError):
                await session._cdp_set_extra_headers({"X-Custom": "value"})


# ===========================================================================
# _cdp_grant_permissions
# ===========================================================================


@pytest.mark.asyncio
class TestCdpGrantPermissions:

    async def test_raises_not_implemented(self):
        session = _make_browser_session()
        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=_make_cdp_session()):
            with pytest.raises(NotImplementedError):
                await session._cdp_grant_permissions(["geolocation"])


# ===========================================================================
# _cdp_set_geolocation / _cdp_clear_geolocation
# ===========================================================================


@pytest.mark.asyncio
class TestCdpGeolocation:

    async def test_set_geolocation(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Emulation.setGeolocationOverride = AsyncMock()  # type: ignore[union-attr]
        await session._cdp_set_geolocation(latitude=40.7, longitude=-74.0, accuracy=50)
        session._cdp_client_root.send.Emulation.setGeolocationOverride.assert_called_once_with(  # type: ignore[union-attr]
            params={"latitude": 40.7, "longitude": -74.0, "accuracy": 50}
        )

    async def test_clear_geolocation(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Emulation.clearGeolocationOverride = AsyncMock()  # type: ignore[union-attr]
        await session._cdp_clear_geolocation()
        session._cdp_client_root.send.Emulation.clearGeolocationOverride.assert_called_once()  # type: ignore[union-attr]


# ===========================================================================
# _cdp_add_init_script / _cdp_remove_init_script
# ===========================================================================


@pytest.mark.asyncio
class TestCdpInitScript:

    async def test_add_init_script(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Page.addScriptToEvaluateOnNewDocument = AsyncMock(
            return_value={"identifier": "script-1"}
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_add_init_script("console.log('init')")
            assert result == "script-1"

    async def test_add_init_script_asserts_root_client(self):
        session = _make_browser_session()
        session._cdp_client_root = None  # type: ignore[attr-defined]
        with pytest.raises(AssertionError):
            await session._cdp_add_init_script("console.log('init')")

    async def test_remove_init_script(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Page.removeScriptToEvaluateOnNewDocument = AsyncMock()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            await session._cdp_remove_init_script("script-1")
            mock_cdp.cdp_client.send.Page.removeScriptToEvaluateOnNewDocument.assert_called_once()


# ===========================================================================
# _cdp_set_viewport
# ===========================================================================


@pytest.mark.asyncio
class TestCdpSetViewport:

    async def test_set_viewport_with_target_id(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Emulation.setDeviceMetricsOverride = AsyncMock()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            await session._cdp_set_viewport(width=1920, height=1080, target_id="t-1")
            mock_cdp.cdp_client.send.Emulation.setDeviceMetricsOverride.assert_called_once()

    async def test_set_viewport_with_agent_focus(self):
        session = _make_browser_session()
        mock_focus = _make_cdp_session()
        mock_focus.cdp_client.send.Emulation.setDeviceMetricsOverride = AsyncMock()
        session.agent_focus = mock_focus

        await session._cdp_set_viewport(width=1280, height=720)
        mock_focus.cdp_client.send.Emulation.setDeviceMetricsOverride.assert_called_once()

    async def test_set_viewport_returns_early_when_no_focus(self):
        session = _make_browser_session()
        session.agent_focus = None
        # Should not raise -- just logs warning and returns
        await session._cdp_set_viewport(width=800, height=600)


# ===========================================================================
# _cdp_get_origins
# ===========================================================================


@pytest.mark.asyncio
class TestCdpGetOrigins:

    async def test_returns_origins_with_storage(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        # Mock CDP calls
        mock_cdp.cdp_client.send.DOMStorage.enable = AsyncMock()
        mock_cdp.cdp_client.send.DOMStorage.disable = AsyncMock()
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "securityOrigin": "https://example.com"},
                    "childFrames": [],
                }
            }
        )
        mock_cdp.cdp_client.send.DOMStorage.getDOMStorageItems = AsyncMock(
            side_effect=[
                # localStorage call
                {"entries": [["key1", "val1"]]},
                # sessionStorage call
                {"entries": []},
            ]
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_origins()
            assert len(result) == 1
            assert result[0]["origin"] == "https://example.com"
            assert "localStorage" in result[0]

    async def test_returns_empty_on_exception(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.DOMStorage.enable = AsyncMock(side_effect=RuntimeError("fail"))

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_origins()
            assert result == []

    async def test_skips_null_origins(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOMStorage.enable = AsyncMock()
        mock_cdp.cdp_client.send.DOMStorage.disable = AsyncMock()
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "securityOrigin": "null"},
                    "childFrames": [],
                }
            }
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_origins()
            assert result == []

    async def test_extracts_origins_from_child_frames(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOMStorage.enable = AsyncMock()
        mock_cdp.cdp_client.send.DOMStorage.disable = AsyncMock()
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "securityOrigin": "https://parent.com"},
                    "childFrames": [
                        {
                            "frame": {"id": "f2", "securityOrigin": "https://child.com"},
                            "childFrames": [],
                        }
                    ],
                }
            }
        )

        # Will be called for each origin + storage type pair
        call_count = 0

        async def mock_storage_items(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # parent localStorage
                return {"entries": [["pk", "pv"]]}
            return {"entries": []}

        mock_cdp.cdp_client.send.DOMStorage.getDOMStorageItems = AsyncMock(side_effect=mock_storage_items)

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_origins()
            # At least one origin should have been found
            assert len(result) >= 1

    async def test_handles_storage_item_failure(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOMStorage.enable = AsyncMock()
        mock_cdp.cdp_client.send.DOMStorage.disable = AsyncMock()
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "securityOrigin": "https://example.com"},
                    "childFrames": [],
                }
            }
        )
        mock_cdp.cdp_client.send.DOMStorage.getDOMStorageItems = AsyncMock(
            side_effect=RuntimeError("storage error")
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_origins()
            # Should not raise, returns empty since storage retrieval failed
            assert result == []


# ===========================================================================
# _cdp_get_storage_state
# ===========================================================================


@pytest.mark.asyncio
class TestCdpGetStorageState:

    async def test_returns_cookies_and_origins(self):
        session = _make_browser_session()
        cookies = [{"name": "c", "value": "v"}]
        origins = [{"origin": "https://example.com", "localStorage": [{"name": "k", "value": "v"}]}]

        with patch.object(BrowserSession, "_cdp_get_cookies", new_callable=AsyncMock, return_value=cookies):
            with patch.object(BrowserSession, "_cdp_get_origins", new_callable=AsyncMock, return_value=origins):
                result = await session._cdp_get_storage_state()
                assert result["cookies"] == cookies
                assert result["origins"] == origins


# ===========================================================================
# _cdp_navigate
# ===========================================================================


@pytest.mark.asyncio
class TestCdpNavigate:

    async def test_navigates_to_url(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Page.navigate = AsyncMock()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            await session._cdp_navigate("https://test.com")
            mock_cdp.cdp_client.send.Page.navigate.assert_called_once()

    async def test_navigate_asserts_root_client(self):
        session = _make_browser_session()
        session._cdp_client_root = None  # type: ignore[attr-defined]
        with pytest.raises(AssertionError, match="CDP client not initialized"):
            await session._cdp_navigate("https://test.com")

    async def test_navigate_asserts_agent_focus(self):
        session = _make_browser_session()
        session.agent_focus = None
        with pytest.raises(AssertionError, match="CDP session not initialized"):
            await session._cdp_navigate("https://test.com")

    async def test_navigate_with_specific_target_id(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Page.navigate = AsyncMock()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            await session._cdp_navigate("https://test.com", target_id="specific-target")
            mock_cdp.cdp_client.send.Page.navigate.assert_called_once()


# ===========================================================================
# _is_valid_target (static method)
# ===========================================================================


class TestIsValidTarget:

    def test_http_page_target(self):
        target = {"type": "page", "url": "https://example.com"}
        assert BrowserSession._is_valid_target(target) is True

    def test_about_blank_page(self):
        target = {"type": "page", "url": "about:blank"}
        assert BrowserSession._is_valid_target(target) is True

    def test_about_srcdoc_excluded(self):
        target = {"type": "page", "url": "about:srcdoc"}
        assert BrowserSession._is_valid_target(target) is False

    def test_chrome_excluded_by_default(self):
        target = {"type": "page", "url": "chrome://settings"}
        assert BrowserSession._is_valid_target(target) is False

    def test_chrome_included_when_flag_set(self):
        target = {"type": "page", "url": "chrome://settings"}
        assert BrowserSession._is_valid_target(target, include_chrome=True) is True

    def test_chrome_extension_excluded_by_default(self):
        target = {"type": "page", "url": "chrome-extension://abc/popup.html"}
        assert BrowserSession._is_valid_target(target) is False

    def test_chrome_extension_included_when_flag_set(self):
        target = {"type": "page", "url": "chrome-extension://abc/popup.html"}
        assert BrowserSession._is_valid_target(target, include_chrome_extensions=True) is True

    def test_chrome_error_excluded_by_default(self):
        target = {"type": "page", "url": "chrome-error://chromewebdata/"}
        assert BrowserSession._is_valid_target(target) is False

    def test_chrome_error_included_when_flag_set(self):
        target = {"type": "page", "url": "chrome-error://chromewebdata/"}
        assert BrowserSession._is_valid_target(target, include_chrome_error=True) is True

    def test_service_worker_excluded_by_default(self):
        target = {"type": "service_worker", "url": "https://example.com/sw.js"}
        assert BrowserSession._is_valid_target(target) is False

    def test_worker_included_when_flag_set(self):
        target = {"type": "service_worker", "url": "https://example.com/sw.js"}
        assert BrowserSession._is_valid_target(target, include_workers=True) is True

    def test_iframe_included_when_flag_set(self):
        target = {"type": "iframe", "url": "https://child.com"}
        assert BrowserSession._is_valid_target(target, include_iframes=True) is True

    def test_iframe_excluded_by_default_include_iframes_false(self):
        # Default include_iframes=True in static method, but include_pages=True (iframe type != page)
        target = {"type": "iframe", "url": "https://child.com"}
        assert BrowserSession._is_valid_target(target, include_iframes=False) is False

    def test_new_tab_page_always_allowed(self):
        target = {"type": "page", "url": "chrome://new-tab-page/"}
        assert BrowserSession._is_valid_target(target) is True

    def test_chrome_newtab_always_allowed(self):
        target = {"type": "page", "url": "chrome://newtab/"}
        assert BrowserSession._is_valid_target(target) is True

    def test_tab_type_treated_as_page(self):
        target = {"type": "tab", "url": "https://example.com"}
        assert BrowserSession._is_valid_target(target) is True

    def test_webview_type_treated_as_iframe(self):
        target = {"type": "webview", "url": "https://example.com"}
        assert BrowserSession._is_valid_target(target, include_iframes=True) is True

    def test_shared_worker_included(self):
        target = {"type": "shared_worker", "url": "https://example.com/worker.js"}
        assert BrowserSession._is_valid_target(target, include_workers=True) is True

    def test_http_excluded_when_flag_false(self):
        target = {"type": "page", "url": "https://example.com"}
        assert BrowserSession._is_valid_target(target, include_http=False) is False

    def test_about_blank_still_allowed_via_new_tab_page(self):
        # about:blank is always allowed because is_new_tab_page() returns True for it
        target = {"type": "page", "url": "about:blank"}
        assert BrowserSession._is_valid_target(target, include_about=False) is True

    def test_pages_excluded_when_flag_false(self):
        target = {"type": "page", "url": "https://example.com"}
        assert BrowserSession._is_valid_target(target, include_pages=False) is False


# ===========================================================================
# get_all_frames
# ===========================================================================


@pytest.mark.asyncio
class TestGetAllFrames:

    async def test_basic_frame_tree(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = False

        mock_cdp = _make_cdp_session(target_id="target-1")
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "frame-1", "url": "https://example.com"},
                    "childFrames": [],
                }
            }
        )

        # Make sure agent_focus matches target
        session.agent_focus = mock_cdp
        session.agent_focus.target_id = "target-1"

        targets = [{"type": "page", "url": "https://example.com", "targetId": "target-1"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            all_frames, target_sessions = await session.get_all_frames()
            assert "frame-1" in all_frames
            assert all_frames["frame-1"]["frameTargetId"] == "target-1"

    async def test_skips_non_focus_targets_when_cross_origin_disabled(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = False

        mock_focus = _make_cdp_session(target_id="focused-target")
        mock_focus.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "url": "https://example.com"},
                    "childFrames": [],
                }
            }
        )
        session.agent_focus = mock_focus

        targets = [
            {"type": "page", "url": "https://example.com", "targetId": "focused-target"},
            {"type": "page", "url": "https://other.com", "targetId": "other-target"},
        ]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            all_frames, target_sessions = await session.get_all_frames()
            assert "f1" in all_frames

    async def test_cross_origin_iframes_enabled(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        mock_cdp = _make_cdp_session(target_id="target-1")
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "main-frame", "url": "https://example.com"},
                    "childFrames": [
                        {
                            "frame": {"id": "child-frame", "url": "https://child.com"},
                            "childFrames": [],
                        }
                    ],
                }
            }
        )

        targets = [{"type": "page", "url": "https://example.com", "targetId": "target-1"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
                with patch.object(BrowserSession, "_populate_frame_metadata", new_callable=AsyncMock):
                    all_frames, target_sessions = await session.get_all_frames()
                    assert "main-frame" in all_frames
                    assert "child-frame" in all_frames
                    assert "child-frame" in all_frames["main-frame"]["childFrameIds"]

    async def test_handles_frame_tree_exception(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        mock_cdp = _make_cdp_session(target_id="target-1")
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(side_effect=RuntimeError("no frames"))

        targets = [{"type": "page", "url": "https://example.com", "targetId": "target-1"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
                with patch.object(BrowserSession, "_populate_frame_metadata", new_callable=AsyncMock):
                    all_frames, target_sessions = await session.get_all_frames()
                    assert all_frames == {}

    async def test_iframe_target_merges_with_existing_frame(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        mock_cdp_page = _make_cdp_session(target_id="page-target")
        mock_cdp_page.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "shared-frame", "url": "https://example.com"},
                    "childFrames": [],
                }
            }
        )

        mock_cdp_iframe = _make_cdp_session(target_id="iframe-target")
        mock_cdp_iframe.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "shared-frame", "url": "https://example.com"},
                    "childFrames": [],
                }
            }
        )

        targets = [
            {"type": "page", "url": "https://example.com", "targetId": "page-target"},
            {"type": "iframe", "url": "https://example.com", "targetId": "iframe-target"},
        ]

        async def mock_get_session(target_id, focus=False):
            if target_id == "page-target":
                return mock_cdp_page
            return mock_cdp_iframe

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, side_effect=mock_get_session):
                with patch.object(BrowserSession, "_populate_frame_metadata", new_callable=AsyncMock):
                    all_frames, target_sessions = await session.get_all_frames()
                    # The iframe target should have updated the frameTargetId
                    assert all_frames["shared-frame"]["frameTargetId"] == "iframe-target"
                    assert all_frames["shared-frame"]["isCrossOrigin"] is True


# ===========================================================================
# _populate_frame_metadata
# ===========================================================================


@pytest.mark.asyncio
class TestPopulateFrameMetadata:

    async def test_populates_backend_node_id(self):
        session = _make_browser_session()
        session._cdp_client_root.send.DOM.enable = AsyncMock()  # type: ignore[union-attr]
        session._cdp_client_root.send.DOM.getFrameOwner = AsyncMock(  # type: ignore[union-attr]
            return_value={"backendNodeId": 99, "nodeId": 100}
        )

        all_frames = {
            "parent-frame": {"frameTargetId": "parent-target", "parentFrameId": None},
            "child-frame": {"frameTargetId": "child-target", "parentFrameId": "parent-frame"},
        }
        target_sessions = {"parent-target": "session-1"}

        await session._populate_frame_metadata(all_frames, target_sessions)

        assert all_frames["child-frame"]["parentTargetId"] == "parent-target"
        assert all_frames["child-frame"]["backendNodeId"] == 99
        assert all_frames["child-frame"]["nodeId"] == 100

    async def test_handles_frame_owner_exception(self):
        session = _make_browser_session()
        session._cdp_client_root.send.DOM.enable = AsyncMock()  # type: ignore[union-attr]
        session._cdp_client_root.send.DOM.getFrameOwner = AsyncMock(  # type: ignore[union-attr]
            side_effect=RuntimeError("cross-origin")
        )

        all_frames = {
            "parent-frame": {"frameTargetId": "parent-target", "parentFrameId": None},
            "child-frame": {"frameTargetId": "child-target", "parentFrameId": "parent-frame"},
        }
        target_sessions = {"parent-target": "session-1"}

        # Should not raise
        await session._populate_frame_metadata(all_frames, target_sessions)
        assert all_frames["child-frame"]["parentTargetId"] == "parent-target"
        # backendNodeId should not be set since it failed
        assert "backendNodeId" not in all_frames["child-frame"]

    async def test_skips_frames_without_parent(self):
        session = _make_browser_session()

        all_frames = {
            "root-frame": {"frameTargetId": "target-1", "parentFrameId": None},
        }
        target_sessions = {"target-1": "session-1"}

        await session._populate_frame_metadata(all_frames, target_sessions)
        # No parentTargetId should be set
        assert "parentTargetId" not in all_frames["root-frame"]


# ===========================================================================
# find_frame_target
# ===========================================================================


@pytest.mark.asyncio
class TestFindFrameTarget:

    async def test_finds_frame_in_provided_frames(self):
        session = _make_browser_session()
        all_frames = {"frame-1": {"url": "https://example.com"}}
        result = await session.find_frame_target("frame-1", all_frames)
        assert result is not None
        assert result["url"] == "https://example.com"

    async def test_returns_none_for_missing_frame(self):
        session = _make_browser_session()
        all_frames = {"frame-1": {"url": "https://example.com"}}
        result = await session.find_frame_target("nonexistent", all_frames)
        assert result is None

    async def test_calls_get_all_frames_when_none_provided(self):
        session = _make_browser_session()
        frames = {"f1": {"url": "https://example.com"}}
        with patch.object(BrowserSession, "get_all_frames", new_callable=AsyncMock, return_value=(frames, {})):
            result = await session.find_frame_target("f1")
            assert result is not None


# ===========================================================================
# cdp_client_for_target
# ===========================================================================


@pytest.mark.asyncio
class TestCdpClientForTarget:

    async def test_returns_session_for_target(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session(target_id="t-1")

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.cdp_client_for_target("t-1")
            assert result is mock_cdp


# ===========================================================================
# get_target_id_from_session_id
# ===========================================================================


class TestGetTargetIdFromSessionId:

    def test_returns_target_id_for_known_session(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session(target_id="t-1", session_id="s-1")
        session._cdp_session_pool = {"t-1": mock_cdp}  # type: ignore[attr-defined]

        result = session.get_target_id_from_session_id("s-1")
        assert result == "t-1"

    def test_returns_none_for_unknown_session(self):
        session = _make_browser_session()
        session._cdp_session_pool = {}  # type: ignore[attr-defined]
        result = session.get_target_id_from_session_id("unknown")
        assert result is None

    def test_returns_none_for_none_session_id(self):
        session = _make_browser_session()
        result = session.get_target_id_from_session_id(None)
        assert result is None


# ===========================================================================
# cdp_client_for_frame
# ===========================================================================


@pytest.mark.asyncio
class TestCdpClientForFrame:

    async def test_returns_main_session_when_cross_origin_disabled(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = False
        mock_cdp = _make_cdp_session()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.cdp_client_for_frame("any-frame")
            assert result is mock_cdp

    async def test_finds_frame_in_hierarchy(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        mock_cdp = _make_cdp_session(target_id="target-1")
        all_frames = {"frame-1": {"frameTargetId": "target-1"}}
        target_sessions = {"target-1": "session-1"}

        with patch.object(BrowserSession, "get_all_frames", new_callable=AsyncMock, return_value=(all_frames, target_sessions)):
            with patch.object(BrowserSession, "find_frame_target", new_callable=AsyncMock, return_value=all_frames["frame-1"]):
                with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
                    result = await session.cdp_client_for_frame("frame-1")
                    assert result is mock_cdp

    async def test_raises_when_frame_not_found(self):
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        with patch.object(BrowserSession, "get_all_frames", new_callable=AsyncMock, return_value=({}, {})):
            with patch.object(BrowserSession, "find_frame_target", new_callable=AsyncMock, return_value=None):
                with pytest.raises(ValueError, match="not found"):
                    await session.cdp_client_for_frame("missing-frame")


# ===========================================================================
# cdp_client_for_node
# ===========================================================================


@pytest.mark.asyncio
class TestCdpClientForNode:

    async def test_strategy1_session_id(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session(target_id="t-1", session_id="s-1")
        session._cdp_session_pool = {"t-1": mock_cdp}  # type: ignore[attr-defined]

        node = _make_mock_node(session_id="s-1")
        result = await session.cdp_client_for_node(node)
        assert result is mock_cdp

    async def test_strategy2_frame_id(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        node = _make_mock_node(session_id=None, frame_id="frame-1")

        with patch.object(BrowserSession, "cdp_client_for_frame", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.cdp_client_for_node(node)
            assert result is mock_cdp

    async def test_strategy3_target_id(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        node = _make_mock_node(session_id=None, frame_id=None, target_id="t-1")

        with patch.object(BrowserSession, "cdp_client_for_frame", new_callable=AsyncMock, side_effect=ValueError("not found")):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
                result = await session.cdp_client_for_node(node)
                assert result is mock_cdp

    async def test_strategy4_agent_focus_fallback(self):
        session = _make_browser_session()
        mock_focus = _make_cdp_session(url="https://focused.com")
        session.agent_focus = mock_focus

        node = _make_mock_node(session_id=None, frame_id=None, target_id=None)

        result = await session.cdp_client_for_node(node)
        assert result is mock_focus

    async def test_fallback_to_main_session(self):
        session = _make_browser_session()
        session.agent_focus = None
        mock_cdp = _make_cdp_session()

        node = _make_mock_node(session_id=None, frame_id=None, target_id=None)

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.cdp_client_for_node(node)
            assert result is mock_cdp

    async def test_strategy2_falls_through_on_exception(self):
        session = _make_browser_session()
        mock_focus = _make_cdp_session(url="https://focused.com")
        session.agent_focus = mock_focus

        node = _make_mock_node(session_id=None, frame_id="f-1", target_id=None)

        with patch.object(BrowserSession, "cdp_client_for_frame", new_callable=AsyncMock, side_effect=ValueError("nope")):
            result = await session.cdp_client_for_node(node)
            assert result is mock_focus

    async def test_strategy3_falls_through_on_exception(self):
        session = _make_browser_session()
        mock_focus = _make_cdp_session(url="https://focused.com")
        session.agent_focus = mock_focus

        node = _make_mock_node(session_id=None, frame_id=None, target_id="bad-target")

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, side_effect=ValueError("bad")):
            result = await session.cdp_client_for_node(node)
            assert result is mock_focus


# ===========================================================================
# take_screenshot
# ===========================================================================


@pytest.mark.asyncio
class TestTakeScreenshot:

    async def test_basic_screenshot(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        encoded = base64.b64encode(png_data).decode()
        mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": encoded}
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.take_screenshot()
            assert result == png_data

    async def test_screenshot_with_path(self, tmp_path):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        encoded = base64.b64encode(png_data).decode()
        mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": encoded}
        )

        file_path = str(tmp_path / "test.png")
        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.take_screenshot(path=file_path)
            assert result == png_data
            # File should have been written
            from pathlib import Path

            assert Path(file_path).read_bytes() == png_data

    async def test_screenshot_with_jpeg_quality(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        jpeg_data = b"\xff\xd8\xff" + b"\x00" * 50
        encoded = base64.b64encode(jpeg_data).decode()
        mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": encoded}
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.take_screenshot(format="jpeg", quality=80)
            assert result == jpeg_data

    async def test_screenshot_with_clip(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        png_data = b"\x89PNG\r\n\x1a\n"
        encoded = base64.b64encode(png_data).decode()
        mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": encoded}
        )

        clip = {"x": 0, "y": 0, "width": 100, "height": 100}
        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.take_screenshot(clip=clip)
            assert result == png_data

    async def test_screenshot_full_page(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        png_data = b"\x89PNG\r\n\x1a\n"
        encoded = base64.b64encode(png_data).decode()
        mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(
            return_value={"data": encoded}
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session.take_screenshot(full_page=True)
            assert result == png_data

    async def test_screenshot_raises_when_no_data(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value={})

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            with pytest.raises(Exception, match="Screenshot failed"):
                await session.take_screenshot()

    async def test_screenshot_raises_when_none_result(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Page.captureScreenshot = AsyncMock(return_value=None)

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            with pytest.raises(Exception, match="Screenshot failed"):
                await session.take_screenshot()


# ===========================================================================
# screenshot_element
# ===========================================================================


@pytest.mark.asyncio
class TestScreenshotElement:

    async def test_screenshot_element_success(self):
        session = _make_browser_session()
        bounds = {"x": 10, "y": 20, "width": 100, "height": 50}
        png_data = b"\x89PNG"

        with patch.object(BrowserSession, "_get_element_bounds", new_callable=AsyncMock, return_value=bounds):
            with patch.object(BrowserSession, "take_screenshot", new_callable=AsyncMock, return_value=png_data):
                result = await session.screenshot_element("#my-element")
                assert result == png_data

    async def test_screenshot_element_not_found(self):
        session = _make_browser_session()

        with patch.object(BrowserSession, "_get_element_bounds", new_callable=AsyncMock, return_value=None):
            with pytest.raises(ValueError, match="not found or has no bounds"):
                await session.screenshot_element("#missing")

    async def test_screenshot_element_with_path_and_quality(self):
        session = _make_browser_session()
        bounds = {"x": 0, "y": 0, "width": 50, "height": 50}
        png_data = b"\xff\xd8\xff"

        with patch.object(BrowserSession, "_get_element_bounds", new_callable=AsyncMock, return_value=bounds):
            with patch.object(BrowserSession, "take_screenshot", new_callable=AsyncMock, return_value=png_data) as mock_ss:
                result = await session.screenshot_element("#elem", path="/tmp/out.jpg", format="jpeg", quality=90)
                assert result == png_data
                mock_ss.assert_called_once_with(path="/tmp/out.jpg", format="jpeg", quality=90, clip=bounds)


# ===========================================================================
# _get_element_bounds
# ===========================================================================


@pytest.mark.asyncio
class TestGetElementBounds:

    async def test_returns_bounds_for_element(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1}}
        )
        mock_cdp.cdp_client.send.DOM.querySelector = AsyncMock(
            return_value={"nodeId": 42}
        )
        # content quad: [x1,y1, x2,y2, x3,y3, x4,y4]
        mock_cdp.cdp_client.send.DOM.getBoxModel = AsyncMock(
            return_value={
                "model": {
                    "content": [10, 20, 110, 20, 110, 70, 10, 70]
                }
            }
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._get_element_bounds("#test")
            assert result is not None
            assert result["x"] == 10
            assert result["y"] == 20
            assert result["width"] == 100
            assert result["height"] == 50

    async def test_returns_none_when_node_not_found(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1}}
        )
        mock_cdp.cdp_client.send.DOM.querySelector = AsyncMock(
            return_value={"nodeId": 0}  # nodeId 0 means not found
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._get_element_bounds("#missing")
            assert result is None

    async def test_returns_none_when_no_box_model(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1}}
        )
        mock_cdp.cdp_client.send.DOM.querySelector = AsyncMock(
            return_value={"nodeId": 42}
        )
        mock_cdp.cdp_client.send.DOM.getBoxModel = AsyncMock(
            return_value={"model": None}
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._get_element_bounds("#no-box")
            assert result is None

    async def test_returns_none_when_box_model_missing_key(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1}}
        )
        mock_cdp.cdp_client.send.DOM.querySelector = AsyncMock(
            return_value={"nodeId": 42}
        )
        mock_cdp.cdp_client.send.DOM.getBoxModel = AsyncMock(
            return_value={}  # No "model" key
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._get_element_bounds("#broken")
            assert result is None


# ===========================================================================
# Additional edge case tests for remaining missed lines
# ===========================================================================


@pytest.mark.asyncio
class TestCdpGetAllPagesFilterCombinations:
    """Test _cdp_get_all_pages with various filter flags."""

    async def test_include_workers(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(  # type: ignore[union-attr]
            return_value={
                "targetInfos": [
                    {"type": "service_worker", "url": "https://example.com/sw.js", "targetId": "sw1"},
                    {"type": "page", "url": "https://example.com", "targetId": "p1"},
                ]
            }
        )
        result = await session._cdp_get_all_pages(include_workers=True)
        target_ids = [t["targetId"] for t in result]
        assert "sw1" in target_ids
        assert "p1" in target_ids

    async def test_include_chrome_extensions(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(  # type: ignore[union-attr]
            return_value={
                "targetInfos": [
                    {"type": "page", "url": "chrome-extension://abc/popup.html", "targetId": "ext1"},
                ]
            }
        )
        result = await session._cdp_get_all_pages(include_chrome_extensions=True)
        assert len(result) == 1
        assert result[0]["targetId"] == "ext1"

    async def test_include_iframes(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(  # type: ignore[union-attr]
            return_value={
                "targetInfos": [
                    {"type": "iframe", "url": "https://child.com", "targetId": "if1"},
                ]
            }
        )
        result = await session._cdp_get_all_pages(include_iframes=True)
        assert len(result) == 1

    async def test_include_chrome_error(self):
        session = _make_browser_session()
        session._cdp_client_root.send.Target.getTargets = AsyncMock(  # type: ignore[union-attr]
            return_value={
                "targetInfos": [
                    {"type": "page", "url": "chrome-error://chromewebdata/", "targetId": "err1"},
                ]
            }
        )
        result = await session._cdp_get_all_pages(include_chrome_error=True)
        assert len(result) == 1


@pytest.mark.asyncio
class TestCdpSetViewportEdgeCases:

    async def test_set_viewport_passes_mobile_and_scale(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()
        mock_cdp.cdp_client.send.Emulation.setDeviceMetricsOverride = AsyncMock()

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            await session._cdp_set_viewport(width=375, height=812, device_scale_factor=3.0, mobile=True, target_id="t-1")
            call_args = mock_cdp.cdp_client.send.Emulation.setDeviceMetricsOverride.call_args
            assert call_args.kwargs["params"]["mobile"] is True
            assert call_args.kwargs["params"]["deviceScaleFactor"] == 3.0


@pytest.mark.asyncio
class TestCdpGetOriginsEdgeCases:

    async def test_origin_without_any_storage_is_excluded(self):
        """An origin that has empty localStorage AND sessionStorage should be excluded from results."""
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOMStorage.enable = AsyncMock()
        mock_cdp.cdp_client.send.DOMStorage.disable = AsyncMock()
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "securityOrigin": "https://empty.com"},
                    "childFrames": [],
                }
            }
        )
        # Both localStorage and sessionStorage return empty
        mock_cdp.cdp_client.send.DOMStorage.getDOMStorageItems = AsyncMock(
            return_value={"entries": []}
        )

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_origins()
            assert result == []

    async def test_origin_with_session_storage_only(self):
        session = _make_browser_session()
        mock_cdp = _make_cdp_session()

        mock_cdp.cdp_client.send.DOMStorage.enable = AsyncMock()
        mock_cdp.cdp_client.send.DOMStorage.disable = AsyncMock()
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "securityOrigin": "https://session-only.com"},
                    "childFrames": [],
                }
            }
        )

        call_count = 0

        async def mock_get_items(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # localStorage: empty
                return {"entries": []}
            else:
                # sessionStorage: has items
                return {"entries": [["token", "abc123"]]}

        mock_cdp.cdp_client.send.DOMStorage.getDOMStorageItems = AsyncMock(side_effect=mock_get_items)

        with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
            result = await session._cdp_get_origins()
            assert len(result) == 1
            assert "sessionStorage" in result[0]


@pytest.mark.asyncio
class TestGetAllFramesEdgeCases:

    async def test_cross_origin_isolated_frame(self):
        """Frames with crossOriginIsolatedContextType are marked as cross-origin."""
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        mock_cdp = _make_cdp_session(target_id="t-1")
        mock_cdp.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {
                        "id": "f1",
                        "url": "https://example.com",
                        "crossOriginIsolatedContextType": "Isolated",
                    },
                    "childFrames": [],
                }
            }
        )

        targets = [{"type": "page", "url": "https://example.com", "targetId": "t-1"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=mock_cdp):
                with patch.object(BrowserSession, "_populate_frame_metadata", new_callable=AsyncMock):
                    all_frames, _ = await session.get_all_frames()
                    assert all_frames["f1"]["isCrossOrigin"] is True

    async def test_skips_iframe_targets_when_cross_origin_disabled(self):
        """When cross_origin_iframes is False, iframe type targets should be skipped."""
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = False

        mock_focus = _make_cdp_session(target_id="page-target")
        mock_focus.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": {"id": "f1", "url": "https://example.com"},
                    "childFrames": [],
                }
            }
        )
        session.agent_focus = mock_focus

        targets = [
            {"type": "page", "url": "https://example.com", "targetId": "page-target"},
            {"type": "iframe", "url": "https://iframe.com", "targetId": "iframe-target"},
        ]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            all_frames, _ = await session.get_all_frames()
            assert "f1" in all_frames

    async def test_session_none_skipped(self):
        """When get_or_create_cdp_session returns None for a target, it's skipped."""
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        targets = [{"type": "page", "url": "https://example.com", "targetId": "t-1"}]

        with patch.object(BrowserSession, "_cdp_get_all_pages", new_callable=AsyncMock, return_value=targets):
            with patch.object(BrowserSession, "get_or_create_cdp_session", new_callable=AsyncMock, return_value=None):
                with patch.object(BrowserSession, "_populate_frame_metadata", new_callable=AsyncMock):
                    all_frames, target_sessions = await session.get_all_frames()
                    assert all_frames == {}
                    assert target_sessions == {}


@pytest.mark.asyncio
class TestCdpClientForFrameEdgeCases:

    async def test_frame_found_but_target_not_in_sessions(self):
        """When frame is found but its target_id is not in target_sessions, raises ValueError."""
        session = _make_browser_session()
        session.browser_profile.cross_origin_iframes = True

        frame_info = {"frameTargetId": "orphaned-target"}
        all_frames = {"frame-1": frame_info}
        target_sessions = {}  # target not present

        with patch.object(BrowserSession, "get_all_frames", new_callable=AsyncMock, return_value=(all_frames, target_sessions)):
            with patch.object(BrowserSession, "find_frame_target", new_callable=AsyncMock, return_value=frame_info):
                with pytest.raises(ValueError, match="not found"):
                    await session.cdp_client_for_frame("frame-1")


@pytest.mark.asyncio
class TestCdpClientForNodeEdgeCases:

    async def test_session_id_not_in_pool(self):
        """When node has session_id but it's not in the pool, falls through to next strategy."""
        session = _make_browser_session()
        session._cdp_session_pool = {}  # type: ignore[attr-defined]
        mock_focus = _make_cdp_session(url="https://focused.com")
        session.agent_focus = mock_focus

        node = _make_mock_node(session_id="unknown-session", frame_id=None, target_id=None)
        result = await session.cdp_client_for_node(node)
        # Falls through to agent_focus
        assert result is mock_focus


@pytest.mark.asyncio
class TestPopulateFrameMetadataEdgeCases:

    async def test_parent_target_not_in_sessions(self):
        """When parent target_id exists but is not in target_sessions, skip backend node lookup."""
        session = _make_browser_session()

        all_frames = {
            "parent-frame": {"frameTargetId": "orphan-target", "parentFrameId": None},
            "child-frame": {"frameTargetId": "child-target", "parentFrameId": "parent-frame"},
        }
        target_sessions = {}  # No sessions at all

        await session._populate_frame_metadata(all_frames, target_sessions)
        # parentTargetId should still be set
        assert all_frames["child-frame"]["parentTargetId"] == "orphan-target"
        # But no backendNodeId since session wasn't available
        assert "backendNodeId" not in all_frames["child-frame"]

    async def test_frame_owner_returns_none(self):
        session = _make_browser_session()
        session._cdp_client_root.send.DOM.enable = AsyncMock()  # type: ignore[union-attr]
        session._cdp_client_root.send.DOM.getFrameOwner = AsyncMock(return_value=None)  # type: ignore[union-attr]

        all_frames = {
            "parent-frame": {"frameTargetId": "parent-target", "parentFrameId": None},
            "child-frame": {"frameTargetId": "child-target", "parentFrameId": "parent-frame"},
        }
        target_sessions = {"parent-target": "session-1"}

        await session._populate_frame_metadata(all_frames, target_sessions)
        # backendNodeId should NOT be set since frame_owner returned None
        assert "backendNodeId" not in all_frames["child-frame"]


class TestIsValidTargetAdditionalEdgeCases:

    def test_worker_type_included(self):
        target = {"type": "worker", "url": "https://example.com/worker.js"}
        assert BrowserSession._is_valid_target(target, include_workers=True) is True

    def test_empty_type_and_url(self):
        target = {"type": "", "url": ""}
        assert BrowserSession._is_valid_target(target) is False

    def test_http_url_but_wrong_type(self):
        """URL is valid but type is not allowed."""
        target = {"type": "browser", "url": "https://example.com"}
        assert BrowserSession._is_valid_target(target) is False
