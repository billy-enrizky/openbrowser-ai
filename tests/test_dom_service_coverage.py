"""Tests for DomService (service.py) - comprehensive coverage."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.dom.service import DomService
from openbrowser.dom.views import (
    DOMRect,
    EnhancedAXNode,
    EnhancedAXProperty,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    SerializedDOMState,
    TargetAllTrees,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_snapshot(bounds=None, computed_styles=None, scroll_rects=None, client_rects=None):
    return EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=bounds or DOMRect(x=0, y=0, width=100, height=50),
        clientRects=client_rects,
        scrollRects=scroll_rects,
        computed_styles=computed_styles,
        paint_order=None,
        stacking_contexts=None,
    )


def _make_node(
    tag="div",
    node_type=NodeType.ELEMENT_NODE,
    node_value="",
    attributes=None,
    children=None,
    is_visible=True,
    snapshot=None,
    content_document=None,
    frame_id=None,
    node_name=None,
):
    _id_counter = getattr(_make_node, "_counter", 0) + 1
    _make_node._counter = _id_counter

    return EnhancedDOMTreeNode(
        node_id=_id_counter,
        backend_node_id=_id_counter + 50000,
        node_type=node_type,
        node_name=node_name or tag,
        node_value=node_value,
        attributes=attributes or {},
        is_scrollable=None,
        is_visible=is_visible,
        absolute_position=None,
        target_id="target-1",
        frame_id=frame_id,
        session_id=None,
        content_document=content_document,
        shadow_root_type=None,
        shadow_roots=None,
        parent_node=None,
        children_nodes=children,
        ax_node=None,
        snapshot_node=snapshot or (_make_snapshot() if is_visible else None),
    )


def _make_browser_session():
    session = MagicMock()
    session.logger = logging.getLogger("test.dom_service")
    session.current_target_id = "target-abc"
    session.agent_focus = MagicMock()
    session.agent_focus.session_id = "session-1"
    session.cdp_client = MagicMock()
    session.get_all_frames = AsyncMock(return_value=({}, {}))
    session.get_or_create_cdp_session = AsyncMock()
    return session


# ────────────────────────────────────────────────────────────────────
# __init__ and context manager
# ────────────────────────────────────────────────────────────────────

class TestDomServiceInit:
    """Test __init__ (lines 46-59) and context manager (lines 61-65)."""

    def test_init_defaults(self):
        session = _make_browser_session()
        svc = DomService(session)
        assert svc.cross_origin_iframes is False
        assert svc.paint_order_filtering is True
        assert svc.max_iframes == 100
        assert svc.max_iframe_depth == 5

    def test_init_custom(self):
        session = _make_browser_session()
        svc = DomService(
            session,
            cross_origin_iframes=True,
            paint_order_filtering=False,
            max_iframes=50,
            max_iframe_depth=3,
        )
        assert svc.cross_origin_iframes is True
        assert svc.paint_order_filtering is False
        assert svc.max_iframes == 50

    def test_init_custom_logger(self):
        session = _make_browser_session()
        custom_logger = logging.getLogger("custom")
        svc = DomService(session, logger=custom_logger)
        assert svc.logger is custom_logger

    @pytest.mark.asyncio
    async def test_aenter_aexit(self):
        session = _make_browser_session()
        async with DomService(session) as svc:
            assert svc is not None


# ────────────────────────────────────────────────────────────────────
# _get_targets_for_page
# ────────────────────────────────────────────────────────────────────

class TestGetTargetsForPage:
    """Test _get_targets_for_page (lines 67-108)."""

    @pytest.mark.asyncio
    async def test_with_target_id(self):
        session = _make_browser_session()
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": "t-1", "type": "page", "title": "Test", "url": "http://test.com"},
            ]
        })
        session.get_all_frames = AsyncMock(return_value=({}, {}))
        svc = DomService(session)
        result = await svc._get_targets_for_page(target_id="t-1")
        assert result.page_session["targetId"] == "t-1"

    @pytest.mark.asyncio
    async def test_no_target_id_uses_current(self):
        session = _make_browser_session()
        session.current_target_id = "t-current"
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": "t-current", "type": "page", "title": "Test", "url": "http://test.com"},
            ]
        })
        session.get_all_frames = AsyncMock(return_value=({}, {}))
        svc = DomService(session)
        result = await svc._get_targets_for_page()
        assert result.page_session["targetId"] == "t-current"

    @pytest.mark.asyncio
    async def test_no_target_id_no_current_raises(self):
        session = _make_browser_session()
        session.current_target_id = None
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": []
        })
        svc = DomService(session)
        with pytest.raises(ValueError, match="No current target"):
            await svc._get_targets_for_page()

    @pytest.mark.asyncio
    async def test_target_not_found_raises(self):
        session = _make_browser_session()
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": "other", "type": "page"},
            ]
        })
        svc = DomService(session)
        with pytest.raises(ValueError, match="No target found"):
            await svc._get_targets_for_page(target_id="missing")

    @pytest.mark.asyncio
    async def test_cross_origin_iframes_detected(self):
        session = _make_browser_session()
        session.cdp_client.send.Target.getTargets = AsyncMock(return_value={
            "targetInfos": [
                {"targetId": "t-1", "type": "page", "title": "Test", "url": "http://test.com"},
                {"targetId": "t-iframe", "type": "iframe", "title": "Iframe", "url": "http://other.com"},
            ]
        })
        session.get_all_frames = AsyncMock(return_value=({
            "frame-1": {
                "isCrossOrigin": True,
                "frameTargetId": "t-iframe",
                "parentTargetId": "t-1",
            }
        }, {}))
        svc = DomService(session)
        result = await svc._get_targets_for_page(target_id="t-1")
        assert len(result.iframe_sessions) == 1


# ────────────────────────────────────────────────────────────────────
# _build_enhanced_ax_node
# ────────────────────────────────────────────────────────────────────

class TestBuildEnhancedAxNode:
    """Test _build_enhanced_ax_node (lines 110-136)."""

    def test_basic_ax_node(self):
        session = _make_browser_session()
        svc = DomService(session)
        ax_node = {
            "nodeId": "ax-1",
            "ignored": False,
            "role": {"value": "button"},
            "name": {"value": "Submit"},
            "description": {"value": "Submit form"},
        }
        result = svc._build_enhanced_ax_node(ax_node)
        assert result.role == "button"
        assert result.name == "Submit"
        assert result.description == "Submit form"

    def test_ax_node_with_properties(self):
        session = _make_browser_session()
        svc = DomService(session)
        ax_node = {
            "nodeId": "ax-1",
            "ignored": False,
            "properties": [
                {"name": "checked", "value": {"value": True}},
                {"name": "disabled", "value": {"value": False}},
            ],
        }
        result = svc._build_enhanced_ax_node(ax_node)
        assert result.properties is not None
        assert len(result.properties) == 2

    def test_ax_node_with_invalid_property(self):
        session = _make_browser_session()
        svc = DomService(session)
        ax_node = {
            "nodeId": "ax-1",
            "ignored": False,
            "properties": [
                {"name": "invalid_random_prop", "value": {"value": True}},
            ],
        }
        # Should handle invalid property names gracefully
        result = svc._build_enhanced_ax_node(ax_node)
        # May either include it or skip it depending on enum validation
        assert result is not None

    def test_ax_node_without_properties(self):
        session = _make_browser_session()
        svc = DomService(session)
        ax_node = {
            "nodeId": "ax-1",
            "ignored": True,
        }
        result = svc._build_enhanced_ax_node(ax_node)
        assert result.ignored is True
        assert result.properties is None

    def test_ax_node_with_child_ids(self):
        session = _make_browser_session()
        svc = DomService(session)
        ax_node = {
            "nodeId": "ax-1",
            "ignored": False,
            "childIds": ["child-1", "child-2"],
        }
        result = svc._build_enhanced_ax_node(ax_node)
        assert result.child_ids == ["child-1", "child-2"]

    def test_ax_node_empty_child_ids(self):
        session = _make_browser_session()
        svc = DomService(session)
        ax_node = {
            "nodeId": "ax-1",
            "ignored": False,
            "childIds": [],
        }
        result = svc._build_enhanced_ax_node(ax_node)
        assert result.child_ids is None


# ────────────────────────────────────────────────────────────────────
# _get_viewport_ratio
# ────────────────────────────────────────────────────────────────────

class TestGetViewportRatio:
    """Test _get_viewport_ratio (lines 138-165)."""

    @pytest.mark.asyncio
    async def test_normal_viewport(self):
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
            "visualViewport": {"clientWidth": 1920},
            "cssVisualViewport": {"clientWidth": 960},
            "cssLayoutViewport": {"clientWidth": 960},
        })
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        ratio = await svc._get_viewport_ratio("target-1")
        assert ratio == 2.0

    @pytest.mark.asyncio
    async def test_viewport_exception_fallback(self):
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(side_effect=Exception("fail"))
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        ratio = await svc._get_viewport_ratio("target-1")
        assert ratio == 1.0

    @pytest.mark.asyncio
    async def test_viewport_zero_css_width(self):
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
            "visualViewport": {"clientWidth": 0},
            "cssVisualViewport": {"clientWidth": 0},
            "cssLayoutViewport": {"clientWidth": 0},
        })
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        ratio = await svc._get_viewport_ratio("target-1")
        assert ratio == 1.0


# ────────────────────────────────────────────────────────────────────
# is_element_visible_according_to_all_parents (class method)
# ────────────────────────────────────────────────────────────────────

class TestIsElementVisible:
    """Test is_element_visible_according_to_all_parents (lines 167-252)."""

    def test_no_snapshot(self):
        node = _make_node(is_visible=True, snapshot=None)
        node.snapshot_node = None
        assert DomService.is_element_visible_according_to_all_parents(node, []) is False

    def test_display_none(self):
        snap = _make_snapshot(computed_styles={"display": "none", "visibility": "visible", "opacity": "1"})
        node = _make_node(snapshot=snap)
        assert DomService.is_element_visible_according_to_all_parents(node, []) is False

    def test_visibility_hidden(self):
        snap = _make_snapshot(computed_styles={"display": "block", "visibility": "hidden", "opacity": "1"})
        node = _make_node(snapshot=snap)
        assert DomService.is_element_visible_according_to_all_parents(node, []) is False

    def test_opacity_zero(self):
        snap = _make_snapshot(computed_styles={"display": "block", "visibility": "visible", "opacity": "0"})
        node = _make_node(snapshot=snap)
        assert DomService.is_element_visible_according_to_all_parents(node, []) is False

    def test_invalid_opacity(self):
        snap = _make_snapshot(computed_styles={"display": "block", "visibility": "visible", "opacity": "invalid"})
        node = _make_node(snapshot=snap)
        # Should not crash, treat as visible
        assert DomService.is_element_visible_according_to_all_parents(node, []) is True

    def test_no_bounds(self):
        snap = EnhancedSnapshotNode(
            is_clickable=None, cursor_style=None, bounds=None,
            clientRects=None, scrollRects=None,
            computed_styles={"display": "block", "visibility": "visible", "opacity": "1"},
            paint_order=None, stacking_contexts=None,
        )
        node = _make_node(snapshot=snap)
        assert DomService.is_element_visible_according_to_all_parents(node, []) is False

    def test_visible_no_frames(self):
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 50),
            computed_styles={"display": "block", "visibility": "visible", "opacity": "1"},
        )
        node = _make_node(snapshot=snap)
        assert DomService.is_element_visible_according_to_all_parents(node, []) is True

    def test_iframe_frame_offset(self):
        """Test element inside iframe with offset."""
        snap = _make_snapshot(
            bounds=DOMRect(10, 10, 50, 30),
            computed_styles={"display": "block", "visibility": "visible", "opacity": "1"},
        )
        node = _make_node(snapshot=snap)

        iframe_snap = _make_snapshot(bounds=DOMRect(100, 100, 800, 600))
        iframe_frame = _make_node(
            tag="IFRAME", node_name="IFRAME",
            snapshot=iframe_snap, frame_id="frame-1",
        )

        result = DomService.is_element_visible_according_to_all_parents(node, [iframe_frame])
        assert result is True

    def test_html_frame_with_scroll(self):
        """Test visibility check with scroll position in HTML frame."""
        snap = _make_snapshot(
            bounds=DOMRect(10, 500, 50, 30),
            computed_styles={"display": "block", "visibility": "visible", "opacity": "1"},
        )
        node = _make_node(snapshot=snap)

        html_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 800, 600),
            scroll_rects=DOMRect(0, 0, 800, 2000),
            client_rects=DOMRect(0, 0, 800, 600),
        )
        html_frame = _make_node(
            tag="HTML", node_name="HTML",
            snapshot=html_snap, frame_id="main-frame",
        )

        result = DomService.is_element_visible_according_to_all_parents(node, [html_frame])
        assert result is True

    def test_element_outside_scroll_viewport(self):
        """Element far outside scroll viewport should not be visible."""
        snap = _make_snapshot(
            bounds=DOMRect(10, 50000, 50, 30),
            computed_styles={"display": "block", "visibility": "visible", "opacity": "1"},
        )
        node = _make_node(snapshot=snap)

        html_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 800, 600),
            scroll_rects=DOMRect(0, 0, 800, 2000),
            client_rects=DOMRect(0, 0, 800, 600),
        )
        html_frame = _make_node(
            tag="HTML", node_name="HTML",
            snapshot=html_snap, frame_id="main-frame",
        )

        result = DomService.is_element_visible_according_to_all_parents(node, [html_frame])
        assert result is False


# ────────────────────────────────────────────────────────────────────
# _get_ax_tree_for_all_frames
# ────────────────────────────────────────────────────────────────────

class TestGetAxTreeForAllFrames:
    """Test _get_ax_tree_for_all_frames (lines 254-289)."""

    @pytest.mark.asyncio
    async def test_single_frame(self):
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(return_value={
            "frameTree": {
                "frame": {"id": "frame-1"},
            }
        })
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(return_value={
            "nodes": [{"nodeId": "n1", "ignored": False}]
        })
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        result = await svc._get_ax_tree_for_all_frames("target-1")
        assert len(result["nodes"]) == 1

    @pytest.mark.asyncio
    async def test_nested_frames(self):
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(return_value={
            "frameTree": {
                "frame": {"id": "frame-1"},
                "childFrames": [
                    {"frame": {"id": "frame-2"}},
                ],
            }
        })
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(return_value={
            "nodes": [{"nodeId": "n1", "ignored": False}]
        })
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        result = await svc._get_ax_tree_for_all_frames("target-1")
        assert len(result["nodes"]) == 2  # One from each frame


# ────────────────────────────────────────────────────────────────────
# _get_all_trees
# ────────────────────────────────────────────────────────────────────

class TestGetAllTrees:
    """Test _get_all_trees (lines 291-448)."""

    @pytest.mark.asyncio
    async def test_basic_tree_retrieval(self):
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"

        # Mock all CDP calls
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={
            "result": {"value": "complete"}
        })
        cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = AsyncMock(return_value={
            "documents": [{"nodes": {}, "layout": {}}],
            "strings": [],
        })
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={
            "root": {"nodeId": 1, "nodeType": 9, "nodeName": "#document"}
        })
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
            "visualViewport": {"clientWidth": 1920},
            "cssVisualViewport": {"clientWidth": 1920},
            "cssLayoutViewport": {"clientWidth": 1920},
        })
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(return_value={
            "frameTree": {"frame": {"id": "f-1"}}
        })
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(return_value={
            "nodes": []
        })
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        result = await svc._get_all_trees("target-1")
        assert result.snapshot is not None
        assert result.dom_tree is not None


# ────────────────────────────────────────────────────────────────────
# detect_pagination_buttons (static method)
# ────────────────────────────────────────────────────────────────────

class TestDetectPaginationButtons:
    """Test detect_pagination_buttons (lines 751-825)."""

    def _make_pagination_node(self, text="", attributes=None, is_clickable=True):
        snap = _make_snapshot() if is_clickable else None
        if snap:
            snap.is_clickable = is_clickable
        text_node = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value=text,
        )
        node = _make_node(
            tag="button", snapshot=snap,
            children=[text_node], attributes=attributes or {},
        )
        return node

    def test_detect_next_button(self):
        node = self._make_pagination_node("Next")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "next" for b in result)

    def test_detect_prev_button(self):
        node = self._make_pagination_node("Previous")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "prev" for b in result)

    def test_detect_first_button(self):
        node = self._make_pagination_node("First")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "first" for b in result)

    def test_detect_last_button(self):
        node = self._make_pagination_node("Last")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "last" for b in result)

    def test_detect_page_number(self):
        node = self._make_pagination_node("5", attributes={"role": "button"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "page_number" for b in result)

    def test_detect_arrow_button(self):
        node = self._make_pagination_node(">")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "next" for b in result)

    def test_detect_disabled_button(self):
        node = self._make_pagination_node("Next", attributes={"disabled": "true"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) > 0
        assert result[0]["is_disabled"] is True

    def test_aria_label_pagination(self):
        node = self._make_pagination_node("", attributes={"aria-label": "next page"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "next" for b in result)

    def test_title_pagination(self):
        node = self._make_pagination_node("", attributes={"title": "previous page"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "prev" for b in result)

    def test_no_snapshot_skipped(self):
        node = self._make_pagination_node("Next", is_clickable=False)
        node.snapshot_node = None
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) == 0

    def test_not_clickable_skipped(self):
        snap = _make_snapshot()
        snap.is_clickable = False
        node = self._make_pagination_node("Next")
        node.snapshot_node = snap
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) == 0

    def test_non_pagination_text_ignored(self):
        node = self._make_pagination_node("Submit Form")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) == 0

    def test_class_based_detection(self):
        node = self._make_pagination_node("", attributes={"class": "next-page btn"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "next" for b in result)

    def test_empty_selector_map(self):
        result = DomService.detect_pagination_buttons({})
        assert result == []

    def test_unicode_arrow_next(self):
        node = self._make_pagination_node("\u00bb")  # >>
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "next" for b in result)

    def test_page_number_too_many_digits(self):
        """Page numbers with more than 2 digits should not match."""
        node = self._make_pagination_node("123", attributes={"role": "button"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert not any(b["button_type"] == "page_number" for b in result)
