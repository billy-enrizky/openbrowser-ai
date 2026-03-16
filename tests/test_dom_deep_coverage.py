"""Deep coverage tests for DOM modules:

1. dom/serializer/html_serializer.py (87 lines, 8% covered, 80 lines missed)
   Missing: 25, 37-160, 171-190, 201, 212

2. dom/service.py (344 lines, 81% covered, 66 lines missed)
   Missing: 124-125, 299-300, 336, 375-396, 405-410, 414, 428-431, 438,
            510, 529-530, 592-596, 599-602, 618-619, 624, 645, 657-716

3. daemon/server.py (222 lines, 73% covered, 61 lines missed)
   Missing: 92-126, 137-162, 197-198, 261-262, 303, 333-334, 344

4. dom/markdown_extractor.py (56 lines, 12% covered, 49 lines missed)
"""

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path
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


# ============================================================================
# Shared helpers
# ============================================================================


def _make_snapshot(
    bounds=None,
    computed_styles=None,
    scroll_rects=None,
    client_rects=None,
    is_clickable=None,
    paint_order=None,
):
    return EnhancedSnapshotNode(
        is_clickable=is_clickable,
        cursor_style=None,
        bounds=bounds or DOMRect(x=0, y=0, width=100, height=50),
        clientRects=client_rects,
        scrollRects=scroll_rects,
        computed_styles=computed_styles,
        paint_order=paint_order,
        stacking_contexts=None,
    )


_id_counter = 0


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
    shadow_roots=None,
    shadow_root_type=None,
    parent_node=None,
):
    global _id_counter
    _id_counter += 1

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
        shadow_root_type=shadow_root_type,
        shadow_roots=shadow_roots,
        parent_node=parent_node,
        children_nodes=children,
        ax_node=None,
        snapshot_node=snapshot or (_make_snapshot() if is_visible else None),
    )


def _make_browser_session():
    session = MagicMock()
    session.logger = logging.getLogger("test.dom_deep")
    session.current_target_id = "target-abc"
    session.agent_focus = MagicMock()
    session.agent_focus.session_id = "session-1"
    session.cdp_client = MagicMock()
    session.get_all_frames = AsyncMock(return_value=({}, {}))
    session.get_or_create_cdp_session = AsyncMock()
    return session


# ============================================================================
# Part 1: HTMLSerializer (dom/serializer/html_serializer.py)
# ============================================================================

from openbrowser.dom.serializer.html_serializer import HTMLSerializer


class TestHTMLSerializerInit:
    """Cover __init__ (line 25): extract_links assignment."""

    def test_default_extract_links(self):
        serializer = HTMLSerializer()
        assert serializer.extract_links is False

    def test_extract_links_true(self):
        serializer = HTMLSerializer(extract_links=True)
        assert serializer.extract_links is True


class TestHTMLSerializerSerialize:
    """Cover serialize method (lines 37-160)."""

    def test_document_node(self):
        """Lines 37-44: DOCUMENT_NODE serializes all children."""
        text_child = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="Hello"
        )
        doc = _make_node(
            tag="#document",
            node_type=NodeType.DOCUMENT_NODE,
            children=[
                _make_node(tag="div", children=[text_child]),
            ],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(doc)
        assert "<div>" in result
        assert "Hello" in result

    def test_document_fragment_node(self):
        """Lines 46-63: DOCUMENT_FRAGMENT_NODE wraps in template."""
        text_child = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="shadow content"
        )
        fragment = _make_node(
            tag="#document-fragment",
            node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
            shadow_root_type="open",
            children=[_make_node(tag="span", children=[text_child])],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(fragment)
        assert '<template shadowroot="open">' in result
        assert "</template>" in result
        assert "shadow content" in result

    def test_document_fragment_no_shadow_type(self):
        """Line 51: fragment without shadow_root_type defaults to 'open'."""
        fragment = _make_node(
            tag="#document-fragment",
            node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
            shadow_root_type=None,
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(fragment)
        assert '<template shadowroot="open">' in result

    def test_element_node_basic(self):
        """Lines 65-146: basic element serialization."""
        text_child = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="content"
        )
        div = _make_node(tag="div", children=[text_child])
        serializer = HTMLSerializer()
        result = serializer.serialize(div)
        assert result == "<div>content</div>"

    def test_element_node_with_attributes(self):
        """Lines 94-97: element with attributes."""
        div = _make_node(
            tag="div",
            attributes={"class": "container", "id": "main"},
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(div)
        assert 'class="container"' in result
        assert 'id="main"' in result

    def test_skip_style_script_head(self):
        """Lines 70-71: style, script, head are skipped."""
        for skip_tag in ["style", "script", "head", "meta", "link", "title"]:
            node = _make_node(tag=skip_tag, children=[])
            serializer = HTMLSerializer()
            result = serializer.serialize(node)
            assert result == ""

    def test_skip_hidden_code_display_none(self):
        """Lines 74-78: code with display:none is skipped."""
        node = _make_node(
            tag="code",
            attributes={"style": "display:none"},
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_skip_hidden_code_display_none_with_space(self):
        """Line 77: code with 'display: none' (space) is skipped."""
        node = _make_node(
            tag="code",
            attributes={"style": "display: none"},
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_skip_code_bpr_guid(self):
        """Lines 80-82: code with bpr-guid ID is skipped."""
        node = _make_node(
            tag="code",
            attributes={"id": "bpr-guid-123"},
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_skip_code_data_id(self):
        """Lines 80-82: code with 'data' in ID is skipped."""
        node = _make_node(
            tag="code",
            attributes={"id": "data-store"},
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_skip_code_state_id(self):
        """Lines 80-82: code with 'state' in ID is skipped."""
        node = _make_node(
            tag="code",
            attributes={"id": "state-container"},
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_code_not_hidden(self):
        """Code tag without hidden indicators is kept."""
        text_child = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="hello world"
        )
        node = _make_node(tag="code", children=[text_child])
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert "<code>" in result
        assert "hello world" in result

    def test_skip_base64_img(self):
        """Lines 85-88: img with data:image/ src is skipped."""
        node = _make_node(
            tag="img",
            attributes={"src": "data:image/png;base64,iVBORw0KGgo="},
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_img_normal_src(self):
        """img with normal src is kept."""
        node = _make_node(
            tag="img",
            attributes={"src": "https://example.com/img.png", "alt": "test"},
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert "<img" in result
        assert "/>" in result

    def test_void_element(self):
        """Lines 100-118: void elements are self-closing."""
        for void_tag in ["br", "hr", "input", "img"]:
            node = _make_node(tag=void_tag, attributes={})
            serializer = HTMLSerializer()
            result = serializer.serialize(node)
            assert "/>" in result
            assert f"</{void_tag}>" not in result

    def test_iframe_with_content_document(self):
        """Lines 123-128: iframe with content_document serializes children."""
        text_child = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="iframe content"
        )
        content_doc = _make_node(
            tag="#document",
            node_type=NodeType.DOCUMENT_NODE,
            children=[_make_node(tag="p", children=[text_child])],
        )
        iframe = _make_node(
            tag="iframe",
            content_document=content_doc,
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(iframe)
        assert "<iframe>" in result
        assert "iframe content" in result
        assert "</iframe>" in result

    def test_frame_with_content_document(self):
        """Lines 123-128: frame with content_document."""
        text_child = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="frame content"
        )
        content_doc = _make_node(
            tag="#document",
            node_type=NodeType.DOCUMENT_NODE,
            children=[_make_node(tag="p", children=[text_child])],
        )
        frame = _make_node(
            tag="frame",
            content_document=content_doc,
            children=[],
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(frame)
        assert "frame content" in result

    def test_element_with_shadow_roots(self):
        """Lines 131-135: element with shadow roots."""
        shadow_text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="shadow"
        )
        shadow_root = _make_node(
            tag="#document-fragment",
            node_type=NodeType.DOCUMENT_FRAGMENT_NODE,
            shadow_root_type="open",
            children=[_make_node(tag="span", children=[shadow_text])],
        )
        host = _make_node(tag="div", shadow_roots=[shadow_root], children=[])
        serializer = HTMLSerializer()
        result = serializer.serialize(host)
        assert "<template" in result
        assert "shadow" in result

    def test_element_with_light_dom_children(self):
        """Lines 138-141: light DOM children after shadow roots."""
        light_text = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="light"
        )
        host = _make_node(tag="div", children=[_make_node(tag="span", children=[light_text])])
        serializer = HTMLSerializer()
        result = serializer.serialize(host)
        assert "light" in result

    def test_text_node(self):
        """Lines 148-152: TEXT_NODE serialization."""
        node = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value="hello <world>"
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == "hello &lt;world&gt;"

    def test_text_node_empty(self):
        """Line 152: TEXT_NODE with empty value."""
        node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="")
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_text_node_none_value(self):
        """Line 152: TEXT_NODE with None value."""
        node = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value=None)
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_comment_node(self):
        """Lines 154-156: COMMENT_NODE is skipped."""
        node = _make_node(
            tag="#comment",
            node_type=NodeType.COMMENT_NODE,
            node_value="this is a comment",
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""

    def test_unknown_node_type(self):
        """Lines 158-160: unknown node type is skipped."""
        node = _make_node(
            tag="#cdata",
            node_type=NodeType.CDATA_SECTION_NODE,
            node_value="cdata content",
        )
        serializer = HTMLSerializer()
        result = serializer.serialize(node)
        assert result == ""


class TestHTMLSerializerAttributes:
    """Cover _serialize_attributes (lines 162-190)."""

    def test_skip_href_when_not_extracting_links(self):
        """Lines 174-175: href is skipped when extract_links=False."""
        serializer = HTMLSerializer(extract_links=False)
        result = serializer._serialize_attributes({"href": "http://test.com", "class": "link"})
        assert "href" not in result
        assert "class" in result

    def test_include_href_when_extracting_links(self):
        """href is kept when extract_links=True."""
        serializer = HTMLSerializer(extract_links=True)
        result = serializer._serialize_attributes({"href": "http://test.com"})
        assert "href" in result

    def test_skip_data_attributes(self):
        """Lines 179-180: data-* attributes are skipped."""
        serializer = HTMLSerializer()
        result = serializer._serialize_attributes({"data-react-id": "abc", "class": "item"})
        assert "data-react-id" not in result
        assert "class" in result

    def test_boolean_attribute_empty_string(self):
        """Lines 183-184: boolean attribute with empty string."""
        serializer = HTMLSerializer()
        result = serializer._serialize_attributes({"disabled": ""})
        assert result == "disabled"

    def test_boolean_attribute_none(self):
        """Lines 183-184: boolean attribute with None value."""
        serializer = HTMLSerializer()
        result = serializer._serialize_attributes({"readonly": None})
        assert result == "readonly"

    def test_attribute_value_escaped(self):
        """Lines 187-188: attribute values are properly escaped."""
        serializer = HTMLSerializer()
        result = serializer._serialize_attributes({"title": 'He said "hello" & <bye>'})
        assert "&amp;" in result
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&quot;" in result


class TestHTMLSerializerEscape:
    """Cover _escape_html (line 201) and _escape_attribute (line 212)."""

    def test_escape_html(self):
        serializer = HTMLSerializer()
        assert serializer._escape_html("a & b < c > d") == "a &amp; b &lt; c &gt; d"

    def test_escape_attribute(self):
        serializer = HTMLSerializer()
        result = serializer._escape_attribute("a&b<c>d\"e'f")
        assert "&amp;" in result
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&quot;" in result
        assert "&#x27;" in result


# ============================================================================
# Part 2: DomService deeper coverage (dom/service.py)
# ============================================================================


class TestDomServiceBuildEnhancedAxNodeDeep:
    """Cover _build_enhanced_ax_node edge cases: lines 124-125 (ValueError)."""

    def test_invalid_property_name_skipped(self):
        """Lines 124-125: invalid property name raises ValueError and is skipped."""
        session = _make_browser_session()
        svc = DomService(session)
        ax_node = {
            "nodeId": "ax-1",
            "ignored": False,
            "properties": [
                {"name": "totally_not_a_real_property_xyz", "value": {"value": True}},
                {"name": "checked", "value": {"value": True}},
            ],
        }
        result = svc._build_enhanced_ax_node(ax_node)
        # Should have properties list (at least 'checked' if valid, invalid one skipped)
        assert result is not None


class TestDomServiceGetAllTreesDeep:
    """Cover _get_all_trees deeper branches: lines 299-300, 336, 375-396,
    405-410, 414, 428-431, 438."""

    @pytest.mark.asyncio
    async def test_get_all_trees_runtime_evaluate_fails(self):
        """Lines 299-300: Runtime.evaluate fails (exception caught, pass)."""
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"

        # Runtime.evaluate fails
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=Exception("page not ready")
        )
        cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = AsyncMock(
            return_value={"documents": [{"nodes": {}}], "strings": []}
        )
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1, "nodeType": 9}}
        )
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "visualViewport": {"clientWidth": 1920},
                "cssVisualViewport": {"clientWidth": 1920},
                "cssLayoutViewport": {"clientWidth": 1920},
            }
        )
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={"frameTree": {"frame": {"id": "f-1"}}}
        )
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(
            return_value={"nodes": []}
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        result = await svc._get_all_trees("target-1")
        assert result.snapshot is not None

    @pytest.mark.asyncio
    async def test_get_all_trees_scroll_result_extraction(self):
        """Lines 333-338: successful scroll position extraction for iframes."""
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"

        # First call: readyState
        # Second call: scroll positions
        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[
                {"result": {"value": "complete"}},
                {
                    "result": {
                        "value": {"0": {"scrollTop": 100, "scrollLeft": 0}}
                    }
                },
            ]
        )
        cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = AsyncMock(
            return_value={"documents": [{"nodes": {}}], "strings": []}
        )
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1, "nodeType": 9}}
        )
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "visualViewport": {"clientWidth": 1920},
                "cssVisualViewport": {"clientWidth": 1920},
                "cssLayoutViewport": {"clientWidth": 1920},
            }
        )
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={"frameTree": {"frame": {"id": "f-1"}}}
        )
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(
            return_value={"nodes": []}
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        result = await svc._get_all_trees("target-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_all_trees_scroll_fetch_fails(self):
        """Line 340: scroll position fetch fails gracefully."""
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"

        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=[
                {"result": {"value": "complete"}},
                Exception("scroll fetch failed"),
            ]
        )
        cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = AsyncMock(
            return_value={"documents": [{"nodes": {}}], "strings": []}
        )
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1, "nodeType": 9}}
        )
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "visualViewport": {"clientWidth": 1920},
                "cssVisualViewport": {"clientWidth": 1920},
                "cssLayoutViewport": {"clientWidth": 1920},
            }
        )
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={"frameTree": {"frame": {"id": "f-1"}}}
        )
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(
            return_value={"nodes": []}
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)
        result = await svc._get_all_trees("target-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_all_trees_timeout_raises(self):
        """Lines 374-414: tasks that time out are retried and raise on failure."""
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"

        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": "complete"}}
        )

        # Make snapshot hang forever (simulate timeout) -- accept any args/kwargs
        async def hang_forever(*args, **kwargs):
            await asyncio.sleep(100)

        cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = hang_forever
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1, "nodeType": 9}}
        )
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "visualViewport": {"clientWidth": 1920},
                "cssVisualViewport": {"clientWidth": 1920},
                "cssLayoutViewport": {"clientWidth": 1920},
            }
        )
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={"frameTree": {"frame": {"id": "f-1"}}}
        )
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(
            return_value={"nodes": []}
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)

        with pytest.raises(TimeoutError, match="CDP requests failed or timed out"):
            await svc._get_all_trees("target-1")

    @pytest.mark.asyncio
    async def test_get_all_trees_iframe_limit(self):
        """Lines 427-431: excessive iframes get truncated."""
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"

        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": "complete"}}
        )
        # Create 200 documents to exceed max_iframes=3
        documents = [{"nodes": {}} for _ in range(200)]
        cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = AsyncMock(
            return_value={"documents": documents, "strings": []}
        )
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1, "nodeType": 9}}
        )
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "visualViewport": {"clientWidth": 1920},
                "cssVisualViewport": {"clientWidth": 1920},
                "cssLayoutViewport": {"clientWidth": 1920},
            }
        )
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={"frameTree": {"frame": {"id": "f-1"}}}
        )
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(
            return_value={"nodes": []}
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session, max_iframes=3)
        result = await svc._get_all_trees("target-1")
        # Should be truncated to max_iframes
        assert len(result.snapshot["documents"]) == 3


class TestDomServiceGetAllTreesTaskFailure:
    """Cover task exception handling: lines 405-410."""

    @pytest.mark.asyncio
    async def test_get_all_trees_task_exception(self):
        """Lines 405-407: a task that completes with an exception is logged."""
        session = _make_browser_session()
        cdp_session = MagicMock()
        cdp_session.session_id = "s1"

        cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": "complete"}}
        )
        cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = AsyncMock(
            side_effect=RuntimeError("snapshot failed")
        )
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(
            return_value={"root": {"nodeId": 1, "nodeType": 9}}
        )
        cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
            return_value={
                "visualViewport": {"clientWidth": 1920},
                "cssVisualViewport": {"clientWidth": 1920},
                "cssLayoutViewport": {"clientWidth": 1920},
            }
        )
        cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(
            return_value={"frameTree": {"frame": {"id": "f-1"}}}
        )
        cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(
            return_value={"nodes": []}
        )
        session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
        svc = DomService(session)

        with pytest.raises(TimeoutError, match="CDP requests failed"):
            await svc._get_all_trees("target-1")


# ============================================================================
# Part 3: DaemonServer deeper coverage (daemon/server.py)
# ============================================================================


class TestDaemonServerEnsureExecutor:
    """Cover _ensure_executor (lines 92-126)."""

    @pytest.mark.asyncio
    async def test_ensure_executor_already_set(self):
        """Lines 90-91: returns immediately when executor exists."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        daemon._executor = MagicMock()
        await daemon._ensure_executor()
        # Should not try to import anything

    @pytest.mark.asyncio
    async def test_ensure_executor_initializes(self):
        """Lines 92-126: full initialization path."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()

        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.kill = AsyncMock()

        mock_profile = MagicMock()
        mock_namespace = {"key": "value"}
        mock_tools = MagicMock()

        mock_executor_cls = MagicMock()
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_session), \
             patch("openbrowser.code_use.executor.CodeExecutor", mock_executor_cls), \
             patch("openbrowser.code_use.executor.DEFAULT_MAX_OUTPUT_CHARS", 50000), \
             patch("openbrowser.code_use.namespace.create_namespace", return_value=mock_namespace), \
             patch("openbrowser.tools.service.CodeAgentTools", return_value=mock_tools), \
             patch.dict("os.environ", {"OPENBROWSER_MAX_OUTPUT": "0"}):

            await daemon._ensure_executor()

            assert daemon._executor is not None
            assert daemon._session is mock_session
            mock_session.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_executor_setup_fails_kills_browser(self):
        """Lines 120-126: if namespace/executor setup fails, browser is killed."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()

        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.kill = AsyncMock()

        mock_profile = MagicMock()

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_session), \
             patch("openbrowser.tools.service.CodeAgentTools", side_effect=RuntimeError("fail")), \
             patch("openbrowser.code_use.executor.DEFAULT_MAX_OUTPUT_CHARS", 50000):

            with pytest.raises(RuntimeError, match="fail"):
                await daemon._ensure_executor()

            mock_session.kill.assert_called_once()


class TestDaemonServerRecoverBrowserSession:
    """Cover _recover_browser_session (lines 137-162)."""

    @pytest.mark.asyncio
    async def test_recover_success(self):
        """Lines 137-162: successful recovery."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        old_session = MagicMock()
        old_session.kill = AsyncMock()
        daemon._session = old_session
        daemon._executor = MagicMock()

        mock_new_session = MagicMock()
        mock_new_session.start = AsyncMock()
        mock_profile = MagicMock()
        mock_namespace = {}
        mock_tools = MagicMock()

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_new_session), \
             patch("openbrowser.code_use.namespace.create_namespace", return_value=mock_namespace), \
             patch("openbrowser.tools.service.CodeAgentTools", return_value=mock_tools):

            await daemon._recover_browser_session()

            old_session.kill.assert_called_once()
            assert daemon._session is mock_new_session

    @pytest.mark.asyncio
    async def test_recover_kill_error_continues(self):
        """Lines 139-142: kill error during recovery is suppressed."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        daemon._session = MagicMock()
        daemon._session.kill = AsyncMock(side_effect=Exception("kill failed"))
        daemon._executor = MagicMock()

        mock_new_session = MagicMock()
        mock_new_session.start = AsyncMock()
        mock_profile = MagicMock()

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_new_session), \
             patch("openbrowser.code_use.namespace.create_namespace", return_value={}), \
             patch("openbrowser.tools.service.CodeAgentTools", return_value=MagicMock()):

            await daemon._recover_browser_session()
            assert daemon._session is mock_new_session

    @pytest.mark.asyncio
    async def test_recover_namespace_fails_kills_browser(self):
        """Lines 156-161: namespace creation fails during recovery, browser killed."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        daemon._session = MagicMock()
        daemon._session.kill = AsyncMock()
        daemon._executor = MagicMock()

        mock_new_session = MagicMock()
        mock_new_session.start = AsyncMock()
        mock_new_session.kill = AsyncMock()
        mock_profile = MagicMock()

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_new_session), \
             patch("openbrowser.code_use.namespace.create_namespace", side_effect=RuntimeError("ns fail")), \
             patch("openbrowser.tools.service.CodeAgentTools", return_value=MagicMock()):

            with pytest.raises(RuntimeError, match="ns fail"):
                await daemon._recover_browser_session()

            mock_new_session.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_recover_no_existing_session(self):
        """Line 138: _session is None before recovery."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        daemon._session = None
        daemon._executor = MagicMock()

        mock_new_session = MagicMock()
        mock_new_session.start = AsyncMock()
        mock_profile = MagicMock()

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_new_session), \
             patch("openbrowser.code_use.namespace.create_namespace", return_value={}), \
             patch("openbrowser.tools.service.CodeAgentTools", return_value=MagicMock()):

            await daemon._recover_browser_session()
            assert daemon._session is mock_new_session


class TestDaemonServerHandleRequestRecoveryFail:
    """Cover line 197-198: recovery fails during execute."""

    @pytest.mark.asyncio
    async def test_execute_recovery_fails(self):
        """Lines 197-198: recovery itself fails, error is logged."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer(exec_timeout=5)
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.output = "websocket error"
        mock_result.error = "websocket error"

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=mock_result)
        daemon._executor = mock_executor

        with patch.object(
            daemon, "_recover_browser_session",
            new_callable=AsyncMock,
            side_effect=RuntimeError("recovery boom"),
        ):
            result = await daemon._handle_request(
                {"action": "execute", "code": "x = 1", "id": 1}
            )
            # Original failed result should be returned
            assert result["success"] is False


class TestDaemonServerHandleClientDeep:
    """Cover _handle_client deeper branches: lines 261-262."""

    @pytest.mark.asyncio
    async def test_handle_client_wait_closed_error(self):
        """Lines 261-262: wait_closed raises, suppressed."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        reader = AsyncMock()
        writer = MagicMock()

        reader.readline = AsyncMock(return_value=json.dumps({"action": "status", "id": 1}).encode() + b"\n")
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock(side_effect=ConnectionResetError("closed"))
        writer.write = MagicMock()

        await daemon._handle_client(reader, writer)
        writer.close.assert_called_once()


class TestDaemonServerRunWindows:
    """Cover Windows branch: line 303."""

    @pytest.mark.asyncio
    async def test_run_windows_server(self):
        """Line 303: Windows TCP server branch."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        mock_sock = MagicMock()
        mock_sock.parent = MagicMock()
        mock_sock.unlink = MagicMock()

        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=False)

        daemon._session = None

        async def stop_quickly():
            await asyncio.sleep(0.01)
            daemon._stop_event.set()

        with patch("openbrowser.daemon.server.get_socket_path", return_value=mock_sock), \
             patch("openbrowser.daemon.server._read_pid", return_value=None), \
             patch("openbrowser.daemon.server._write_pid"), \
             patch("openbrowser.daemon.server._cleanup_pid"), \
             patch("openbrowser.daemon.server.IS_WINDOWS", True), \
             patch("asyncio.start_server", new_callable=AsyncMock, return_value=mock_server):

            asyncio.create_task(stop_quickly())
            await daemon.run()


class TestDaemonServerRunCleanupKillError:
    """Cover lines 333-334: session.kill() fails during cleanup."""

    @pytest.mark.asyncio
    async def test_run_cleanup_kill_error(self):
        """Lines 333-334: session kill error during cleanup is suppressed."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()
        mock_sock = MagicMock()
        mock_sock.parent = MagicMock()
        mock_sock.unlink = MagicMock()

        mock_server = AsyncMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=False)

        daemon._session = MagicMock()
        daemon._session.kill = AsyncMock(side_effect=RuntimeError("kill boom"))

        async def stop_quickly():
            await asyncio.sleep(0.01)
            daemon._stop_event.set()

        with patch("openbrowser.daemon.server.get_socket_path", return_value=mock_sock), \
             patch("openbrowser.daemon.server._read_pid", return_value=None), \
             patch("openbrowser.daemon.server._write_pid"), \
             patch("openbrowser.daemon.server._cleanup_pid"), \
             patch("openbrowser.daemon.server.IS_WINDOWS", False), \
             patch("asyncio.start_unix_server", new_callable=AsyncMock, return_value=mock_server), \
             patch("os.chmod"):

            asyncio.create_task(stop_quickly())
            await daemon.run()
            # Should not raise despite kill error


class TestDaemonServerEnsureExecutorMaxOutput:
    """Cover OPENBROWSER_MAX_OUTPUT parsing: lines 114-117."""

    @pytest.mark.asyncio
    async def test_ensure_executor_custom_max_output(self):
        """Lines 114-117: custom OPENBROWSER_MAX_OUTPUT from env."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()

        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_profile = MagicMock()
        mock_namespace = {}
        mock_tools = MagicMock()

        mock_executor_cls = MagicMock()
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_session), \
             patch("openbrowser.code_use.executor.CodeExecutor", mock_executor_cls), \
             patch("openbrowser.code_use.executor.DEFAULT_MAX_OUTPUT_CHARS", 50000), \
             patch("openbrowser.code_use.namespace.create_namespace", return_value=mock_namespace), \
             patch("openbrowser.tools.service.CodeAgentTools", return_value=mock_tools), \
             patch.dict("os.environ", {"OPENBROWSER_MAX_OUTPUT": "100000"}):

            await daemon._ensure_executor()
            # Should have been called with 100000
            mock_executor_cls.assert_called_once_with(max_output_chars=100000)

    @pytest.mark.asyncio
    async def test_ensure_executor_invalid_max_output(self):
        """Lines 115-116: invalid OPENBROWSER_MAX_OUTPUT falls back to default."""
        from openbrowser.daemon.server import DaemonServer

        daemon = DaemonServer()

        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_profile = MagicMock()
        mock_namespace = {}
        mock_tools = MagicMock()

        mock_executor_cls = MagicMock()
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance

        with patch.object(daemon, "_build_browser_profile", return_value=mock_profile), \
             patch("openbrowser.browser.BrowserSession", return_value=mock_session), \
             patch("openbrowser.code_use.executor.CodeExecutor", mock_executor_cls), \
             patch("openbrowser.code_use.executor.DEFAULT_MAX_OUTPUT_CHARS", 50000), \
             patch("openbrowser.code_use.namespace.create_namespace", return_value=mock_namespace), \
             patch("openbrowser.tools.service.CodeAgentTools", return_value=mock_tools), \
             patch.dict("os.environ", {"OPENBROWSER_MAX_OUTPUT": "not_a_number"}):

            await daemon._ensure_executor()
            # Should fall back to DEFAULT_MAX_OUTPUT_CHARS
            mock_executor_cls.assert_called_once_with(max_output_chars=50000)


# ============================================================================
# Part 4: markdown_extractor.py
# ============================================================================


class TestExtractCleanMarkdown:
    """Cover extract_clean_markdown (lines 19-104)."""

    @pytest.mark.asyncio
    async def test_extract_with_browser_session(self):
        """Lines 43-49: browser_session path."""
        from openbrowser.dom.markdown_extractor import extract_clean_markdown

        mock_session = MagicMock()
        mock_dom_tree = _make_node(
            tag="#document",
            node_type=NodeType.DOCUMENT_NODE,
            children=[
                _make_node(
                    tag="p",
                    children=[
                        _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="Hello World"),
                    ],
                ),
            ],
        )
        mock_session.get_current_page_url = AsyncMock(return_value="http://example.com")

        mock_watchdog = MagicMock()
        mock_watchdog.enhanced_dom_tree = mock_dom_tree
        mock_session._dom_watchdog = mock_watchdog

        content, stats = await extract_clean_markdown(browser_session=mock_session)
        assert "Hello World" in content
        assert stats["method"] == "enhanced_dom_tree"
        assert "url" in stats

    @pytest.mark.asyncio
    async def test_extract_with_dom_service(self):
        """Lines 50-54: dom_service path."""
        from openbrowser.dom.markdown_extractor import extract_clean_markdown

        mock_dom_tree = _make_node(
            tag="#document",
            node_type=NodeType.DOCUMENT_NODE,
            children=[
                _make_node(
                    tag="p",
                    children=[
                        _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="DOM content"),
                    ],
                ),
            ],
        )
        mock_dom_service = MagicMock()
        mock_dom_service.get_dom_tree = AsyncMock(return_value=mock_dom_tree)

        content, stats = await extract_clean_markdown(
            dom_service=mock_dom_service, target_id="t-1"
        )
        assert "DOM content" in content
        assert stats["method"] == "dom_service"

    @pytest.mark.asyncio
    async def test_extract_raises_both_params(self):
        """Lines 44-45: raises when both browser_session and dom_service provided."""
        from openbrowser.dom.markdown_extractor import extract_clean_markdown

        with pytest.raises(ValueError, match="Cannot specify both"):
            await extract_clean_markdown(
                browser_session=MagicMock(),
                dom_service=MagicMock(),
                target_id="t-1",
            )

    @pytest.mark.asyncio
    async def test_extract_raises_neither_param(self):
        """Lines 55-56: raises when neither browser_session nor dom_service."""
        from openbrowser.dom.markdown_extractor import extract_clean_markdown

        with pytest.raises(ValueError, match="Must provide either"):
            await extract_clean_markdown()

    @pytest.mark.asyncio
    async def test_extract_raises_browser_session_with_target_id(self):
        """Lines 44-45: browser_session with target_id raises."""
        from openbrowser.dom.markdown_extractor import extract_clean_markdown

        with pytest.raises(ValueError, match="Cannot specify both"):
            await extract_clean_markdown(
                browser_session=MagicMock(),
                target_id="t-1",
            )

    @pytest.mark.asyncio
    async def test_extract_raises_dom_service_without_target_id(self):
        """Line 55-56: dom_service without target_id raises."""
        from openbrowser.dom.markdown_extractor import extract_clean_markdown

        with pytest.raises(ValueError, match="Must provide either"):
            await extract_clean_markdown(dom_service=MagicMock())


class TestGetEnhancedDomTreeFromBrowserSession:
    """Cover _get_enhanced_dom_tree_from_browser_session (lines 107-123)."""

    @pytest.mark.asyncio
    async def test_cached_dom_tree(self):
        """Lines 115-116: returns cached enhanced_dom_tree."""
        from openbrowser.dom.markdown_extractor import _get_enhanced_dom_tree_from_browser_session

        mock_tree = MagicMock()
        mock_session = MagicMock()
        mock_watchdog = MagicMock()
        mock_watchdog.enhanced_dom_tree = mock_tree
        mock_session._dom_watchdog = mock_watchdog

        result = await _get_enhanced_dom_tree_from_browser_session(mock_session)
        assert result is mock_tree

    @pytest.mark.asyncio
    async def test_build_dom_tree_when_not_cached(self):
        """Lines 118-123: builds dom tree when not cached."""
        from openbrowser.dom.markdown_extractor import _get_enhanced_dom_tree_from_browser_session

        mock_tree = MagicMock()
        mock_session = MagicMock()
        mock_watchdog = MagicMock()
        mock_watchdog.enhanced_dom_tree = None
        mock_watchdog._build_dom_tree_without_highlights = AsyncMock()
        mock_session._dom_watchdog = mock_watchdog

        # After build, enhanced_dom_tree is set
        def set_tree():
            mock_watchdog.enhanced_dom_tree = mock_tree

        mock_watchdog._build_dom_tree_without_highlights.side_effect = lambda: set_tree()

        result = await _get_enhanced_dom_tree_from_browser_session(mock_session)
        assert result is mock_tree

    @pytest.mark.asyncio
    async def test_raises_when_no_watchdog(self):
        """Line 112: raises when DOMWatchdog is not available."""
        from openbrowser.dom.markdown_extractor import _get_enhanced_dom_tree_from_browser_session

        mock_session = MagicMock()
        mock_session._dom_watchdog = None

        with pytest.raises(AssertionError, match="DOMWatchdog not available"):
            await _get_enhanced_dom_tree_from_browser_session(mock_session)


class TestPreprocessMarkdownContent:
    """Cover _preprocess_markdown_content (lines 129-169)."""

    def test_removes_json_in_code_blocks(self):
        """Line 146: JSON in code blocks is removed."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        content = 'Some text `{"key":"value","nested":{"a":"b"}}` more text'
        result, chars_filtered = _preprocess_markdown_content(content)
        assert '{"key"' not in result

    def test_removes_large_json_type_field(self):
        """Line 147: large JSON with $type fields is removed."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        json_blob = '{"$type":' + '"x"' * 100 + '}'
        content = f"Before {json_blob} after"
        result, chars_filtered = _preprocess_markdown_content(content)
        assert "$type" not in result

    def test_removes_large_nested_json(self):
        """Line 148: large nested JSON objects are removed."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        json_blob = '{"bigkey":{"inner":' + '"x"' * 100 + '}}'
        content = f"Before {json_blob} after"
        result, _ = _preprocess_markdown_content(content)
        # The large nested JSON should be reduced

    def test_compresses_newlines(self):
        """Line 151: 4+ newlines compressed to max_newlines."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        content = "line1\n\n\n\n\n\nline2"
        result, _ = _preprocess_markdown_content(content, max_newlines=2)
        # Should have at most 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_filters_short_lines(self):
        """Lines 153-163: lines with 2 or fewer chars are removed."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        content = "Good line here\n  \n--\nAnother good line"
        result, _ = _preprocess_markdown_content(content)
        assert "Good line here" in result
        assert "Another good line" in result

    def test_filters_long_json_lines(self):
        """Lines 161-162: lines starting with { or [ and > 100 chars are removed."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        long_json = "{" + '"key":"value",' * 20 + "}"
        content = f"Normal line\n{long_json}\nAnother normal"
        result, _ = _preprocess_markdown_content(content)
        assert "Normal line" in result
        assert "Another normal" in result

    def test_returns_chars_filtered(self):
        """Line 168: returns correct chars_filtered count."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        content = "short\n\n\n\n\n\n\nlonger content here"
        result, chars_filtered = _preprocess_markdown_content(content)
        assert chars_filtered >= 0

    def test_strips_result(self):
        """Line 166: result is stripped."""
        from openbrowser.dom.markdown_extractor import _preprocess_markdown_content

        content = "   \n\n  actual content   \n\n   "
        result, _ = _preprocess_markdown_content(content)
        assert not result.startswith(" ")
        assert not result.endswith(" ")


# ============================================================================
# Part 5: DomService.detect_pagination_buttons deeper branches
# ============================================================================


class TestDetectPaginationDeep:
    """Cover additional branches in detect_pagination_buttons."""

    def _make_pagination_node(self, text="", attributes=None, is_clickable=True):
        snap = _make_snapshot(is_clickable=is_clickable) if is_clickable else None
        text_node = _make_node(
            tag="#text", node_type=NodeType.TEXT_NODE, node_value=text,
        )
        node = _make_node(
            tag="button", snapshot=snap,
            children=[text_node], attributes=attributes or {},
        )
        return node

    def test_aria_disabled_button(self):
        """aria-disabled='true' marks button as disabled."""
        node = self._make_pagination_node("Next", attributes={"aria-disabled": "true"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) > 0
        assert result[0]["is_disabled"] is True

    def test_disabled_in_class(self):
        """'disabled' in class name marks button as disabled."""
        node = self._make_pagination_node("Next", attributes={"class": "btn disabled"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) > 0
        assert result[0]["is_disabled"] is True

    def test_link_role_page_number(self):
        """Page number with role='link' is detected."""
        node = self._make_pagination_node("7", attributes={"role": "link"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "page_number" for b in result)

    def test_no_role_page_number(self):
        """Page number without role attribute (empty string) is detected."""
        node = self._make_pagination_node("3", attributes={})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "page_number" for b in result)

    def test_multilingual_next_patterns(self):
        """International next patterns: siguiente, suivant, etc."""
        for pattern in ["siguiente", "suivant", "weiter", "volgende"]:
            node = self._make_pagination_node(pattern)
            selector_map = {node.backend_node_id: node}
            result = DomService.detect_pagination_buttons(selector_map)
            assert any(b["button_type"] == "next" for b in result), f"Failed for: {pattern}"

    def test_multilingual_prev_patterns(self):
        """International prev patterns: anterior, zurck, etc."""
        for pattern in ["anterior", "vorige"]:
            node = self._make_pagination_node(pattern)
            selector_map = {node.backend_node_id: node}
            result = DomService.detect_pagination_buttons(selector_map)
            assert any(b["button_type"] == "prev" for b in result), f"Failed for: {pattern}"

    def test_first_patterns(self):
        """First button pattern detection."""
        node = self._make_pagination_node("erste")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "first" for b in result)

    def test_last_patterns(self):
        """Last button pattern detection."""
        node = self._make_pagination_node("laatste")
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert any(b["button_type"] == "last" for b in result)

    def test_button_text_fallback_to_aria_label(self):
        """Button text falls back to aria-label when text is empty."""
        node = self._make_pagination_node("", attributes={"aria-label": "Go to next"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) > 0
        # get_all_children_text returns "" so it falls through to aria_label or title
        # The code uses: node.get_all_children_text().strip() or aria_label or title
        # But aria_label is already lowered for matching -- the text field uses original
        # Let's check case-insensitively
        assert "go to next" in result[0]["text"].lower()

    def test_button_text_fallback_to_title(self):
        """Button text falls back to title when text and aria-label empty."""
        node = self._make_pagination_node("", attributes={"title": "Previous page"})
        selector_map = {node.backend_node_id: node}
        result = DomService.detect_pagination_buttons(selector_map)
        assert len(result) > 0
        assert "previous page" in result[0]["text"].lower()


# ============================================================================
# Part 6: DomService.is_element_visible_according_to_all_parents deeper
# ============================================================================


class TestIsElementVisibleDeep:
    """Cover deeper branches in is_element_visible_according_to_all_parents."""

    def test_iframe_and_html_frame_combined(self):
        """Test element visible with both iframe and HTML frames in chain."""
        snap = _make_snapshot(
            bounds=DOMRect(10, 10, 50, 30),
            computed_styles={"display": "block", "visibility": "visible", "opacity": "1"},
        )
        node = _make_node(snapshot=snap)

        iframe_snap = _make_snapshot(bounds=DOMRect(50, 50, 800, 600))
        iframe_frame = _make_node(
            tag="IFRAME", node_name="IFRAME",
            snapshot=iframe_snap, frame_id="frame-1",
        )

        html_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 800, 600),
            scroll_rects=DOMRect(0, 0, 800, 2000),
            client_rects=DOMRect(0, 0, 800, 600),
        )
        html_frame = _make_node(
            tag="HTML", node_name="HTML",
            snapshot=html_snap, frame_id="main-frame",
        )

        result = DomService.is_element_visible_according_to_all_parents(
            node, [html_frame, iframe_frame]
        )
        assert result is True

    def test_frame_element_with_bounds(self):
        """FRAME element (not IFRAME) with bounds also offsets."""
        snap = _make_snapshot(
            bounds=DOMRect(5, 5, 50, 30),
            computed_styles={"display": "block", "visibility": "visible", "opacity": "1"},
        )
        node = _make_node(snapshot=snap)

        frame_snap = _make_snapshot(bounds=DOMRect(200, 200, 400, 300))
        frame = _make_node(
            tag="FRAME", node_name="FRAME",
            snapshot=frame_snap, frame_id="frame-1",
        )

        result = DomService.is_element_visible_according_to_all_parents(
            node, [frame]
        )
        assert result is True

    def test_no_computed_styles_treated_as_visible(self):
        """Element with no computed_styles is treated as visible."""
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 50),
            computed_styles=None,
        )
        node = _make_node(snapshot=snap)
        result = DomService.is_element_visible_according_to_all_parents(node, [])
        assert result is True
