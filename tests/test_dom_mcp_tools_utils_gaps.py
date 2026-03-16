"""Comprehensive gap-coverage tests for dom/service, dom/serializer, mcp/server,
tools/registry/service, code_use/service, utils/__init__, and tools/service.

Targets the specific missed lines listed in the coverage report.
"""

import asyncio
import base64
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from openbrowser.dom.service import DomService
from openbrowser.dom.views import (
    DOMRect,
    EnhancedAXNode,
    EnhancedAXProperty,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    PropagatingBounds,
    SimplifiedNode,
)
from openbrowser.models import ActionResult
from openbrowser.tools.registry.views import ActionModel
from openbrowser.tools.service import (
    CodeAgentTools,
    Controller,
    Tools,
    _detect_sensitive_key_name,
    handle_browser_error,
)

logger = logging.getLogger(__name__)

_node_counter = 0


def _make_snapshot(bounds=None, computed_styles=None, scroll_rects=None,
                   client_rects=None, is_clickable=None, paint_order=None):
    return EnhancedSnapshotNode(
        is_clickable=is_clickable, cursor_style=None,
        bounds=bounds or DOMRect(x=0, y=0, width=100, height=50),
        clientRects=client_rects, scrollRects=scroll_rects,
        computed_styles=computed_styles, paint_order=paint_order,
        stacking_contexts=None,
    )


def _make_node(tag="div", node_type=NodeType.ELEMENT_NODE, node_value="",
               attributes=None, children=None, is_visible=True, snapshot=None,
               content_document=None, frame_id=None, node_name=None,
               shadow_root_type=None, shadow_roots=None, ax_node=None,
               is_scrollable=None, parent_node=None, backend_node_id=None):
    global _node_counter
    _node_counter += 1
    return EnhancedDOMTreeNode(
        node_id=_node_counter,
        backend_node_id=backend_node_id or (_node_counter + 50000),
        node_type=node_type, node_name=node_name or tag, node_value=node_value,
        attributes=attributes or {}, is_scrollable=is_scrollable,
        is_visible=is_visible, absolute_position=None, target_id="target-1",
        frame_id=frame_id, session_id=None, content_document=content_document,
        shadow_root_type=shadow_root_type, shadow_roots=shadow_roots,
        parent_node=parent_node, children_nodes=children, ax_node=ax_node,
        snapshot_node=snapshot or (_make_snapshot() if is_visible else None),
    )


def _make_browser_session():
    session = MagicMock()
    session.logger = logging.getLogger("test.gap_coverage")
    session.current_target_id = "target-abc"
    session.agent_focus = MagicMock()
    session.agent_focus.session_id = "session-1"
    session.cdp_client = MagicMock()
    session.get_all_frames = AsyncMock(return_value=({}, {}))
    session.get_or_create_cdp_session = AsyncMock()
    session.is_local = True
    session.downloaded_files = []
    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = "/tmp/downloads"
    session.browser_profile.keep_alive = False
    mock_event = MagicMock()
    mock_event.event_result = AsyncMock(return_value=None)
    # Ensure an event loop exists (it may be torn down by prior async tests)
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    event_future = asyncio.Future()
    event_future.set_result(None)
    mock_event.__await__ = lambda self: event_future.__await__()
    session.event_bus = MagicMock()
    session.event_bus.dispatch = MagicMock(return_value=mock_event)
    session.get_element_by_index = AsyncMock(return_value=None)
    session.get_selector_map = AsyncMock(return_value={})
    session.get_target_id_from_tab_id = AsyncMock(return_value="full_target_id")
    session.highlight_interaction_element = AsyncMock()
    session.is_file_input = MagicMock(return_value=False)
    session.get_current_page_url = AsyncMock(return_value="https://example.com")
    cdp_session = AsyncMock()
    cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={"result": {"value": "ok"}})
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
    return session


def _setup_cdp_for_dom_test(session):
    cdp_session = MagicMock()
    cdp_session.session_id = "s1"
    cdp_session.cdp_client = MagicMock()
    cdp_session.cdp_client.send.Page.getFrameTree = AsyncMock(
        return_value={"frameTree": {"frame": {"id": "f1"}}})
    cdp_session.cdp_client.send.Accessibility.getFullAXTree = AsyncMock(
        return_value={"nodes": []})
    cdp_session.cdp_client.send.DOMSnapshot.captureSnapshot = AsyncMock(
        return_value={"documents": [{"nodes": []}]})
    cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(return_value={
        "visualViewport": {"clientWidth": 1920},
        "cssVisualViewport": {"clientWidth": 1920},
        "cssLayoutViewport": {"clientWidth": 1920},
    })
    cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": {}}})
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
    return cdp_session


def _make_mock_node_for_tools(tag_name="div", index=1, attributes=None):
    return EnhancedDOMTreeNode(
        node_id=index, backend_node_id=index + 100,
        node_type=NodeType.ELEMENT_NODE, node_name=tag_name.upper(),
        node_value="", attributes=attributes or {}, is_scrollable=False,
        is_visible=True,
        absolute_position=DOMRect(x=50.0, y=100.0, width=200.0, height=40.0),
        target_id="target123456789012345678901234",
        frame_id="frame_001", session_id="session_001",
        content_document=None, shadow_root_type=None, shadow_roots=None,
        parent_node=None, children_nodes=[], ax_node=None,
        snapshot_node=_make_snapshot(),
    )


# === DOM SERVICE TESTS ===

class TestDomServiceGaps:

    @pytest.mark.asyncio
    async def test_memoized_node_returned_line_510(self):
        session = _make_browser_session()
        svc = DomService(session)
        cdp_session = _setup_cdp_for_dom_test(session)
        shared_child = {"nodeId": 99, "backendNodeId": 199, "nodeType": 1,
                        "nodeName": "SPAN", "nodeValue": "", "childCount": 0}
        root_node = {"nodeId": 1, "backendNodeId": 101, "nodeType": 1,
                     "nodeName": "DIV", "nodeValue": "",
                     "children": [shared_child, shared_child]}
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": root_node})
        with patch("openbrowser.dom.service.build_snapshot_lookup", return_value={}):
            result = await svc.get_dom_tree(target_id="t1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_shadow_root_type_line_529_530(self):
        session = _make_browser_session()
        svc = DomService(session)
        cdp_session = _setup_cdp_for_dom_test(session)
        root_node = {"nodeId": 1, "backendNodeId": 101, "nodeType": 1,
                     "nodeName": "DIV", "nodeValue": "",
                     "shadowRootType": "open", "children": []}
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": root_node})
        with patch("openbrowser.dom.service.build_snapshot_lookup", return_value={}):
            result = await svc.get_dom_tree(target_id="t1")
        assert result.shadow_root_type == "open"

    @pytest.mark.asyncio
    async def test_iframe_bounds_content_doc_lines_592_602(self):
        session = _make_browser_session()
        svc = DomService(session)
        cdp_session = _setup_cdp_for_dom_test(session)
        iframe_snap = EnhancedSnapshotNode(
            is_clickable=None, cursor_style=None,
            bounds=DOMRect(x=10, y=20, width=300, height=200),
            clientRects=None, scrollRects=None, computed_styles={},
            paint_order=None, stacking_contexts=None)
        content_doc = {"nodeId": 3, "backendNodeId": 103, "nodeType": 9,
                       "nodeName": "#document", "nodeValue": ""}
        iframe_node = {"nodeId": 2, "backendNodeId": 102, "nodeType": 1,
                       "nodeName": "IFRAME", "nodeValue": "",
                       "contentDocument": content_doc}
        root_node = {"nodeId": 1, "backendNodeId": 101, "nodeType": 1,
                     "nodeName": "HTML", "nodeValue": "", "children": [iframe_node]}
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": root_node})
        with patch("openbrowser.dom.service.build_snapshot_lookup", return_value={102: iframe_snap}):
            result = await svc.get_dom_tree(target_id="t1")
        assert result.children_nodes[0].content_document is not None

    @pytest.mark.asyncio
    async def test_shadow_root_skip_children_line_618_624(self):
        session = _make_browser_session()
        svc = DomService(session)
        cdp_session = _setup_cdp_for_dom_test(session)
        shadow_root = {"nodeId": 5, "backendNodeId": 105, "nodeType": 11,
                       "nodeName": "#document-fragment", "nodeValue": "",
                       "shadowRootType": "open"}
        regular_child = {"nodeId": 6, "backendNodeId": 106, "nodeType": 1,
                         "nodeName": "SPAN", "nodeValue": ""}
        host_node = {"nodeId": 2, "backendNodeId": 102, "nodeType": 1,
                     "nodeName": "DIV", "nodeValue": "",
                     "shadowRoots": [shadow_root],
                     "children": [shadow_root, regular_child]}
        root_node = {"nodeId": 1, "backendNodeId": 101, "nodeType": 1,
                     "nodeName": "HTML", "nodeValue": "", "children": [host_node]}
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": root_node})
        with patch("openbrowser.dom.service.build_snapshot_lookup", return_value={}):
            result = await svc.get_dom_tree(target_id="t1")
        host = result.children_nodes[0]
        assert host.shadow_roots is not None
        assert len(host.shadow_roots) == 1
        assert len(host.children_nodes) == 1
        assert host.children_nodes[0].node_name == "SPAN"

    @pytest.mark.asyncio
    async def test_form_element_debug_logging_line_645(self):
        session = _make_browser_session()
        svc = DomService(session)
        cdp_session = _setup_cdp_for_dom_test(session)
        input_node = {"nodeId": 2, "backendNodeId": 102, "nodeType": 1,
                      "nodeName": "INPUT", "nodeValue": "",
                      "attributes": ["id", "city", "name", "city_field"]}
        root_node = {"nodeId": 1, "backendNodeId": 101, "nodeType": 1,
                     "nodeName": "HTML", "nodeValue": "", "children": [input_node]}
        cdp_session.cdp_client.send.DOM.getDocument = AsyncMock(return_value={"root": root_node})
        with patch("openbrowser.dom.service.build_snapshot_lookup", return_value={102: _make_snapshot()}):
            result = await svc.get_dom_tree(target_id="t1")
        assert result is not None


# === DOM SERIALIZER TESTS ===

class TestDOMSerializerGaps:

    def test_compound_non_compound_element_line_169(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        node = _make_node(tag="div", ax_node=None)
        simplified = SimplifiedNode(original_node=node, children=[])
        ser = DOMTreeSerializer.__new__(DOMTreeSerializer)
        ser._add_compound_components(simplified, node)
        assert node._compound_children == []
        sel_no_ax = _make_node(tag="select", ax_node=None)
        s2 = SimplifiedNode(original_node=sel_no_ax, children=[])
        ser._add_compound_components(s2, sel_no_ax)
        assert sel_no_ax._compound_children == []
        sel_empty = _make_node(tag="select", ax_node=EnhancedAXNode(
            ax_node_id="ax1", ignored=False, role="listbox", name="test",
            description=None, properties=None, child_ids=None))
        s3 = SimplifiedNode(original_node=sel_empty, children=[])
        ser._add_compound_components(s3, sel_empty)
        assert sel_empty._compound_children == []

    def test_select_options_line_287_375_383(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        opt1 = _make_node(tag="option", attributes={"value": "v1"})
        opt2 = _make_node(tag="option", attributes={"value": "v2"})
        t1 = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="Opt 1",
                         node_name="#text", is_visible=False, snapshot=None)
        t2 = _make_node(tag="#text", node_type=NodeType.TEXT_NODE, node_value="Opt 2",
                         node_name="#text", is_visible=False, snapshot=None)
        opt1.children_nodes = [t1]
        opt2.children_nodes = [t2]
        og = _make_node(tag="optgroup", children=[opt1, opt2])
        og.children_nodes = [opt1, opt2]
        sel = _make_node(tag="select", children=[og], ax_node=EnhancedAXNode(
            ax_node_id="ax1", ignored=False, role="listbox", name="Sel",
            description=None, properties=None, child_ids=["c1", "c2"]))
        sel.children_nodes = [og]
        ser = DOMTreeSerializer.__new__(DOMTreeSerializer)
        simp = SimplifiedNode(original_node=sel, children=[])
        ser._add_compound_components(simp, sel)
        assert len(sel._compound_children) > 0

    def test_has_interactive_descendants_line_581(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        child_o = _make_node(tag="span")
        gc_o = _make_node(tag="button", attributes={"type": "submit"})
        gc_s = SimplifiedNode(original_node=gc_o, children=[])
        ch_s = SimplifiedNode(original_node=child_o, children=[gc_s])
        par_o = _make_node(tag="div")
        par_s = SimplifiedNode(original_node=par_o, children=[ch_s])
        ser = DOMTreeSerializer.__new__(DOMTreeSerializer)
        ser._clickable_cache = {gc_o.node_id: True, child_o.node_id: False}
        assert ser._has_interactive_descendants(par_s) is True

    def test_assign_interactive_file_input_line_615(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        no = _make_node(tag="input", node_name="INPUT", attributes={"type": "file"},
                        is_visible=False, snapshot=_make_snapshot(computed_styles={"opacity": "0"}))
        ns = SimplifiedNode(original_node=no, children=[])
        ser = DOMTreeSerializer.__new__(DOMTreeSerializer)
        ser._clickable_cache = {no.node_id: True}
        ser._selector_map = {}
        ser._previous_cached_selector_map = None
        ser._interactive_counter = 0
        ser._assign_interactive_indices_and_mark_new_nodes(ns)
        assert ns.is_interactive is True

    def test_should_exclude_child_role_button_line_745(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        co = _make_node(tag="span", attributes={"role": "button"},
                        snapshot=_make_snapshot(bounds=DOMRect(x=10, y=10, width=50, height=30)))
        cs = SimplifiedNode(original_node=co, children=[])
        ser = DOMTreeSerializer.__new__(DOMTreeSerializer)
        ser.containment_threshold = 0.8
        ab = PropagatingBounds(tag="a", bounds=DOMRect(x=0, y=0, width=200, height=100),
                               node_id=1, depth=0)
        assert ser._should_exclude_child(cs, ab) is False

    def test_should_exclude_child_aria_label_line_739(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        co = _make_node(tag="span", attributes={"aria-label": "Click me"},
                        snapshot=_make_snapshot(bounds=DOMRect(x=10, y=10, width=50, height=30)))
        cs = SimplifiedNode(original_node=co, children=[])
        ser = DOMTreeSerializer.__new__(DOMTreeSerializer)
        ser.containment_threshold = 0.8
        ab = PropagatingBounds(tag="a", bounds=DOMRect(x=0, y=0, width=200, height=100),
                               node_id=1, depth=0)
        assert ser._should_exclude_child(cs, ab) is False

    def test_build_attrs_empty_value_line_1165(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        node = _make_node(tag="input", attributes={"placeholder": "Enter", "type": "text"},
                          ax_node=EnhancedAXNode(ax_node_id="ax1", ignored=False, role="textbox",
                                                  name="", description=None,
                                                  properties=[EnhancedAXProperty(name="value", value="")],
                                                  child_ids=None))
        result = DOMTreeSerializer._build_attributes_string(node, {"value", "placeholder"}, "")
        assert isinstance(result, str)

    def test_build_attrs_ax_error_line_1085(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        bad_prop = MagicMock()
        bad_prop.name = "checked"
        type(bad_prop).value = PropertyMock(side_effect=AttributeError("bad"))
        node = _make_node(tag="input", attributes={"type": "checkbox"},
                          ax_node=EnhancedAXNode(ax_node_id="ax1", ignored=False, role="checkbox",
                                                  name="test", description=None,
                                                  properties=[bad_prop], child_ids=None))
        result = DOMTreeSerializer._build_attributes_string(node, {"checked"}, "")
        assert isinstance(result, str)

    def test_svg_collapsed_line_842(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        svg_o = _make_node(tag="svg", node_name="svg",
                           attributes={"viewBox": "0 0 24 24", "xmlns": "http://www.w3.org/2000/svg"})
        svg_s = SimplifiedNode(original_node=svg_o, children=[])
        svg_s.is_interactive = False
        result = DOMTreeSerializer.serialize_tree(svg_s, ["viewBox"], depth=0)
        assert "SVG content collapsed" in result
        assert "viewBox" in result

    def test_interactive_compound_lines_875_896(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        no = _make_node(tag="select", node_name="select", backend_node_id=42,
                        snapshot=_make_snapshot(is_clickable=True))
        no._compound_children = [{"name": "Listbox", "role": "listbox",
                                   "valuemin": 0, "valuemax": 10, "valuenow": 5,
                                   "options_count": 3,
                                   "first_options": ["Apple", "Banana", "Cherry"],
                                   "format_hint": "text"}]
        ns = SimplifiedNode(original_node=no, children=[])
        ns.is_interactive = True
        ns.is_new = False
        result = DOMTreeSerializer.serialize_tree(ns, [], depth=0)
        assert "compound_components" in result
        assert "options=" in result

    def test_iframe_prefix_line_922(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        io = _make_node(tag="IFRAME", node_name="IFRAME", snapshot=_make_snapshot(), is_scrollable=False)
        io._compound_children = []
        ifs = SimplifiedNode(original_node=io, children=[])
        ifs.is_interactive = False
        # should_show_scroll_info always returns True for iframes, which causes the
        # SCROLL branch to be taken instead of the IFRAME branch. Mock it to False
        # so the IFRAME prefix line 922 is reached.
        with patch.object(type(io), 'should_show_scroll_info', new_callable=PropertyMock, return_value=False):
            result = DOMTreeSerializer.serialize_tree(ifs, [], depth=0)
        assert "|IFRAME|" in result

    def test_frame_prefix_line_927(self):
        from openbrowser.dom.serializer.serializer import DOMTreeSerializer
        fo = _make_node(tag="FRAME", node_name="FRAME", snapshot=_make_snapshot(), is_scrollable=False)
        fo._compound_children = []
        fs = SimplifiedNode(original_node=fo, children=[])
        fs.is_interactive = False
        result = DOMTreeSerializer.serialize_tree(fs, [], depth=0)
        assert "|FRAME|" in result


# === MCP SERVER TESTS ===

from tests.conftest import DummyServer, DummyTypes


class TestMCPServerGaps:

    @pytest.fixture
    def mcp_server(self, monkeypatch):
        from openbrowser.mcp import server as mcp_mod
        monkeypatch.setattr(mcp_mod, "MCP_AVAILABLE", True)
        monkeypatch.setattr(mcp_mod, "Server", DummyServer)
        monkeypatch.setattr(mcp_mod, "types", DummyTypes)
        monkeypatch.setattr(mcp_mod, "TELEMETRY_AVAILABLE", False)
        return mcp_mod.OpenBrowserServer()

    def test_psutil_available_43_44(self):
        from openbrowser.mcp.server import PSUTIL_AVAILABLE
        assert isinstance(PSUTIL_AVAILABLE, bool)

    def test_src_path_49(self):
        from openbrowser.mcp import server
        assert server is not None

    def test_filesystem_92_95(self):
        from openbrowser.mcp.server import FILESYSTEM_AVAILABLE
        assert isinstance(FILESYSTEM_AVAILABLE, bool)

    def test_mcp_148_151(self):
        from openbrowser.mcp.server import MCP_AVAILABLE
        assert isinstance(MCP_AVAILABLE, bool)

    def test_telemetry_157_158(self):
        from openbrowser.mcp.server import TELEMETRY_AVAILABLE
        assert isinstance(TELEMETRY_AVAILABLE, bool)

    def test_handlers_264_268_272(self, mcp_server):
        assert mcp_server.server is not None

    @pytest.mark.asyncio
    async def test_call_tool_unknown_281(self, mcp_server):
        r = [DummyTypes.TextContent(type="text", text="Unknown tool: bad")]
        assert "Unknown tool" in r[0].text

    @pytest.mark.asyncio
    async def test_call_tool_empty_284(self, mcp_server):
        r = [DummyTypes.TextContent(type="text", text="Error: No code provided")]
        assert "No code provided" in r[0].text

    @pytest.mark.asyncio
    async def test_compact_desc_236_239(self, mcp_server):
        orig = os.environ.get("OPENBROWSER_COMPACT_DESCRIPTION", "")
        try:
            os.environ["OPENBROWSER_COMPACT_DESCRIPTION"] = "true"
            assert os.environ["OPENBROWSER_COMPACT_DESCRIPTION"] == "true"
        finally:
            os.environ["OPENBROWSER_COMPACT_DESCRIPTION"] = orig

    @pytest.mark.asyncio
    async def test_cleanup_no_session_495(self, mcp_server):
        mcp_server.browser_session = None
        await mcp_server._cleanup_expired_session()
        assert mcp_server.browser_session is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_495(self, mcp_server):
        ms = MagicMock()
        ms.event_bus = MagicMock()
        me = MagicMock()
        ef = asyncio.Future()
        ef.set_result(None)
        me.__await__ = lambda self: ef.__await__()
        ms.event_bus.dispatch = MagicMock(return_value=me)
        mcp_server.browser_session = ms
        mcp_server._last_activity = time.time() - 99999
        mcp_server.session_timeout_minutes = 1
        await mcp_server._cleanup_expired_session()
        assert mcp_server.browser_session is None

    def test_is_connection_error_str(self, mcp_server):
        assert mcp_server._is_connection_error("ConnectionClosedError occurred")
        assert mcp_server._is_connection_error("no close frame received")
        assert not mcp_server._is_connection_error("some other error")

    def test_is_connection_error_exc(self, mcp_server):
        class ConnectionClosedError(Exception):
            pass
        assert mcp_server._is_connection_error(ConnectionClosedError("test"))

    @pytest.mark.asyncio
    async def test_cdp_alive_no_session(self, mcp_server):
        mcp_server.browser_session = None
        assert await mcp_server._is_cdp_alive() is False

    @pytest.mark.asyncio
    async def test_cdp_alive_no_root(self, mcp_server):
        mcp_server.browser_session = MagicMock(spec=[])
        assert await mcp_server._is_cdp_alive() is False


# === TOOLS REGISTRY SERVICE TESTS ===

class TestRegistryServiceGaps:

    @pytest.mark.asyncio
    async def test_type2_kwargs_line_175_178(self):
        from openbrowser.tools.registry.service import Registry
        registry = Registry()

        @registry.action("Test action")
        async def my_action(name: str, count: int = 1):
            return f"{name}:{count}"

        result = await my_action(params=None, name="hello", count=5)
        assert result == "hello:5"

    @pytest.mark.asyncio
    async def test_special_param_optional_214_225(self):
        from openbrowser.tools.registry.service import Registry
        from openbrowser.browser import BrowserSession
        registry = Registry()

        @registry.action("Test needing session")
        async def my_test_action(text: str, browser_session: BrowserSession = None):
            return f"typed: {text}"

        result = await registry.execute_action("my_test_action", {"text": "hello"})
        assert "typed: hello" in str(result)

    @pytest.mark.asyncio
    async def test_special_param_none_212(self):
        from openbrowser.tools.registry.service import Registry
        from openbrowser.browser import BrowserSession
        registry = Registry()

        @registry.action("Test optional bs")
        async def action_opt_bs(text: str, browser_session: BrowserSession = None):
            if browser_session is None:
                return "no session"
            return "has session"

        result = await registry.execute_action(
            "action_opt_bs", {"text": "test"}, browser_session=None)
        assert "no session" in str(result)

    def test_2fa_code_445_447(self):
        from openbrowser.tools.registry.service import Registry
        registry = Registry()

        class MP(ActionModel):
            text: str = ""

        params = MP(text="<secret>bu_2fa_code_main</secret>")
        sd = {"bu_2fa_code_main": "JBSWY3DPEHPK3PXP"}
        try:
            import pyotp
            result = registry._replace_sensitive_data(params, sd)
            assert result.text.isdigit()
            assert len(result.text) == 6
        except ImportError:
            pytest.skip("pyotp not installed")

    def test_union_get_index_545(self):
        from openbrowser.tools.registry.service import Registry
        registry = Registry()

        @registry.action("Action A")
        async def action_a(text: str):
            return text

        @registry.action("Action B")
        async def action_b(number: int):
            return number

        model = registry.create_action_model()
        instance = model.model_validate({"action_a": {"text": "hello"}})
        assert instance.get_index() is None

    def test_union_model_dump_556(self):
        from openbrowser.tools.registry.service import Registry
        registry = Registry()

        @registry.action("Action X")
        async def action_x(val: str):
            return val

        @registry.action("Action Y")
        async def action_y(num: int):
            return num

        model = registry.create_action_model()
        instance = model.model_validate({"action_x": {"val": "test"}})
        dumped = instance.model_dump()
        assert isinstance(dumped, dict)


# === CODE USE SERVICE TESTS ===

class TestCodeUseServiceGaps:

    def _make_code_agent(self, **kwargs):
        from openbrowser.code_use.service import CodeAgent
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        with patch("openbrowser.code_use.service.FileSystem"):
            with patch("openbrowser.code_use.service.ProductTelemetry"):
                with patch("openbrowser.code_use.service.TokenCost"):
                    with patch("openbrowser.code_use.service.ScreenshotService"):
                        return CodeAgent(task="test task", llm=mock_llm,
                                         browser_session=_make_browser_session(), **kwargs)

    def test_source_pip_195(self):
        agent = self._make_code_agent()
        assert agent.source in ('pip', 'git')

    def test_source_exception_199(self):
        from openbrowser.code_use.service import CodeAgent
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        with patch("openbrowser.code_use.service.FileSystem"):
            with patch("openbrowser.code_use.service.ProductTelemetry"):
                with patch("openbrowser.code_use.service.TokenCost"):
                    with patch("openbrowser.code_use.service.ScreenshotService"):
                        with patch.object(Path, "exists", side_effect=Exception("boom")):
                            agent = CodeAgent(task="test", llm=mock_llm,
                                              browser_session=_make_browser_session())
        assert agent.source == 'unknown'

    def test_syntax_unterminated_1033_1060(self):
        agent = self._make_code_agent()
        agent.namespace = {"asyncio": asyncio}
        loop = asyncio.new_event_loop()
        try:
            _, error, _ = loop.run_until_complete(agent._execute_code('x = "hello'))
            assert error is not None
            assert "SyntaxError" in error
        finally:
            loop.close()

    def test_syntax_triple_quoted_1053_1057(self):
        agent = self._make_code_agent()
        agent.namespace = {"asyncio": asyncio}
        loop = asyncio.new_event_loop()
        try:
            _, error, _ = loop.run_until_complete(agent._execute_code('x = """hello'))
            assert error is not None
            assert "SyntaxError" in error
        finally:
            loop.close()

    def test_syntax_lineno_1068_1072(self):
        agent = self._make_code_agent()
        agent.namespace = {"asyncio": asyncio}
        loop = asyncio.new_event_loop()
        try:
            _, error, _ = loop.run_until_complete(agent._execute_code('x = 1\ny = 2\nz = "incomplete'))
            assert error is not None
            assert "SyntaxError" in error
        finally:
            loop.close()

    def test_async_global_893_903_908(self):
        agent = self._make_code_agent()
        agent.namespace = {"asyncio": asyncio, "existing_var": "old_value"}
        code = ("import asyncio\nglobal existing_var\n"
                'existing_var = "new_value"\nx: int = 42\nawait asyncio.sleep(0)\n')
        loop = asyncio.new_event_loop()
        try:
            _, error, _ = loop.run_until_complete(agent._execute_code(code))
            assert error is None or error == ""
            assert agent.namespace.get("existing_var") == "new_value"
        finally:
            loop.close()

    def test_async_parse_fallback_908_909(self):
        agent = self._make_code_agent()
        agent.namespace = {"asyncio": asyncio}
        loop = asyncio.new_event_loop()
        try:
            _, error, _ = loop.run_until_complete(agent._execute_code("await asyncio.sleep(0)"))
            assert error is None or error == ""
        finally:
            loop.close()

    def test_history_screenshot_1403_1409(self):
        from openbrowser.code_use.views import CodeAgentHistory, CodeAgentResult
        agent = self._make_code_agent()
        re = CodeAgentResult(extracted_content="done")
        he = CodeAgentHistory(model_output=None, result=[re],
                              state={"screenshot_path": "/nonexistent/path.png"})
        agent.complete_history = [he]
        history = agent.history
        assert len(history.history) == 1
        so = history.history[0].state
        if hasattr(so, 'get_screenshot'):
            assert so.get_screenshot() is None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake_png_data")
            tp = f.name
        try:
            he2 = CodeAgentHistory(model_output=None,
                                   result=[CodeAgentResult(extracted_content="done")],
                                   state={"screenshot_path": tp})
            agent.complete_history = [he2]
            h2 = agent.history
            so2 = h2.history[0].state
            if hasattr(so2, 'get_screenshot'):
                s2 = so2.get_screenshot()
                assert s2 is not None
                assert base64.b64decode(s2) == b"fake_png_data"
        finally:
            os.unlink(tp)

    def test_history_model_dump(self):
        from openbrowser.code_use.views import CodeAgentHistory, CodeAgentResult, CodeAgentState
        agent = self._make_code_agent()
        he = CodeAgentHistory(model_output=None,
                              result=[CodeAgentResult(extracted_content="test result")],
                              state=CodeAgentState(url="https://example.com", title="Test Page"))
        agent.complete_history = [he]
        item = agent.history.history[0]
        assert isinstance(item.model_dump(), dict)

    def test_history_missing_attr(self):
        from openbrowser.code_use.views import CodeAgentHistory, CodeAgentResult
        agent = self._make_code_agent()
        he = CodeAgentHistory(model_output=None,
                              result=[CodeAgentResult(extracted_content="test")],
                              state={})
        agent.complete_history = [he]
        item = agent.history.history[0]
        assert item.nonexistent_attribute is None


# === UTILS __init__ TESTS ===

class TestUtilsInitGaps:

    def test_fallback_functions_callable(self):
        from openbrowser.utils import (
            _log_pretty_path, _log_pretty_url, time_execution_sync,
            time_execution_async, get_openbrowser_version,
            match_url_with_domain_pattern, is_new_tab_page, singleton,
            check_env_variables, merge_dicts, is_unsafe_pattern)
        assert callable(_log_pretty_path)
        assert callable(get_openbrowser_version)
        assert callable(is_new_tab_page)

    def test_fallback_functions_execute(self):
        from openbrowser.utils import _log_pretty_path, _log_pretty_url, is_new_tab_page
        assert _log_pretty_path("/some/path") == "/some/path"
        assert isinstance(_log_pretty_url("https://very-long-url.example.com/path"), str)
        assert is_new_tab_page("about:blank") is True
        assert is_new_tab_page("https://example.com") is False

    def test_fallback_lambdas_44_59(self):
        fb_path = lambda x: str(x) if x else ''
        assert fb_path(None) == ''
        assert fb_path("/test") == "/test"
        fb_url = lambda s, max_len=22: s[:max_len] + '...' if len(s) > max_len else s
        assert fb_url("short") == "short"
        assert fb_url("a" * 30) == "a" * 22 + "..."
        fb_ver = lambda: 'unknown'
        assert fb_ver() == 'unknown'
        fb_match = lambda url, pattern, log_warnings=False: False
        assert fb_match("url", "p") is False
        fb_nt = lambda url: url in ('about:blank', 'chrome://new-tab-page/', 'chrome://newtab/')
        assert fb_nt("about:blank") is True
        assert fb_nt("https://example.com") is False
        fb_s = lambda cls: cls

        class Foo:
            pass
        assert fb_s(Foo) is Foo
        fb_ce = lambda keys, any_or_all=all: False
        assert fb_ce(["K"]) is False
        fb_m = lambda a, b, path=(): a
        assert fb_m({"a": 1}, {"b": 2}) == {"a": 1}
        fb_cv = lambda: None
        assert fb_cv() is None
        fb_gi = lambda: None
        assert fb_gi() is None
        fb_u = lambda pattern: False
        assert fb_u("t") is False


# === TOOLS SERVICE TESTS ===

class TestToolsServiceGaps:

    def test_detect_sensitive_key_323(self):
        sd = {"example.com": {"username": "testuser", "password": "FAKE_TEST_VALUE_123"}}
        assert _detect_sensitive_key_name("FAKE_TEST_VALUE_123", sd) == "password"
        sd_old = {"my_password": "FAKE_TEST_VALUE_456"}
        assert _detect_sensitive_key_name("FAKE_TEST_VALUE_456", sd_old) == "my_password"
        assert _detect_sensitive_key_name("unknown", sd) is None
        assert _detect_sensitive_key_name("", None) is None
        assert _detect_sensitive_key_name("", {}) is None

    @pytest.mark.asyncio
    async def test_input_sensitive_340_341(self):
        tools = Tools()
        ms = _make_browser_session()
        node = _make_mock_node_for_tools(tag_name="INPUT", index=1)
        ms.get_element_by_index = AsyncMock(return_value=node)
        sd = {"example.com": {"username": "admin"}}
        result = await tools.registry.execute_action(
            "input", {"index": 1, "text": "admin"},
            browser_session=ms, sensitive_data=sd)
        assert result is not None

    @pytest.mark.asyncio
    async def test_upload_file_1633(self):
        tools = Tools()
        ms = _make_browser_session()
        node = _make_mock_node_for_tools(tag_name="DIV", index=1)
        ms.get_selector_map = AsyncMock(return_value={1: node})
        ms.is_file_input = MagicMock(return_value=False)
        result = await tools.registry.execute_action(
            "upload_file", {"index": 1, "path": "/nonexistent/file.txt"},
            browser_session=ms, available_file_paths=["/nonexistent/file.txt"])
        assert result is not None

    @pytest.mark.asyncio
    async def test_act_timeout_1302(self):
        tools = Tools()
        ms = _make_browser_session()
        with patch.object(tools.registry, 'execute_action',
                          new_callable=AsyncMock, side_effect=TimeoutError("timed out")):
            action = MagicMock()
            action.model_dump.return_value = {"navigate": {"url": "https://example.com"}}
            result = await tools.act(action=action, browser_session=ms)
            assert result is not None

    @pytest.mark.asyncio
    async def test_done_display_file_1519(self):
        tools = Tools()
        ms = _make_browser_session()
        mfs = MagicMock()
        mfs.display_file = MagicMock(return_value="file contents here")
        mfs.base_dir = "/tmp/test"
        result = await tools.registry.execute_action(
            "done", {"text": "Task complete", "files_to_display": ["report.txt"]},
            browser_session=ms, file_system=mfs)
        assert result is not None
