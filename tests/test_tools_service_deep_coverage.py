"""Deep coverage tests for openbrowser.tools.service module.

Targets all missed lines from the coverage report:
323, 340-341, 374-527, 612-613, 630-646, 657, 711-713, 777-778, 786, 795-796,
897, 926, 1273, 1299-1300, 1302-1303, 1310, 1316-1319, 1444, 1504-1505, 1508,
1514-1532, 1543-1548, 1585-1599, 1603, 1605, 1632-1640, 1648-1664, 1675-1711,
1725-1727
"""

import asyncio
import enum
import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import BaseModel

from openbrowser.browser.views import BrowserError
from openbrowser.dom.views import DOMRect, EnhancedDOMTreeNode, NodeType
from openbrowser.models import ActionResult
from openbrowser.tools.registry.views import ActionModel
from openbrowser.tools.service import (
    CodeAgentTools,
    Tools,
    _detect_sensitive_key_name,
    handle_browser_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_browser_session():
    """Create a fully mocked BrowserSession."""
    session = MagicMock()
    session.is_local = True
    session.current_target_id = "target123456789012345678901234"
    session.downloaded_files = []
    session.logger = MagicMock()

    mock_event = MagicMock()
    mock_event.__aiter__ = MagicMock(return_value=iter([]))
    mock_event.event_result = AsyncMock(return_value=None)
    event_future = asyncio.Future()
    event_future.set_result(None)
    mock_event.__await__ = lambda self: event_future.__await__()

    session.event_bus.dispatch = MagicMock(return_value=mock_event)
    session.get_element_by_index = AsyncMock(return_value=None)
    session.get_selector_map = AsyncMock(return_value={})
    session.get_target_id_from_tab_id = AsyncMock(return_value="full_target_id_1234")
    session.highlight_interaction_element = AsyncMock()
    session.is_file_input = MagicMock(return_value=False)
    session.get_current_page_url = AsyncMock(return_value="https://example.com")

    cdp_session = AsyncMock()
    cdp_session.cdp_client.send.Runtime.evaluate = AsyncMock(
        return_value={"result": {"value": "test_result"}}
    )
    cdp_session.cdp_client.send.Page.getLayoutMetrics = AsyncMock(
        return_value={
            "cssVisualViewport": {"clientHeight": 800},
            "cssLayoutViewport": {"clientHeight": 800},
        }
    )
    session.get_or_create_cdp_session = AsyncMock(return_value=cdp_session)
    session.cdp_client = MagicMock()

    session.browser_profile = MagicMock()
    session.browser_profile.downloads_path = "/tmp/downloads"
    session.browser_profile.keep_alive = False

    return session


def _make_mock_node(tag_name="div", index=1, attributes=None, parent=None, children=None):
    """Create a real EnhancedDOMTreeNode."""
    node = EnhancedDOMTreeNode(
        node_id=index,
        backend_node_id=index + 100,
        node_type=NodeType.ELEMENT_NODE,
        node_name=tag_name.upper(),
        node_value="",
        attributes=attributes or {},
        is_scrollable=False,
        is_visible=True,
        absolute_position=DOMRect(x=50.0, y=100.0, width=200.0, height=40.0),
        target_id="target123456789012345678901234",
        frame_id="frame_001",
        session_id="session_001",
        content_document=None,
        shadow_root_type=None,
        shadow_roots=[],
        parent_node=parent,
        children_nodes=children or [],
        ax_node=None,
        snapshot_node=None,
    )
    return node


def _make_mock_file_system():
    """Create a mock FileSystem."""
    fs = MagicMock()
    fs.display_file = MagicMock(return_value="file content")
    fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))
    fs.write_file = AsyncMock(return_value="Wrote file test.txt")
    fs.append_file = AsyncMock(return_value="Appended to test.txt")
    fs.read_file = AsyncMock(return_value="file contents")
    fs.replace_file_str = AsyncMock(return_value="Replaced in test.txt")
    fs.save_extracted_content = AsyncMock(return_value="extracted_001.md")
    fs.get_file = MagicMock(return_value=None)
    return fs


class _AwaitableEvent:
    """Mock event that supports 'await event' and 'await event.event_result()'."""

    def __init__(self, result=None, side_effect=None):
        self._result = result
        self._side_effect = side_effect

    def __await__(self):
        f = asyncio.Future()
        f.set_result(None)
        return f.__await__()

    async def event_result(self, **kwargs):
        if self._side_effect:
            raise self._side_effect
        return self._result


def _make_awaitable_event(result=None, side_effect=None):
    return _AwaitableEvent(result=result, side_effect=side_effect)


# ---------------------------------------------------------------------------
# Line 323: input action with has_sensitive_data=True and detected key name
# ---------------------------------------------------------------------------


class TestInputSensitiveKeyDetection:
    """Covers line 323 (_detect_sensitive_key_name call inside input action)
    and lines 340-341 (sensitive_key_name branch)."""

    @pytest.mark.asyncio
    async def test_input_sensitive_data_with_detected_key_name(self):
        """When has_sensitive_data=True and sensitive_data matches, line 323 + 340-341 are hit."""
        tools = Tools()
        session = _make_mock_browser_session()
        node = _make_mock_node("input")
        session.get_element_by_index = AsyncMock(return_value=node)
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        result = await tools.registry.execute_action(
            "input",
            {"index": 3, "text": "hunter2"},
            browser_session=session,
            sensitive_data={"password": "hunter2"},
        )
        assert isinstance(result, ActionResult)
        # The sensitive key should be detected; message should say "Typed password"
        assert "Typed" in result.extracted_content

    @pytest.mark.asyncio
    async def test_input_sensitive_data_no_key_match(self):
        """has_sensitive_data=True but text doesn't match any key -> 'Typed sensitive data'."""
        tools = Tools()
        session = _make_mock_browser_session()
        node = _make_mock_node("input")
        session.get_element_by_index = AsyncMock(return_value=node)
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        result = await tools.registry.execute_action(
            "input",
            {"index": 3, "text": "nomatch_text"},
            browser_session=session,
            sensitive_data={"password": "different_value"},
        )
        assert isinstance(result, ActionResult)
        assert "Typed" in result.extracted_content


# ---------------------------------------------------------------------------
# Lines 374-527: upload_file action in base Tools class
# ---------------------------------------------------------------------------


class TestBaseToolsUploadFile:
    """Covers upload_file in base Tools.__init__ (lines 374-527)."""

    @pytest.mark.asyncio
    async def test_upload_file_path_in_available_file_paths(self):
        """Happy path: file is in available_file_paths."""
        tools = Tools()
        session = _make_mock_browser_session()
        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert isinstance(result, ActionResult)
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_path_in_downloaded_files(self):
        """File not in available_file_paths but in downloaded_files."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.downloaded_files = ["/tmp/downloaded.pdf"]
        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/downloaded.pdf"},
                browser_session=session,
                available_file_paths=[],
            )
        assert isinstance(result, ActionResult)
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_path_in_filesystem_service(self):
        """File found in FileSystem service (lines 379-386)."""
        tools = Tools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=MagicMock())  # file found
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "myfile.txt"},
                browser_session=session,
                available_file_paths=[],
                file_system=fs,
            )
        assert isinstance(result, ActionResult)
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_not_in_fs_remote_browser(self):
        """File not found in FileSystem, browser is remote (lines 388-390)."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.is_local = False
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=None)
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        result = await tools.registry.execute_action(
            "upload_file",
            {"index": 1, "path": "/remote/path.txt"},
            browser_session=session,
            available_file_paths=[],
            file_system=fs,
        )
        assert isinstance(result, ActionResult)
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_not_in_fs_local_abs_exists(self):
        """File not found in FileSystem, local browser, absolute path exists (lines 391-394)."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.is_local = True
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=None)
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.isabs", return_value=True), \
             patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/absolute_file.txt"},
                browser_session=session,
                available_file_paths=[],
                file_system=fs,
            )
        assert isinstance(result, ActionResult)
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_not_in_fs_local_not_found_error(self):
        """File not found in FileSystem, local, not absolute or doesn't exist (lines 395-398)."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.is_local = True
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=None)
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        result = await tools.registry.execute_action(
            "upload_file",
            {"index": 1, "path": "relative_file.txt"},
            browser_session=session,
            available_file_paths=[],
            file_system=fs,
        )
        assert result.error is not None
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_upload_file_no_filesystem_remote_browser(self):
        """No FileSystem, browser is remote (lines 400-402)."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.is_local = False

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        result = await tools.registry.execute_action(
            "upload_file",
            {"index": 1, "path": "/remote/file.txt"},
            browser_session=session,
            available_file_paths=[],
            file_system=None,
        )
        assert isinstance(result, ActionResult)
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_no_filesystem_local_abs_exists(self):
        """No FileSystem, local browser, absolute path exists (lines 403-406)."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.is_local = True

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.isabs", return_value=True), \
             patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/local_file.txt"},
                browser_session=session,
                available_file_paths=[],
                file_system=None,
            )
        assert isinstance(result, ActionResult)
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_no_filesystem_local_not_found_raises(self):
        """No FileSystem, local browser, relative path -> raises BrowserError (lines 407-409).
        execute_action wraps BrowserError in RuntimeError."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.is_local = True

        with pytest.raises(RuntimeError, match="not available"):
            await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "no_such_file.txt"},
                browser_session=session,
                available_file_paths=[],
                file_system=None,
            )

    @pytest.mark.asyncio
    async def test_upload_file_local_file_not_exists(self):
        """Local browser, file doesn't exist on disk (lines 412-415)."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.is_local = True

        with patch("os.path.exists", return_value=False):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/nonexistent.txt"},
                browser_session=session,
                available_file_paths=["/tmp/nonexistent.txt"],
            )
        assert result.error is not None
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_upload_file_element_not_in_selector_map(self):
        """Element not found in selector map (lines 419-421)."""
        tools = Tools()
        session = _make_mock_browser_session()
        session.get_selector_map = AsyncMock(return_value={})

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert result.error is not None
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_upload_file_find_file_input_near_element_direct(self):
        """File input IS the selected element (line 445-446)."""
        tools = Tools()
        session = _make_mock_browser_session()
        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_find_file_input_in_descendants(self):
        """File input found in descendants of selected element (lines 431-440, 448-450)."""
        tools = Tools()
        session = _make_mock_browser_session()

        child_node = _make_mock_node("input", index=2, attributes={"type": "file"})
        parent_node = _make_mock_node("div", index=1, children=[child_node])
        child_node.parent_node = parent_node

        call_count = [0]
        def is_file_input_side_effect(n):
            if n.node_id == 2:
                return True
            return False
        session.is_file_input = MagicMock(side_effect=is_file_input_side_effect)
        session.get_selector_map = AsyncMock(return_value={1: parent_node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_find_file_input_in_sibling(self):
        """File input found in a sibling of the selected element (lines 452-460)."""
        tools = Tools()
        session = _make_mock_browser_session()

        grandparent = _make_mock_node("div", index=10)
        selected = _make_mock_node("div", index=1, parent=grandparent)
        sibling_file_input = _make_mock_node("input", index=3, parent=grandparent)
        grandparent.children_nodes = [selected, sibling_file_input]

        def is_file_input_side_effect(n):
            return n.node_id == 3
        session.is_file_input = MagicMock(side_effect=is_file_input_side_effect)
        session.get_selector_map = AsyncMock(return_value={1: selected})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_fallback_scroll_position_closest(self):
        """File input not near element, found via scroll position fallback (lines 474-507)."""
        tools = Tools()
        session = _make_mock_browser_session()

        # selected node is not file input
        selected = _make_mock_node("div", index=1)
        # file input element at a known position
        file_input = _make_mock_node("input", index=5, attributes={"type": "file"})
        file_input.absolute_position = DOMRect(x=50.0, y=200.0, width=200.0, height=40.0)

        def is_file_input_fn(n):
            return n.node_id == 5
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: selected, 5: file_input})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": 150}}
        )

        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_fallback_scroll_position_exception(self):
        """CDP scroll position throws exception, falls back to 0 (line 486-487)."""
        tools = Tools()
        session = _make_mock_browser_session()

        selected = _make_mock_node("div", index=1)
        file_input = _make_mock_node("input", index=5, attributes={"type": "file"})
        file_input.absolute_position = DOMRect(x=50.0, y=200.0, width=200.0, height=40.0)

        def is_file_input_fn(n):
            return n.node_id == 5
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: selected, 5: file_input})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=RuntimeError("CDP error")
        )

        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_no_file_input_on_page_raises(self):
        """No file input found anywhere on the page (lines 508-511).
        execute_action wraps BrowserError in RuntimeError."""
        tools = Tools()
        session = _make_mock_browser_session()

        selected = _make_mock_node("div", index=1)
        session.is_file_input = MagicMock(return_value=False)
        session.get_selector_map = AsyncMock(return_value={1: selected})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": 0}}
        )

        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="No file upload element"):
                await tools.registry.execute_action(
                    "upload_file",
                    {"index": 1, "path": "/tmp/test.txt"},
                    browser_session=session,
                    available_file_paths=["/tmp/test.txt"],
                )

    @pytest.mark.asyncio
    async def test_upload_file_dispatch_failure(self):
        """Upload event dispatch fails (lines 525-527).
        execute_action wraps BrowserError in RuntimeError."""
        tools = Tools()
        session = _make_mock_browser_session()

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event(side_effect=RuntimeError("upload failed"))
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="Failed to upload file"):
                await tools.registry.execute_action(
                    "upload_file",
                    {"index": 1, "path": "/tmp/test.txt"},
                    browser_session=session,
                    available_file_paths=["/tmp/test.txt"],
                )

    @pytest.mark.asyncio
    async def test_upload_file_find_in_descendants_depth_exceeded(self):
        """find_file_input_in_descendants returns None when depth < 0 (line 432-433)."""
        tools = Tools()
        session = _make_mock_browser_session()

        # Create a deep tree but file input is too deep
        deep_child = _make_mock_node("input", index=99, attributes={"type": "file"})
        level3 = _make_mock_node("div", index=4, children=[deep_child])
        deep_child.parent_node = level3
        level2 = _make_mock_node("div", index=3, children=[level3])
        level3.parent_node = level2
        level1 = _make_mock_node("div", index=2, children=[level2])
        level2.parent_node = level1
        root = _make_mock_node("div", index=1, children=[level1])
        level1.parent_node = root

        # Only mark the deep child as file input
        def is_file_input_fn(n):
            return n.node_id == 99
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: root, 99: deep_child})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": 0}}
        )

        # The tree is deeper than max_descendant_depth=3, and max_height=3
        # The file input at depth 4 should be found via the selector_map fallback
        deep_child.absolute_position = DOMRect(x=50.0, y=100.0, width=200.0, height=40.0)

        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()


# ---------------------------------------------------------------------------
# Lines 612-613: extract action markdown extraction failure
# ---------------------------------------------------------------------------


class TestExtractActionErrors:
    """Covers extract action error branches."""

    @pytest.mark.asyncio
    async def test_extract_markdown_extraction_failure(self):
        """extract_clean_markdown raises -> RuntimeError (lines 612-613)."""
        tools = Tools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        mock_llm = MagicMock()

        with patch(
            "openbrowser.dom.markdown_extractor.extract_clean_markdown",
            new_callable=AsyncMock,
            side_effect=ValueError("DOM parse error"),
        ):
            with pytest.raises(RuntimeError, match="Could not extract clean markdown"):
                await tools.registry.execute_action(
                    "extract",
                    {"params": {"query": "data", "extract_links": False, "start_from_char": 0}},
                    browser_session=session,
                    page_extraction_llm=mock_llm,
                    file_system=fs,
                )

    @pytest.mark.asyncio
    async def test_extract_llm_invocation_failure(self):
        """LLM ainvoke fails -> RuntimeError (lines 711-713)."""
        tools = Tools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        with patch(
            "openbrowser.dom.markdown_extractor.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(
                "Content",
                {
                    "original_html_chars": 100,
                    "initial_markdown_chars": 50,
                    "final_filtered_chars": 20,
                    "filtered_chars_removed": 30,
                },
            ),
        ):
            with pytest.raises(RuntimeError, match="LLM timeout"):
                await tools.registry.execute_action(
                    "extract",
                    {"params": {"query": "data", "extract_links": False, "start_from_char": 0}},
                    browser_session=session,
                    page_extraction_llm=mock_llm,
                    file_system=fs,
                )


# ---------------------------------------------------------------------------
# Lines 630-646, 657: extract content truncation branches
# ---------------------------------------------------------------------------


class TestExtractTruncation:
    """Covers smart truncation in extract action."""

    @pytest.mark.asyncio
    async def test_extract_truncation_paragraph_break(self):
        """Content exceeds MAX_CHAR_LIMIT, truncated at paragraph break (lines 630-635)."""
        tools = Tools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Extracted data")
        )

        # Create content larger than 30000 with paragraph break near the end
        content = "A" * 29600 + "\n\n" + "B" * 500
        assert len(content) > 30000

        with patch(
            "openbrowser.dom.markdown_extractor.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(
                content,
                {
                    "original_html_chars": 40000,
                    "initial_markdown_chars": 35000,
                    "final_filtered_chars": len(content),
                    "filtered_chars_removed": 5000,
                },
            ),
        ):
            result = await tools.registry.execute_action(
                "extract",
                {"params": {"query": "all data", "extract_links": False, "start_from_char": 0}},
                browser_session=session,
                page_extraction_llm=mock_llm,
                file_system=fs,
            )
        assert isinstance(result, ActionResult)

    @pytest.mark.asyncio
    async def test_extract_truncation_sentence_break(self):
        """No paragraph break, truncated at sentence break (lines 636-640)."""
        tools = Tools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Extracted data")
        )

        # Content with no paragraph break but has sentence break near 30000
        content = "A" * 29850 + ". " + "B" * 200
        assert len(content) > 30000

        with patch(
            "openbrowser.dom.markdown_extractor.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(
                content,
                {
                    "original_html_chars": 40000,
                    "initial_markdown_chars": 35000,
                    "final_filtered_chars": len(content),
                    "filtered_chars_removed": 5000,
                },
            ),
        ):
            result = await tools.registry.execute_action(
                "extract",
                {"params": {"query": "data", "extract_links": False, "start_from_char": 0}},
                browser_session=session,
                page_extraction_llm=mock_llm,
                file_system=fs,
            )
        assert isinstance(result, ActionResult)

    @pytest.mark.asyncio
    async def test_extract_truncation_at_max_limit(self):
        """No paragraph or sentence break, truncated at MAX_CHAR_LIMIT (lines 642-646)."""
        tools = Tools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Extracted data")
        )

        # Content with no breaks at all
        content = "A" * 35000
        with patch(
            "openbrowser.dom.markdown_extractor.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(
                content,
                {
                    "original_html_chars": 40000,
                    "initial_markdown_chars": 35000,
                    "final_filtered_chars": len(content),
                    "filtered_chars_removed": 5000,
                },
            ),
        ):
            result = await tools.registry.execute_action(
                "extract",
                {"params": {"query": "data", "extract_links": False, "start_from_char": 0}},
                browser_session=session,
                page_extraction_llm=mock_llm,
                file_system=fs,
            )
        assert isinstance(result, ActionResult)

    @pytest.mark.asyncio
    async def test_extract_truncated_with_stats_message(self):
        """Truncated content includes 'use start_from_char' in result (line 657)."""
        tools = Tools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(completion="Extracted data")
        )

        content = "A" * 35000
        with patch(
            "openbrowser.dom.markdown_extractor.extract_clean_markdown",
            new_callable=AsyncMock,
            return_value=(
                content,
                {
                    "original_html_chars": 40000,
                    "initial_markdown_chars": 35000,
                    "final_filtered_chars": len(content),
                    "filtered_chars_removed": 5000,
                },
            ),
        ):
            result = await tools.registry.execute_action(
                "extract",
                {"params": {"query": "data", "extract_links": False, "start_from_char": 0}},
                browser_session=session,
                page_extraction_llm=mock_llm,
                file_system=fs,
            )
        assert isinstance(result, ActionResult)
        assert result.extracted_content is not None


# ---------------------------------------------------------------------------
# Lines 777-778, 786, 795-796: scroll with errors during multi-page and fractional
# ---------------------------------------------------------------------------


class TestScrollErrorPaths:
    """Covers scroll action error paths in multi-page and fractional scrolling."""

    @pytest.mark.asyncio
    async def test_scroll_multi_page_one_scroll_fails(self):
        """One scroll in multi-page fails but continues (lines 777-778)."""
        tools = Tools()
        session = _make_mock_browser_session()

        call_count = [0]
        def dispatch_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                return _make_awaitable_event(side_effect=RuntimeError("scroll 2 failed"))
            return _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(side_effect=dispatch_side_effect)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await tools.registry.execute_action(
                "scroll",
                {"down": True, "pages": 3.0},
                browser_session=session,
            )
        assert isinstance(result, ActionResult)
        assert "Scrolled" in result.extracted_content

    @pytest.mark.asyncio
    async def test_scroll_multi_page_upward_with_fractional(self):
        """Multi-page scroll upward with fractional remainder (line 786)."""
        tools = Tools()
        session = _make_mock_browser_session()
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await tools.registry.execute_action(
                "scroll",
                {"down": False, "pages": 2.5},
                browser_session=session,
            )
        assert isinstance(result, ActionResult)
        assert "up" in result.long_term_memory.lower()

    @pytest.mark.asyncio
    async def test_scroll_fractional_remainder_fails(self):
        """Fractional scroll after full pages fails (lines 795-796)."""
        tools = Tools()
        session = _make_mock_browser_session()

        call_count = [0]
        def dispatch_side_effect(*args, **kwargs):
            call_count[0] += 1
            # The fractional scroll is the last dispatch call
            if call_count[0] > 2:
                return _make_awaitable_event(side_effect=RuntimeError("fractional failed"))
            return _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(side_effect=dispatch_side_effect)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await tools.registry.execute_action(
                "scroll",
                {"down": True, "pages": 2.5},
                browser_session=session,
            )
        assert isinstance(result, ActionResult)


# ---------------------------------------------------------------------------
# Line 897: dropdown_options returns falsy data
# ---------------------------------------------------------------------------


class TestDropdownOptionsNoData:
    @pytest.mark.asyncio
    async def test_dropdown_options_no_data_returned(self):
        """dropdown_data is falsy -> raises ValueError (line 897).
        execute_action wraps ValueError in RuntimeError."""
        tools = Tools()
        session = _make_mock_browser_session()
        node = _make_mock_node("select")
        session.get_element_by_index = AsyncMock(return_value=node)
        # Return empty dict (falsy-like when checked)
        event = _make_awaitable_event(result=None)
        session.event_bus.dispatch = MagicMock(return_value=event)

        with pytest.raises(RuntimeError, match="Failed to get dropdown options"):
            await tools.registry.execute_action(
                "dropdown_options",
                {"index": 5},
                browser_session=session,
            )


# ---------------------------------------------------------------------------
# Line 926: select_dropdown returns falsy data
# ---------------------------------------------------------------------------


class TestSelectDropdownNoData:
    @pytest.mark.asyncio
    async def test_select_dropdown_no_data_returned(self):
        """selection_data is falsy -> raises ValueError (line 926).
        execute_action wraps ValueError in RuntimeError."""
        tools = Tools()
        session = _make_mock_browser_session()
        node = _make_mock_node("select")
        session.get_element_by_index = AsyncMock(return_value=node)
        event = _make_awaitable_event(result=None)
        session.event_bus.dispatch = MagicMock(return_value=event)

        with pytest.raises(RuntimeError, match="Failed to select dropdown option"):
            await tools.registry.execute_action(
                "select_dropdown",
                {"index": 5, "text": "Option A"},
                browser_session=session,
            )


# ---------------------------------------------------------------------------
# Line 1273: act() with Laminar available
# ---------------------------------------------------------------------------


class TestActWithLaminar:
    @pytest.mark.asyncio
    async def test_act_with_laminar_span(self):
        """When Laminar is not None, span_context is created (lines 1273, 1310)."""
        tools = Tools()
        fs = _make_mock_file_system()

        mock_laminar = MagicMock()
        mock_span_ctx = MagicMock()
        mock_span_ctx.__enter__ = MagicMock(return_value=None)
        mock_span_ctx.__exit__ = MagicMock(return_value=False)
        mock_laminar.start_as_current_span = MagicMock(return_value=mock_span_ctx)
        mock_laminar.set_span_output = MagicMock()

        from pydantic import create_model
        from openbrowser.tools.views import DoneAction

        DynamicModel = create_model(
            "TestLaminarModel",
            __base__=ActionModel,
            done=(DoneAction | None, None),
        )
        action = DynamicModel(done=DoneAction(text="Complete", success=True, files_to_display=[]))

        with patch("openbrowser.tools.service.Laminar", mock_laminar):
            result = await tools.act(
                action=action,
                browser_session=_make_mock_browser_session(),
                file_system=fs,
            )
        assert isinstance(result, ActionResult)
        assert result.is_done is True
        mock_laminar.start_as_current_span.assert_called_once()
        mock_laminar.set_span_output.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 1299-1300: act() BrowserError handling
# ---------------------------------------------------------------------------


class TestActBrowserError:
    @pytest.mark.asyncio
    async def test_act_catches_browser_error(self):
        """BrowserError in execute_action -> handle_browser_error (lines 1299-1300).
        We must mock execute_action to raise BrowserError directly (since the real
        execute_action wraps all exceptions in RuntimeError)."""
        tools = Tools()

        err = BrowserError(
            "test browser error",
            long_term_memory="Browser error occurred",
        )

        from pydantic import create_model
        from openbrowser.tools.views import DoneAction

        DynamicModel = create_model(
            "BrowserFailModel",
            __base__=ActionModel,
            done=(DoneAction | None, None),
        )
        action = DynamicModel(done=DoneAction(text="test", success=True, files_to_display=[]))

        with patch.object(tools.registry, "execute_action", new_callable=AsyncMock, side_effect=err):
            result = await tools.act(
                action=action,
                browser_session=_make_mock_browser_session(),
            )
        assert result.error == "Browser error occurred"


# ---------------------------------------------------------------------------
# Lines 1302-1303: act() TimeoutError handling
# ---------------------------------------------------------------------------


class TestActTimeoutError:
    @pytest.mark.asyncio
    async def test_act_catches_timeout_error(self):
        """TimeoutError -> 'not executed due to timeout' (lines 1302-1303)."""
        tools = Tools()

        @tools.action("Timeout action")
        async def timeout_action():
            raise TimeoutError("operation timed out")

        from pydantic import create_model

        DynamicModel = create_model(
            "TimeoutModel",
            __base__=ActionModel,
            timeout_action=(dict | None, None),
        )
        action = DynamicModel(timeout_action={})

        result = await tools.act(
            action=action,
            browser_session=_make_mock_browser_session(),
        )
        assert "timeout" in result.error.lower()


# ---------------------------------------------------------------------------
# Lines 1316-1319: act() returns None or invalid type
# ---------------------------------------------------------------------------


class TestActReturnTypes:
    @pytest.mark.asyncio
    async def test_act_none_result(self):
        """Action returns None -> ActionResult() (lines 1316-1317)."""
        tools = Tools()

        @tools.action("None action")
        async def none_action():
            return None

        from pydantic import create_model

        DynamicModel = create_model(
            "NoneModel",
            __base__=ActionModel,
            none_action=(dict | None, None),
        )
        action = DynamicModel(none_action={})

        result = await tools.act(
            action=action,
            browser_session=_make_mock_browser_session(),
        )
        assert isinstance(result, ActionResult)

    @pytest.mark.asyncio
    async def test_act_invalid_result_type(self):
        """Action returns invalid type -> ValueError (lines 1318-1319)."""
        tools = Tools()

        @tools.action("Bad action")
        async def bad_action():
            return 42  # Invalid return type

        from pydantic import create_model

        DynamicModel = create_model(
            "BadModel",
            __base__=ActionModel,
            bad_action=(dict | None, None),
        )
        action = DynamicModel(bad_action={})

        with pytest.raises(ValueError, match="Invalid action result type"):
            await tools.act(
                action=action,
                browser_session=_make_mock_browser_session(),
            )


# ---------------------------------------------------------------------------
# Line 1444: CodeAgentTools._register_code_use_done_action with output_model
# ---------------------------------------------------------------------------


class TestCodeAgentToolsStructuredOutput:
    def test_code_agent_with_output_model_returns_early(self):
        """output_model is not None -> _register_code_use_done_action returns early (line 1444)."""
        class MyOutput(BaseModel):
            answer: str

        tools = CodeAgentTools(output_model=MyOutput)
        assert "done" in tools.registry.registry.actions

    @pytest.mark.asyncio
    async def test_structured_output_done_action(self):
        """Test structured output done action with enum values (lines 1181-1195)."""
        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        class MyOutput(BaseModel):
            color: Color
            name: str

        tools = Tools(output_model=MyOutput)

        from openbrowser.tools.views import StructuredOutputAction

        result = await tools.registry.execute_action(
            "done",
            {"success": True, "data": {"color": "red", "name": "test"}},
        )
        assert result.is_done is True
        assert result.success is True
        data = json.loads(result.extracted_content)
        assert data["color"] == "red"
        assert data["name"] == "test"


# ---------------------------------------------------------------------------
# Lines 1504-1508, 1514-1532: CodeAgentTools done file handling edge cases
# ---------------------------------------------------------------------------


class TestCodeAgentDoneFileHandling:
    @pytest.mark.asyncio
    async def test_code_agent_done_file_read_exception(self):
        """File exists but reading fails (lines 1504-1505)."""
        tools = CodeAgentTools()
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value=None)
        fs.get_file = MagicMock(return_value=None)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name
            f.write("some content")

        try:
            # Make reading fail by patching open to raise
            original_open = open
            call_idx = [0]

            def patched_open(*args, **kwargs):
                if args and isinstance(args[0], str) and args[0] == temp_path:
                    call_idx[0] += 1
                    if call_idx[0] <= 3:
                        raise PermissionError("Cannot read")
                return original_open(*args, **kwargs)

            with patch("builtins.open", side_effect=patched_open):
                result = await tools.registry.execute_action(
                    "done",
                    {"text": "Done", "success": True, "files_to_display": [temp_path]},
                    file_system=fs,
                )
            assert result.is_done is True
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_code_agent_done_file_not_found_anywhere(self):
        """File not found in any location (lines 1507-1508, 1514)."""
        tools = CodeAgentTools()
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value=None)
        fs.get_file = MagicMock(return_value=None)

        result = await tools.registry.execute_action(
            "done",
            {"text": "Done", "success": True, "files_to_display": ["nonexistent.txt"]},
            file_system=fs,
        )
        assert result.is_done is True
        # Agent wanted to display files but none were found -> warning logged

    @pytest.mark.asyncio
    async def test_code_agent_done_display_false_file_not_in_fs_on_disk(self):
        """display_files_in_done_text=False, file not in FileSystem but on disk (lines 1515-1532)."""
        tools = CodeAgentTools(display_files_in_done_text=False)
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value=None)
        fs.get_file = MagicMock(return_value=None)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name
            f.write("disk content")

        try:
            result = await tools.registry.execute_action(
                "done",
                {"text": "Done", "success": True, "files_to_display": [temp_path]},
                file_system=fs,
            )
            assert result.is_done is True
            # Attachment should be the temp_path
            assert any(temp_path in att for att in result.attachments)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_code_agent_done_display_false_file_in_fs_dir(self):
        """display_files_in_done_text=False, file found via fs_dir path (lines 1525-1526)."""
        tools = CodeAgentTools(display_files_in_done_text=False)
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value=None)
        fs.get_file = MagicMock(return_value=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            fs.get_dir = MagicMock(return_value=Path(tmpdir))
            # Create the file in the fs dir
            file_path = os.path.join(tmpdir, "report.txt")
            with open(file_path, "w") as f:
                f.write("report content")

            result = await tools.registry.execute_action(
                "done",
                {"text": "Done", "success": True, "files_to_display": ["report.txt"]},
                file_system=fs,
            )
            assert result.is_done is True

    @pytest.mark.asyncio
    async def test_code_agent_done_display_false_file_not_found(self):
        """display_files_in_done_text=False, file not found anywhere (no attachment added)."""
        tools = CodeAgentTools(display_files_in_done_text=False)
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value=None)
        fs.get_file = MagicMock(return_value=None)

        result = await tools.registry.execute_action(
            "done",
            {"text": "Done", "success": True, "files_to_display": ["nonexistent.txt"]},
            file_system=fs,
        )
        assert result.is_done is True


# ---------------------------------------------------------------------------
# Lines 1543-1548: CodeAgentTools done attachment path resolution
# ---------------------------------------------------------------------------


class TestCodeAgentDoneAttachmentResolution:
    @pytest.mark.asyncio
    async def test_code_agent_done_resolve_absolute_path(self):
        """Attachment is already absolute path (line 1537-1539)."""
        tools = CodeAgentTools()
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value="content")
        fs.get_file = MagicMock(return_value=MagicMock())  # managed by FileSystem

        result = await tools.registry.execute_action(
            "done",
            {"text": "Done", "success": True, "files_to_display": ["managed_file.txt"]},
            file_system=fs,
        )
        assert result.is_done is True
        # Attachment should include the fs dir path
        assert any("/tmp/test_fs" in att for att in result.attachments)

    @pytest.mark.asyncio
    async def test_code_agent_done_resolve_relative_existing(self):
        """Attachment is relative, file exists on disk (lines 1543-1545)."""
        tools = CodeAgentTools()
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=None)  # not managed

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir=os.getcwd()) as f:
            temp_name = os.path.basename(f.name)
            temp_path = f.name
            f.write("content")

        try:
            fs.display_file = MagicMock(return_value="content")

            result = await tools.registry.execute_action(
                "done",
                {"text": "Done", "success": True, "files_to_display": [temp_name]},
                file_system=fs,
            )
            assert result.is_done is True
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_code_agent_done_resolve_nonexistent_fallback(self):
        """Attachment doesn't exist anywhere -> uses fs dir path fallback (lines 1546-1548)."""
        tools = CodeAgentTools()
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value="content")
        fs.get_file = MagicMock(return_value=None)

        result = await tools.registry.execute_action(
            "done",
            {"text": "Done", "success": True, "files_to_display": ["ghost_file.txt"]},
            file_system=fs,
        )
        assert result.is_done is True
        # Should fall back to fs.get_dir() / file_name
        assert any("test_fs" in att for att in result.attachments)


# ---------------------------------------------------------------------------
# Lines 1585-1599, 1603, 1605: CodeAgentTools upload_file path validation
# ---------------------------------------------------------------------------


class TestCodeAgentUploadFilePathValidation:
    @pytest.mark.asyncio
    async def test_upload_file_in_filesystem_service(self):
        """File found in FileSystem service (lines 1585-1589)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=MagicMock())
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "myfile.txt"},
                browser_session=session,
                available_file_paths=["/some/other.txt"],
                file_system=fs,
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_not_in_fs_remote_browser(self):
        """File not in FileSystem, remote browser (lines 1592-1593)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = False
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=None)
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        result = await tools.registry.execute_action(
            "upload_file",
            {"index": 1, "path": "/remote/file.txt"},
            browser_session=session,
            available_file_paths=["/some/other.txt"],
            file_system=fs,
        )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_not_in_fs_local_abs_exists(self):
        """File not in FileSystem, local, absolute path exists (lines 1594-1595)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = True
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=None)
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.isabs", return_value=True), \
             patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/existing.txt"},
                browser_session=session,
                available_file_paths=["/some/other.txt"],
                file_system=fs,
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_not_in_fs_local_error(self):
        """File not in FileSystem, local, relative path -> error (lines 1596-1599)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = True
        fs = _make_mock_file_system()
        fs.get_file = MagicMock(return_value=None)
        fs.get_dir = MagicMock(return_value=Path("/tmp/test_fs"))

        result = await tools.registry.execute_action(
            "upload_file",
            {"index": 1, "path": "relative_file.txt"},
            browser_session=session,
            available_file_paths=["/some/other.txt"],
            file_system=fs,
        )
        assert result.error is not None
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_upload_file_no_fs_remote(self):
        """No FileSystem, remote browser (lines 1602-1603)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = False

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        result = await tools.registry.execute_action(
            "upload_file",
            {"index": 1, "path": "/remote/file.txt"},
            browser_session=session,
            available_file_paths=["/some/other.txt"],
            file_system=None,
        )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_no_fs_local_abs_exists(self):
        """No FileSystem, local, absolute path exists (lines 1604-1605)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = True

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.isabs", return_value=True), \
             patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/local_file.txt"},
                browser_session=session,
                available_file_paths=["/some/other.txt"],
                file_system=None,
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_no_fs_local_error(self):
        """No FileSystem, local, relative path -> error (lines 1606-1609)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = True

        result = await tools.registry.execute_action(
            "upload_file",
            {"index": 1, "path": "relative_file.txt"},
            browser_session=session,
            available_file_paths=["/some/other.txt"],
            file_system=None,
        )
        assert result.error is not None
        assert "not available" in result.error


# ---------------------------------------------------------------------------
# Lines 1632-1640, 1648-1664: CodeAgentTools upload_file find_file_input_near_element
# ---------------------------------------------------------------------------


class TestCodeAgentUploadFileFindInput:
    @pytest.mark.asyncio
    async def test_code_agent_upload_find_in_descendants(self):
        """File input found in descendants (lines 1631-1640)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        child = _make_mock_node("input", index=2, attributes={"type": "file"})
        parent = _make_mock_node("div", index=1, children=[child])
        child.parent_node = parent

        def is_file_input_fn(n):
            return n.node_id == 2
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: parent})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_code_agent_upload_find_in_siblings(self):
        """File input found in sibling (lines 1652-1660)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        grandparent = _make_mock_node("div", index=10)
        selected = _make_mock_node("div", index=1, parent=grandparent)
        sibling = _make_mock_node("input", index=3, parent=grandparent)
        grandparent.children_nodes = [selected, sibling]

        def is_file_input_fn(n):
            return n.node_id == 3
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: selected})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_code_agent_upload_find_in_sibling_descendants(self):
        """File input found in sibling's descendants (lines 1658-1660)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        grandparent = _make_mock_node("div", index=10)
        selected = _make_mock_node("div", index=1, parent=grandparent)
        sibling = _make_mock_node("div", index=3, parent=grandparent)
        nested_file_input = _make_mock_node("input", index=4, parent=sibling)
        sibling.children_nodes = [nested_file_input]
        grandparent.children_nodes = [selected, sibling]

        def is_file_input_fn(n):
            return n.node_id == 4
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: selected})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_code_agent_upload_parent_node_none_break(self):
        """current.parent_node is None -> break (line 1662-1663).
        execute_action wraps BrowserError in RuntimeError."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        # Node with no parent
        node = _make_mock_node("div", index=1)
        assert node.parent_node is None

        # No file input found locally, and no parent to climb
        session.is_file_input = MagicMock(return_value=False)
        session.get_selector_map = AsyncMock(return_value={1: node})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": 0}}
        )

        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="No file upload element"):
                await tools.registry.execute_action(
                    "upload_file",
                    {"index": 1, "path": "/tmp/test.txt"},
                    browser_session=session,
                    available_file_paths=["/tmp/test.txt"],
                )


# ---------------------------------------------------------------------------
# Lines 1675-1711: CodeAgentTools upload_file fallback via scroll position
# ---------------------------------------------------------------------------


class TestCodeAgentUploadFileFallback:
    @pytest.mark.asyncio
    async def test_code_agent_upload_fallback_finds_closest(self):
        """Fallback finds closest file input via scroll position (lines 1675-1707)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        selected = _make_mock_node("div", index=1)
        file_input = _make_mock_node("input", index=5)
        file_input.absolute_position = DOMRect(x=50.0, y=200.0, width=200.0, height=40.0)

        def is_file_input_fn(n):
            return n.node_id == 5
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: selected, 5: file_input})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": 100}}
        )

        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_code_agent_upload_fallback_cdp_exception(self):
        """CDP scroll position throws exception (lines 1686-1687)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        selected = _make_mock_node("div", index=1)
        file_input = _make_mock_node("input", index=5)
        file_input.absolute_position = DOMRect(x=50.0, y=100.0, width=200.0, height=40.0)

        def is_file_input_fn(n):
            return n.node_id == 5
        session.is_file_input = MagicMock(side_effect=is_file_input_fn)
        session.get_selector_map = AsyncMock(return_value={1: selected, 5: file_input})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            side_effect=RuntimeError("CDP error")
        )

        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/test.txt"},
                browser_session=session,
                available_file_paths=["/tmp/test.txt"],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_code_agent_upload_fallback_no_file_input_raises(self):
        """No file input found on page -> BrowserError (lines 1708-1711).
        execute_action wraps BrowserError in RuntimeError."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        selected = _make_mock_node("div", index=1)
        session.is_file_input = MagicMock(return_value=False)
        session.get_selector_map = AsyncMock(return_value={1: selected})

        cdp = await session.get_or_create_cdp_session()
        cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
            return_value={"result": {"value": 0}}
        )

        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="No file upload element"):
                await tools.registry.execute_action(
                    "upload_file",
                    {"index": 1, "path": "/tmp/test.txt"},
                    browser_session=session,
                    available_file_paths=["/tmp/test.txt"],
                )


# ---------------------------------------------------------------------------
# Lines 1725-1727: CodeAgentTools upload_file dispatch failure
# ---------------------------------------------------------------------------


class TestCodeAgentUploadFileDispatchFailure:
    @pytest.mark.asyncio
    async def test_code_agent_upload_dispatch_failure(self):
        """Upload event dispatch fails -> BrowserError (lines 1725-1727).
        execute_action wraps BrowserError in RuntimeError."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event(side_effect=RuntimeError("upload dispatch error"))
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="Failed to upload file"):
                await tools.registry.execute_action(
                    "upload_file",
                    {"index": 1, "path": "/tmp/test.txt"},
                    browser_session=session,
                    available_file_paths=["/tmp/test.txt"],
                )


# ---------------------------------------------------------------------------
# CodeAgentTools default exclusions
# ---------------------------------------------------------------------------


class TestCodeAgentToolsExclusions:
    def test_default_exclusions(self):
        """Default CodeAgentTools excludes extract, find_text, screenshot, search, file ops."""
        tools = CodeAgentTools()
        actions = tools.registry.registry.actions
        assert "extract" not in actions
        assert "find_text" not in actions
        assert "screenshot" not in actions
        assert "search" not in actions
        assert "write_file" not in actions
        assert "read_file" not in actions
        assert "replace_file" not in actions
        # Should still have these
        assert "click" in actions
        assert "navigate" in actions
        assert "done" in actions

    def test_custom_exclusions(self):
        """Custom exclusions override defaults."""
        tools = CodeAgentTools(exclude_actions=["click"])
        actions = tools.registry.registry.actions
        assert "click" not in actions
        assert "navigate" in actions


# ---------------------------------------------------------------------------
# CodeAgentTools upload_file with no whitelist
# ---------------------------------------------------------------------------


class TestCodeAgentUploadNoWhitelist:
    @pytest.mark.asyncio
    async def test_upload_file_no_whitelist_local_exists(self):
        """No whitelist, local browser, file exists (lines 1611-1615)."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = True

        node = _make_mock_node("input", attributes={"type": "file"})
        session.is_file_input = MagicMock(return_value=True)
        session.get_selector_map = AsyncMock(return_value={1: node})
        event = _make_awaitable_event()
        session.event_bus.dispatch = MagicMock(return_value=event)

        with patch("os.path.exists", return_value=True):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/existing_file.txt"},
                browser_session=session,
                available_file_paths=[],
            )
        assert "uploaded" in result.extracted_content.lower()

    @pytest.mark.asyncio
    async def test_upload_file_no_whitelist_local_not_exists(self):
        """No whitelist, local browser, file doesn't exist."""
        tools = CodeAgentTools()
        session = _make_mock_browser_session()
        session.is_local = True

        with patch("os.path.exists", return_value=False):
            result = await tools.registry.execute_action(
                "upload_file",
                {"index": 1, "path": "/tmp/nonexistent.txt"},
                browser_session=session,
                available_file_paths=[],
            )
        assert result.error is not None
        assert "does not exist" in result.error


# ---------------------------------------------------------------------------
# CodeAgentTools done with file in fs_dir path (display_files_in_done_text=True)
# ---------------------------------------------------------------------------


class TestCodeAgentDoneFileInFsDir:
    @pytest.mark.asyncio
    async def test_done_file_found_via_fs_dir(self):
        """File not in FileSystem display, but found via fs_dir path (lines 1484-1486)."""
        tools = CodeAgentTools()
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value=None)
        fs.get_file = MagicMock(return_value=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            fs.get_dir = MagicMock(return_value=Path(tmpdir))
            file_path = os.path.join(tmpdir, "report.txt")
            with open(file_path, "w") as f:
                f.write("report content from disk")

            result = await tools.registry.execute_action(
                "done",
                {"text": "Done", "success": True, "files_to_display": ["report.txt"]},
                file_system=fs,
            )
        assert result.is_done is True
        assert "report content from disk" in result.extracted_content

    @pytest.mark.asyncio
    async def test_done_file_found_via_cwd(self):
        """File found via current working directory (lines 1487-1488)."""
        tools = CodeAgentTools()
        fs = _make_mock_file_system()
        fs.display_file = MagicMock(return_value=None)
        fs.get_file = MagicMock(return_value=None)
        fs.get_dir = MagicMock(return_value=Path("/nonexistent/dir"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir=os.getcwd()
        ) as f:
            temp_name = os.path.basename(f.name)
            temp_path = f.name
            f.write("cwd content")

        try:
            result = await tools.registry.execute_action(
                "done",
                {"text": "Done", "success": True, "files_to_display": [temp_name]},
                file_system=fs,
            )
            assert result.is_done is True
        finally:
            os.unlink(temp_path)
