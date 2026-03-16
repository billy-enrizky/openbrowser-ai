"""Tests for openbrowser.agent.prompts module - comprehensive coverage.

Covers SystemPrompt and AgentMessagePrompt classes,
including all prompt generation, page statistics, browser state descriptions,
agent state descriptions, and user message construction.
"""

import logging
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from openbrowser.agent.prompts import AgentMessagePrompt, SystemPrompt
from openbrowser.agent.views import AgentStepInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_browser_state(
    url="https://example.com",
    title="Example Page",
    tabs=None,
    page_info=None,
    dom_state=None,
    is_pdf_viewer=False,
    recent_events=None,
    closed_popup_messages=None,
    screenshot=None,
):
    """Create a mock BrowserStateSummary."""
    state = MagicMock()
    state.url = url
    state.title = title
    state.screenshot = screenshot

    if tabs is None:
        tab = MagicMock()
        tab.url = url
        tab.title = title
        tab.target_id = "target-id-abcdef1234567890"
        tabs = [tab]
    state.tabs = tabs

    state.page_info = page_info
    state.is_pdf_viewer = is_pdf_viewer
    state.recent_events = recent_events
    state.closed_popup_messages = closed_popup_messages or []

    if dom_state is None:
        dom_state = MagicMock()
        dom_state.llm_representation = MagicMock(return_value="[1] button Click me")
        dom_state._root = None
        dom_state.selector_map = {}
    state.dom_state = dom_state

    return state


def _make_file_system(todo_contents="", describe_result="No files"):
    """Create a mock FileSystem."""
    fs = MagicMock()
    fs.get_todo_contents = MagicMock(return_value=todo_contents)
    fs.describe = MagicMock(return_value=describe_result)
    return fs


def _make_page_info(
    viewport_width=1280,
    viewport_height=720,
    page_width=1280,
    page_height=3000,
    scroll_x=0,
    scroll_y=500,
    pixels_above=500,
    pixels_below=1780,
    pixels_left=0,
    pixels_right=0,
):
    """Create a mock PageInfo."""
    pi = MagicMock()
    pi.viewport_width = viewport_width
    pi.viewport_height = viewport_height
    pi.page_width = page_width
    pi.page_height = page_height
    pi.scroll_x = scroll_x
    pi.scroll_y = scroll_y
    pi.pixels_above = pixels_above
    pi.pixels_below = pixels_below
    pi.pixels_left = pixels_left
    pi.pixels_right = pixels_right
    return pi


# ---------------------------------------------------------------------------
# SystemPrompt tests (additional coverage beyond test_agent_service.py)
# ---------------------------------------------------------------------------


class TestSystemPromptCoverage:
    """Additional SystemPrompt tests for full coverage."""

    def test_load_prompt_template_default(self):
        sp = SystemPrompt()
        assert sp.prompt_template is not None
        assert len(sp.prompt_template) > 0

    def test_load_prompt_template_flash(self):
        sp = SystemPrompt(flash_mode=True)
        msg = sp.get_system_message()
        assert msg.content is not None

    def test_load_prompt_template_no_thinking(self):
        sp = SystemPrompt(use_thinking=False)
        msg = sp.get_system_message()
        assert msg.content is not None

    def test_max_actions_formatting(self):
        sp = SystemPrompt(max_actions_per_step=15)
        msg = sp.get_system_message()
        assert "15" in msg.content

    def test_system_message_cache_enabled(self):
        sp = SystemPrompt()
        msg = sp.get_system_message()
        assert msg.cache is True

    def test_override_does_not_load_template(self):
        sp = SystemPrompt(override_system_message="Custom only")
        # Should not have prompt_template since override was used
        assert not hasattr(sp, "prompt_template") or sp.system_message.content == "Custom only"

    def test_extend_appended_to_default(self):
        ext = "EXTRA INSTRUCTION HERE"
        sp = SystemPrompt(extend_system_message=ext)
        msg = sp.get_system_message()
        assert ext in msg.content

    def test_extend_appended_to_override(self):
        sp = SystemPrompt(
            override_system_message="Base",
            extend_system_message="Extension",
        )
        msg = sp.get_system_message()
        assert "Base" in msg.content
        assert "Extension" in msg.content


# ---------------------------------------------------------------------------
# AgentMessagePrompt - _extract_page_statistics
# ---------------------------------------------------------------------------


class TestExtractPageStatistics:
    """Tests for _extract_page_statistics method."""

    def test_no_dom_state(self):
        browser_state = _make_browser_state()
        browser_state.dom_state = None
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        stats = prompt._extract_page_statistics()
        assert stats["total_elements"] == 0

    def test_no_root(self):
        dom_state = MagicMock()
        dom_state._root = None
        dom_state.llm_representation = MagicMock(return_value="content")
        browser_state = _make_browser_state(dom_state=dom_state)

        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        stats = prompt._extract_page_statistics()
        assert stats["total_elements"] == 0

    def test_with_elements(self):
        from openbrowser.dom.views import NodeType

        # Build a mock DOM tree
        root = MagicMock()
        root.original_node = MagicMock()
        root.original_node.node_type = NodeType.ELEMENT_NODE
        root.original_node.tag_name = "div"
        root.original_node.is_actually_scrollable = False
        root.is_interactive = False
        root.is_shadow_host = False
        root.children = []

        # Add a link child
        link_node = MagicMock()
        link_node.original_node = MagicMock()
        link_node.original_node.node_type = NodeType.ELEMENT_NODE
        link_node.original_node.tag_name = "a"
        link_node.original_node.is_actually_scrollable = False
        link_node.is_interactive = True
        link_node.is_shadow_host = False
        link_node.children = []
        root.children.append(link_node)

        # Add an iframe child
        iframe_node = MagicMock()
        iframe_node.original_node = MagicMock()
        iframe_node.original_node.node_type = NodeType.ELEMENT_NODE
        iframe_node.original_node.tag_name = "iframe"
        iframe_node.original_node.is_actually_scrollable = False
        iframe_node.is_interactive = False
        iframe_node.is_shadow_host = False
        iframe_node.children = []
        root.children.append(iframe_node)

        # Add an image child
        img_node = MagicMock()
        img_node.original_node = MagicMock()
        img_node.original_node.node_type = NodeType.ELEMENT_NODE
        img_node.original_node.tag_name = "img"
        img_node.original_node.is_actually_scrollable = False
        img_node.is_interactive = False
        img_node.is_shadow_host = False
        img_node.children = []
        root.children.append(img_node)

        # Add a scrollable div
        scroll_node = MagicMock()
        scroll_node.original_node = MagicMock()
        scroll_node.original_node.node_type = NodeType.ELEMENT_NODE
        scroll_node.original_node.tag_name = "div"
        scroll_node.original_node.is_actually_scrollable = True
        scroll_node.is_interactive = True
        scroll_node.is_shadow_host = False
        scroll_node.children = []
        root.children.append(scroll_node)

        # Add a shadow host
        shadow_host = MagicMock()
        shadow_host.original_node = MagicMock()
        shadow_host.original_node.node_type = NodeType.ELEMENT_NODE
        shadow_host.original_node.tag_name = "custom-element"
        shadow_host.original_node.is_actually_scrollable = False
        shadow_host.is_interactive = False
        shadow_host.is_shadow_host = True

        # Shadow child (open shadow)
        shadow_child = MagicMock()
        shadow_child.original_node = MagicMock()
        shadow_child.original_node.node_type = NodeType.DOCUMENT_FRAGMENT_NODE
        shadow_child.original_node.shadow_root_type = "open"
        shadow_child.is_interactive = False
        shadow_child.is_shadow_host = False
        shadow_child.children = []
        shadow_host.children = [shadow_child]
        root.children.append(shadow_host)

        # Add a closed shadow host
        closed_shadow_host = MagicMock()
        closed_shadow_host.original_node = MagicMock()
        closed_shadow_host.original_node.node_type = NodeType.ELEMENT_NODE
        closed_shadow_host.original_node.tag_name = "closed-element"
        closed_shadow_host.original_node.is_actually_scrollable = False
        closed_shadow_host.is_interactive = False
        closed_shadow_host.is_shadow_host = True

        closed_shadow_child = MagicMock()
        closed_shadow_child.original_node = MagicMock()
        closed_shadow_child.original_node.node_type = NodeType.DOCUMENT_FRAGMENT_NODE
        closed_shadow_child.original_node.shadow_root_type = "closed"
        closed_shadow_child.is_interactive = False
        closed_shadow_child.is_shadow_host = False
        closed_shadow_child.children = []
        closed_shadow_host.children = [closed_shadow_child]
        root.children.append(closed_shadow_host)

        dom_state = MagicMock()
        dom_state._root = root
        dom_state.llm_representation = MagicMock(return_value="[1] button Submit")
        dom_state.selector_map = {}

        browser_state = _make_browser_state(dom_state=dom_state)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        stats = prompt._extract_page_statistics()
        assert stats["links"] == 1
        assert stats["iframes"] == 1
        assert stats["images"] == 1
        assert stats["scroll_containers"] == 1
        assert stats["interactive_elements"] == 2  # link + scrollable div
        assert stats["shadow_open"] == 1
        assert stats["shadow_closed"] == 1
        assert stats["total_elements"] >= 7

    def test_document_fragment_node_no_double_count(self):
        from openbrowser.dom.views import NodeType

        # A standalone document fragment (not as shadow child)
        root = MagicMock()
        root.original_node = MagicMock()
        root.original_node.node_type = NodeType.DOCUMENT_FRAGMENT_NODE
        root.original_node.shadow_root_type = None
        root.is_interactive = False
        root.is_shadow_host = False
        root.children = []

        dom_state = MagicMock()
        dom_state._root = root
        dom_state.llm_representation = MagicMock(return_value="content")

        browser_state = _make_browser_state(dom_state=dom_state)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        stats = prompt._extract_page_statistics()
        assert stats["total_elements"] == 1

    def test_node_with_no_original(self):
        root = MagicMock()
        root.original_node = None
        root.children = []

        dom_state = MagicMock()
        dom_state._root = root
        dom_state.llm_representation = MagicMock(return_value="content")

        browser_state = _make_browser_state(dom_state=dom_state)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        stats = prompt._extract_page_statistics()
        assert stats["total_elements"] == 0


# ---------------------------------------------------------------------------
# AgentMessagePrompt - _get_browser_state_description
# ---------------------------------------------------------------------------


class TestGetBrowserStateDescription:
    """Tests for _get_browser_state_description method."""

    def test_basic_description(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "Interactive elements" in desc

    def test_description_with_page_info_scrolled(self):
        pi = _make_page_info(pixels_above=720, pixels_below=1560)
        browser_state = _make_browser_state(page_info=pi)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "pages above" in desc
        assert "pages below" in desc

    def test_description_at_top_of_page(self):
        pi = _make_page_info(pixels_above=0, pixels_below=2000)
        browser_state = _make_browser_state(page_info=pi)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "[Start of page]" in desc

    def test_description_at_bottom_of_page(self):
        pi = _make_page_info(pixels_above=2000, pixels_below=0)
        browser_state = _make_browser_state(page_info=pi)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "[End of page]" in desc

    def test_description_empty_page(self):
        dom_state = MagicMock()
        dom_state.llm_representation = MagicMock(return_value="")
        dom_state._root = None
        browser_state = _make_browser_state(dom_state=dom_state)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "empty page" in desc

    def test_description_truncated_elements(self):
        dom_state = MagicMock()
        dom_state.llm_representation = MagicMock(return_value="x" * 50000)
        dom_state._root = None
        browser_state = _make_browser_state(dom_state=dom_state)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            max_clickable_elements_length=100,
        )
        desc = prompt._get_browser_state_description()
        assert "truncated" in desc

    def test_description_pdf_viewer(self):
        browser_state = _make_browser_state(is_pdf_viewer=True)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "PDF viewer" in desc

    def test_description_recent_events(self):
        browser_state = _make_browser_state(recent_events="Download started")
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            include_recent_events=True,
        )
        desc = prompt._get_browser_state_description()
        assert "Download started" in desc

    def test_description_recent_events_not_included(self):
        browser_state = _make_browser_state(recent_events="Download started")
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            include_recent_events=False,
        )
        desc = prompt._get_browser_state_description()
        assert "Download started" not in desc

    def test_description_closed_popups(self):
        browser_state = _make_browser_state(
            closed_popup_messages=["Alert: Cookie consent", "Confirm: Delete?"]
        )
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "Cookie consent" in desc
        assert "Delete?" in desc

    def test_description_multiple_tabs(self):
        tab1 = MagicMock()
        tab1.url = "https://example.com"
        tab1.title = "Example"
        tab1.target_id = "abcdefgh12345678"

        tab2 = MagicMock()
        tab2.url = "https://other.com"
        tab2.title = "Other"
        tab2.target_id = "ijklmnop87654321"

        browser_state = _make_browser_state(tabs=[tab1, tab2])
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "5678" in desc  # tab_id[-4:]
        assert "4321" in desc

    def test_description_current_tab_identified(self):
        tab1 = MagicMock()
        tab1.url = "https://example.com"
        tab1.title = "Example"
        tab1.target_id = "abcdefgh12345678"

        browser_state = _make_browser_state(tabs=[tab1], url="https://example.com", title="Example")
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "Current tab" in desc

    def test_description_ambiguous_current_tab(self):
        tab1 = MagicMock()
        tab1.url = "https://example.com"
        tab1.title = "Example"
        tab1.target_id = "abcdefgh12345678"

        tab2 = MagicMock()
        tab2.url = "https://example.com"
        tab2.title = "Example"
        tab2.target_id = "ijklmnop87654321"

        browser_state = _make_browser_state(tabs=[tab1, tab2], url="https://example.com", title="Example")
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        # Ambiguous match (2 tabs with same URL+title), should NOT show "Current tab"
        assert "Current tab" not in desc, (
            "With ambiguous tab matches, current tab should not be identified"
        )

    def test_low_element_count_spa_warning(self):
        from openbrowser.dom.views import NodeType

        # Create a root with fewer than 10 elements
        root = MagicMock()
        root.original_node = MagicMock()
        root.original_node.node_type = NodeType.ELEMENT_NODE
        root.original_node.tag_name = "div"
        root.original_node.is_actually_scrollable = False
        root.is_interactive = False
        root.is_shadow_host = False
        root.children = []

        dom_state = MagicMock()
        dom_state._root = root
        dom_state.llm_representation = MagicMock(return_value="[1] div content")
        dom_state.selector_map = {}

        browser_state = _make_browser_state(dom_state=dom_state)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "empty" in desc.lower() or "page_stats" in desc

    def test_page_info_zero_viewport_height(self):
        pi = _make_page_info(viewport_height=0, pixels_above=0, pixels_below=0)
        browser_state = _make_browser_state(page_info=pi)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "page_info" in desc


# ---------------------------------------------------------------------------
# AgentMessagePrompt - _get_agent_state_description
# ---------------------------------------------------------------------------


class TestGetAgentStateDescription:
    """Tests for _get_agent_state_description method."""

    def test_includes_task(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Search for flights",
        )
        desc = prompt._get_agent_state_description()
        assert "Search for flights" in desc

    def test_includes_step_info(self):
        browser_state = _make_browser_state()
        step_info = AgentStepInfo(step_number=4, max_steps=20)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            step_info=step_info,
        )
        desc = prompt._get_agent_state_description()
        assert "Step5" in desc
        assert "maximum:20" in desc

    def test_no_step_info(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
        )
        desc = prompt._get_agent_state_description()
        assert "Today:" in desc

    def test_includes_sensitive_data(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Login",
            sensitive_data="password: <secret>pass</secret>",
        )
        desc = prompt._get_agent_state_description()
        assert "<sensitive_data>" in desc

    def test_includes_available_file_paths(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Upload file",
            available_file_paths=["/tmp/report.pdf", "/tmp/data.csv"],
        )
        desc = prompt._get_agent_state_description()
        assert "/tmp/report.pdf" in desc
        assert "Use with absolute paths" in desc

    def test_empty_todo_placeholder(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(todo_contents=""),
            task="Task",
        )
        desc = prompt._get_agent_state_description()
        assert "empty todo.md" in desc

    def test_todo_with_content(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(todo_contents="- Step 1\n- Step 2"),
            task="Task",
        )
        desc = prompt._get_agent_state_description()
        assert "Step 1" in desc

    def test_file_system_none(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=None,
            task="Task",
        )
        desc = prompt._get_agent_state_description()
        assert "No file system available" in desc


# ---------------------------------------------------------------------------
# AgentMessagePrompt - get_user_message
# ---------------------------------------------------------------------------


class TestGetUserMessage:
    """Tests for get_user_message method."""

    def test_text_only(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Find price",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert isinstance(msg.content, str)
        assert "agent_history" in msg.content
        assert "browser_state" in msg.content

    def test_with_screenshots(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Find price",
            screenshots=["base64data1", "base64data2"],
        )
        msg = prompt.get_user_message(use_vision=True)
        assert isinstance(msg.content, list)
        # Should have text + labels + images
        assert len(msg.content) >= 3

    def test_new_tab_disables_vision(self):
        browser_state = _make_browser_state(url="about:blank")
        step_info = AgentStepInfo(step_number=0, max_steps=10)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            step_info=step_info,
            screenshots=["base64"],
        )
        msg = prompt.get_user_message(use_vision=True)
        # Should be text-only for about:blank on step 0
        assert isinstance(msg.content, str)

    def test_chrome_newtab_disables_vision(self):
        browser_state = _make_browser_state(url="chrome://newtab")
        step_info = AgentStepInfo(step_number=0, max_steps=10)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            step_info=step_info,
            screenshots=["base64"],
        )
        msg = prompt.get_user_message(use_vision=True)
        assert isinstance(msg.content, str)

    def test_new_tab_multiple_tabs_keeps_vision(self):
        tab1 = MagicMock()
        tab1.url = "about:blank"
        tab1.title = "New Tab"
        tab1.target_id = "abcdefghijklmnop"

        tab2 = MagicMock()
        tab2.url = "https://example.com"
        tab2.title = "Example"
        tab2.target_id = "ijklmnop12345678"

        browser_state = _make_browser_state(url="about:blank", tabs=[tab1, tab2])
        step_info = AgentStepInfo(step_number=0, max_steps=10)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            step_info=step_info,
            screenshots=["base64"],
        )
        msg = prompt.get_user_message(use_vision=True)
        # Multiple tabs, so vision should not be disabled
        assert isinstance(msg.content, list)

    def test_with_page_filtered_actions(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            page_filtered_actions="custom_action: Do something special",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "page_specific_actions" in msg.content

    def test_with_read_state(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            read_state_description="Previous extraction content",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "read_state" in msg.content

    def test_empty_read_state_not_included(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            read_state_description="   ",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "<read_state>" not in msg.content

    def test_with_sample_images(self):
        from openbrowser.llm.messages import ContentPartTextParam

        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            screenshots=["base64data"],
            sample_images=[ContentPartTextParam(text="Sample image label")],
        )
        msg = prompt.get_user_message(use_vision=True)
        assert isinstance(msg.content, list)

    def test_cache_enabled_on_message(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert msg.cache is True

    def test_single_screenshot_label(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            screenshots=["base64data"],
        )
        msg = prompt.get_user_message(use_vision=True)
        assert isinstance(msg.content, list)
        # Check for "Current screenshot:" label
        text_parts = [p for p in msg.content if hasattr(p, "text")]
        has_current_label = any("Current screenshot" in p.text for p in text_parts)
        assert has_current_label

    def test_multiple_screenshots_labels(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            screenshots=["base64_1", "base64_2"],
        )
        msg = prompt.get_user_message(use_vision=True)
        assert isinstance(msg.content, list)
        text_parts = [p for p in msg.content if hasattr(p, "text")]
        has_previous = any("Previous screenshot" in p.text for p in text_parts)
        has_current = any("Current screenshot" in p.text for p in text_parts)
        assert has_previous
        assert has_current

    def test_not_step_zero_keeps_vision(self):
        browser_state = _make_browser_state(url="about:blank")
        step_info = AgentStepInfo(step_number=3, max_steps=10)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            step_info=step_info,
            screenshots=["base64"],
        )
        msg = prompt.get_user_message(use_vision=True)
        # Not step 0, so vision should remain
        assert isinstance(msg.content, list)

    def test_agent_history_description(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            agent_history_description="Step 1: Clicked button\nStep 2: Filled form",
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "Clicked button" in msg.content
        assert "Filled form" in msg.content

    def test_no_agent_history(self):
        browser_state = _make_browser_state()
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
            task="Task",
            agent_history_description=None,
        )
        msg = prompt.get_user_message(use_vision=False)
        assert "agent_history" in msg.content


# ---------------------------------------------------------------------------
# Additional coverage for remaining gaps
# ---------------------------------------------------------------------------


class TestSystemPromptTemplateLoadFailure:
    """Tests for SystemPrompt template loading failure (lines 54-55)."""

    def test_template_load_failure_raises_runtime_error(self):
        with patch("importlib.resources.files", side_effect=FileNotFoundError("not found")):
            with pytest.raises(RuntimeError, match="Failed to load system prompt template"):
                SystemPrompt()


class TestBrowserStateDescriptionWithShadowAndImages:
    """Tests for _get_browser_state_description including shadow DOM and image stats (lines 186, 188)."""

    def test_description_includes_shadow_and_image_stats(self):
        from openbrowser.dom.views import NodeType

        # Build DOM with shadow and images
        root = MagicMock()
        root.original_node = MagicMock()
        root.original_node.node_type = NodeType.ELEMENT_NODE
        root.original_node.tag_name = "div"
        root.original_node.is_actually_scrollable = False
        root.is_interactive = False
        root.is_shadow_host = False
        root.children = []

        # Add several elements to exceed the 10-element threshold
        for i in range(12):
            node = MagicMock()
            node.original_node = MagicMock()
            node.original_node.node_type = NodeType.ELEMENT_NODE
            node.original_node.tag_name = "div"
            node.original_node.is_actually_scrollable = False
            node.is_interactive = False
            node.is_shadow_host = False
            node.children = []
            root.children.append(node)

        # Add image nodes
        for _ in range(3):
            img = MagicMock()
            img.original_node = MagicMock()
            img.original_node.node_type = NodeType.ELEMENT_NODE
            img.original_node.tag_name = "img"
            img.original_node.is_actually_scrollable = False
            img.is_interactive = False
            img.is_shadow_host = False
            img.children = []
            root.children.append(img)

        # Add shadow host
        shadow_host = MagicMock()
        shadow_host.original_node = MagicMock()
        shadow_host.original_node.node_type = NodeType.ELEMENT_NODE
        shadow_host.original_node.tag_name = "custom-element"
        shadow_host.original_node.is_actually_scrollable = False
        shadow_host.is_interactive = False
        shadow_host.is_shadow_host = True

        shadow_child = MagicMock()
        shadow_child.original_node = MagicMock()
        shadow_child.original_node.node_type = NodeType.DOCUMENT_FRAGMENT_NODE
        shadow_child.original_node.shadow_root_type = "open"
        shadow_child.is_interactive = False
        shadow_child.is_shadow_host = False
        shadow_child.children = []
        shadow_host.children = [shadow_child]
        root.children.append(shadow_host)

        dom_state = MagicMock()
        dom_state._root = root
        dom_state.llm_representation = MagicMock(return_value="[1] button Submit")
        dom_state.selector_map = {i: MagicMock() for i in range(15)}

        browser_state = _make_browser_state(dom_state=dom_state)
        prompt = AgentMessagePrompt(
            browser_state_summary=browser_state,
            file_system=_make_file_system(),
        )
        desc = prompt._get_browser_state_description()
        assert "shadow(open)" in desc
        assert "images" in desc
