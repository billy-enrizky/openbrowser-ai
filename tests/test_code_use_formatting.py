"""Comprehensive tests for openbrowser.code_use.formatting module.

Covers: format_browser_state_for_llm() with all branches.
"""

import logging
from unittest.mock import MagicMock, AsyncMock

import pytest

from openbrowser.code_use.formatting import format_browser_state_for_llm

logger = logging.getLogger(__name__)


def _make_state(
    url="https://example.com",
    title="Example",
    tabs=None,
    page_info=None,
    pending_network_requests=None,
    dom_html="<div>Hello</div>",
):
    """Create a mock BrowserStateSummary."""
    state = MagicMock()
    state.url = url
    state.title = title

    # DOM state
    dom_state = MagicMock()
    dom_state.eval_representation.return_value = dom_html
    state.dom_state = dom_state

    # Tabs
    if tabs is None:
        tab = MagicMock()
        tab.url = url
        tab.title = title
        tab.target_id = "ABCDEFGH12345678"
        tabs = [tab]
    state.tabs = tabs

    # Page info
    state.page_info = page_info

    # Network requests
    state.pending_network_requests = pending_network_requests

    return state


def _make_page_info(
    pixels_above=0,
    pixels_below=0,
    viewport_height=800,
    page_height=800,
):
    pi = MagicMock()
    pi.pixels_above = pixels_above
    pi.pixels_below = pixels_below
    pi.viewport_height = viewport_height
    pi.page_height = page_height
    return pi


class TestFormatBrowserStateForLLM:
    @pytest.mark.asyncio
    async def test_basic_state_output(self):
        state = _make_state()
        ns = {"evaluate": lambda: None, "navigate": lambda: None}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "## Browser State" in result
        assert "https://example.com" in result
        assert "Example" in result
        assert "DOM Structure" in result
        assert "<div>Hello</div>" in result

    @pytest.mark.asyncio
    async def test_empty_dom_tree(self):
        state = _make_state(dom_html="")
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "Empty DOM tree" in result

    @pytest.mark.asyncio
    async def test_multiple_tabs_shown(self):
        tab1 = MagicMock()
        tab1.url = "https://example.com"
        tab1.title = "Example"
        tab1.target_id = "AAAA1111BBBB2222"

        tab2 = MagicMock()
        tab2.url = "https://other.com"
        tab2.title = "Other Page"
        tab2.target_id = "CCCC3333DDDD4444"

        state = _make_state(tabs=[tab1, tab2])
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "Tabs:" in result
        assert "2222" in result  # last 4 of target_id
        assert "4444" in result

    @pytest.mark.asyncio
    async def test_single_tab_with_current_marker(self):
        tab1 = MagicMock()
        tab1.url = "https://example.com"
        tab1.title = "Example"
        tab1.target_id = "AAAA1111BBBB2222"

        tab2 = MagicMock()
        tab2.url = "https://other.com"
        tab2.title = "Other Page"
        tab2.target_id = "CCCC3333DDDD4444"

        state = _make_state(url="https://example.com", title="Example", tabs=[tab1, tab2])
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "(current)" in result

    @pytest.mark.asyncio
    async def test_page_info_scroll_shown(self):
        page_info = _make_page_info(
            pixels_above=800,
            pixels_below=1600,
            viewport_height=800,
            page_height=3200,
        )
        state = _make_state(page_info=page_info)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "pages above" in result
        assert "pages below" in result
        assert "total pages" in result

    @pytest.mark.asyncio
    async def test_page_info_no_scroll_below(self):
        page_info = _make_page_info(
            pixels_above=0,
            pixels_below=0,
            viewport_height=800,
            page_height=800,
        )
        state = _make_state(page_info=page_info)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "[Start of page]" in result
        assert "[End of page]" in result

    @pytest.mark.asyncio
    async def test_page_info_scroll_hints_pages_above(self):
        page_info = _make_page_info(
            pixels_above=1600,
            pixels_below=0,
            viewport_height=800,
            page_height=2400,
        )
        state = _make_state(page_info=page_info)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "pages above" in result
        assert "[End of page]" in result

    @pytest.mark.asyncio
    async def test_page_info_total_pages_hidden_when_small(self):
        """Total pages not shown when page_height / viewport_height <= 1.2."""
        page_info = _make_page_info(
            pixels_above=0,
            pixels_below=100,
            viewport_height=800,
            page_height=900,
        )
        state = _make_state(page_info=page_info)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "total pages" not in result

    @pytest.mark.asyncio
    async def test_pending_network_requests_shown(self):
        req1 = MagicMock()
        req1.url = "https://api.example.com/data"
        req1.loading_duration_ms = 1500.0

        req2 = MagicMock()
        req2.url = "https://api.example.com/other"
        req2.loading_duration_ms = 500.0

        state = _make_state(pending_network_requests=[req1, req2])
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "Loading:" in result
        assert "2 network requests" in result
        assert "Tip:" in result

    @pytest.mark.asyncio
    async def test_pending_network_deduplicates(self):
        req1 = MagicMock()
        req1.url = "https://api.example.com/data"
        req1.loading_duration_ms = 1500.0

        req2 = MagicMock()
        req2.url = "https://api.example.com/data"  # duplicate
        req2.loading_duration_ms = 2000.0

        state = _make_state(pending_network_requests=[req1, req2])
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "1 network requests" in result

    @pytest.mark.asyncio
    async def test_more_than_20_requests_shows_overflow(self):
        requests_list = []
        for i in range(25):
            req = MagicMock()
            req.url = f"https://api.example.com/item{i}"
            req.loading_duration_ms = 100.0
            requests_list.append(req)

        state = _make_state(pending_network_requests=requests_list)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "... and" in result
        assert "more" in result

    @pytest.mark.asyncio
    async def test_long_url_truncated_in_network_requests(self):
        req = MagicMock()
        req.url = "https://api.example.com/" + "a" * 100
        req.loading_duration_ms = 100.0

        state = _make_state(pending_network_requests=[req])
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "..." in result

    @pytest.mark.asyncio
    async def test_namespace_variables_shown(self):
        ns = {
            "evaluate": lambda: None,
            "my_var": 42,
            "_private": "hidden",
            "browser": MagicMock(),
        }
        state = _make_state()
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "my_var" in result
        assert "_private" not in result

    @pytest.mark.asyncio
    async def test_code_block_variables_shown_with_details(self):
        ns = {
            "_code_block_vars": {"my_data"},
            "my_data": "hello world value",
        }
        state = _make_state()
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "Code block variables:" in result
        assert "my_data" in result

    @pytest.mark.asyncio
    async def test_code_block_variable_function_display(self):
        ns = {
            "_code_block_vars": {"my_func"},
            "my_func": "(function(){return 1})()",
        }
        state = _make_state()
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "my_func" in result

    @pytest.mark.asyncio
    async def test_code_block_variable_long_value(self):
        ns = {
            "_code_block_vars": {"long_var"},
            "long_var": "a" * 50,  # > 20 chars
        }
        state = _make_state()
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "long_var" in result
        # When first_20 == last_20 (e.g., all same char), no ... is shown
        # Only truncated to first 20 chars
        assert "long_var(str): \"aaaaaaaaaaaaaaaaaaaa\"" in result

    @pytest.mark.asyncio
    async def test_dom_truncation_for_long_content(self):
        long_dom = "x" * 70000
        state = _make_state(dom_html=long_dom)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "DOM truncated" in result
        assert "60000 characters" in result

    @pytest.mark.asyncio
    async def test_viewport_height_zero_no_division_error(self):
        page_info = _make_page_info(
            pixels_above=0,
            pixels_below=0,
            viewport_height=0,
            page_height=0,
        )
        state = _make_state(page_info=page_info)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "## Browser State" in result

    @pytest.mark.asyncio
    async def test_skip_vars_are_hidden(self):
        ns = {
            "browser": MagicMock(),
            "file_system": MagicMock(),
            "np": MagicMock(),
            "pd": MagicMock(),
            "plt": MagicMock(),
            "requests": MagicMock(),
            "BeautifulSoup": MagicMock(),
            "wait": MagicMock(),
            "user_var": 42,
        }
        state = _make_state()
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        # Skip vars (navigate, click, etc.) should not appear in the result
        assert "user_var" in result
        # Internal helper vars that are in the skip list must NOT appear anywhere
        assert "navigate" not in result, "navigate should be filtered from output"
        for skip_name in ["click", "type_text", "wait", "BeautifulSoup"]:
            # These are pre-imported helpers, not user variables - they should be filtered
            assert skip_name not in result, f"{skip_name} should be filtered from output"

    @pytest.mark.asyncio
    async def test_code_block_variable_with_none_value(self):
        ns = {
            "_code_block_vars": {"empty_var"},
            "empty_var": None,
        }
        state = _make_state()
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        # Should not crash even though value is None
        assert "## Browser State" in result

    @pytest.mark.asyncio
    async def test_no_page_info(self):
        state = _make_state(page_info=None)
        ns = {}
        bs = MagicMock()

        result = await format_browser_state_for_llm(state, ns, bs)
        assert "pages above" not in result
        assert "pages below" not in result
