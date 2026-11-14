"""Tests for tab_id auto-truncation and multi-tab display features."""

from dataclasses import field as dataclass_field
from unittest.mock import MagicMock

import pytest

from openbrowser.browser.views import BrowserStateSummary, TabInfo
from openbrowser.dom.views import SerializedDOMState


def _make_tab(url: str, title: str, target_id: str) -> TabInfo:
	"""Create a TabInfo with the given target_id."""
	return TabInfo(url=url, title=title, target_id=target_id)


def _make_state(url: str, title: str, tabs: list[TabInfo]) -> BrowserStateSummary:
	"""Create a BrowserStateSummary with the given tabs."""
	dom_state = MagicMock(spec=SerializedDOMState)
	dom_state.selector_map = {}
	dom_state.eval_representation.return_value = ''
	return BrowserStateSummary(
		dom_state=dom_state,
		url=url,
		title=title,
		tabs=tabs,
	)


class TestMultiTabDisplay:
	"""Tests for BrowserStateSummary.__str__ multi-tab display."""

	def test_single_tab_no_tab_listing(self):
		tab = _make_tab('https://example.com', 'Example', 'A' * 32)
		state = _make_state('https://example.com', 'Example', [tab])
		output = str(state)
		assert 'Tabs' not in output

	def test_multi_tab_shows_tab_listing(self):
		tab1 = _make_tab('https://example.com', 'Example', 'ABCD1234ABCD1234ABCD1234ABCD1234')
		tab2 = _make_tab('https://httpbin.org', 'HTTPBin', 'EFGH5678EFGH5678EFGH5678EFGH5678')
		state = _make_state('https://example.com', 'Example', [tab1, tab2])
		output = str(state)
		assert 'Tabs (2):' in output
		assert '[1234]' in output
		assert '[5678]' in output

	def test_active_tab_marked(self):
		tab1 = _make_tab('https://example.com', 'Example', 'A' * 28 + 'AAAA')
		tab2 = _make_tab('https://other.com', 'Other', 'B' * 28 + 'BBBB')
		state = _make_state('https://example.com', 'Example', [tab1, tab2])
		output = str(state)
		assert '(active)' in output
		lines = output.split('\n')
		tab_lines = [l for l in lines if '[' in l and ']' in l]
		assert any('(active)' in l and 'Example' in l for l in tab_lines)
		assert not any('(active)' in l and 'Other' in l for l in tab_lines)

	def test_long_title_truncated_with_ellipsis(self):
		long_title = 'A' * 80
		tab1 = _make_tab('https://example.com', long_title, 'A' * 32)
		tab2 = _make_tab('https://other.com', 'Short', 'B' * 32)
		state = _make_state('https://example.com', long_title, [tab1, tab2])
		output = str(state)
		assert '...' in output

	def test_short_title_no_ellipsis(self):
		tab1 = _make_tab('https://example.com', 'Short Title', 'A' * 32)
		tab2 = _make_tab('https://other.com', 'Other', 'B' * 32)
		state = _make_state('https://example.com', 'Short Title', [tab1, tab2])
		output = str(state)
		tab_lines = [l for l in output.split('\n') if 'Short Title' in l]
		assert tab_lines
		assert '...' not in tab_lines[0]


class TestTabIdInput:
	"""Switch and close accept short IDs and full CDP target IDs."""

	def test_switch_accepts_full_target_id(self):
		from openbrowser.tools.views import SwitchTabAction

		full_id = 'F8863CF9860B598226D1F7C08B1AE9CD'
		action = SwitchTabAction(tab_id=full_id)
		assert action.tab_id == full_id

	def test_close_accepts_full_target_id(self):
		from openbrowser.tools.views import CloseTabAction

		full_id = 'AABBCCDD11223344AABBCCDD11223344'
		action = CloseTabAction(tab_id=full_id)
		assert action.tab_id == full_id

	def test_switch_accepts_short_id(self):
		from openbrowser.tools.views import SwitchTabAction

		assert SwitchTabAction(tab_id='E9CD').tab_id == 'E9CD'

	def test_rejects_ids_shorter_than_four_characters(self):
		from pydantic import ValidationError

		from openbrowser.tools.views import CloseTabAction, SwitchTabAction

		with pytest.raises(ValidationError):
			SwitchTabAction(tab_id='ABC')
		with pytest.raises(ValidationError):
			CloseTabAction(tab_id='ABC')
