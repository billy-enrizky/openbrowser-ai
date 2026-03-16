"""Tests for openbrowser.tools.utils module."""

import logging
from unittest.mock import MagicMock, PropertyMock

import pytest

from openbrowser.tools.utils import get_click_description

logger = logging.getLogger(__name__)


class TestGetClickDescription:
    """Tests for the get_click_description utility function."""

    def _make_node(
        self,
        tag_name="div",
        attributes=None,
        text="",
        children=None,
        ax_node=None,
        is_visible=True,
        snapshot_node=None,
    ):
        """Helper to build a mock EnhancedDOMTreeNode."""
        node = MagicMock()
        node.tag_name = tag_name
        node.attributes = attributes or {}
        node.get_all_children_text = MagicMock(return_value=text)
        node.children = children or []
        node.ax_node = ax_node
        node.is_visible = is_visible
        node.snapshot_node = snapshot_node
        return node

    # -- Basic tag name --

    def test_basic_tag_name(self):
        node = self._make_node(tag_name="button")
        desc = get_click_description(node)
        assert desc.startswith("button")

    def test_div_with_text(self):
        node = self._make_node(tag_name="div", text="Hello World")
        desc = get_click_description(node)
        assert "div" in desc
        assert '"Hello World"' in desc

    def test_long_text_is_truncated(self):
        long_text = "A" * 50
        node = self._make_node(tag_name="span", text=long_text)
        desc = get_click_description(node)
        assert "..." in desc
        assert len(desc) < len(long_text) + 50

    # -- Input types --

    def test_input_with_type(self):
        node = self._make_node(tag_name="input", attributes={"type": "text"})
        desc = get_click_description(node)
        assert "type=text" in desc

    def test_input_checkbox_unchecked(self):
        node = self._make_node(
            tag_name="input",
            attributes={"type": "checkbox", "checked": "false"},
        )
        desc = get_click_description(node)
        assert "checkbox-state=unchecked" in desc

    def test_input_checkbox_checked_via_attribute(self):
        node = self._make_node(
            tag_name="input",
            attributes={"type": "checkbox", "checked": "true"},
        )
        desc = get_click_description(node)
        assert "checkbox-state=checked" in desc

    def test_input_checkbox_checked_via_ax_node(self):
        ax_prop = MagicMock()
        ax_prop.name = "checked"
        ax_prop.value = True

        ax_node = MagicMock()
        ax_node.properties = [ax_prop]

        node = self._make_node(
            tag_name="input",
            attributes={"type": "checkbox", "checked": "false"},
            ax_node=ax_node,
        )
        desc = get_click_description(node)
        assert "checkbox-state=checked" in desc

    def test_input_checkbox_unchecked_via_ax_node(self):
        ax_prop = MagicMock()
        ax_prop.name = "checked"
        ax_prop.value = False

        ax_node = MagicMock()
        ax_node.properties = [ax_prop]

        node = self._make_node(
            tag_name="input",
            attributes={"type": "checkbox", "checked": "true"},
            ax_node=ax_node,
        )
        desc = get_click_description(node)
        assert "checkbox-state=unchecked" in desc

    # -- Role attribute --

    def test_role_attribute(self):
        node = self._make_node(tag_name="div", attributes={"role": "button"})
        desc = get_click_description(node)
        assert "role=button" in desc

    def test_role_checkbox_unchecked(self):
        node = self._make_node(
            tag_name="div",
            attributes={"role": "checkbox", "aria-checked": "false"},
        )
        desc = get_click_description(node)
        assert "role=checkbox" in desc
        assert "checkbox-state=unchecked" in desc

    def test_role_checkbox_checked(self):
        node = self._make_node(
            tag_name="div",
            attributes={"role": "checkbox", "aria-checked": "true"},
        )
        desc = get_click_description(node)
        assert "checkbox-state=checked" in desc

    def test_role_checkbox_checked_via_ax_node(self):
        ax_prop = MagicMock()
        ax_prop.name = "checked"
        ax_prop.value = "true"

        ax_node = MagicMock()
        ax_node.properties = [ax_prop]

        node = self._make_node(
            tag_name="div",
            attributes={"role": "checkbox", "aria-checked": "false"},
            ax_node=ax_node,
        )
        desc = get_click_description(node)
        assert "checkbox-state=checked" in desc

    # -- Label/span/div with hidden checkbox child --

    def test_label_with_hidden_checkbox_child(self):
        # Create a child checkbox that is hidden
        child = MagicMock()
        child.tag_name = "input"
        child.attributes = {"type": "checkbox", "checked": "true"}
        child.is_visible = False
        child.ax_node = None
        child.snapshot_node = None

        node = self._make_node(
            tag_name="label",
            text="Accept Terms",
            children=[child],
        )
        desc = get_click_description(node)
        assert "checkbox-state=checked" in desc

    def test_label_with_hidden_checkbox_via_opacity(self):
        child = MagicMock()
        child.tag_name = "input"
        child.attributes = {"type": "checkbox", "checked": "false"}
        child.is_visible = True  # Visible but opacity 0
        child.ax_node = None
        snapshot = MagicMock()
        snapshot.computed_styles = {"opacity": "0"}
        child.snapshot_node = snapshot

        node = self._make_node(
            tag_name="span",
            text="Option",
            children=[child],
        )
        desc = get_click_description(node)
        assert "checkbox-state=unchecked" in desc

    def test_label_with_visible_checkbox_not_included(self):
        """Visible checkboxes in label children should not be detected as hidden."""
        child = MagicMock()
        child.tag_name = "input"
        child.attributes = {"type": "checkbox", "checked": "true"}
        child.is_visible = True
        child.ax_node = None
        child.snapshot_node = None

        node = self._make_node(
            tag_name="label",
            text="Visible checkbox",
            children=[child],
        )
        desc = get_click_description(node)
        # Should NOT include checkbox-state because child is visible
        assert "checkbox-state" not in desc

    # -- Key attributes --

    def test_id_attribute(self):
        node = self._make_node(tag_name="button", attributes={"id": "submit-btn"})
        desc = get_click_description(node)
        assert "id=submit-btn" in desc

    def test_name_attribute(self):
        node = self._make_node(tag_name="input", attributes={"name": "email"})
        desc = get_click_description(node)
        assert "name=email" in desc

    def test_aria_label_attribute(self):
        node = self._make_node(
            tag_name="button",
            attributes={"aria-label": "Close dialog"},
        )
        desc = get_click_description(node)
        assert "aria-label=Close dialog" in desc

    def test_attribute_value_truncated_at_20_chars(self):
        node = self._make_node(
            tag_name="div",
            attributes={"id": "a" * 10 + "b" * 20},
        )
        desc = get_click_description(node)
        # The id should be truncated to 20 chars — the full 30-char value must NOT appear
        assert "a" * 10 + "b" * 20 not in desc
        assert "a" * 10 + "b" * 10 in desc

    # -- Empty text --

    def test_empty_text_not_included(self):
        node = self._make_node(tag_name="div", text="   ")
        desc = get_click_description(node)
        # Stripped whitespace should not produce a quoted text part
        assert '"' not in desc
