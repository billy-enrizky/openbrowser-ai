"""Tests for DOM views data classes."""

import logging
from unittest.mock import MagicMock, patch

from openbrowser.dom.views import (
    DOMInteractedElement,
    DOMRect,
    NodeType,
    SerializedDOMState,
)

logger = logging.getLogger(__name__)


class TestSerializedDOMState:
    """Tests for SerializedDOMState."""

    def test_llm_representation_empty_root(self):
        """llm_representation returns fallback message when root is None."""
        state = SerializedDOMState(_root=None, selector_map={})
        result = state.llm_representation()
        assert "Empty DOM tree" in result

    def test_eval_representation_empty_root(self):
        """eval_representation returns fallback message when root is None."""
        state = SerializedDOMState(_root=None, selector_map={})
        result = state.eval_representation()
        assert "Empty DOM tree" in result

    def test_llm_representation_with_root(self):
        """llm_representation calls DOMTreeSerializer when root exists."""
        mock_root = MagicMock()
        state = SerializedDOMState(_root=mock_root, selector_map={1: MagicMock()})

        mock_serializer_cls = MagicMock()
        mock_serializer_cls.serialize_tree.return_value = "<div>hello</div>"
        with patch("openbrowser.dom.serializer.serializer.DOMTreeSerializer", mock_serializer_cls):
            result = state.llm_representation()
            assert result == "<div>hello</div>"
            mock_serializer_cls.serialize_tree.assert_called_once()

    def test_eval_representation_with_root(self):
        """eval_representation calls DOMEvalSerializer when root exists."""
        mock_root = MagicMock()
        state = SerializedDOMState(_root=mock_root, selector_map={})

        mock_serializer_cls = MagicMock()
        mock_serializer_cls.serialize_tree.return_value = "<span>eval</span>"
        with patch("openbrowser.dom.serializer.eval_serializer.DOMEvalSerializer", mock_serializer_cls):
            result = state.eval_representation()
            assert result == "<span>eval</span>"
            mock_serializer_cls.serialize_tree.assert_called_once()


class TestDOMInteractedElement:
    """Tests for DOMInteractedElement serialization."""

    def test_to_dict_with_bounds(self):
        """to_dict includes bounds as dictionary."""
        rect = DOMRect(x=10.0, y=20.0, width=100.0, height=50.0)
        elem = DOMInteractedElement(
            node_id=1,
            backend_node_id=100,
            frame_id="ABCD",
            node_type=NodeType.ELEMENT_NODE,
            node_value="",
            node_name="BUTTON",
            attributes={"id": "submit"},
            bounds=rect,
            x_path="button",
            element_hash=12345,
        )
        d = elem.to_dict()
        assert d["node_id"] == 1
        assert d["backend_node_id"] == 100
        assert d["bounds"]["x"] == 10.0
        assert d["attributes"]["id"] == "submit"

    def test_to_dict_without_bounds(self):
        """to_dict handles None bounds."""
        elem = DOMInteractedElement(
            node_id=2,
            backend_node_id=200,
            frame_id=None,
            node_type=NodeType.TEXT_NODE,
            node_value="hello",
            node_name="#text",
            attributes=None,
            bounds=None,
            x_path="",
            element_hash=67890,
        )
        d = elem.to_dict()
        assert d["bounds"] is None
        assert d["frame_id"] is None


class TestDOMRect:
    """Tests for DOMRect."""

    def test_dom_rect_to_dict(self):
        """DOMRect.to_dict returns correct dictionary."""
        rect = DOMRect(x=10.0, y=20.0, width=100.0, height=50.0)
        d = rect.to_dict()
        assert d == {"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0}

    def test_dom_rect_json(self):
        """DOMRect.__json__ returns same as to_dict."""
        rect = DOMRect(x=1.0, y=2.0, width=3.0, height=4.0)
        assert rect.__json__() == rect.to_dict()


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_values(self):
        """NodeType enum has expected values."""
        assert NodeType.ELEMENT_NODE == 1
        assert NodeType.TEXT_NODE == 3
        assert NodeType.DOCUMENT_NODE == 9
