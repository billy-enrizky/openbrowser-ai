"""Tests for paint order module (paint_order.py) - comprehensive coverage."""

import logging

import pytest

from openbrowser.dom.serializer.paint_order import (
    PaintOrderRemover,
    Rect,
    RectUnionPure,
)
from openbrowser.dom.views import (
    DOMRect,
    EnhancedDOMTreeNode,
    EnhancedSnapshotNode,
    NodeType,
    SimplifiedNode,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_snapshot(bounds=None, paint_order=None, computed_styles=None):
    return EnhancedSnapshotNode(
        is_clickable=None,
        cursor_style=None,
        bounds=bounds or DOMRect(x=0, y=0, width=100, height=50),
        clientRects=None,
        scrollRects=None,
        computed_styles=computed_styles,
        paint_order=paint_order,
        stacking_contexts=None,
    )


def _make_node(tag="div", snapshot=None, children=None):
    _id_counter = getattr(_make_node, "_counter", 0) + 1
    _make_node._counter = _id_counter

    return EnhancedDOMTreeNode(
        node_id=_id_counter,
        backend_node_id=_id_counter + 40000,
        node_type=NodeType.ELEMENT_NODE,
        node_name=tag,
        node_value="",
        attributes={},
        is_scrollable=None,
        is_visible=True,
        absolute_position=None,
        target_id="target-1",
        frame_id=None,
        session_id=None,
        content_document=None,
        shadow_root_type=None,
        shadow_roots=None,
        parent_node=None,
        children_nodes=children,
        ax_node=None,
        snapshot_node=snapshot,
    )


# ────────────────────────────────────────────────────────────────────
# Rect data class
# ────────────────────────────────────────────────────────────────────

class TestRect:
    """Test Rect dataclass (lines 11-33)."""

    def test_valid_rect(self):
        r = Rect(0, 0, 10, 10)
        assert r.x1 == 0
        assert r.y2 == 10

    def test_invalid_rect_post_init(self):
        """Invalid rect (x1 > x2) should trigger __post_init__ returning False."""
        r = Rect(10, 0, 5, 10)  # x1 > x2
        # __post_init__ returns False but doesn't raise
        assert r.x1 == 10

    def test_area(self):
        r = Rect(0, 0, 10, 20)
        assert r.area() == 200.0

    def test_area_zero(self):
        r = Rect(5, 5, 5, 5)
        assert r.area() == 0.0

    def test_intersects_true(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(5, 5, 15, 15)
        assert a.intersects(b) is True

    def test_intersects_false_no_overlap(self):
        a = Rect(0, 0, 5, 5)
        b = Rect(10, 10, 20, 20)
        assert a.intersects(b) is False

    def test_intersects_edge_touch(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(10, 0, 20, 10)
        # Edge-touching rectangles should NOT intersect (x2 <= other.x1)
        assert a.intersects(b) is False

    def test_contains_true(self):
        a = Rect(0, 0, 20, 20)
        b = Rect(5, 5, 10, 10)
        assert a.contains(b) is True

    def test_contains_false(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(5, 5, 15, 15)
        assert a.contains(b) is False

    def test_contains_same_rect(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(0, 0, 10, 10)
        assert a.contains(b) is True


# ────────────────────────────────────────────────────────────────────
# RectUnionPure
# ────────────────────────────────────────────────────────────────────

class TestRectUnionPure:
    """Test RectUnionPure (lines 36-128)."""

    def test_empty_union_contains_false(self):
        u = RectUnionPure()
        assert u.contains(Rect(0, 0, 10, 10)) is False

    def test_add_first_rect(self):
        u = RectUnionPure()
        result = u.add(Rect(0, 0, 10, 10))
        assert result is True

    def test_add_duplicate_rect(self):
        u = RectUnionPure()
        u.add(Rect(0, 0, 10, 10))
        result = u.add(Rect(0, 0, 10, 10))
        assert result is False  # Already covered

    def test_add_subset_rect(self):
        u = RectUnionPure()
        u.add(Rect(0, 0, 20, 20))
        result = u.add(Rect(5, 5, 10, 10))
        assert result is False  # Already contained

    def test_contains_after_add(self):
        u = RectUnionPure()
        u.add(Rect(0, 0, 10, 10))
        assert u.contains(Rect(2, 2, 8, 8)) is True

    def test_contains_non_covered(self):
        u = RectUnionPure()
        u.add(Rect(0, 0, 10, 10))
        assert u.contains(Rect(15, 15, 20, 20)) is False

    def test_add_partially_overlapping(self):
        u = RectUnionPure()
        u.add(Rect(0, 0, 10, 10))
        result = u.add(Rect(5, 5, 15, 15))
        assert result is True  # New area added

    def test_add_non_overlapping(self):
        u = RectUnionPure()
        u.add(Rect(0, 0, 10, 10))
        result = u.add(Rect(20, 20, 30, 30))
        assert result is True

    def test_split_diff_bottom_slice(self):
        u = RectUnionPure()
        a = Rect(0, 0, 10, 10)
        b = Rect(0, 5, 10, 15)
        parts = u._split_diff(a, b)
        # Should have bottom slice (y from 0 to 5)
        assert any(p.y1 == 0 and p.y2 == 5 for p in parts)

    def test_split_diff_top_slice(self):
        u = RectUnionPure()
        a = Rect(0, 5, 10, 15)
        b = Rect(0, 0, 10, 10)
        parts = u._split_diff(a, b)
        # Should have top slice (y from 10 to 15)
        assert any(p.y1 == 10 and p.y2 == 15 for p in parts)

    def test_split_diff_left_slice(self):
        u = RectUnionPure()
        a = Rect(0, 0, 10, 10)
        b = Rect(5, 0, 15, 10)
        parts = u._split_diff(a, b)
        # Should have left slice
        assert any(p.x1 == 0 and p.x2 == 5 for p in parts)

    def test_split_diff_right_slice(self):
        u = RectUnionPure()
        a = Rect(5, 0, 15, 10)
        b = Rect(0, 0, 10, 10)
        parts = u._split_diff(a, b)
        # Should have right slice
        assert any(p.x1 == 10 and p.x2 == 15 for p in parts)

    def test_split_diff_center_removal(self):
        """When b is in the center of a, should get 4 slices."""
        u = RectUnionPure()
        a = Rect(0, 0, 20, 20)
        b = Rect(5, 5, 15, 15)
        parts = u._split_diff(a, b)
        assert len(parts) == 4

    def test_multiple_rects_full_coverage(self):
        u = RectUnionPure()
        u.add(Rect(0, 0, 10, 10))
        u.add(Rect(10, 0, 20, 10))
        u.add(Rect(0, 10, 10, 20))
        u.add(Rect(10, 10, 20, 20))
        assert u.contains(Rect(0, 0, 20, 20)) is True


# ────────────────────────────────────────────────────────────────────
# PaintOrderRemover
# ────────────────────────────────────────────────────────────────────

class TestPaintOrderRemover:
    """Test PaintOrderRemover (lines 131-197)."""

    def test_empty_tree(self):
        node = _make_node(snapshot=_make_snapshot(paint_order=None, bounds=None))
        node.snapshot_node = None
        root = SimplifiedNode(original_node=node, children=[])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        assert root.ignored_by_paint_order is False

    def test_single_node_not_ignored(self):
        snap = _make_snapshot(bounds=DOMRect(0, 0, 100, 100), paint_order=1)
        node = _make_node(snapshot=snap)
        root = SimplifiedNode(original_node=node, children=[])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        assert root.ignored_by_paint_order is False

    def test_occluded_node_ignored(self):
        """Node with lower paint order covered by higher paint order node should be ignored."""
        # Foreground node (paint order 2) covers background node (paint order 1)
        fg_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100), paint_order=2,
            computed_styles={"background-color": "rgb(255, 255, 255)", "opacity": "1"},
        )
        bg_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100), paint_order=1,
            computed_styles={"background-color": "rgb(255, 255, 255)", "opacity": "1"},
        )
        fg_node = _make_node(snapshot=fg_snap)
        bg_node = _make_node(snapshot=bg_snap)

        fg_simp = SimplifiedNode(original_node=fg_node, children=[])
        bg_simp = SimplifiedNode(original_node=bg_node, children=[])
        root_node = _make_node(snapshot=_make_snapshot(paint_order=None, bounds=None))
        root_node.snapshot_node = None
        root = SimplifiedNode(original_node=root_node, children=[fg_simp, bg_simp])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        # bg_node should be ignored since fg_node covers it
        assert bg_simp.ignored_by_paint_order is True

    def test_transparent_node_does_not_occlude(self):
        """Node with transparent background should not occlude nodes behind it."""
        fg_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100), paint_order=2,
            computed_styles={"background-color": "rgba(0, 0, 0, 0)", "opacity": "1"},
        )
        bg_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100), paint_order=1,
            computed_styles={"background-color": "rgb(255, 255, 255)", "opacity": "1"},
        )
        fg_node = _make_node(snapshot=fg_snap)
        bg_node = _make_node(snapshot=bg_snap)

        fg_simp = SimplifiedNode(original_node=fg_node, children=[])
        bg_simp = SimplifiedNode(original_node=bg_node, children=[])
        root_node = _make_node(snapshot=_make_snapshot(paint_order=None, bounds=None))
        root_node.snapshot_node = None
        root = SimplifiedNode(original_node=root_node, children=[fg_simp, bg_simp])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        # bg_node should NOT be ignored (fg is transparent)
        assert bg_simp.ignored_by_paint_order is False

    def test_low_opacity_does_not_occlude(self):
        """Node with low opacity should not occlude nodes behind it."""
        fg_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100), paint_order=2,
            computed_styles={"background-color": "rgb(255, 255, 255)", "opacity": "0.5"},
        )
        bg_snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100), paint_order=1,
            computed_styles={"background-color": "rgb(255, 255, 255)", "opacity": "1"},
        )
        fg_node = _make_node(snapshot=fg_snap)
        bg_node = _make_node(snapshot=bg_snap)

        fg_simp = SimplifiedNode(original_node=fg_node, children=[])
        bg_simp = SimplifiedNode(original_node=bg_node, children=[])
        root_node = _make_node(snapshot=_make_snapshot(paint_order=None, bounds=None))
        root_node.snapshot_node = None
        root = SimplifiedNode(original_node=root_node, children=[fg_simp, bg_simp])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        assert bg_simp.ignored_by_paint_order is False

    def test_node_no_snapshot_skipped(self):
        """Node without snapshot data should be skipped in paint order calculation."""
        node = _make_node(snapshot=None)
        node.snapshot_node = None
        root = SimplifiedNode(original_node=node, children=[])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        assert root.ignored_by_paint_order is False

    def test_node_no_bounds_skipped(self):
        snap = _make_snapshot(bounds=None, paint_order=1)
        node = _make_node(snapshot=snap)
        root = SimplifiedNode(original_node=node, children=[])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        assert root.ignored_by_paint_order is False

    def test_nested_children_collected(self):
        """Paint order calculation should collect nodes from nested children."""
        inner_snap = _make_snapshot(bounds=DOMRect(0, 0, 50, 50), paint_order=1)
        inner = _make_node(snapshot=inner_snap)
        inner_simp = SimplifiedNode(original_node=inner, children=[])

        outer_snap = _make_snapshot(bounds=DOMRect(0, 0, 100, 100), paint_order=2,
                                    computed_styles={"background-color": "rgb(255,255,255)", "opacity": "1"})
        outer = _make_node(snapshot=outer_snap)
        outer_simp = SimplifiedNode(original_node=outer, children=[inner_simp])

        root_node = _make_node(snapshot=_make_snapshot(paint_order=None, bounds=None))
        root_node.snapshot_node = None
        root = SimplifiedNode(original_node=root_node, children=[outer_simp])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        # Inner should be ignored because outer covers it with higher paint order
        assert inner_simp.ignored_by_paint_order is True

    def test_no_computed_styles_node_still_added(self):
        """Node without computed_styles should still be added to rect union."""
        snap = _make_snapshot(
            bounds=DOMRect(0, 0, 100, 100), paint_order=1,
            computed_styles=None,
        )
        node = _make_node(snapshot=snap)
        root = SimplifiedNode(original_node=node, children=[])
        remover = PaintOrderRemover(root)
        remover.calculate_paint_order()
        # Should not crash and not be ignored
        assert root.ignored_by_paint_order is False
