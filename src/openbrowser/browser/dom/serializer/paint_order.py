"""Paint order filtering to remove hidden/overlapping elements.

This module provides algorithms for determining element visibility based
on paint order (z-index stacking). Elements covered by higher paint order
elements with opaque backgrounds are filtered out.

The algorithm processes elements in reverse paint order (highest first),
building a union of covered rectangles. Elements fully covered by the
union are marked as hidden.

Classes:
    Rect: Axis-aligned rectangle for geometry operations.
    RectUnionPure: Maintains disjoint set of rectangles.
    PaintOrderRemover: Marks elements hidden by paint order.

Example:
    >>> remover = PaintOrderRemover(simplified_root)
    >>> remover.calculate_paint_order()
    >>> # Nodes now have ignored_by_paint_order=True if hidden
"""

from collections import defaultdict
from dataclasses import dataclass

from src.openbrowser.browser.dom.views import SimplifiedNode


@dataclass(frozen=True, slots=True)
class Rect:
    """Closed axis-aligned rectangle with (x1,y1) bottom-left, (x2,y2) top-right.

    Immutable rectangle used for paint order visibility calculations.
    Coordinates follow screen convention (y increases downward).

    Attributes:
        x1: Left edge x-coordinate.
        y1: Top edge y-coordinate.
        x2: Right edge x-coordinate (must be >= x1).
        y2: Bottom edge y-coordinate (must be >= y1).

    Raises:
        ValueError: If x1 > x2 or y1 > y2.

    Example:
        >>> rect = Rect(0, 0, 100, 50)
        >>> rect.area()
        5000.0
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        """Validate rectangle coordinates.

        Raises:
            ValueError: If coordinates are invalid (x1 > x2 or y1 > y2).
        """
        if not (self.x1 <= self.x2 and self.y1 <= self.y2):
            raise ValueError(f"Invalid rectangle: x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}")

    def area(self) -> float:
        """Calculate rectangle area.

        Returns:
            Area in square units (width * height).
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def intersects(self, other: 'Rect') -> bool:
        """Check if this rectangle intersects with another.

        Rectangles touching at edges only do not intersect.

        Args:
            other: Rectangle to check intersection with.

        Returns:
            True if rectangles have overlapping area.
        """
        return not (self.x2 <= other.x1 or other.x2 <= self.x1 or self.y2 <= other.y1 or other.y2 <= self.y1)

    def contains(self, other: 'Rect') -> bool:
        """Check if this rectangle fully contains another.

        Args:
            other: Rectangle to check containment of.

        Returns:
            True if other is entirely inside this rectangle.
        """
        return self.x1 <= other.x1 and self.y1 <= other.y1 and self.x2 >= other.x2 and self.y2 >= other.y2


class RectUnionPure:
    """Maintains a disjoint set of rectangles for coverage tracking.

    Pure Python implementation without external dependencies.
    Suitable for a few thousand rectangles.

    Used to track which screen areas are covered by opaque elements
    during paint order processing.

    Example:
        >>> union = RectUnionPure()
        >>> union.add(Rect(0, 0, 100, 100))  # Returns True (grew)
        >>> union.contains(Rect(10, 10, 50, 50))  # Returns True (covered)
    """

    __slots__ = ('_rects',)

    def __init__(self):
        self._rects: list[Rect] = []

    def _split_diff(self, a: Rect, b: Rect) -> list[Rect]:
        r"""Return list of up to 4 rectangles representing a \ b.

        Computes the set difference of rectangle a minus rectangle b.
        Assumes a intersects b.

        Args:
            a: Rectangle to subtract from.
            b: Rectangle to subtract.

        Returns:
            List of up to 4 rectangles covering a but not b.
        """
        parts = []

        # Bottom slice
        if a.y1 < b.y1:
            parts.append(Rect(a.x1, a.y1, a.x2, b.y1))
        # Top slice
        if b.y2 < a.y2:
            parts.append(Rect(a.x1, b.y2, a.x2, a.y2))

        # Middle (vertical) strip: y overlap is [max(a.y1,b.y1), min(a.y2,b.y2)]
        y_lo = max(a.y1, b.y1)
        y_hi = min(a.y2, b.y2)

        # Left slice
        if a.x1 < b.x1:
            parts.append(Rect(a.x1, y_lo, b.x1, y_hi))
        # Right slice
        if b.x2 < a.x2:
            parts.append(Rect(b.x2, y_lo, a.x2, y_hi))

        return parts

    def contains(self, r: Rect) -> bool:
        """Check if rectangle r is fully covered by the current union.

        Args:
            r: Rectangle to check coverage of.

        Returns:
            True if r is completely covered by existing rectangles.
        """
        if not self._rects:
            return False

        stack = [r]
        for s in self._rects:
            new_stack = []
            for piece in stack:
                if s.contains(piece):
                    # piece completely gone
                    continue
                if piece.intersects(s):
                    new_stack.extend(self._split_diff(piece, s))
                else:
                    new_stack.append(piece)
            if not new_stack:  # everything eaten – covered
                return True
            stack = new_stack
        return False  # something survived

    def add(self, r: Rect) -> bool:
        """Insert rectangle r unless it is already covered.

        If r is partially covered, only the uncovered portions are added.

        Args:
            r: Rectangle to add to the union.

        Returns:
            True if the union grew (r was not fully covered).
        """
        if self.contains(r):
            return False

        pending = [r]
        i = 0
        while i < len(self._rects):
            s = self._rects[i]
            new_pending = []
            changed = False
            for piece in pending:
                if piece.intersects(s):
                    new_pending.extend(self._split_diff(piece, s))
                    changed = True
                else:
                    new_pending.append(piece)
            pending = new_pending
            if changed:
                # s unchanged; proceed with next existing rectangle
                i += 1
            else:
                i += 1

        # Any left‑over pieces are new, non‑overlapping areas
        self._rects.extend(pending)
        return True


class PaintOrderRemover:
    """Calculates which elements should be removed based on paint order.

    Processes elements in reverse paint order (highest first), tracking
    covered screen areas. Elements fully covered by opaque higher-order
    elements are marked with ignored_by_paint_order=True.

    Transparency handling:
        - Elements with opacity < 0.8 don't block visibility
        - Elements with transparent backgrounds don't block visibility

    Attributes:
        root: Root SimplifiedNode of the tree to process.

    Example:
        >>> remover = PaintOrderRemover(simplified_root)
        >>> remover.calculate_paint_order()
        >>> for node in get_all_nodes(simplified_root):
        ...     if node.ignored_by_paint_order:
        ...         print(f'{node.original_node.tag_name} is hidden')
    """

    def __init__(self, root: SimplifiedNode):
        self.root = root

    def calculate_paint_order(self) -> None:
        """Calculate paint order and mark elements that should be ignored.

        Traverses the tree, groups nodes by paint order, then processes
        from highest to lowest. Elements covered by the accumulated
        opaque area are marked with ignored_by_paint_order=True.

        Note:
            Modifies nodes in place. Does not remove nodes, only marks them.
        """
        all_simplified_nodes_with_paint_order: list[SimplifiedNode] = []

        def collect_paint_order(node: SimplifiedNode) -> None:
            if (
                node.original_node.snapshot_node
                and node.original_node.snapshot_node.paint_order is not None
                and node.original_node.snapshot_node.bounds is not None
            ):
                all_simplified_nodes_with_paint_order.append(node)

            for child in node.children:
                collect_paint_order(child)

        collect_paint_order(self.root)

        grouped_by_paint_order: defaultdict[int, list[SimplifiedNode]] = defaultdict(list)

        for node in all_simplified_nodes_with_paint_order:
            if node.original_node.snapshot_node and node.original_node.snapshot_node.paint_order is not None:
                grouped_by_paint_order[node.original_node.snapshot_node.paint_order].append(node)

        rect_union = RectUnionPure()

        for paint_order, nodes in sorted(grouped_by_paint_order.items(), key=lambda x: -x[0]):
            rects_to_add = []

            for node in nodes:
                if not node.original_node.snapshot_node or not node.original_node.snapshot_node.bounds:
                    continue  # shouldn't happen by how we filter them out in the first place

                bounds = node.original_node.snapshot_node.bounds
                rect = Rect(
                    x1=bounds.x,
                    y1=bounds.y,
                    x2=bounds.x + bounds.width,
                    y2=bounds.y + bounds.height,
                )

                if rect_union.contains(rect):
                    node.ignored_by_paint_order = True

                # don't add to the nodes if opacity is less then 0.95 or background-color is transparent
                computed_styles = node.original_node.snapshot_node.computed_styles
                if computed_styles:
                    bg_color = computed_styles.get('background-color', 'rgba(0, 0, 0, 0)')
                    opacity_str = computed_styles.get('opacity', '1')
                    
                    # Check for transparent background
                    is_transparent_bg = bg_color == 'rgba(0, 0, 0, 0)' or bg_color == 'transparent'
                    
                    # Check opacity
                    try:
                        opacity = float(opacity_str)
                        is_low_opacity = opacity < 0.8
                    except (ValueError, TypeError):
                        is_low_opacity = False

                    if is_transparent_bg or is_low_opacity:
                        continue

                rects_to_add.append(rect)

            for rect in rects_to_add:
                rect_union.add(rect)

        return None

