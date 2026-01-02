"""Enhanced snapshot processing for openbrowser-ai DOM tree extraction.

This module provides stateless functions for parsing Chrome DevTools Protocol
(CDP) DOMSnapshot data to extract visibility, clickability, cursor styles,
and other layout information.

The DOMSnapshot API provides a complete snapshot of the DOM including:
- Bounding boxes for all elements
- Computed styles
- Paint order for visibility determination
- Clickability markers

Constants:
    REQUIRED_COMPUTED_STYLES: CSS properties extracted from snapshots.

Functions:
    build_snapshot_lookup: Create backend_node_id -> EnhancedSnapshotNode map.
    is_element_visible: Check visibility from snapshot data.
    is_element_interactive: Check interactivity from snapshot data.
    get_computed_style_value: Extract specific CSS property values.

Example:
    >>> snapshot = await cdp.DOMSnapshot.captureSnapshot(...)
    >>> lookup = build_snapshot_lookup(snapshot, device_pixel_ratio=2.0)
    >>> node_data = lookup.get(backend_node_id)
    >>> if node_data and is_element_visible(node_data):
    ...     print('Element is visible')
"""

import logging
from typing import Any

from src.openbrowser.browser.dom.views import DOMRect, EnhancedSnapshotNode

logger = logging.getLogger(__name__)

# Only the ESSENTIAL computed styles for interactivity and visibility detection
REQUIRED_COMPUTED_STYLES = [
    # Styles actually used for visibility and interactivity detection
    'display',  # visibility detection
    'visibility',  # visibility detection
    'opacity',  # visibility detection
    'overflow',  # scrollability detection
    'overflow-x',  # scrollability detection
    'overflow-y',  # scrollability detection
    'cursor',  # cursor extraction
    'pointer-events',  # clickability logic
    'position',  # visibility logic
    'background-color',  # visibility logic
]


def _parse_rare_boolean_data(rare_data: dict, index: int) -> bool | None:
    """Parse rare boolean data from CDP snapshot.

    CDP uses 'rare boolean data' format for sparse boolean arrays where
    only indices with True values are stored.

    Args:
        rare_data: RareBooleanData dict with 'index' key containing
            list of indices that are True.
        index: Snapshot node index to check.

    Returns:
        True if index is in the rare data indices, None otherwise.

    Example:
        >>> rare = {'index': [0, 5, 10]}  # Indices 0, 5, 10 are True
        >>> _parse_rare_boolean_data(rare, 5)
        True
        >>> _parse_rare_boolean_data(rare, 3)
        None
    """
    indices = rare_data.get('index', [])
    return index in indices if indices else None


def _parse_computed_styles(strings: list[str], style_indices: list[int]) -> dict[str, str]:
    """Parse computed styles from layout tree using string indices.

    CDP stores computed styles as indices into a shared string table.
    This function resolves those indices to actual style values.

    Args:
        strings: String table from snapshot (snapshot.strings).
        style_indices: Indices into strings for each REQUIRED_COMPUTED_STYLES.

    Returns:
        Dict mapping CSS property name to its computed value.

    Note:
        Only REQUIRED_COMPUTED_STYLES properties are extracted to
        minimize memory usage and processing time.
    """
    styles = {}
    for i, style_index in enumerate(style_indices):
        if i < len(REQUIRED_COMPUTED_STYLES) and 0 <= style_index < len(strings):
            styles[REQUIRED_COMPUTED_STYLES[i]] = strings[style_index]
    return styles


def build_snapshot_lookup(
    snapshot: dict[str, Any],
    device_pixel_ratio: float = 1.0,
) -> dict[int, EnhancedSnapshotNode]:
    """Build a lookup table of backend node ID to enhanced snapshot data.

    Parses CDP DOMSnapshot.captureSnapshot result and creates a lookup table
    mapping backend node IDs to EnhancedSnapshotNode objects containing:
    - Bounding boxes (converted from device pixels to CSS pixels)
    - Computed styles (display, visibility, opacity, cursor, etc.)
    - Paint order for visibility filtering
    - Clickability markers
    - Client and scroll rects

    Args:
        snapshot: CDP DOMSnapshot.captureSnapshot result dict.
        device_pixel_ratio: Device pixel ratio for coordinate conversion
            (e.g., 2.0 for Retina displays).

    Returns:
        Dict mapping backend_node_id to EnhancedSnapshotNode.

    Performance:
        Uses pre-built layout index map to achieve O(n) complexity
        instead of O(n^2) double lookups.

    Example:
        >>> snapshot = await cdp.DOMSnapshot.captureSnapshot(
        ...     computedStyles=REQUIRED_COMPUTED_STYLES,
        ...     includePaintOrder=True
        ... )
        >>> lookup = build_snapshot_lookup(snapshot, device_pixel_ratio=2.0)
        >>> node = lookup.get(123)  # Get data for backend_node_id 123
    """
    snapshot_lookup: dict[int, EnhancedSnapshotNode] = {}

    if not snapshot.get('documents'):
        return snapshot_lookup

    strings = snapshot.get('strings', [])

    for document in snapshot['documents']:
        nodes = document.get('nodes', {})
        layout = document.get('layout', {})

        # Build backend node id to snapshot index lookup
        backend_node_to_snapshot_index = {}
        backend_node_ids = nodes.get('backendNodeId', [])
        for i, backend_node_id in enumerate(backend_node_ids):
            backend_node_to_snapshot_index[backend_node_id] = i

        # PERFORMANCE: Pre-build layout index map to eliminate O(n^2) double lookups
        # Preserve original behavior: use FIRST occurrence for duplicates
        layout_index_map = {}
        node_indices = layout.get('nodeIndex', [])
        for layout_idx, node_index in enumerate(node_indices):
            if node_index not in layout_index_map:  # Only store first occurrence
                layout_index_map[node_index] = layout_idx

        # Build snapshot lookup for each backend node id
        for backend_node_id, snapshot_index in backend_node_to_snapshot_index.items():
            # Parse clickability from rare boolean data
            is_clickable = None
            if 'isClickable' in nodes:
                is_clickable = _parse_rare_boolean_data(nodes['isClickable'], snapshot_index)

            # Initialize layout-related fields
            cursor_style = None
            bounding_box = None
            computed_styles = {}
            paint_order = None
            client_rects = None
            scroll_rects = None
            stacking_contexts = None

            # Look for layout tree node that corresponds to this snapshot node
            if snapshot_index in layout_index_map:
                layout_idx = layout_index_map[snapshot_index]
                
                # Parse bounding box
                bounds_list = layout.get('bounds', [])
                if layout_idx < len(bounds_list):
                    bounds = bounds_list[layout_idx]
                    if len(bounds) >= 4:
                        # CDP coordinates are in device pixels, convert to CSS pixels
                        raw_x, raw_y, raw_width, raw_height = bounds[0], bounds[1], bounds[2], bounds[3]
                        bounding_box = DOMRect(
                            x=raw_x / device_pixel_ratio,
                            y=raw_y / device_pixel_ratio,
                            width=raw_width / device_pixel_ratio,
                            height=raw_height / device_pixel_ratio,
                        )

                # Parse computed styles for this layout node
                styles_list = layout.get('styles', [])
                if layout_idx < len(styles_list):
                    style_indices = styles_list[layout_idx]
                    computed_styles = _parse_computed_styles(strings, style_indices)
                    cursor_style = computed_styles.get('cursor')

                # Extract paint order if available
                paint_orders = layout.get('paintOrders', [])
                if layout_idx < len(paint_orders):
                    paint_order = paint_orders[layout_idx]

                # Extract client rects if available
                client_rects_data = layout.get('clientRects', [])
                if layout_idx < len(client_rects_data):
                    client_rect = client_rects_data[layout_idx]
                    if client_rect and len(client_rect) >= 4:
                        client_rects = DOMRect(
                            x=client_rect[0],
                            y=client_rect[1],
                            width=client_rect[2],
                            height=client_rect[3],
                        )

                # Extract scroll rects if available
                scroll_rects_data = layout.get('scrollRects', [])
                if layout_idx < len(scroll_rects_data):
                    scroll_rect = scroll_rects_data[layout_idx]
                    if scroll_rect and len(scroll_rect) >= 4:
                        scroll_rects = DOMRect(
                            x=scroll_rect[0],
                            y=scroll_rect[1],
                            width=scroll_rect[2],
                            height=scroll_rect[3],
                        )

                # Extract stacking contexts if available
                stacking_ctx_data = layout.get('stackingContexts', {})
                stacking_indices = stacking_ctx_data.get('index', [])
                if layout_idx < len(stacking_indices):
                    stacking_contexts = stacking_indices[layout_idx]

            snapshot_lookup[backend_node_id] = EnhancedSnapshotNode(
                is_clickable=is_clickable,
                cursor_style=cursor_style,
                bounds=bounding_box,
                client_rects=client_rects,
                scroll_rects=scroll_rects,
                computed_styles=computed_styles if computed_styles else None,
                paint_order=paint_order,
                stacking_contexts=stacking_contexts,
            )

    return snapshot_lookup


def is_element_visible(
    snapshot_node: EnhancedSnapshotNode | None,
    viewport_width: float = 1920,
    viewport_height: float = 1080,
) -> bool:
    """Check if an element is visible based on snapshot data.

    Determines visibility by checking:
    1. Computed styles (display:none, visibility:hidden, opacity:0)
    2. Bounding box existence and size (>0 width/height)
    3. Position relative to viewport (not completely off-screen)

    Args:
        snapshot_node: Enhanced snapshot node data, or None.
        viewport_width: Viewport width in CSS pixels.
        viewport_height: Viewport height in CSS pixels.

    Returns:
        True if element appears to be visible, False otherwise.

    Note:
        This is a heuristic check. Elements may be visually hidden
        by other means (z-index, clipping) not detected here.
    """
    if snapshot_node is None:
        return False

    # Check computed styles
    styles = snapshot_node.computed_styles or {}
    
    # Hidden by display
    if styles.get('display') == 'none':
        return False
    
    # Hidden by visibility
    if styles.get('visibility') == 'hidden':
        return False
    
    # Hidden by opacity
    opacity = styles.get('opacity', '1')
    try:
        if float(opacity) <= 0:
            return False
    except ValueError:
        pass

    # Check bounding box
    bounds = snapshot_node.bounds
    if bounds is None:
        return False

    # Zero size
    if bounds.width <= 0 or bounds.height <= 0:
        return False

    # Completely off-screen
    if bounds.x + bounds.width < 0 or bounds.y + bounds.height < 0:
        return False
    if bounds.x > viewport_width or bounds.y > viewport_height:
        return False

    return True


def is_element_interactive(snapshot_node: EnhancedSnapshotNode | None) -> bool:
    """Check if an element is interactive based on snapshot data.

    Determines interactivity by checking:
    1. CDP isClickable marker (from event listeners)
    2. cursor:pointer style (visual interactivity hint)
    3. pointer-events CSS property (not 'none')

    Args:
        snapshot_node: Enhanced snapshot node data, or None.

    Returns:
        True if element appears to be interactive, False otherwise.

    Note:
        This supplements tag-based interactivity detection. An element
        may be interactive even if this returns False (e.g., JavaScript
        event handlers not detected by CDP).
    """
    if snapshot_node is None:
        return False

    # Explicitly marked as clickable by CDP
    if snapshot_node.is_clickable:
        return True

    # Check cursor style for pointer
    if snapshot_node.cursor_style == 'pointer':
        return True

    # Check pointer-events
    styles = snapshot_node.computed_styles or {}
    if styles.get('pointer-events') == 'none':
        return False

    return False


def get_computed_style_value(
    snapshot_node: EnhancedSnapshotNode | None,
    style_name: str,
    default: str | None = None,
) -> str | None:
    """Get a computed style value from snapshot node.

    Safe accessor for computed styles that handles None nodes and
    missing style properties.

    Args:
        snapshot_node: Enhanced snapshot node data, or None.
        style_name: CSS property name (e.g., 'display', 'cursor').
        default: Default value if style not found.

    Returns:
        Computed style value, or default if not available.

    Example:
        >>> display = get_computed_style_value(node, 'display', 'block')
        >>> cursor = get_computed_style_value(node, 'cursor')
    """
    if snapshot_node is None or snapshot_node.computed_styles is None:
        return default
    return snapshot_node.computed_styles.get(style_name, default)

