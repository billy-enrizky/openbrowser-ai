"""Enhanced clickable element detection following browser-use pattern.

This module provides sophisticated detection of interactive/clickable elements
using multiple signals: tag names, ARIA roles, accessibility properties,
event handlers, and visual indicators.

The detection algorithm uses a tiered approach:
1. Skip non-element nodes and html/body
2. Check iframe sizing for scrollability
3. Detect search-related elements
4. Check accessibility properties (focusable, editable, etc.)
5. Check interactive tag names
6. Check event handlers and ARIA roles
7. Check accessibility tree roles
8. Check icon-sized elements with interactive attributes
9. Fallback to cursor:pointer style

Classes:
    ClickableElementDetector: Static methods for interactivity detection.
"""

from openbrowser.browser.dom.views import EnhancedDOMTreeNode, NodeType


class ClickableElementDetector:
    """Detector for interactive/clickable elements with enhanced detection logic.

    Uses multiple signals to determine if an element is interactive:
    - HTML tag name (button, input, a, etc.)
    - ARIA roles and properties
    - Accessibility tree data
    - Event handler attributes
    - Cursor styles
    - Element sizing (for icons)

    Example:
        >>> if ClickableElementDetector.is_interactive(node):
        ...     print(f'{node.tag_name} is clickable')
    """

    @staticmethod
    def is_interactive(node: EnhancedDOMTreeNode) -> bool:
        """Check if this node is clickable/interactive using enhanced scoring.

        Uses a tiered detection approach:
        1. Filter out non-elements and html/body
        2. Check iframe sizing (>100x100 for potential scrolling)
        3. Search element detection (class/id/data attributes)
        4. Accessibility properties (focusable, editable, checked, etc.)
        5. Interactive tag names (button, input, a, select, etc.)
        6. Event handlers (onclick, onkeydown, etc.)
        7. ARIA roles (button, link, checkbox, etc.)
        8. Icon detection (10-50px elements with interactive attrs)
        9. Cursor:pointer fallback

        Args:
            node: EnhancedDOMTreeNode to evaluate.

        Returns:
            True if the element should be considered interactive.

        Note:
            Labels are excluded to avoid destroying real clickable elements
            when the label has a 'for' attribute.
        """

        # Skip non-element nodes
        if node.node_type != NodeType.ELEMENT_NODE:
            return False

        # remove html and body nodes
        if node.tag_name in {'html', 'body'}:
            return False

        # IFRAME elements should be interactive if they're large enough to potentially need scrolling
        # Small iframes (< 100px width or height) are unlikely to have scrollable content
        if node.tag_name and (node.tag_name.upper() == 'IFRAME' or node.tag_name.upper() == 'FRAME'):
            if node.snapshot_node and node.snapshot_node.bounds:
                width = node.snapshot_node.bounds.width
                height = node.snapshot_node.bounds.height
                # Only include iframes larger than 100x100px
                if width > 100 and height > 100:
                    return True

        # RELAXED SIZE CHECK: Allow all elements including size 0 (they might be interactive overlays, etc.)
        # Note: Size 0 elements can still be interactive (e.g., invisible clickable overlays)
        # Visibility is determined separately by CSS styles, not just bounding box size

        # SEARCH ELEMENT DETECTION: Check for search-related classes and attributes
        if node.attributes:
            search_indicators = {
                'search',
                'magnify',
                'glass',
                'lookup',
                'find',
                'query',
                'search-icon',
                'search-btn',
                'search-button',
                'searchbox',
            }

            # Check class names for search indicators
            class_list = node.attributes.get('class', '').lower().split()
            if any(indicator in ' '.join(class_list) for indicator in search_indicators):
                return True

            # Check id for search indicators
            element_id = node.attributes.get('id', '').lower()
            if any(indicator in element_id for indicator in search_indicators):
                return True

            # Check data attributes for search functionality
            for attr_name, attr_value in node.attributes.items():
                if attr_name.startswith('data-') and any(indicator in attr_value.lower() for indicator in search_indicators):
                    return True

        # Enhanced accessibility property checks - direct clear indicators only
        if node.ax_node and node.ax_node.properties:
            for prop in node.ax_node.properties:
                try:
                    # Handle both dict and object formats
                    if isinstance(prop, dict):
                        prop_name = prop.get('name')
                        prop_value = prop.get('value')
                    else:
                        prop_name = getattr(prop, 'name', None)
                        prop_value = getattr(prop, 'value', None)

                    if prop_name is None:
                        continue

                    # aria disabled
                    if prop_name == 'disabled' and prop_value:
                        return False

                    # aria hidden
                    if prop_name == 'hidden' and prop_value:
                        return False

                    # Direct interactiveness indicators
                    if prop_name in ['focusable', 'editable', 'settable'] and prop_value:
                        return True

                    # Interactive state properties (presence indicates interactive widget)
                    if prop_name in ['checked', 'expanded', 'pressed', 'selected']:
                        # These properties only exist on interactive elements
                        return True

                    # Form-related interactiveness
                    if prop_name in ['required', 'autocomplete'] and prop_value:
                        return True

                    # Elements with keyboard shortcuts are interactive
                    if prop_name == 'keyshortcuts' and prop_value:
                        return True
                except (AttributeError, ValueError, TypeError):
                    # Skip properties we can't process
                    continue

        # ENHANCED TAG CHECK: Include truly interactive elements
        # Note: 'label' removed - labels are handled by other attribute checks below
        # Otherwise labels with "for" attribute can destroy the real clickable element
        interactive_tags = {
            'button',
            'input',
            'select',
            'textarea',
            'a',
            'details',
            'summary',
            'option',
            'optgroup',
        }
        # Check with case-insensitive comparison
        if node.tag_name and node.tag_name.lower() in interactive_tags:
            return True

        # Tertiary check: elements with interactive attributes
        if node.attributes:
            # Check for event handlers or interactive attributes
            interactive_attributes = {'onclick', 'onmousedown', 'onmouseup', 'onkeydown', 'onkeyup', 'tabindex'}
            if any(attr in node.attributes for attr in interactive_attributes):
                return True

            # Check for interactive ARIA roles
            if 'role' in node.attributes:
                interactive_roles = {
                    'button',
                    'link',
                    'menuitem',
                    'option',
                    'radio',
                    'checkbox',
                    'tab',
                    'textbox',
                    'combobox',
                    'slider',
                    'spinbutton',
                    'search',
                    'searchbox',
                }
                if node.attributes['role'] in interactive_roles:
                    return True

        # Quaternary check: accessibility tree roles
        if node.ax_node and node.ax_node.role:
            interactive_ax_roles = {
                'button',
                'link',
                'menuitem',
                'option',
                'radio',
                'checkbox',
                'tab',
                'textbox',
                'combobox',
                'slider',
                'spinbutton',
                'listbox',
                'search',
                'searchbox',
            }
            if node.ax_node.role in interactive_ax_roles:
                return True

        # ICON AND SMALL ELEMENT CHECK: Elements that might be icons
        if (
            node.snapshot_node
            and node.snapshot_node.bounds
            and 10 <= node.snapshot_node.bounds.width <= 50  # Icon-sized elements
            and 10 <= node.snapshot_node.bounds.height <= 50
        ):
            # Check if this small element has interactive properties
            if node.attributes:
                # Small elements with these attributes are likely interactive icons
                icon_attributes = {'class', 'role', 'onclick', 'data-action', 'aria-label'}
                if any(attr in node.attributes for attr in icon_attributes):
                    return True

        # Final fallback: cursor style indicates interactivity (for cases Chrome missed)
        if node.snapshot_node and node.snapshot_node.cursor_style and node.snapshot_node.cursor_style == 'pointer':
            return True

        return False

