"""Shared markdown extraction utilities for browser content processing.

This module provides a unified interface for extracting clean markdown from
browser content, used by both the tools service and page actor. It converts
the enhanced DOM tree to HTML, then to markdown using markdownify.

The extraction pipeline:
1. Get enhanced DOM tree (from browser session or DOM service)
2. Serialize to HTML using HTMLSerializer
3. Convert to markdown using markdownify
4. Clean up whitespace, remove JSON blobs, filter artifacts

Functions:
    extract_clean_markdown: Main entry point for markdown extraction.

Requirements:
    markdownify: Install with `pip install markdownify`

Example:
    >>> content, stats = await extract_clean_markdown(browser_session)
    >>> print(content[:500])
    >>> print(f"Extracted {stats['final_filtered_chars']} chars")
"""

import re
from typing import TYPE_CHECKING, Any

from openbrowser.browser.dom.serializer.html_serializer import HTMLSerializer
from openbrowser.browser.dom.service import DomService

if TYPE_CHECKING:
    from openbrowser.browser.session import BrowserSession
    from openbrowser.browser.watchdogs.dom_watchdog import DOMWatchdog


async def extract_clean_markdown(
    browser_session: 'BrowserSession | None' = None,
    dom_service: DomService | None = None,
    target_id: str | None = None,
    extract_links: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Extract clean markdown from browser content using enhanced DOM tree.

    Unified function for markdown extraction supporting two paths:
    1. Browser session path (tools service): Uses browser_session directly
    2. DOM service path (page actor): Uses dom_service + target_id

    The extraction process:
    1. Get enhanced DOM tree from DOMWatchdog cache or build it
    2. Serialize to clean HTML using HTMLSerializer
    3. Convert HTML to markdown using markdownify
    4. Clean up: remove URL encoding, JSON blobs, excessive whitespace

    Args:
        browser_session: Browser session for extraction (tools service path).
            Mutually exclusive with dom_service/target_id.
        dom_service: DOM service instance (page actor path).
        target_id: Target ID for the page (required with dom_service).
        extract_links: Whether to preserve hyperlinks in markdown output.

    Returns:
        Tuple of (markdown_content, statistics_dict) where statistics include:
            - method: Extraction method used
            - original_html_chars: HTML length before conversion
            - initial_markdown_chars: Markdown length before filtering
            - filtered_chars_removed: Characters removed by filtering
            - final_filtered_chars: Final markdown length
            - url: Current page URL (if available)

    Raises:
        ValueError: If neither browser_session nor (dom_service + target_id)
            are provided, or if conflicting arguments are given.
        ImportError: If markdownify is not installed.
        NotImplementedError: If dom_service path is used (not yet implemented).

    Example:
        >>> content, stats = await extract_clean_markdown(browser_session)
        >>> print(f"Extracted {stats['final_filtered_chars']} chars from {stats['url']}")
    """
    # Validate input parameters
    if browser_session is not None:
        if dom_service is not None or target_id is not None:
            raise ValueError('Cannot specify both browser_session and dom_service/target_id')
        # Browser session path (tools service)
        enhanced_dom_tree = await _get_enhanced_dom_tree_from_browser_session(browser_session)
        current_url = await browser_session.get_current_page_url()
        method = 'enhanced_dom_tree'
    elif dom_service is not None and target_id is not None:
        # DOM service path (page actor) - not fully implemented yet
        # For now, raise error as this path requires additional implementation
        raise NotImplementedError('dom_service + target_id path not yet implemented')
        # current_url = None  # Not available via DOM service
        # method = 'dom_service'
    else:
        raise ValueError('Must provide either browser_session or both dom_service and target_id')

    # Use the HTML serializer with the enhanced DOM tree
    html_serializer = HTMLSerializer(extract_links=extract_links)
    page_html = html_serializer.serialize(enhanced_dom_tree)

    original_html_length = len(page_html)

    # Use markdownify for clean markdown conversion
    try:
        from markdownify import markdownify as md
    except ImportError:
        raise ImportError('markdownify is required for markdown extraction. Install with: pip install markdownify')

    content = md(
        page_html,
        heading_style='ATX',  # Use # style headings
        strip=['script', 'style'],  # Remove these tags
        bullets='-',  # Use - for unordered lists
        code_language='',  # Don't add language to code blocks
        escape_asterisks=False,  # Don't escape asterisks (cleaner output)
        escape_underscores=False,  # Don't escape underscores (cleaner output)
        escape_misc=False,  # Don't escape other characters (cleaner output)
        autolinks=False,  # Don't convert URLs to <> format
        default_title=False,  # Don't add default title attributes
        keep_inline_images_in=[],  # Don't keep inline images in any tags (we already filter base64 in HTML)
    )

    initial_markdown_length = len(content)

    # Minimal cleanup - markdownify already does most of the work
    content = re.sub(r'%[0-9A-Fa-f]{2}', '', content)  # Remove any remaining URL encoding

    # Apply light preprocessing to clean up excessive whitespace
    content, chars_filtered = _preprocess_markdown_content(content)

    final_filtered_length = len(content)

    # Content statistics
    stats = {
        'method': method,
        'original_html_chars': original_html_length,
        'initial_markdown_chars': initial_markdown_length,
        'filtered_chars_removed': chars_filtered,
        'final_filtered_chars': final_filtered_length,
    }

    # Add URL to stats if available
    if current_url:
        stats['url'] = current_url

    return content, stats


async def _get_enhanced_dom_tree_from_browser_session(browser_session: 'BrowserSession'):
    """Get enhanced DOM tree from browser session via DOMWatchdog.

    Attempts to use cached enhanced DOM tree from DOMWatchdog for
    performance. Falls back to building the tree via CDP if not cached.

    Args:
        browser_session: Active browser session with agent focus.

    Returns:
        EnhancedDOMTreeNode representing the document root.

    Raises:
        RuntimeError: If browser session not started (no agent_focus).

    Note:
        The result is cached in DOMWatchdog for subsequent calls.
    """
    # Get the enhanced DOM tree from DOMWatchdog
    # This captures the current state of the page including dynamic content, shadow roots, etc.
    dom_watchdog: 'DOMWatchdog | None' = getattr(browser_session, '_dom_watchdog', None)
    
    # Use cached enhanced DOM tree if available
    if dom_watchdog and hasattr(dom_watchdog, 'enhanced_dom_tree') and dom_watchdog.enhanced_dom_tree is not None:
        return dom_watchdog.enhanced_dom_tree

    # Build the enhanced DOM tree if not cached
    if not browser_session.agent_focus:
        raise RuntimeError('Browser session not started')
    
    cdp_session = browser_session.agent_focus
    
    # Get document using CDP
    await cdp_session.cdp_client.send.DOM.enable(session_id=cdp_session.session_id)
    doc_result = await cdp_session.cdp_client.send.DOM.getDocument(
        params={'depth': -1, 'pierce': True},
        session_id=cdp_session.session_id
    )
    
    # Build enhanced DOM tree from root node
    root_node = doc_result['root']
    enhanced_dom_tree = DomService._build_enhanced_node(
        root_node,
        target_id=browser_session.current_target_id,
        session_id=cdp_session.session_id
    )
    
    # Cache it in DOMWatchdog if available
    if dom_watchdog and hasattr(dom_watchdog, 'enhanced_dom_tree'):
        dom_watchdog.enhanced_dom_tree = enhanced_dom_tree
    
    return enhanced_dom_tree


def _preprocess_markdown_content(content: str, max_newlines: int = 3) -> tuple[str, int]:
    """Light preprocessing of markdown output with JSON blob removal.

    Cleans up markdown output by:
    1. Removing JSON blobs (common in SPAs like LinkedIn, Facebook)
    2. Compressing excessive newlines (4+ becomes max_newlines)
    3. Filtering lines that are too short or look like JSON artifacts

    Args:
        content: Raw markdown content from markdownify.
        max_newlines: Maximum consecutive newlines to preserve (default: 3).

    Returns:
        Tuple of (filtered_content, chars_removed).

    Note:
        JSON detection targets objects >100 chars to avoid removing
        small legitimate inline JSON snippets.
    """
    original_length = len(content)

    # Remove JSON blobs (common in SPAs like LinkedIn, Facebook, etc.)
    # These are often embedded as `{"key":"value",...}` and can be massive
    # Match JSON objects/arrays that are at least 100 chars long
    # This catches SPA state/config data without removing small inline JSON
    content = re.sub(r'`\{["\w].*?\}`', '', content, flags=re.DOTALL)  # Remove JSON in code blocks
    content = re.sub(r'\{"\$type":[^}]{100,}\}', '', content)  # Remove JSON with $type fields (common pattern)
    content = re.sub(r'\{"[^"]{5,}":\{[^}]{100,}\}', '', content)  # Remove nested JSON objects

    # Compress consecutive newlines (4+ newlines become max_newlines)
    content = re.sub(r'\n{4,}', '\n' * max_newlines, content)

    # Remove lines that are only whitespace or very short (likely artifacts)
    lines = content.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep lines with substantial content
        if len(stripped) > 2:
            # Skip lines that look like JSON (start with { or [ and are very long)
            if (stripped.startswith('{') or stripped.startswith('[')) and len(stripped) > 100:
                continue
            filtered_lines.append(line)

    content = '\n'.join(filtered_lines)
    content = content.strip()

    chars_filtered = original_length - len(content)
    return content, chars_filtered

