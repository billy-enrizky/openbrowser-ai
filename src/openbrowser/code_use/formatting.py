"""Browser state formatting helpers for code-use agent."""

import logging
from typing import Any

from src.openbrowser.browser.session import BrowserSession

logger = logging.getLogger(__name__)


async def format_browser_state_for_llm(
    url: str,
    title: str,
    dom_html: str,
    namespace: dict[str, Any],
    browser_session: BrowserSession,
    screenshot: str | None = None,
    page_info: dict | None = None,
) -> str:
    """
    Format browser state summary for LLM consumption in code-use mode.

    Args:
        url: Current page URL
        title: Current page title
        dom_html: DOM representation
        namespace: The code execution namespace (for showing available variables)
        browser_session: Browser session for additional checks
        screenshot: Optional screenshot base64 string
        page_info: Optional page scroll/size information

    Returns:
        Formatted browser state text for LLM
    """
    if dom_html == "":
        dom_html = "Empty DOM tree (you might have to wait for the page to load)"

    # Format with URL and title header
    lines = ["## Browser State"]
    lines.append(f"**URL:** {url}")
    lines.append(f"**Title:** {title}")
    lines.append("")

    # Add page scroll info if available
    if page_info:
        pixels_above = page_info.get("pixels_above", 0)
        pixels_below = page_info.get("pixels_below", 0)
        viewport_height = page_info.get("viewport_height", 1)
        page_height = page_info.get("page_height", 1)

        pages_above = pixels_above / viewport_height if viewport_height > 0 else 0
        pages_below = pixels_below / viewport_height if viewport_height > 0 else 0
        total_pages = page_height / viewport_height if viewport_height > 0 else 0

        scroll_info = f"**Page:** {pages_above:.1f} pages above, {pages_below:.1f} pages below"
        if total_pages > 1.2:  # Only mention total if significantly > 1 page
            scroll_info += f", {total_pages:.1f} total pages"
        lines.append(scroll_info)
        lines.append("")

    # Add available variables and functions BEFORE DOM structure
    skip_vars = {
        "browser",
        "file_system",
        "np",
        "pd",
        "plt",
        "numpy",
        "pandas",
        "matplotlib",
        "requests",
        "BeautifulSoup",
        "bs4",
        "pypdf",
        "PdfReader",
        "wait",
    }

    # Highlight code block variables separately from regular variables
    code_block_vars = []
    regular_vars = []
    tracked_code_blocks = namespace.get("_code_block_vars", set())
    for name in namespace.keys():
        # Skip private vars and system objects/actions
        if not name.startswith("_") and name not in skip_vars:
            if name in tracked_code_blocks:
                code_block_vars.append(name)
            else:
                regular_vars.append(name)

    # Sort for consistent display
    available_vars_sorted = sorted(regular_vars)
    code_block_vars_sorted = sorted(code_block_vars)

    # Build available line with code blocks and variables
    parts = []
    if code_block_vars_sorted:
        # Show detailed info for code block variables
        code_block_details = []
        for var_name in code_block_vars_sorted:
            value = namespace.get(var_name)
            if value is not None:
                type_name = type(value).__name__
                value_str = str(value) if not isinstance(value, str) else value

                # Check if it's a function
                is_function = value_str.strip().startswith("(function") or value_str.strip().startswith("(async function")

                if is_function:
                    detail = f"{var_name}({type_name})"
                else:
                    first_20 = value_str[:20].replace("\n", "\\n").replace("\t", "\\t")
                    last_20 = value_str[-20:].replace("\n", "\\n").replace("\t", "\\t") if len(value_str) > 20 else ""

                    if last_20 and first_20 != last_20:
                        detail = f'{var_name}({type_name}): "{first_20}...{last_20}"'
                    else:
                        detail = f'{var_name}({type_name}): "{first_20}"'
                code_block_details.append(detail)

        parts.append(f'**Code block variables:** {" | ".join(code_block_details)}')
    if available_vars_sorted:
        parts.append(f'**Variables:** {", ".join(available_vars_sorted)}')

    lines.append(f'**Available:** {" | ".join(parts)}')
    lines.append("")

    # Add DOM structure
    lines.append("**DOM Structure:**")

    # Add scroll position hints for DOM
    if page_info:
        pixels_above = page_info.get("pixels_above", 0)
        pixels_below = page_info.get("pixels_below", 0)
        viewport_height = page_info.get("viewport_height", 1)

        pages_above = pixels_above / viewport_height if viewport_height > 0 else 0
        pages_below = pixels_below / viewport_height if viewport_height > 0 else 0

        if pages_above > 0:
            dom_html = f"... {pages_above:.1f} pages above \n{dom_html}"
        else:
            dom_html = "[Start of page]\n" + dom_html

        if pages_below > 0:
            dom_html += f"\n... {pages_below:.1f} pages below "
        else:
            dom_html += "\n[End of page]"

    # Truncate DOM if too long and notify LLM
    max_dom_length = 60000
    if len(dom_html) > max_dom_length:
        lines.append(dom_html[:max_dom_length])
        lines.append(
            f"\n[DOM truncated after {max_dom_length} characters. Full page contains {len(dom_html)} characters total. Use evaluate to explore more.]"
        )
    else:
        lines.append(dom_html)

    browser_state_text = "\n".join(lines)
    return browser_state_text

