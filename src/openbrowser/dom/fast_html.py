"""Fast HTML extraction for the HTTP scraping path.

The browser path keeps the enhanced DOM and markdownify pipeline. Static
scraping uses Lexbor through selectolax so a one-kilobyte page parses in
microseconds instead of milliseconds.
"""

from __future__ import annotations

import re
from typing import Any

from selectolax.lexbor import LexborHTMLParser

_HEADINGS = {f"h{i}" for i in range(1, 7)}
_BLOCK_CONTAINERS = {
    "div",
    "section",
    "article",
    "header",
    "footer",
    "main",
    "aside",
    "nav",
    "figure",
    "figcaption",
    "details",
    "summary",
    "form",
    "fieldset",
    "address",
    "dl",
    "dt",
    "dd",
}
_REMOVED = {"script", "style", "noscript", "template", "svg"}
_WHITESPACE = re.compile(r"\s+")
_EXCESS_NEWLINES = re.compile(r"\n{4,}")


def _children(node):
    child = node.first_child
    while child is not None:
        yield child
        child = child.next


def _inline(node) -> str:
    parts: list[str] = []
    for child in _children(node):
        tag = child.tag
        if tag == "-text":
            parts.append(_WHITESPACE.sub(" ", child.text_content or ""))
        elif tag == "a":
            parts.append(f"[{_inline(child).strip()}]({child.attrs.get('href', '')})")
        elif tag in ("strong", "b"):
            parts.append(f"**{_inline(child).strip()}**")
        elif tag in ("em", "i"):
            parts.append(f"*{_inline(child).strip()}*")
        elif tag == "code":
            parts.append(f"`{(child.text_content or '').strip()}`")
        elif tag == "br":
            parts.append("\n")
        elif tag == "img":
            parts.append(f"![{child.attrs.get('alt', '')}]({child.attrs.get('src', '')})")
        else:
            parts.append(_inline(child))
    return "".join(parts)


def _render_children(node) -> str:
    return "".join(_render(child) for child in _children(node))


def _render(node) -> str:
    tag = node.tag
    if tag == "-text":
        text = _WHITESPACE.sub(" ", node.text_content or "")
        return text if text.strip() else ""
    if tag in _REMOVED:
        return ""
    if tag in _HEADINGS:
        return "#" * int(tag[1]) + " " + _inline(node).strip() + "\n\n"
    if tag == "p":
        body = _inline(node).strip()
        return body + "\n\n" if body else ""
    if tag == "br":
        return "\n"
    if tag == "hr":
        return "---\n\n"
    if tag == "pre":
        return "```\n" + (node.text_content or "").strip("\n") + "\n```\n\n"
    if tag == "blockquote":
        body = _render_children(node).strip()
        return "\n".join("> " + line for line in body.splitlines()) + "\n\n"
    if tag in ("ul", "ol"):
        lines: list[str] = []
        items = [child for child in _children(node) if child.tag == "li"]
        for index, item in enumerate(items, 1):
            prefix = f"{index}. " if tag == "ol" else "- "
            body = _render_children(item).strip()
            body_lines = body.splitlines() or [""]
            lines.append(prefix + body_lines[0])
            lines.extend("  " + line for line in body_lines[1:])
        return "\n".join(lines) + "\n\n"
    if tag == "table":
        return _render_table(node)
    if tag == "a":
        return f"[{_inline(node).strip()}]({node.attrs.get('href', '')})"
    if tag in ("strong", "b"):
        return f"**{_inline(node).strip()}**"
    if tag in ("em", "i"):
        return f"*{_inline(node).strip()}*"
    if tag == "code":
        return f"`{(node.text_content or '').strip()}`"
    if tag == "img":
        return f"![{node.attrs.get('alt', '')}]({node.attrs.get('src', '')})"
    body = _render_children(node)
    if tag in _BLOCK_CONTAINERS:
        body = body.strip()
        return body + "\n\n" if body else ""
    return body


def _render_table(node) -> str:
    rows: list[list[str]] = []
    for row in node.css("tr"):
        cells = [_inline(cell).strip() for cell in row.css("th,td")]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines) + "\n\n"


def _parse(html: str) -> LexborHTMLParser:
    tree = LexborHTMLParser(html)
    for node in tree.css("script, style, noscript, template, svg"):
        node.decompose()
    return tree


def to_markdown(html: str, selector: str | None = None) -> str:
    """Convert HTML to compact markdown with the fast Lexbor tree."""
    tree = _parse(html)
    node = tree.css_first(selector) if selector else None
    if node is None:
        node = tree.body or tree.root
    text = _render_children(node)
    return _EXCESS_NEWLINES.sub("\n\n\n", text).strip()


def to_title(html: str) -> str:
    """Return the document title or an empty string."""
    tree = _parse(html)
    node = tree.css_first("title")
    return node.text(separator=" ", strip=True) if node is not None else ""


def to_text(html: str, selector: str | None = None) -> str:
    """Return visible text with one block per line."""
    tree = _parse(html)
    node = tree.css_first(selector) if selector else None
    if node is None:
        node = tree.body or tree.root
    assert node is not None
    return node.text(separator="\n", strip=True)


def to_links(html: str) -> list[dict[str, Any]]:
    """Return every link with text, href, and title."""
    tree = _parse(html)
    return [
        {
            "text": node.text(separator=" ", strip=True),
            "href": node.attrs.get("href"),
            "title": node.attrs.get("title"),
        }
        for node in tree.css("a[href]")
    ]


def to_nodes(html: str, selector: str) -> list[dict[str, Any]]:
    """Return matching nodes with text, inner HTML, and attributes."""
    tree = _parse(html)
    return [
        {
            "text": node.text(separator=" ", strip=True),
            "html": node.html or "",
            "attributes": dict(node.attrs),
        }
        for node in tree.css(selector)
    ]
