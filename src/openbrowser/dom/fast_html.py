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
_INLINE_TAGS = {
    "a",
    "abbr",
    "b",
    "br",
    "cite",
    "code",
    "del",
    "em",
    "i",
    "img",
    "ins",
    "mark",
    "q",
    "s",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "time",
    "u",
}
_WHITESPACE = re.compile(r"\s+")
_EXCESS_NEWLINES = re.compile(r"\n{4,}")


def _children(node):
    child = node.first_child
    while child is not None:
        yield child
        child = child.next


def _inline(node) -> str:
    parts: list[str] = []
    pending_space = False
    for child in _children(node):
        tag = child.tag
        if tag == "-comment":
            continue
        if tag == "-text":
            text = _WHITESPACE.sub(" ", child.text_content or "")
            if not text:
                continue
            if not text.strip():
                pending_space = True
                continue
            leading_space = text.startswith(" ")
            trailing_space = text.endswith(" ")
            if parts and (pending_space or leading_space) and not parts[-1].endswith(" "):
                parts.append(" ")
            parts.append(text.strip())
            pending_space = trailing_space
            continue
        if tag == "a":
            rendered = f"[{_inline(child).strip()}]({child.attrs.get('href', '')})"
        elif tag in ("strong", "b"):
            rendered = f"**{_inline(child).strip()}**"
        elif tag in ("em", "i"):
            rendered = f"*{_inline(child).strip()}*"
        elif tag == "code":
            rendered = f"`{(child.text() or '').strip()}`"
        elif tag == "br":
            rendered = "\n"
        elif tag == "img":
            rendered = f"![{child.attrs.get('alt', '')}]({child.attrs.get('src', '')})"
        else:
            rendered = _inline(child)
        if not rendered:
            continue
        if parts and pending_space and not parts[-1].endswith(" "):
            parts.append(" ")
        parts.append(rendered)
        pending_space = False
    return "".join(parts)


def _is_whitespace_text(node) -> bool:
    return node.tag == "-text" and not (node.text_content or "").strip()


def _is_inline_node(node) -> bool:
    return node.tag == "-text" or node.tag in _INLINE_TAGS


def _render_children(node) -> str:
    parts: list[str] = []
    previous = None
    pending_space = False
    for child in _children(node):
        if child.tag == "-comment":
            continue
        if _is_whitespace_text(child):
            pending_space = True
            continue
        if pending_space and previous is not None and _is_inline_node(previous) and _is_inline_node(child):
            parts.append(" ")
        parts.append(_render(child))
        previous = child
        pending_space = False
    return "".join(parts)


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
        return "```\n" + (node.text() or "").strip("\n") + "\n```\n\n"
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
        return f"`{(node.text() or '').strip()}`"
    if tag == "img":
        return f"![{node.attrs.get('alt', '')}]({node.attrs.get('src', '')})"
    body = _render_children(node)
    if tag in _BLOCK_CONTAINERS:
        body = body.strip()
        return body + "\n\n" if body else ""
    return body


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _render_table(node) -> str:
    rows: list[list[str]] = []
    for row in node.css("tr"):
        cells = [_escape_table_cell(_inline(cell).strip()) for cell in row.css("th,td")]
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


def _has_empty_application_root(tree: LexborHTMLParser) -> bool:
    for node in tree.css("#root, #app, #__next"):
        child = node.first_child
        empty = True
        while child is not None:
            if child.tag != "-text" or (child.text_content or "").strip():
                empty = False
                break
            child = child.next
        if empty:
            return True
    return False


class ParsedHTML:
    """Read several extraction results from one cleaned Lexbor tree."""

    __slots__ = ("_html", "_tree")

    def __init__(self, html: str) -> None:
        self._html = html
        self._tree = _parse(html)

    def probe_static(self) -> tuple[str, bool]:
        node = self._tree.body or self._tree.root
        text = node.text(separator="\n", strip=True) if node is not None else ""
        return text, _has_empty_application_root(self._tree)

    def has_empty_application_root(self) -> bool:
        return _has_empty_application_root(self._tree)

    def title(self) -> str:
        node = self._tree.css_first("title")
        return node.text(separator=" ", strip=True) if node is not None else ""

    def markdown(self, selector: str | None = None) -> str:
        selected = self._tree.css_first(selector) if selector else None
        if selected is not None:
            text = _render(selected)
        else:
            node = self._tree.body or self._tree.root
            text = _render_children(node)
        return _EXCESS_NEWLINES.sub("\n\n\n", text).strip()

    def text(self, selector: str | None = None) -> str:
        node = self._tree.css_first(selector) if selector else None
        if node is None:
            node = self._tree.body or self._tree.root
        assert node is not None
        return node.text(separator="\n", strip=True)

    def links(self) -> list[dict[str, Any]]:
        return [
            {
                "text": node.text(separator=" ", strip=True),
                "href": node.attrs.get("href"),
                "title": node.attrs.get("title"),
            }
            for node in self._tree.css("a[href]")
        ]

    def nodes(self, selector: str) -> list[dict[str, Any]]:
        return [
            {
                "text": node.text(separator=" ", strip=True),
                "html": node.html or "",
                "attributes": dict(node.attrs),
            }
            for node in self._tree.css(selector)
        ]

    def extract(self, format: str, selector: str | None = None) -> Any:
        if format == "markdown":
            return self.markdown(selector)
        if format == "text":
            return self.text(selector)
        if format == "html":
            return self._html
        if format == "links":
            return self.links()
        return self.nodes(selector) if selector else []


def parse_document(html: str) -> ParsedHTML:
    """Parse HTML once for several static extraction operations."""
    return ParsedHTML(html)


def to_markdown(html: str, selector: str | None = None) -> str:
    """Convert HTML to compact markdown with the fast Lexbor tree."""
    return parse_document(html).markdown(selector)


def has_empty_application_root(html: str) -> bool:
    """Return True when a common SPA mount point has no text or element children."""
    return parse_document(html).has_empty_application_root()


def probe_static_html(html: str) -> tuple[str, bool]:
    """Return visible text and whether a common SPA mount point is empty."""
    return parse_document(html).probe_static()


def to_title(html: str) -> str:
    """Return the document title or an empty string."""
    return parse_document(html).title()


def to_text(html: str, selector: str | None = None) -> str:
    """Return visible text with one block per line."""
    return parse_document(html).text(selector)


def to_links(html: str) -> list[dict[str, Any]]:
    """Return every link with text, href, and title."""
    return parse_document(html).links()


def to_nodes(html: str, selector: str) -> list[dict[str, Any]]:
    """Return matching nodes with text, inner HTML, and attributes."""
    return parse_document(html).nodes(selector)
