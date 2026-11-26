"""Tests for the fast Lexbor HTML extraction path."""

from types import SimpleNamespace

import pytest
from selectolax.lexbor import SelectolaxError

from openbrowser.dom.fast_html import (
    apply_content_filters,
    has_empty_application_root,
    parse_document,
    to_links,
    to_markdown,
    to_nodes,
    to_text,
    to_title,
)


def test_markdown_headings_paragraphs_and_inline():
    html = '<h1>Title</h1><p>Hello <a href="/x">link</a> and <strong>bold</strong>.</p>'
    assert to_markdown(html) == "# Title\n\nHello [link](/x) and **bold**."


def test_markdown_lists_are_numbered_and_bulleted():
    html = "<ul><li>A</li><li>B</li></ul><ol><li>One</li><li>Two</li></ol>"
    assert to_markdown(html) == "- A\n- B\n\n1. One\n2. Two"


def test_markdown_table_uses_a_pipe_table():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    assert to_markdown(html) == "| A | B |\n| --- | --- |\n| 1 | 2 |"


def test_markdown_strips_scripts_and_styles():
    html = "<html><head><style>.x{}</style></head><body><h1>H</h1><script>bad()</script><p>Good</p></body></html>"
    content = to_markdown(html)
    assert "# H" in content
    assert "Good" in content
    assert "bad()" not in content
    assert ".x{}" not in content


def test_markdown_selector_limits_output():
    html = '<div><h1>Outer</h1><section id="target"><h2>Inner</h2><p>Body</p></section></div>'
    assert to_markdown(html, "#target") == "## Inner\n\nBody"


def test_text_returns_visible_lines():
    html = "<html><body><h1>H</h1><p>Line one</p><p>Line two</p></body></html>"
    assert to_text(html) == "H\nLine one\nLine two"


def test_links_return_text_href_and_title():
    html = '<a href="/a" title="A title">A</a><a href="/b">B</a>'
    assert to_links(html) == [
        {"text": "A", "href": "/a", "title": "A title"},
        {"text": "B", "href": "/b", "title": None},
    ]


def test_nodes_return_text_html_and_attributes():
    html = '<ul><li data-id="a">A</li><li data-id="b">B</li></ul>'
    nodes = to_nodes(html, "li")
    assert [node["text"] for node in nodes] == ["A", "B"]
    assert [node["attributes"]["data-id"] for node in nodes] == ["a", "b"]
    assert nodes[0]["html"] == '<li data-id="a">A</li>'


def test_title_returns_document_title():
    assert to_title("<html><head><title>Bench Title</title></head></html>") == "Bench Title"
    assert to_title("<html><body>No title</body></html>") == ""


def test_markdown_keeps_code_and_pre_content():
    assert to_markdown("<p><code>x = 1</code></p>") == "`x = 1`"
    assert to_markdown("<pre><code>x = 1\ny = 2</code></pre>") == "```\nx = 1\ny = 2\n```"


def test_markdown_preserves_inline_spaces_but_drops_indentation():
    assert to_markdown("<div><span>Hello</span> <span>world</span></div>") == "Hello world"
    assert to_markdown("<div>\n  <span>Hello</span>\n  <span>world</span>\n</div>") == "Hello world"
    assert to_markdown("<div><p>First</p> <p>Second</p></div>") == "First\n\nSecond"


def test_markdown_escapes_pipes_inside_table_cells():
    html = "<table><tr><th>A|B</th><th>C</th></tr><tr><td>1</td><td>2</td></tr></table>"
    assert to_markdown(html) == "| A\\|B | C |\n| --- | --- |\n| 1 | 2 |"


def test_markdown_selector_renders_the_selected_node():
    assert to_markdown("<table><tr><td>A</td><td>B</td></tr></table>", "table") == "| A | B |\n| --- | --- |"
    assert to_markdown('<img src="x.png" alt="X">', "img") == "![X](x.png)"


def test_empty_application_root_detection():
    assert has_empty_application_root('<div id="root"></div><nav>Static navigation text that is long enough</nav>')
    assert has_empty_application_root('<div id="app"> \n </div>')
    assert not has_empty_application_root('<div id="root"><p>Mounted</p></div>')
    assert not has_empty_application_root("<main>No application shell</main>")


def test_markdown_preserves_inline_space_around_html_comments():
    html = "<div><span>A</span> <!--split--> <span>B</span></div>"
    assert to_markdown(html) == "A B"


def test_markdown_does_not_add_space_for_adjacent_nodes_around_html_comments():
    html = "<div><span>A</span><!--split--><span>B</span></div>"
    assert to_markdown(html) == "AB"


def test_markdown_collapses_inline_whitespace_split_by_a_comment():
    html = "<p>A <!--split--> B</p>"
    assert to_markdown(html) == "A B"


RAW_HTML = """<!DOCTYPE html>
<html>
<head><style>.page { color: red; }</style></head>
<body>
<nav>Navigation</nav>
<article id="target">
  <style>.article { color: blue; }</style>
  <script>window.inlineValue = 1;</script>
  <p>A &amp; B</p>
  <svg viewBox="0 0 1 1"><circle cx="0.5" cy="0.5" r="0.5"/></svg>
  <div class="ad">Advertisement</div>
</article>
<footer>Footer</footer>
</body>
</html>
"""


def html_document():
    return parse_document(RAW_HTML)


def test_html_unscoped_unfiltered_preserves_original_bytes():
    document = html_document()
    assert document.extract("html") == RAW_HTML
    assert document.extract("html", None) == RAW_HTML
    document.markdown()
    assert document.extract("html") == RAW_HTML


def test_html_selector_returns_raw_selected_node_with_inline_assets():
    document = html_document()
    selected = document.extract("html", "#target")
    assert selected.startswith('<article id="target">')
    assert selected.endswith("</article>")
    assert "<style>.article { color: blue; }</style>" in selected
    assert "<script>window.inlineValue = 1;</script>" in selected
    assert "<svg" in selected
    assert "Navigation" not in selected
    assert "Footer" not in selected
    assert selected != RAW_HTML
    assert document.extract("html") == RAW_HTML


def test_html_document_body_and_head_scopes():
    document = html_document()
    full = document.extract("html", "html")
    body = document.extract("html", "body")
    head = document.extract("html", "head")
    assert full.startswith("<html>")
    assert "Navigation" in full
    assert body.startswith("<body>")
    assert "Navigation" in body
    assert head.startswith("<head>")
    assert "<style>.page { color: red; }</style>" in head
    assert "Navigation" not in head


def test_html_missing_selector_returns_empty_html():
    document = html_document()
    assert document.extract("html", "article.missing") == ""
    assert document.extract("html", "footer.missing") == ""


def test_html_invalid_selector_raises_css_error():
    document = html_document()
    with pytest.raises(SelectolaxError):
        document.extract("html", "div[")


def test_html_only_main_and_exclude_filters_preserve_inline_assets():
    document = html_document()
    stats = apply_content_filters(document, only_main_content=True, exclude_tags=[".ad"])
    assert stats["removed"] >= 3
    selected = document.extract("html")
    assert "Navigation" not in selected
    assert "Footer" not in selected
    assert "Advertisement" not in selected
    assert "<style>.article { color: blue; }</style>" in selected
    assert "<script>window.inlineValue = 1;</script>" in selected
    assert "<svg" in selected
    assert selected != RAW_HTML


def test_html_include_tags_selects_the_include_root():
    document = html_document()
    stats = apply_content_filters(document, include_tags=["article"])
    assert stats["include_root"] == "article"
    selected = document.extract("html")
    assert selected.startswith('<article id="target">')
    assert "Footer" not in selected
    assert "<script>window.inlineValue = 1;</script>" in selected


def test_html_include_tags_without_match_returns_original_bytes():
    document = html_document()
    stats = apply_content_filters(document, include_tags=["article.missing"])
    assert stats["include_root"] is None
    assert document.extract("html") == RAW_HTML


def test_html_explicit_selector_wins_over_include_root():
    document = html_document()
    apply_content_filters(document, include_tags=["footer"])
    selected = document.extract("html", "article")
    assert selected.startswith('<article id="target">')
    assert "Footer" not in selected


def test_html_selector_removed_by_filter_returns_empty_html():
    document = html_document()
    apply_content_filters(document, only_main_content=True)
    assert document.extract("html", "footer") == ""


def test_html_exclude_inline_assets_only_when_explicitly_filtered():
    document = html_document()
    apply_content_filters(document, exclude_tags=["script"])
    selected = document.extract("html", "article")
    assert "<script" not in selected
    assert "<style>.article { color: blue; }</style>" in selected
    assert "<svg" in selected


def test_http_html_selector_and_filters_use_raw_selected_html():
    from openbrowser.scrape.http import _http_result_from_response

    response = SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=RAW_HTML,
        url="https://example.test/page",
    )

    selected = _http_result_from_response(response, format="html", selector="#target", engine="http")
    assert selected is not None
    assert selected["content"].startswith('<article id="target">')
    assert "<script>window.inlineValue = 1;</script>" in selected["content"]

    filtered = _http_result_from_response(
        response,
        format="html",
        selector=None,
        engine="http",
        only_main_content=True,
        exclude_tags=[".ad"],
    )
    assert filtered is not None
    assert "Navigation" not in filtered["content"]
    assert "Footer" not in filtered["content"]
    assert "Advertisement" not in filtered["content"]
    assert "<script>window.inlineValue = 1;</script>" in filtered["content"]
    assert "<svg" in filtered["content"]


def test_html_filters_do_not_change_markdown_text_or_links_paths():
    document = html_document()
    selected = document.extract("html", "article")
    assert "<script" in selected
    markdown = document.markdown()
    assert "A & B" in markdown
    assert "window.inlineValue" not in markdown
    assert ".article" not in markdown
    assert "A & B" in document.text()
    assert [link["href"] for link in document.links()] == []


def test_html_raw_parse_is_lazy_until_scoped_or_filtered():
    document = html_document()
    assert document._raw_tree is None
    document.markdown()
    assert document._raw_tree is None
    assert document.extract("html") == RAW_HTML
    assert document._raw_tree is None
    assert "<article" in document.extract("html", "article")
    assert document._raw_tree is not None
