"""Tests for the fast Lexbor HTML extraction path."""

from openbrowser.dom.fast_html import to_links, to_markdown, to_nodes, to_text, to_title


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
