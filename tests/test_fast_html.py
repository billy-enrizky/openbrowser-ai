"""Tests for the fast Lexbor HTML extraction path."""

from openbrowser.dom.fast_html import (
    has_empty_application_root,
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
