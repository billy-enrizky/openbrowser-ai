"""Comprehensive tests for openbrowser.code_use.utils module.

Covers: truncate_message_content, detect_token_limit_issue,
extract_url_from_task, extract_code_blocks.
"""

import logging

import pytest

from openbrowser.code_use.utils import (
    detect_token_limit_issue,
    extract_code_blocks,
    extract_url_from_task,
    truncate_message_content,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# truncate_message_content
# ---------------------------------------------------------------------------


class TestTruncateMessageContent:
    def test_short_content_unchanged(self):
        result = truncate_message_content("hello", max_length=100)
        assert result == "hello"

    def test_exact_limit_unchanged(self):
        content = "x" * 100
        result = truncate_message_content(content, max_length=100)
        assert result == content

    def test_long_content_truncated(self):
        content = "x" * 200
        result = truncate_message_content(content, max_length=100)
        assert len(result) > 100
        assert "truncated" in result
        assert "100 characters" in result

    def test_default_max_length(self):
        content = "x" * 10001
        result = truncate_message_content(content)
        assert "truncated" in result

    def test_empty_string(self):
        result = truncate_message_content("", max_length=100)
        assert result == ""


# ---------------------------------------------------------------------------
# detect_token_limit_issue
# ---------------------------------------------------------------------------


class TestDetectTokenLimitIssue:
    def test_stop_reason_max_tokens(self):
        is_bad, msg = detect_token_limit_issue("output", 100, 200, "max_tokens")
        assert is_bad is True
        assert "max_tokens" in msg

    def test_high_usage_ratio(self):
        is_bad, msg = detect_token_limit_issue("output", 95, 100, "end_turn")
        assert is_bad is True
        assert "95.0%" in msg

    def test_exactly_90_percent(self):
        is_bad, msg = detect_token_limit_issue("output", 90, 100, "end_turn")
        assert is_bad is True

    def test_below_90_percent(self):
        is_bad, msg = detect_token_limit_issue("output", 89, 100, "end_turn")
        assert is_bad is False
        assert msg is None

    def test_repetitive_output(self):
        content = "abcdef" * 50  # 300 chars, last 6 = "abcdef" appears 50 times
        is_bad, msg = detect_token_limit_issue(content, 100, 200, "end_turn")
        assert is_bad is True
        assert "Repetitive" in msg

    def test_non_repetitive_output(self):
        content = "a" * 5 + "b"  # only 6 chars, last 6 not repeated enough
        is_bad, msg = detect_token_limit_issue(content, 10, 200, "end_turn")
        assert is_bad is False

    def test_short_content_no_repetition_check(self):
        content = "short"
        is_bad, msg = detect_token_limit_issue(content, 10, 200, "end_turn")
        assert is_bad is False

    def test_no_max_tokens(self):
        is_bad, msg = detect_token_limit_issue("output", 100, None, "end_turn")
        assert is_bad is False

    def test_max_tokens_zero(self):
        is_bad, msg = detect_token_limit_issue("output", 100, 0, "end_turn")
        assert is_bad is False

    def test_completion_tokens_none(self):
        is_bad, msg = detect_token_limit_issue("output", None, 100, "end_turn")
        assert is_bad is False

    def test_all_none(self):
        is_bad, msg = detect_token_limit_issue("output", None, None, None)
        assert is_bad is False

    def test_stop_reason_none(self):
        is_bad, msg = detect_token_limit_issue("output", 10, 100, None)
        assert is_bad is False


# ---------------------------------------------------------------------------
# extract_url_from_task
# ---------------------------------------------------------------------------


class TestExtractUrlFromTask:
    def test_extract_simple_url(self):
        result = extract_url_from_task("Go to https://google.com")
        assert result == "https://google.com"

    def test_extract_url_with_path(self):
        result = extract_url_from_task("Navigate to https://example.com/page?q=1")
        assert result == "https://example.com/page?q=1"

    def test_extract_domain_without_protocol(self):
        result = extract_url_from_task("Open google.com")
        assert result == "https://google.com"

    def test_extract_domain_with_www(self):
        result = extract_url_from_task("Go to www.google.com")
        assert result == "https://www.google.com"

    def test_multiple_urls_returns_none(self):
        result = extract_url_from_task("Compare https://google.com and https://bing.com")
        assert result is None

    def test_no_url_returns_none(self):
        result = extract_url_from_task("Search for the best restaurants")
        assert result is None

    def test_email_excluded(self):
        result = extract_url_from_task("Contact user@example.com about the project")
        assert result is None

    def test_url_with_trailing_punctuation(self):
        result = extract_url_from_task("Visit https://example.com.")
        assert result == "https://example.com"

    def test_url_with_trailing_comma(self):
        result = extract_url_from_task("Go to https://example.com, then do X")
        assert result == "https://example.com"

    def test_http_url(self):
        # http:// URL also matches the domain pattern which becomes https://
        # So two unique URLs => returns None (ambiguous)
        result = extract_url_from_task("Open http://example.com")
        assert result is None

    def test_subdomain_url(self):
        result = extract_url_from_task("Check api.example.com/status")
        assert result == "https://api.example.com/status"

    def test_url_with_parentheses_removed(self):
        result = extract_url_from_task("See (https://example.com)")
        assert result is not None
        assert "example.com" in result

    def test_empty_task(self):
        result = extract_url_from_task("")
        assert result is None


# ---------------------------------------------------------------------------
# extract_code_blocks
# ---------------------------------------------------------------------------


class TestExtractCodeBlocks:
    def test_single_python_block(self):
        text = "```python\nprint('hello')\n```"
        blocks = extract_code_blocks(text)
        assert "python" in blocks
        assert "print('hello')" in blocks["python"]

    def test_single_js_block(self):
        text = "```javascript\nconsole.log('hi')\n```"
        blocks = extract_code_blocks(text)
        assert "js" in blocks

    def test_js_shorthand(self):
        text = "```js\nconsole.log('hi')\n```"
        blocks = extract_code_blocks(text)
        assert "js" in blocks

    def test_bash_block_not_matched(self):
        """The literal 'bash' language tag is not explicitly handled -- only 'sh' and 'shell' are."""
        text = "```bash\necho hello\n```"
        blocks = extract_code_blocks(text)
        # 'bash' falls through to the else/skip branch in the normalizer
        assert "bash" not in blocks

    def test_sh_block(self):
        text = "```sh\necho hello\n```"
        blocks = extract_code_blocks(text)
        assert "bash" in blocks

    def test_shell_block(self):
        text = "```shell\necho hello\n```"
        blocks = extract_code_blocks(text)
        assert "bash" in blocks

    def test_markdown_block(self):
        text = "```markdown\n# Header\n```"
        blocks = extract_code_blocks(text)
        assert "markdown" in blocks

    def test_md_block(self):
        text = "```md\n# Header\n```"
        blocks = extract_code_blocks(text)
        assert "markdown" in blocks

    def test_unknown_language_skipped(self):
        text = "```ruby\nputs 'hi'\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0

    def test_named_block(self):
        text = "```js my_variable\nvar x = 1;\n```"
        blocks = extract_code_blocks(text)
        assert "my_variable" in blocks
        assert "var x = 1;" in blocks["my_variable"]

    def test_multiple_python_blocks(self):
        text = "```python\nx = 1\ny = 2\n```\nsome text\n```python\na = 3\nb = 4\n```"
        blocks = extract_code_blocks(text)
        assert "python_0" in blocks
        assert "python_1" in blocks
        assert "python" in blocks  # backward compat points to python_0
        assert blocks["python"] == blocks["python_0"]

    def test_empty_code_block_skipped(self):
        text = "```python\n\n```"
        blocks = extract_code_blocks(text)
        assert "python" not in blocks

    def test_mixed_languages(self):
        text = "```python\nx = 1\n```\n```js\nvar y = 2\n```\n```sh\necho z\n```"
        blocks = extract_code_blocks(text)
        assert "python" in blocks
        assert "js" in blocks
        assert "bash" in blocks

    def test_generic_code_block_fallback(self):
        """Generic ``` blocks (no language) should be treated as python."""
        text = "```\nfallback code\n```"
        blocks = extract_code_blocks(text)
        assert "python" in blocks
        assert "fallback code" in blocks["python"]

    def test_nested_backticks(self):
        text = "````python\ncode with ``` inside\n````"
        blocks = extract_code_blocks(text)
        assert "python" in blocks
        assert "code with ``` inside" in blocks["python"]

    def test_no_code_blocks(self):
        text = "Just regular text without any code blocks."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0

    def test_multiple_js_blocks_last_wins(self):
        """Multiple unnamed js blocks should keep the last one."""
        text = "```js\nvar x = 'first';\nvar y = 1;\n```\n```js\nvar a = 'second';\nvar b = 2;\n```"
        blocks = extract_code_blocks(text)
        assert "js" in blocks
        assert "second" in blocks["js"]

    def test_trailing_whitespace_stripped(self):
        text = "```python\ncode = 1\nresult = 2  \n  \n```"
        blocks = extract_code_blocks(text)
        assert "python" in blocks
        assert not blocks["python"].endswith("  \n  ")

    def test_generic_block_combined(self):
        text = "```\npart1 = 1\n```\n```\npart2 = 2\n```"
        blocks = extract_code_blocks(text)
        assert "python" in blocks
        assert "part1" in blocks["python"]
        assert "part2" in blocks["python"]
