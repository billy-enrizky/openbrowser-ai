"""
E2E LLM Performance Benchmark: OpenBrowser vs Playwright vs Chrome DevTools MCP.

Runs identical browser tasks through Claude (via claude-agent-sdk) and measures
task success, tool call count, wall-clock time, and cost.
"""
import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _verify_fact_lookup(result: str) -> bool:
    """Output must contain 'Guido van Rossum' AND '1991'."""
    lower = result.lower()
    return "guido van rossum" in lower and "1991" in lower


def _verify_form_fill(result: str) -> bool:
    """Output must mention form submission or response data."""
    lower = result.lower()
    return any(kw in lower for kw in ["submitted", "response", "custname", "post"])


def _verify_multi_page_extract(result: str) -> bool:
    """Output must contain at least 3 distinct multi-word strings (story titles)."""
    lines = [line.strip() for line in result.split("\n") if len(line.split()) >= 3]
    return len(lines) >= 3


def _verify_search_navigate(result: str) -> bool:
    """Output must contain 'Mozilla'."""
    return "mozilla" in result.lower()


def _verify_deep_navigation(result: str) -> bool:
    """Output must contain a version number pattern (digits.digits.digits)."""
    return bool(re.search(r"\d+\.\d+\.\d+", result))


def _verify_content_analysis(result: str) -> bool:
    """Output must contain numeric counts for headings, links, and paragraphs."""
    lower = result.lower()
    has_numbers = bool(re.search(r"\d+", result))
    has_terms = sum(1 for term in ["heading", "link", "paragraph"] if term in lower)
    return has_numbers and has_terms >= 2


TASKS = [
    {
        "name": "fact_lookup",
        "prompt": (
            "Go to the Python Wikipedia page "
            "(https://en.wikipedia.org/wiki/Python_(programming_language)) "
            "and find who created Python and in what year."
        ),
        "verify": _verify_fact_lookup,
    },
    {
        "name": "form_fill",
        "prompt": (
            "Navigate to httpbin.org/forms/post, fill in Customer name: John Doe, "
            "choose Medium pizza size, select Mushroom topping, and submit the form."
        ),
        "verify": _verify_form_fill,
    },
    {
        "name": "multi_page_extract",
        "prompt": (
            "Go to news.ycombinator.com and extract the titles of the top 5 stories."
        ),
        "verify": _verify_multi_page_extract,
    },
    {
        "name": "search_navigate",
        "prompt": (
            "Go to en.wikipedia.org, search for 'Rust programming language', "
            "click the result, and tell me what company originally developed it."
        ),
        "verify": _verify_search_navigate,
    },
    {
        "name": "deep_navigation",
        "prompt": (
            "Go to github.com/anthropics/claude-code, "
            "find the latest release version number."
        ),
        "verify": _verify_deep_navigation,
    },
    {
        "name": "content_analysis",
        "prompt": (
            "Go to example.com and describe the page structure: "
            "how many headings, links, and paragraphs."
        ),
        "verify": _verify_content_analysis,
    },
]
