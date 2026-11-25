"""Stateless MCP tools that never start or import the browser stack."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .catalog import load_tool_definitions

if TYPE_CHECKING:
    from .schema import CompiledSchema

STATELESS_TOOL_NAMES = frozenset({"search_web", "scrape_web"})


@lru_cache(maxsize=2)
def _compiled_schema(name: str) -> CompiledSchema:
    """Compile the advertised schema once without importing jsonschema."""
    from .schema import compile_schema

    for tool in load_tool_definitions(modern=True):
        if tool["name"] == name:
            return compile_schema(tool["inputSchema"])
    raise ValueError(f"Unknown stateless tool: {name}")


def validate_stateless_arguments(name: str, arguments: dict[str, Any]) -> None:
    """Apply the advertised input schema before any network call."""
    from .schema import iter_errors

    error = next(iter_errors(arguments, _compiled_schema(name)), None)
    if error is None:
        return
    location = f" at {error.json_path}" if error.json_path and error.json_path != "$" else ""
    raise ValueError(f"Invalid arguments{location}: {error.message}")


def _json_error(error: str, message: str) -> tuple[str, bool]:
    return json.dumps({"error": error, "message": message}, ensure_ascii=False), True


async def _search_web(arguments: dict[str, Any]) -> tuple[str, bool]:
    from openbrowser.search import SearchError, asearch

    try:
        results = await asearch(
            arguments["query"],
            max_results=arguments.get("max_results", 5),
            backend=arguments.get("backend", "auto"),
            region=arguments.get("region", "us-en"),
            safe_search=arguments.get("safe_search", "moderate"),
            timelimit=arguments.get("timelimit"),
        )
    except SearchError as error:
        return _json_error("search_failed", str(error))
    except (TypeError, ValueError) as error:
        return _json_error("invalid_search_arguments", str(error))
    return json.dumps([result.as_dict() for result in results], ensure_ascii=False), False


async def _scrape_web(arguments: dict[str, Any]) -> tuple[str, bool]:
    from openbrowser.scrape.http import _truncate_scrape_result, scrape_http

    url = arguments["url"]
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("url must be an absolute http:// or https:// URL")
    try:
        result = await scrape_http(
            url,
            format=arguments.get("format", "markdown"),
            selector=arguments.get("selector"),
            extract_links=bool(arguments.get("extract_links", False)),
            timeout_ms=arguments.get("timeout_ms"),
            engine="http",
            only_main_content=bool(arguments.get("only_main_content", False)),
            include_tags=arguments.get("include_tags"),
            exclude_tags=arguments.get("exclude_tags"),
        )
    except Exception as error:
        return _json_error("http_fetch_failed", f"{type(error).__name__}: {error}")
    if result is None:
        return _json_error(
            "http_fetch_failed",
            "The URL did not return parseable HTML, text, or JSON. "
            "Use execute_code with a browser context for JavaScript pages.",
        )
    result["ok"] = True
    _truncate_scrape_result(result, arguments.get("max_chars"))
    return json.dumps(result, ensure_ascii=False), False


async def execute_stateless_tool(name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Run one no-browser MCP tool and return JSON text plus its error flag."""
    if name not in STATELESS_TOOL_NAMES:
        raise ValueError(f"Unknown stateless tool: {name}")
    validate_stateless_arguments(name, arguments)
    if name == "search_web":
        return await _search_web(arguments)
    return await _scrape_web(arguments)
