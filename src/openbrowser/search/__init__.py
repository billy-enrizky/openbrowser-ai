"""Compatibility shim for the canonical :mod:`openbrowser.web_search` module.

Prefer ``openbrowser.web_search`` and the ``web_search`` and ``aweb_search``
names in new code. The old ``openbrowser.search`` imports remain available for
existing callers.
"""

from openbrowser.web_search import (
    MAX_QUERY_LENGTH,
    MAX_WEB_SEARCH_RESULTS,
    SafeSearch,
    WebSearchError,
    WebSearchResult,
    aweb_search,
    web_search,
)

MAX_SEARCH_RESULTS = MAX_WEB_SEARCH_RESULTS
SearchError = WebSearchError
SearchResult = WebSearchResult
asearch = aweb_search
search = web_search

__all__ = [
    "MAX_QUERY_LENGTH",
    "MAX_SEARCH_RESULTS",
    "SafeSearch",
    "SearchError",
    "SearchResult",
    "asearch",
    "search",
]
