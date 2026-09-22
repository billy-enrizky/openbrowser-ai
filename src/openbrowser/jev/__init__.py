"""Source-backed Jev search support used by Context Atlas."""

from openbrowser.jev.coordinator import BoundedSemanticSearch
from openbrowser.jev.semantic_search import (
    DEFAULT_RELEVANCE_THRESHOLD,
    MAX_PASSAGES,
    MAX_PASSAGE_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_TOTAL_TEXT_LENGTH,
    SemanticSearch,
    SemanticSearchError,
)
from openbrowser.jev.views import (
    SearchPassage,
    SearchSentence,
    SemanticMatch,
    SemanticSearchResult,
)

__all__ = [
    "BoundedSemanticSearch",
    "DEFAULT_RELEVANCE_THRESHOLD",
    "MAX_PASSAGES",
    "MAX_PASSAGE_LENGTH",
    "MAX_QUERY_LENGTH",
    "MAX_TOTAL_TEXT_LENGTH",
    "SearchPassage",
    "SearchSentence",
    "SemanticMatch",
    "SemanticSearch",
    "SemanticSearchError",
    "SemanticSearchResult",
]
