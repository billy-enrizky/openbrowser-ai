"""Bounded multi-request coordination for source-backed Jev search."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openbrowser.jev.semantic_search import (
	DEFAULT_RELEVANCE_THRESHOLD,
	MAX_PASSAGE_LENGTH,
	MAX_PASSAGES,
	MAX_QUERY_LENGTH,
	MAX_TOTAL_TEXT_LENGTH,
	SemanticSearchError,
)
from openbrowser.jev.views import (
	SearchPassage,
	SearchSentence,
	SemanticMatch,
	SemanticSearchResult,
)


MAX_BATCH_PASSAGES = 12
MAX_BATCH_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class SearchBatch:
	"""A deterministic slice of the caller's source passages."""

	index: int
	passages: tuple[SearchPassage, ...]


def plan_search_batches(
	*,
	query: str,
	passages: tuple[SearchPassage, ...] | Sequence[SearchPassage],
) -> tuple[SearchBatch, ...]:
	"""Validate a source and split it into provider-safe windows."""
	clean_query = _validate_query(query)
	del clean_query
	try:
		items = tuple(passages)
	except TypeError as exc:
		raise SemanticSearchError("passages must be an iterable of SearchPassage values") from exc
	if len(items) > MAX_PASSAGES:
		raise SemanticSearchError(f"passages must contain at most {MAX_PASSAGES} items")

	_validate_passages(items)
	return tuple(
		SearchBatch(index=start // MAX_BATCH_PASSAGES, passages=items[start : start + MAX_BATCH_PASSAGES])
		for start in range(0, len(items), MAX_BATCH_PASSAGES)
	)


class BoundedSemanticSearch:
	"""Run one-batch Jev searches concurrently and merge source-backed scores."""

	def __init__(
		self,
		*,
		searcher: object,
		max_concurrency: int = MAX_BATCH_CONCURRENCY,
	) -> None:
		if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
			raise ValueError("max_concurrency must be a positive integer")
		self._searcher = searcher
		self._max_concurrency = max_concurrency
		self.threshold = float(getattr(searcher, "threshold", DEFAULT_RELEVANCE_THRESHOLD))

	async def search(
		self,
		*,
		query: str,
		passages: tuple[SearchPassage, ...] | Sequence[SearchPassage],
	) -> SemanticSearchResult:
		"""Search every bounded window and return one stable source-backed result."""
		started = time.perf_counter()
		batches = plan_search_batches(query=query, passages=passages)
		if not batches:
			return SemanticSearchResult(
				scores=(),
				matches=(),
				threshold=self.threshold,
				elapsed_ms=_elapsed_ms(started),
			)

		semaphore = asyncio.Semaphore(self._max_concurrency)

		async def run_batch(batch: SearchBatch) -> SemanticSearchResult:
			async with semaphore:
				try:
					search = getattr(self._searcher, "search")  # noqa: B009
					return await search(query=query.strip(), passages=batch.passages)
				except SemanticSearchError as exc:
					raise _normalize_provider_error(exc) from exc
				except Exception as exc:
					raise _normalize_provider_error(exc) from exc

		tasks = [asyncio.create_task(run_batch(batch)) for batch in batches]
		try:
			results = await asyncio.gather(*tasks)
		except BaseException:
			for task in tasks:
				if not task.done():
					task.cancel()
			await asyncio.gather(*tasks, return_exceptions=True)
			raise
		return _merge_results(
			results=results,
			batches=batches,
			threshold=self.threshold,
			elapsed_ms=_elapsed_ms(started),
		)


def _validate_query(query: str) -> str:
	if not isinstance(query, str):
		raise SemanticSearchError("query must be a string")
	clean_query = query.strip()
	if not clean_query:
		raise SemanticSearchError("query must not be empty")
	if len(clean_query) > MAX_QUERY_LENGTH:
		raise SemanticSearchError(f"query must be at most {MAX_QUERY_LENGTH} characters")
	return clean_query


def _validate_passages(passages: tuple[SearchPassage, ...]) -> None:
	seen_ids: set[str] = set()
	total_text_length = 0
	for passage in passages:
		if not isinstance(passage, SearchPassage):
			raise SemanticSearchError("passages must contain SearchPassage values")
		if not isinstance(passage.id, str) or not passage.id.strip():
			raise SemanticSearchError("passage IDs must not be empty")
		if passage.id in seen_ids:
			raise SemanticSearchError("passage IDs must be unique")
		seen_ids.add(passage.id)
		if not isinstance(passage.text, str) or not passage.text.strip():
			raise SemanticSearchError("passage text must not be empty")
		if len(passage.text) > MAX_PASSAGE_LENGTH:
			raise SemanticSearchError(
				f"passage {passage.id!r} must be at most {MAX_PASSAGE_LENGTH:,} characters"
			)
		total_text_length += len(passage.text)
		if total_text_length > MAX_TOTAL_TEXT_LENGTH:
			raise SemanticSearchError(
				f"passage text must total at most {MAX_TOTAL_TEXT_LENGTH:,} characters"
			)
		_validate_sentences(passage)


def _validate_sentences(passage: SearchPassage) -> None:
	if not passage.sentences:
		return
	for sentence in passage.sentences:
		if not isinstance(sentence, SearchSentence):
			raise SemanticSearchError(
				f"passage {passage.id!r} sentences must contain SearchSentence values"
			)
		if not isinstance(sentence.index, int) or isinstance(sentence.index, bool):
			raise SemanticSearchError(
				f"passage {passage.id!r} sentence indexes must be integers"
			)
		if not isinstance(sentence.text, str) or not sentence.text.strip():
			raise SemanticSearchError(
				f"passage {passage.id!r} sentence text must not be empty"
			)
	if tuple(sentence.index for sentence in passage.sentences) != tuple(range(len(passage.sentences))):
		raise SemanticSearchError(
			f"passage {passage.id!r} sentence indexes must be contiguous from zero"
		)


def _normalize_provider_error(error: Exception) -> SemanticSearchError:
	message = str(error).lower()
	if any(marker in message for marker in ("max_tokens", "max tokens", "context length", "input too large", "413")):
		return SemanticSearchError(
			"Jev provider input limit reached; retry with a shorter source"
		)
	if "rate" in message or "429" in message:
		return SemanticSearchError("Jev rate limit reached; retry in a moment")
	if isinstance(error, SemanticSearchError):
		return error
	return SemanticSearchError(f"Jev semantic search failed: {type(error).__name__}")


def _merge_results(
	*,
	results: Sequence[SemanticSearchResult],
	batches: Sequence[SearchBatch],
	threshold: float,
	elapsed_ms: int,
) -> SemanticSearchResult:
	passage_order = {
		passage.id: index
		for index, passage in enumerate(passage for batch in batches for passage in batch.passages)
	}
	all_scores: list[tuple[int, SemanticMatch]] = []
	usage_values: list[object] = []
	for result in results:
		if result.usage is not None:
			usage_values.append(result.usage)
		for match in result.scores:
			_validate_match(match, passage_order, batches)
			all_scores.append((passage_order[match.passage_id], match))
	ranked_scores = tuple(match for _, match in sorted(all_scores, key=lambda item: (-item[1].probability, item[0])))
	matches = tuple(match for match in ranked_scores if match.probability >= threshold)
	return SemanticSearchResult(
		scores=ranked_scores,
		matches=matches,
		threshold=threshold,
		elapsed_ms=elapsed_ms,
		usage=_aggregate_usage(usage_values),
	)


def _validate_match(
	match: SemanticMatch,
	passage_order: Mapping[str, int],
	batches: Sequence[SearchBatch],
) -> None:
	if match.passage_id not in passage_order:
		raise SemanticSearchError("Jev returned a match for an unknown source passage")
	passage = next(
		passage
		for batch in batches
		for passage in batch.passages
		if passage.id == match.passage_id
	)
	if not 0.0 <= float(match.probability) <= 1.0:
		raise SemanticSearchError("Jev returned an invalid match probability")
	if not any(
		sentence.index == match.sentence_index and sentence.text == match.sentence_text
		for sentence in passage.sentences or (SearchSentence(index=0, text=passage.text),)
	):
		raise SemanticSearchError("Jev returned a sentence that is not present in the source")


def _aggregate_usage(values: Sequence[object]) -> object | None:
	if not values:
		return None
	if not all(isinstance(value, Mapping) for value in values):
		return None
	mapping_values = tuple(cast(Mapping[Any, Any], value) for value in values)
	keys = {key for value in mapping_values for key in value}
	aggregated: dict[object, object] = {}
	for key in keys:
		raw_values = [value[key] for value in mapping_values if key in value]
		if raw_values and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw_values):
			aggregated[key] = sum(raw_values)
		elif raw_values and all(item == raw_values[0] for item in raw_values):
			aggregated[key] = raw_values[0]
	return aggregated or None


def _elapsed_ms(started: float) -> int:
	return int((time.perf_counter() - started) * 1_000)


__all__ = [
	"MAX_BATCH_CONCURRENCY",
	"MAX_BATCH_PASSAGES",
	"BoundedSemanticSearch",
	"SearchBatch",
	"plan_search_batches",
	]
