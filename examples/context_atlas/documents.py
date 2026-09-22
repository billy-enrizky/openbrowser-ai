"""Prepare bounded, source-backed documents for Context Atlas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from openbrowser.jev.semantic_search import (
	MAX_PASSAGES,
	MAX_PASSAGE_LENGTH,
	MAX_QUERY_LENGTH,
	MAX_TOTAL_TEXT_LENGTH,
)
from openbrowser.jev.views import SearchPassage, SearchSentence


class DocumentInputError(ValueError):
	"""Raised when a document cannot be converted into bounded source passages."""


def segment_text(text: str) -> tuple[dict[str, object], ...]:
	"""Split text deterministically while preserving each source sentence."""
	if not isinstance(text, str):
		raise DocumentInputError("passage text must be a string")
	if not text.strip():
		return ()

	sentences: list[dict[str, object]] = []
	start = 0
	for position, character in enumerate(text):
		if character not in ".!?":
			continue
		if not _ends_sentence(text, position):
			continue
		candidate = text[start : position + 1].strip()
		if candidate:
			sentences.append({"index": len(sentences), "text": candidate})
		start = position + 1

	tail = text[start:].strip()
	if tail:
		sentences.append({"index": len(sentences), "text": tail})
	return tuple(sentences)


def prepare_passages(
	blocks: Iterable[Mapping[str, Any]],
	*,
	query: str | None = None,
) -> tuple[SearchPassage, ...]:
	"""Validate request blocks and convert them to immutable Jev contracts."""
	if query is not None:
		_validate_query(query)
	try:
		items = tuple(blocks)
	except TypeError as exc:
		raise DocumentInputError("blocks must be an iterable") from exc
	if len(items) > MAX_PASSAGES:
		raise DocumentInputError(f"blocks must contain at most {MAX_PASSAGES} items")

	seen_ids: set[str] = set()
	total_length = 0
	passages: list[SearchPassage] = []
	for block in items:
		if not isinstance(block, Mapping):
			raise DocumentInputError("each block must be an object")
		block_id = block.get("id")
		text = block.get("text")
		if not isinstance(block_id, str) or not block_id.strip():
			raise DocumentInputError("block IDs must not be empty")
		if block_id in seen_ids:
			raise DocumentInputError("block IDs must be unique")
		seen_ids.add(block_id)
		if not isinstance(text, str) or not text.strip():
			raise DocumentInputError("block text must not be empty")
		if len(text) > MAX_PASSAGE_LENGTH:
			raise DocumentInputError(
				f"block {block_id!r} must be at most {MAX_PASSAGE_LENGTH:,} characters"
			)
		total_length += len(text)
		if total_length > MAX_TOTAL_TEXT_LENGTH:
			raise DocumentInputError(
				f"block text must total at most {MAX_TOTAL_TEXT_LENGTH:,} characters"
			)

		sentences = _read_sentences(block, block_id, text)
		if not sentences:
			raise DocumentInputError(f"block {block_id!r} must contain source text")
		passages.append(SearchPassage(id=block_id, text=text, sentences=sentences))
	return tuple(passages)


def _validate_query(query: str) -> str:
	if not isinstance(query, str) or not query.strip():
		raise DocumentInputError("query must not be empty")
	clean_query = query.strip()
	if len(clean_query) > MAX_QUERY_LENGTH:
		raise DocumentInputError(f"query must be at most {MAX_QUERY_LENGTH} characters")
	return clean_query


def _read_sentences(
	block: Mapping[str, Any],
	block_id: str,
	text: str,
) -> tuple[SearchSentence, ...]:
	if "sentences" not in block or block["sentences"] is None:
		return tuple(SearchSentence(index=item["index"], text=item["text"]) for item in segment_text(text))
	raw_sentences = block["sentences"]
	if not isinstance(raw_sentences, (list, tuple)) or not raw_sentences:
		raise DocumentInputError(f"block {block_id!r} sentences must be a non-empty list")
	sentences: list[SearchSentence] = []
	for expected_index, raw_sentence in enumerate(raw_sentences):
		if not isinstance(raw_sentence, Mapping):
			raise DocumentInputError(f"block {block_id!r} sentences must be objects")
		index = raw_sentence.get("index")
		sentence_text = raw_sentence.get("text")
		if index != expected_index:
			raise DocumentInputError(
				f"block {block_id!r} sentence indexes must be contiguous from zero"
			)
		if not isinstance(sentence_text, str) or not sentence_text.strip():
			raise DocumentInputError(f"block {block_id!r} sentence text must not be empty")
		sentences.append(SearchSentence(index=index, text=sentence_text))
	return tuple(sentences)


def _ends_sentence(text: str, position: int) -> bool:
	closing = "\"'”’)]}"
	next_position = position + 1
	while next_position < len(text) and text[next_position] in closing:
		next_position += 1
	return next_position == len(text) or text[next_position].isspace()


__all__ = ["DocumentInputError", "prepare_passages", "segment_text"]
