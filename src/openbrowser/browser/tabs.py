"""Explicit tab identity and state for a browser session."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from openbrowser.browser.focus import FOCUS_TEXT_LIMIT, _bounded_text


class TabState(str, Enum):
	"""Lifecycle state for one browser tab or page target."""

	OPENING = 'opening'
	OPEN = 'open'
	CLOSING = 'closing'
	CLOSED = 'closed'
	CRASHED = 'crashed'


class TabError(ValueError):
	"""Raised when a tab ID cannot be resolved unambiguously."""

	def __init__(
		self,
		message: str,
		*,
		tab_id: str,
		candidates: Iterable[str] | None = None,
		hint: str | None = None,
	):
		self.tab_id = tab_id
		self.candidates = list(candidates or [])
		self.hint = hint
		super().__init__(message)


@dataclass
class TabRecord:
	"""One tab identity and its last known state."""

	target_id: str
	short_id: str
	url: str = ''
	title: str = ''
	state: TabState = TabState.OPEN
	created_at: float = 0.0
	updated_at: float = 0.0
	parent_target_id: str | None = None
	error: str | None = None
	close_attempt: int = 0

	def touch(self, *, now: float | None = None) -> None:
		self.updated_at = time.time() if now is None else now


_PROTECTED_STATES = frozenset({TabState.CLOSING, TabState.CLOSED, TabState.CRASHED})


def _can_transition(current: TabState, requested: TabState, *, allow_reopen: bool) -> bool:
	if current is requested:
		return True
	if current is TabState.OPENING:
		return requested in {TabState.OPEN, TabState.CLOSING, TabState.CLOSED, TabState.CRASHED}
	if current is TabState.OPEN:
		return requested is not TabState.OPENING
	if current is TabState.CLOSING:
		if requested in {TabState.CLOSED, TabState.CRASHED}:
			return True
		return allow_reopen and requested in {TabState.OPEN, TabState.OPENING}
	if current is TabState.CLOSED:
		return allow_reopen and requested in {TabState.OPEN, TabState.OPENING}
	if current is TabState.CRASHED:
		if requested is TabState.CLOSED:
			return True
		return allow_reopen and requested in {TabState.OPEN, TabState.OPENING}
	return False


class TabRegistry:
	"""Map full and short target IDs to one explicit tab state.

	The short ID is always the last four characters of the target ID. The
	registry never invents a different short ID. If two live targets share a
	short ID, resolution raises ``TabError`` and lists every full ID so the
	caller can retry with an unambiguous value.
	"""

	def __init__(self) -> None:
		self._tabs: dict[str, TabRecord] = {}

	def __len__(self) -> int:
		return len(self._tabs)

	@staticmethod
	def short_id_for(target_id: str) -> str:
		return target_id[-4:]

	def register(
		self,
		target_id: str,
		*,
		url: str | None = None,
		title: str | None = None,
		parent_target_id: str | None = None,
		state: TabState = TabState.OPENING,
	) -> TabRecord:
		now = time.time()
		record = self._tabs.get(target_id)
		if record is None:
			record = TabRecord(
				target_id=target_id,
				short_id=self.short_id_for(target_id),
				url=url or '',
				title=title or '',
				state=state,
				created_at=now,
				updated_at=now,
				parent_target_id=parent_target_id,
			)
			self._tabs[target_id] = record
			return record

		if url is not None:
			record.url = url
		if title is not None:
			record.title = title
		if parent_target_id is not None:
			record.parent_target_id = parent_target_id
		if _can_transition(record.state, state, allow_reopen=False):
			record.state = state
		record.touch(now=now)
		return record

	def get(self, target_id: str) -> TabRecord | None:
		"""Return one tab record by full target ID."""
		return self._tabs.get(target_id)

	def update(
		self,
		target_id: str,
		*,
		url: str | None = None,
		title: str | None = None,
		state: TabState | None = None,
		allow_reopen: bool = False,
	) -> TabRecord | None:
		record = self._tabs.get(target_id)
		if record is None:
			return None
		now = time.time()
		if url is not None:
			record.url = url
		if title is not None:
			record.title = title
		if state is not None and _can_transition(record.state, state, allow_reopen=allow_reopen):
			if record.state in _PROTECTED_STATES and state in {TabState.OPEN, TabState.OPENING}:
				record.created_at = now
				record.error = None
			record.state = state
		record.touch(now=now)
		return record

	def mark_closing(self, target_id: str) -> None:
		record = self.update(target_id, state=TabState.CLOSING)
		if record is not None:
			record.close_attempt += 1

	def mark_closed(self, target_id: str) -> None:
		self.update(target_id, state=TabState.CLOSED)

	def mark_crashed(self, target_id: str, error: str | None = None) -> None:
		record = self.update(target_id, state=TabState.CRASHED)
		if record is None:
			return
		record.error = _bounded_text(error, limit=FOCUS_TEXT_LIMIT)

	def sync_from_pages(
		self,
		pages: Iterable[dict[str, Any]],
		*,
		allow_reopen: bool = False,
		complete: bool = True,
	) -> None:
		"""Update the registry from a CDP page target list.

		Set ``complete=False`` for URL-filtered eligibility lists, which must not
		close registered tabs that are merely omitted from the filter.
		"""
		seen: set[str] = set()
		now = time.time()
		for page in pages:
			target_id = str(page.get('targetId') or page.get('target_id') or '')
			if not target_id:
				continue
			seen.add(target_id)
			url = str(page.get('url') or '')
			title = str(page.get('title') or '')
			record = self._tabs.get(target_id)
			created = record is None
			if record is None:
				self.register(target_id, url=url, title=title, state=TabState.OPEN)
			else:
				self.update(target_id, url=url, title=title, state=TabState.OPEN, allow_reopen=allow_reopen)
			record = self._tabs[target_id]
			if created:
				record.created_at = now
			record.touch(now=now)

		if complete:
			for target_id, record in self._tabs.items():
				if target_id not in seen and record.state is not TabState.CLOSED:
					record.state = TabState.CLOSED
					record.touch(now=now)

	def live_page_candidates(
		self,
		pages: Iterable[dict[str, Any]],
		*,
		exclude_target_id: str | None = None,
	) -> list[TabRecord]:
		"""Return live page records in deterministic recency order."""
		page_list = list(pages)
		self.sync_from_pages(page_list, complete=False)
		live_ids = {
			str(page.get('targetId') or page.get('target_id') or '')
			for page in page_list
			if str(page.get('type') or '') == 'page'
		}
		candidates = [
			record
			for target_id, record in self._tabs.items()
			if target_id in live_ids
			and target_id != exclude_target_id
			and record.state in {TabState.OPEN, TabState.OPENING}
		]
		return sorted(candidates, key=lambda record: (record.created_at, record.target_id))

	def known_closed(self, identifier: str) -> str | None:
		"""Return the full target ID of one uniquely known closed tab."""
		identifier = identifier.strip()
		if not identifier:
			return None
		if len(identifier) != 4:
			record = self._tabs.get(identifier)
			if record is not None and record.state is TabState.CLOSED:
				return identifier
			return None
		matches = [
			target_id
			for target_id, record in self._tabs.items()
			if record.state is TabState.CLOSED and record.short_id == identifier
		]
		return matches[0] if len(matches) == 1 else None

	def candidates(self, identifier: str) -> list[str]:
		identifier = identifier.strip()
		if not identifier:
			return []
		if len(identifier) != 4:
			record = self._tabs.get(identifier)
			if record is not None and record.state is not TabState.CLOSED:
				return [identifier]
			return []
		return [
			target_id
			for target_id, record in self._tabs.items()
			if record.state is not TabState.CLOSED and record.short_id == identifier
		]

	def resolve(self, identifier: str) -> str:
		identifier = identifier.strip()
		if not identifier:
			raise TabError(
				'Tab ID is empty.',
				tab_id=identifier,
				hint='Read the tabs list from get_browser_state_summary() and pass a tab_id.',
			)

		exact = self._tabs.get(identifier)
		if exact is not None:
			if exact.state is TabState.CLOSED:
				raise TabError(
					f'Tab {identifier} is already closed.',
					tab_id=identifier,
					hint='Read the current tabs list and choose a live tab.',
				)
			return identifier

		candidates = self.candidates(identifier)
		if len(candidates) == 1:
			return candidates[0]
		if len(candidates) > 1:
			raise TabError(
				f'Tab ID {identifier} is ambiguous; it matches {len(candidates)} live tabs.',
				tab_id=identifier,
				candidates=candidates,
				hint='Pass the full CDP target_id, not the four-character suffix.',
			)
		raise TabError(
			f'No live tab matches tab_id {identifier}.',
			tab_id=identifier,
			hint='Read the current tabs list. The tab may have closed during the action.',
		)

	def snapshot(self, *, active_target_id: str | None = None) -> list[dict[str, Any]]:
		records = sorted(self._tabs.values(), key=lambda record: (record.created_at, record.target_id))
		return [
			{
				'target_id': record.target_id,
				'short_id': record.short_id,
				'url': record.url,
				'title': record.title,
				'state': record.state.value,
				'active': active_target_id == record.target_id,
				'parent_target_id': record.parent_target_id,
				'error': record.error,
				'close_attempt': record.close_attempt,
				'created_at': record.created_at,
				'updated_at': record.updated_at,
			}
			for record in records
		]

	def clear(self) -> None:
		self._tabs.clear()
