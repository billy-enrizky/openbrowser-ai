"""Explicit tab identity and state for a browser session."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


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

	def touch(self, *, now: float | None = None) -> None:
		self.updated_at = time.time() if now is None else now


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
		if record.state is TabState.CLOSED and state is not TabState.CLOSED:
			record.created_at = now
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
	) -> TabRecord | None:
		record = self._tabs.get(target_id)
		if record is None:
			return None
		if url is not None:
			record.url = url
		if title is not None:
			record.title = title
		if state is not None:
			record.state = state
		record.touch()
		return record

	def mark_closing(self, target_id: str) -> None:
		self.update(target_id, state=TabState.CLOSING)

	def mark_closed(self, target_id: str) -> None:
		self.update(target_id, state=TabState.CLOSED)

	def mark_crashed(self, target_id: str) -> None:
		self.update(target_id, state=TabState.CRASHED)

	def sync_from_pages(self, pages: Iterable[dict[str, Any]]) -> None:
		"""Update the registry from an authoritative CDP page target list."""
		seen: set[str] = set()
		now = time.time()
		for page in pages:
			target_id = str(page.get('targetId') or page.get('target_id') or '')
			if not target_id:
				continue
			seen.add(target_id)
			self.register(
				target_id,
				url=str(page.get('url') or ''),
				title=str(page.get('title') or ''),
				state=TabState.OPEN,
			)
			record = self._tabs[target_id]
			record.touch(now=now)

		for target_id, record in self._tabs.items():
			if target_id not in seen and record.state is not TabState.CLOSED:
				record.state = TabState.CLOSED
				record.touch(now=now)

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
				'created_at': record.created_at,
				'updated_at': record.updated_at,
			}
			for record in records
		]

	def clear(self) -> None:
		self._tabs.clear()
