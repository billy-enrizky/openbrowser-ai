"""Explicit terminal lifecycle for browser downloads.

Every download path in the watchdog reports into one registry so a caller can
answer three questions from one record: did the download start, did it finish,
and if not, where did it fail.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlsplit, urlunsplit


class DownloadPhase(str, Enum):
	"""Phases of one download. `COMPLETED`, `FAILED`, and `CANCELLED` are terminal."""

	REQUESTED = 'requested'
	STARTED = 'started'
	COMPLETED = 'completed'
	FAILED = 'failed'
	CANCELLED = 'cancelled'


TERMINAL_PHASES = frozenset({DownloadPhase.COMPLETED, DownloadPhase.FAILED, DownloadPhase.CANCELLED})


@dataclass
class DownloadOutcome:
	"""One download attempt, with the last observed phase and terminal detail."""

	download_id: str
	phase: DownloadPhase
	url: str = ''
	path: str | None = None
	file_name: str | None = None
	file_size: int | None = None
	file_type: str | None = None
	mime_type: str | None = None
	source: str = 'unknown'
	stage: str | None = None
	error: str | None = None
	detail: str | None = None
	from_cache: bool = False
	auto_download: bool = False
	requested_at: float = field(default_factory=time.time)
	finished_at: float | None = None
	_requested_monotonic: float = field(default_factory=time.monotonic, repr=False, compare=False)
	_finished_monotonic: float | None = field(default=None, repr=False, compare=False)

	@property
	def terminal(self) -> bool:
		return self.phase in TERMINAL_PHASES

	@property
	def duration_ms(self) -> float | None:
		if self._finished_monotonic is None:
			return None
		return max(0.0, (self._finished_monotonic - self._requested_monotonic) * 1000.0)

	def to_dict(self) -> dict[str, object]:
		return {
			'download_id': self.download_id,
			'phase': self.phase.value,
			'terminal': self.terminal,
			'url': self.url,
			'path': self.path,
			'file_name': self.file_name,
			'file_size': self.file_size,
			'file_type': self.file_type,
			'mime_type': self.mime_type,
			'source': self.source,
			'stage': self.stage,
			'error': self.error,
			'detail': self.detail,
			'from_cache': self.from_cache,
			'auto_download': self.auto_download,
			'requested_at': self.requested_at,
			'finished_at': self.finished_at,
			'duration_ms': self.duration_ms,
		}


class DownloadTimeoutError(TimeoutError):
	"""No terminal download outcome appeared within the wait budget."""

	def __init__(self, message: str, outcome: DownloadOutcome | None = None):
		super().__init__(message)
		self.outcome = outcome.to_dict() if outcome is not None else None


class DownloadRegistry:
	"""Bounded record of download attempts and their terminal outcomes.

	The first terminal outcome wins. Watchdog paths duplicate events on
	purpose (progress, polling, and fetch can all report the same file), so a
	later duplicate must never flip a completed download to failed.
	"""

	def __init__(self, max_outcomes: int = 200, max_active: int = 100):
		self._max_outcomes = max(1, max_outcomes)
		self._max_active = max(1, max_active)
		self._outcomes: dict[str, DownloadOutcome] = {}
		# Terminal order is completion order, not insertion order, so trimming
		# never discards the outcome a waiter just observed.
		self._terminal_order: list[str] = []
		# Active waiters pin only the outcomes they could match.
		self._active_waiters: list[tuple[str | None, str | None, float]] = []

	@property
	def max_active(self) -> int:
		return self._max_active

	def active_count(self) -> int:
		return sum(1 for outcome in self._outcomes.values() if not outcome.terminal)

	def get(self, download_id: str) -> DownloadOutcome | None:
		return self._outcomes.get(download_id)

	def outcomes(self) -> list[DownloadOutcome]:
		"""Return retained outcomes in creation order."""
		return list(self._outcomes.values())

	def pending(self) -> list[DownloadOutcome]:
		return [outcome for outcome in self._outcomes.values() if not outcome.terminal]

	def latest(
		self,
		*,
		url: str | None = None,
		terminal: bool | None = None,
	) -> DownloadOutcome | None:
		for outcome in reversed(list(self._outcomes.values())):
			if url is not None and outcome.url != url:
				continue
			if terminal is not None and outcome.terminal is not terminal:
				continue
			return outcome
		return None

	def clear(self) -> None:
		self._outcomes.clear()
		self._terminal_order.clear()

	def request(
		self,
		download_id: str,
		*,
		url: str = '',
		file_name: str | None = None,
		source: str = 'unknown',
		detail: str | None = None,
	) -> DownloadOutcome:
		outcome = self._get_or_create(download_id, url=url, file_name=file_name, source=source)
		self._update_metadata(outcome, url=url, file_name=file_name, source=source, detail=detail)
		return outcome

	def start(
		self,
		download_id: str,
		*,
		url: str | None = None,
		file_name: str | None = None,
		source: str | None = None,
	) -> DownloadOutcome:
		outcome = self._get_or_create(
			download_id,
			url=url or '',
			file_name=file_name,
			source=source or 'unknown',
		)
		self._update_metadata(outcome, url=url, file_name=file_name, source=source)
		self._advance(outcome, DownloadPhase.STARTED)
		return outcome

	def complete(
		self,
		download_id: str,
		*,
		path: str,
		file_size: int | None = None,
		file_name: str | None = None,
		file_type: str | None = None,
		mime_type: str | None = None,
		from_cache: bool = False,
		auto_download: bool = False,
		url: str | None = None,
		source: str | None = None,
		detail: str | None = None,
	) -> DownloadOutcome:
		outcome = self._get_or_create(
			download_id,
			url=url or '',
			file_name=file_name,
			source=source or 'unknown',
		)
		if outcome.phase is DownloadPhase.COMPLETED:
			# A completion is final. A duplicate must not change its path.
			return outcome
		if outcome.phase is DownloadPhase.CANCELLED:
			# A cancelled attempt stays cancelled even if a late file appears.
			return outcome
		# A file that appears after a timeout is stronger evidence than the
		# earlier failure, so allow that one upgrade.
		self._update_metadata(outcome, url=url, file_name=file_name, source=source, detail=detail)
		outcome.path = path
		outcome.file_name = file_name or outcome.file_name
		outcome.file_size = file_size if file_size is not None else outcome.file_size
		outcome.file_type = file_type or outcome.file_type
		outcome.mime_type = mime_type or outcome.mime_type
		outcome.from_cache = from_cache
		outcome.auto_download = auto_download
		outcome.stage = None
		outcome.error = None
		self._advance(outcome, DownloadPhase.COMPLETED)
		return outcome

	def fail(
		self,
		download_id: str,
		*,
		stage: str,
		error: str,
		url: str | None = None,
		path: str | None = None,
		source: str | None = None,
		detail: str | None = None,
	) -> DownloadOutcome:
		outcome = self._get_or_create(download_id, url=url or '', source=source or 'unknown')
		if outcome.terminal:
			return outcome
		self._update_metadata(outcome, url=url, source=source, detail=detail)
		if path is not None:
			outcome.path = path
		outcome.stage = stage
		outcome.error = error
		self._advance(outcome, DownloadPhase.FAILED)
		return outcome

	def cancel(self, download_id: str, *, detail: str | None = None) -> DownloadOutcome:
		outcome = self._get_or_create(download_id, url='', source='unknown')
		if outcome.terminal:
			return outcome
		if detail is not None:
			outcome.detail = detail
		self._advance(outcome, DownloadPhase.CANCELLED)
		return outcome

	def cancel_pending(self, *, detail: str | None = None) -> list[DownloadOutcome]:
		return [self.cancel(outcome.download_id, detail=detail) for outcome in self.pending()]

	async def wait_for_terminal(
		self,
		*,
		timeout: float,
		download_id: str | None = None,
		url: str | None = None,
		poll_interval: float = 0.1,
		lookback: float = 0.0,
	) -> DownloadOutcome:
		"""Wait for a terminal outcome, optionally for one specific download.

		An attempt that is already pending when the wait starts still counts.
		Only outcomes that were already terminal at call time are excluded, so
		an old finished download cannot satisfy a new wait. An explicit
		`download_id` returns its current terminal state even if it predates
		the call. `lookback` additionally accepts a terminal outcome that
		finished within that many seconds before the call, so "click, then
		wait" still works when the download was very fast.
		"""
		excluded = (
			frozenset()
			if download_id is not None
			else frozenset(
				outcome.download_id for outcome in self._outcomes.values() if outcome.terminal
			)
		)
		lookback_start = time.monotonic() - max(0.0, lookback)
		# A pending attempt that this wait could match is stronger evidence than
		# recent history, so the lookback only applies when no such attempt exists.
		if self._pending_matches(download_id=download_id, url=url):
			lookback_start = None
		deadline = time.monotonic() + max(0.0, timeout)
		waiter = (download_id, url, time.monotonic())
		self._active_waiters.append(waiter)

		try:
			while True:
				match = self._match(
					download_id=download_id,
					url=url,
					excluded_terminal_ids=excluded,
					terminal=True,
					lookback_start=lookback_start,
				)
				if match is not None:
					return match
				remaining = deadline - time.monotonic()
				if remaining <= 0:
					if download_id is not None or url is not None:
						last = self._match(
							download_id=download_id,
							url=url,
							excluded_terminal_ids=None,
							terminal=None,
							lookback_start=None,
						)
					else:
						# Report the newest observed attempt even if it began before
						# this wait. It is the best diagnostic for the timeout.
						last = self.latest()
					raise DownloadTimeoutError(
						f'No terminal download outcome within {timeout:g}s'
						+ (f' (last phase: {last.phase.value})' if last is not None else ' (no download observed)'),
						outcome=last,
					)
				await asyncio.sleep(min(poll_interval, remaining))
		finally:
			self._active_waiters.remove(waiter)
			# Release anything pinned by this waiter and re-bound the history.
			self._trim()

	def _match(
		self,
		*,
		download_id: str | None,
		url: str | None,
		excluded_terminal_ids: frozenset[str] | None,
		terminal: bool | None,
		lookback_start: float | None,
	) -> DownloadOutcome | None:
		for outcome in reversed(list(self._outcomes.values())):
			if download_id is not None and outcome.download_id != download_id:
				continue
			if url is not None and outcome.url != url:
				continue
			if (
				excluded_terminal_ids
				and outcome.terminal
				and outcome.download_id in excluded_terminal_ids
			):
				recent_enough = (
					lookback_start is not None
					and outcome._finished_monotonic is not None
					and outcome._finished_monotonic >= lookback_start
				)
				if not recent_enough:
					continue
			if terminal is not None and outcome.terminal is not terminal:
				continue
			return outcome
		return None

	def _get_or_create(
		self,
		download_id: str,
		*,
		url: str,
		file_name: str | None = None,
		source: str = 'unknown',
	) -> DownloadOutcome:
		existing = self._outcomes.get(download_id)
		if existing is not None:
			return existing
		outcome = DownloadOutcome(
			download_id=download_id,
			phase=DownloadPhase.REQUESTED,
			url=url,
			file_name=file_name,
			source=source,
		)
		self._outcomes[download_id] = outcome
		self._trim()
		return outcome

	def _pending_matches(self, *, download_id: str | None, url: str | None) -> bool:
		for outcome in self._outcomes.values():
			if outcome.terminal:
				continue
			if download_id is not None and outcome.download_id != download_id:
				continue
			if url is not None and outcome.url != url:
				continue
			return True
		return False

	def _protected_by_waiter(self, outcome: DownloadOutcome) -> bool:
		"""Whether an active waiter could still match this terminal outcome."""
		for download_id, url, started in list(self._active_waiters):
			if download_id is not None:
				if outcome.download_id == download_id:
					return True
				continue
			if url is not None:
				if outcome.url == url:
					return True
				continue
			if outcome._finished_monotonic is not None and outcome._finished_monotonic >= started:
				return True
		return False

	def _record_terminal_order(self, download_id: str) -> None:
		"""Move an id to the end of the completion order."""
		if download_id in self._terminal_order:
			self._terminal_order.remove(download_id)
		self._terminal_order.append(download_id)

	def _trim(self) -> None:
		# Never evict an active attempt, and always keep the newest terminal
		# outcome so a waiter can still observe it. Only older history goes.
		pending = [
			download_id for download_id, outcome in self._outcomes.items() if not outcome.terminal
		]
		while len(pending) > self._max_active:
			# Explicit capacity instead of unbounded growth. The oldest active
			# attempt is failed with a capacity error the caller can inspect.
			download_id = pending.pop(0)
			outcome = self._outcomes[download_id]
			outcome.stage = 'capacity'
			outcome.error = f'Too many active downloads (limit {self._max_active})'
			outcome.phase = DownloadPhase.FAILED
			outcome.finished_at = time.time()
			outcome._finished_monotonic = time.monotonic()
			self._record_terminal_order(download_id)

		while len(self._outcomes) > self._max_outcomes:
			if len(self._terminal_order) <= 1:
				return
			evictable = None
			for download_id in self._terminal_order:
				outcome = self._outcomes.get(download_id)
				if outcome is None:
					continue
				if self._protected_by_waiter(outcome):
					continue
				evictable = download_id
				break
			if evictable is None:
				return
			self._terminal_order.remove(evictable)
			self._outcomes.pop(evictable, None)

	@staticmethod
	def _update_metadata(
		outcome: DownloadOutcome,
		*,
		url: str | None = None,
		file_name: str | None = None,
		source: str | None = None,
		detail: str | None = None,
	) -> None:
		if url:
			outcome.url = url
		if file_name:
			outcome.file_name = file_name
		if source:
			outcome.source = source
		if detail is not None:
			outcome.detail = detail

	def _advance(self, outcome: DownloadOutcome, phase: DownloadPhase) -> None:
		if outcome.phase is DownloadPhase.COMPLETED:
			return
		if phase is not DownloadPhase.COMPLETED and outcome.terminal:
			return
		outcome.phase = phase
		if phase in TERMINAL_PHASES:
			outcome.finished_at = time.time()
			outcome._finished_monotonic = time.monotonic()
			self._record_terminal_order(outcome.download_id)
			# Re-trim now that this record became eligible for eviction.
			self._trim()


def redacted_outcome(payload: dict[str, object], *, max_text: int = 300) -> dict[str, object]:
	"""Return a status-safe copy of an outcome.

	Query strings are stripped so signed tokens do not leak into status, and
	long free text is capped so one failure cannot enlarge the response.
	"""

	def _truncate(value: object, limit: int = max_text) -> object:
		if isinstance(value, str) and len(value) > limit:
			return value[:limit] + '...'
		return value

	def _redact_url(value: object) -> object:
		if not isinstance(value, str):
			return '<redacted>'
		if not value:
			return value
		try:
			parsed = urlsplit(value)
		except ValueError:
			return '<invalid-url>'
		# Drop userinfo (user:password@host) and the query for every URL form,
		# including file: and bare paths.
		host = parsed.netloc.rsplit('@', 1)[-1]
		query = 'redacted' if parsed.query else ''
		return _truncate(urlunsplit((parsed.scheme, host, parsed.path, query, '')))

	redacted = dict(payload)
	for key in ('file_name', 'path', 'error', 'detail', 'mime_type'):
		if key in redacted:
			redacted[key] = _truncate(redacted[key])
	# Never expose an internal key that may embed a URL or token. Keep a stable
	# opaque reference so a status reader can still correlate entries.
	internal_id = redacted.pop('download_id', None)
	if isinstance(internal_id, str):
		redacted['download_ref'] = hashlib.sha256(internal_id.encode('utf-8')).hexdigest()[:12]
	if 'url' in redacted:
		redacted['url'] = _redact_url(redacted['url'])
	return redacted
