from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

WaitUntil = Literal['load', 'domcontentloaded', 'networkidle', 'commit']

logger = logging.getLogger(__name__)


class NavigationError(RuntimeError):
	"""A navigation failed without requiring generic focus repair."""


class NavigationBusyError(NavigationError):
	"""Another navigation still owns this session."""


class NavigationBlockedError(NavigationError, ValueError):
	"""A URL was refused by the configured policy."""


class NavigationTimeoutError(NavigationError, TimeoutError):
	"""The request's single deadline expired."""


class NavigationPhase(str, Enum):
	PREPARING = 'preparing'
	NAVIGATING = 'navigating'
	WAITING = 'waiting'
	NOTIFYING = 'notifying'
	SUCCEEDED = 'succeeded'
	DOWNLOAD = 'download'
	FAILED = 'failed'
	TIMED_OUT = 'timed_out'
	CANCELLED = 'cancelled'


TERMINAL_PHASES = frozenset({NavigationPhase.SUCCEEDED, NavigationPhase.DOWNLOAD, NavigationPhase.FAILED,
	NavigationPhase.TIMED_OUT, NavigationPhase.CANCELLED})
_NEXT = {
	NavigationPhase.PREPARING: {NavigationPhase.NAVIGATING},
	NavigationPhase.NAVIGATING: {NavigationPhase.WAITING, NavigationPhase.NOTIFYING},
	NavigationPhase.WAITING: {NavigationPhase.NOTIFYING},
	NavigationPhase.NOTIFYING: {NavigationPhase.SUCCEEDED, NavigationPhase.DOWNLOAD},
}


@dataclass(frozen=True)
class NavigationCommit:
	frame_id: str
	loader_id: str
	url: str


class ContainmentPhase(str, Enum):
	PREPARING = 'preparing'
	ACTION_IN_FLIGHT = 'action_in_flight'
	VERIFYING = 'verifying'
	CONTAINED = 'contained'
	SKIPPED = 'skipped'
	FAILED = 'failed'
	UNCONFIRMED = 'unconfirmed'
	REVOKED = 'revoked'


@dataclass
class ContainmentAttempt:
	commit: NavigationCommit
	phase: ContainmentPhase = ContainmentPhase.PREPARING
	error: str | None = None
	world_name: str = ''
	context: tuple[int, str] | None = None
	object_id: str | None = None
	deadline: float | None = None
	owned_browser: tuple[int, float, str, str] | None = None

	@property
	def requires_retirement(self) -> bool:
		return self.phase in {ContainmentPhase.ACTION_IN_FLIGHT, ContainmentPhase.VERIFYING, ContainmentPhase.UNCONFIRMED}

	def snapshot(self) -> dict[str, Any]:
		return {'phase': self.phase.value, 'url': self.commit.url, 'frame_id': self.commit.frame_id,
			'loader_id': self.commit.loader_id, 'error': self.error}


def _same_history_url(url: Any, expected: str | None) -> bool:
	"""Compare a within-document event URL with the selected history entry.

	URLs cannot distinguish two entries with identical URLs and different
	history state. A racing same-URL event is an accepted limit.
	"""
	if expected is None:
		return True
	if not isinstance(url, str):
		return False
	left, right = urlsplit(url), urlsplit(expected)
	return (left.scheme, left.netloc, left.path, left.query, left.fragment) == (
		right.scheme, right.netloc, right.path, right.query, right.fragment
	)


@dataclass
class NavigationOperation:
	"""One event-owned request and the evidence that can complete it."""

	navigation_id: str
	generation: int
	requested_url: str
	wait_until: WaitUntil
	target_id: str | None = None
	session_id: str | None = None
	frame_id: str | None = None
	loader_id: str | None = None
	observed_url: str | None = None
	phase: NavigationPhase = NavigationPhase.PREPARING
	last_error: str | None = None
	deadline: float | None = None
	publication_attempted: bool = False
	child_tasks: set[asyncio.Task[Any]] = field(default_factory=set, repr=False)
	_changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
	_evidence: list[tuple[str, dict[str, Any]]] = field(default_factory=list, repr=False)
	_current_frame: str | None = None
	_current_loader: str | None = None
	_response_kind: Literal['document', 'same_document', 'download', 'history'] | None = None
	policy_commit: NavigationCommit | None = None
	containment: ContainmentAttempt | None = None
	policy_task: asyncio.Task[None] | None = field(default=None, repr=False)
	policy_changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
	_policy_open: bool = False
	_trigger_issued: bool = False
	_within_committed: bool = False
	_expected_within_url: str | None = None
	EVIDENCE_LIMIT: ClassVar[int] = 256

	@property
	def terminal(self) -> bool:
		return self.phase in TERMINAL_PHASES

	@property
	def is_download(self) -> bool:
		return self._response_kind == 'download'

	def transition(self, phase: NavigationPhase) -> None:
		if self.terminal:
			if phase == self.phase:
				return
			raise NavigationError(f'Navigation is already {self.phase.value}')
		if phase not in _NEXT[self.phase]:
			raise NavigationError(f'Illegal navigation transition {self.phase.value} -> {phase.value}')
		self.phase = phase

	def finish(self, phase: NavigationPhase, error: BaseException | str | None = None) -> None:
		if self.terminal:
			return
		if phase not in TERMINAL_PHASES:
			raise NavigationError(f'Not a terminal navigation phase: {phase.value}')
		if phase in {NavigationPhase.SUCCEEDED, NavigationPhase.DOWNLOAD}:
			self.transition(phase)
		else:
			self.phase = phase
		self.last_error = str(error) if error is not None else None
		self._evidence.clear()
		self._changed.set()

	def select_target(self, target_id: str, session_id: str) -> None:
		if self.target_id is not None:
			raise NavigationError('Navigation target binding is immutable')
		self.target_id, self.session_id = target_id, session_id

	def capture_document(self, frame: Mapping[str, Any]) -> None:
		self._current_frame = frame['id']
		self._current_loader = frame.get('loaderId')
		self.observed_url = frame.get('url', '') + frame.get('urlFragment', '')

	def arm_policy(self) -> None:
		self._policy_open = True

	def close_policy_capture(self) -> None:
		self._policy_open = False
		self.policy_changed.set()

	@property
	def policy_open(self) -> bool:
		return self._policy_open

	@property
	def policy_pending(self) -> bool:
		return (self.policy_task is not None and not self.policy_task.done()
			and (self.containment is None or self.containment.phase is not ContainmentPhase.REVOKED)) or (
			self.containment is not None and self.containment.requires_retirement)

	def record_commit(self, frame: Mapping[str, Any]) -> None:
		if not self._policy_open or frame.get('parentId') or frame.get('id') != self._current_frame:
			return
		loader, url = frame.get('loaderId'), frame.get('url')
		if not isinstance(loader, str) or not loader or not isinstance(url, str):
			return
		self.policy_commit = NavigationCommit(frame['id'], loader, url + frame.get('urlFragment', ''))
		if not self.terminal and self.containment is None:
			self.observed_url = self.policy_commit.url
		self.policy_changed.set()

	def record_policy_context(self, event: dict[str, Any]) -> None:
		attempt = self.containment
		context = event.get('context', {})
		if (attempt is not None and context.get('name') == attempt.world_name
			and context.get('auxData', {}).get('frameId') == attempt.commit.frame_id
			and isinstance(context.get('id'), int) and isinstance(context.get('uniqueId'), str)):
			attempt.context = (context['id'], context['uniqueId'])
			self.policy_changed.set()

	def observe(self, kind: str, event: dict[str, Any]) -> None:
		if kind == 'frame':
			self.record_commit(event.get('frame', {}))
		elif kind == 'within' and event.get('frameId') == self._current_frame:
			loader = self.policy_commit.loader_id if self.policy_commit else self._current_loader
			self.record_commit({'id': event['frameId'], 'loaderId': loader, 'url': event.get('url')})
		if self.terminal:
			return
		if kind == 'frame':
			frame = event.get('frame', {})
			if frame.get('parentId'):
				return
			frame_id = frame.get('id')
		else:
			frame_id = event.get('frameId')
		if self._current_frame is not None and frame_id != self._current_frame:
			return
		payload = event.get('frame', {}) if kind == 'frame' else event
		if kind == 'within' and self._response_kind == 'history':
			if self._trigger_issued and self.loader_id is None:
				if not _same_history_url(payload.get('url'), self._expected_within_url):
					return
				self.loader_id = self._current_loader
				self.observed_url = payload.get('url', self.observed_url)
				self._within_committed = True
			elif self.loader_id is not None:
				# A pushState after a document commit updates the URL without
				# replacing the loader this operation is waiting for.
				self.observed_url = payload.get('url', self.observed_url)
			else:
				return
			if len(self._evidence) >= self.EVIDENCE_LIMIT:
				self.finish(NavigationPhase.FAILED, 'Navigation evidence limit exceeded')
				return
			self._evidence.append(
				('within', {key: payload[key] for key in ('frameId', 'loaderId', 'name', 'url') if key in payload})
			)
			self._changed.set()
			return
		if (
			kind == 'frame'
			and self._response_kind in ('document', 'history')
			and self.loader_id is None
			and self._trigger_issued
		):
			loader = payload.get('loaderId')
			if isinstance(loader, str) and loader and loader != self._current_loader:
				self.loader_id = loader
		if (
			self._response_kind is not None
			and kind != 'within'
			and payload.get('loaderId') != self.loader_id
		):
			return
		if kind == 'frame' and self.loader_id is not None and payload.get('loaderId') == self.loader_id:
			self.observed_url = payload.get('url', '') + payload.get('urlFragment', '')
		if len(self._evidence) >= self.EVIDENCE_LIMIT:
			self.finish(NavigationPhase.FAILED, 'Navigation evidence limit exceeded')
			return
		keys = ('id', 'loaderId', 'url', 'urlFragment') if kind == 'frame' else ('frameId', 'loaderId', 'name', 'url')
		self._evidence.append((kind, {key: payload[key] for key in keys if key in payload}))
		self._changed.set()

	def accept_response(self, response: dict[str, Any]) -> None:
		if self.terminal:
			raise NavigationError(self.last_error or f'Navigation is {self.phase.value}')
		if not isinstance(response, dict):
			raise NavigationError('Page.navigate returned a malformed response')
		for name in ('frameId', 'loaderId', 'errorText'):
			if name in response and (not isinstance(response[name], str) or (name != 'errorText' and not response[name])):
				raise NavigationError(f'Page.navigate returned invalid {name}')
		if 'isDownload' in response and not isinstance(response['isDownload'], bool):
			raise NavigationError('Page.navigate returned invalid isDownload')
		if response.get('errorText') and response.get('isDownload') is not True:
			raise NavigationError(response['errorText'])
		self.frame_id = response.get('frameId')
		if not self.frame_id:
			raise NavigationError('Page.navigate returned no frameId')
		self._response_kind = 'download' if response.get('isDownload') is True else 'document' if response.get('loaderId') else 'same_document'
		if self.is_download:
			self.transition(NavigationPhase.NOTIFYING)
			return
		self.loader_id = response.get('loaderId') or self._current_loader
		if not response.get('loaderId') and self.frame_id != self._current_frame:
			raise NavigationError('Same-document response changed main frame')
		self.transition(NavigationPhase.WAITING)
		self._changed.set()

	def begin_trigger(self) -> None:
		"""Mark that the history or reload command is about to be issued.

		Evidence received before this point cannot be the commit for this
		request, so it is discarded.
		"""
		self._trigger_issued = True
		self._evidence.clear()

	def expect_history_commit(self, expected_url: str | None = None) -> None:
		"""Accept either a new document or a within-document commit.

		History traversal can produce either shape for the same entry,
		including SPA entries created with ``history.pushState`` that change
		the path or query without a new document. A within event only counts
		when its URL matches the selected history entry, so a page timer that
		changes the URL during the request cannot satisfy it.
		"""
		if self._response_kind is not None:
			raise NavigationError('Navigation already has a commit expectation')
		self._response_kind = 'history'
		self._expected_within_url = expected_url
		self.frame_id = self._current_frame
		self.loader_id = None
		self.transition(NavigationPhase.WAITING)
		self._changed.set()

	def expect_reload_commit(self) -> None:
		"""Expect a new document loader from a reload command."""
		if self._response_kind is not None:
			raise NavigationError('Navigation already has a commit expectation')
		self._response_kind = 'document'
		self.frame_id = self._current_frame
		self.loader_id = None
		self.transition(NavigationPhase.WAITING)
		self._changed.set()

	@property
	def ready(self) -> bool:
		if self.terminal or self._response_kind is None:
			return False
		if self.is_download:
			return True
		# A loader-less response acknowledges same-document navigation.
		if self._response_kind == 'history' and self._within_committed:
			return True
		if self._response_kind == 'history' and self.loader_id is None:
			return False
		same_document = self._response_kind == 'same_document'
		milestone = {'domcontentloaded': 'DOMContentLoaded', 'load': 'load', 'networkidle': 'networkIdle'}.get(self.wait_until)
		ready = same_document and self.wait_until == 'commit'
		for kind, data in self._evidence:
			frame_id = data.get('id') if kind == 'frame' else data.get('frameId')
			if frame_id != self.frame_id:
				continue
			if kind == 'within' and self._response_kind == 'history':
				self.observed_url = data.get('url', self.observed_url)
			elif kind == 'within' and same_document:
				self.observed_url = data.get('url', self.observed_url)
			elif data.get('loaderId') == self.loader_id:
				if kind == 'frame':
					self.observed_url = data.get('url', '') + data.get('urlFragment', '')
				if self.wait_until == 'commit' or (kind == 'lifecycle' and data.get('name') == milestone):
					ready = True
		return ready

	@property
	def committed(self) -> bool:
		"""True once the selected target has produced a document or within commit."""
		if self._response_kind == 'history' and self._within_committed:
			return True
		if self.loader_id is None:
			return False
		return any(
			kind == 'frame' and data.get('loaderId') == self.loader_id for kind, data in self._evidence
		)

	async def wait_commit(self) -> None:
		"""Wait for the commit only, before the document load milestone."""
		while not self.committed:
			if self.terminal:
				raise NavigationError(self.last_error or f'Navigation is {self.phase.value}')
			self._changed.clear()
			await self._changed.wait()

	async def wait_ready(self) -> None:
		while not self.ready:
			if self.terminal:
				raise NavigationError(self.last_error or f'Navigation is {self.phase.value}')
			self._changed.clear()
			await self._changed.wait()

	def snapshot(self) -> dict[str, Any]:
		return {name: getattr(self, name) for name in (
			'navigation_id', 'generation', 'requested_url', 'wait_until', 'target_id', 'session_id',
			'frame_id', 'loader_id', 'observed_url', 'last_error',
		)} | {'phase': self.phase.value, 'containment': self.containment.snapshot() if self.containment else None}


class NavigationMonitor:
	"""Own one request and one synchronous callback set per root CDP binding."""

	def __init__(self) -> None:
		self.active: NavigationOperation | None = None
		self._last: dict[str, Any] | None = None
		self._root: Any = None
		self._observer: Callable[[str, dict[str, Any], str | None], None] | None = None

	def reserve(self, operation: NavigationOperation) -> None:
		if self.active is not None and self.active is not operation:
			raise NavigationBusyError('Another navigation is in progress')
		self.active = operation

	def complete(self, operation: NavigationOperation) -> None:
		if self.active is operation:
			self._last = operation.snapshot()
			self.active = None

	def snapshot(self) -> dict[str, Any] | None:
		return self.active.snapshot() if self.active else dict(self._last) if self._last else None

	def invalidate(self) -> None:
		self._root = None

	def bind(
		self,
		root: Any,
		observer: Callable[[str, dict[str, Any], str | None], None] | None = None,
	) -> None:
		"""Bind one callback set per root and keep one document observer.

		The observer runs for every event, including when no request is active,
		so page-initiated document changes can invalidate caches. It is a
		best-effort early signal; read-time revalidation is the fence.
		"""
		if observer is not None:
			self._observer = observer
		if root is self._root:
			return
		self._root = root
		for method, kind in (('frameNavigated', 'frame'), ('lifecycleEvent', 'lifecycle'), ('navigatedWithinDocument', 'within')):
			def callback(event: dict[str, Any], session_id: str | None = None, *, _kind: str = kind) -> None:
				if self._root is not root:
					return
				observer = self._observer
				if observer is not None:
					try:
						observer(_kind, event, session_id)
					except Exception:
						logger.exception('Document observer failed for %s', _kind)
				operation = self.active
				if operation is not None and operation.session_id == session_id:
					operation.observe(_kind, event)
			getattr(root.register.Page, method)(callback)
		def context_callback(event: dict[str, Any], session_id: str | None = None) -> None:
			operation = self.active
			if self._root is root and operation is not None and operation.session_id == session_id:
				operation.record_policy_context(event)
		root.register.Runtime.executionContextCreated(context_callback)


class DocumentReadinessPhase(StrEnum):
	"""Progress and terminal phase of one actor navigation barrier.

	A success returns a ready snapshot. A CDP or navigation failure raises
	NavigationError. The enum records the phases that can appear in a timeout.
	"""

	REQUESTED = 'requested'
	COMMITTED = 'committed'
	TIMED_OUT = 'timed_out'


@dataclass(frozen=True)
class DocumentSnapshot:
	"""The main document a target currently shows."""

	frame_id: str | None
	loader_id: str | None
	url: str
	ready_state: str | None


async def read_document_snapshot(client: Any, session_id: str) -> DocumentSnapshot:
	"""Read the main frame and its ready state without registering events."""
	tree = await client.send.Page.getFrameTree(session_id=session_id)
	frame = ((tree or {}).get('frameTree') or {}).get('frame')
	if not isinstance(frame, dict):
		raise NavigationError('Page.getFrameTree returned no main frame')
	frame_id = frame.get('id') if isinstance(frame.get('id'), str) else None
	raw_loader = frame.get('loaderId')
	loader_id = raw_loader if isinstance(raw_loader, str) and raw_loader else None
	url = str(frame.get('url') or '') + str(frame.get('urlFragment') or '')
	ready_state = None
	try:
		result = await client.send.Runtime.evaluate(
			params={'expression': 'document.readyState', 'returnByValue': True},
			session_id=session_id,
		)
	except Exception:
		# A navigating document destroys its execution context. Retry.
		ready_state = None
	else:
		value = ((result or {}).get('result') or {}).get('value')
		if isinstance(value, str):
			ready_state = value
	return DocumentSnapshot(frame_id=frame_id, loader_id=loader_id, url=url, ready_state=ready_state)


def _document_committed(
	snapshot: DocumentSnapshot,
	baseline: DocumentSnapshot | None,
	requested_url: str | None,
	expected_loader: str | None,
	require_document_change: bool = False,
) -> bool:
	if expected_loader and snapshot.loader_id == expected_loader:
		return True
	if baseline is not None:
		if baseline.loader_id is not None and snapshot.loader_id != baseline.loader_id:
			return True
		if baseline.loader_id is None and snapshot.loader_id is not None:
			return True
		if (
			expected_loader is None
			and requested_url
			and snapshot.url
			and (not require_document_change or snapshot.url != baseline.url)
			and _same_history_url(snapshot.url, requested_url)
		):
			return snapshot.ready_state is not None
		return bool(requested_url and snapshot.url and snapshot.url != baseline.url)
	if expected_loader or require_document_change:
		return False
	return bool(requested_url) and snapshot.ready_state is not None and _same_history_url(
		snapshot.url, requested_url or ''
	)


def _milestone_met(snapshot: DocumentSnapshot, wait_until: WaitUntil) -> bool:
	if wait_until == 'commit':
		return True
	if wait_until == 'domcontentloaded':
		return snapshot.ready_state in ('interactive', 'complete')
	return snapshot.ready_state == 'complete'


ACTOR_WAIT_UNTIL = frozenset({'commit', 'domcontentloaded', 'load'})
_FATAL_SESSION_MARKERS = ('-32001', 'Session with given id not found')


def _fatal_cdp_error(error: BaseException) -> bool:
	text = str(error)
	return any(marker in text for marker in _FATAL_SESSION_MARKERS)


def validate_actor_wait_until(wait_until: WaitUntil) -> None:
	"""Reject an actor milestone the barrier cannot prove."""
	if wait_until not in ACTOR_WAIT_UNTIL:
		raise NavigationError(
			f'Actor navigation does not support wait_until={wait_until!r}; '
			'use commit, domcontentloaded, or load, or use session.navigate_to for networkidle'
		)


async def await_document_ready(
	client: Any,
	session_id: str,
	*,
	target_id: str,
	baseline: DocumentSnapshot | None,
	requested_url: str | None,
	expected_loader: str | None = None,
	wait_until: WaitUntil = 'commit',
	timeout_seconds: float = 30.0,
	poll_interval: float = 0.05,
	require_document_change: bool = False,
) -> DocumentSnapshot:
	"""Wait for one actor navigation to reach the requested milestone.

	This is a target-scoped barrier, not a second navigation operation. It
	registers no events and never changes focus. The session navigation
	machine remains the authority for policy, request correlation, and
	network-idle readiness.
	"""
	validate_actor_wait_until(wait_until)
	loop = asyncio.get_running_loop()
	deadline = loop.time() + max(0.01, timeout_seconds)
	interval = max(0.01, poll_interval)
	last_error: BaseException | None = None
	last_snapshot: DocumentSnapshot | None = None
	committed = False
	while True:
		remaining = deadline - loop.time()
		if remaining <= 0:
			break
		try:
			async with asyncio.timeout(remaining):
				snapshot = await read_document_snapshot(client, session_id)
		except TimeoutError:
			break
		except Exception as error:
			if _fatal_cdp_error(error):
				raise NavigationError(f'Actor navigation lost its CDP session: {error}') from error
			last_error = error
		else:
			last_snapshot = snapshot
			committed = committed or _document_committed(
				snapshot, baseline, requested_url, expected_loader, require_document_change
			)
			if committed and _milestone_met(snapshot, wait_until):
				return snapshot
		remaining = deadline - loop.time()
		if remaining <= 0:
			break
		await asyncio.sleep(min(interval, remaining))
	last_phase = DocumentReadinessPhase.COMMITTED if committed else DocumentReadinessPhase.REQUESTED
	loader = last_snapshot.loader_id if last_snapshot else None
	detail = (
		f'Actor navigation on target {target_id} to {requested_url or "the current entry"} '
		f'did not reach {wait_until} within {timeout_seconds:g}s '
		f'(phase {DocumentReadinessPhase.TIMED_OUT.value}, last observed {last_phase.value}, '
		f'loader {loader or "unknown"}). The page may still be loading.'
	)
	error = NavigationTimeoutError(detail)
	if last_error is not None:
		raise error from last_error
	raise error
