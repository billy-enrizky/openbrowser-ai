from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal

WaitUntil = Literal['load', 'domcontentloaded', 'networkidle', 'commit']


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
	_response_kind: Literal['document', 'same_document', 'download'] | None = None
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

	def capture_document(self, frame: dict[str, Any]) -> None:
		self._current_frame = frame['id']
		self._current_loader = frame.get('loaderId')
		self.observed_url = frame.get('url', '') + frame.get('urlFragment', '')

	def observe(self, kind: str, event: dict[str, Any]) -> None:
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
		if self._response_kind is not None and kind != 'within' and payload.get('loaderId') != self.loader_id:
			return
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

	@property
	def ready(self) -> bool:
		if self.terminal or self._response_kind is None:
			return False
		if self.is_download:
			return True
		# A loader-less response acknowledges same-document navigation.
		same_document = self._response_kind == 'same_document'
		milestone = {'domcontentloaded': 'DOMContentLoaded', 'load': 'load', 'networkidle': 'networkIdle'}.get(self.wait_until)
		ready = same_document and self.wait_until == 'commit'
		for kind, data in self._evidence:
			frame_id = data.get('id') if kind == 'frame' else data.get('frameId')
			if frame_id != self.frame_id:
				continue
			if kind == 'within' and same_document:
				self.observed_url = data.get('url', self.observed_url)
			elif data.get('loaderId') == self.loader_id:
				if kind == 'frame':
					self.observed_url = data.get('url', '') + data.get('urlFragment', '')
				if self.wait_until == 'commit' or (kind == 'lifecycle' and data.get('name') == milestone):
					ready = True
		return ready

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
		)} | {'phase': self.phase.value}


class NavigationMonitor:
	"""Own one request and one synchronous callback set per root CDP binding."""

	def __init__(self) -> None:
		self.active: NavigationOperation | None = None
		self._last: dict[str, Any] | None = None
		self._root: Any = None

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

	def bind(self, root: Any) -> None:
		if root is self._root:
			return
		self._root = root
		for method, kind in (('frameNavigated', 'frame'), ('lifecycleEvent', 'lifecycle'), ('navigatedWithinDocument', 'within')):
			def callback(event: dict[str, Any], session_id: str | None = None, *, _kind: str = kind) -> None:
				operation = self.active
				if self._root is root and operation is not None and operation.session_id == session_id:
					operation.observe(_kind, event)
			getattr(root.register.Page, method)(callback)
