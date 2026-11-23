"""Event-driven CDP session management.

Manages CDP sessions by listening to Target.attachedToTarget and Target.detachedFromTarget
events, ensuring the session pool always reflects the current browser state.
"""

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from cdp_use.cdp.target import AttachedToTargetEvent, DetachedFromTargetEvent, SessionID, TargetID
from cdp_use.cdp.target.events import TargetCrashedEvent

from openbrowser.browser.focus import FocusChangeReason, FocusPhase, FocusTransitionBudget
from openbrowser.browser.tabs import TabState

if TYPE_CHECKING:
	from openbrowser.browser.session import BrowserSession, CDPSession


class SessionManager:
	"""Event-driven CDP session manager.

	Automatically synchronizes the CDP session pool with browser state via CDP events.

	Key features:
	- Sessions added/removed automatically via Target attach/detach events
	- Multiple sessions can attach to the same target
	- Targets only removed when ALL sessions detach
	- No stale sessions - pool always reflects browser reality
	"""

	def __init__(self, browser_session: 'BrowserSession'):
		self.browser_session = browser_session
		self.logger = browser_session.logger

		# Target -> set of sessions attached to it
		self._target_sessions: dict[TargetID, set[SessionID]] = {}

		# Session -> target mapping for reverse lookup
		self._session_to_target: dict[SessionID, TargetID] = {}

		# Target -> type cache (page, iframe, worker, etc.) - types are immutable
		self._target_types: dict[TargetID, str] = {}

		# Lock for thread-safe access
		self._lock = asyncio.Lock()

		# Track callbacks so reset can retire them before dropping the client.
		self._event_tasks: set[asyncio.Task[Any]] = set()
		self._closed = False
		self._crash_listener_registered = False
		self._monitored_root = getattr(browser_session, '_cdp_client_root', None)
		self._monitored_connection_generation = self._session_connection_generation()

	def _session_connection_generation(self) -> int | None:
		"""Return the CDP connection generation when the session exposes one."""
		health = getattr(self.browser_session, '_cdp_health', None)
		generation = getattr(health, 'connection_generation', None)
		return generation if isinstance(generation, int) else None

	def _crash_context_is_current(self) -> bool:
		"""Return True while the monitored root and connection generation still own crashes."""
		if self._closed:
			return False
		root = self.browser_session._cdp_client_root
		if root is None or root is not self._monitored_root:
			return False
		if self._monitored_connection_generation is None:
			return True
		return self._session_connection_generation() == self._monitored_connection_generation

	def _health_machine(self) -> Any | None:
		"""Return the session health tracker when it is the real explicit model."""
		from openbrowser.browser.health import BrowserHealthMachine

		health = getattr(self.browser_session, '_health', None)
		return health if isinstance(health, BrowserHealthMachine) else None

	async def start_monitoring(self) -> None:
		"""Start monitoring Target attach/detach events.

		Registers CDP event handlers to keep the session pool synchronized with browser state.
		"""
		if not self.browser_session._cdp_client_root:
			raise RuntimeError('CDP client not initialized')

		self._closed = False
		# Capture cdp_client_root in closure to avoid type errors
		cdp_client = self.browser_session._cdp_client_root
		self._monitored_root = cdp_client
		self._monitored_connection_generation = self._session_connection_generation()

		# Register synchronous event handlers (CDP requirement)
		def on_attached(event: AttachedToTargetEvent, session_id: SessionID | None = None):
			event_session_id = event['sessionId']
			target_type = event['targetInfo'].get('type', 'unknown')

			# Enable auto-attach for this session's children
			async def _enable_auto_attach():
				try:
					await cdp_client.send.Target.setAutoAttach(
						params={'autoAttach': True, 'waitForDebuggerOnStart': False, 'flatten': True},
						session_id=event_session_id,
					)
					self.logger.debug(
						f'[SessionManager] Auto-attach enabled for {target_type} session {event_session_id[:8]}...'
					)
				except Exception as e:
					error_str = str(e)
					# Expected for short-lived targets (workers, temp iframes) that detach before task executes
					if '-32001' in error_str or 'Session with given id not found' in error_str:
						self.logger.debug(
							f'[SessionManager] Auto-attach skipped for {target_type} session {event_session_id[:8]}... '
							f'(already detached - normal for short-lived targets)'
						)
					else:
						self.logger.debug(f'[SessionManager] Auto-attach failed for {target_type}: {e}')

			# Schedule auto-attach and pool management
			self._schedule_task(_enable_auto_attach())
			self._schedule_task(self._handle_target_attached(event))

		def on_detached(event: DetachedFromTargetEvent, session_id: SessionID | None = None):
			self._schedule_task(self._handle_target_detached(event))

		def on_crashed(event: TargetCrashedEvent, session_id: SessionID | None = None):
			target_id = str(event.get('targetId') or '')
			if not target_id:
				return
			status = str(event.get('status') or 'crashed')
			error_code = event.get('errorCode')
			self._schedule_task(self._handle_target_crashed(target_id, status=status, error_code=error_code))

		self.browser_session._cdp_client_root.register.Target.attachedToTarget(on_attached)
		self.browser_session._cdp_client_root.register.Target.detachedFromTarget(on_detached)
		if not self._crash_listener_registered:
			self.browser_session._cdp_client_root.register.Target.targetCrashed(on_crashed)
			self._crash_listener_registered = True

		self.logger.debug('[SessionManager] Event monitoring started')

	def _schedule_task(self, coroutine: Any) -> asyncio.Task[Any]:
		"""Track one CDP callback task so reset can retire it."""
		if self._closed:
			if inspect.iscoroutine(coroutine):
				coroutine.close()
			raise RuntimeError('SessionManager is closed')
		task = asyncio.create_task(coroutine)
		self._event_tasks.add(task)
		task.add_done_callback(self._event_task_done)
		return task

	def _event_task_done(self, task: asyncio.Task[Any]) -> None:
		self._event_tasks.discard(task)
		if task.cancelled():
			return
		try:
			error = task.exception()
		except asyncio.CancelledError:
			return
		if error is not None:
			self.logger.warning(f'[SessionManager] Event task failed: {type(error).__name__}: {error}')

	async def get_session_for_target(self, target_id: TargetID) -> 'CDPSession | None':
		"""Get the current valid session for a target.

		Args:
                target_id: Target ID to get session for

		Returns:
                CDPSession if exists, None if target has detached
		"""
		async with self._lock:
			return self.browser_session._cdp_session_pool.get(target_id)

	async def active_session_ids(
		self,
		target_id: TargetID,
		*,
		budget: FocusTransitionBudget | None = None,
	) -> tuple[SessionID, ...]:
		"""Return one target's active session IDs in deterministic order."""
		acquired = False
		try:
			if budget is None:
				await self._lock.acquire()
			else:
				if budget.remaining() <= 0:
					raise TimeoutError('Focus budget expired before session lookup')
				await asyncio.wait_for(self._lock.acquire(), timeout=max(0.001, budget.remaining()))
			acquired = True
			if budget is not None and budget.remaining() <= 0:
				raise TimeoutError('Focus budget expired during session lookup')
			return tuple(sorted(self._target_sessions.get(target_id, set())))
		finally:
			if acquired:
				self._lock.release()

	async def get_live_session_for_target(
		self,
		target_id: TargetID,
		*,
		budget: FocusTransitionBudget | None = None,
	) -> 'CDPSession | None':
		"""Return a wrapper only when its exact session ID is active.

		A stale pooled wrapper is replaced with a new wrapper for the lowest
		active session ID. The focused wrapper object is never mutated here.
		"""
		acquired = False
		try:
			if budget is None:
				await self._lock.acquire()
			else:
				if budget.remaining() <= 0:
					raise TimeoutError('Focus budget expired before session lookup')
				await asyncio.wait_for(self._lock.acquire(), timeout=max(0.001, budget.remaining()))
			acquired = True
			if budget is not None and budget.remaining() <= 0:
				raise TimeoutError('Focus budget expired during session lookup')
			active_ids = tuple(sorted(self._target_sessions.get(target_id, set())))
			if not active_ids:
				return None
			wrapper = self.browser_session._cdp_session_pool.get(target_id)
			if wrapper is not None and wrapper.session_id in active_ids:
				return wrapper

			from openbrowser.browser.session import CDPSession

			root = self.browser_session._cdp_client_root
			if root is None:
				return None
			replacement = CDPSession(
				cdp_client=root,
				target_id=target_id,
				session_id=active_ids[0],
				title=getattr(wrapper, 'title', 'Unknown title') if wrapper is not None else 'Unknown title',
				url=getattr(wrapper, 'url', 'about:blank') if wrapper is not None else 'about:blank',
			)
			self.browser_session._cdp_session_pool[target_id] = replacement
			return replacement
		finally:
			if acquired:
				self._lock.release()

	async def validate_session(self, target_id: TargetID) -> bool:
		"""Check if a target still has active sessions.

		Args:
                target_id: Target ID to validate

		Returns:
                True if target has active sessions, False if it should be removed
		"""
		async with self._lock:
			if target_id not in self._target_sessions:
				return False

			return len(self._target_sessions[target_id]) > 0

	def cancel_pending(self) -> None:
		"""Cancel callback tasks without awaiting, for forced runtime reset."""
		self._closed = True
		current = asyncio.current_task()
		for task in list(self._event_tasks):
			if task is not current and not task.done():
				task.cancel()
		self._event_tasks.clear()

	async def clear(self) -> None:
		"""Clear all session tracking and retire tracked callbacks."""
		self._closed = True
		current = asyncio.current_task()
		tasks = {task for task in self._event_tasks if task is not current and not task.done()}
		for task in tasks:
			task.cancel()
		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)
		self._event_tasks.clear()

		async with self._lock:
			self._target_sessions.clear()
			self._session_to_target.clear()
			self._target_types.clear()

		self.logger.info('[SessionManager] Cleared all session tracking')

	async def is_target_valid(self, target_id: TargetID) -> bool:
		"""Check if a target is still valid and has active sessions.

		Args:
                target_id: Target ID to validate

		Returns:
                True if target is valid and has active sessions, False otherwise
		"""
		async with self._lock:
			if target_id not in self._target_sessions:
				return False
			return len(self._target_sessions[target_id]) > 0

	async def _handle_target_attached(self, event: AttachedToTargetEvent) -> None:
		"""Handle Target.attachedToTarget event.

		Called automatically by Chrome when a new target/session is created.
		This is the ONLY place where sessions are added to the pool.
		"""
		target_id = event['targetInfo']['targetId']
		session_id = event['sessionId']
		target_type = event['targetInfo']['type']
		waiting_for_debugger = event.get('waitingForDebugger', False)

		self.logger.debug(
			f'[SessionManager] Target attached: {target_id[:8]}... (session={session_id[:8]}..., '
			f'type={target_type}, waitingForDebugger={waiting_for_debugger})'
		)

		rebound_session = None
		async with self._lock:
			if self._closed:
				return
			# Track this session for the target
			if target_id not in self._target_sessions:
				self._target_sessions[target_id] = set()

			self._target_sessions[target_id].add(session_id)
			self._session_to_target[session_id] = target_id

			# Cache target type (immutable, set once)
			if target_id not in self._target_types:
				self._target_types[target_id] = target_type

			# Create or replace the CDPSession wrapper. Never mutate a wrapper
			# that a committed focus snapshot may still reference.
			from openbrowser.browser.session import CDPSession

			assert self.browser_session._cdp_client_root is not None, 'Root CDP client required'
			existing = self.browser_session._cdp_session_pool.get(target_id)
			rebound_session = CDPSession(
				cdp_client=self.browser_session._cdp_client_root,
				target_id=target_id,
				session_id=session_id,
				title=event['targetInfo'].get('title', getattr(existing, 'title', 'Unknown title')),
				url=event['targetInfo'].get('url', getattr(existing, 'url', 'about:blank')),
			)
			self.browser_session._cdp_session_pool[target_id] = rebound_session

			if existing is None:
				self.logger.debug(
					f'[SessionManager] Created session for target {target_id[:8]}... '
					f'(pool size: {len(self.browser_session._cdp_session_pool)})'
				)
			else:
				self.logger.debug(
					f'[SessionManager] Replaced session wrapper for target {target_id[:8]}... '
					f'(session={session_id[:8]}...)'
				)

			if target_type == 'page':
				tab_url = event['targetInfo'].get('url')
				tab_title = event['targetInfo'].get('title')
				if self.browser_session._tab_registry.get(target_id) is None:
					self.browser_session._tab_registry.register(
						target_id,
						url=tab_url,
						title=tab_title,
						state=TabState.OPEN,
					)
				else:
					self.browser_session._tab_registry.update(
						target_id,
						url=tab_url,
						title=tab_title,
						state=TabState.OPEN,
					)

		if rebound_session is not None:
			handler = getattr(self.browser_session, '_handle_attached_session', None)
			if callable(handler):
				try:
					result = handler(target_id, rebound_session)
					if inspect.isawaitable(result):
						await result
				except Exception as error:
					self.logger.warning(
						f'[SessionManager] Focus rebind failed for target {target_id[:8]}...: '
						f'{type(error).__name__}: {error}'
					)

		# Resume execution if waiting for debugger
		if waiting_for_debugger:
			try:
				assert self.browser_session._cdp_client_root is not None
				await self.browser_session._cdp_client_root.send.Runtime.runIfWaitingForDebugger(session_id=session_id)
				self.logger.debug(f'[SessionManager] Resumed execution for session {session_id[:8]}...')
			except Exception as e:
				self.logger.warning(f'[SessionManager] Failed to resume execution: {e}')

	async def _handle_target_detached(self, event: DetachedFromTargetEvent) -> None:
		"""Handle Target.detachedFromTarget event.

		Called automatically by Chrome when a target/session is destroyed.
		This is the ONLY place where sessions are removed from the pool.
		"""
		session_id = event['sessionId']
		target_id = event.get('targetId')  # May be empty

		# If targetId not in event, look it up via session mapping
		if not target_id:
			async with self._lock:
				target_id = self._session_to_target.get(session_id)

		if not target_id:
			self.logger.warning(f'[SessionManager] Session detached but target unknown (session={session_id[:8]}...)')
			return

		target_fully_removed = False
		target_type = None
		remaining_session_ids: tuple[SessionID, ...] = ()

		async with self._lock:
			if self._closed:
				return
			# Remove this session from target's session set
			if target_id in self._target_sessions:
				self._target_sessions[target_id].discard(session_id)

				remaining_sessions = len(self._target_sessions[target_id])
				remaining_session_ids = tuple(sorted(self._target_sessions[target_id]))

				self.logger.debug(
					f'[SessionManager] Session detached: target={target_id[:8]}... '
					f'session={session_id[:8]}... (remaining={remaining_sessions})'
				)

				# Only remove target when NO sessions remain
				if remaining_sessions == 0:
					self.logger.debug(
						f'[SessionManager] No sessions remain for target {target_id[:8]}..., removing from pool'
					)

					target_fully_removed = True

					# Remove from pool
					if target_id in self.browser_session._cdp_session_pool:
						self.browser_session._cdp_session_pool.pop(target_id)
						self.browser_session.forget_dom_cache(target_id)
						self.logger.debug(
							f'[SessionManager] Removed target {target_id[:8]}... from pool '
							f'(pool size: {len(self.browser_session._cdp_session_pool)})'
						)

					# Clean up tracking
					del self._target_sessions[target_id]
			else:
				# Target not tracked - already removed or never attached
				self.logger.debug(
					f'[SessionManager] Session detached from untracked target: target={target_id[:8]}... '
					f'session={session_id[:8]}... (target was already removed or attach event was missed)'
				)

			# Get target type before cleaning up cache (needed for TabClosedEvent dispatch)
			target_type = self._target_types.get(target_id)

			# Clean up target type cache if target fully removed
			if target_id not in self._target_sessions and target_id in self._target_types:
				del self._target_types[target_id]

			# Remove from reverse mapping
			if session_id in self._session_to_target:
				del self._session_to_target[session_id]

		target_exists = await self._target_exists(target_id) if target_fully_removed else None

		# Detaching the last CDP session does not destroy the target. Dispatch a
		# terminal event only when the authoritative target list omits it.
		if target_fully_removed:
			if target_type in ('page', 'tab'):
				if target_exists is False:
					from openbrowser.browser.events import TabClosedEvent

					self.browser_session.event_bus.dispatch(TabClosedEvent(target_id=target_id))
					self.logger.debug(f'[SessionManager] Dispatched TabClosedEvent for page target {target_id[:8]}...')
				elif target_exists is True:
					self.logger.debug(
						f'[SessionManager] Last session detached from live target {target_id[:8]}...; '
						'not dispatching TabClosedEvent'
					)
				else:
					self.logger.debug(
						f'[SessionManager] Target absence is unconfirmed for {target_id[:8]}...; '
						'not dispatching TabClosedEvent'
					)
			elif target_type:
				self.logger.debug(
					f'[SessionManager] Target {target_id[:8]}... fully removed (type={target_type}) - not dispatching TabClosedEvent'
				)

		# Resolve exact focused-session loss outside the manager lock.
		handler = getattr(self.browser_session, '_handle_detached_session', None)
		if callable(handler):
			try:
				result = handler(target_id, session_id, remaining_session_ids)
				if inspect.isawaitable(result):
					await result
			except Exception as error:
				self.logger.warning(
					f'[SessionManager] Focus detach repair failed for target {target_id[:8]}...: '
					f'{type(error).__name__}: {error}'
				)

	async def _target_exists(self, target_id: TargetID) -> bool | None:
		"""Return the authoritative target-list verdict, or None when unknown."""
		root = self.browser_session._cdp_client_root
		if root is None:
			return None
		try:
			result = await asyncio.wait_for(root.send.Target.getTargets(), timeout=2.0)
		except Exception:
			return None
		if not isinstance(result, dict):
			return None
		target_infos = result.get('targetInfos')
		if not isinstance(target_infos, list):
			return None
		return any(isinstance(target, dict) and target.get('targetId') == target_id for target in target_infos)

	async def _handle_target_crashed(
		self,
		target_id: TargetID,
		*,
		status: str,
		error_code: Any = None,
	) -> None:
		"""Mark a page crash and start focused recovery."""
		if not self._crash_context_is_current():
			return
		target_type = self._target_types.get(target_id)
		if target_type != 'page':
			return
		message = f'Target crashed: status={status}, errorCode={error_code}'
		health = self._health_machine()
		generation = self._monitored_connection_generation or 0
		token = None
		if health is not None:
			token = health.record_target_crash(generation, target_id, message)
			if token is None:
				return
		self.browser_session._tab_registry.mark_crashed(target_id, message)
		from openbrowser.browser.events import BrowserErrorEvent

		self.browser_session.event_bus.dispatch(
			BrowserErrorEvent(
				error_type='TargetCrash',
				message=f'Target {target_id[-8:]} crashed',
				details={
					'target_id': target_id,
					'status': status,
					'error_code': error_code,
				},
			)
		)
		snapshot = self.browser_session.focus_snapshot()
		if snapshot.phase is not FocusPhase.FOCUSED or snapshot.target_id != target_id:
			return
		if (
			health is not None
			and token is not None
			and not health.record_focus_recovery_started(token, generation)
		):
			return
		try:
			recovered = await self.browser_session._recover_agent_focus(
				target_id,
				reason=FocusChangeReason.CRASH_RECOVERY,
				budget=FocusTransitionBudget.start(),
				expected_generation=snapshot.generation,
			)
		except asyncio.CancelledError:
			raise
		except Exception as error:
			if not self._crash_context_is_current():
				return
			if (
				health is not None
				and token is not None
				and not health.record_focus_recovery_failure(token, generation)
			):
				return
			report = getattr(self.browser_session, '_report_focus_recovery_failure', None)
			if callable(report):
				report(target_id, error)
			else:
				self.logger.warning(
					f'[SessionManager] Focus recovery after crash failed for target {target_id[:8]}...: '
					f'{type(error).__name__}: {error}'
				)
		else:
			if not self._crash_context_is_current():
				return
			if health is not None and token is not None and recovered.target_id is not None:
				health.record_focus_recovery_success(
					token,
					generation,
					recovered.target_id,
				)
