"""Cancellation-safe draining for application-owned work.

The MCP adapter owns browser, startup, and shutdown work. A caller can be
cancelled while that work is still settling. These helpers keep the caller's
first cancellation as control flow and absorb later cancellation of the drain
loop so the owned worker always reaches its final state.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio


@dataclass(frozen=True)
class _CleanupOutcome:
	succeeded: bool
	caller_cancellation: asyncio.CancelledError | None = None


async def _drain_owned_completion(
	completion: asyncio.Future[Any],
	*,
	on_first_cancellation: Callable[[], object] | None = None,
) -> asyncio.CancelledError | None:
	"""Await an owned operation and return the caller's first cancellation.

	Once the caller cancels, the AnyIO shield prevents the MCP request scope
	from re-cancelling this drain loop. Direct ``Task.cancel()`` calls are
	still absorbed, so the owned operation always reaches its final state
	before the request unwinds.
	"""
	caller_cancellation: asyncio.CancelledError | None = None
	while not completion.done():
		try:
			await asyncio.shield(completion)
		except asyncio.CancelledError as error:
			if caller_cancellation is not None:
				continue
			caller_cancellation = error
			if on_first_cancellation is not None:
				on_first_cancellation()
			with anyio.CancelScope(shield=True):
				while not completion.done():
					try:
						await asyncio.shield(completion)
					except asyncio.CancelledError:
						continue
			break
	return caller_cancellation
