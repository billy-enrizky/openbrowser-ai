"""Explicit storage-state load phases shared by the watchdog and daemon."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StorageStatePhase(str, Enum):
	"""Lifecycle of one file-backed storage-state load."""

	NOT_CONFIGURED = 'not_configured'
	PENDING = 'pending'
	LOADING = 'loading'
	LOADED = 'loaded'
	FAILED = 'failed'
	SKIPPED = 'skipped'
	CANCELLED = 'cancelled'


@dataclass(frozen=True)
class StorageStateSnapshot:
	"""JSON-safe view of a storage-state load."""

	phase: StorageStatePhase
	path: str | None = None
	bytes: int | None = None
	load_ms: float | None = None
	elapsed_ms: float | None = None
	error: str | None = None
	detail: str | None = None
	cookies_count: int = 0
	origins_count: int = 0
	script_count: int = 0
	storage_script_attempts: int = 0

	def to_dict(self) -> dict[str, Any]:
		return {
			'phase': self.phase.value,
			'path': self.path,
			'bytes': self.bytes,
			'load_ms': self.load_ms,
			'elapsed_ms': self.elapsed_ms,
			'error': self.error,
			'detail': self.detail,
			'cookies_count': self.cookies_count,
			'origins_count': self.origins_count,
			'script_count': self.script_count,
			'storage_script_attempts': self.storage_script_attempts,
		}
