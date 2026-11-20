"""Explicit storage-state load and save phases shared by browser owners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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


class StorageStateSavePhase(str, Enum):
	"""Lifecycle of one file-backed storage-state save."""

	IDLE = 'idle'
	NOT_CONFIGURED = 'not_configured'
	SAVING = 'saving'
	SAVED = 'saved'
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
	skipped_items: int = 0
	skipped_origins: int = 0

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
			'skipped_items': self.skipped_items,
			'skipped_origins': self.skipped_origins,
		}


@dataclass(frozen=True)
class StorageStateSaveSnapshot:
	"""JSON-safe view of a storage-state save."""

	phase: StorageStateSavePhase
	path: str | None = None
	bytes: int | None = None
	save_ms: float | None = None
	elapsed_ms: float | None = None
	error: str | None = None
	detail: str | None = None
	cookies_count: int = 0
	origins_count: int = 0

	def to_dict(self) -> dict[str, Any]:
		return {
			'phase': self.phase.value,
			'path': self.path,
			'bytes': self.bytes,
			'save_ms': self.save_ms,
			'elapsed_ms': self.elapsed_ms,
			'error': self.error,
			'detail': self.detail,
			'cookies_count': self.cookies_count,
			'origins_count': self.origins_count,
		}


def _expand_path(path: str | Path) -> Path:
	return Path(path).expanduser().resolve()


def load_snapshot_for_configuration(configured: object) -> StorageStateSnapshot:
	"""Derive the truthful pre-load view from a configured storage source."""
	if configured is None or configured == '':
		return StorageStateSnapshot(phase=StorageStatePhase.NOT_CONFIGURED)
	if not isinstance(configured, (str, Path)):
		return StorageStateSnapshot(
			phase=StorageStatePhase.SKIPPED,
			detail='in-memory storage state is not loaded from a file',
		)
	path: Path | None = None
	try:
		path = _expand_path(configured)
		size = path.stat().st_size
	except FileNotFoundError:
		return StorageStateSnapshot(
			phase=StorageStatePhase.SKIPPED,
			path=str(path) if path is not None else str(configured),
			detail='storage-state file does not exist',
		)
	except (OSError, RuntimeError) as error:
		return StorageStateSnapshot(
			phase=StorageStatePhase.FAILED,
			path=str(configured),
			error=f'{type(error).__name__}: {error}',
			detail='storage-state path could not be resolved or inspected',
		)
	return StorageStateSnapshot(
		phase=StorageStatePhase.PENDING,
		path=str(path),
		bytes=size,
	)


def save_snapshot_for_configuration(configured: object) -> StorageStateSaveSnapshot:
	"""Derive the truthful pre-save view from a configured storage source."""
	if configured is None or configured == '':
		return StorageStateSaveSnapshot(phase=StorageStateSavePhase.NOT_CONFIGURED)
	if not isinstance(configured, (str, Path)):
		return StorageStateSaveSnapshot(
			phase=StorageStateSavePhase.SKIPPED,
			detail='in-memory storage state is not saved to a file',
		)
	try:
		path = _expand_path(configured)
	except (OSError, RuntimeError) as error:
		return StorageStateSaveSnapshot(
			phase=StorageStateSavePhase.FAILED,
			path=str(configured),
			error=f'{type(error).__name__}: {error}',
			detail='storage-state path could not be resolved',
		)
	return StorageStateSaveSnapshot(phase=StorageStateSavePhase.IDLE, path=str(path))
