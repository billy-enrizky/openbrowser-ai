"""Cross-platform ownership leases for browser user-data directories."""

from __future__ import annotations

import errno
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import psutil

if os.name == 'nt':
	import msvcrt  # type: ignore[import-not-found]
else:
	import fcntl


IS_WINDOWS = os.name == 'nt'


class ProfileInUseError(RuntimeError):
	"""Raised when another owner holds a browser profile lease."""

	def __init__(self, profile_dir: str | Path):
		self.profile_dir = str(Path(profile_dir).expanduser().resolve())
		super().__init__(f'Browser profile already in use: {self.profile_dir}')


class ProfileLease:
	"""Hold an exclusive OS lock and ownership record for one profile."""

	LOCK_FILENAME = '.openbrowser-profile.lock'
	METADATA_FILENAME = '.openbrowser-profile.json'
	SCHEMA_VERSION = 1

	def __init__(
		self,
		profile_dir: str | Path,
		*,
		instance_id: str | None = None,
		owner_pid: int | None = None,
		owner_start_time: float | None = None,
	):
		self.profile_dir = Path(profile_dir).expanduser().resolve()
		self.instance_id = instance_id or uuid.uuid4().hex
		self.owner_pid = owner_pid if owner_pid is not None else os.getpid()
		self.owner_start_time = (
			float(owner_start_time)
			if owner_start_time is not None
			else self._get_process_start_time(self.owner_pid)
		)
		self._lock_handle: BinaryIO | None = None
		self.previous_metadata: dict[str, Any] | None = None

	@property
	def lock_path(self) -> Path:
		return self.profile_dir / self.LOCK_FILENAME

	@property
	def metadata_path(self) -> Path:
		return self.profile_dir / self.METADATA_FILENAME

	@property
	def acquired(self) -> bool:
		return self._lock_handle is not None

	def acquire(self) -> dict[str, Any] | None:
		"""Acquire the profile lock and return metadata from the prior owner."""
		if self.acquired:
			return self.previous_metadata

		self.profile_dir.mkdir(parents=True, exist_ok=True)
		try:
			handle = self.lock_path.open('a+b')
		except OSError:
			raise

		try:
			self._lock(handle)
		except OSError as exc:
			handle.close()
			if isinstance(exc, BlockingIOError) or exc.errno in (errno.EACCES, errno.EAGAIN):
				raise ProfileInUseError(self.profile_dir) from exc
			raise

		self._lock_handle = handle
		self.previous_metadata = self.read_metadata(self.profile_dir)
		return self.previous_metadata

	def write_metadata(self, browser: dict[str, Any] | None = None) -> None:
		"""Atomically write metadata for this acquired lease."""
		if not self.acquired:
			raise RuntimeError('Cannot write profile metadata before acquiring the lease')

		metadata = {
			'schema_version': self.SCHEMA_VERSION,
			'profile_dir': str(self.profile_dir),
			'instance_id': self.instance_id,
			'owner_pid': self.owner_pid,
			'owner_start_time': self.owner_start_time,
			'browser': browser,
		}
		self._atomic_write(metadata)

	def record_browser(
		self,
		process: psutil.Process,
		*,
		ownership_marker: str,
		executable: str,
		cdp_port: int,
	) -> dict[str, Any]:
		"""Record the identity and launch details of the owned browser."""
		browser = {
			'pid': process.pid,
			'start_time': process.create_time(),
			'profile_dir': str(self.profile_dir),
			'instance_id': self.instance_id,
			'ownership_marker': ownership_marker,
			'executable': executable,
			'cdp_port': cdp_port,
		}
		self.write_metadata(browser=browser)
		return browser

	def clear_browser(self) -> None:
		"""Keep the owner record but clear the browser process record."""
		self.write_metadata(browser=None)

	def release(self) -> None:
		"""Remove only this owner's metadata and release the OS lock."""
		handle = self._lock_handle
		if handle is None:
			return

		try:
			metadata = self.read_metadata(self.profile_dir)
			if metadata and self._metadata_belongs_to_this_lease(metadata):
				self.metadata_path.unlink(missing_ok=True)
		finally:
			try:
				self._unlock(handle)
			finally:
				handle.close()
				self._lock_handle = None
				self.previous_metadata = None

	@classmethod
	def read_metadata(cls, profile_dir: str | Path) -> dict[str, Any] | None:
		"""Read valid ownership metadata, returning None for stale/corrupt data."""
		path = Path(profile_dir).expanduser().resolve() / cls.METADATA_FILENAME
		try:
			value = json.loads(path.read_text(encoding='utf-8'))
		except (OSError, ValueError, TypeError):
			return None
		if not isinstance(value, dict) or value.get('schema_version') != cls.SCHEMA_VERSION:
			return None
		return value

	@staticmethod
	def process_matches_identity(process: Any, pid: int | str, start_time: float | str) -> bool:
		"""Return true only when PID and OS creation time both match."""
		try:
			return int(process.pid) == int(pid) and float(process.create_time()) == float(start_time)
		except (AttributeError, TypeError, ValueError, OSError, psutil.Error):
			return False

	@classmethod
	def process_matches_browser(cls, process: Any, browser: dict[str, Any]) -> bool:
		"""Verify the recorded PID, profile, and OpenBrowser marker."""
		try:
			if not cls.process_matches_identity(process, browser['pid'], browser['start_time']):
				return False
			profile_dir = str(Path(browser['profile_dir']).expanduser().resolve())
			marker = str(browser['ownership_marker'])
			cmdline = process.cmdline()
			if not isinstance(cmdline, (list, tuple)):
				return False
			profile_matches = False
			for argument in cmdline:
				if not isinstance(argument, str) or not argument.startswith('--user-data-dir='):
					continue
				try:
					profile_matches = str(Path(argument.split('=', 1)[1]).expanduser().resolve()) == profile_dir
				except OSError:
					profile_matches = False
				if profile_matches:
					break
			return profile_matches and marker in cmdline and str(browser['instance_id']) in marker
		except (AttributeError, KeyError, TypeError, ValueError, OSError, psutil.Error):
			return False

	@staticmethod
	def process_is_alive(pid: int | str, start_time: float | str) -> bool:
		"""Check a process identity without trusting a reused PID."""
		try:
			process = psutil.Process(int(pid))
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ValueError, OSError):
			return False
		return ProfileLease.process_matches_identity(process, pid, start_time)

	@staticmethod
	def _get_process_start_time(pid: int) -> float:
		try:
			return float(psutil.Process(pid).create_time())
		except (psutil.Error, OSError, ValueError) as exc:
			raise RuntimeError(f'Unable to determine owner process start time for pid={pid}') from exc

	def _metadata_belongs_to_this_lease(self, metadata: dict[str, Any]) -> bool:
		return (
			metadata.get('instance_id') == self.instance_id
			and metadata.get('owner_pid') == self.owner_pid
			and float(metadata.get('owner_start_time')) == self.owner_start_time
		)

	def _atomic_write(self, metadata: dict[str, Any]) -> None:
		file_descriptor, temporary_name = tempfile.mkstemp(
			prefix='.openbrowser-profile-', suffix='.tmp', dir=self.profile_dir
		)
		try:
			with os.fdopen(file_descriptor, 'w', encoding='utf-8') as temporary_file:
				json.dump(metadata, temporary_file, sort_keys=True)
				temporary_file.flush()
				os.fsync(temporary_file.fileno())
			os.replace(temporary_name, self.metadata_path)
		finally:
			try:
				Path(temporary_name).unlink(missing_ok=True)
			except OSError:
				pass

	@staticmethod
	def _lock(handle: BinaryIO) -> None:
		if IS_WINDOWS:
			handle.seek(0, os.SEEK_END)
			if handle.tell() == 0:
				handle.write(b'\0')
				handle.flush()
			handle.seek(0)
			msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[name-defined]
		else:
			fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

	@staticmethod
	def _unlock(handle: BinaryIO) -> None:
		if IS_WINDOWS:
			handle.seek(0)
			msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[name-defined]
		else:
			fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
