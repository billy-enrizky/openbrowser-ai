"""Cross-process ownership lock for one daemon socket."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import BinaryIO


class DaemonOwnerLock:
    """Hold one exclusive OS lock for the lifetime of a daemon process."""

    def __init__(self, path: Path):
        self.path = path
        self._handle: BinaryIO | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        """Try to take the lock without blocking."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, 'r+b', buffering=0)
        try:
            if os.name == 'nt':
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b'\0')
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise
        self._handle = handle
        return True

    def release(self) -> None:
        """Release the lock. Safe to call more than once."""
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == 'nt':
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> 'DaemonOwnerLock':
        if not self.acquire():
            raise RuntimeError(f'Daemon ownership lock is already held: {self.path}')
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
