"""Persistent, server-owned storage for the Context Atlas Jev credential."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from openbrowser.config import Config


logger = logging.getLogger(__name__)
CREDENTIAL_FILENAME = "jev_credentials.json"


class CredentialStoreError(RuntimeError):
	"""Raised when the persistent Jev credential cannot be read or written."""


class CredentialStore:
	"""Read and atomically update one local Context Atlas credential."""

	def __init__(self, *, path: Path | None = None) -> None:
		self._path = Path(path) if path is not None else _default_path()

	@property
	def path(self) -> Path:
		return self._path

	def load(self) -> str | None:
		if not self._path.exists():
			return None
		try:
			payload = json.loads(self._path.read_text(encoding="utf-8"))
		except (OSError, UnicodeError, json.JSONDecodeError) as exc:
			raise CredentialStoreError("could not read the stored credential") from exc
		if not isinstance(payload, dict):
			raise CredentialStoreError("stored credential has an invalid format")
		value = payload.get("api_key")
		if not isinstance(value, str) or not value.strip():
			raise CredentialStoreError("stored credential must contain a non-empty key")
		try:
			os.chmod(self._path, 0o600)
		except OSError as exc:
			raise CredentialStoreError("could not secure the stored credential") from exc
		return value.strip()

	def configured(self) -> bool:
		return self.load() is not None

	def save(self, api_key: str) -> None:
		if not isinstance(api_key, str) or not api_key.strip():
			raise CredentialStoreError("credential must be non-empty")
		value = api_key.strip()
		parent = self._path.parent
		temporary_path: Path | None = None
		try:
			parent.mkdir(parents=True, exist_ok=True)
			os.chmod(parent, 0o700)
			file_descriptor, temporary_name = tempfile.mkstemp(
				prefix=f".{self._path.stem}.",
				dir=parent,
				text=True,
			)
			temporary_path = Path(temporary_name)
			os.chmod(temporary_path, 0o600)
			with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
				json.dump({"api_key": value}, handle, ensure_ascii=False, separators=(",", ":"))
				handle.flush()
				os.fsync(handle.fileno())
			os.replace(temporary_path, self._path)
			os.chmod(self._path, 0o600)
		except (OSError, TypeError, ValueError) as exc:
			if temporary_path is not None:
				try:
					temporary_path.unlink(missing_ok=True)
				except OSError:
					logger.debug("unable to remove temporary credential file", exc_info=True)
			raise CredentialStoreError("could not persist the credential") from exc

	def clear(self) -> None:
		try:
			self._path.unlink(missing_ok=True)
		except OSError as exc:
			raise CredentialStoreError("could not clear the stored credential") from exc


def _default_path() -> Path:
	return Config().OPENBROWSER_CONFIG_DIR / "context-atlas" / CREDENTIAL_FILENAME


__all__ = ["CREDENTIAL_FILENAME", "CredentialStore", "CredentialStoreError"]
