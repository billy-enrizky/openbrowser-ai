"""Explicit policy for carrying user variables across browser recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_WARNINGS = 20

# Names the new namespace always provides. They are rebuilt, never carried.
FRAMEWORK_NAMES = frozenset(
	{
		'browser',
		'file_system',
		'json',
		'asyncio',
		'Path',
		'csv',
		're',
		'datetime',
		'requests',
		'np',
		'numpy',
		'pd',
		'pandas',
		'plt',
		'matplotlib',
		'BeautifulSoup',
		'bs4',
		'PdfReader',
		'pypdf',
		'tabulate',
		'evaluate',
		'done',
		'navigate',
		'click',
		'input_text',
		'scroll',
		'send_keys',
		'select_dropdown',
		'click_xy',
		'hover_xy',
		'scroll_xy',
		'get_selector_from_index',
		'download_file',
		'list_downloads',
		'wait_for_download',
		'download_outcomes',
		'download_failures',
		'close',
		'dropdown_options',
		'go_back',
		'input',
		'switch',
		'upload_file',
		'wait',
	}
)

_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes, bytearray)
_MAX_LABEL_LENGTH = 120


def _truncate_label(value: str, limit: int = _MAX_LABEL_LENGTH) -> str:
	if len(value) <= limit:
		return value
	return value[:limit] + '...'


class NamespaceValueKind(str, Enum):
	"""How one namespace value is treated during a transfer."""

	SCALAR = 'scalar'
	FRAMEWORK = 'framework'
	OPAQUE = 'opaque'


@dataclass
class NamespaceTransfer:
	"""Result of classifying one namespace before a browser rebuild."""

	variables: dict[str, Any] = field(default_factory=dict)
	framework_names: list[str] = field(default_factory=list)
	dropped: dict[str, str] = field(default_factory=dict)
	warnings: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, object]:
		"""Return a bounded, JSON-safe summary for status and reset responses."""
		preserved = [_truncate_label(name) for name in sorted(self.variables)]
		framework = [_truncate_label(name) for name in sorted(self.framework_names)]
		dropped_items = [
			(_truncate_label(name), _truncate_label(reason, 200))
			for name, reason in self.dropped.items()
		]
		return {
			'preserved_count': len(preserved),
			'preserved': preserved[:20],
			'preserved_truncated': len(preserved) > 20,
			'framework_count': len(framework),
			'framework': framework[:20],
			'framework_truncated': len(framework) > 20,
			'dropped_count': len(dropped_items),
			'dropped': dict(dropped_items[:20]),
			'dropped_truncated': len(dropped_items) > 20,
			'warnings': [_truncate_label(warning, 200) for warning in self.warnings],
		}


def _is_framework_name(name: str) -> bool:
	"""Only names the replacement namespace actually binds are framework."""
	return name in FRAMEWORK_NAMES


def _classify(value: Any, old_browser: Any, seen: set[int]) -> tuple[NamespaceValueKind, str | None]:
	if type(value) in _SCALAR_TYPES:
		return NamespaceValueKind.SCALAR, None
	if old_browser is not None and value is old_browser:
		return NamespaceValueKind.OPAQUE, 'references the previous browser session'

	if type(value) is dict:
		if id(value) in seen:
			return NamespaceValueKind.OPAQUE, 'cyclic container'
		seen.add(id(value))
		try:
			for key, item in value.items():
				if type(key) not in _SCALAR_TYPES:
					return NamespaceValueKind.OPAQUE, f'non-scalar dict key ({type(key).__name__})'
				kind, reason = _classify(item, old_browser, seen)
				if kind is not NamespaceValueKind.SCALAR:
					return NamespaceValueKind.OPAQUE, reason or 'contains a non-scalar value'
			return NamespaceValueKind.SCALAR, None
		finally:
			seen.discard(id(value))

	if type(value) in (list, tuple, set, frozenset):
		if id(value) in seen:
			return NamespaceValueKind.OPAQUE, 'cyclic container'
		seen.add(id(value))
		try:
			for item in value:
				kind, reason = _classify(item, old_browser, seen)
				if kind is not NamespaceValueKind.SCALAR:
					if reason and 'previous browser' in reason:
						return NamespaceValueKind.OPAQUE, reason
					return NamespaceValueKind.OPAQUE, f'contains a non-scalar value ({type(item).__name__})'
			return NamespaceValueKind.SCALAR, None
		finally:
			seen.discard(id(value))

	return NamespaceValueKind.OPAQUE, f'unsupported type {type(value).__name__}'


def _copy_scalar(value: Any, memo: dict[int, Any]) -> Any:
	"""Deep-copy a scalar-only value while preserving shared references."""
	if isinstance(value, (bytes, str, int, float, complex, bool, type(None))):
		return value
	existing = memo.get(id(value))
	if existing is not None:
		return existing
	if type(value) is list:
		copied: Any = []
		memo[id(value)] = copied
		copied.extend(_copy_scalar(item, memo) for item in value)
		return copied
	if type(value) is tuple:
		copied = tuple(_copy_scalar(item, memo) for item in value)
		memo[id(value)] = copied
		return copied
	if type(value) is set:
		copied = {_copy_scalar(item, memo) for item in value}
		memo[id(value)] = copied
		return copied
	if type(value) is frozenset:
		copied = frozenset(_copy_scalar(item, memo) for item in value)
		memo[id(value)] = copied
		return copied
	if type(value) is dict:
		copied = {}
		memo[id(value)] = copied
		copied.update({key: _copy_scalar(item, memo) for key, item in value.items()})
		return copied
	if isinstance(value, bytearray):
		copied = bytearray(value)
		memo[id(value)] = copied
		return copied
	return value


def transfer_namespace(
	old_namespace: dict[str, Any],
	*,
	old_browser_session: Any = None,
) -> NamespaceTransfer:
	"""Classify a namespace before a rebuild.

	Private names are internal. Framework bindings are rebuilt by the new
	namespace. Scalars and scalar-only containers are copied. Everything else
	is dropped with a reason, and a value that is or contains the previous
	browser session is named explicitly.
	"""
	result = NamespaceTransfer()
	memo: dict[int, Any] = {}
	for name, value in old_namespace.items():
		if not isinstance(name, str) or name.startswith('_'):
			continue
		if _is_framework_name(name):
			result.framework_names.append(name)
			continue
		kind, reason = _classify(value, old_browser_session, set())
		if kind is NamespaceValueKind.SCALAR:
			result.variables[name] = _copy_scalar(value, memo)
			continue
		reason = reason or 'unsupported value'
		result.dropped[name] = reason
		if len(result.warnings) < MAX_WARNINGS:
			result.warnings.append(
				f"Dropped namespace variable '{name}': {reason}. Reassign it after recovery."
			)
	return result
