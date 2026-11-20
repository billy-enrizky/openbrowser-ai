"""Explicit policy for carrying user variables across browser recovery."""

from __future__ import annotations

import hashlib
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

MAX_WARNINGS = 20

# Names the new namespace always provides. They are rebuilt, never carried.
ALWAYS_FRAMEWORK_NAMES = frozenset(
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
		'input_text',
		'close',
		'dropdown_options',
		'go_back',
		'switch',
		'upload_file',
		'wait',
	}
)

OPTIONAL_FRAMEWORK_NAMES = frozenset(
	{
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
	}
)

FRAMEWORK_NAMES = ALWAYS_FRAMEWORK_NAMES | OPTIONAL_FRAMEWORK_NAMES

_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes, bytearray)
_MAX_LABEL_LENGTH = 120
_MAX_DEPTH = 50


def _truncate_label(value: str, limit: int = _MAX_LABEL_LENGTH, *, unique: bool = False) -> str:
	if len(value) <= limit:
		return value
	suffix = '#' + hashlib.sha256(value.encode('utf-8')).hexdigest()[:6] if unique else ''
	return value[:limit] + '...' + suffix


class NamespaceValueKind(str, Enum):
	"""How one namespace value is treated during a transfer."""

	SCALAR = 'scalar'
	FRAMEWORK = 'framework'
	OPAQUE = 'opaque'


_Classification = tuple[NamespaceValueKind, str | None, int]


@dataclass
class NamespaceTransfer:
	"""Result of classifying one namespace before a browser rebuild."""

	variables: dict[str, Any] = field(default_factory=dict)
	framework_names: list[str] = field(default_factory=list)
	dropped: dict[str, str] = field(default_factory=dict)
	warnings: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, object]:
		"""Return a bounded, JSON-safe summary for status and reset responses."""
		preserved = [_truncate_label(name, unique=True) for name in sorted(self.variables)]
		framework = [_truncate_label(name, unique=True) for name in sorted(self.framework_names)]
		dropped_items = [
			(_truncate_label(name, unique=True), _truncate_label(reason, 200))
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


def _looks_like_framework_value(name: str, value: Any) -> bool:
	"""Whether a value is the kind of object this framework name binds."""
	if name == 'file_system' and value is None:
		return True
	if isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, type)):
		return True
	if type(value).__name__.startswith('_Missing'):
		return True
	if name in ('browser', 'file_system') and type(value) not in _SCALAR_TYPES:
		return True
	return False


def _is_framework_binding(name: str, value: Any) -> bool:
	"""A framework name is rebuilt only when it holds a framework value."""
	if name in ALWAYS_FRAMEWORK_NAMES or name in OPTIONAL_FRAMEWORK_NAMES:
		return _looks_like_framework_value(name, value)
	return False


def _classify(
	value: Any,
	old_browser: Any,
	active: set[int],
	cache: dict[int, _Classification],
	depth: int = 0,
) -> _Classification:
	"""Classify a value and return its kind, reason, and nesting height."""
	if type(value) in _SCALAR_TYPES:
		return NamespaceValueKind.SCALAR, None, 0
	if old_browser is not None and value is old_browser:
		return NamespaceValueKind.OPAQUE, 'references the previous browser session', 0
	if depth > _MAX_DEPTH:
		return NamespaceValueKind.OPAQUE, 'nested too deeply to transfer', 0

	key = id(value)
	cached = cache.get(key)
	if cached is not None:
		kind, reason, height = cached
		if kind is NamespaceValueKind.SCALAR and depth + height > _MAX_DEPTH:
			# The same container can be shallow in one place and too deep in
			# another, so the cached height must be checked at each use.
			return NamespaceValueKind.OPAQUE, 'nested too deeply to transfer', 0
		return cached

	if type(value) is dict:
		if key in active:
			return NamespaceValueKind.OPAQUE, 'cyclic container', 0
		active.add(key)
		try:
			height = 0
			for dict_key, item in value.items():
				if type(dict_key) not in _SCALAR_TYPES:
					result: _Classification = (
						NamespaceValueKind.OPAQUE,
						f'non-scalar dict key ({type(dict_key).__name__})',
						0,
					)
					cache[key] = result
					return result
				kind, reason, child_height = _classify(item, old_browser, active, cache, depth + 1)
				if kind is not NamespaceValueKind.SCALAR:
					result = (NamespaceValueKind.OPAQUE, reason or 'contains a non-scalar value', 0)
					cache[key] = result
					return result
				height = max(height, child_height)
			height += 1
			if depth + height > _MAX_DEPTH:
				result = (NamespaceValueKind.OPAQUE, 'nested too deeply to transfer', 0)
			else:
				result = (NamespaceValueKind.SCALAR, None, height)
			cache[key] = result
			return result
		finally:
			active.discard(key)

	if type(value) in (list, tuple, set, frozenset):
		if key in active:
			return NamespaceValueKind.OPAQUE, 'cyclic container', 0
		active.add(key)
		try:
			height = 0
			for item in value:
				kind, reason, child_height = _classify(item, old_browser, active, cache, depth + 1)
				if kind is not NamespaceValueKind.SCALAR:
					result = (NamespaceValueKind.OPAQUE, reason or f'contains a non-scalar value ({type(item).__name__})', 0)
					cache[key] = result
					return result
				height = max(height, child_height)
			height += 1
			if depth + height > _MAX_DEPTH:
				result = (NamespaceValueKind.OPAQUE, 'nested too deeply to transfer', 0)
			else:
				result = (NamespaceValueKind.SCALAR, None, height)
			cache[key] = result
			return result
		finally:
			active.discard(key)

	cache[key] = (NamespaceValueKind.OPAQUE, f'unsupported type {type(value).__name__}', 0)
	return cache[key]


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
	cache: dict[int, _Classification] = {}
	for name, value in old_namespace.items():
		if not isinstance(name, str) or name.startswith('_'):
			continue
		if name in FRAMEWORK_NAMES:
			if _is_framework_binding(name, value):
				result.framework_names.append(name)
				continue
			if name in ALWAYS_FRAMEWORK_NAMES:
				reason = 'shadows framework binding'
				result.dropped[name] = reason
				if len(result.warnings) < MAX_WARNINGS:
					result.warnings.append(
						f"Dropped namespace variable '{name}': {reason}. Reassign it after recovery."
					)
				continue
			# An optional binding that is not present in this process is a
			# user value under a reserved name. Fall through to classification.
		try:
			kind, reason, _height = _classify(value, old_browser_session, set(), cache)
			if kind is NamespaceValueKind.SCALAR:
				result.variables[name] = _copy_scalar(value, memo)
				continue
		except RecursionError:
			reason = 'nested too deeply to transfer'
		reason = reason or 'unsupported value'
		result.dropped[name] = reason
		if len(result.warnings) < MAX_WARNINGS:
			result.warnings.append(
				f"Dropped namespace variable '{name}': {reason}. Reassign it after recovery."
			)
	return result
