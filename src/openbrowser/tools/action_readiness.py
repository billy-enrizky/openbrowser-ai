"""Explicit readiness rules for browser element indexes.

Element indexes come from the latest serialized browser state. Using one
before state exists, or after the page changed, is a readiness failure, not
an action failure. These rules give the agent one typed error with a refresh
hint instead of a raw schema error or a success-shaped message.

The accepted index forms are the same numeric forms the Pydantic integer
field accepted before this module existed: integers, integral floats, and
numeric strings. Boolean values are rejected explicitly because ``True``
silently became index 1.

Readiness proves that an index belongs to the last serialized browser state.
It cannot prove that the live DOM still matches that snapshot. JavaScript
that changes the DOM without a state refresh, and navigations that have not
completed, are outside what this check can see. Re-read browser state after
either one.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import TypeAdapter, ValidationError

REFRESH_HINT = (
	'Refresh with `state = await browser.get_browser_state_summary()` '
	'(in MCP, run it inside execute_code; get_status does not refresh indexes), '
	'then retry with an [i_N] index from the current state.'
)
_AVAILABLE_SAMPLE = 10
_INDEX_LABEL = re.compile(r'^\[?i_(\d+)\]?$')
_INT_INDEX = TypeAdapter(int)


class ActionReadinessCode(str, Enum):
	"""Why an element index cannot be used."""

	MISSING = 'element_index_missing'
	INVALID = 'element_index_invalid'
	STATE_NOT_READY = 'browser_state_not_ready'
	STALE = 'element_index_stale'


class ActionReadinessError(ValueError):
	"""A typed action-readiness failure with a refresh hint."""

	def __init__(
		self,
		code: ActionReadinessCode,
		message: str,
		*,
		action: str | None = None,
		hint: str = REFRESH_HINT,
	):
		self.code = code
		self.action = action
		self.detail = message
		self.hint = hint
		super().__init__(self._format())

	def _format(self) -> str:
		prefix = f'{self.action}() ' if self.action else ''
		return f'Action readiness [{self.code.value}]: {prefix}{self.detail} {self.hint}'

	def with_action(self, action: str) -> 'ActionReadinessError':
		"""Return the same failure named by the public callable."""
		return ActionReadinessError(self.code, self.detail, action=action, hint=self.hint)


def _bounded_repr(value: Any, limit: int = 120) -> str:
	"""Render a value safely, even beyond CPython's integer digit limit."""
	try:
		text = repr(value)
	except Exception:
		return f'<unrepresentable {type(value).__name__}>'
	if len(text) <= limit:
		return text
	return f'{text[:limit]}... ({len(text)} chars)'


def _invalid(index: Any, detail: str, *, action: str | None) -> ActionReadinessError:
	return ActionReadinessError(
		ActionReadinessCode.INVALID,
		f'{detail} Got {type(index).__name__} ({_bounded_repr(index)}).',
		action=action,
	)


def _coerce_index_value(index: Any, *, action: str | None) -> int:
	if isinstance(index, bool):
		raise _invalid(index, 'index must be an integer, not a boolean.', action=action)
	if isinstance(index, int):
		return index
	if isinstance(index, str):
		label = _INDEX_LABEL.match(index.strip())
		if label:
			digits = label.group(1)
			target = digits if len(digits) <= 12 else 'the bare integer'
			raise _invalid(
				index,
				f'index is a browser-state label. Pass {target} without the "i_" prefix or brackets.',
				action=action,
			)
	# Delegate every other type to the same coercion the Pydantic field uses,
	# so the guard and the model cannot disagree about which values are valid.
	try:
		return _INT_INDEX.validate_python(index)
	except ValidationError:
		if isinstance(index, float) and not index.is_integer():
			raise _invalid(index, 'index must be a whole number.', action=action) from None
		raise _invalid(index, 'index must be an integer.', action=action) from None


def normalize_optional_index(index: Any, *, action: str | None = None, sentinel: int = 0) -> int | None:
	"""Normalize an optional index, preserving the action's sentinel value."""
	if index is None:
		return None
	if isinstance(index, bool):
		return validate_index_value(index, action=action)
	try:
		number = _INT_INDEX.validate_python(index)
	except ValidationError:
		return validate_index_value(index, action=action)
	if number == sentinel:
		return sentinel
	return validate_index_value(number, action=action)


def validate_index_value(index: Any, *, action: str | None = None) -> int:
	"""Validate an index value without consulting browser state.

	This is the model-level and namespace-level guard for direct calls.
	"""
	if index is None:
		raise ActionReadinessError(
			ActionReadinessCode.MISSING,
			'index is required, but None was passed.',
			action=action,
		)
	value = _coerce_index_value(index, action=action)
	if value < 1:
		raise ActionReadinessError(
			ActionReadinessCode.INVALID,
			f'element indexes are 1-based; got {_bounded_repr(value)}.',
			action=action,
		)
	return value


def available_index_summary(selector_map: dict[int, Any]) -> str:
	"""Return a bounded, sorted sample of usable indexes."""
	indexes = sorted(selector_map)
	if not indexes:
		return 'none'
	sample = indexes[:_AVAILABLE_SAMPLE]
	text = ', '.join(_bounded_repr(index) for index in sample)
	if len(indexes) > _AVAILABLE_SAMPLE:
		text += f' (showing {_AVAILABLE_SAMPLE} of {len(indexes)})'
	return text


def require_element_index(
	index: Any,
	*,
	selector_map: Any,
	action: str | None = None,
) -> int:
	"""Require an index that exists in the current serialized browser state."""
	index = validate_index_value(index, action=action)
	if not isinstance(selector_map, dict) or not selector_map:
		raise ActionReadinessError(
			ActionReadinessCode.STATE_NOT_READY,
			'browser state has no element indexes; the page may still be loading or may have no interactive elements.',
			action=action,
		)
	if index not in selector_map:
		available = available_index_summary(selector_map)
		raise ActionReadinessError(
			ActionReadinessCode.STALE,
			f'index {_bounded_repr(index)} is not in the current browser state. Available indexes: {available}.',
			action=action,
		)
	return index


def readiness_error_from_validation_error(error: ValidationError) -> ActionReadinessError | None:
	"""Recover the typed readiness failure that Pydantic wrapped.

	Pydantic converts validator exceptions into ``ValidationError`` while
	keeping the original exception in ``ctx['error']``.
	"""
	for details in error.errors():
		cause = (details.get('ctx') or {}).get('error')
		if isinstance(cause, ActionReadinessError):
			return cause
	return None
