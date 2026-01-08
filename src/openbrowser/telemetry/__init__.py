"""Telemetry for OpenBrowser.

Performance optimizations:
- Added import cache for lazy imports
- Optimized __getattr__ with cache lookup as fast path
"""

from typing import TYPE_CHECKING

# Type stubs for lazy imports
if TYPE_CHECKING:
	from openbrowser.telemetry.service import ProductTelemetry
	from openbrowser.telemetry.views import (
		BaseTelemetryEvent,
		CLITelemetryEvent,
		MCPClientTelemetryEvent,
		MCPServerTelemetryEvent,
	)

# Lazy imports mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
	'ProductTelemetry': ('openbrowser.telemetry.service', 'ProductTelemetry'),
	'BaseTelemetryEvent': ('openbrowser.telemetry.views', 'BaseTelemetryEvent'),
	'CLITelemetryEvent': ('openbrowser.telemetry.views', 'CLITelemetryEvent'),
	'MCPClientTelemetryEvent': ('openbrowser.telemetry.views', 'MCPClientTelemetryEvent'),
	'MCPServerTelemetryEvent': ('openbrowser.telemetry.views', 'MCPServerTelemetryEvent'),
}

# Cache for imported modules/attributes
_import_cache: dict[str, object] = {}


def __getattr__(name: str):
	"""Lazy import mechanism for telemetry components.
	
	Optimized with caching to avoid repeated imports.
	"""
	# Fast path: check cache first
	if name in _import_cache:
		return _import_cache[name]
	
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			attr = getattr(module, attr_name)
			# Cache the imported attribute
			_import_cache[name] = attr
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'BaseTelemetryEvent',
	'ProductTelemetry',
	'CLITelemetryEvent',
	'MCPClientTelemetryEvent',
	'MCPServerTelemetryEvent',
]
