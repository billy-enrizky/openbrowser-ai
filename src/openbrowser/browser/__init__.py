from typing import TYPE_CHECKING

# Type stubs for lazy imports
if TYPE_CHECKING:
	from .lifecycle import LifecycleMachine, LifecyclePhase, LifecycleSnapshot, LifecycleTransitionError
	from .profile import BrowserProfile, ProxySettings
	from .session import BrowserSession
	from .tabs import TabError, TabRecord, TabRegistry, TabState


# Lazy imports mapping for heavy browser components
_LAZY_IMPORTS = {
	'ProxySettings': ('.profile', 'ProxySettings'),
	'BrowserProfile': ('.profile', 'BrowserProfile'),
	'BrowserSession': ('.session', 'BrowserSession'),
	'LifecycleMachine': ('.lifecycle', 'LifecycleMachine'),
	'LifecyclePhase': ('.lifecycle', 'LifecyclePhase'),
	'LifecycleSnapshot': ('.lifecycle', 'LifecycleSnapshot'),
	'LifecycleTransitionError': ('.lifecycle', 'LifecycleTransitionError'),
	'TabError': ('.tabs', 'TabError'),
	'TabRecord': ('.tabs', 'TabRecord'),
	'TabRegistry': ('.tabs', 'TabRegistry'),
	'TabState': ('.tabs', 'TabState'),
}


def __getattr__(name: str):
	"""Lazy import mechanism for heavy browser components."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			# Use relative import for current package
			full_module_path = f'openbrowser.browser{module_path}'
			module = import_module(full_module_path)
			attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {full_module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'BrowserSession',
	'BrowserProfile',
	'ProxySettings',
	'LifecycleMachine',
	'LifecyclePhase',
	'LifecycleSnapshot',
	'LifecycleTransitionError',
	'TabError',
	'TabRecord',
	'TabRegistry',
	'TabState',
]
