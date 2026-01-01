"""Browser module for low-level CDP logic."""

from typing import TYPE_CHECKING

# Type stubs for lazy imports
if TYPE_CHECKING:
    from src.openbrowser.browser.profile import BrowserProfile, ProxySettings
    from src.openbrowser.browser.session import BrowserSession

# Lazy imports mapping for heavy browser components
_LAZY_IMPORTS = {
    'ProxySettings': ('src.openbrowser.browser.profile', 'ProxySettings'),
    'BrowserProfile': ('src.openbrowser.browser.profile', 'BrowserProfile'),
    'BrowserSession': ('src.openbrowser.browser.session', 'BrowserSession'),
}


def __getattr__(name: str):
    """Lazy import mechanism for heavy browser components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module
            module = import_module(module_path)
            attr = getattr(module, attr_name)
            # Cache the imported attribute in the module's globals
            globals()[name] = attr
            return attr
        except ImportError as e:
            raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'BrowserSession',
    'BrowserProfile',
    'ProxySettings',
]

