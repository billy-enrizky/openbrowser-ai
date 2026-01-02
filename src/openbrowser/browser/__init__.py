"""Browser module for low-level CDP (Chrome DevTools Protocol) logic.

This module provides the core browser automation functionality using CDP for direct
communication with Chromium-based browsers. It implements an event-driven architecture
following the browser-use pattern for managing browser sessions, profiles, and state.

Key Components:
    BrowserSession: Main class for managing browser lifecycle and CDP connections.
    BrowserProfile: Configuration for browser launch parameters and settings.
    ProxySettings: Typed configuration for HTTP/SOCKS proxy settings.

The module uses lazy imports to avoid loading heavy browser components until needed.

Example:
    >>> from src.openbrowser.browser import BrowserSession, BrowserProfile
    >>> profile = BrowserProfile(headless=True)
    >>> session = BrowserSession(browser_profile=profile)
    >>> await session.start()
    >>> await session.navigate_to("https://example.com")
    >>> await session.stop()
"""

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
    """Lazy import mechanism for heavy browser components.

    This function enables lazy loading of browser module components to improve
    import performance. Components are only loaded when first accessed.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The requested module attribute (class or function).

    Raises:
        ImportError: If the requested component cannot be imported.
        AttributeError: If the requested name is not a valid module attribute.
    """
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

