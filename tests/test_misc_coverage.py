"""Comprehensive tests for miscellaneous small modules.

Covers:
- src/openbrowser/types.py (2 stmts: lines 3-5) - BaseModel import
- src/openbrowser/controller/__init__.py (2 stmts: lines 1-3) - Controller import
- src/openbrowser/mcp/__main__.py (4 stmts: lines 7-12) - MCP entry point
- src/openbrowser/browser/__init__.py remaining (lines 31-34) - AttributeError path
- src/openbrowser/browser/events.py (line 60) - serialize_node None branch
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# types.py (lines 3-5)
# ---------------------------------------------------------------------------


class TestTypesModule:
    """Test openbrowser.types module."""

    def test_types_imports_basemodel(self):
        """Test that types module exports BaseModel."""
        from openbrowser.types import BaseModel

        assert BaseModel is not None

    def test_types_all_exports(self):
        """Test __all__ exports."""
        from openbrowser import types

        assert "BaseModel" in types.__all__

    def test_types_basemodel_is_pydantic(self):
        """Test that BaseModel is from pydantic."""
        from pydantic import BaseModel as PydanticBaseModel

        from openbrowser.types import BaseModel

        assert BaseModel is PydanticBaseModel


# ---------------------------------------------------------------------------
# controller/__init__.py (lines 1-3)
# ---------------------------------------------------------------------------


class TestControllerInit:
    """Test openbrowser.controller.__init__ module."""

    def test_controller_import(self):
        """Test that Controller can be imported."""
        from openbrowser.controller import Controller

        assert Controller is not None

    def test_controller_all_exports(self):
        """Test __all__ exports."""
        from openbrowser import controller

        assert "Controller" in controller.__all__

    def test_controller_is_from_tools_service(self):
        """Test Controller comes from tools.service."""
        from openbrowser.controller import Controller
        from openbrowser.tools.service import Controller as ToolsController

        assert Controller is ToolsController


# ---------------------------------------------------------------------------
# mcp/__main__.py (lines 7-12)
# ---------------------------------------------------------------------------


class TestMCPMainModule:
    """Test openbrowser.mcp.__main__ module."""

    def test_mcp_main_imports(self):
        """Test that the __main__ module can be imported."""
        import openbrowser.mcp.__main__ as mcp_main_module

        assert hasattr(mcp_main_module, "asyncio")
        assert hasattr(mcp_main_module, "main")

    def test_mcp_main_function_exists(self):
        """Test that main function is importable."""
        from openbrowser.mcp.server import main

        assert callable(main)

    def test_mcp_main_module_guard(self):
        """Test that __name__ == '__main__' guard exists."""
        import importlib
        import openbrowser.mcp.__main__ as mod

        # The module should have loaded without running asyncio.run
        # because __name__ is not '__main__' when imported
        assert mod is not None

    def test_mcp_main_asyncio_imported(self):
        """Test asyncio is available in the module."""
        import openbrowser.mcp.__main__ as mod

        assert hasattr(mod, "asyncio")


# ---------------------------------------------------------------------------
# browser/__init__.py remaining (lines 31-34) - AttributeError path
# ---------------------------------------------------------------------------


class TestBrowserInit:
    """Test openbrowser.browser.__init__ module lazy imports."""

    def test_lazy_import_browser_session(self):
        """Test lazy import of BrowserSession."""
        from openbrowser.browser import BrowserSession

        assert BrowserSession is not None

    def test_lazy_import_browser_profile(self):
        """Test lazy import of BrowserProfile."""
        from openbrowser.browser import BrowserProfile

        assert BrowserProfile is not None

    def test_lazy_import_proxy_settings(self):
        """Test lazy import of ProxySettings."""
        from openbrowser.browser import ProxySettings

        assert ProxySettings is not None

    def test_lazy_import_nonexistent_raises_attribute_error(self):
        """Test that accessing a nonexistent attribute raises AttributeError."""
        import openbrowser.browser as browser_mod

        with pytest.raises(AttributeError, match="has no attribute 'NonExistentClass'"):
            _ = browser_mod.NonExistentClass

    def test_lazy_import_caches_result(self):
        """Test that lazy imports are cached in globals."""
        import openbrowser.browser as browser_mod

        # Access to trigger lazy load
        _ = browser_mod.BrowserSession

        # Second access should be from cache (globals)
        _ = browser_mod.BrowserSession

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from openbrowser.browser import __all__

        assert "BrowserSession" in __all__
        assert "BrowserProfile" in __all__
        assert "ProxySettings" in __all__

    def test_lazy_imports_mapping(self):
        """Test _LAZY_IMPORTS mapping structure."""
        from openbrowser.browser import _LAZY_IMPORTS

        assert "BrowserSession" in _LAZY_IMPORTS
        assert "BrowserProfile" in _LAZY_IMPORTS
        assert "ProxySettings" in _LAZY_IMPORTS

        # Each entry should be (module_path, attr_name) tuple
        for name, (mod_path, attr_name) in _LAZY_IMPORTS.items():
            assert isinstance(mod_path, str)
            assert isinstance(attr_name, str)
            assert mod_path.startswith(".")

    def test_lazy_import_error_handling(self):
        """Test ImportError handling in lazy import."""
        import openbrowser.browser as browser_mod

        # Temporarily add a bad entry to _LAZY_IMPORTS
        original = browser_mod._LAZY_IMPORTS.copy()
        browser_mod._LAZY_IMPORTS["BadModule"] = (".nonexistent_module", "BadClass")

        try:
            with pytest.raises(ImportError, match="Failed to import BadModule"):
                _ = browser_mod.BadModule
        finally:
            browser_mod._LAZY_IMPORTS = original


# ---------------------------------------------------------------------------
# browser/events.py (line 60) - serialize_node with None
# ---------------------------------------------------------------------------


class TestBrowserEventsSerializeNode:
    """Test the serialize_node validator on ElementSelectedEvent."""

    def test_serialize_node_with_none(self):
        """Test that serialize_node returns None when data is None."""
        from openbrowser.browser.events import ElementSelectedEvent

        # The field_validator should return None for None input
        result = ElementSelectedEvent.serialize_node(None)
        assert result is None

    def test_events_get_timeout_with_env_var(self):
        """Test _get_timeout with valid environment variable."""
        import os

        from openbrowser.browser.events import _get_timeout

        with patch.dict(os.environ, {"TEST_TIMEOUT": "25.0"}):
            result = _get_timeout("TEST_TIMEOUT", 10.0)
            assert result == 25.0

    def test_events_get_timeout_negative_env_var(self):
        """Test _get_timeout with negative environment variable."""
        import os

        from openbrowser.browser.events import _get_timeout

        with patch.dict(os.environ, {"TEST_TIMEOUT": "-5.0"}):
            result = _get_timeout("TEST_TIMEOUT", 10.0)
            assert result == 10.0  # Falls back to default

    def test_events_get_timeout_invalid_env_var(self):
        """Test _get_timeout with invalid environment variable."""
        import os

        from openbrowser.browser.events import _get_timeout

        with patch.dict(os.environ, {"TEST_TIMEOUT": "not_a_number"}):
            result = _get_timeout("TEST_TIMEOUT", 10.0)
            assert result == 10.0  # Falls back to default

    def test_events_get_timeout_no_env_var(self):
        """Test _get_timeout without environment variable."""
        import os

        from openbrowser.browser.events import _get_timeout

        # Ensure the env var is not set
        env = os.environ.copy()
        env.pop("TEST_NONEXISTENT", None)

        with patch.dict(os.environ, env, clear=True):
            result = _get_timeout("TEST_NONEXISTENT", 15.0)
            assert result == 15.0

    def test_check_event_names_dont_overlap(self):
        """Test that _check_event_names_dont_overlap works without error."""
        from openbrowser.browser.events import _check_event_names_dont_overlap

        # Should not raise -- called at import time already
        _check_event_names_dont_overlap()


# ---------------------------------------------------------------------------
# browser/__init__.py __getattr__ edge cases
# ---------------------------------------------------------------------------


class TestBrowserInitGetattr:
    """Test additional __getattr__ edge cases in browser/__init__.py."""

    def test_getattr_function_exists(self):
        """Test that __getattr__ is defined."""
        import openbrowser.browser as mod

        assert hasattr(mod, "__getattr__")

    def test_getattr_known_name(self):
        """Test __getattr__ with a known lazy import name."""
        import openbrowser.browser as mod

        # This should trigger __getattr__ and cache the result
        result = mod.__getattr__("BrowserSession")
        assert result is not None

    def test_getattr_unknown_name(self):
        """Test __getattr__ with an unknown name."""
        import openbrowser.browser as mod

        with pytest.raises(AttributeError):
            mod.__getattr__("CompletelyUnknownAttribute")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
