"""Comprehensive tests for openbrowser.__init__ module.

Covers: lazy imports, _patched_del, __getattr__, import caching, __all__.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

import openbrowser
from openbrowser import __all__, _LAZY_IMPORTS, _import_cache, _patched_del

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _patched_del
# ---------------------------------------------------------------------------


class TestPatchedDel:
    def test_closed_loop_returns_early(self):
        """When event loop is closed, _patched_del should not crash."""
        mock_transport = MagicMock()
        mock_transport._loop = MagicMock()
        mock_transport._loop.is_closed.return_value = True

        # Should not raise
        _patched_del(mock_transport)

    def test_no_loop_attribute(self):
        """When transport has no _loop, should attempt original __del__."""
        mock_transport = MagicMock(spec=[])

        # Mock _original_del to avoid calling real __del__ on a mock without _closed
        with patch("openbrowser._original_del") as mock_orig_del:
            _patched_del(mock_transport)
            # Without _loop, it falls through to _original_del
            mock_orig_del.assert_called_once_with(mock_transport)

    def test_runtime_error_event_loop_closed_silenced(self):
        """RuntimeError with 'Event loop is closed' should be silenced."""
        mock_transport = MagicMock()
        mock_transport._loop = MagicMock()
        mock_transport._loop.is_closed.return_value = False

        with patch("openbrowser._original_del", side_effect=RuntimeError("Event loop is closed")):
            # Should not raise
            _patched_del(mock_transport)

    def test_other_runtime_error_reraised(self):
        """Other RuntimeErrors should be re-raised."""
        mock_transport = MagicMock()
        mock_transport._loop = MagicMock()
        mock_transport._loop.is_closed.return_value = False

        with patch("openbrowser._original_del", side_effect=RuntimeError("Something else")):
            with pytest.raises(RuntimeError, match="Something else"):
                _patched_del(mock_transport)


# ---------------------------------------------------------------------------
# __getattr__ and lazy imports
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_lazy_imports_mapping_exists(self):
        assert isinstance(_LAZY_IMPORTS, dict)
        assert len(_LAZY_IMPORTS) > 0

    def test_import_cache_is_dict(self):
        assert isinstance(_import_cache, dict)

    def test_all_exports_defined(self):
        assert isinstance(__all__, list)
        assert "BrowserProfile" in __all__
        assert "BrowserSession" in __all__
        assert "Agent" in __all__

    def test_getattr_raises_for_unknown(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            openbrowser.__getattr__("NONEXISTENT_ATTR_XYZ_12345")

    def test_lazy_import_browser_profile(self):
        """Test that BrowserProfile can be lazy-imported."""
        bp = openbrowser.BrowserProfile
        assert bp is not None
        # Should be cached now
        assert "BrowserProfile" in _import_cache

    def test_lazy_import_browser_session(self):
        bs = openbrowser.BrowserSession
        assert bs is not None

    def test_lazy_import_action_result(self):
        ar = openbrowser.ActionResult
        assert ar is not None

    def test_lazy_import_tools(self):
        t = openbrowser.Tools
        assert t is not None

    def test_lazy_import_dom_service(self):
        ds = openbrowser.DomService
        assert ds is not None

    def test_browser_alias(self):
        """Browser should be an alias for BrowserSession."""
        browser = openbrowser.Browser
        browser_session = openbrowser.BrowserSession
        assert browser is browser_session

    def test_browser_agent_alias(self):
        """BrowserAgent should be an alias for Agent."""
        agent = openbrowser.Agent
        browser_agent = openbrowser.BrowserAgent
        assert agent is browser_agent

    def test_controller_alias(self):
        """Controller should be an alias for Tools."""
        # Controller is listed as alias for Tools in _LAZY_IMPORTS
        controller = openbrowser.Controller
        assert controller is not None

    def test_models_import(self):
        """models should return the module itself."""
        models = openbrowser.models
        assert models is not None
        assert hasattr(models, "__name__")

    def test_cache_hit_returns_same_object(self):
        """Accessing the same attribute twice should return cached object."""
        obj1 = openbrowser.BrowserProfile
        obj2 = openbrowser.BrowserProfile
        assert obj1 is obj2

    def test_system_prompt_lazy_import(self):
        sp = openbrowser.SystemPrompt
        assert sp is not None


# ---------------------------------------------------------------------------
# Module-level attributes
# ---------------------------------------------------------------------------


class TestModuleAttributes:
    def test_logger_exists(self):
        """openbrowser should have a logger attribute."""
        assert hasattr(openbrowser, "logger")
        assert openbrowser.logger.name == "openbrowser"
