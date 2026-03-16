"""Coverage tests for openbrowser.code_use.__init__.py lazy imports.

Covers lines 27-32: __getattr__ lazy import mechanism for CodeAgent,
create_namespace, export_to_ipynb, session_to_python_script, CodeCell,
ExecutionStatus, NotebookSession.
"""

import logging
from unittest.mock import patch, MagicMock

import pytest

logger = logging.getLogger(__name__)


class TestCodeUseLazyImports:
    """Tests for __getattr__ lazy import in code_use/__init__.py."""

    def test_import_code_agent(self):
        """Accessing CodeAgent triggers lazy import."""
        from openbrowser.code_use import CodeAgent
        assert CodeAgent is not None

    def test_import_create_namespace(self):
        """Accessing create_namespace triggers lazy import."""
        from openbrowser.code_use import create_namespace
        assert create_namespace is not None
        assert callable(create_namespace)

    def test_import_code_cell(self):
        """Accessing CodeCell triggers lazy import."""
        from openbrowser.code_use import CodeCell
        assert CodeCell is not None

    def test_import_execution_status(self):
        """Accessing ExecutionStatus triggers lazy import."""
        from openbrowser.code_use import ExecutionStatus
        assert ExecutionStatus is not None
        assert hasattr(ExecutionStatus, "SUCCESS")

    def test_import_notebook_session(self):
        """Accessing NotebookSession triggers lazy import."""
        from openbrowser.code_use import NotebookSession
        assert NotebookSession is not None

    def test_import_export_to_ipynb(self):
        """Accessing export_to_ipynb triggers lazy import."""
        from openbrowser.code_use import export_to_ipynb
        assert export_to_ipynb is not None
        assert callable(export_to_ipynb)

    def test_import_session_to_python_script(self):
        """Accessing session_to_python_script triggers lazy import."""
        from openbrowser.code_use import session_to_python_script
        assert session_to_python_script is not None
        assert callable(session_to_python_script)

    def test_unknown_attribute_raises(self):
        """Accessing unknown attribute raises AttributeError."""
        import openbrowser.code_use as code_use_mod
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = code_use_mod.nonexistent_thing

    def test_all_list_matches_lazy_imports(self):
        """__all__ should list the same names as _LAZY_IMPORTS."""
        import openbrowser.code_use as code_use_mod
        all_set = set(code_use_mod.__all__)
        lazy_set = set(code_use_mod._LAZY_IMPORTS.keys())
        assert all_set == lazy_set, (
            f"__all__ and _LAZY_IMPORTS are out of sync: "
            f"in __all__ only: {all_set - lazy_set}, "
            f"in _LAZY_IMPORTS only: {lazy_set - all_set}"
        )
        for name in code_use_mod.__all__:
            # Each name should be importable
            obj = getattr(code_use_mod, name)
            assert obj is not None
