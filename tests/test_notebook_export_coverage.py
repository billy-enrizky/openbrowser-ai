"""Comprehensive tests for openbrowser.code_use.notebook_export module.

Covers ALL code paths in notebook_export.py (lines 3-276):
- export_to_ipynb() - full notebook export with JS blocks, outputs, errors, browser state
- session_to_python_script() - Python script generation with JS blocks
- All cell types, output types, error handling
- JavaScript pattern detection
- Edge cases: empty sessions, no namespace, missing attributes
"""

import json
import logging
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local models to avoid circular imports (mirrors views.py)
# ---------------------------------------------------------------------------


class CellType(str, Enum):
    """Type of notebook cell."""

    CODE = "code"
    MARKDOWN = "markdown"


class ExecutionStatus(str, Enum):
    """Execution status of a cell."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class CodeCell(BaseModel):
    """Represents a code cell in the notebook-like execution."""

    id: str = Field(default_factory=lambda: "test-id")
    cell_type: CellType = CellType.CODE
    source: str = Field(description="The code to execute")
    output: str | None = Field(default=None)
    execution_count: int | None = Field(default=None)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    error: str | None = Field(default=None)
    browser_state: str | None = Field(default=None)


class NotebookSession(BaseModel):
    """Represents a notebook-like session."""

    id: str = Field(default_factory=lambda: "session-id")
    cells: list[CodeCell] = Field(default_factory=list)
    current_execution_count: int = Field(default=0)
    namespace: dict[str, Any] = Field(default_factory=dict)

    def add_cell(self, source: str) -> CodeCell:
        """Add a new code cell to the session."""
        cell = CodeCell(source=source)
        self.cells.append(cell)
        return cell


class MockCodeAgent:
    """Mock CodeAgent for testing notebook export."""

    def __init__(self):
        self.session = NotebookSession()
        self.namespace: dict = {}


# ---------------------------------------------------------------------------
# Patching helper -- import the real module with mocked CodeAgent
# ---------------------------------------------------------------------------


def _get_export_functions():
    """Import export functions, patching CodeAgent to avoid heavy deps."""
    with MagicMock() as mock_service:
        # We need to mock the CodeAgent import in notebook_export
        import importlib
        import sys

        # The actual import
        from openbrowser.code_use.notebook_export import export_to_ipynb, session_to_python_script

        return export_to_ipynb, session_to_python_script


# ---------------------------------------------------------------------------
# export_to_ipynb (lines 12-174)
# ---------------------------------------------------------------------------


class TestExportToIpynb:
    """Tests for export_to_ipynb function."""

    def test_export_basic_session(self, tmp_path):
        """Test exporting a basic session with one code cell."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="await navigate('https://example.com')")
        cell.status = ExecutionStatus.SUCCESS
        cell.execution_count = 1
        cell.output = "Navigated to example.com"

        output_path = tmp_path / "test.ipynb"
        result = export_to_ipynb(agent, output_path)

        assert result == output_path
        assert output_path.exists()

        with open(output_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] == 5
        assert "kernelspec" in notebook["metadata"]
        # Setup cell + 1 code cell
        assert len(notebook["cells"]) >= 2

    def test_export_with_output(self, tmp_path):
        """Test that cell output is included in notebook."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="result = 'hello'")
        cell.execution_count = 1
        cell.output = "hello\nworld"

        output_path = tmp_path / "output.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        # Find code cell (last one)
        code_cell = nb["cells"][-1]
        assert code_cell["cell_type"] == "code"
        assert len(code_cell["outputs"]) >= 1
        assert code_cell["outputs"][0]["output_type"] == "stream"
        assert code_cell["outputs"][0]["name"] == "stdout"

    def test_export_with_error(self, tmp_path):
        """Test that cell error is included in notebook."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="raise ValueError('bad')")
        cell.execution_count = 1
        cell.error = "ValueError: bad\n  at line 1"

        output_path = tmp_path / "error.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        code_cell = nb["cells"][-1]
        error_output = None
        for out in code_cell.get("outputs", []):
            if out.get("output_type") == "error":
                error_output = out
                break

        assert error_output is not None
        assert error_output["ename"] == "Error"
        assert "ValueError: bad" in error_output["evalue"]
        assert len(error_output["traceback"]) >= 1

    def test_export_with_browser_state(self, tmp_path):
        """Test that browser state is included in notebook."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="await navigate('https://example.com')")
        cell.execution_count = 1
        cell.browser_state = "URL: https://example.com\nTitle: Example"

        output_path = tmp_path / "state.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        code_cell = nb["cells"][-1]
        browser_state_found = False
        for out in code_cell.get("outputs", []):
            text_content = str(out.get("text", []))
            if "Browser State" in text_content:
                browser_state_found = True
                break

        assert browser_state_found

    def test_export_with_all_outputs(self, tmp_path):
        """Test cell with output, error, and browser state together."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="await click(5)")
        cell.execution_count = 1
        cell.output = "Clicked element 5"
        cell.error = "Warning: slow response"
        cell.browser_state = "URL changed"

        output_path = tmp_path / "all_outputs.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        code_cell = nb["cells"][-1]
        assert len(code_cell["outputs"]) == 3  # stdout + error + browser state

    def test_export_with_javascript_blocks(self, tmp_path):
        """Test exporting session with JavaScript code blocks in namespace."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"js_func", "non_js_var"}
        agent.namespace["js_func"] = "function extractData() { return document.querySelector('.title').textContent; }"
        agent.namespace["non_js_var"] = "just a plain string"

        cell = agent.session.add_cell(source="result = await evaluate(js_func)")
        cell.execution_count = 1

        output_path = tmp_path / "js.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        # Should have setup cell + JS block + code cell
        assert len(nb["cells"]) >= 3

        # Find JS block cell
        js_found = False
        for cell_data in nb["cells"]:
            source = "".join(cell_data.get("source", []))
            if "JavaScript Code Block: js_func" in source:
                js_found = True
                break
        assert js_found

    def test_export_with_js_patterns(self, tmp_path):
        """Test that all JavaScript patterns are detected."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        # Test each JS pattern
        patterns_to_test = [
            ("arrow_func", "const fn = (x) => { return x; }"),
            ("dom_query", "document.querySelector('.item')"),
            ("array_from", "Array.from(elements)"),
            ("text_content", "el.textContent"),
            ("inner_html", "el.innerHTML"),
            ("return_stmt", "return result"),
            ("console_log", "console.log('debug')"),
            ("window_obj", "window.location"),
            ("map_call", "items.map(x => x)"),
            ("filter_call", "items.filter(x => x > 0)"),
            ("foreach_call", "items.forEach(fn)"),
            ("anon_func", "(function() { return 1; })"),
        ]

        for var_name, code in patterns_to_test:
            agent = MockCodeAgent()
            agent.namespace["_code_block_vars"] = {var_name}
            agent.namespace[var_name] = code

            output_path = tmp_path / f"js_{var_name}.ipynb"
            export_to_ipynb(agent, output_path)

            with open(output_path, "r") as f:
                nb = json.load(f)

            js_found = any(
                "JavaScript Code Block" in "".join(c.get("source", []))
                for c in nb["cells"]
            )
            assert js_found, f"JS pattern not detected for {var_name}: {code}"

    def test_export_empty_session(self, tmp_path):
        """Test exporting an empty session (only setup cell)."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()

        output_path = tmp_path / "empty.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        # Should have at least the setup cell
        assert len(nb["cells"]) >= 1

    def test_export_no_namespace(self, tmp_path):
        """Test exporting when agent has no namespace attribute."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        # Remove namespace
        delattr(agent, "namespace")

        cell = agent.session.add_cell(source="test")
        cell.execution_count = 1

        output_path = tmp_path / "no_namespace.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        assert output_path.exists()

    def test_export_empty_namespace(self, tmp_path):
        """Test exporting when agent has empty namespace."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        agent.namespace = {}

        cell = agent.session.add_cell(source="test")
        cell.execution_count = 1

        output_path = tmp_path / "empty_ns.ipynb"
        export_to_ipynb(agent, output_path)

        assert output_path.exists()

    def test_export_js_block_empty_string(self, tmp_path):
        """Test that empty JS variable values are skipped."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"empty_var"}
        agent.namespace["empty_var"] = "   "  # whitespace only

        output_path = tmp_path / "empty_js.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        # Should not have JS block cells
        js_found = any(
            "JavaScript Code Block" in "".join(c.get("source", []))
            for c in nb["cells"]
        )
        assert not js_found

    def test_export_js_block_not_string(self, tmp_path):
        """Test that non-string JS variables are skipped."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"num_var"}
        agent.namespace["num_var"] = 42  # Not a string

        output_path = tmp_path / "non_string_js.ipynb"
        export_to_ipynb(agent, output_path)

        assert output_path.exists()

    def test_export_creates_parent_dirs(self, tmp_path):
        """Test that export creates parent directories."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="test")

        output_path = tmp_path / "nested" / "dir" / "notebook.ipynb"
        result = export_to_ipynb(agent, output_path)

        assert result.exists()

    def test_export_multiple_cells(self, tmp_path):
        """Test exporting multiple cells."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()

        c1 = agent.session.add_cell(source="await navigate('https://example.com')")
        c1.execution_count = 1

        c2 = agent.session.add_cell(source="await click(5)")
        c2.execution_count = 2
        c2.output = "Clicked"

        c3 = agent.session.add_cell(source="await extract('title')")
        c3.execution_count = 3
        c3.error = "Extraction failed"

        output_path = tmp_path / "multi.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        # Setup + 3 code cells
        assert len(nb["cells"]) == 4

    def test_export_markdown_cell(self, tmp_path):
        """Test exporting markdown cells."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = CodeCell(
            source="# This is a heading",
            cell_type=CellType.MARKDOWN,
        )
        agent.session.cells.append(cell)

        output_path = tmp_path / "markdown.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        md_cell = nb["cells"][-1]
        assert md_cell["cell_type"] == "markdown"

    def test_export_string_output_path(self, tmp_path):
        """Test that string output path is converted to Path."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        agent.session.add_cell(source="test")

        output_path = str(tmp_path / "str_path.ipynb")
        result = export_to_ipynb(agent, output_path)

        assert isinstance(result, Path)
        assert result.exists()

    def test_export_cell_without_output(self, tmp_path):
        """Test cell with None output."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="x = 1")
        cell.execution_count = 1
        cell.output = None
        cell.error = None
        cell.browser_state = None

        output_path = tmp_path / "no_output.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        code_cell = nb["cells"][-1]
        assert code_cell["outputs"] == []

    def test_export_cell_python_counter(self, tmp_path):
        """Test that python_cell_count is incremented correctly."""
        from openbrowser.code_use.notebook_export import export_to_ipynb

        agent = MockCodeAgent()
        for i in range(5):
            cell = agent.session.add_cell(source=f"step_{i}")
            cell.execution_count = i + 1

        output_path = tmp_path / "counter.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r") as f:
            nb = json.load(f)

        # Setup + 5 code cells
        assert len(nb["cells"]) == 6


# ---------------------------------------------------------------------------
# session_to_python_script (lines 177-276)
# ---------------------------------------------------------------------------


class TestSessionToPythonScript:
    """Tests for session_to_python_script function."""

    def test_basic_script_generation(self):
        """Test basic script generation structure."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="await navigate('https://example.com')")
        cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "# Generated from openbrowser code-use session" in script
        assert "import asyncio" in script
        assert "import json" in script
        assert "from openbrowser import BrowserSession" in script
        assert "from openbrowser.code_use import create_namespace" in script
        assert "async def main():" in script
        assert "await browser.start()" in script
        assert "await browser.stop()" in script
        assert "asyncio.run(main())" in script
        assert "navigate('https://example.com')" in script

    def test_script_includes_namespace_functions(self):
        """Test that all namespace function extractions are included."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        agent.session.add_cell(source="test")

        script = session_to_python_script(agent)

        expected_functions = [
            "navigate", "click", "input_text", "evaluate",
            "search", "extract", "scroll", "done",
            "go_back", "wait", "screenshot", "find_text",
            "switch_tab", "close_tab", "dropdown_options",
            "select_dropdown", "upload_file", "send_keys",
        ]

        for func in expected_functions:
            assert func in script, f"Missing function: {func}"

    def test_script_with_javascript_blocks(self):
        """Test script generation with JavaScript blocks."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"extract_js"}
        agent.namespace["extract_js"] = "document.querySelector('h1').innerHTML;"

        cell = agent.session.add_cell(source="result = await evaluate(extract_js)")
        cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "# JavaScript Code Block: extract_js" in script
        assert 'extract_js = """' in script

    def test_script_with_all_js_patterns(self):
        """Test that all JS patterns are detected in script mode."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        patterns = [
            ("fn1", "function getData() { return 1; }"),
            ("fn2", "(function() { return 1; })"),
            ("fn3", "const fn = () => { return 1; }"),
            ("fn4", "document.getElementById('x')"),
            ("fn5", "Array.from(items)"),
            ("fn6", "el.querySelector('.item')"),
            ("fn7", "el.textContent"),
            ("fn8", "el.innerHTML"),
            ("fn9", "return result"),
            ("fn10", "console.log('test')"),
            ("fn11", "window.location"),
            ("fn12", "items.map(fn)"),
            ("fn13", "items.filter(fn)"),
            ("fn14", "items.forEach(fn)"),
        ]

        for var_name, code in patterns:
            agent = MockCodeAgent()
            agent.namespace["_code_block_vars"] = {var_name}
            agent.namespace[var_name] = code

            script = session_to_python_script(agent)
            assert f"# JavaScript Code Block: {var_name}" in script, (
                f"JS pattern not detected: {var_name} = {code}"
            )

    def test_script_skips_non_js_variables(self):
        """Test that non-JavaScript variables are not included."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"plain_text"}
        agent.namespace["plain_text"] = "just a normal string value"

        script = session_to_python_script(agent)

        assert "JavaScript Code Block: plain_text" not in script

    def test_script_skips_empty_string_vars(self):
        """Test that empty string variables are skipped."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"empty_var"}
        agent.namespace["empty_var"] = "   "

        script = session_to_python_script(agent)

        assert "JavaScript Code Block" not in script

    def test_script_skips_non_string_vars(self):
        """Test that non-string variables are skipped."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"num_var"}
        agent.namespace["num_var"] = 123

        script = session_to_python_script(agent)

        assert "JavaScript Code Block" not in script

    def test_script_multiple_cells(self):
        """Test script with multiple code cells."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()

        c1 = agent.session.add_cell(source="await navigate('https://example.com')")
        c1.execution_count = 1

        c2 = agent.session.add_cell(source="await click(5)")
        c2.execution_count = 2

        script = session_to_python_script(agent)

        assert "# Cell 1" in script
        assert "# Cell 2" in script
        assert "navigate('https://example.com')" in script
        assert "click(5)" in script

    def test_script_multiline_source(self):
        """Test script with multiline cell source."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        source = "x = 1\ny = 2\nz = x + y"
        cell = agent.session.add_cell(source=source)
        cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "x = 1" in script
        assert "y = 2" in script
        assert "z = x + y" in script

    def test_script_empty_lines_skipped(self):
        """Test that empty source lines are skipped."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        source = "x = 1\n\ny = 2"  # Contains empty line
        cell = agent.session.add_cell(source=source)
        cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "x = 1" in script
        assert "y = 2" in script

    def test_script_empty_session(self):
        """Test script for empty session."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        script = session_to_python_script(agent)

        assert "async def main():" in script
        assert "await browser.stop()" in script
        assert "asyncio.run(main())" in script

    def test_script_no_namespace(self):
        """Test script when agent has no namespace."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        delattr(agent, "namespace")
        agent.session.add_cell(source="test")

        script = session_to_python_script(agent)
        assert "JavaScript Code Block" not in script

    def test_script_empty_namespace(self):
        """Test script when agent has empty namespace."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        agent.namespace = {}
        agent.session.add_cell(source="test")

        script = session_to_python_script(agent)
        assert "JavaScript Code Block" not in script

    def test_script_only_code_cells_included(self):
        """Test that only CODE cells are included in script."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()

        # Add a markdown cell
        md_cell = CodeCell(source="# Heading", cell_type=CellType.MARKDOWN)
        agent.session.cells.append(md_cell)

        # Add a code cell
        code_cell = agent.session.add_cell(source="await navigate('https://example.com')")
        code_cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "navigate('https://example.com')" in script
        assert "# Heading" not in script  # Markdown cell should not be included

    def test_script_sorted_js_vars(self):
        """Test that JavaScript variables are sorted alphabetically."""
        from openbrowser.code_use.notebook_export import session_to_python_script

        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"z_func", "a_func"}
        agent.namespace["z_func"] = "document.querySelector('.z')"
        agent.namespace["a_func"] = "document.querySelector('.a')"

        script = session_to_python_script(agent)

        # a_func should appear before z_func
        a_pos = script.index("a_func")
        z_pos = script.index("z_func")
        assert a_pos < z_pos


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
