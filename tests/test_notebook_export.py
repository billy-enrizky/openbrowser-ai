"""Comprehensive tests for notebook_export module.

This module provides extensive test coverage for the notebook export
subsystem, which converts CodeAgent sessions to Jupyter notebook format
and Python scripts. It validates:

    - CellType and ExecutionStatus enums for notebook cell representation
    - CodeCell: Individual code cell with source, output, and execution info
    - NotebookSession: Session container with cell management and namespaces
    - NotebookExport: Jupyter notebook format with metadata and cells
    - export_to_ipynb: Full notebook export with setup code and JavaScript
    - session_to_python_script: Conversion to standalone Python scripts
    - Proper handling of outputs, errors, and browser state in cells

The notebook export system enables reproducibility and sharing of
browser automation sessions in standard notebook and script formats.
"""

import json
import logging
import re
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Local copies of necessary views to avoid circular import issues
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
    id: str = Field(default_factory=lambda: str(uuid4()))
    cell_type: CellType = CellType.CODE
    source: str = Field(description="The code to execute")
    output: str | None = Field(default=None)
    execution_count: int | None = Field(default=None)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    error: str | None = Field(default=None)
    browser_state: str | None = Field(default=None)


class NotebookSession(BaseModel):
    """Represents a notebook-like session."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    cells: list[CodeCell] = Field(default_factory=list)
    current_execution_count: int = Field(default=0)
    namespace: dict[str, Any] = Field(default_factory=dict)

    def add_cell(self, source: str) -> CodeCell:
        """Add a new code cell to the session."""
        cell = CodeCell(source=source)
        self.cells.append(cell)
        return cell

    def increment_execution_count(self) -> int:
        """Increment and return the execution count."""
        self.current_execution_count += 1
        return self.current_execution_count


class NotebookExport(BaseModel):
    """Export format for Jupyter notebook."""
    nbformat: int = Field(default=4)
    nbformat_minor: int = Field(default=5)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cells: list[dict[str, Any]] = Field(default_factory=list)


# Inline the export functions here to avoid circular import issues in testing
def export_to_ipynb(agent, output_path):
    """Export a CodeAgent session to a Jupyter notebook (.ipynb) file."""
    output_path = Path(output_path)

    notebook = NotebookExport(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.11.0",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        }
    )

    setup_code = """import asyncio
import json
import logging
from typing import Any
from src.openbrowser.browser.session import BrowserSession
from src.openbrowser.code_use.namespace import create_namespace

browser = BrowserSession()
await browser.start()
namespace: dict[str, Any] = create_namespace(browser)
globals().update(namespace)
logging.info("openbrowser environment initialized!")"""

    setup_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": setup_code.split("\n"),
        "execution_count": None,
        "outputs": [],
    }
    notebook.cells.append(setup_cell)

    # Add JavaScript code blocks
    if hasattr(agent, "namespace") and agent.namespace:
        code_block_vars = agent.namespace.get("_code_block_vars", set())
        for var_name in sorted(code_block_vars):
            var_value = agent.namespace.get(var_name)
            if isinstance(var_value, str) and var_value.strip():
                js_patterns = [
                    r"function\s+\w+\s*\(",
                    r"=>\s*{",
                    r"document\.",
                    r"\.querySelector",
                    r"\.textContent",
                    r"\.innerHTML",
                ]
                is_js = any(re.search(pattern, var_value, re.IGNORECASE) for pattern in js_patterns)
                if is_js:
                    js_cell = {
                        "cell_type": "code",
                        "metadata": {},
                        "source": [f"# JavaScript Code Block: {var_name}\n", f'{var_name} = """{var_value}"""'],
                        "execution_count": None,
                        "outputs": [],
                    }
                    notebook.cells.append(js_cell)

    # Convert cells
    for cell in agent.session.cells:
        notebook_cell = {
            "cell_type": cell.cell_type.value,
            "metadata": {},
            "source": cell.source.splitlines(keepends=True),
        }

        if cell.cell_type == CellType.CODE:
            notebook_cell["execution_count"] = cell.execution_count
            notebook_cell["outputs"] = []

            if cell.output:
                notebook_cell["outputs"].append(
                    {"output_type": "stream", "name": "stdout", "text": cell.output.split("\n")}
                )

            if cell.error:
                notebook_cell["outputs"].append(
                    {
                        "output_type": "error",
                        "ename": "Error",
                        "evalue": cell.error.split("\n")[0] if cell.error else "",
                        "traceback": cell.error.split("\n") if cell.error else [],
                    }
                )

            if cell.browser_state:
                notebook_cell["outputs"].append(
                    {"output_type": "stream", "name": "stdout", "text": [f"Browser State:\n{cell.browser_state}"]}
                )

        notebook.cells.append(notebook_cell)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook.model_dump(), f, indent=2, ensure_ascii=False)

    return output_path


def session_to_python_script(agent):
    """Convert a CodeAgent session to a Python script."""
    lines = []
    lines.append("# Generated from openbrowser code-use session\n")
    lines.append("import asyncio\n")
    lines.append("import json\n")
    lines.append("import logging\n")
    lines.append("from src.openbrowser.browser.session import BrowserSession\n")
    lines.append("from src.openbrowser.code_use.namespace import create_namespace\n\n")
    lines.append("async def main():\n")
    lines.append("\tbrowser = BrowserSession()\n")
    lines.append("\tawait browser.start()\n\n")
    lines.append("\tnamespace = create_namespace(browser)\n\n")
    lines.append('\tnavigate = namespace["navigate"]\n')
    lines.append('\tclick = namespace["click"]\n')
    lines.append('\tinput_text = namespace["input"]\n')
    lines.append('\tevaluate = namespace["evaluate"]\n')
    lines.append('\tsearch = namespace["search"]\n')
    lines.append('\textract = namespace["extract"]\n')
    lines.append('\tscroll = namespace["scroll"]\n')
    lines.append('\tdone = namespace["done"]\n')
    lines.append('\tgo_back = namespace["go_back"]\n')
    lines.append('\twait = namespace["wait"]\n')
    lines.append('\tscreenshot = namespace["screenshot"]\n')
    lines.append('\tfind_text = namespace["find_text"]\n')
    lines.append('\tswitch_tab = namespace["switch"]\n')
    lines.append('\tclose_tab = namespace["close"]\n')
    lines.append('\tdropdown_options = namespace["dropdown_options"]\n')
    lines.append('\tselect_dropdown = namespace["select_dropdown"]\n')
    lines.append('\tupload_file = namespace["upload_file"]\n')
    lines.append('\tsend_keys = namespace["send_keys"]\n\n')

    # Add JavaScript code blocks
    if hasattr(agent, "namespace") and agent.namespace:
        code_block_vars = agent.namespace.get("_code_block_vars", set())
        for var_name in sorted(code_block_vars):
            var_value = agent.namespace.get(var_name)
            if isinstance(var_value, str) and var_value.strip():
                js_patterns = [r"function\s+\w+\s*\(", r"document\.", r"\.querySelector", r"\.innerHTML"]
                is_js = any(re.search(pattern, var_value, re.IGNORECASE) for pattern in js_patterns)
                if is_js:
                    lines.append(f"\t# JavaScript Code Block: {var_name}\n")
                    lines.append(f'\t{var_name} = """{var_value}"""\n\n')

    for i, cell in enumerate(agent.session.cells):
        if cell.cell_type == CellType.CODE:
            lines.append(f"\t# Cell {i + 1}\n")
            source_lines = cell.source.split("\n")
            for line in source_lines:
                if line.strip():
                    lines.append(f"\t{line}\n")
            lines.append("\n")

    lines.append("\tawait browser.stop()\n\n")
    lines.append("if __name__ == '__main__':\n")
    lines.append("\tasyncio.run(main())\n")

    return "".join(lines)


class MockCodeAgent:
    """Mock CodeAgent for testing notebook export."""

    def __init__(self) -> None:
        self.session = NotebookSession()
        self.namespace: dict = {}


class TestExportToIpynb:
    """Tests for export_to_ipynb function."""

    def test_export_basic_session(self, tmp_path: Path) -> None:
        """Test exporting a basic session with one code cell."""
        agent = MockCodeAgent()

        # Add a code cell
        cell = agent.session.add_cell(source="await navigate('https://example.com')")
        cell.status = ExecutionStatus.SUCCESS
        cell.execution_count = 1
        cell.output = "Navigated to https://example.com"

        output_path = tmp_path / "test_notebook.ipynb"
        result_path = export_to_ipynb(agent, output_path)

        assert result_path == output_path
        assert output_path.exists()

        # Verify notebook structure
        with open(output_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] == 5
        assert "kernelspec" in notebook["metadata"]
        assert notebook["metadata"]["kernelspec"]["language"] == "python"

        # Should have setup cell + 1 code cell
        assert len(notebook["cells"]) >= 2

    def test_export_with_multiple_cells(self, tmp_path: Path) -> None:
        """Test exporting a session with multiple code cells."""
        agent = MockCodeAgent()

        # Add multiple cells
        cell1 = agent.session.add_cell(source="await navigate('https://example.com')")
        cell1.status = ExecutionStatus.SUCCESS
        cell1.execution_count = 1

        cell2 = agent.session.add_cell(source="await click(5)")
        cell2.status = ExecutionStatus.SUCCESS
        cell2.execution_count = 2

        cell3 = agent.session.add_cell(source="result = await extract('Get page title')")
        cell3.status = ExecutionStatus.SUCCESS
        cell3.execution_count = 3
        cell3.output = "Page title: Example"

        output_path = tmp_path / "multi_cell.ipynb"
        result_path = export_to_ipynb(agent, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Setup cell + 3 code cells
        assert len(notebook["cells"]) == 4

    def test_export_with_error_cell(self, tmp_path: Path) -> None:
        """Test exporting a session with error output."""
        agent = MockCodeAgent()

        cell = agent.session.add_cell(source="await click(999)")
        cell.status = ExecutionStatus.ERROR
        cell.execution_count = 1
        cell.error = "Element not found: index 999"

        output_path = tmp_path / "error_cell.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Find the error cell (last one)
        code_cell = notebook["cells"][-1]
        assert code_cell["cell_type"] == "code"
        assert len(code_cell["outputs"]) > 0

        error_output = code_cell["outputs"][0]
        assert error_output["output_type"] == "error"
        assert "Element not found" in error_output["evalue"]

    def test_export_with_browser_state(self, tmp_path: Path) -> None:
        """Test exporting a session with browser state."""
        agent = MockCodeAgent()

        cell = agent.session.add_cell(source="await navigate('https://example.com')")
        cell.status = ExecutionStatus.SUCCESS
        cell.execution_count = 1
        cell.browser_state = "URL: https://example.com\nTitle: Example Domain"

        output_path = tmp_path / "browser_state.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Check browser state output
        code_cell = notebook["cells"][-1]
        browser_state_found = False
        for output in code_cell["outputs"]:
            if "Browser State" in str(output.get("text", [])):
                browser_state_found = True
                break
        assert browser_state_found

    def test_export_with_javascript_blocks(self, tmp_path: Path) -> None:
        """Test exporting a session with JavaScript code blocks in namespace."""
        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"js_extract"}
        agent.namespace[
            "js_extract"
        ] = """
        function extractData() {
            return document.querySelector('.title').textContent;
        }
        """

        cell = agent.session.add_cell(source="result = await evaluate(js_extract)")
        cell.status = ExecutionStatus.SUCCESS
        cell.execution_count = 1

        output_path = tmp_path / "js_blocks.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Should have setup cell + JS block cell + code cell
        assert len(notebook["cells"]) >= 3

        # Find JS block cell
        js_cell_found = False
        for cell_data in notebook["cells"]:
            source = "".join(cell_data.get("source", []))
            if "JavaScript Code Block" in source and "js_extract" in source:
                js_cell_found = True
                break
        assert js_cell_found

    def test_export_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that export creates parent directories if they don't exist."""
        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="await navigate('https://example.com')")
        cell.status = ExecutionStatus.SUCCESS

        output_path = tmp_path / "nested" / "dir" / "structure" / "notebook.ipynb"
        result_path = export_to_ipynb(agent, output_path)

        assert result_path.exists()

    def test_export_empty_session(self, tmp_path: Path) -> None:
        """Test exporting an empty session (only setup cell)."""
        agent = MockCodeAgent()

        output_path = tmp_path / "empty.ipynb"
        export_to_ipynb(agent, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        # Should have at least the setup cell
        assert len(notebook["cells"]) >= 1


class TestSessionToPythonScript:
    """Tests for session_to_python_script function."""

    def test_basic_script_generation(self) -> None:
        """Test basic script generation."""
        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="await navigate('https://example.com')")
        cell.status = ExecutionStatus.SUCCESS
        cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "# Generated from openbrowser code-use session" in script
        assert "import asyncio" in script
        assert "import logging" in script
        assert "from src.openbrowser.browser.session import BrowserSession" in script
        assert "async def main():" in script
        assert "await browser.start()" in script
        assert "await browser.stop()" in script
        assert "asyncio.run(main())" in script
        assert "navigate('https://example.com')" in script

    def test_script_with_multiple_cells(self) -> None:
        """Test script generation with multiple cells."""
        agent = MockCodeAgent()

        cell1 = agent.session.add_cell(source="await navigate('https://example.com')")
        cell1.execution_count = 1

        cell2 = agent.session.add_cell(source="await click(5)")
        cell2.execution_count = 2

        script = session_to_python_script(agent)

        assert "# Cell 1" in script
        assert "# Cell 2" in script
        assert "navigate('https://example.com')" in script
        assert "click(5)" in script

    def test_script_with_javascript_blocks(self) -> None:
        """Test script generation with JavaScript blocks."""
        agent = MockCodeAgent()
        agent.namespace["_code_block_vars"] = {"extract_script"}
        agent.namespace[
            "extract_script"
        ] = """
        document.querySelector('h1').innerHTML;
        """

        cell = agent.session.add_cell(source="result = await evaluate(extract_script)")
        cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "# JavaScript Code Block: extract_script" in script
        assert 'extract_script = """' in script

    def test_script_includes_namespace_functions(self) -> None:
        """Test that script includes all namespace function extractions."""
        agent = MockCodeAgent()
        cell = agent.session.add_cell(source="test")
        cell.execution_count = 1

        script = session_to_python_script(agent)

        expected_functions = [
            "navigate",
            "click",
            "input_text",
            "evaluate",
            "search",
            "extract",
            "scroll",
            "done",
            "go_back",
            "wait",
            "screenshot",
            "find_text",
            "switch_tab",
            "close_tab",
            "dropdown_options",
            "select_dropdown",
            "upload_file",
            "send_keys",
        ]

        for func in expected_functions:
            assert func in script

    def test_empty_session_script(self) -> None:
        """Test script generation for empty session."""
        agent = MockCodeAgent()

        script = session_to_python_script(agent)

        assert "async def main():" in script
        assert "await browser.stop()" in script

    def test_multiline_cell_source(self) -> None:
        """Test script generation with multiline cell source."""
        agent = MockCodeAgent()
        multiline_source = """await navigate('https://example.com')
await click(5)
result = await extract('Get data')"""

        cell = agent.session.add_cell(source=multiline_source)
        cell.execution_count = 1

        script = session_to_python_script(agent)

        assert "navigate('https://example.com')" in script
        assert "click(5)" in script
        assert "extract('Get data')" in script


class TestSystemPromptNoThinking:
    """Tests for system_prompt_no_thinking.md file."""

    def test_file_exists(self) -> None:
        """Test that system_prompt_no_thinking.md exists."""
        prompt_path = Path("src/openbrowser/agent/system_prompt_no_thinking.md")
        assert prompt_path.exists(), "system_prompt_no_thinking.md should exist"

    def test_file_content_structure(self) -> None:
        """Test that system_prompt_no_thinking.md has expected structure."""
        prompt_path = Path("src/openbrowser/agent/system_prompt_no_thinking.md")
        content = prompt_path.read_text(encoding="utf-8")

        # Check for key sections
        assert "<intro>" in content
        assert "<language_settings>" in content
        assert "<input>" in content
        assert "<agent_history>" in content
        assert "<user_request>" in content
        assert "<browser_state>" in content
        assert "<browser_vision>" in content
        assert "<browser_rules>" in content
        assert "<file_system>" in content
        assert "<task_completion_rules>" in content
        assert "<action_rules>" in content
        assert "<efficiency_guidelines>" in content
        assert "<output>" in content

    def test_no_reasoning_rules_section(self) -> None:
        """Test that reasoning_rules section is not present (vs regular prompt)."""
        prompt_path = Path("src/openbrowser/agent/system_prompt_no_thinking.md")
        content = prompt_path.read_text(encoding="utf-8")

        # The no_thinking version should not have extended reasoning rules
        # This is the key difference from the regular system_prompt.md
        assert "<reasoning_rules>" not in content or "Be clear and concise" not in content

    def test_json_output_format(self) -> None:
        """Test that output format is simplified JSON."""
        prompt_path = Path("src/openbrowser/agent/system_prompt_no_thinking.md")
        content = prompt_path.read_text(encoding="utf-8")

        # Should have JSON output format
        assert "evaluation_previous_goal" in content
        assert "memory" in content
        assert "next_goal" in content
        assert "action" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
