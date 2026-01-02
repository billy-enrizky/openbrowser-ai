"""Export code-use session to Jupyter notebook format.

This module provides utilities to export CodeAgent sessions to Jupyter
notebook (.ipynb) files or Python scripts. This allows users to:

- Review and replay browser automation sessions
- Modify and re-run automation code
- Share automation workflows with others
- Debug and refine automation scripts

The exported notebooks include setup code for the openbrowser environment
and all code cells from the session with their outputs.
"""

import json
import logging
import re
from pathlib import Path

from .views import CellType, NotebookExport

# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import CodeAgent

logger = logging.getLogger(__name__)


def export_to_ipynb(agent: "CodeAgent", output_path: str | Path) -> Path:
    """Export a CodeAgent session to a Jupyter notebook (.ipynb) file.

    Creates a complete Jupyter notebook containing:
    - A setup cell that initializes the openbrowser environment
    - JavaScript code block variables defined during the session
    - All Python code cells executed during the session
    - Outputs and errors from each cell execution
    - Browser state snapshots where available

    The exported notebook can be opened in JupyterLab or VS Code and
    re-executed to replay the automation session.

    Args:
        agent: The CodeAgent instance whose session should be exported.
            Must have already run (session.cells populated).
        output_path: File path where the notebook should be saved.
            Parent directories are created if they don't exist.

    Returns:
        Path object pointing to the saved notebook file.

    Raises:
        OSError: If the file cannot be written.

    Example:
        ```python
        agent = CodeAgent(task="Search Google", llm=llm)
        session = await agent.run()

        # Export to notebook
        notebook_path = export_to_ipynb(agent, 'my_automation.ipynb')
        logger.info(f'Notebook saved to {notebook_path}')

        # Can also use Path objects
        from pathlib import Path
        export_to_ipynb(agent, Path('output') / 'session.ipynb')
        ```
    """
    output_path = Path(output_path)

    # Create notebook structure
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

    # Add setup cell at the beginning with proper type hints
    setup_code = """import asyncio
import json
import logging
from typing import Any
from src.openbrowser.browser.session import BrowserSession
from src.openbrowser.code_use.namespace import create_namespace

# Initialize browser and namespace
browser = BrowserSession()
await browser.start()

# Create namespace with all browser control functions
namespace: dict[str, Any] = create_namespace(browser)

# Import all functions into the current namespace
globals().update(namespace)

# Type hints for better IDE support (these are now available globally)
# navigate, click, input, evaluate, search, extract, scroll, done, etc.

logging.info("openbrowser environment initialized!")
logging.info("Available functions: navigate, click, input, evaluate, search, extract, done, etc.")"""

    setup_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": setup_code.split("\n"),
        "execution_count": None,
        "outputs": [],
    }
    notebook.cells.append(setup_cell)

    # Add JavaScript code blocks as variables FIRST
    if hasattr(agent, "namespace") and agent.namespace:
        # Look for JavaScript variables in the namespace
        code_block_vars = agent.namespace.get("_code_block_vars", set())

        for var_name in sorted(code_block_vars):
            var_value = agent.namespace.get(var_name)
            if isinstance(var_value, str) and var_value.strip():
                # Check if this looks like JavaScript code
                # Look for common JS patterns
                js_patterns = [
                    r"function\s+\w+\s*\(",
                    r"\(\s*function\s*\(\)",
                    r"=>\s*{",
                    r"document\.",
                    r"Array\.from\(",
                    r"\.querySelector",
                    r"\.textContent",
                    r"\.innerHTML",
                    r"return\s+",
                    r"console\.log",
                    r"window\.",
                    r"\.map\(",
                    r"\.filter\(",
                    r"\.forEach\(",
                ]

                is_js = any(re.search(pattern, var_value, re.IGNORECASE) for pattern in js_patterns)

                if is_js:
                    # Create a code cell with the JavaScript variable
                    js_cell = {
                        "cell_type": "code",
                        "metadata": {},
                        "source": [f"# JavaScript Code Block: {var_name}\n", f'{var_name} = """{var_value}"""'],
                        "execution_count": None,
                        "outputs": [],
                    }
                    notebook.cells.append(js_cell)

    # Convert cells
    python_cell_count = 0
    for cell in agent.session.cells:
        notebook_cell: dict = {
            "cell_type": cell.cell_type.value,
            "metadata": {},
            "source": cell.source.splitlines(keepends=True),
        }

        if cell.cell_type == CellType.CODE:
            python_cell_count += 1
            notebook_cell["execution_count"] = cell.execution_count
            notebook_cell["outputs"] = []

            # Add output if available
            if cell.output:
                notebook_cell["outputs"].append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": cell.output.split("\n"),
                    }
                )

            # Add error if available
            if cell.error:
                notebook_cell["outputs"].append(
                    {
                        "output_type": "error",
                        "ename": "Error",
                        "evalue": cell.error.split("\n")[0] if cell.error else "",
                        "traceback": cell.error.split("\n") if cell.error else [],
                    }
                )

            # Add browser state as a separate output
            if cell.browser_state:
                notebook_cell["outputs"].append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": [f"Browser State:\n{cell.browser_state}"],
                    }
                )

        notebook.cells.append(notebook_cell)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info(f"Exported notebook to {output_path}")
    return output_path


def session_to_python_script(agent: "CodeAgent") -> str:
    """Convert a CodeAgent session to a standalone Python script.

    Creates an async Python script that can be run independently to
    replay the browser automation session. The script includes:
    - Import statements for required modules
    - Browser session initialization
    - Namespace creation and function extraction
    - JavaScript code block variables
    - All code cells from the session
    - Browser cleanup on completion

    The generated script uses asyncio.run() to execute the async main()
    function, making it directly runnable with `python script.py`.

    Args:
        agent: The CodeAgent instance whose session should be converted.
            Must have already run (session.cells populated).

    Returns:
        Complete Python script as a string, ready to be saved to a file
        or executed.

    Example:
        ```python
        agent = CodeAgent(task="Scrape data", llm=llm)
        await agent.run()

        # Convert to script
        script = session_to_python_script(agent)

        # Save to file
        with open('automation.py', 'w') as f:
            f.write(script)

        # Or print for review
        print(script)
        ```

    Note:
        The generated script extracts common functions from the namespace
        for direct access (navigate, click, input_text, evaluate, etc.).
        Additional functions can be accessed via the namespace dictionary.
    """
    lines = []

    lines.append("# Generated from openbrowser code-use session\n")
    lines.append("import asyncio\n")
    lines.append("import json\n")
    lines.append("import logging\n")
    lines.append("from src.openbrowser.browser.session import BrowserSession\n")
    lines.append("from src.openbrowser.code_use.namespace import create_namespace\n\n")

    lines.append("async def main():\n")
    lines.append("\t# Initialize browser and namespace\n")
    lines.append("\tbrowser = BrowserSession()\n")
    lines.append("\tawait browser.start()\n\n")
    lines.append("\t# Create namespace with all browser control functions\n")
    lines.append("\tnamespace = create_namespace(browser)\n\n")
    lines.append("\t# Extract functions from namespace for direct access\n")
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

    # Add JavaScript code blocks as variables FIRST
    if hasattr(agent, "namespace") and agent.namespace:
        code_block_vars = agent.namespace.get("_code_block_vars", set())

        for var_name in sorted(code_block_vars):
            var_value = agent.namespace.get(var_name)
            if isinstance(var_value, str) and var_value.strip():
                # Check if this looks like JavaScript code
                js_patterns = [
                    r"function\s+\w+\s*\(",
                    r"\(\s*function\s*\(\)",
                    r"=>\s*{",
                    r"document\.",
                    r"Array\.from\(",
                    r"\.querySelector",
                    r"\.textContent",
                    r"\.innerHTML",
                    r"return\s+",
                    r"console\.log",
                    r"window\.",
                    r"\.map\(",
                    r"\.filter\(",
                    r"\.forEach\(",
                ]

                is_js = any(re.search(pattern, var_value, re.IGNORECASE) for pattern in js_patterns)

                if is_js:
                    lines.append(f"\t# JavaScript Code Block: {var_name}\n")
                    lines.append(f'\t{var_name} = """{var_value}"""\n\n')

    for i, cell in enumerate(agent.session.cells):
        if cell.cell_type == CellType.CODE:
            lines.append(f"\t# Cell {i + 1}\n")

            # Indent each line of source
            source_lines = cell.source.split("\n")
            for line in source_lines:
                if line.strip():  # Only add non-empty lines
                    lines.append(f"\t{line}\n")

            lines.append("\n")

    lines.append("\tawait browser.stop()\n\n")
    lines.append("if __name__ == '__main__':\n")
    lines.append("\tasyncio.run(main())\n")

    return "".join(lines)
