"""Code-use agent module - Jupyter notebook-like code execution for browser automation.

This module provides a Jupyter notebook-like interface for browser automation where
an LLM writes and executes Python code in a persistent namespace with browser control
functions available.

Key Components:
    CodeAgent: The main agent class that orchestrates code execution and browser control.
    NotebookSession: Represents a session of executed code cells.
    CodeCell: Individual code cell with source, output, and execution status.
    create_namespace: Creates the execution namespace with browser tools.
    export_to_ipynb: Exports session to Jupyter notebook format.

Example:
    ```python
    from openbrowser.code_use import CodeAgent
    from openbrowser.browser.session import BrowserSession

    async def main():
        browser = BrowserSession()
        await browser.start()

        agent = CodeAgent(
            task="Navigate to google.com and search for 'Python'",
            llm=my_llm,
            browser=browser,
        )
        session = await agent.run()
    ```
"""

from openbrowser.code_use.namespace import create_namespace
from openbrowser.code_use.notebook_export import export_to_ipynb, session_to_python_script
from openbrowser.code_use.service import CodeAgent
from openbrowser.code_use.views import (
    CellType,
    CodeAgentHistory,
    CodeAgentModelOutput,
    CodeAgentResult,
    CodeAgentState,
    CodeAgentStepMetadata,
    CodeCell,
    ExecutionStatus,
    NotebookExport,
    NotebookSession,
)

__all__ = [
    "CellType",
    "CodeAgent",
    "CodeAgentHistory",
    "CodeAgentModelOutput",
    "CodeAgentResult",
    "CodeAgentState",
    "CodeAgentStepMetadata",
    "CodeCell",
    "ExecutionStatus",
    "NotebookExport",
    "NotebookSession",
    "create_namespace",
    "export_to_ipynb",
    "session_to_python_script",
]

