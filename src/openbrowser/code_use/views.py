"""Data models for code-use mode."""

import base64
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class CellType(str, Enum):
    """Type of notebook cell.

    Defines the type of content a cell can contain, similar to Jupyter notebooks.

    Attributes:
        CODE: A cell containing executable Python code.
        MARKDOWN: A cell containing Markdown-formatted documentation.
    """

    CODE = "code"
    MARKDOWN = "markdown"


class ExecutionStatus(str, Enum):
    """Execution status of a cell.

    Tracks the lifecycle state of a code cell's execution.

    Attributes:
        PENDING: Cell has not yet been executed.
        RUNNING: Cell is currently executing.
        SUCCESS: Cell executed successfully without errors.
        ERROR: Cell execution failed with an error.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class CodeCell(BaseModel):
    """Represents a code cell in the notebook-like execution environment.

    A CodeCell contains Python source code and tracks its execution state,
    output, and any errors that occurred during execution.

    Attributes:
        id: Unique identifier for the cell.
        cell_type: Type of cell (code or markdown).
        source: The Python code to execute.
        output: Standard output captured during execution.
        execution_count: Sequential execution number (like Jupyter's In[n]).
        status: Current execution status of the cell.
        error: Error message if execution failed.
        browser_state: Snapshot of browser state after cell execution.

    Example:
        ```python
        cell = CodeCell(source="print('Hello, world!')")
        cell.status = ExecutionStatus.SUCCESS
        cell.output = "Hello, world!"
        ```
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    cell_type: CellType = CellType.CODE
    source: str = Field(description="The code to execute")
    output: str | None = Field(default=None, description="The output of the code execution")
    execution_count: int | None = Field(default=None, description="The execution count")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    error: str | None = Field(default=None, description="Error message if execution failed")
    browser_state: str | None = Field(default=None, description="Browser state after execution")


class NotebookSession(BaseModel):
    """Represents a notebook-like session containing multiple code cells.

    A NotebookSession manages a collection of CodeCells, tracking their
    execution order and maintaining the shared namespace state.

    Attributes:
        id: Unique identifier for the session.
        cells: List of code cells in execution order.
        current_execution_count: Counter for cell execution numbering.
        namespace: Current state of variables in the execution namespace.

    Example:
        ```python
        session = NotebookSession()
        cell = session.add_cell("x = 42")
        execution_num = session.increment_execution_count()
        cell.execution_count = execution_num
        ```
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    cells: list[CodeCell] = Field(default_factory=list)
    current_execution_count: int = Field(default=0)
    namespace: dict[str, Any] = Field(default_factory=dict, description="Current namespace state")

    def add_cell(self, source: str) -> CodeCell:
        """Add a new code cell to the session.

        Creates a new CodeCell with the given source code and appends it
        to the session's cell list.

        Args:
            source: The Python source code for the cell.

        Returns:
            The newly created CodeCell instance.
        """
        cell = CodeCell(source=source)
        self.cells.append(cell)
        return cell

    def get_cell(self, cell_id: str) -> CodeCell | None:
        """Get a cell by its unique identifier.

        Args:
            cell_id: The unique ID of the cell to retrieve.

        Returns:
            The CodeCell with the matching ID, or None if not found.
        """
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        return None

    def get_latest_cell(self) -> CodeCell | None:
        """Get the most recently added cell.

        Returns:
            The last CodeCell in the session, or None if the session is empty.
        """
        if self.cells:
            return self.cells[-1]
        return None

    def increment_execution_count(self) -> int:
        """Increment and return the execution count.

        Used to assign sequential execution numbers to cells, similar to
        Jupyter's In[n] numbering.

        Returns:
            The new execution count after incrementing.
        """
        self.current_execution_count += 1
        return self.current_execution_count


class NotebookExport(BaseModel):
    """Export format for Jupyter notebook (.ipynb) files.

    Represents the structure of a Jupyter notebook for serialization.
    Follows the nbformat v4 specification.

    Attributes:
        nbformat: Major version of notebook format (default: 4).
        nbformat_minor: Minor version of notebook format (default: 5).
        metadata: Notebook-level metadata (kernelspec, language_info, etc.).
        cells: List of cell dictionaries in Jupyter notebook format.

    Example:
        ```python
        notebook = NotebookExport(
            metadata={"kernelspec": {"name": "python3"}},
            cells=[{"cell_type": "code", "source": ["print('hello')"]}]
        )
        with open("output.ipynb", "w") as f:
            json.dump(notebook.model_dump(), f)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    nbformat: int = Field(default=4)
    nbformat_minor: int = Field(default=5)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cells: list[dict[str, Any]] = Field(default_factory=list)


class CodeAgentModelOutput(BaseModel):
    """Model output for CodeAgent containing code and full LLM response.

    Captures both the extracted Python code and the complete LLM response
    which may include reasoning, explanations, and other text.

    Attributes:
        model_output: The extracted Python code from the LLM response.
        full_response: The complete LLM response including explanations.

    Example:
        ```python
        output = CodeAgentModelOutput(
            model_output="await navigate('https://example.com')",
            full_response="I'll navigate to the website first.\n```python\nawait navigate('https://example.com')\n```"
        )
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model_output: str = Field(description="The extracted code from the LLM response")
    full_response: str = Field(description="The complete LLM response including any text/reasoning")


class CodeAgentResult(BaseModel):
    """Result of executing a code cell in CodeAgent.

    Contains the outcome of a single code cell execution, including
    any output, errors, and task completion status.

    Attributes:
        extracted_content: Standard output or return value from execution.
        error: Error message if the execution failed.
        is_done: Whether the task has been marked as complete via done().
        success: Self-reported success status from the done() call.

    Example:
        ```python
        result = CodeAgentResult(
            extracted_content="Successfully scraped 10 items",
            is_done=True,
            success=True
        )
        ```
    """

    model_config = ConfigDict(extra="forbid")

    extracted_content: str | None = Field(default=None, description="Output from code execution")
    error: str | None = Field(default=None, description="Error message if execution failed")
    is_done: bool = Field(default=False, description="Whether task is marked as done")
    success: bool | None = Field(default=None, description="Self-reported success from done() call")


class CodeAgentState(BaseModel):
    """State information for a CodeAgent step.

    Captures the browser state at a particular point during agent execution,
    including the current URL, page title, and an optional screenshot.

    Attributes:
        url: The current page URL in the browser.
        title: The current page title.
        screenshot_path: File path to the saved screenshot image.

    Example:
        ```python
        state = CodeAgentState(
            url="https://example.com",
            title="Example Domain",
            screenshot_path="/tmp/screenshot_1.jpg"
        )
        screenshot_b64 = state.get_screenshot()
        ```
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    url: str | None = Field(default=None, description="Current page URL")
    title: str | None = Field(default=None, description="Current page title")
    screenshot_path: str | None = Field(default=None, description="Path to screenshot file")

    def get_screenshot(self) -> str | None:
        """Load screenshot from disk and return as base64-encoded string.

        Reads the screenshot file from disk and encodes it as a base64 string
        suitable for embedding in HTML or sending via API.

        Returns:
            Base64-encoded screenshot data, or None if the file doesn't exist
            or cannot be read.
        """
        if not self.screenshot_path:
            return None

        path_obj = Path(self.screenshot_path)
        if not path_obj.exists():
            return None

        try:
            with open(path_obj, "rb") as f:
                screenshot_data = f.read()
            return base64.b64encode(screenshot_data).decode("utf-8")
        except Exception:
            return None


class CodeAgentStepMetadata(BaseModel):
    """Metadata for a single CodeAgent step including timing and token information.

    Tracks performance metrics for each step of the agent's execution,
    including LLM token usage and execution timing.

    Attributes:
        input_tokens: Number of input tokens sent to the LLM.
        output_tokens: Number of output tokens received from the LLM.
        step_start_time: Unix timestamp when the step began.
        step_end_time: Unix timestamp when the step completed.

    Example:
        ```python
        metadata = CodeAgentStepMetadata(
            input_tokens=1500,
            output_tokens=200,
            step_start_time=1704067200.0,
            step_end_time=1704067205.5
        )
        print(f"Step took {metadata.duration_seconds:.2f} seconds")
        ```
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, description="Number of input tokens used")
    output_tokens: int | None = Field(default=None, description="Number of output tokens used")
    step_start_time: float = Field(description="Step start timestamp (Unix time)")
    step_end_time: float = Field(description="Step end timestamp (Unix time)")

    @property
    def duration_seconds(self) -> float:
        """Calculate the step duration in seconds.

        Returns:
            The elapsed time between step start and end in seconds.
        """
        return self.step_end_time - self.step_start_time


class CodeAgentHistory(BaseModel):
    """History item for CodeAgent actions.

    Represents a complete record of a single step in the agent's execution,
    including the LLM's output, execution results, browser state, and metadata.

    Attributes:
        model_output: The LLM's response for this step (code and explanation).
        result: List of execution results from the code cell(s).
        state: Browser state snapshot after this step.
        metadata: Timing and token usage metrics for this step.
        screenshot_path: Path to screenshot file (legacy, prefer state.screenshot_path).

    Example:
        ```python
        history_item = CodeAgentHistory(
            model_output=CodeAgentModelOutput(model_output="...", full_response="..."),
            result=[CodeAgentResult(extracted_content="Done")],
            state=CodeAgentState(url="https://example.com"),
            metadata=CodeAgentStepMetadata(step_start_time=0.0, step_end_time=1.0)
        )
        ```
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_output: CodeAgentModelOutput | None = Field(default=None, description="LLM output for this step")
    result: list[CodeAgentResult] = Field(default_factory=list, description="Results from code execution")
    state: CodeAgentState = Field(description="Browser state at this step")
    metadata: CodeAgentStepMetadata | None = Field(default=None, description="Step timing and token metadata")
    screenshot_path: str | None = Field(default=None, description="Legacy field for screenshot path")

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serialize the history item to a dictionary.

        Custom serialization that properly handles nested Pydantic models.

        Args:
            **kwargs: Additional arguments passed to nested model_dump calls.

        Returns:
            Dictionary representation of the history item.
        """
        return {
            "model_output": self.model_output.model_dump() if self.model_output else None,
            "result": [r.model_dump() for r in self.result],
            "state": self.state.model_dump(),
            "metadata": self.metadata.model_dump() if self.metadata else None,
            "screenshot_path": self.screenshot_path,
        }

