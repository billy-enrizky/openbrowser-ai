"""Comprehensive tests for openbrowser.code_use.views module.

Covers: CellType, ExecutionStatus, CodeCell, NotebookSession,
NotebookExport, CodeAgentModelOutput, CodeAgentResult, CodeAgentState,
CodeAgentStepMetadata, CodeAgentHistory.
"""

import base64
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestCellType:
    def test_code_value(self):
        assert CellType.CODE == "code"

    def test_markdown_value(self):
        assert CellType.MARKDOWN == "markdown"


class TestExecutionStatus:
    def test_all_values(self):
        assert ExecutionStatus.PENDING == "pending"
        assert ExecutionStatus.RUNNING == "running"
        assert ExecutionStatus.SUCCESS == "success"
        assert ExecutionStatus.ERROR == "error"


# ---------------------------------------------------------------------------
# CodeCell
# ---------------------------------------------------------------------------


class TestCodeCell:
    def test_default_values(self):
        cell = CodeCell(source="x = 1")
        assert cell.source == "x = 1"
        assert cell.cell_type == CellType.CODE
        assert cell.output is None
        assert cell.execution_count is None
        assert cell.status == ExecutionStatus.PENDING
        assert cell.error is None
        assert cell.browser_state is None
        assert cell.id is not None  # auto-generated UUID7

    def test_custom_values(self):
        cell = CodeCell(
            source="print(1)",
            cell_type=CellType.MARKDOWN,
            output="1",
            execution_count=5,
            status=ExecutionStatus.SUCCESS,
            error=None,
            browser_state="some state",
        )
        assert cell.execution_count == 5
        assert cell.status == ExecutionStatus.SUCCESS
        assert cell.browser_state == "some state"

    def test_extra_fields_forbidden(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CodeCell(source="x", extra_field="bad")


# ---------------------------------------------------------------------------
# NotebookSession
# ---------------------------------------------------------------------------


class TestNotebookSession:
    def test_default_empty(self):
        session = NotebookSession()
        assert session.cells == []
        assert session.current_execution_count == 0
        assert session.namespace == {}

    def test_add_cell(self):
        session = NotebookSession()
        cell = session.add_cell("x = 1")
        assert len(session.cells) == 1
        assert cell.source == "x = 1"
        assert cell is session.cells[0]

    def test_get_cell_found(self):
        session = NotebookSession()
        cell = session.add_cell("x = 1")
        found = session.get_cell(cell.id)
        assert found is cell

    def test_get_cell_not_found(self):
        session = NotebookSession()
        session.add_cell("x = 1")
        found = session.get_cell("nonexistent-id")
        assert found is None

    def test_get_latest_cell(self):
        session = NotebookSession()
        session.add_cell("a")
        session.add_cell("b")
        latest = session.get_latest_cell()
        assert latest.source == "b"

    def test_get_latest_cell_empty(self):
        session = NotebookSession()
        assert session.get_latest_cell() is None

    def test_increment_execution_count(self):
        session = NotebookSession()
        assert session.increment_execution_count() == 1
        assert session.increment_execution_count() == 2
        assert session.increment_execution_count() == 3
        assert session.current_execution_count == 3


# ---------------------------------------------------------------------------
# NotebookExport
# ---------------------------------------------------------------------------


class TestNotebookExport:
    def test_defaults(self):
        export = NotebookExport()
        assert export.nbformat == 4
        assert export.nbformat_minor == 5
        assert export.metadata == {}
        assert export.cells == []

    def test_custom_values(self):
        export = NotebookExport(
            nbformat=3,
            nbformat_minor=4,
            metadata={"kernel": "python3"},
            cells=[{"cell_type": "code", "source": "x = 1"}],
        )
        assert export.nbformat == 3
        assert len(export.cells) == 1


# ---------------------------------------------------------------------------
# CodeAgentModelOutput
# ---------------------------------------------------------------------------


class TestCodeAgentModelOutput:
    def test_creation(self):
        output = CodeAgentModelOutput(
            model_output="print(1)",
            full_response="Sure, here is the code:\n```python\nprint(1)\n```",
        )
        assert output.model_output == "print(1)"
        assert "Sure" in output.full_response


# ---------------------------------------------------------------------------
# CodeAgentResult
# ---------------------------------------------------------------------------


class TestCodeAgentResult:
    def test_defaults(self):
        result = CodeAgentResult()
        assert result.extracted_content is None
        assert result.error is None
        assert result.is_done is False
        assert result.success is None

    def test_custom_values(self):
        result = CodeAgentResult(
            extracted_content="output",
            error="some error",
            is_done=True,
            success=True,
        )
        assert result.is_done is True
        assert result.success is True


# ---------------------------------------------------------------------------
# CodeAgentState
# ---------------------------------------------------------------------------


class TestCodeAgentState:
    def test_defaults(self):
        state = CodeAgentState()
        assert state.url is None
        assert state.title is None
        assert state.screenshot_path is None

    def test_get_screenshot_no_path(self):
        state = CodeAgentState()
        assert state.get_screenshot() is None

    def test_get_screenshot_nonexistent_file(self):
        state = CodeAgentState(screenshot_path="/nonexistent/path/screenshot.png")
        assert state.get_screenshot() is None

    def test_get_screenshot_valid_file(self, tmp_path):
        img_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img_path = tmp_path / "screenshot.png"
        img_path.write_bytes(img_data)

        state = CodeAgentState(screenshot_path=str(img_path))
        result = state.get_screenshot()
        assert result is not None
        decoded = base64.b64decode(result)
        assert decoded == img_data

    def test_get_screenshot_permission_error(self, tmp_path):
        img_path = tmp_path / "screenshot.png"
        img_path.write_bytes(b"data")

        state = CodeAgentState(screenshot_path=str(img_path))
        with patch("builtins.open", side_effect=PermissionError("mocked permission denied")):
            result = state.get_screenshot()
        # Should return None on permission error
        assert result is None


# ---------------------------------------------------------------------------
# CodeAgentStepMetadata
# ---------------------------------------------------------------------------


class TestCodeAgentStepMetadata:
    def test_duration_seconds(self):
        meta = CodeAgentStepMetadata(
            step_start_time=100.0,
            step_end_time=105.5,
        )
        assert meta.duration_seconds == pytest.approx(5.5)

    def test_optional_tokens(self):
        meta = CodeAgentStepMetadata(
            step_start_time=0.0,
            step_end_time=1.0,
            input_tokens=500,
            output_tokens=200,
        )
        assert meta.input_tokens == 500
        assert meta.output_tokens == 200

    def test_none_tokens(self):
        meta = CodeAgentStepMetadata(
            step_start_time=0.0,
            step_end_time=1.0,
        )
        assert meta.input_tokens is None
        assert meta.output_tokens is None


# ---------------------------------------------------------------------------
# CodeAgentHistory
# ---------------------------------------------------------------------------


class TestCodeAgentHistory:
    def test_model_dump_with_all_fields(self):
        history = CodeAgentHistory(
            model_output=CodeAgentModelOutput(
                model_output="code", full_response="full"
            ),
            result=[CodeAgentResult(extracted_content="output")],
            state=CodeAgentState(url="https://example.com"),
            metadata=CodeAgentStepMetadata(step_start_time=0.0, step_end_time=1.0),
            screenshot_path="/tmp/screenshot.png",
        )
        d = history.model_dump()
        assert d["model_output"]["model_output"] == "code"
        assert len(d["result"]) == 1
        assert d["state"]["url"] == "https://example.com"
        assert d["metadata"]["step_start_time"] == 0.0
        assert d["screenshot_path"] == "/tmp/screenshot.png"

    def test_model_dump_with_none_fields(self):
        history = CodeAgentHistory(
            model_output=None,
            result=[],
            state=CodeAgentState(),
            metadata=None,
            screenshot_path=None,
        )
        d = history.model_dump()
        assert d["model_output"] is None
        assert d["result"] == []
        assert d["metadata"] is None
        assert d["screenshot_path"] is None

    def test_default_result_list(self):
        history = CodeAgentHistory(
            state=CodeAgentState(),
        )
        assert history.result == []
        assert history.model_output is None
