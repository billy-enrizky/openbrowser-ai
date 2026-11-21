"""File system ownership tests for CodeAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

from openbrowser.code_use.service import CodeAgent
from openbrowser.filesystem.file_system import FileSystem


def make_agent(**kwargs) -> CodeAgent:
    return CodeAgent(task="Test task", llm=MagicMock(), browser=MagicMock(), **kwargs)


def test_default_file_system_is_owned_by_one_agent() -> None:
    first = make_agent()
    second = make_agent()

    assert first.file_system.base_dir == first.agent_directory
    assert second.file_system.base_dir == second.agent_directory
    assert first.file_system.data_dir != second.file_system.data_dir
    assert first.file_system.data_dir.exists()
    assert second.file_system.data_dir.exists()


def test_explicit_file_system_is_preserved(tmp_path) -> None:
    file_system = FileSystem(base_dir=tmp_path)
    agent = make_agent(file_system=file_system)

    assert agent.file_system is file_system
    assert agent.file_system.base_dir == tmp_path
