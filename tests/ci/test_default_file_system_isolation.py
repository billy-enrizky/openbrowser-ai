from __future__ import annotations

from unittest.mock import MagicMock

from openbrowser.code_use.service import CodeAgent


def test_default_code_agent_file_systems_do_not_share_a_directory() -> None:
    first = CodeAgent(task="first", llm=MagicMock(), browser=MagicMock())
    second = CodeAgent(task="second", llm=MagicMock(), browser=MagicMock())

    assert first.file_system.data_dir != second.file_system.data_dir
    assert first.file_system.data_dir.exists()
    assert second.file_system.data_dir.exists()
