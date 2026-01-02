"""File system views and models.

This module defines Pydantic models for file system operations,
including file information and state tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Information about a file or directory.

    Contains metadata about a file system entry, used when listing
    directory contents.

    Attributes:
        path: Relative path from the base directory.
        size: File size in bytes.
        modified_time: Last modification time as Unix timestamp.
        is_directory: True if this entry is a directory, False for files.

    Example:
        ```python
        info = FileInfo(
            path="data/output.json",
            size=1024,
            modified_time=1704067200.0,
            is_directory=False
        )
        ```
    """

    path: str
    size: int
    modified_time: float
    is_directory: bool = False


class FileSystemState(BaseModel):
    """State of the file system for persistence and tracking.

    Tracks which files have been created or modified during a session,
    along with configuration settings. Useful for auditing and cleanup.

    Attributes:
        base_path: Base directory path for file operations.
        created_files: List of relative paths to files created during session.
        modified_files: List of relative paths to files modified during session.
        allowed_extensions: List of permitted file extensions (with dots).

    Example:
        ```python
        state = FileSystemState(
            base_path="/tmp/agent_workspace",
            allowed_extensions=[".txt", ".json", ".csv"]
        )
        # After operations:
        print(f"Created: {state.created_files}")
        print(f"Modified: {state.modified_files}")
        ```
    """

    base_path: str = Field(default=".", description="Base path for file operations")
    created_files: list[str] = Field(default_factory=list, description="List of files created during session")
    modified_files: list[str] = Field(default_factory=list, description="List of files modified during session")
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".json", ".csv", ".pdf", ".py", ".js", ".html", ".css"],
        description="Allowed file extensions",
    )

