"""File system views and models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Information about a file."""

    path: str
    size: int
    modified_time: float
    is_directory: bool = False


class FileSystemState(BaseModel):
    """State of the file system for persistence."""

    base_path: str = Field(default=".", description="Base path for file operations")
    created_files: list[str] = Field(default_factory=list, description="List of files created during session")
    modified_files: list[str] = Field(default_factory=list, description="List of files modified during session")
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".json", ".csv", ".pdf", ".py", ".js", ".html", ".css"],
        description="Allowed file extensions",
    )

