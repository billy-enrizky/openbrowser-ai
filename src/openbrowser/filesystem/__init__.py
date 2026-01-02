"""File system module for agent file operations.

This module provides file system access for browser automation agents,
allowing controlled reading, writing, and modification of files during
automation tasks.

Key Components:
    FileSystem: Main service class for file operations.
    FileSystemState: Pydantic model tracking file system state.

Security Features:
    - Path validation to prevent directory traversal
    - Configurable allowed file extensions
    - Base path confinement

Example:
    ```python
    from src.openbrowser.filesystem import FileSystem

    fs = FileSystem(base_path="./output")
    fs.write_file("data.json", json.dumps(scraped_data))
    content = fs.read_file("config.txt")
    ```
"""

from .service import FileSystem
from .views import FileSystemState

__all__ = ["FileSystem", "FileSystemState"]

