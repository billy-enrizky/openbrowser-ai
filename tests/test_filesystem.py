"""Tests for the filesystem module.

This module provides comprehensive test coverage for the filesystem
subsystem, which provides secure file operations for browser automation
agents. It validates:

    - FileSystem: Core file operations (read, write, replace, list, exists)
    - FileSystemState: State tracking for created and modified files
    - Path security validation to prevent directory traversal attacks
    - Extension filtering for allowed file types
    - Automatic parent directory creation during write operations
    - State tracking of file operations for audit and rollback

The filesystem module enables agents to safely interact with the local
file system while maintaining security boundaries and operation logs.
"""

import pytest
import tempfile
from pathlib import Path

from src.openbrowser.filesystem import FileSystem, FileSystemState


class TestFileSystem:
    """Tests for the FileSystem class.

    Validates file operations including reading, writing, replacing,
    listing, and security features like path validation and extension filtering.
    """

    def test_filesystem_init(self):
        """Test FileSystem initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            assert fs.base_path == Path(tmpdir).resolve()
            assert fs.state.base_path == str(Path(tmpdir).resolve())

    def test_write_and_read_file(self):
        """Test writing and reading a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            
            # Write file
            result = fs.write_file("test.txt", "Hello, World!")
            assert "Successfully wrote" in result
            
            # Read file
            content = fs.read_file("test.txt")
            assert content == "Hello, World!"

    def test_write_creates_directories(self):
        """Test that write creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            
            fs.write_file("subdir/nested/test.txt", "Content")
            content = fs.read_file("subdir/nested/test.txt")
            assert content == "Content"

    def test_replace_in_file(self):
        """Test replacing text in a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            
            fs.write_file("test.txt", "Hello, World!")
            fs.replace_in_file("test.txt", "World", "Python")
            content = fs.read_file("test.txt")
            assert content == "Hello, Python!"

    def test_file_exists(self):
        """Test file exists check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            
            assert fs.exists("nonexistent.txt") is False
            fs.write_file("exists.txt", "Content")
            assert fs.exists("exists.txt") is True

    def test_list_files(self):
        """Test listing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            
            fs.write_file("file1.txt", "Content 1")
            fs.write_file("file2.txt", "Content 2")
            
            files = fs.list_files()
            file_names = [f.path for f in files]
            assert "file1.txt" in file_names
            assert "file2.txt" in file_names

    def test_path_validation_security(self):
        """Test that paths outside base_path are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            
            with pytest.raises(ValueError, match="outside allowed base path"):
                fs.read_file("/etc/passwd")

    def test_extension_validation(self):
        """Test that disallowed extensions are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir, allowed_extensions=[".txt"])
            
            with pytest.raises(ValueError, match="not allowed"):
                fs.write_file("test.exe", "Content")

    def test_state_tracking(self):
        """Test that file operations are tracked in state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(base_path=tmpdir)
            
            fs.write_file("new.txt", "Content")
            assert "new.txt" in fs.state.created_files
            
            fs.replace_in_file("new.txt", "Content", "New Content")
            assert "new.txt" in fs.state.modified_files


class TestFileSystemState:
    """Tests for the FileSystemState class.

    Validates state tracking for file operations including default values
    and lists of created and modified files.
    """

    def test_state_defaults(self):
        """Test FileSystemState defaults."""
        state = FileSystemState()
        assert state.base_path == "."
        assert state.created_files == []
        assert state.modified_files == []
        assert ".txt" in state.allowed_extensions

