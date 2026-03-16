"""Tests for openbrowser.filesystem.file_system module."""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openbrowser.filesystem.file_system import (
    BaseFile,
    CsvFile,
    FileSystem,
    FileSystemError,
    FileSystemState,
    JsonFile,
    JsonlFile,
    MarkdownFile,
    PdfFile,
    TxtFile,
    INVALID_FILENAME_ERROR_MESSAGE,
    DEFAULT_FILE_SYSTEM_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseFile subclass tests
# ---------------------------------------------------------------------------


class TestFileTypes:
    """Tests for file type classes."""

    def test_markdown_file_extension(self):
        f = MarkdownFile(name="test")
        assert f.extension == "md"
        assert f.full_name == "test.md"

    def test_txt_file_extension(self):
        f = TxtFile(name="test")
        assert f.extension == "txt"
        assert f.full_name == "test.txt"

    def test_json_file_extension(self):
        f = JsonFile(name="test")
        assert f.extension == "json"
        assert f.full_name == "test.json"

    def test_csv_file_extension(self):
        f = CsvFile(name="test")
        assert f.extension == "csv"
        assert f.full_name == "test.csv"

    def test_jsonl_file_extension(self):
        f = JsonlFile(name="test")
        assert f.extension == "jsonl"
        assert f.full_name == "test.jsonl"

    def test_pdf_file_extension(self):
        f = PdfFile(name="test")
        assert f.extension == "pdf"
        assert f.full_name == "test.pdf"


class TestBaseFileOperations:
    """Tests for BaseFile methods."""

    def test_write_file_content(self):
        f = MarkdownFile(name="test")
        f.write_file_content("# Hello")
        assert f.content == "# Hello"

    def test_append_file_content(self):
        f = MarkdownFile(name="test", content="Line 1\n")
        f.append_file_content("Line 2\n")
        assert f.content == "Line 1\nLine 2\n"

    def test_update_content(self):
        f = TxtFile(name="test")
        f.update_content("New content")
        assert f.content == "New content"

    def test_read(self):
        f = TxtFile(name="test", content="Hello World")
        assert f.read() == "Hello World"

    def test_get_size(self):
        f = TxtFile(name="test", content="12345")
        assert f.get_size == 5

    def test_get_line_count(self):
        f = TxtFile(name="test", content="line1\nline2\nline3")
        assert f.get_line_count == 3

    def test_sync_to_disk_sync(self, tmp_path):
        f = MarkdownFile(name="test", content="# Test")
        f.sync_to_disk_sync(tmp_path)
        file_path = tmp_path / "test.md"
        assert file_path.exists()
        assert file_path.read_text() == "# Test"

    @pytest.mark.asyncio
    async def test_write_async(self, tmp_path):
        f = TxtFile(name="test")
        await f.write("Hello Async", tmp_path)
        assert f.content == "Hello Async"
        file_path = tmp_path / "test.txt"
        assert file_path.exists()

    @pytest.mark.asyncio
    async def test_append_async(self, tmp_path):
        f = TxtFile(name="test", content="First\n")
        # First sync to disk so the file exists
        f.sync_to_disk_sync(tmp_path)
        await f.append("Second\n", tmp_path)
        assert f.content == "First\nSecond\n"


# ---------------------------------------------------------------------------
# FileSystem tests
# ---------------------------------------------------------------------------


class TestFileSystemInit:
    """Tests for FileSystem initialization."""

    def test_creates_directories(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.data_dir.exists()
        assert fs.data_dir == tmp_path / DEFAULT_FILE_SYSTEM_PATH

    def test_creates_default_files(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert "todo.md" in fs.files
        # Check that the file was synced to disk
        assert (fs.data_dir / "todo.md").exists()

    def test_no_default_files_when_disabled(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path, create_default_files=False)
        assert len(fs.files) == 0

    def test_cleans_existing_data_dir(self, tmp_path):
        # Create a data dir with some content
        data_dir = tmp_path / DEFAULT_FILE_SYSTEM_PATH
        data_dir.mkdir(parents=True)
        (data_dir / "old_file.txt").write_text("old content")

        # Creating FileSystem should clean the directory
        fs = FileSystem(base_dir=tmp_path)
        assert not (data_dir / "old_file.txt").exists()


class TestFileSystemValidation:
    """Tests for filename validation."""

    def test_valid_filenames(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs._is_valid_filename("test.md") is True
        assert fs._is_valid_filename("my-file_1.txt") is True
        assert fs._is_valid_filename("data.json") is True
        assert fs._is_valid_filename("report.csv") is True
        assert fs._is_valid_filename("log.jsonl") is True
        assert fs._is_valid_filename("doc.pdf") is True

    def test_invalid_filenames(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs._is_valid_filename("test") is False  # No extension
        assert fs._is_valid_filename("test.exe") is False  # Unsupported extension
        assert fs._is_valid_filename("test file.md") is False  # Space
        assert fs._is_valid_filename("") is False  # Empty string
        assert fs._is_valid_filename(".md") is False  # No name part

    def test_parse_filename(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        name, ext = fs._parse_filename("my-report.MD")
        assert name == "my-report"
        assert ext == "md"  # Lowercased


class TestFileSystemOperations:
    """Tests for FileSystem file operations."""

    def test_get_allowed_extensions(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        extensions = fs.get_allowed_extensions()
        assert "md" in extensions
        assert "txt" in extensions
        assert "json" in extensions
        assert "csv" in extensions
        assert "jsonl" in extensions
        assert "pdf" in extensions

    def test_get_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        # todo.md should exist by default
        f = fs.get_file("todo.md")
        assert f is not None
        assert f.full_name == "todo.md"

    def test_get_file_returns_none_for_invalid(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.get_file("bad name") is None

    def test_get_file_returns_none_for_nonexistent(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.get_file("nonexistent.md") is None

    def test_list_files(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        files = fs.list_files()
        assert "todo.md" in files

    def test_display_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.display_file("todo.md") is not None

    def test_display_file_invalid_returns_none(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.display_file("invalid name") is None

    def test_display_file_nonexistent_returns_none(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.display_file("nonexistent.md") is None

    def test_get_dir(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.get_dir() == fs.data_dir

    def test_get_todo_contents(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        contents = fs.get_todo_contents()
        assert isinstance(contents, str)

    def test_get_todo_contents_no_todo(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path, create_default_files=False)
        assert fs.get_todo_contents() == ""


class TestFileSystemReadWrite:
    """Tests for read_file and write_file."""

    @pytest.mark.asyncio
    async def test_write_file_creates_new(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.write_file("new_file.txt", "Hello World")
        assert "successfully" in result
        assert "new_file.txt" in fs.files

    @pytest.mark.asyncio
    async def test_write_file_overwrites_existing(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("test.txt", "First")
        await fs.write_file("test.txt", "Second")
        content = fs.files["test.txt"].read()
        assert content == "Second"

    @pytest.mark.asyncio
    async def test_write_file_invalid_name(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.write_file("bad name!", "content")
        assert result == INVALID_FILENAME_ERROR_MESSAGE

    @pytest.mark.asyncio
    async def test_read_file_success(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("test.md", "# Hello")
        result = await fs.read_file("test.md")
        assert "# Hello" in result
        assert "Read from file" in result

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.read_file("nonexistent.md")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_read_file_invalid_name(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.read_file("bad name!")
        assert result == INVALID_FILENAME_ERROR_MESSAGE

    @pytest.mark.asyncio
    async def test_read_external_file_txt(self, tmp_path):
        # Create an external file
        ext_file = tmp_path / "external.txt"
        ext_file.write_text("External content")

        fs = FileSystem(base_dir=tmp_path)
        result = await fs.read_file(str(ext_file), external_file=True)
        assert "External content" in result

    @pytest.mark.asyncio
    async def test_read_external_file_not_found(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.read_file("/nonexistent/path/file.txt", external_file=True)
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_read_external_file_unsupported_extension(self, tmp_path):
        ext_file = tmp_path / "file.exe"
        ext_file.write_text("binary")

        fs = FileSystem(base_dir=tmp_path)
        result = await fs.read_file(str(ext_file), external_file=True)
        assert "not supported" in result or "Invalid" in result


class TestFileSystemAppend:
    """Tests for append_file."""

    @pytest.mark.asyncio
    async def test_append_to_existing_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("test.txt", "Line 1\n")
        result = await fs.append_file("test.txt", "Line 2\n")
        assert "appended" in result
        content = fs.files["test.txt"].read()
        assert "Line 1\nLine 2\n" == content

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.append_file("nonexistent.txt", "content")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_append_invalid_name(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.append_file("bad name!", "content")
        assert result == INVALID_FILENAME_ERROR_MESSAGE


class TestFileSystemReplace:
    """Tests for replace_file_str."""

    @pytest.mark.asyncio
    async def test_replace_string(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("test.md", "Hello World")
        result = await fs.replace_file_str("test.md", "World", "Universe")
        assert "Successfully replaced" in result
        content = fs.files["test.md"].read()
        assert "Hello Universe" == content

    @pytest.mark.asyncio
    async def test_replace_empty_old_str_error(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("test.md", "Hello")
        result = await fs.replace_file_str("test.md", "", "New")
        assert "Cannot replace empty string" in result

    @pytest.mark.asyncio
    async def test_replace_nonexistent_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.replace_file_str("nope.md", "old", "new")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_replace_invalid_name(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        result = await fs.replace_file_str("bad name!", "old", "new")
        assert result == INVALID_FILENAME_ERROR_MESSAGE


class TestFileSystemSaveExtractedContent:
    """Tests for save_extracted_content."""

    @pytest.mark.asyncio
    async def test_saves_extracted_content(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        filename = await fs.save_extracted_content("Extracted data here")
        assert filename == "extracted_content_0.md"
        assert filename in fs.files
        assert fs.extracted_content_count == 1

    @pytest.mark.asyncio
    async def test_increments_counter(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        f1 = await fs.save_extracted_content("First")
        f2 = await fs.save_extracted_content("Second")
        assert f1 == "extracted_content_0.md"
        assert f2 == "extracted_content_1.md"
        assert fs.extracted_content_count == 2


class TestFileSystemDescribe:
    """Tests for describe method."""

    def test_empty_file_description(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        # Only has todo.md which is skipped in describe
        desc = fs.describe()
        assert "todo.md" not in desc

    @pytest.mark.asyncio
    async def test_describe_with_small_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("test.txt", "Short content")
        desc = fs.describe()
        assert "test.txt" in desc
        assert "Short content" in desc

    @pytest.mark.asyncio
    async def test_describe_with_large_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        # Create a large file that exceeds 1.5 * DISPLAY_CHARS
        large_content = "\n".join([f"Line {i}" for i in range(200)])
        await fs.write_file("large.txt", large_content)
        desc = fs.describe()
        assert "large.txt" in desc
        assert "more lines" in desc

    @pytest.mark.asyncio
    async def test_describe_with_empty_file(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("empty.txt", "")
        desc = fs.describe()
        assert "empty file" in desc


class TestFileSystemState:
    """Tests for state serialization/deserialization."""

    def test_get_state(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        state = fs.get_state()
        assert isinstance(state, FileSystemState)
        assert state.base_dir == str(tmp_path)
        assert "todo.md" in state.files

    def test_from_state(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        state = fs.get_state()

        # Restore from state in a different tmp directory
        fs2 = FileSystem.from_state(state)
        assert "todo.md" in fs2.files
        assert fs2.extracted_content_count == 0

    @pytest.mark.asyncio
    async def test_state_round_trip_preserves_content(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        await fs.write_file("test.md", "# My Doc")
        await fs.save_extracted_content("Extracted")

        state = fs.get_state()
        fs2 = FileSystem.from_state(state)

        assert fs2.files["test.md"].read() == "# My Doc"
        assert fs2.extracted_content_count == 1


class TestFileSystemNuke:
    """Tests for nuke method."""

    def test_nuke_removes_data_dir(self, tmp_path):
        fs = FileSystem(base_dir=tmp_path)
        assert fs.data_dir.exists()
        fs.nuke()
        assert not fs.data_dir.exists()


class TestPdfFileSync:
    """Tests for PdfFile sync_to_disk_sync."""

    def test_pdf_without_reportlab_raises(self, tmp_path):
        f = PdfFile(name="test", content="# Hello")
        with patch("openbrowser.filesystem.file_system.REPORTLAB_AVAILABLE", False):
            with pytest.raises(FileSystemError, match="reportlab"):
                f.sync_to_disk_sync(tmp_path)
