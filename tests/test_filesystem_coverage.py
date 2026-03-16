"""Comprehensive tests for src/openbrowser/filesystem/file_system.py to cover remaining gaps.

Missing lines: 16-17, 40, 139-167, 170-171, 227, 279-280, 288-298, 303-306,
318-321, 332, 344-347, 361-364, 383-386, 454-455, 462, 505-517
"""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def file_system(temp_dir):
    """Create a FileSystem instance."""
    from openbrowser.filesystem.file_system import FileSystem

    return FileSystem(base_dir=temp_dir)


class TestFileSystemImports:
    """Test import fallbacks (lines 16-17)."""

    def test_reportlab_available(self):
        """Lines 16-17: REPORTLAB_AVAILABLE flag."""
        from openbrowser.filesystem.file_system import REPORTLAB_AVAILABLE

        assert isinstance(REPORTLAB_AVAILABLE, bool)


class TestBaseFile:
    """Test BaseFile and subclass functionality."""

    def test_extension_abstract(self):
        """Line 40: extension is abstract."""
        from openbrowser.filesystem.file_system import BaseFile

        with pytest.raises(TypeError):
            BaseFile(name="test")

    def test_markdown_file(self):
        """Test MarkdownFile."""
        from openbrowser.filesystem.file_system import MarkdownFile

        f = MarkdownFile(name="test")
        assert f.extension == "md"
        assert f.full_name == "test.md"

    def test_txt_file(self):
        """Test TxtFile."""
        from openbrowser.filesystem.file_system import TxtFile

        f = TxtFile(name="test")
        assert f.extension == "txt"

    def test_json_file(self):
        """Test JsonFile."""
        from openbrowser.filesystem.file_system import JsonFile

        f = JsonFile(name="test")
        assert f.extension == "json"

    def test_csv_file(self):
        """Test CsvFile."""
        from openbrowser.filesystem.file_system import CsvFile

        f = CsvFile(name="test")
        assert f.extension == "csv"

    def test_jsonl_file(self):
        """Test JsonlFile."""
        from openbrowser.filesystem.file_system import JsonlFile

        f = JsonlFile(name="test")
        assert f.extension == "jsonl"

    def test_file_properties(self):
        """Test get_size and get_line_count."""
        from openbrowser.filesystem.file_system import TxtFile

        f = TxtFile(name="test", content="line1\nline2\nline3")
        assert f.get_size == 17
        assert f.get_line_count == 3


class TestPdfFile:
    """Test PdfFile (lines 139-167, 170-171)."""

    def test_pdf_extension(self):
        """Test PdfFile extension."""
        from openbrowser.filesystem.file_system import PdfFile

        f = PdfFile(name="test")
        assert f.extension == "pdf"

    def test_pdf_sync_to_disk_no_reportlab(self):
        """Lines 137-138: PDF sync without reportlab."""
        from openbrowser.filesystem.file_system import PdfFile

        f = PdfFile(name="test", content="Hello")
        with patch("openbrowser.filesystem.file_system.REPORTLAB_AVAILABLE", False):
            with pytest.raises(Exception, match="reportlab"):
                f.sync_to_disk_sync(Path("/tmp"))

    def test_pdf_sync_to_disk_with_reportlab(self):
        """Lines 139-167: PDF sync with reportlab (mocked)."""
        from openbrowser.filesystem.file_system import PdfFile

        f = PdfFile(name="test", content="# Title\n## Subtitle\n### Section\nNormal text\n\nEmpty line above")

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_doc = MagicMock()
            mock_styles = MagicMock()
            mock_styles.__getitem__ = MagicMock(return_value=MagicMock())

            with patch("openbrowser.filesystem.file_system.REPORTLAB_AVAILABLE", True):
                with patch("openbrowser.filesystem.file_system.SimpleDocTemplate", return_value=mock_doc):
                    with patch("openbrowser.filesystem.file_system.getSampleStyleSheet", return_value=mock_styles):
                        with patch("openbrowser.filesystem.file_system.Paragraph", return_value=MagicMock()):
                            with patch("openbrowser.filesystem.file_system.Spacer", return_value=MagicMock()):
                                f.sync_to_disk_sync(Path(tmpdir))
                                mock_doc.build.assert_called_once()

    def test_pdf_sync_to_disk_error(self):
        """Lines 166-167: PDF sync error."""
        from openbrowser.filesystem.file_system import PdfFile, FileSystemError

        f = PdfFile(name="test", content="Hello")

        with patch("openbrowser.filesystem.file_system.REPORTLAB_AVAILABLE", True):
            with patch(
                "openbrowser.filesystem.file_system.SimpleDocTemplate",
                side_effect=Exception("build error"),
            ):
                with pytest.raises(FileSystemError, match="Could not write"):
                    f.sync_to_disk_sync(Path("/tmp"))

    @pytest.mark.asyncio
    async def test_pdf_async_sync_to_disk(self):
        """Lines 170-171: PDF async sync."""
        from openbrowser.filesystem.file_system import PdfFile

        f = PdfFile(name="test", content="Hello")

        with patch.object(PdfFile, "sync_to_disk_sync") as mock_sync:
            await f.sync_to_disk(Path("/tmp"))
            mock_sync.assert_called_once()


class TestFileSystem:
    """Test FileSystem class."""

    def test_init_creates_directories(self, temp_dir):
        """Test FileSystem creates directories on init."""
        from openbrowser.filesystem.file_system import FileSystem

        fs = FileSystem(base_dir=temp_dir)
        assert fs.data_dir.exists()

    def test_init_cleans_existing(self, temp_dir):
        """Test FileSystem cleans existing data dir."""
        from openbrowser.filesystem.file_system import FileSystem, DEFAULT_FILE_SYSTEM_PATH

        data_dir = Path(temp_dir) / DEFAULT_FILE_SYSTEM_PATH
        data_dir.mkdir(parents=True)
        (data_dir / "old_file.txt").write_text("old content")

        fs = FileSystem(base_dir=temp_dir)
        assert not (data_dir / "old_file.txt").exists()

    def test_default_files_created(self, file_system):
        """Line 227: default files created."""
        assert "todo.md" in file_system.list_files()

    def test_valid_filename(self, file_system):
        """Test _is_valid_filename."""
        assert file_system._is_valid_filename("test.md") is True
        assert file_system._is_valid_filename("test.py") is False
        assert file_system._is_valid_filename("") is False
        assert file_system._is_valid_filename("test") is False

    def test_get_file_invalid(self, file_system):
        """Test get_file with invalid filename."""
        assert file_system.get_file("!invalid.md") is None

    def test_get_file_not_found(self, file_system):
        """Test get_file for non-existent file."""
        assert file_system.get_file("nonexistent.md") is None

    @pytest.mark.asyncio
    async def test_read_file_external_text(self, file_system):
        """Lines 279-280, 288-298: read external text files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("external content")
            f.flush()

            result = await file_system.read_file(f.name, external_file=True)
            assert "external content" in result

    @pytest.mark.asyncio
    async def test_read_file_external_invalid_filename(self, file_system):
        """Lines 279-280: read external file with invalid format."""
        result = await file_system.read_file("no_extension", external_file=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_file_external_pdf(self, file_system):
        """Lines 288-298: read external PDF file."""
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF content"
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = await file_system.read_file("/tmp/test.pdf", external_file=True)
            assert "PDF content" in result

    @pytest.mark.asyncio
    async def test_read_file_external_unsupported(self, file_system):
        """Lines 303-306: read external unsupported extension."""
        result = await file_system.read_file("/tmp/test.xyz", external_file=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_file_external_not_found(self, file_system):
        """Line 303: read external file not found."""
        result = await file_system.read_file("/nonexistent/file.txt", external_file=True)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_file_external_permission_error(self, file_system):
        """Lines 303-306: read external permission denied."""
        with patch("anyio.open_file", side_effect=PermissionError("denied")):
            result = await file_system.read_file("/tmp/perm.txt", external_file=True)
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_file_external_generic_error(self, file_system):
        """Lines 305-306: read external generic error."""
        with patch("anyio.open_file", side_effect=Exception("generic error")):
            result = await file_system.read_file("/tmp/err.txt", external_file=True)
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_file_invalid_name(self, file_system):
        """Test read_file with invalid internal filename."""
        result = await file_system.read_file("!invalid.md")
        assert "Invalid" in result

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, file_system):
        """Test read_file for non-existent internal file."""
        result = await file_system.read_file("notfound.md")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_read_file_success(self, file_system):
        """Lines 318-319: read file success."""
        await file_system.write_file("test.md", "Hello world")
        result = await file_system.read_file("test.md")
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_read_file_filesystem_error(self, file_system):
        """Lines 318-321: read file raises FileSystemError."""
        from openbrowser.filesystem.file_system import FileSystemError, MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "read",
            side_effect=FileSystemError("test error"),
        ):
            result = await file_system.read_file("test.md")
            assert "test error" in result

    @pytest.mark.asyncio
    async def test_read_file_generic_error(self, file_system):
        """Lines 320-321: read file generic error."""
        from openbrowser.filesystem.file_system import MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "read",
            side_effect=Exception("generic error"),
        ):
            result = await file_system.read_file("test.md")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_write_file_invalid(self, file_system):
        """Test write_file with invalid filename."""
        result = await file_system.write_file("!invalid.md", "content")
        assert "Invalid" in result

    @pytest.mark.asyncio
    async def test_write_file_new(self, file_system):
        """Line 332: write new file."""
        result = await file_system.write_file("newfile.md", "Hello")
        assert "successfully" in result

    @pytest.mark.asyncio
    async def test_write_file_existing(self, file_system):
        """Test write to existing file."""
        await file_system.write_file("existing.md", "First")
        result = await file_system.write_file("existing.md", "Second")
        assert "successfully" in result

    @pytest.mark.asyncio
    async def test_write_file_filesystem_error(self, file_system):
        """Lines 344-347: write file raises FileSystemError."""
        from openbrowser.filesystem.file_system import FileSystemError, MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "write",
            side_effect=FileSystemError("write error"),
        ):
            result = await file_system.write_file("test.md", "new content")
            assert "write error" in result

    @pytest.mark.asyncio
    async def test_write_file_generic_error(self, file_system):
        """Lines 346-347: write file generic error."""
        from openbrowser.filesystem.file_system import MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "write",
            side_effect=Exception("generic"),
        ):
            result = await file_system.write_file("test.md", "new content")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_append_file_invalid(self, file_system):
        """Test append with invalid filename."""
        result = await file_system.append_file("!invalid.md", "content")
        assert "Invalid" in result

    @pytest.mark.asyncio
    async def test_append_file_not_found(self, file_system):
        """Test append to non-existent file."""
        result = await file_system.append_file("nofile.md", "content")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_append_file_success(self, file_system):
        """Lines 361-364: append success."""
        await file_system.write_file("test.md", "First")
        result = await file_system.append_file("test.md", " Second")
        assert "successfully" in result

    @pytest.mark.asyncio
    async def test_append_file_filesystem_error(self, file_system):
        """Lines 361-364: append raises FileSystemError."""
        from openbrowser.filesystem.file_system import FileSystemError, MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "append",
            side_effect=FileSystemError("append error"),
        ):
            result = await file_system.append_file("test.md", "more")
            assert "append error" in result

    @pytest.mark.asyncio
    async def test_append_file_generic_error(self, file_system):
        """Lines 363-364: append generic error."""
        from openbrowser.filesystem.file_system import MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "append",
            side_effect=Exception("generic"),
        ):
            result = await file_system.append_file("test.md", "more")
            assert "Error" in result

    @pytest.mark.asyncio
    async def test_replace_file_str_invalid(self, file_system):
        """Test replace with invalid filename."""
        result = await file_system.replace_file_str("!inv.md", "a", "b")
        assert "Invalid" in result

    @pytest.mark.asyncio
    async def test_replace_file_str_empty_old(self, file_system):
        """Test replace with empty old_str."""
        result = await file_system.replace_file_str("todo.md", "", "new")
        assert "Cannot replace empty" in result

    @pytest.mark.asyncio
    async def test_replace_file_str_not_found(self, file_system):
        """Test replace on non-existent file."""
        result = await file_system.replace_file_str("nofile.md", "a", "b")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_replace_file_str_success(self, file_system):
        """Lines 383-386: replace success."""
        await file_system.write_file("test.md", "Hello World")
        result = await file_system.replace_file_str("test.md", "World", "Test")
        assert "Successfully replaced" in result

    @pytest.mark.asyncio
    async def test_replace_file_str_filesystem_error(self, file_system):
        """Lines 383-386: replace raises FileSystemError."""
        from openbrowser.filesystem.file_system import FileSystemError, MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "write",
            side_effect=FileSystemError("replace error"),
        ):
            result = await file_system.replace_file_str("test.md", "content", "new")
            assert "replace error" in result

    @pytest.mark.asyncio
    async def test_replace_file_str_generic_error(self, file_system):
        """Lines 385-386: replace generic error."""
        from openbrowser.filesystem.file_system import MarkdownFile

        await file_system.write_file("test.md", "content")
        with patch.object(
            MarkdownFile,
            "write",
            side_effect=Exception("generic"),
        ):
            result = await file_system.replace_file_str("test.md", "content", "new")
            assert "Error" in result

    def test_describe_with_files(self, file_system):
        """Lines 454-455, 462: describe with files."""
        from openbrowser.filesystem.file_system import MarkdownFile

        # Add a file with content
        file_obj = MarkdownFile(name="report", content="Short content")
        file_system.files["report.md"] = file_obj

        result = file_system.describe()
        assert "report.md" in result

    def test_describe_empty_file(self, file_system):
        """Test describe with empty file."""
        from openbrowser.filesystem.file_system import MarkdownFile

        file_obj = MarkdownFile(name="empty", content="")
        file_system.files["empty.md"] = file_obj

        result = file_system.describe()
        assert "empty" in result

    def test_describe_large_file(self, file_system):
        """Lines 454-455: describe with large file (start/end preview)."""
        from openbrowser.filesystem.file_system import MarkdownFile

        content = "\n".join([f"Line {i}: " + "x" * 50 for i in range(100)])
        file_obj = MarkdownFile(name="large", content=content)
        file_system.files["large.md"] = file_obj

        result = file_system.describe()
        assert "large.md" in result
        assert "more lines" in result

    @pytest.mark.asyncio
    async def test_save_extracted_content(self, file_system):
        """Test save_extracted_content."""
        filename = await file_system.save_extracted_content("Extracted data")
        assert filename == "extracted_content_0.md"
        assert file_system.extracted_content_count == 1

    def test_get_todo_contents(self, file_system):
        """Test get_todo_contents."""
        result = file_system.get_todo_contents()
        assert isinstance(result, str)

    def test_get_todo_contents_no_file(self, temp_dir):
        """Test get_todo_contents when no todo file."""
        from openbrowser.filesystem.file_system import FileSystem

        fs = FileSystem(base_dir=temp_dir, create_default_files=False)
        result = fs.get_todo_contents()
        assert result == ""

    def test_get_state(self, file_system):
        """Test get_state serialization."""
        state = file_system.get_state()
        assert state.base_dir is not None
        assert "todo.md" in state.files

    def test_nuke(self, file_system):
        """Test nuke deletes directory."""
        assert file_system.data_dir.exists()
        file_system.nuke()
        assert not file_system.data_dir.exists()

    def test_from_state(self, file_system):
        """Lines 505-517: from_state restores file system."""
        from openbrowser.filesystem.file_system import FileSystem

        # Add some files
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(file_system.write_file("test.txt", "Hello"))
        finally:
            loop.close()

        state = file_system.get_state()
        file_system.nuke()

        # Restore from state
        restored = FileSystem.from_state(state)
        assert "todo.md" in restored.list_files()
        assert "test.txt" in restored.list_files()

    def test_from_state_all_file_types(self, temp_dir):
        """Lines 505-517: from_state with all file types."""
        from openbrowser.filesystem.file_system import (
            FileSystem,
            FileSystemState,
            MarkdownFile,
            TxtFile,
            JsonFile,
            JsonlFile,
            CsvFile,
            PdfFile,
        )

        state = FileSystemState(
            base_dir=temp_dir,
            extracted_content_count=2,
            files={
                "test.md": {"type": "MarkdownFile", "data": {"name": "test", "content": "md"}},
                "test.txt": {"type": "TxtFile", "data": {"name": "test", "content": "txt"}},
                "test.json": {"type": "JsonFile", "data": {"name": "test", "content": "{}"}},
                "test.jsonl": {"type": "JsonlFile", "data": {"name": "test", "content": "{}"}},
                "test.csv": {"type": "CsvFile", "data": {"name": "test", "content": "a,b"}},
                "test.pdf": {"type": "PdfFile", "data": {"name": "test", "content": "pdf"}},
                "test.unknown": {"type": "UnknownFile", "data": {"name": "test", "content": ""}},
            },
        )

        with patch.object(PdfFile, "sync_to_disk_sync"):
            restored = FileSystem.from_state(state)
            assert restored.extracted_content_count == 2
            assert "test.md" in restored.files
            assert "test.txt" in restored.files
            assert "test.json" in restored.files
            assert "test.jsonl" in restored.files
            assert "test.csv" in restored.files
            assert "test.pdf" in restored.files
            assert "test.unknown" not in restored.files  # Unknown types skipped
