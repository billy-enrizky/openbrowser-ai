"""File system service for agent file operations."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from src.openbrowser.filesystem.views import FileInfo, FileSystemState

logger = logging.getLogger(__name__)


class FileSystem:
    """
    File system service for agent file operations.
    Supports reading, writing, and replacing file content.
    """

    def __init__(
        self,
        base_path: str | Path = ".",
        allowed_extensions: list[str] | None = None,
    ):
        self.base_path = Path(base_path).resolve()
        self.state = FileSystemState(
            base_path=str(self.base_path),
            allowed_extensions=allowed_extensions or FileSystemState().allowed_extensions,
        )
        self.extracted_content_count = 0

    def _validate_path(self, file_path: str | Path) -> Path:
        """Validate and resolve file path."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.base_path / path
        path = path.resolve()

        # Security check - ensure path is within base_path
        try:
            path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(f"Path {path} is outside allowed base path {self.base_path}")

        return path

    def _validate_extension(self, path: Path) -> None:
        """Validate file extension is allowed."""
        suffix = path.suffix.lower()
        if suffix and suffix not in self.state.allowed_extensions:
            raise ValueError(f"File extension {suffix} is not allowed. Allowed: {self.state.allowed_extensions}")

    def read_file(self, file_path: str | Path, max_size: int = 1_000_000) -> str:
        """
        Read file content.

        Args:
            file_path: Path to the file
            max_size: Maximum file size in bytes (default 1MB)

        Returns:
            File content as string
        """
        path = self._validate_path(file_path)
        self._validate_extension(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.stat().st_size > max_size:
            raise ValueError(f"File too large: {path.stat().st_size} bytes (max: {max_size})")

        logger.info(f"Reading file: {path}")

        # Handle different file types
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    def _read_pdf(self, path: Path) -> str:
        """Read PDF file content."""
        try:
            import pypdf
        except ImportError:
            raise ImportError("Please install pypdf: pip install pypdf")

        reader = pypdf.PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)

    def write_file(self, file_path: str | Path, content: str) -> str:
        """
        Write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write

        Returns:
            Success message
        """
        path = self._validate_path(file_path)
        self._validate_extension(path)

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing file: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._write_pdf(path, content)
        elif suffix == ".json":
            # Validate JSON
            try:
                data = json.loads(content)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except json.JSONDecodeError:
                # Write as plain text if not valid JSON
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        # Track created file
        relative_path = str(path.relative_to(self.base_path))
        if relative_path not in self.state.created_files:
            self.state.created_files.append(relative_path)

        return f"Successfully wrote {len(content)} characters to {path}"

    def _write_pdf(self, path: Path, content: str) -> str:
        """Write content as PDF using markdown conversion."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
        except ImportError:
            raise ImportError("Please install reportlab: pip install reportlab")

        c = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter

        # Simple text rendering
        y = height - inch
        for line in content.split("\n"):
            if y < inch:
                c.showPage()
                y = height - inch
            c.drawString(inch, y, line[:80])  # Truncate long lines
            y -= 14  # Line spacing

        c.save()
        return f"Successfully wrote PDF to {path}"

    def replace_in_file(
        self,
        file_path: str | Path,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        """
        Replace text in a file.

        Args:
            file_path: Path to the file
            old_text: Text to find
            new_text: Replacement text
            replace_all: Replace all occurrences (default: first only)

        Returns:
            Success message
        """
        path = self._validate_path(file_path)
        self._validate_extension(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_text not in content:
            raise ValueError(f"Text not found in file: {old_text[:50]}...")

        if replace_all:
            new_content = content.replace(old_text, new_text)
            count = content.count(old_text)
        else:
            new_content = content.replace(old_text, new_text, 1)
            count = 1

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # Track modified file
        relative_path = str(path.relative_to(self.base_path))
        if relative_path not in self.state.modified_files:
            self.state.modified_files.append(relative_path)

        logger.info(f"Replaced {count} occurrence(s) in {path}")
        return f"Replaced {count} occurrence(s) in {path}"

    def list_files(
        self,
        directory: str | Path = ".",
        recursive: bool = False,
    ) -> list[FileInfo]:
        """
        List files in a directory.

        Args:
            directory: Directory path
            recursive: Include subdirectories

        Returns:
            List of FileInfo objects
        """
        path = self._validate_path(directory)

        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        files = []
        if recursive:
            for item in path.rglob("*"):
                if item.is_file():
                    files.append(self._get_file_info(item))
        else:
            for item in path.iterdir():
                files.append(self._get_file_info(item))

        return files

    def _get_file_info(self, path: Path) -> FileInfo:
        """Get file information."""
        stat = path.stat()
        return FileInfo(
            path=str(path.relative_to(self.base_path)),
            size=stat.st_size,
            modified_time=stat.st_mtime,
            is_directory=path.is_dir(),
        )

    def exists(self, file_path: str | Path) -> bool:
        """Check if file exists."""
        try:
            path = self._validate_path(file_path)
            return path.exists()
        except ValueError:
            return False

    def get_state(self) -> FileSystemState:
        """Get current file system state."""
        return self.state

    def get_dir(self) -> Path:
        """Get the base directory path."""
        return self.base_path

    async def save_extracted_content(self, content: str) -> str:
        """Save extracted content to a numbered markdown file.

        Args:
            content: Content to save

        Returns:
            Filename of the saved file
        """
        filename = f'extracted_content_{self.extracted_content_count}.md'
        file_path = self.base_path / filename
        
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.extracted_content_count += 1
        
        # Track created file
        relative_path = str(file_path.relative_to(self.base_path))
        if relative_path not in self.state.created_files:
            self.state.created_files.append(relative_path)
        
        logger.info(f"Saved extracted content to {filename}")
        return filename

    def display_file(self, full_filename: str) -> str | None:
        """Display file content for inclusion in done action.

        Args:
            full_filename: Filename (relative to base_path or absolute)

        Returns:
            File content as string, or None if file not found
        """
        try:
            path = self._validate_path(full_filename)
            if not path.exists():
                return None
            
            # Read file content
            return self.read_file(path)
        except (ValueError, FileNotFoundError):
            return None
        except Exception as e:
            logger.warning(f"Error displaying file {full_filename}: {e}")
            return None

