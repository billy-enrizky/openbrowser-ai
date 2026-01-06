"""File system service for agent file operations.

This module provides the FileSystem class for controlled file system access
during browser automation. It includes security features like path validation
and extension filtering to prevent unauthorized access.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from openbrowser.filesystem.views import FileInfo, FileSystemState

logger = logging.getLogger(__name__)


class FileSystem:
    """File system service for agent file operations.

    Provides controlled file system access for browser automation agents,
    supporting reading, writing, and replacing file content. Includes
    security features to prevent unauthorized access.

    Security Features:
        - Path validation to prevent directory traversal attacks
        - Configurable base path confinement
        - Extension allowlist to limit file types
        - File size limits for read operations

    Attributes:
        base_path: Resolved absolute path to the working directory.
        state: FileSystemState tracking created/modified files.
        extracted_content_count: Counter for auto-numbered extracted files.

    Example:
        ```python
        fs = FileSystem(
            base_path="./output",
            allowed_extensions=[".json", ".csv", ".txt"]
        )

        # Write data
        fs.write_file("results.json", json.dumps(data))

        # Read file
        content = fs.read_file("config.txt")

        # List files
        files = fs.list_files(recursive=True)
        ```
    """

    def __init__(
        self,
        base_path: str | Path = ".",
        allowed_extensions: list[str] | None = None,
    ):
        """Initialize the FileSystem service.

        Args:
            base_path: Base directory path for file operations. All file
                paths are resolved relative to this directory. Defaults
                to the current working directory.
            allowed_extensions: List of permitted file extensions (with dots).
                Defaults to common text and code formats.

        Example:
            ```python
            # Use current directory
            fs = FileSystem()

            # Use specific directory with limited extensions
            fs = FileSystem(
                base_path="/tmp/agent_workspace",
                allowed_extensions=[".txt", ".json"]
            )
            ```
        """
        self.base_path = Path(base_path).resolve()
        self.state = FileSystemState(
            base_path=str(self.base_path),
            allowed_extensions=allowed_extensions or FileSystemState().allowed_extensions,
        )
        self.extracted_content_count = 0

    def _validate_path(self, file_path: str | Path) -> Path:
        """Validate and resolve a file path.

        Ensures the path is within the allowed base directory to prevent
        directory traversal attacks.

        Args:
            file_path: Path to validate (relative or absolute).

        Returns:
            Resolved absolute Path object.

        Raises:
            ValueError: If the resolved path is outside the base directory.
        """
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
        """Validate that a file extension is allowed.

        Args:
            path: Path object with extension to validate.

        Raises:
            ValueError: If the file extension is not in the allowed list.
        """
        suffix = path.suffix.lower()
        if suffix and suffix not in self.state.allowed_extensions:
            raise ValueError(f"File extension {suffix} is not allowed. Allowed: {self.state.allowed_extensions}")

    def read_file(self, file_path: str | Path, max_size: int = 1_000_000) -> str:
        """Read file content.

        Reads the content of a file, with special handling for different
        file types (PDF, JSON, text).

        Args:
            file_path: Path to the file (relative to base_path or absolute).
            max_size: Maximum file size in bytes. Defaults to 1MB.

        Returns:
            File content as a string. For JSON files, returns formatted JSON.
            For PDF files, returns extracted text.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file exceeds max_size or has disallowed extension.
            ImportError: If reading PDF files without pypdf installed.

        Example:
            ```python
            content = fs.read_file("data/config.json")
            pdf_text = fs.read_file("documents/report.pdf")
            ```
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
        """Read text content from a PDF file.

        Extracts text from all pages of a PDF document.

        Args:
            path: Path to the PDF file.

        Returns:
            Extracted text from all pages, separated by double newlines.

        Raises:
            ImportError: If pypdf is not installed.
        """
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
        """Write content to a file.

        Creates a new file or overwrites an existing file. Parent
        directories are created automatically if they don't exist.

        Args:
            file_path: Path to the file (relative to base_path or absolute).
            content: Content to write to the file.

        Returns:
            Success message indicating the number of characters written.

        Raises:
            ValueError: If the file extension is not allowed.

        Note:
            - JSON files are validated and pretty-printed
            - PDF files are created using reportlab (if installed)
            - Created files are tracked in state.created_files

        Example:
            ```python
            fs.write_file("output.json", json.dumps({"key": "value"}))
            fs.write_file("notes.txt", "Some notes here")
            ```
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
        """Write content as a PDF file using markdown conversion.

        Creates a simple PDF document with the text content. Lines are
        wrapped at 80 characters and automatically paginated.

        Args:
            path: Path where the PDF should be saved.
            content: Text content to write to the PDF.

        Returns:
            Success message with the file path.

        Raises:
            ImportError: If reportlab is not installed.
        """
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
        """Replace text in a file.

        Finds and replaces text content within an existing file.

        Args:
            file_path: Path to the file (relative to base_path or absolute).
            old_text: Text to find in the file.
            new_text: Replacement text.
            replace_all: If True, replace all occurrences. If False (default),
                replace only the first occurrence.

        Returns:
            Success message indicating the number of replacements made.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If old_text is not found in the file, or if the
                file extension is not allowed.

        Note:
            Modified files are tracked in state.modified_files.

        Example:
            ```python
            # Replace first occurrence
            fs.replace_in_file("config.json", '"debug": false', '"debug": true')

            # Replace all occurrences
            fs.replace_in_file("template.html", "{{name}}", "John", replace_all=True)
            ```
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
        """List files in a directory.

        Returns information about files and subdirectories within the
        specified directory.

        Args:
            directory: Directory path to list (relative to base_path or
                absolute). Defaults to base_path.
            recursive: If True, include files in subdirectories. If False
                (default), only list immediate children.

        Returns:
            List of FileInfo objects with file metadata.

        Raises:
            ValueError: If the path is not a directory or is outside base_path.

        Example:
            ```python
            # List current directory
            files = fs.list_files()

            # List subdirectory recursively
            all_files = fs.list_files("data", recursive=True)
            for f in all_files:
                print(f"{f.path}: {f.size} bytes")
            ```
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
        """Get file information for a path.

        Args:
            path: Absolute path to the file or directory.

        Returns:
            FileInfo object with path, size, modification time, and type.
        """
        stat = path.stat()
        return FileInfo(
            path=str(path.relative_to(self.base_path)),
            size=stat.st_size,
            modified_time=stat.st_mtime,
            is_directory=path.is_dir(),
        )

    def exists(self, file_path: str | Path) -> bool:
        """Check if a file exists.

        Args:
            file_path: Path to check (relative to base_path or absolute).

        Returns:
            True if the file exists and is within base_path, False otherwise.
        """
        try:
            path = self._validate_path(file_path)
            return path.exists()
        except ValueError:
            return False

    def get_state(self) -> FileSystemState:
        """Get the current file system state.

        Returns:
            FileSystemState object with base_path, created_files,
            modified_files, and allowed_extensions.
        """
        return self.state

    def get_dir(self) -> Path:
        """Get the base directory path.

        Returns:
            The resolved base directory Path object.
        """
        return self.base_path

    async def save_extracted_content(self, content: str) -> str:
        """Save extracted content to a numbered markdown file.

        Creates files with auto-incrementing names (extracted_content_0.md,
        extracted_content_1.md, etc.) for storing extracted data.

        Args:
            content: The content to save.

        Returns:
            The filename of the saved file (e.g., "extracted_content_0.md").

        Example:
            ```python
            filename = await fs.save_extracted_content("# Results\n...")
            # Returns "extracted_content_0.md"
            ```
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

        Reads and returns file content, intended for displaying results
        to the user or including in agent responses.

        Args:
            full_filename: Filename relative to base_path or absolute path.

        Returns:
            File content as a string, or None if the file doesn't exist
            or cannot be read.

        Example:
            ```python
            content = fs.display_file("extracted_content_0.md")
            if content:
                print(content)
            ```
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

