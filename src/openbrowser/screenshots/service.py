"""Screenshot service for storage and management."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openbrowser.browser.session import BrowserSession, CDPSession

logger = logging.getLogger(__name__)


class ScreenshotService:
    """
    Service for managing screenshots during agent execution.
    Handles capture, storage, and retrieval of screenshots.
    """

    def __init__(
        self,
        save_path: str | Path | None = None,
        format: str = "png",
        quality: int = 90,
    ):
        """
        Initialize screenshot service.

        Args:
            save_path: Directory to save screenshots (None for in-memory only)
            format: Image format (png or jpeg)
            quality: JPEG quality (1-100)
        """
        self.save_path = Path(save_path) if save_path else None
        self.format = format
        self.quality = quality
        self._screenshots: list[str] = []  # Base64 encoded screenshots
        self._step_counter = 0

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

    async def take_screenshot(
        self,
        cdp_session: CDPSession,
        step_number: int | None = None,
        save: bool = True,
    ) -> str:
        """
        Take a screenshot of the current page.

        Args:
            cdp_session: CDP session to use
            step_number: Optional step number for naming
            save: Whether to save to disk

        Returns:
            Base64-encoded screenshot data
        """
        try:
            result = await cdp_session.cdp_client.send.Page.captureScreenshot(
                params={
                    "format": self.format,
                    "quality": self.quality if self.format == "jpeg" else None,
                },
                session_id=cdp_session.session_id,
            )

            screenshot_b64 = result.get("data", "")

            if not screenshot_b64:
                logger.warning("Empty screenshot data received")
                return ""

            # Store in memory
            self._screenshots.append(screenshot_b64)

            # Save to disk if configured
            if save and self.save_path:
                step = step_number if step_number is not None else self._step_counter
                self._step_counter += 1
                file_path = self.save_path / f"step_{step:04d}.{self.format}"
                self._save_to_file(screenshot_b64, file_path)
                logger.debug(f"Screenshot saved to {file_path}")

            return screenshot_b64

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return ""

    def _save_to_file(self, screenshot_b64: str, file_path: Path) -> None:
        """Save base64 screenshot to file."""
        try:
            image_data = base64.b64decode(screenshot_b64)
            with open(file_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            logger.error(f"Failed to save screenshot to {file_path}: {e}")

    def add_screenshot(self, screenshot_b64: str) -> None:
        """Add a screenshot to the collection."""
        self._screenshots.append(screenshot_b64)

    def get_screenshots(self) -> list[str]:
        """Get all collected screenshots."""
        return self._screenshots.copy()

    def get_latest(self) -> str | None:
        """Get the latest screenshot."""
        return self._screenshots[-1] if self._screenshots else None

    def get_by_step(self, step: int) -> str | None:
        """Get screenshot by step number."""
        if 0 <= step < len(self._screenshots):
            return self._screenshots[step]
        return None

    def count(self) -> int:
        """Get the number of screenshots."""
        return len(self._screenshots)

    def clear(self) -> None:
        """Clear all screenshots from memory."""
        self._screenshots.clear()
        self._step_counter = 0

    def get_screenshot_paths(self) -> list[Path]:
        """Get paths to all saved screenshots."""
        if not self.save_path:
            return []

        paths = sorted(self.save_path.glob(f"*.{self.format}"))
        return paths

    @staticmethod
    def decode_to_bytes(screenshot_b64: str) -> bytes:
        """Decode base64 screenshot to bytes."""
        return base64.b64decode(screenshot_b64)

    @staticmethod
    def encode_from_bytes(image_bytes: bytes) -> str:
        """Encode bytes to base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")

    def resize_screenshot(
        self,
        screenshot_b64: str,
        width: int,
        height: int,
    ) -> str:
        """
        Resize a screenshot.

        Args:
            screenshot_b64: Base64-encoded screenshot
            width: Target width
            height: Target height

        Returns:
            Resized base64-encoded screenshot
        """
        try:
            from PIL import Image

            image_data = base64.b64decode(screenshot_b64)
            image = Image.open(io.BytesIO(image_data))
            resized = image.resize((width, height), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            resized.save(buffer, format=self.format.upper())
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        except ImportError:
            logger.warning("PIL not installed, cannot resize screenshot")
            return screenshot_b64
        except Exception as e:
            logger.error(f"Failed to resize screenshot: {e}")
            return screenshot_b64

