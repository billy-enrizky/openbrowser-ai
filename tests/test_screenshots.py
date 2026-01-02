"""Tests for the screenshot service module.

This module provides comprehensive test coverage for the ScreenshotService
class, which manages browser screenshot capture, storage, and retrieval.
It validates:

    - Service initialization with default and custom save paths
    - Screenshot addition, counting, and retrieval (latest, by step, all)
    - Screenshot clearing and path listing
    - Base64 encoding and decoding for screenshot data
    - File-based screenshot storage with configurable format and quality

The screenshot service enables visual debugging, history tracking, and
evidence collection during browser automation tasks.
"""

import pytest
import base64
import tempfile
from pathlib import Path

from src.openbrowser.screenshots import ScreenshotService


class TestScreenshotService:
    """Tests for the ScreenshotService class.

    Validates screenshot management functionality including storage,
    retrieval, encoding/decoding, and file system operations.
    """

    def test_screenshot_service_init(self):
        """Test ScreenshotService initialization."""
        service = ScreenshotService()
        assert service.save_path is None
        assert service.format == "png"
        assert service.quality == 90
        assert service._screenshots == []

    def test_screenshot_service_with_save_path(self):
        """Test ScreenshotService with save path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ScreenshotService(save_path=tmpdir)
            assert service.save_path == Path(tmpdir)

    def test_add_screenshot(self):
        """Test adding a screenshot."""
        service = ScreenshotService()
        
        # Create a simple test image (1x1 red PNG)
        test_image = base64.b64encode(b"test_image_data").decode()
        service.add_screenshot(test_image)
        
        assert service.count() == 1
        assert service.get_latest() == test_image

    def test_get_screenshots(self):
        """Test getting all screenshots."""
        service = ScreenshotService()
        
        service.add_screenshot("screenshot1")
        service.add_screenshot("screenshot2")
        
        screenshots = service.get_screenshots()
        assert len(screenshots) == 2
        assert screenshots[0] == "screenshot1"
        assert screenshots[1] == "screenshot2"

    def test_get_by_step(self):
        """Test getting screenshot by step."""
        service = ScreenshotService()
        
        service.add_screenshot("step0")
        service.add_screenshot("step1")
        service.add_screenshot("step2")
        
        assert service.get_by_step(0) == "step0"
        assert service.get_by_step(1) == "step1"
        assert service.get_by_step(2) == "step2"
        assert service.get_by_step(99) is None

    def test_clear(self):
        """Test clearing screenshots."""
        service = ScreenshotService()
        
        service.add_screenshot("test")
        assert service.count() == 1
        
        service.clear()
        assert service.count() == 0

    def test_decode_encode(self):
        """Test decode and encode methods."""
        original = b"test image bytes"
        encoded = ScreenshotService.encode_from_bytes(original)
        decoded = ScreenshotService.decode_to_bytes(encoded)
        
        assert decoded == original

    def test_get_screenshot_paths(self):
        """Test getting screenshot paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ScreenshotService(save_path=tmpdir)
            
            # Create some test files
            (Path(tmpdir) / "step_0000.png").write_bytes(b"test")
            (Path(tmpdir) / "step_0001.png").write_bytes(b"test")
            
            paths = service.get_screenshot_paths()
            assert len(paths) == 2

