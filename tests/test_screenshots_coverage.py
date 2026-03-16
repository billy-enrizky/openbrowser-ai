"""Tests for openbrowser.screenshots.service module -- targeting 100% coverage."""

import base64
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


class TestScreenshotServiceInit:
    """Tests for ScreenshotService.__init__."""

    def test_init_creates_screenshots_dir(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        agent_dir = tmp_path / "agent_output"
        agent_dir.mkdir()
        svc = ScreenshotService(agent_directory=agent_dir)
        assert svc.screenshots_dir == agent_dir / "screenshots"
        assert svc.screenshots_dir.exists()

    def test_init_with_string_path(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        agent_dir = tmp_path / "agent_str"
        agent_dir.mkdir()
        svc = ScreenshotService(agent_directory=str(agent_dir))
        assert isinstance(svc.agent_directory, Path)
        assert svc.screenshots_dir.exists()

    def test_init_creates_parents(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        agent_dir = tmp_path / "deep" / "nested" / "agent"
        svc = ScreenshotService(agent_directory=agent_dir)
        assert svc.screenshots_dir.exists()


@pytest.mark.asyncio
class TestStoreScreenshot:
    """Tests for ScreenshotService.store_screenshot -- lines 34-38, 43-52."""

    async def test_store_screenshot_writes_file(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        svc = ScreenshotService(agent_directory=tmp_path)
        # Create a small PNG-like payload
        raw_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(raw_data).decode()

        result_path = await svc.store_screenshot(b64_data, step_number=1)

        assert result_path == str(svc.screenshots_dir / "step_1.png")
        assert Path(result_path).exists()
        assert Path(result_path).read_bytes() == raw_data

    async def test_store_screenshot_step_numbering(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        svc = ScreenshotService(agent_directory=tmp_path)
        raw = b"fake-png-data"
        b64 = base64.b64encode(raw).decode()

        for step in (0, 5, 99):
            path = await svc.store_screenshot(b64, step_number=step)
            assert f"step_{step}.png" in path


@pytest.mark.asyncio
class TestGetScreenshot:
    """Tests for ScreenshotService.get_screenshot -- lines 57-68."""

    async def test_get_screenshot_returns_base64(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        svc = ScreenshotService(agent_directory=tmp_path)
        raw = b"hello-screenshot-data"
        b64 = base64.b64encode(raw).decode()

        path = await svc.store_screenshot(b64, step_number=7)
        result = await svc.get_screenshot(path)
        assert result == b64

    async def test_get_screenshot_empty_path_returns_none(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        svc = ScreenshotService(agent_directory=tmp_path)
        result = await svc.get_screenshot("")
        assert result is None

    async def test_get_screenshot_nonexistent_returns_none(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        svc = ScreenshotService(agent_directory=tmp_path)
        result = await svc.get_screenshot("/nonexistent/path/step_1.png")
        assert result is None

    async def test_get_screenshot_reads_actual_file(self, tmp_path):
        from openbrowser.screenshots.service import ScreenshotService

        svc = ScreenshotService(agent_directory=tmp_path)
        # Manually write a file
        file_path = svc.screenshots_dir / "manual.png"
        raw = b"manual-data"
        file_path.write_bytes(raw)

        result = await svc.get_screenshot(str(file_path))
        assert result == base64.b64encode(raw).decode("utf-8")
