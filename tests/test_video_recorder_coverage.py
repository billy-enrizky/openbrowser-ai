"""Tests for openbrowser.browser.video_recorder module -- targeting 100% coverage.

Covers lines: 19-20, 27-29, 50-55, 64-85, 95-144, 152-162
"""

import base64
import logging
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tests for _get_padded_size (lines 27-29)
# ---------------------------------------------------------------------------
class TestGetPaddedSize:
    """Tests for _get_padded_size helper function."""

    def test_already_aligned(self):
        from openbrowser.browser.video_recorder import _get_padded_size

        result = _get_padded_size({"width": 1280, "height": 720})
        assert result["width"] == 1280
        assert result["height"] == 720

    def test_needs_padding(self):
        from openbrowser.browser.video_recorder import _get_padded_size

        result = _get_padded_size({"width": 1281, "height": 721})
        assert result["width"] == 1296  # ceil(1281/16)*16
        assert result["height"] == 736  # ceil(721/16)*16

    def test_small_values(self):
        from openbrowser.browser.video_recorder import _get_padded_size

        result = _get_padded_size({"width": 1, "height": 1})
        assert result["width"] == 16
        assert result["height"] == 16

    def test_custom_macro_block(self):
        from openbrowser.browser.video_recorder import _get_padded_size

        result = _get_padded_size({"width": 100, "height": 100}, macro_block_size=32)
        assert result["width"] == 128
        assert result["height"] == 128


# ---------------------------------------------------------------------------
# Tests for VideoRecorderService.__init__ (lines 50-55)
# ---------------------------------------------------------------------------
class TestVideoRecorderServiceInit:
    """Tests for VideoRecorderService constructor."""

    def test_init_attributes(self, tmp_path):
        from openbrowser.browser.video_recorder import VideoRecorderService

        output = tmp_path / "video.mp4"
        size = {"width": 1280, "height": 720}
        svc = VideoRecorderService(output_path=output, size=size, framerate=30)

        assert svc.output_path == output
        assert svc.size == size
        assert svc.framerate == 30
        assert svc._writer is None
        assert svc._is_active is False
        assert svc.padded_size["width"] == 1280
        assert svc.padded_size["height"] == 720

    def test_init_unaligned_size(self, tmp_path):
        from openbrowser.browser.video_recorder import VideoRecorderService

        output = tmp_path / "video.mp4"
        size = {"width": 1281, "height": 721}
        svc = VideoRecorderService(output_path=output, size=size, framerate=24)

        assert svc.padded_size["width"] % 16 == 0
        assert svc.padded_size["height"] % 16 == 0


# ---------------------------------------------------------------------------
# Tests for VideoRecorderService.start (lines 64-85)
# ---------------------------------------------------------------------------
class TestVideoRecorderServiceStart:
    """Tests for VideoRecorderService.start method."""

    def test_start_when_imageio_unavailable(self, tmp_path):
        """When IMAGEIO_AVAILABLE is False, start should log error and return."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 1280, "height": 720},
            framerate=30,
        )

        with patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", False):
            svc.start()

        assert svc._is_active is False
        assert svc._writer is None

    @patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", True)
    def test_start_success(self, tmp_path):
        """When imageio is available and writer creation succeeds."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()
        with patch("openbrowser.browser.video_recorder.iio") as mock_iio:
            mock_iio.get_writer.return_value = mock_writer

            svc = VideoRecorderService(
                output_path=tmp_path / "video.mp4",
                size={"width": 1280, "height": 720},
                framerate=30,
            )
            svc.start()

            assert svc._is_active is True
            assert svc._writer is mock_writer
            mock_iio.get_writer.assert_called_once()

    @patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", True)
    def test_start_writer_creation_fails(self, tmp_path):
        """When writer creation raises an exception."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        with patch("openbrowser.browser.video_recorder.iio") as mock_iio:
            mock_iio.get_writer.side_effect = RuntimeError("ffmpeg not found")

            svc = VideoRecorderService(
                output_path=tmp_path / "video.mp4",
                size={"width": 1280, "height": 720},
                framerate=30,
            )
            svc.start()

            assert svc._is_active is False
            assert svc._writer is None

    @patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", True)
    def test_start_creates_parent_directories(self, tmp_path):
        """start() should create parent directories for the output path."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()
        with patch("openbrowser.browser.video_recorder.iio") as mock_iio:
            mock_iio.get_writer.return_value = mock_writer

            deep_path = tmp_path / "deep" / "nested" / "video.mp4"
            svc = VideoRecorderService(
                output_path=deep_path,
                size={"width": 640, "height": 480},
                framerate=15,
            )
            svc.start()

            assert deep_path.parent.exists()
            assert svc._is_active is True


# ---------------------------------------------------------------------------
# Tests for VideoRecorderService.add_frame (lines 95-144)
# ---------------------------------------------------------------------------
class TestVideoRecorderServiceAddFrame:
    """Tests for VideoRecorderService.add_frame method."""

    def test_add_frame_when_not_active(self, tmp_path):
        """Should do nothing when not active."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 1280, "height": 720},
            framerate=30,
        )
        # Not started, should silently return
        svc.add_frame("dGVzdA==")
        assert svc._is_active is False

    def test_add_frame_when_no_writer(self, tmp_path):
        """Should do nothing when writer is None."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 1280, "height": 720},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = None
        svc.add_frame("dGVzdA==")

    @patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", True)
    def test_add_frame_success(self, tmp_path):
        """Successful frame addition with ffmpeg processing."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()

        # Create a small PNG image as frame data
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode()

        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 100, "height": 100},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = mock_writer

        # Mock subprocess.Popen
        mock_proc = MagicMock()
        # Create fake raw RGB output of correct size
        padded_w = svc.padded_size["width"]
        padded_h = svc.padded_size["height"]
        raw_out = b"\x00" * (padded_w * padded_h * 3)
        mock_proc.communicate.return_value = (raw_out, b"")
        mock_proc.returncode = 0

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("openbrowser.browser.video_recorder.imageio_ffmpeg") as mock_ffmpeg,
            patch("openbrowser.browser.video_recorder.np") as mock_np,
        ):
            mock_ffmpeg.get_ffmpeg_exe.return_value = "/usr/bin/ffmpeg"
            fake_array = MagicMock()
            mock_np.frombuffer.return_value.reshape.return_value = fake_array

            svc.add_frame(frame_b64)

            mock_writer.append_data.assert_called_once_with(fake_array)

    @patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", True)
    def test_add_frame_ffmpeg_error(self, tmp_path):
        """Frame add with ffmpeg returning non-zero exit code."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()
        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 100, "height": 100},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = mock_writer

        frame_b64 = base64.b64encode(b"fake-png").decode()

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"", b"some ffmpeg error")
        mock_proc.returncode = 1

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("openbrowser.browser.video_recorder.imageio_ffmpeg") as mock_ffmpeg,
        ):
            mock_ffmpeg.get_ffmpeg_exe.return_value = "/usr/bin/ffmpeg"
            # The OSError is caught internally; add_frame should not raise
            svc.add_frame(frame_b64)

        # Writer should NOT have been called because the error was raised before append
        mock_writer.append_data.assert_not_called()

    @patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", True)
    def test_add_frame_ffmpeg_deprecated_pixel_format_warning(self, tmp_path):
        """Frame add with ffmpeg returning deprecated pixel format warning."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()
        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 100, "height": 100},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = mock_writer

        frame_b64 = base64.b64encode(b"fake-png").decode()

        padded_w = svc.padded_size["width"]
        padded_h = svc.padded_size["height"]
        raw_out = b"\x00" * (padded_w * padded_h * 3)

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (raw_out, b"deprecated pixel format used, make sure you did set range correctly")
        mock_proc.returncode = 1  # non-zero but contains deprecated pixel format

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("openbrowser.browser.video_recorder.imageio_ffmpeg") as mock_ffmpeg,
            patch("openbrowser.browser.video_recorder.np") as mock_np,
        ):
            mock_ffmpeg.get_ffmpeg_exe.return_value = "/usr/bin/ffmpeg"
            fake_array = MagicMock()
            mock_np.frombuffer.return_value.reshape.return_value = fake_array

            svc.add_frame(frame_b64)

            mock_writer.append_data.assert_called_once_with(fake_array)

    @patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", True)
    def test_add_frame_exception_caught(self, tmp_path):
        """General exception during add_frame is caught."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()
        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 100, "height": 100},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = mock_writer

        with patch("base64.b64decode", side_effect=Exception("decode error")):
            # Should not raise
            svc.add_frame("not-valid-b64")


# ---------------------------------------------------------------------------
# Tests for VideoRecorderService.stop_and_save (lines 152-162)
# ---------------------------------------------------------------------------
class TestVideoRecorderServiceStopAndSave:
    """Tests for VideoRecorderService.stop_and_save method."""

    def test_stop_when_not_active(self, tmp_path):
        """Should do nothing when not active."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 1280, "height": 720},
            framerate=30,
        )
        svc.stop_and_save()
        assert svc._is_active is False

    def test_stop_when_no_writer(self, tmp_path):
        """Should do nothing when writer is None."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 1280, "height": 720},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = None
        svc.stop_and_save()

    def test_stop_success(self, tmp_path):
        """Successful stop closes the writer."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()
        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 1280, "height": 720},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = mock_writer

        svc.stop_and_save()

        mock_writer.close.assert_called_once()
        assert svc._is_active is False
        assert svc._writer is None

    def test_stop_close_fails(self, tmp_path):
        """If writer.close() raises, still cleans up."""
        from openbrowser.browser.video_recorder import VideoRecorderService

        mock_writer = MagicMock()
        mock_writer.close.side_effect = RuntimeError("close failed")

        svc = VideoRecorderService(
            output_path=tmp_path / "video.mp4",
            size={"width": 1280, "height": 720},
            framerate=30,
        )
        svc._is_active = True
        svc._writer = mock_writer

        svc.stop_and_save()

        # Finally block still cleans up
        assert svc._is_active is False
        assert svc._writer is None


# ---------------------------------------------------------------------------
# Test IMAGEIO_AVAILABLE flag (lines 19-20)
# ---------------------------------------------------------------------------
class TestImageioAvailableFlag:
    """Test the IMAGEIO_AVAILABLE import flag behavior."""

    def test_imageio_available_is_bool(self):
        from openbrowser.browser.video_recorder import IMAGEIO_AVAILABLE

        assert isinstance(IMAGEIO_AVAILABLE, bool)

    def test_imageio_unavailable_path(self):
        """Test that when imageio is not importable, IMAGEIO_AVAILABLE is False."""
        import importlib
        import sys

        # We cannot actually un-import it reliably, but we test the flag
        # exists and works with our mock
        with patch.dict(sys.modules, {"imageio": None, "imageio.v2": None}):
            # The module is already loaded, just verify the flag path works
            from openbrowser.browser.video_recorder import VideoRecorderService

            svc = VideoRecorderService(
                output_path=Path("/tmp/test.mp4"),
                size={"width": 640, "height": 480},
                framerate=24,
            )
            with patch("openbrowser.browser.video_recorder.IMAGEIO_AVAILABLE", False):
                svc.start()
                assert svc._is_active is False
