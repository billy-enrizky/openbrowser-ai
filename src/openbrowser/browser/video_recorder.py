"""Video Recording Service for Browser Use Sessions.

This module provides video recording functionality for browser sessions using
CDP screencast capabilities and ffmpeg for encoding. It captures individual
frames from the browser and compiles them into a video file.

Requirements:
    Optional dependencies: imageio, imageio-ffmpeg, numpy
    Install with: pip install imageio imageio-ffmpeg numpy

Classes:
    VideoRecorderService: Handles frame capture and video encoding.

Functions:
    _get_padded_size: Calculates codec-compatible dimensions.

Example:
    >>> recorder = VideoRecorderService(
    ...     output_path=Path('/tmp/session.mp4'),
    ...     size=ViewportSize(width=1920, height=1080),
    ...     framerate=30
    ... )
    >>> recorder.start()
    >>> recorder.add_frame(base64_png_data)
    >>> recorder.stop_and_save()
"""

import base64
import logging
import math
import subprocess
from pathlib import Path
from typing import Optional

from src.openbrowser.browser.profile import ViewportSize

try:
    import imageio.v2 as iio  # type: ignore[import-not-found]
    import imageio_ffmpeg  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    from imageio.core.format import Format  # type: ignore[import-not-found]

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_padded_size(size: ViewportSize, macro_block_size: int = 16) -> ViewportSize:
    """Calculate dimensions padded to the nearest multiple of macro_block_size.

    Video codecs like H.264 require dimensions that are multiples of the
    macro block size (typically 16 pixels). This function rounds up.

    Args:
        size: Original viewport size.
        macro_block_size: Block size for codec compatibility (default: 16).

    Returns:
        ViewportSize with dimensions rounded up to nearest macro block multiple.

    Example:
        >>> _get_padded_size(ViewportSize(width=1920, height=1080))
        ViewportSize(width=1920, height=1088)  # 1080 rounds up to 1088
    """
    width = int(math.ceil(size['width'] / macro_block_size)) * macro_block_size
    height = int(math.ceil(size['height'] / macro_block_size)) * macro_block_size
    return ViewportSize(width=width, height=height)


class VideoRecorderService:
    """Handles video encoding for browser sessions using imageio.

    This service captures individual frames from CDP screencast, decodes them,
    and appends them to a video file using a pip-installable ffmpeg backend.
    It automatically resizes and pads frames to match codec requirements.

    The recording workflow is:
    1. Create instance with output path, size, and framerate
    2. Call start() to initialize the video writer
    3. Call add_frame() for each captured frame (base64 PNG)
    4. Call stop_and_save() to finalize the video file

    Attributes:
        output_path: Path where the video will be saved.
        size: Target video dimensions.
        framerate: Video framerate (frames per second).
        padded_size: Codec-compatible dimensions (multiple of 16).

    Example:
        >>> recorder = VideoRecorderService(
        ...     output_path=Path('/tmp/video.mp4'),
        ...     size=ViewportSize(width=1920, height=1080),
        ...     framerate=30
        ... )
        >>> recorder.start()
        >>> for frame_b64 in frames:
        ...     recorder.add_frame(frame_b64)
        >>> recorder.stop_and_save()
    """

    def __init__(self, output_path: Path, size: ViewportSize, framerate: int):
        """Initialize the video recorder.

        Args:
            output_path: The full path where the video will be saved.
                Parent directories will be created if needed.
            size: A ViewportSize specifying the width and height of the video.
            framerate: The desired framerate for the output video (fps).

        Example:
            >>> recorder = VideoRecorderService(
            ...     output_path=Path('/tmp/session.mp4'),
            ...     size=ViewportSize(width=1920, height=1080),
            ...     framerate=30
            ... )
        """
        self.output_path = output_path
        self.size = size
        self.framerate = framerate
        self._writer: Optional['Format.Writer'] = None
        self._is_active = False
        self.padded_size = _get_padded_size(self.size)

    def start(self) -> None:
        """Prepare and start the video writer.

        Initializes the imageio video writer with H.264 codec and yuv420p
        pixel format for maximum compatibility. Creates parent directories
        if needed.

        If required dependencies (imageio, imageio-ffmpeg, numpy) are not
        installed, logs an error and returns without starting.

        Note:
            Call this before add_frame(). The recorder is active after
            this method returns successfully (_is_active will be True).
        """
        if not IMAGEIO_AVAILABLE:
            logger.error(
                'MP4 recording requires optional dependencies. Please install them with: pip install imageio imageio-ffmpeg numpy'
            )
            return

        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # The macro_block_size is set to None because we handle padding ourselves
            self._writer = iio.get_writer(
                str(self.output_path),
                fps=self.framerate,
                codec='libx264',
                quality=8,  # A good balance of quality and file size (1-10 scale)
                pixelformat='yuv420p',  # Ensures compatibility with most players
                macro_block_size=None,
            )
            self._is_active = True
            logger.debug(f'[VideoRecorderService] Video recorder started. Output will be saved to {self.output_path}')
        except Exception as e:
            logger.error(f'[VideoRecorderService] Failed to initialize video writer: {e}')
            self._is_active = False

    def add_frame(self, frame_data_b64: str) -> None:
        """Decode, resize, pad, and append a frame to the video.

        Processes a base64-encoded PNG frame from CDP screencast:
        1. Decodes base64 to PNG bytes
        2. Scales to target video dimensions
        3. Pads to codec-compatible dimensions (macro block alignment)
        4. Appends to video file

        Args:
            frame_data_b64: A base64-encoded string of PNG frame data
                from CDP Page.screencastFrame.

        Note:
            Silently returns if recorder is not active. Logs warnings
            for individual frame processing failures without raising.
        """
        if not self._is_active or not self._writer:
            return

        try:
            frame_bytes = base64.b64decode(frame_data_b64)

            # Build a filter chain for ffmpeg:
            # 1. scale: Resizes the frame to the user-specified dimensions.
            # 2. pad: Adds black bars to meet codec's macro-block requirements,
            #    centering the original content.
            vf_chain = (
                f'scale={self.size["width"]}:{self.size["height"]},'
                f'pad={self.padded_size["width"]}:{self.padded_size["height"]}:(ow-iw)/2:(oh-ih)/2:color=black'
            )

            output_pix_fmt = 'rgb24'
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                '-f',
                'image2pipe',  # Input format from a pipe
                '-c:v',
                'png',  # Specify input codec is PNG
                '-i',
                '-',  # Input from stdin
                '-vf',
                vf_chain,  # Video filter for resizing and padding
                '-f',
                'rawvideo',  # Output format is raw video
                '-pix_fmt',
                output_pix_fmt,  # Output pixel format
                '-',  # Output to stdout
            ]

            # Execute ffmpeg as a subprocess
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate(input=frame_bytes)

            if proc.returncode != 0:
                err_msg = err.decode(errors='ignore').strip()
                if 'deprecated pixel format used' not in err_msg.lower():
                    raise OSError(f'ffmpeg error during resizing/padding: {err_msg}')
                else:
                    logger.debug(f'[VideoRecorderService] ffmpeg warning during resizing/padding: {err_msg}')

            # Convert the raw output bytes to a numpy array with the padded dimensions
            img_array = np.frombuffer(out, dtype=np.uint8).reshape((self.padded_size['height'], self.padded_size['width'], 3))

            self._writer.append_data(img_array)
        except Exception as e:
            logger.warning(f'[VideoRecorderService] Could not process and add video frame: {e}')

    def stop_and_save(self) -> None:
        """Finalize the video file by closing the writer.

        Writes any buffered frames and closes the video file. Should be
        called when the recording session is complete.

        After calling this method, the recorder is no longer active and
        add_frame() will have no effect.

        Note:
            This method is idempotent - safe to call multiple times.
            Logs success message with output path on completion.
        """
        if not self._is_active or not self._writer:
            return

        try:
            self._writer.close()
            logger.info(f'[VideoRecorderService] Video recording saved successfully to: {self.output_path}')
        except Exception as e:
            logger.error(f'[VideoRecorderService] Failed to finalize and save video: {e}')
        finally:
            self._is_active = False
            self._writer = None

