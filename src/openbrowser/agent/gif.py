"""GIF generation from agent history screenshots."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openbrowser.agent.views import AgentHistoryList

logger = logging.getLogger(__name__)


def create_history_gif(
    history: AgentHistoryList,
    output_path: str | Path = "agent_history.gif",
    duration: float = 1.0,
    add_step_annotations: bool = True,
    font_size: int = 24,
) -> Path:
    """
    Create a GIF from agent history screenshots.

    Args:
        history: Agent history containing screenshots
        output_path: Path to save the GIF
        duration: Duration per frame in seconds
        add_step_annotations: Whether to add step number annotations
        font_size: Font size for annotations

    Returns:
        Path to the saved GIF
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Please install pillow: pip install pillow")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect screenshots from history
    screenshots = history.screenshots()
    if not screenshots:
        raise ValueError("No screenshots found in agent history")

    logger.info(f"Creating GIF from {len(screenshots)} screenshots")

    frames = []
    for i, screenshot_b64 in enumerate(screenshots):
        if not screenshot_b64:
            continue

        # Decode base64 image
        try:
            image_data = base64.b64decode(screenshot_b64)
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary (GIF requires palette mode)
            if image.mode in ("RGBA", "P"):
                # Create white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "RGBA":
                    background.paste(image, mask=image.split()[3])
                else:
                    background.paste(image)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Add step annotation if requested
            if add_step_annotations:
                image = _add_step_annotation(image, i + 1, font_size)

            frames.append(image)

        except Exception as e:
            logger.warning(f"Failed to process screenshot {i}: {e}")
            continue

    if not frames:
        raise ValueError("No valid screenshots could be processed")

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 1000),  # Convert to milliseconds
        loop=0,  # Infinite loop
        optimize=True,
    )

    logger.info(f"GIF saved to {output_path}")
    return output_path


def _add_step_annotation(image: "Image.Image", step_number: int, font_size: int = 24) -> "Image.Image":
    """Add step number annotation to an image.
    
    Overlays a step number label on the top-left corner of the image
    with a semi-transparent background for visibility.
    
    Args:
        image: PIL Image to annotate.
        step_number: Step number to display.
        font_size: Font size for the annotation text.
        
    Returns:
        New PIL Image with the step annotation overlay.
    """
    from PIL import ImageDraw, ImageFont

    # Create a copy to avoid modifying original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Create annotation text
    text = f"Step {step_number}"

    # Try to use a system font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Draw background rectangle
    padding = 10
    x = 10
    y = 10
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180),
    )

    # Draw text
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return annotated


def create_gif_from_screenshots(
    screenshots: list[str],
    output_path: str | Path = "screenshots.gif",
    duration: float = 1.0,
) -> Path:
    """
    Create a GIF from a list of base64-encoded screenshots.

    Args:
        screenshots: List of base64-encoded screenshot strings
        output_path: Path to save the GIF
        duration: Duration per frame in seconds

    Returns:
        Path to the saved GIF
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Please install pillow: pip install pillow")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for screenshot_b64 in screenshots:
        if not screenshot_b64:
            continue

        try:
            image_data = base64.b64decode(screenshot_b64)
            image = Image.open(io.BytesIO(image_data))

            if image.mode in ("RGBA", "P"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "RGBA":
                    background.paste(image, mask=image.split()[3])
                else:
                    background.paste(image)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            frames.append(image)

        except Exception as e:
            logger.warning(f"Failed to process screenshot: {e}")
            continue

    if not frames:
        raise ValueError("No valid screenshots could be processed")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 1000),
        loop=0,
        optimize=True,
    )

    logger.info(f"GIF saved to {output_path}")
    return output_path

