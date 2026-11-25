"""Add direction labels to the historical comparison charts.

The original charts are preserved below a light banner. Values, series, error
bars, and source notes are not changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parents[1]
IMAGES = ROOT / "docs" / "images"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)
LABELS = {
    "benchmark_comparison.png": ("Duration (seconds) and API tokens. Lower is better ↓. Ideal is lower-left."),
    "benchmark_cost.png": "Cost per six-task run (USD). Lower is better ↓.",
    "benchmark_per_task_duration.png": "Duration (seconds). Lower is better ↓.",
    "benchmark_per_task_tokens.png": "API input tokens. Lower is better ↓.",
    "benchmark_response_size.png": "MCP response characters (log scale). Lower is better ↓.",
    "benchmark_token_breakdown.png": "API input and output tokens. Lower is better ↓.",
    "benchmark_tool_calls.png": "Tool calls per task. Lower is better ↓.",
}
FOOTER = "Historical run, 2026-02-21. Values unchanged."


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def annotate(path: Path, label: str) -> None:
    source = Image.open(path).convert("RGB")
    if source.getpixel((0, 0)) == (255, 255, 255):
        print(f"{path}: already labeled")
        return
    width, height = source.size
    font_size = max(22, width // 48)
    font = _font(font_size)
    footer_font = _font(max(16, width // 80))
    banner_height = font_size + 32
    footer_height = font_size // 2 + 26
    canvas = Image.new("RGB", (width, height + banner_height + footer_height), "#ffffff")
    canvas.paste(source, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, banner_height // 2), label, font=font, fill="#171717", anchor="lm")
    draw.text(
        (24, height + banner_height + footer_height // 2),
        FOOTER,
        font=footer_font,
        fill="#6b7280",
        anchor="lm",
    )
    canvas.save(path, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Only report the planned labels.")
    args = parser.parse_args()
    for name, label in LABELS.items():
        path = IMAGES / name
        if not path.is_file():
            raise SystemExit(f"Missing chart: {path}")
        if args.check:
            print(f"{name}: {label}")
        else:
            annotate(path, label)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
