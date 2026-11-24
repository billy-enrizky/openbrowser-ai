"""Render asset policy for a browser profile.

`RenderMode` is a capability preset: full, lightweight, or no-render. The
granular flags below change only which resource types reach the network, so a
caller can keep screenshots and still skip external CSS.

The module holds no imports from the browser package, so both the profile and
the session can use it without a cycle.
"""

from __future__ import annotations

from typing import Final

# Resource types the render mode keeps off the network. Stylesheet is only
# blocked for no-render; lightweight keeps CSS so layout and visibility match.
LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES: Final[tuple[str, ...]] = ("Image", "Font", "Media", "TextTrack")
NO_RENDER_BLOCKED_RESOURCE_TYPES: Final[tuple[str, ...]] = (*LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES, "Stylesheet")
RESOURCE_TYPE_ORDER: Final[tuple[str, ...]] = (
    "Image",
    "Font",
    "Media",
    "TextTrack",
    "Stylesheet",
    "Script",
)

TEXT_ONLY_RESOURCE_TYPES: Final[tuple[str, ...]] = (
    "Image",
    "Font",
    "Media",
    "TextTrack",
    "Stylesheet",
)

# Chrome launch args that match each blocked type. Fetch interception is the
# authoritative gate; these stop Blink from creating or decoding the asset.
CHROME_BLOCK_ARGS: Final[dict[str, tuple[str, ...]]] = {
    "Image": ("--blink-settings=imagesEnabled=false",),
    "Font": ("--disable-remote-fonts",),
    "Media": (
        "--mute-audio",
        "--autoplay-policy=user-gesture-required",
        "--disable-accelerated-video-decode",
    ),
    "TextTrack": (),
    "Stylesheet": (),
    "Script": (),
}


def blocked_resource_types_for_render_mode(mode: object) -> tuple[str, ...]:
    """Return the CDP resource types a render mode must not fetch."""
    value = getattr(mode, "value", mode)
    if value == "no-render":
        return NO_RENDER_BLOCKED_RESOURCE_TYPES
    if value == "lightweight":
        return LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES
    return ()


def resolve_blocked_resource_types(
    mode: object,
    *,
    text_only: bool = False,
    block_css: bool | None = None,
    block_images: bool | None = None,
    block_fonts: bool | None = None,
    block_media: bool | None = None,
    block_scripts: bool | None = None,
) -> frozenset[str]:
    """Resolve the blocked set from a render mode, a preset, and overrides.

    ``text_only`` adds the visual asset types. Each explicit flag then adds or
    removes one type. ``None`` leaves the preset alone, ``True`` blocks, and
    ``False`` allows.
    """
    blocked: set[str] = set(blocked_resource_types_for_render_mode(mode))
    if text_only:
        blocked.update(TEXT_ONLY_RESOURCE_TYPES)
    overrides = {
        "Stylesheet": block_css,
        "Image": block_images,
        "Font": block_fonts,
        "Media": block_media,
        "TextTrack": block_media,
        "Script": block_scripts,
    }
    for resource_type, value in overrides.items():
        if value is True:
            blocked.add(resource_type)
        elif value is False:
            blocked.discard(resource_type)
    return frozenset(blocked)


def ordered_resource_types(blocked: frozenset[str] | set[str]) -> tuple[str, ...]:
    """Return blocked types in a stable order for reproducible Fetch patterns."""
    return tuple(resource_type for resource_type in RESOURCE_TYPE_ORDER if resource_type in blocked)


def chrome_args_for_blocked(blocked: frozenset[str] | set[str]) -> list[str]:
    """Return launch args for the blocked asset types, in a stable order."""
    args: list[str] = []
    for resource_type in ("Image", "Font", "Media"):
        if resource_type in blocked:
            args.extend(CHROME_BLOCK_ARGS[resource_type])
    return args
