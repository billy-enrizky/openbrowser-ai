"""Granular render asset policy on top of the render mode presets."""

from __future__ import annotations

import tempfile

from openbrowser.browser.profile import BrowserProfile
from openbrowser.browser.render_policy import (
    LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES,
    NO_RENDER_BLOCKED_RESOURCE_TYPES,
    TEXT_ONLY_RESOURCE_TYPES,
    resolve_blocked_resource_types,
)


def test_legacy_modes_resolve_to_their_existing_sets():
    assert resolve_blocked_resource_types("full") == frozenset()
    assert resolve_blocked_resource_types("lightweight") == frozenset(LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES)
    assert resolve_blocked_resource_types("no-render") == frozenset(NO_RENDER_BLOCKED_RESOURCE_TYPES)


def test_text_only_matches_the_no_render_asset_set():
    assert resolve_blocked_resource_types("full", text_only=True) == frozenset(TEXT_ONLY_RESOURCE_TYPES)
    assert resolve_blocked_resource_types("full", text_only=True) == frozenset(NO_RENDER_BLOCKED_RESOURCE_TYPES)


def test_granular_overrides_add_and_remove_types():
    assert resolve_blocked_resource_types("lightweight", block_css=True) == frozenset(
        (*LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES, "Stylesheet")
    )
    assert "Image" not in resolve_blocked_resource_types("lightweight", block_images=False)
    assert "Script" in resolve_blocked_resource_types("full", block_scripts=True)
    assert resolve_blocked_resource_types("no-render", block_css=False) == frozenset(LIGHTWEIGHT_BLOCKED_RESOURCE_TYPES)


def _profile(**kwargs) -> BrowserProfile:
    kwargs.setdefault("user_data_dir", tempfile.mkdtemp(prefix="ob-render-policy-"))
    return BrowserProfile(**kwargs)


def test_profile_exposes_the_resolved_policy_and_launch_args():
    full = _profile()
    assert full.blocked_resource_types() == ()
    assert not any("imagesEnabled" in arg for arg in full.get_args())

    text_only = _profile(text_only=True)
    assert "Stylesheet" in text_only.blocked_resource_types()
    args = text_only.get_args()
    assert "--blink-settings=imagesEnabled=false" in args
    assert "--disable-remote-fonts" in args
    assert text_only.effective_render_mode.value == "full"

    no_render = _profile(render_mode="no-render")
    assert no_render.blocked_resource_types() == NO_RENDER_BLOCKED_RESOURCE_TYPES

    no_images = _profile(block_images=True)
    assert no_images.blocked_resource_types() == ("Image",)
    assert "--blink-settings=imagesEnabled=false" in no_images.get_args()
    assert "--disable-gpu" not in no_images.get_args()


def test_lightweight_launch_args_are_preserved():
    args = _profile(render_mode="lightweight").get_args()
    for expected in (
        "--blink-settings=imagesEnabled=false",
        "--disable-remote-fonts",
        "--disable-gpu",
        "--disable-accelerated-2d-canvas",
        "--disable-accelerated-video-decode",
        "--mute-audio",
        "--autoplay-policy=user-gesture-required",
    ):
        assert expected in args


def test_headless_shell_is_automatic_for_headless_profiles_without_extensions(monkeypatch):
    monkeypatch.delenv("OPENBROWSER_USE_HEADLESS_SHELL", raising=False)

    assert _profile(headless=True, enable_default_extensions=False).prefers_headless_shell() is True
    assert _profile(headless=True, enable_default_extensions=True).prefers_headless_shell() is False
    assert _profile(headless=False, enable_default_extensions=False).prefers_headless_shell() is False
    assert (
        _profile(use_headless_shell=True, headless=True, enable_default_extensions=True).prefers_headless_shell()
        is True
    )


def test_headless_shell_environment_override_wins(monkeypatch):
    monkeypatch.setenv("OPENBROWSER_USE_HEADLESS_SHELL", "0")
    assert _profile(headless=True, enable_default_extensions=False).prefers_headless_shell() is False

    monkeypatch.setenv("OPENBROWSER_USE_HEADLESS_SHELL", "1")
    assert _profile(headless=True, enable_default_extensions=True).prefers_headless_shell() is True
