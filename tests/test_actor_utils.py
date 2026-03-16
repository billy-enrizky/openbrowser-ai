"""Tests for openbrowser.actor.utils module."""

import logging

import pytest

from openbrowser.actor.utils import Utils, get_key_info

logger = logging.getLogger(__name__)


class TestGetKeyInfoFunction:
    """Tests for the module-level get_key_info function."""

    def test_known_navigation_keys(self):
        """Test that known navigation keys return correct code and VK code."""
        assert get_key_info('Enter') == ('Enter', 13)
        assert get_key_info('Backspace') == ('Backspace', 8)
        assert get_key_info('Tab') == ('Tab', 9)
        assert get_key_info('Escape') == ('Escape', 27)
        assert get_key_info('Space') == ('Space', 32)
        assert get_key_info(' ') == ('Space', 32)

    def test_arrow_keys(self):
        """Test arrow key mappings."""
        assert get_key_info('ArrowLeft') == ('ArrowLeft', 37)
        assert get_key_info('ArrowUp') == ('ArrowUp', 38)
        assert get_key_info('ArrowRight') == ('ArrowRight', 39)
        assert get_key_info('ArrowDown') == ('ArrowDown', 40)

    def test_modifier_keys(self):
        """Test modifier key mappings."""
        assert get_key_info('Shift') == ('ShiftLeft', 16)
        assert get_key_info('Control') == ('ControlLeft', 17)
        assert get_key_info('Alt') == ('AltLeft', 18)
        assert get_key_info('Meta') == ('MetaLeft', 91)

    def test_function_keys(self):
        """Test function key mappings F1-F12."""
        assert get_key_info('F1') == ('F1', 112)
        assert get_key_info('F5') == ('F5', 116)
        assert get_key_info('F12') == ('F12', 123)
        assert get_key_info('F24') == ('F24', 135)

    def test_numpad_keys(self):
        """Test numpad key mappings."""
        assert get_key_info('Numpad0') == ('Numpad0', 96)
        assert get_key_info('Numpad9') == ('Numpad9', 105)
        assert get_key_info('NumpadAdd') == ('NumpadAdd', 107)

    def test_punctuation_keys(self):
        """Test OEM/punctuation key mappings."""
        assert get_key_info(';') == ('Semicolon', 186)
        assert get_key_info('=') == ('Equal', 187)
        assert get_key_info(',') == ('Comma', 188)
        assert get_key_info('-') == ('Minus', 189)
        assert get_key_info('.') == ('Period', 190)
        assert get_key_info('/') == ('Slash', 191)
        assert get_key_info('`') == ('Backquote', 192)
        assert get_key_info('[') == ('BracketLeft', 219)
        assert get_key_info('\\') == ('Backslash', 220)
        assert get_key_info(']') == ('BracketRight', 221)
        assert get_key_info("'") == ('Quote', 222)

    def test_alpha_single_char_lowercase(self):
        """Test dynamic alpha key generation for lowercase letters."""
        code, vk = get_key_info('a')
        assert code == 'KeyA'
        assert vk == 65

        code, vk = get_key_info('z')
        assert code == 'KeyZ'
        assert vk == 90

    def test_alpha_single_char_uppercase(self):
        """Test dynamic alpha key generation for uppercase letters."""
        code, vk = get_key_info('A')
        assert code == 'KeyA'
        assert vk == 65

        code, vk = get_key_info('Z')
        assert code == 'KeyZ'
        assert vk == 90

    def test_digit_single_char(self):
        """Test dynamic digit key generation."""
        code, vk = get_key_info('0')
        assert code == 'Digit0'
        assert vk == 48

        code, vk = get_key_info('9')
        assert code == 'Digit9'
        assert vk == 57

    def test_unknown_key_fallback(self):
        """Test fallback behavior for unknown keys."""
        code, vk = get_key_info('UnknownSpecialKey')
        assert code == 'UnknownSpecialKey'
        assert vk is None

    def test_media_keys(self):
        """Test media key mappings."""
        assert get_key_info('AudioVolumeMute') == ('AudioVolumeMute', 173)
        assert get_key_info('MediaPlayPause') == ('MediaPlayPause', 179)

    def test_lock_keys(self):
        """Test lock key mappings."""
        assert get_key_info('CapsLock') == ('CapsLock', 20)
        assert get_key_info('NumLock') == ('NumLock', 144)
        assert get_key_info('ScrollLock') == ('ScrollLock', 145)

    def test_additional_keys(self):
        """Test additional common key mappings."""
        assert get_key_info('Delete') == ('Delete', 46)
        assert get_key_info('Insert') == ('Insert', 45)
        assert get_key_info('Home') == ('Home', 36)
        assert get_key_info('End') == ('End', 35)
        assert get_key_info('PageUp') == ('PageUp', 33)
        assert get_key_info('PageDown') == ('PageDown', 34)


class TestUtilsClass:
    """Tests for the Utils class static method."""

    def test_get_key_info_matches_module_function(self):
        """Test that Utils.get_key_info matches the module-level function."""
        assert Utils.get_key_info('Enter') == get_key_info('Enter')
        assert Utils.get_key_info('a') == get_key_info('a')
        assert Utils.get_key_info('5') == get_key_info('5')
        assert Utils.get_key_info('F12') == get_key_info('F12')
        assert Utils.get_key_info('UnknownKey') == get_key_info('UnknownKey')

    def test_utils_has_slots(self):
        """Test that Utils uses __slots__ for performance."""
        assert hasattr(Utils, '__slots__')
        assert Utils.__slots__ == ()
