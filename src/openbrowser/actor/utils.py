"""Utility functions for the actor module."""

import logging

logger = logging.getLogger(__name__)


def get_key_info(key: str) -> tuple[str, int | None]:
    """Get the proper code and virtual key code for a key.
    
    Args:
        key: Key name (e.g., 'Enter', 'Tab', 'a', 'Control')
        
    Returns:
        Tuple of (code, windowsVirtualKeyCode)
    """
    # Mapping of key names to (code, virtualKeyCode)
    key_mappings: dict[str, tuple[str, int]] = {
        # Modifier keys
        'Alt': ('AltLeft', 18),
        'AltLeft': ('AltLeft', 18),
        'AltRight': ('AltRight', 18),
        'Control': ('ControlLeft', 17),
        'ControlLeft': ('ControlLeft', 17),
        'ControlRight': ('ControlRight', 17),
        'Meta': ('MetaLeft', 91),
        'MetaLeft': ('MetaLeft', 91),
        'MetaRight': ('MetaRight', 92),
        'Shift': ('ShiftLeft', 16),
        'ShiftLeft': ('ShiftLeft', 16),
        'ShiftRight': ('ShiftRight', 16),
        
        # Function keys
        'F1': ('F1', 112),
        'F2': ('F2', 113),
        'F3': ('F3', 114),
        'F4': ('F4', 115),
        'F5': ('F5', 116),
        'F6': ('F6', 117),
        'F7': ('F7', 118),
        'F8': ('F8', 119),
        'F9': ('F9', 120),
        'F10': ('F10', 121),
        'F11': ('F11', 122),
        'F12': ('F12', 123),
        
        # Navigation keys
        'ArrowDown': ('ArrowDown', 40),
        'ArrowLeft': ('ArrowLeft', 37),
        'ArrowRight': ('ArrowRight', 39),
        'ArrowUp': ('ArrowUp', 38),
        'End': ('End', 35),
        'Home': ('Home', 36),
        'PageDown': ('PageDown', 34),
        'PageUp': ('PageUp', 33),
        
        # Editing keys
        'Backspace': ('Backspace', 8),
        'Delete': ('Delete', 46),
        'Insert': ('Insert', 45),
        
        # Whitespace keys
        'Enter': ('Enter', 13),
        'Tab': ('Tab', 9),
        'Space': ('Space', 32),
        ' ': ('Space', 32),
        
        # Special keys
        'Escape': ('Escape', 27),
        'CapsLock': ('CapsLock', 20),
        'NumLock': ('NumLock', 144),
        'ScrollLock': ('ScrollLock', 145),
        'Pause': ('Pause', 19),
        'PrintScreen': ('PrintScreen', 44),
        
        # Numpad keys
        'Numpad0': ('Numpad0', 96),
        'Numpad1': ('Numpad1', 97),
        'Numpad2': ('Numpad2', 98),
        'Numpad3': ('Numpad3', 99),
        'Numpad4': ('Numpad4', 100),
        'Numpad5': ('Numpad5', 101),
        'Numpad6': ('Numpad6', 102),
        'Numpad7': ('Numpad7', 103),
        'Numpad8': ('Numpad8', 104),
        'Numpad9': ('Numpad9', 105),
        'NumpadAdd': ('NumpadAdd', 107),
        'NumpadDecimal': ('NumpadDecimal', 110),
        'NumpadDivide': ('NumpadDivide', 111),
        'NumpadEnter': ('NumpadEnter', 13),
        'NumpadMultiply': ('NumpadMultiply', 106),
        'NumpadSubtract': ('NumpadSubtract', 109),
    }
    
    if key in key_mappings:
        return key_mappings[key]
    
    # Handle single character keys
    if len(key) == 1:
        if key.isalpha():
            return (f'Key{key.upper()}', ord(key.upper()))
        elif key.isdigit():
            return (f'Digit{key}', ord(key))
        else:
            # Special characters - use their unicode value
            return (key, ord(key))
    
    # Fallback for unknown keys
    logger.warning(f'Unknown key: {key}, using default handling')
    return (key, None)


def calculate_modifier_bitmask(modifiers: list[str] | None) -> int:
    """Calculate the CDP modifier bitmask from a list of modifier names.
    
    Args:
        modifiers: List of modifier names ('Alt', 'Control', 'Meta', 'Shift')
        
    Returns:
        Integer bitmask for CDP Input events
    """
    if not modifiers:
        return 0
    
    modifier_map = {
        'Alt': 1,
        'Control': 2,
        'Meta': 4,
        'Shift': 8,
    }
    
    bitmask = 0
    for mod in modifiers:
        bitmask |= modifier_map.get(mod, 0)
    
    return bitmask

