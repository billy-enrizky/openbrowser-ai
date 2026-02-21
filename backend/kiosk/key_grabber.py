"""X11 key grabber daemon -- intercepts dangerous key combinations.

Runs on the VNC display and grabs keyboard shortcuts that could
close the browser or access the desktop. Grabbed keys are silently
consumed and never reach applications.

Usage:
    DISPLAY=:99 python key_grabber.py
"""

import logging
import os
import signal
import sys

logger = logging.getLogger("key_grabber")


def main():
    """Grab dangerous key combinations on the X11 display."""
    try:
        from Xlib import X, XK
        from Xlib.display import Display
    except ImportError:
        logger.error("python-xlib not installed, key grabber disabled")
        sys.exit(1)

    display_name = os.environ.get("DISPLAY")
    if not display_name:
        logger.error("DISPLAY not set, key grabber cannot start")
        sys.exit(1)

    logger.info("Starting key grabber on display %s", display_name)
    display = Display(display_name)
    root = display.screen().root

    # Modifier masks
    MOD_ALT = X.Mod1Mask
    MOD_CTRL = X.ControlMask
    MOD_SHIFT = X.ShiftMask

    # Key combinations to intercept.
    # Each entry is (modifier_mask, keysym_name).
    # A modifier_mask of 0 means grab the key with no modifiers.
    dangerous_combos = [
        # Close window
        (MOD_ALT, "F4"),
        # Close tab
        (MOD_CTRL, "w"),
        # Quit browser
        (MOD_CTRL, "q"),
        # Switch window
        (MOD_ALT, "Tab"),
        # Ctrl+Alt+Delete
        (MOD_CTRL | MOD_ALT, "Delete"),
        # VT switching (Ctrl+Alt+F1 through F12)
        *[(MOD_CTRL | MOD_ALT, f"F{i}") for i in range(1, 13)],
        # Super/Windows key (could open desktop menu)
        (0, "Super_L"),
        (0, "Super_R"),
        # WM menus / launchers
        (MOD_ALT, "F1"),
        (MOD_ALT, "F2"),
        # Terminal shortcut in some desktop environments
        (MOD_CTRL | MOD_ALT, "t"),
        # xkill shortcut
        (MOD_CTRL | MOD_ALT, "Escape"),
        # Browser shortcuts that could be abused
        (MOD_CTRL | MOD_SHIFT, "i"),    # DevTools (Ctrl+Shift+I)
        (MOD_CTRL | MOD_SHIFT, "j"),    # DevTools Console (Ctrl+Shift+J)
        (MOD_CTRL, "u"),                # View source (Ctrl+U)
        (MOD_CTRL, "l"),                # Address bar focus (Ctrl+L)
        (MOD_ALT, "d"),                 # Address bar focus (Alt+D)
        (MOD_CTRL, "t"),                # New tab (Ctrl+T)
        (MOD_CTRL, "n"),                # New window (Ctrl+N)
        (MOD_CTRL | MOD_SHIFT, "n"),    # New incognito window
        (0, "F5"),                      # Refresh (allow agent to use CDP instead)
        (MOD_CTRL, "r"),                # Refresh (same reason)
        (0, "F11"),                     # Fullscreen toggle
        (0, "F12"),                     # DevTools shortcut
        (MOD_CTRL | MOD_SHIFT, "Delete"),  # Clear browsing data
    ]

    # Extra modifier masks for NumLock and CapsLock combinations.
    # We need to grab with these because they affect the modifier state.
    NUMLOCK = X.Mod2Mask
    CAPSLOCK = X.LockMask
    extra_modifiers = [0, NUMLOCK, CAPSLOCK, NUMLOCK | CAPSLOCK]

    grabbed_count = 0
    for mod_mask, key_name in dangerous_combos:
        keysym = XK.string_to_keysym(key_name)
        if keysym == 0:
            logger.warning("Unknown keysym: %s", key_name)
            continue

        keycode = display.keysym_to_keycode(keysym)
        if keycode == 0:
            logger.warning("No keycode for keysym: %s", key_name)
            continue

        for extra in extra_modifiers:
            try:
                root.grab_key(
                    keycode,
                    mod_mask | extra,
                    True,               # owner_events
                    X.GrabModeAsync,
                    X.GrabModeAsync,
                )
                grabbed_count += 1
            except Exception:
                logger.warning(
                    "Failed to grab %s (mod=0x%x)", key_name, mod_mask | extra
                )

    logger.info("Grabbed %d key combinations", grabbed_count)

    # Clean shutdown on SIGTERM/SIGINT
    def handle_signal(signum, _frame):
        logger.info("Key grabber shutting down (signal %d)", signum)
        display.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Event loop: consume grabbed key events silently.
    # The key press is intercepted at the root window and discarded here,
    # so it never reaches the browser or window manager.
    while True:
        display.next_event()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(levelname)s: %(message)s",
    )
    main()
