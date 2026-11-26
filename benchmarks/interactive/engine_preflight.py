"""Export and validate a common browser pin before native task execution."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from .engine_binding import ENGINE_PIN_ENV, EngineBindingError, EnginePin, read_pin_environment, write_pin_environment


def configure_engine_environment(pin: EnginePin, env: MutableMapping[str, str]) -> None:
    """Declare one executable and its metadata for every fresh native worker."""
    updates: dict[str, str] = {}
    write_pin_environment(pin, updates)
    updates.update(
        CHROME_BIN=str(pin.launcher),
        PLAYWRIGHT_BENCH_BROWSER_VERSION=pin.version,
        OPENBROWSER_USE_HEADLESS_SHELL="0",
    )
    env.update(updates)


def configured_engine_pin(env: Mapping[str, str] | None = None) -> EnginePin | None:
    """Return the declared pin, or preserve an explicitly unconfigured lane.

    A present pin requires the matching launch path and version. A worker uses
    stat fences only. The lane controller performs the full pre/post hashes.
    """
    env = os.environ if env is None else env
    if ENGINE_PIN_ENV not in env:
        return None
    pin = read_pin_environment(env)
    path = env.get("CHROME_BIN")
    try:
        matched_path = False
        if isinstance(path, str) and path and Path(path).is_absolute():
            matched_path = Path(path).resolve(strict=True) == pin.launcher
    except (OSError, RuntimeError):
        matched_path = False
    if (
        not matched_path
        or env.get("PLAYWRIGHT_BENCH_BROWSER_VERSION") != pin.version
        or env.get("OPENBROWSER_USE_HEADLESS_SHELL") != "0"
    ):
        raise EngineBindingError("engine_configuration_mismatch")
    pin.fence()
    return pin
