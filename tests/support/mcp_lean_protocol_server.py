"""Lean stdio fixture. Only browser namespace startup is replaced."""

import asyncio
import os
from pathlib import Path

from openbrowser.mcp import server as module
from openbrowser.mcp.lean import LeanStdioServer


def _marker(name: str) -> None:
    value = os.environ.get(name)
    if value:
        Path(value).write_text(name)


async def ensure_namespace(self, *, start_timeout: float | None = None) -> None:
    if self._namespace is not None:
        return

    async def wait_for_cancel() -> None:
        _marker("OPENBROWSER_TEST_CANCEL_ENTERED")
        try:
            await asyncio.sleep(3600)
        finally:
            self._pending_variables["cancelled"] = True
            _marker("OPENBROWSER_TEST_CANCEL_UNWOUND")

    self._namespace = {
        "__builtins__": __builtins__,
        "asyncio": asyncio,
        "os": os,
        "wait_for_cancel": wait_for_cancel,
    }
    for key, value in self._pending_variables.items():
        if not key.startswith("__") and key not in self._namespace:
            self._namespace[key] = value
    self._pending_variables.clear()


async def main() -> None:
    module.TELEMETRY_AVAILABLE = False
    module.OpenBrowserServer._ensure_namespace = ensure_namespace
    module.OpenBrowserContext._ensure_namespace = ensure_namespace
    await LeanStdioServer().serve()


if __name__ == "__main__":
    asyncio.run(main())
