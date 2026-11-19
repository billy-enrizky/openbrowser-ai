"""Real app/SDK stdio fixture. Only browser namespace startup is replaced.

This fixture runs CodeExecutor and application lifecycle/locking unchanged.
It does not validate Chrome or browser-agent behavior.
"""

import asyncio
import os
from pathlib import Path

from openbrowser.mcp import server as module


def _marker(name: str) -> None:
    value = os.environ.get(name)
    if value:
        Path(value).write_text(name)


async def main() -> None:
    module.TELEMETRY_AVAILABLE = False
    server = module.OpenBrowserServer()

    async def wait_for_cancel() -> None:
        _marker("OPENBROWSER_TEST_CANCEL_ENTERED")
        try:
            await asyncio.sleep(3600)
        finally:
            server._pending_variables["cancelled"] = True
            _marker("OPENBROWSER_TEST_CANCEL_UNWOUND")

    async def ensure_namespace() -> None:
        if server._namespace is None:
            server._namespace = {
                "__builtins__": __builtins__,
                "asyncio": asyncio,
                "wait_for_cancel": wait_for_cancel,
            }
            for key, value in server._pending_variables.items():
                if not key.startswith("__") and key not in server._namespace:
                    server._namespace[key] = value
            server._pending_variables.clear()
            server._executor.set_namespace(server._namespace)

    server._ensure_namespace = ensure_namespace
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
