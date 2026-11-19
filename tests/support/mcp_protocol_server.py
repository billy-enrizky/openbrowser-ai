"""Real app/SDK stdio fixture. Only browser namespace startup is replaced.

This fixture runs CodeExecutor and application lifecycle/locking unchanged.
It does not validate Chrome or browser-agent behavior.
"""

import asyncio

from openbrowser.mcp import server as module


async def main() -> None:
    module.TELEMETRY_AVAILABLE = False
    server = module.OpenBrowserServer()

    async def wait_for_cancel() -> None:
        try:
            await asyncio.sleep(3600)
        finally:
            server._pending_variables["cancelled"] = True

    async def ensure_namespace() -> None:
        if server._namespace is None:
            server._namespace = {
                "__builtins__": __builtins__,
                "asyncio": asyncio,
                "wait_for_cancel": wait_for_cancel,
            }
            server._executor.set_namespace(server._namespace)

    server._ensure_namespace = ensure_namespace
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
