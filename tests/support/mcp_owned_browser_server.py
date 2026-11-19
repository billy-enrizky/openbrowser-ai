"""Actual MCP/application lifecycle with only the external browser replaced.

File barriers make cancellation land during owned cleanup without timing guesses.
No Chrome or browser-agent integration correctness is established here.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from openbrowser.mcp import server as module


async def main() -> None:
    module.TELEMETRY_AVAILABLE = False
    directory = Path(os.environ["OPENBROWSER_TEST_BARRIERS"])
    mode = os.environ["OPENBROWSER_TEST_OWNED_MODE"]
    server = module.OpenBrowserServer()
    owners = []
    killed = []
    contexts = []
    original_context_init = module.OpenBrowserContext.__init__

    def capture_context(self, owner_server, context_id):
        original_context_init(self, owner_server, context_id)
        contexts.append(self)

    module.OpenBrowserContext.__init__ = capture_context

    class Browser:
        def __init__(self, **kwargs):
            self.ident = len(owners) + 1
            owners.append(self)
            self.kill_attempts = 0

        async def start(self):
            (directory / f"start-{self.ident}").touch()
            if (mode == "initial-start-self-cancel" and self.ident == 1) or (
                mode == "replacement-start-self-cancel" and self.ident == 2
            ):
                raise asyncio.CancelledError("browser start worker terminated itself")
            if mode == "cancel-start" and self.ident == 1:
                await asyncio.sleep(3600)
            if (mode == "start-failure" and self.ident == 1) or (
                mode == "replacement-failure" and self.ident == 2
            ):
                raise RuntimeError("controlled partial browser start failure")

        def seed_generation(self, generation):
            self.generation = generation

        async def kill(self):
            self.kill_attempts += 1
            (directory / f"kill-{self.ident}").touch()
            blocking = 2 if mode == "replacement-failure" else 1
            if self.ident == blocking:
                while not (directory / f"release-{self.ident}").exists():
                    await asyncio.sleep(0.005)
            if mode == "reset-failure" and self.ident == 1 and self.kill_attempts == 1:
                raise RuntimeError("controlled owned cleanup failure")
            if (
                mode in ("reset-child-cancel", "reset-both-cancel")
                and self.ident == 1
                and self.kill_attempts == 1
            ):
                raise asyncio.CancelledError("owned cleanup terminated itself")
            killed.append(self.ident)

    module.BrowserSession = Browser
    module.CodeAgentTools = object
    module.create_namespace = lambda **kwargs: {
        "__builtins__": __builtins__,
        "asyncio": asyncio,
        "browser": kwargs["browser_session"],
    }
    server._build_browser_profile = lambda: None
    module.OpenBrowserContext._build_browser_profile = lambda self: None

    async def alive(self=None, *, deadline=None):
        if (directory / "health-self-cancel").exists():
            (directory / "health-self-cancel").unlink()
            raise asyncio.CancelledError("browser health worker terminated itself")
        return not (directory / "recover").exists()

    server._is_cdp_alive = alive
    module.OpenBrowserContext._is_cdp_alive = alive
    try:
        await server.run()
    finally:
        sys.stderr.write(
            "OWNED_FINAL "
            + json.dumps(
                {
                    "owner": getattr(server.browser_session, "ident", None),
                    "execution_locked": server._execution_lock.locked(),
                    "lifecycle_locked": server._lifecycle_lock.locked(),
                    "killed": killed,
                }
            )
            + "\n"
        )
        if contexts:
            modern = contexts[-1]
            sys.stderr.write(
                "MODERN_FINAL "
                + json.dumps(
                    {
                        "owner": getattr(modern.browser_session, "ident", None),
                        "execution_locked": modern._execution_lock.locked(),
                        "lifecycle_locked": modern._lifecycle_lock.locked(),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    asyncio.run(main())
