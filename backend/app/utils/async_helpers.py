"""Shared async helpers for wrapping blocking calls."""

import asyncio
from functools import partial


async def run_sync(func, *args, **kwargs):
    """Run a blocking function without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))
