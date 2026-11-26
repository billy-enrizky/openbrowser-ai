"""Drop-in scraping arm for the native Playwright MCP and CLI sessions.

``benchmarks.scraping.scrape_compare`` owns the lane and its exact oracles.
This class maps the five scraping tasks to the shared trusted native programs
and returns the same ``(ok, output, detail)`` shape as the other scraping
arms. Each lane keeps one native session, matching the session and page reuse
of the browser and service arms in the same lane.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..warm_benchmark import make_context
from .playwright_common import assert_no_secret
from .playwright_tasks import task_program

ContextFactory = Callable[[Path, str], Any]
SCRAPE_ARMS = ("playwright-mcp", "playwright-cli")


def _oracle(name: str):
    from benchmarks.scraping import scrape_compare

    return getattr(scrape_compare, name)


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


class PlaywrightScrapeArm:
    """Scraping surface for one native Playwright session.

    ``start`` only allocates the run directory. ``prepare`` warms the native
    session once, before the lane timer, so every task measures one program on
    a ready session. ``stop`` closes the owned session exactly once.
    """

    last_engine = "browser"

    def __init__(
        self,
        name: str,
        arm: Any,
        *,
        workdir: Path | None = None,
        context_factory: ContextFactory = make_context,
    ) -> None:
        self.name = name
        self.arm = arm
        self._workdir = Path(workdir) if workdir is not None else None
        self._owns_workdir = workdir is None
        self._context_factory = context_factory
        self._ctx: Any = None
        self._prepared = False

    async def start(self) -> None:
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="pws-", dir="/tmp" if os.name != "nt" else None))

    async def prepare(self, task_name: str) -> None:
        if self._ctx is None:
            if self._workdir is None:
                await self.start()
            assert self._workdir is not None
            self._ctx = self._context_factory(self._workdir, self.name)
        if not self._prepared:
            self.arm.warmup(self._ctx)
            self._prepared = True

    async def stop(self) -> None:
        if self._ctx is not None:
            self.arm.cleanup(self._ctx)
            self._ctx = None
            self._prepared = False
        if self._owns_workdir and self._workdir is not None:
            shutil.rmtree(self._workdir)
            self._workdir = None

    # -- scrape_compare task surface --------------------------------------
    async def run_static(self, base: str):
        return self._run("static_article", base, "content", _oracle("exact_static"))

    async def run_js_list(self, base: str):
        return self._run("js_list", base, "skus", _oracle("exact_js_list"))

    async def run_pagination(self, base: str):
        return self._run("pagination", base, "ids", _oracle("exact_pagination"))

    async def run_table(self, base: str):
        return self._run("table", base, "rows", _oracle("exact_table"))

    async def run_assets(self, base: str):
        return self._run("asset_heavy", base, "content", _oracle("exact_assets"))

    def _run(self, task: str, base: str, key: str, check):
        if self._ctx is None or not self._prepared:
            raise RuntimeError("prepare() must run before a scraping task")
        ctx = replace(self._ctx, task=task, base_url=base)
        result = self.arm.execute_program(ctx, task_program(task, base, ctx.downloads))
        try:
            assert_no_secret(result.stdout, result.stderr)
        except RuntimeError as error:
            return False, result.stdout, str(error)
        if not result.ok:
            return False, result.stdout, result.error or "native program failed"
        reported = result.reported if isinstance(result.reported, dict) else {}
        value = reported.get(key)
        if value is None:
            return False, "", f"missing {key} in native result"
        if not check(value):
            return False, _render(value), "oracle rejected native result"
        return True, _render(value), ""


def build_scrape_arm(name: str, *, workdir: Path | None = None, **arm_options: Any) -> PlaywrightScrapeArm:
    """Build a native scraping arm by name, importing the adapter lazily."""
    if name == "playwright-mcp":
        from .playwright_mcp import PlaywrightMcpArm

        return PlaywrightScrapeArm(name, PlaywrightMcpArm(**arm_options), workdir=workdir)
    if name == "playwright-cli":
        from .playwright_cli import PlaywrightCliArm

        return PlaywrightScrapeArm(name, PlaywrightCliArm(**arm_options), workdir=workdir)
    raise ValueError(f"unknown Playwright scraping arm: {name}")
