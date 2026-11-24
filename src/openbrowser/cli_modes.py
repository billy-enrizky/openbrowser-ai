"""Explicit CLI modes selected before Click loads heavy dependencies."""

from __future__ import annotations

from enum import StrEnum


class CliMode(StrEnum):
    """Explicit CLI modes with a fast path around Click."""

    MCP = "mcp"
    EXECUTE_CODE = "execute_code"
    CONFIG = "config"
    DAEMON = "daemon"
    WARM = "warm"
    INSTALL = "install"
    INIT = "init"
    TEMPLATE = "template"


def select_cli_mode(args: list[str]) -> CliMode | None:
    """Return the explicit fast-path mode for argv, or None for Click."""
    prompt_values = {i + 1 for i, arg in enumerate(args[:-1]) if arg in ("-p", "--prompt")}

    def has_flag(flag: str) -> bool:
        return any(arg == flag and index not in prompt_values for index, arg in enumerate(args))

    if has_flag("--mcp"):
        return CliMode.MCP
    if has_flag("-c"):
        return CliMode.EXECUTE_CODE
    if any(arg in ("-p", "--prompt") or arg.startswith("--prompt=") for arg in args):
        # Prompt mode owns its value. "openbrowser-ai -p init" and
        # "openbrowser-ai -p --mcp" are prompts, not modes.
        return None
    if args and args[0] == "config":
        return CliMode.CONFIG
    if args and args[0] == "daemon":
        return CliMode.DAEMON
    if args and args[0] == "warm":
        return CliMode.WARM
    if args and args[0] == "install":
        return CliMode.INSTALL
    if args and args[0] == "init":
        return CliMode.INIT
    if has_flag("--template"):
        return CliMode.TEMPLATE
    return None
