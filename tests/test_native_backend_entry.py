"""The Python backend retains the public CLI display identity."""

from __future__ import annotations

import sys

import pytest


@pytest.mark.parametrize(
    "program,expected",
    [
        ("openbrowser-ai-py", "openbrowser-ai"),
        ("/tmp/bin/openbrowser-ai-py", "/tmp/bin/openbrowser-ai"),
        ("/tmp/openbrowser-ai-py/bin/openbrowser-ai-py", "/tmp/openbrowser-ai-py/bin/openbrowser-ai"),
        (r"C:\tools\openbrowser-ai-py.exe", r"C:\tools\openbrowser-ai.exe"),
        ("openbrowser-ai", "openbrowser-ai"),
        ("/tmp/other-openbrowser-ai-py", "/tmp/other-openbrowser-ai-py"),
    ],
)
def test_backend_preserves_public_program_identity(monkeypatch, program, expected):
    from openbrowser import cli_entry

    seen = []
    monkeypatch.setattr(sys, "argv", [program, "--version"])
    monkeypatch.setattr(cli_entry, "_version_fast_path", lambda argv: seen.append(list(argv)) or True)
    cli_entry.cli()
    assert seen == [[expected, "--version"]]
