"""Structural guard for the backend's one-app-per-process rule."""

from __future__ import annotations

import ast
from pathlib import Path


def test_backend_main_constructs_one_module_level_app() -> None:
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "backend" / "app" / "main.py").read_text(encoding="utf-8"))
    module_app_assignments = []
    fastapi_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FastAPI":
            fastapi_calls.append(node)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "app":
                module_app_assignments.append(node)
    assert len(fastapi_calls) == 1
    assert len(module_app_assignments) == 1
    value = module_app_assignments[0].value
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "FastAPI"
