#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --all-extras
./scripts/ci/run_static.sh .

(
  cd frontend
  bun install --frozen-lockfile
  bun check
  bun run lint:oxlint
  bun run fmt:check
  bun run lint
  bun run build
)

(
  cd backend
  uv sync --locked
  uv run --locked python -c \
    'import app.main; from fastapi import FastAPI; assert isinstance(app.main.app, FastAPI); import app.main as again; assert again.app is app.main.app'
)

mcp_root="${RUNNER_TEMP:-$(mktemp -d)}/merge-queue-mcp1"
uv venv --python "${PYTHON_VERSION:-3.12}" "$mcp_root"
uv pip install --python "$mcp_root/bin/python" "mcp==1.26.0"
"$mcp_root/bin/python" -c \
  'from importlib.metadata import version; assert version("mcp") == "1.26.0"'

export OPENBROWSER_TEST_MCP1_PYTHON="$mcp_root/bin/python"
uv run --locked pytest tests/ -q -m "not integration" \
  --ignore=tests/test_navigation.py -n 4 --dist load --tb=short
uv run --locked pytest tests/test_navigation.py -q -m "not integration" -n 0 --tb=short
