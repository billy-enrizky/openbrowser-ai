#!/usr/bin/env bash
set -euo pipefail

queue_tmp="${RUNNER_TEMP:-$(mktemp -d)}"
parallel_junit="$queue_tmp/merge-queue-parallel.xml"
serial_junit="$queue_tmp/merge-queue-serial.xml"
timing_output="$queue_tmp/merge-queue-timings.json"

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

mcp_root="$queue_tmp/merge-queue-mcp1"
uv venv --python "${PYTHON_VERSION:-3.12}" "$mcp_root"
uv pip install --python "$mcp_root/bin/python" "mcp==1.26.0"
"$mcp_root/bin/python" -c \
  'from importlib.metadata import version; assert version("mcp") == "1.26.0"'

export OPENBROWSER_TEST_MCP1_PYTHON="$mcp_root/bin/python"
uv run --locked pytest tests/ -q -m "not integration" \
  --ignore=tests/test_navigation.py -n 4 --dist load --tb=short \
  --junitxml="$parallel_junit"
uv run --locked pytest tests/test_navigation.py -q -m "not integration" -n 0 --tb=short \
  --junitxml="$serial_junit"

cp tests/ci/test_timings.json "$timing_output"
uv run --locked python scripts/ci/record_timings.py \
  --junit "$parallel_junit" "$serial_junit" --output "$timing_output"
