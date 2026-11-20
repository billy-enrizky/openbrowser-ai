#!/usr/bin/env bash
set -u

root="${1:-.}"
cd "$root" || exit 2
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

set +e
uv run --locked python scripts/ci/check_baseline.py --tool ruff >"$tmp/ruff.txt" 2>&1 &
ruff_pid=$!
uv run --locked python scripts/ci/check_baseline.py --tool pyright >"$tmp/pyright.txt" 2>&1 &
pyright_pid=$!
uv run --locked python scripts/ci/check_baseline.py --tool format >"$tmp/format.txt" 2>&1 &
format_pid=$!
uv run --locked python -m compileall -q backend/app >"$tmp/compile.txt" 2>&1 &
compile_pid=$!

wait "$ruff_pid"; ruff_status=$?
wait "$pyright_pid"; pyright_status=$?
wait "$format_pid"; format_status=$?
wait "$compile_pid"; compile_status=$?

cat "$tmp/ruff.txt" "$tmp/pyright.txt" "$tmp/format.txt" "$tmp/compile.txt"

uv run --locked pytest tests/ci/test_backend_singleton.py -q --tb=short
backend_status=$?
set -e

if [ "$ruff_status" -ne 0 ] || [ "$pyright_status" -ne 0 ] || [ "$format_status" -ne 0 ] || [ "$compile_status" -ne 0 ] || [ "$backend_status" -ne 0 ]; then
  echo "static checks failed: ruff=$ruff_status pyright=$pyright_status format=$format_status compile=$compile_status backend=$backend_status" >&2
  exit 1
fi
echo "static checks passed"
