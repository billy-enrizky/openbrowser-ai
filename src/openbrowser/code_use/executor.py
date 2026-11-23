# src/openbrowser/code_use/executor.py
"""Shared code executor for MCP server and daemon.

Wraps user Python code in an async function, executes it against a
persistent namespace, captures stdout, and returns structured results.
"""

import asyncio
import contextlib
import io
import logging
import traceback
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 10_000


def output_limit_for_executor(executor: Any) -> int:
    """Return the executor's positive output cap or the default cap."""
    limit = getattr(executor, 'max_output_chars', None)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return DEFAULT_MAX_OUTPUT_CHARS
    return limit


@dataclass
class ExecutionResult:
    """Result of a code execution."""
    success: bool
    output: str
    error: str | None = None  # raw exception message (not truncated), for callers to inspect


class CodeExecutor:
    """Execute Python code in a persistent namespace.

    Shared between the MCP server and the CLI daemon. The namespace
    holds browser automation functions (navigate, click, evaluate, etc.)
    and user-defined variables that persist across calls.
    """

    def __init__(self, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS):
        self._namespace: dict[str, Any] | None = None
        self._max_output_chars = max_output_chars
        self._lock = asyncio.Lock()

    @property
    def max_output_chars(self) -> int:
        """Return the configured output cap used for retained diagnostics."""
        return self._max_output_chars

    @property
    def initialized(self) -> bool:
        return self._namespace is not None

    @property
    def namespace(self) -> dict[str, Any] | None:
        """Return the live namespace for inspection or transfer by an owner."""
        return self._namespace

    def set_namespace(self, namespace: dict[str, Any] | None) -> None:
        """Set or clear the execution namespace (for external initialization)."""
        self._namespace = namespace

    def clear_namespace(self) -> None:
        """Drop the active namespace without executing anything."""
        self._namespace = None

    async def execute(self, code: str) -> ExecutionResult:
        """Execute Python code and return the result.

        Code is wrapped in an async function so ``await`` works.
        stdout is captured and returned. Variables persist across calls.
        Serialized via asyncio.Lock to protect sys.stdout from concurrent access.
        """
        namespace = self._namespace
        if namespace is None:
            return ExecutionResult(success=False, output='Error: namespace not initialized', error='namespace not initialized')

        if not code.strip():
            return ExecutionResult(success=False, output='Error: No code provided', error='No code provided')

        async with self._lock:
            # Use try/finally so namespace update always runs, even if user code
            # contains an early ``return``.
            indented = '\n'.join(f'        {line}' for line in code.split('\n'))
            wrapped = (
                'async def __ob_exec__(__ns__):\n'
                '    __ob_result__ = None\n'
                '    try:\n'
                f'{indented}\n'
                '    finally:\n'
                '        __ns__.update({k: v for k, v in locals().items() if not k.startswith("__")})\n'
                '    return __ob_result__\n'
            )

            stdout_capture = io.StringIO()

            try:
                compiled = compile(wrapped, '<execute_code>', 'exec')
                exec(compiled, namespace)

                with contextlib.redirect_stdout(stdout_capture):
                    result = await namespace['__ob_exec__'](namespace)

                output = stdout_capture.getvalue()

                if result is not None and not output.strip():
                    output = repr(result)

                if not output.strip():
                    output = '(executed successfully, no output)'

                output = self._truncate(output)
                return ExecutionResult(success=True, output=output)

            except Exception as e:
                captured = stdout_capture.getvalue()
                raw_error = f'{type(e).__name__}: {e}'
                tb = traceback.format_exc()
                error_output = f'Error: {raw_error}\n\nTraceback:\n{tb}'
                if captured.strip():
                    error_output = f'Output before error:\n{captured}\n\n{error_output}'
                error_output = self._truncate(error_output)
                return ExecutionResult(success=False, output=error_output, error=raw_error)

            finally:
                namespace.pop('__ob_exec__', None)
                namespace.pop('__ob_result__', None)

    def _truncate(self, text: str) -> str:
        if self._max_output_chars and len(text) > self._max_output_chars:
            return text[:self._max_output_chars] + f'\n... (truncated, {len(text)} chars total)'
        return text
