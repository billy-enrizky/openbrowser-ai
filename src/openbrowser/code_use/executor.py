# src/openbrowser/code_use/executor.py
"""Shared code executor for MCP server and daemon.

Wraps user Python code in an async function, executes it against a
persistent namespace, captures stdout, and returns structured results.
"""

import io
import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 10_000


@dataclass
class ExecutionResult:
    """Result of a code execution."""
    success: bool
    output: str


class CodeExecutor:
    """Execute Python code in a persistent namespace.

    Shared between the MCP server and the CLI daemon. The namespace
    holds browser automation functions (navigate, click, evaluate, etc.)
    and user-defined variables that persist across calls.
    """

    def __init__(self, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS):
        self._namespace: dict[str, Any] | None = None
        self._max_output_chars = max_output_chars

    @property
    def initialized(self) -> bool:
        return self._namespace is not None

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        """Set the execution namespace (for external initialization)."""
        self._namespace = namespace

    async def execute(self, code: str) -> ExecutionResult:
        """Execute Python code and return the result.

        Code is wrapped in an async function so ``await`` works.
        stdout is captured and returned. Variables persist across calls.
        """
        if self._namespace is None:
            return ExecutionResult(success=False, output='Error: namespace not initialized')

        indented = '\n'.join(f'    {line}' for line in code.split('\n'))
        wrapped = (
            'async def __ob_exec__(__ns__):\n'
            f'{indented}\n'
            '    __ns__.update({k: v for k, v in locals().items() if not k.startswith("__")})\n'
        )

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        try:
            compiled = compile(wrapped, '<execute_code>', 'exec')
            exec(compiled, self._namespace)

            sys.stdout = stdout_capture
            try:
                result = await self._namespace['__ob_exec__'](self._namespace)
            finally:
                sys.stdout = old_stdout

            output = stdout_capture.getvalue()

            if result is not None and not output.strip():
                output = repr(result)

            if not output.strip():
                output = '(executed successfully, no output)'

            output = self._truncate(output)
            return ExecutionResult(success=True, output=output)

        except Exception as e:
            sys.stdout = old_stdout
            captured = stdout_capture.getvalue()
            tb = traceback.format_exc()
            error_output = f'Error: {type(e).__name__}: {e}\n\nTraceback:\n{tb}'
            if captured.strip():
                error_output = f'Output before error:\n{captured}\n\n{error_output}'
            error_output = self._truncate(error_output)
            return ExecutionResult(success=False, output=error_output)

        finally:
            self._namespace.pop('__ob_exec__', None)

    def _truncate(self, text: str) -> str:
        if self._max_output_chars and len(text) > self._max_output_chars:
            return text[:self._max_output_chars] + f'\n... (truncated, {len(text)} chars total)'
        return text
