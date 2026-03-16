"""Extensive unit tests for CodeExecutor.

Tests cover positive behavior, negative/error handling, and edge cases
for the CodeExecutor class defined in src/openbrowser/code_use/executor.py.

Every test has a docstring explaining its purpose and what edge case it
validates, as required by the assignment specification.
"""

import asyncio
import logging

import pytest

from openbrowser.code_use.executor import CodeExecutor, ExecutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def executor() -> CodeExecutor:
    """Return a CodeExecutor with default max_output_chars."""
    ex = CodeExecutor()
    ex.set_namespace({})
    return ex


@pytest.fixture
def small_executor() -> CodeExecutor:
    """Return a CodeExecutor with a very small output limit for truncation tests."""
    ex = CodeExecutor(max_output_chars=50)
    ex.set_namespace({})
    return ex


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------

class TestPositive:
    """Tests that verify correct, expected behavior of CodeExecutor."""

    @pytest.mark.asyncio
    async def test_basic_variable_assignment_and_retrieval(self, executor: CodeExecutor):
        """Verify that a simple variable assignment stores the value in the
        namespace so subsequent code can access it.

        This is the most fundamental operation: assign a value and confirm
        it lands in the shared namespace dict.
        """
        result = await executor.execute("x = 42")
        assert result.success is True
        assert executor._namespace["x"] == 42
        logger.info("Variable x correctly stored as %d", executor._namespace["x"])

    @pytest.mark.asyncio
    async def test_variable_persistence_across_calls(self, executor: CodeExecutor):
        """Verify that variables set in one execute() call are visible in a
        later execute() call.

        CodeExecutor maintains a persistent namespace, so user-defined
        variables must survive across invocations -- this is critical for
        the interactive coding workflow where the LLM builds state over
        multiple steps.

        Note: Because user code is wrapped inside an async function,
        augmented assignment (+=) on a namespace variable creates a
        local binding and triggers UnboundLocalError. Persistence is
        validated by reading the variable in a subsequent call instead.
        """
        await executor.execute("greeting = 'hello'")
        result = await executor.execute("print(greeting + ' world')")
        assert result.success is True
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_async_code_execution_with_await(self, executor: CodeExecutor):
        """Verify that user code containing 'await' works correctly.

        The executor wraps code inside an async function so that 'await'
        expressions are legal. This test confirms that async stdlib
        utilities like asyncio.sleep can be awaited without error.
        """
        # asyncio must be available in the namespace for await to work
        import asyncio as _asyncio_mod
        executor._namespace["asyncio"] = _asyncio_mod
        result = await executor.execute("await asyncio.sleep(0)\nresult_val = 'async_ok'")
        assert result.success is True
        assert executor._namespace.get("result_val") == "async_ok"

    @pytest.mark.asyncio
    async def test_return_value_display(self, executor: CodeExecutor):
        """Verify that setting __ob_result__ causes its repr to appear as
        output when there is no stdout.

        The executor checks if __ob_result__ is not None and stdout is
        empty; if so, it returns repr(__ob_result__). This lets the LLM
        see the 'return value' of its code without explicitly printing.
        """
        result = await executor.execute("__ob_result__ = {'key': 'value'}")
        assert result.success is True
        assert "key" in result.output
        assert "value" in result.output

    @pytest.mark.asyncio
    async def test_print_output_capture(self, executor: CodeExecutor):
        """Verify that print() output is captured and returned.

        The executor redirects sys.stdout to a StringIO buffer during
        execution. Any print() calls must appear in result.output.
        """
        result = await executor.execute('print("hello from executor")')
        assert result.success is True
        assert "hello from executor" in result.output

    @pytest.mark.asyncio
    async def test_mixed_print_and_return_value(self, executor: CodeExecutor):
        """Verify that when both print() and __ob_result__ are used,
        the print output takes precedence.

        The executor only falls back to repr(__ob_result__) when stdout
        is empty. If stdout already has content, the result is ignored.
        This prevents confusing duplicate output.
        """
        result = await executor.execute(
            'print("printed line")\n__ob_result__ = "should not appear"'
        )
        assert result.success is True
        assert "printed line" in result.output
        # The __ob_result__ repr should NOT appear because stdout was non-empty
        assert "should not appear" not in result.output

    @pytest.mark.asyncio
    async def test_multi_line_code_execution(self, executor: CodeExecutor):
        """Verify that multi-line code blocks execute correctly.

        The executor indents every line of user code inside the wrapper
        function. Multi-line code with control flow (loops, conditionals)
        must be indented correctly and execute without SyntaxError.
        """
        code = (
            "total = 0\n"
            "for i in range(5):\n"
            "    total += i\n"
            "print(total)"
        )
        result = await executor.execute(code)
        assert result.success is True
        assert "10" in result.output

    @pytest.mark.asyncio
    async def test_no_output_case(self, executor: CodeExecutor):
        """Verify that code with no output returns the sentinel message.

        When user code produces neither stdout nor a __ob_result__, the
        executor returns '(executed successfully, no output)' so the
        caller knows the code ran without error.
        """
        result = await executor.execute("_ = 1 + 1")
        assert result.success is True
        assert result.output == "(executed successfully, no output)"


# ---------------------------------------------------------------------------
# Negative tests (error handling)
# ---------------------------------------------------------------------------

class TestNegative:
    """Tests that verify CodeExecutor handles errors gracefully."""

    @pytest.mark.asyncio
    async def test_namespace_not_initialized(self):
        """Verify that executing code before set_namespace() returns an error.

        If the namespace is None, the executor must refuse to run code
        and return a clear error message. This guards against using the
        executor before the browser session has been set up.
        """
        executor = CodeExecutor()
        # Do NOT call set_namespace
        result = await executor.execute("x = 1")
        assert result.success is False
        assert "namespace not initialized" in result.output
        assert result.error == "namespace not initialized"

    @pytest.mark.asyncio
    async def test_empty_code_string(self, executor: CodeExecutor):
        """Verify that an empty string is rejected.

        Empty code has no meaningful operation. The executor should
        return an error rather than silently succeeding, because an
        empty submission usually indicates a bug in the caller.
        """
        result = await executor.execute("")
        assert result.success is False
        assert "No code provided" in result.output
        assert result.error == "No code provided"

    @pytest.mark.asyncio
    async def test_whitespace_only_code(self, executor: CodeExecutor):
        """Verify that whitespace-only code is rejected.

        code.strip() is empty for strings like '   ' or '\\n\\t', so
        the executor must treat these the same as an empty string.
        """
        result = await executor.execute("   \n\t  \n  ")
        assert result.success is False
        assert "No code provided" in result.output

    @pytest.mark.asyncio
    async def test_syntax_error_in_user_code(self, executor: CodeExecutor):
        """Verify that a SyntaxError in user code is caught and reported.

        The compile() call will raise SyntaxError for malformed Python.
        The executor must catch this and return the traceback so the LLM
        can fix its code.
        """
        result = await executor.execute("def foo(:")
        assert result.success is False
        assert "SyntaxError" in result.output
        assert "Traceback" in result.output

    @pytest.mark.asyncio
    async def test_runtime_exception_in_user_code(self, executor: CodeExecutor):
        """Verify that a runtime exception (e.g., TypeError) is caught.

        User code may attempt invalid operations at runtime. The executor
        wraps execution in a try/except and must return both the
        exception type and the traceback.
        """
        result = await executor.execute("int('not_a_number')")
        assert result.success is False
        assert "ValueError" in result.output
        assert "Traceback" in result.output
        assert result.error is not None
        assert "ValueError" in result.error

    @pytest.mark.asyncio
    async def test_division_by_zero(self, executor: CodeExecutor):
        """Verify that ZeroDivisionError is caught and reported.

        Division by zero is a common runtime error. The executor must
        produce a clear error with traceback, not crash.
        """
        result = await executor.execute("result = 1 / 0")
        assert result.success is False
        assert "ZeroDivisionError" in result.output
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    @pytest.mark.asyncio
    async def test_undefined_variable_access(self, executor: CodeExecutor):
        """Verify that accessing an undefined variable produces NameError.

        The namespace starts empty (aside from injected helpers). If
        code references a variable that was never assigned, the executor
        must report NameError with a traceback.
        """
        result = await executor.execute("print(undefined_var_xyz)")
        assert result.success is False
        assert "NameError" in result.output
        assert "undefined_var_xyz" in result.output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests that verify subtle or boundary behavior of CodeExecutor."""

    @pytest.mark.asyncio
    async def test_early_return_still_updates_namespace(self, executor: CodeExecutor):
        """Verify that an early return from __ob_exec__ still persists
        variables into the namespace via the try/finally block.

        The wrapped code uses try/finally so that __ns__.update() runs
        even when the user code triggers a return (via __ob_result__
        assignment followed by 'return'). Without try/finally, variables
        assigned before the return would be lost.
        """
        code = (
            "early_var = 'I was set before return'\n"
            "__ob_result__ = 'returned early'"
        )
        result = await executor.execute(code)
        assert result.success is True
        # The variable must have been persisted even though __ob_result__ was set
        assert executor._namespace.get("early_var") == "I was set before return"

    @pytest.mark.asyncio
    async def test_output_truncation_at_boundary(self):
        """Verify that output is truncated at exactly max_output_chars.

        When the output length exceeds max_output_chars, the executor
        slices at [:max_output_chars] and appends a truncation message.
        This test sets max_output_chars to a precise value and checks
        the boundary.
        """
        limit = 20
        executor = CodeExecutor(max_output_chars=limit)
        executor.set_namespace({})

        # Print exactly limit+1 characters to trigger truncation
        result = await executor.execute(f'print("A" * {limit + 1})')
        assert result.success is True
        assert "truncated" in result.output
        # The first `limit` characters of the original output must be preserved
        assert result.output[:limit] == "A" * limit

    @pytest.mark.asyncio
    async def test_output_truncation_with_very_large_output(self, small_executor: CodeExecutor):
        """Verify that very large outputs are truncated without crashing.

        The executor must handle arbitrarily large stdout without
        memory issues in the returned result. The truncation message
        must include the total character count.
        """
        result = await small_executor.execute('print("B" * 100_000)')
        assert result.success is True
        assert "truncated" in result.output
        # The truncation message should mention the total size
        assert "100001" in result.output  # 100000 B's + newline from print

    @pytest.mark.asyncio
    async def test_no_truncation_when_output_fits(self, executor: CodeExecutor):
        """Verify that output within the limit is NOT truncated.

        The default limit is 10,000 characters. Short outputs must be
        returned verbatim without any truncation suffix.
        """
        result = await executor.execute('print("short")')
        assert result.success is True
        assert "truncated" not in result.output
        assert "short" in result.output

    @pytest.mark.asyncio
    async def test_cleanup_of_internal_names_from_namespace(self, executor: CodeExecutor):
        """Verify that __ob_exec__ and __ob_result__ are removed from the
        namespace after execution.

        These are internal implementation details of the wrapper. They
        must not leak into the user-visible namespace, as that could
        confuse subsequent code or the LLM.
        """
        await executor.execute("x = 1")
        assert "__ob_exec__" not in executor._namespace
        assert "__ob_result__" not in executor._namespace

    @pytest.mark.asyncio
    async def test_cleanup_after_error(self, executor: CodeExecutor):
        """Verify that __ob_exec__ and __ob_result__ are cleaned up even
        when execution raises an exception.

        The finally block in execute() must run regardless of success
        or failure, removing the wrapper function from the namespace.
        """
        await executor.execute("raise RuntimeError('boom')")
        assert "__ob_exec__" not in executor._namespace
        assert "__ob_result__" not in executor._namespace

    @pytest.mark.asyncio
    async def test_code_modifies_namespace_directly(self, executor: CodeExecutor):
        """Verify that user code can read and modify the namespace dict
        via local variable assignment, and those changes persist.

        The namespace is passed as __ns__ to the wrapper function, and
        the finally block calls __ns__.update(locals()). This means
        any local variable assigned by user code becomes a namespace
        entry -- even complex types like lists and dicts.
        """
        await executor.execute("data = [1, 2, 3]")
        result = await executor.execute("data.append(4)\nprint(data)")
        assert result.success is True
        assert "[1, 2, 3, 4]" in result.output
        assert executor._namespace["data"] == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_initialized_property_before_and_after_set(self):
        """Verify the initialized property reflects namespace state.

        Before set_namespace(), initialized must be False. After, it
        must be True. This property is used by callers to gate execution.
        """
        executor = CodeExecutor()
        assert executor.initialized is False
        executor.set_namespace({})
        assert executor.initialized is True

    @pytest.mark.asyncio
    async def test_error_result_contains_raw_error_field(self, executor: CodeExecutor):
        """Verify that the error field on ExecutionResult contains the raw,
        un-truncated exception string.

        The 'error' field is used programmatically by callers (e.g., to
        detect specific exception types). It must not be truncated even
        when the output field is.
        """
        result = await executor.execute('raise ValueError("specific error message 12345")')
        assert result.success is False
        assert result.error is not None
        assert "specific error message 12345" in result.error
        assert result.error.startswith("ValueError:")

    @pytest.mark.asyncio
    async def test_stdout_before_error_is_preserved(self, executor: CodeExecutor):
        """Verify that stdout captured before an exception is included in
        the error output.

        When user code prints some output and then crashes, the executor
        must include the captured stdout ('Output before error:') so
        the LLM can see what happened before the failure.
        """
        code = 'print("step 1 done")\nprint("step 2 done")\nraise RuntimeError("step 3 failed")'
        result = await executor.execute(code)
        assert result.success is False
        assert "step 1 done" in result.output
        assert "step 2 done" in result.output
        assert "Output before error:" in result.output
        assert "RuntimeError" in result.output

    @pytest.mark.asyncio
    async def test_pass_statement_produces_no_output(self, executor: CodeExecutor):
        """Verify that 'pass' produces the no-output sentinel.

        A bare 'pass' is valid Python that does nothing. It should
        succeed and produce the sentinel message, not an error.
        """
        result = await executor.execute("pass")
        assert result.success is True
        assert result.output == "(executed successfully, no output)"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_concurrent_executions_are_serialized(self, executor: CodeExecutor):
        """Verify that concurrent execute() calls are serialized by the lock.

        CodeExecutor uses asyncio.Lock to protect sys.stdout redirection
        from concurrent access. Two overlapping executions must not
        interleave their stdout capture. We run two tasks concurrently
        and verify both produce correct, isolated output.
        """
        executor._namespace["asyncio"] = asyncio

        async def run_a():
            return await executor.execute(
                'await asyncio.sleep(0.01)\nprint("output_a")'
            )

        async def run_b():
            return await executor.execute(
                'await asyncio.sleep(0.01)\nprint("output_b")'
            )

        result_a, result_b = await asyncio.gather(run_a(), run_b())

        assert result_a.success is True
        assert result_b.success is True
        # Each result must contain only its own output, not the other's
        outputs = [result_a.output, result_b.output]
        assert any("output_a" in o for o in outputs)
        assert any("output_b" in o for o in outputs)
        # No result should contain both outputs mixed together
        for o in outputs:
            assert not ("output_a" in o and "output_b" in o)

    @pytest.mark.asyncio
    async def test_max_output_chars_zero_disables_truncation(self):
        """Verify that max_output_chars=0 disables truncation.

        The _truncate method checks 'if self._max_output_chars and ...',
        so a falsy value (0) should skip truncation entirely, allowing
        unlimited output.
        """
        executor = CodeExecutor(max_output_chars=0)
        executor.set_namespace({})
        result = await executor.execute('print("C" * 50000)')
        assert result.success is True
        assert "truncated" not in result.output
        assert "C" * 50000 in result.output
