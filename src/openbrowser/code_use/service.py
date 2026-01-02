"""Code-use agent service - Jupyter notebook-like code execution for browser automation.

This module provides the CodeAgent class, which orchestrates an LLM to write
and execute Python code for browser automation. The agent operates in a
Jupyter notebook-like environment where code is executed in a persistent
namespace with browser control functions available.

Key Features:
    - LLM-driven Python code generation and execution
    - Persistent namespace across execution steps
    - Browser state awareness with screenshots and DOM
    - Automatic URL extraction and navigation
    - Error recovery and retry logic
    - Session export to Jupyter notebooks

Example:
    ```python
    from src.openbrowser.code_use import CodeAgent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4")
    agent = CodeAgent(
        task="Go to amazon.com and search for 'laptop'",
        llm=llm,
    )
    session = await agent.run()
    ```
"""

import ast
import asyncio
import datetime
import io
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.openbrowser.browser.profile import BrowserProfile
from src.openbrowser.browser.session import BrowserSession
from src.openbrowser.tools.actions import CodeAgentTools, Tools

from .formatting import format_browser_state_for_llm
from .namespace import EvaluateError, create_namespace
from .utils import detect_token_limit_issue, extract_code_blocks, extract_url_from_task, truncate_message_content
from .views import (
    CodeAgentHistory,
    CodeAgentModelOutput,
    CodeAgentResult,
    CodeAgentState,
    CodeAgentStepMetadata,
    ExecutionStatus,
    NotebookSession,
)

logger = logging.getLogger(__name__)


class CodeAgent:
    """Agent that executes Python code in a notebook-like environment for browser automation.

    This agent provides a Jupyter notebook-like interface where the LLM writes Python code
    that gets executed in a persistent namespace with browser control functions available.
    The agent maintains conversation history with the LLM, provides browser state updates,
    and handles error recovery.

    The execution flow is:
    1. Agent receives a task description
    2. Browser session is initialized (if not provided)
    3. Initial URL is extracted from task and navigated to (if present)
    4. Main loop: LLM generates code -> Code executes -> Results fed back
    5. Loop continues until done() is called or limits are reached

    Attributes:
        task: The task description for the agent to complete.
        llm: The language model instance for code generation.
        browser_session: The browser session for automation.
        tools: Tools instance containing available browser actions.
        sensitive_data: Dictionary of sensitive data for secure injection.
        max_steps: Maximum number of execution steps allowed.
        max_failures: Maximum consecutive errors before termination.
        use_vision: Whether to include screenshots in LLM messages.
        session: The NotebookSession containing all executed cells.
        namespace: The execution namespace with browser tools.
        complete_history: List of CodeAgentHistory items for each step.
        id: Unique identifier for this agent instance.
        agent_directory: Temporary directory for screenshots and artifacts.

    Example:
        ```python
        async with CodeAgent(
            task="Search for Python tutorials on Google",
            llm=ChatOpenAI(model="gpt-4"),
        ) as agent:
            session = await agent.run(max_steps=20)
            print(f"Completed in {len(session.cells)} steps")
        ```
    """

    def __init__(
        self,
        task: str,
        # Optional parameters
        llm: Any | None = None,
        browser_session: BrowserSession | None = None,
        browser: BrowserSession | None = None,  # Alias for browser_session
        tools: Tools | None = None,
        controller: Tools | None = None,  # Alias for tools
        # Agent settings
        sensitive_data: dict[str, str | dict[str, str]] | None = None,
        max_steps: int = 100,
        max_failures: int = 8,
        use_vision: bool = True,
        **kwargs,
    ):
        """Initialize the code-use agent.

        Creates a new CodeAgent instance configured to complete the given task.
        The agent will use the provided LLM to generate Python code and execute
        it in a browser automation environment.

        Args:
            task: The task description for the agent to complete. May contain
                URLs which will be automatically extracted and navigated to.
            llm: LLM instance (required). Must support ainvoke() method.
                Compatible with LangChain chat models.
            browser_session: Optional pre-configured browser session.
                If not provided, a new session is created automatically.
            browser: Alias for browser_session (cleaner API). Cannot be used
                together with browser_session.
            tools: Optional Tools instance with custom action registry.
                If not provided, CodeAgentTools are created automatically.
            controller: Alias for tools parameter. Cannot be used together
                with tools.
            sensitive_data: Optional dictionary of sensitive data (passwords,
                API keys, etc.) to inject into the namespace securely.
            max_steps: Maximum number of execution steps. Defaults to 100.
            max_failures: Maximum consecutive errors before automatic
                termination. Defaults to 8.
            use_vision: Whether to include screenshots in LLM messages for
                visual understanding. Defaults to True.
            **kwargs: Additional keyword arguments are logged and ignored
                for forward compatibility.

        Raises:
            ValueError: If llm is None, or if both browser/browser_session
                or both controller/tools are specified.

        Example:
            ```python
            # Minimal usage
            agent = CodeAgent(
                task="Go to google.com",
                llm=ChatOpenAI(model="gpt-4")
            )

            # With custom browser
            browser = BrowserSession(headless=False)
            agent = CodeAgent(
                task="Fill out form",
                llm=llm,
                browser=browser,
                sensitive_data={"password": "secret"}
            )
            ```
        """
        # Log and ignore unknown kwargs for compatibility
        if kwargs:
            logger.debug(f"Ignoring additional kwargs for CodeAgent compatibility: {list(kwargs.keys())}")

        if llm is None:
            raise ValueError("llm parameter is required for CodeAgent")

        # Handle browser vs browser_session parameter (browser takes precedence)
        if browser and browser_session:
            raise ValueError('Cannot specify both "browser" and "browser_session" parameters.')
        browser_session = browser or browser_session

        # Handle controller vs tools parameter (controller takes precedence)
        if controller and tools:
            raise ValueError('Cannot specify both "controller" and "tools" parameters.')
        tools = controller or tools

        # Store browser_profile for creating browser session if needed
        self._browser_profile_for_init = BrowserProfile() if browser_session is None else None

        self.task = task
        self.llm = llm
        self.browser_session = browser_session
        self.tools = tools
        self.sensitive_data = sensitive_data
        self.max_steps = max_steps
        self.max_failures = max_failures
        self.use_vision = use_vision

        self.session = NotebookSession()
        self.namespace: dict[str, Any] = {}
        self._llm_messages: list[dict[str, Any]] = []  # Internal LLM conversation history
        self.complete_history: list[CodeAgentHistory] = []  # Type-safe history
        self._last_browser_state_text: str | None = None
        self._last_screenshot: str | None = None
        self._consecutive_errors = 0
        self._step_start_time = 0.0

        # Initialize agent ID and directory
        self.id = str(uuid4())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_tmp = Path("/tmp")
        self.agent_directory = base_tmp / f"browser_use_code_agent_{self.id}_{timestamp}"

    async def run(self, max_steps: int | None = None) -> NotebookSession:
        """Run the agent to complete the task.

        Executes the main agent loop: getting code from the LLM, executing it,
        providing feedback, and repeating until the task is complete or limits
        are reached. The browser is automatically started if not already running.

        Args:
            max_steps: Optional override for maximum number of steps.
                If provided, overrides the value set in __init__.

        Returns:
            The NotebookSession containing all executed code cells, their
            outputs, and execution status.

        Raises:
            Exception: Various exceptions may be raised during execution.
                The agent attempts to handle errors gracefully and continue.

        Note:
            The agent will automatically:
            - Start a browser session if not provided
            - Extract and navigate to URLs in the task
            - Warn the LLM when approaching step/error limits
            - Close the browser session when complete

        Example:
            ```python
            agent = CodeAgent(task="Search Google", llm=llm)

            # Run with default max_steps
            session = await agent.run()

            # Run with custom limit
            session = await agent.run(max_steps=10)

            # Check results
            for cell in session.cells:
                print(f"[{cell.status}] {cell.source[:50]}...")
            ```
        """
        # Use override if provided
        steps_to_run = max_steps if max_steps is not None else self.max_steps
        self.max_steps = steps_to_run

        # Start browser if not provided
        if self.browser_session is None:
            assert self._browser_profile_for_init is not None
            self.browser_session = BrowserSession(browser_profile=self._browser_profile_for_init)
            await self.browser_session.start()

        # Initialize tools if not provided
        if self.tools is None:
            self.tools = CodeAgentTools(self.browser_session)

        # Create namespace with all tools
        self.namespace = create_namespace(
            browser_session=self.browser_session,
            tools=self.tools,
            sensitive_data=self.sensitive_data,
        )

        # Initialize conversation with task
        self._llm_messages.append({"role": "user", "content": f"Task: {self.task}"})

        # Extract URL from task and navigate if found
        initial_url = extract_url_from_task(self.task)
        if initial_url:
            try:
                logger.info(f"Extracted URL from task, navigating to: {initial_url}")
                # Use the navigate action from namespace
                await self.namespace["go_to_url"](initial_url)
                # Wait for page load
                await asyncio.sleep(2)

                # Record this navigation as a cell in the notebook
                nav_code = f"await go_to_url('{initial_url}')"
                cell = self.session.add_cell(source=nav_code)
                cell.status = ExecutionStatus.SUCCESS
                cell.execution_count = self.session.increment_execution_count()
                cell.output = f"Navigated to {initial_url}"

            except Exception as e:
                logger.warning(f"Failed to navigate to extracted URL {initial_url}: {e}")
                # Record failed navigation as error cell
                nav_code = f"await go_to_url('{initial_url}')"
                cell = self.session.add_cell(source=nav_code)
                cell.status = ExecutionStatus.ERROR
                cell.execution_count = self.session.increment_execution_count()
                cell.error = str(e)

        # Get initial browser state before first LLM call
        if self.browser_session:
            try:
                browser_state_text, screenshot = await self._get_browser_state()
                self._last_browser_state_text = browser_state_text
                self._last_screenshot = screenshot
            except Exception as e:
                logger.warning(f"Failed to get initial browser state: {e}")

        # Main execution loop
        for step in range(self.max_steps):
            logger.info(f"\n\n\nStep {step + 1}/{self.max_steps}")

            # Start timing this step
            self._step_start_time = datetime.datetime.now().timestamp()

            # Check if we're approaching the step limit or error limit
            steps_remaining = self.max_steps - step - 1
            errors_remaining = self.max_failures - self._consecutive_errors

            should_warn = (
                steps_remaining <= 1  # Last step or next to last
                or errors_remaining <= 1  # One more error will terminate
                or (steps_remaining <= 2 and self._consecutive_errors >= 2)
            )

            if should_warn:
                warning_message = (
                    f"\n\n⚠️ CRITICAL WARNING: You are approaching execution limits!\n"
                    f"- Steps remaining: {steps_remaining + 1}\n"
                    f"- Consecutive errors: {self._consecutive_errors}/{self.max_failures}\n\n"
                    f"YOU MUST call done() in your NEXT response, even if the task is incomplete:\n"
                    f"- Set success=False if you couldn't complete the task\n"
                    f"- Return EVERYTHING you found so far (partial data is better than nothing)\n"
                    f"Without done(), the user will receive NOTHING."
                )
                self._llm_messages.append({"role": "user", "content": warning_message})

            try:
                # Fetch fresh browser state right before LLM call
                if not self._last_browser_state_text and self.browser_session:
                    try:
                        logger.debug("Fetching browser state before LLM call...")
                        browser_state_text, screenshot = await self._get_browser_state()
                        self._last_browser_state_text = browser_state_text
                        self._last_screenshot = screenshot
                    except Exception as e:
                        logger.warning(f"Failed to get browser state before LLM call: {e}")

                # Get code from LLM
                try:
                    code, full_llm_response = await self._get_code_from_llm()
                except Exception as llm_error:
                    self._consecutive_errors += 1
                    logger.warning(
                        f"LLM call failed (consecutive errors: {self._consecutive_errors}/{self.max_failures}): {llm_error}"
                    )

                    if self._consecutive_errors >= self.max_failures:
                        logger.error(f"Terminating: {self.max_failures} consecutive LLM failures")
                        break

                    await asyncio.sleep(1)
                    continue

                if not code or code.strip() == "":
                    # If task is already done, empty code is fine
                    if self._is_task_done():
                        logger.info("Task already marked as done, LLM provided explanation without code")
                        await self._add_step_to_complete_history(
                            model_output_code="",
                            full_llm_response=full_llm_response,
                            output=full_llm_response,
                            error=None,
                            screenshot_path=await self._capture_screenshot(step + 1),
                        )
                        break

                    logger.warning("LLM returned empty code")
                    self._consecutive_errors += 1

                    if self.browser_session:
                        try:
                            browser_state_text, screenshot = await self._get_browser_state()
                            self._last_browser_state_text = browser_state_text
                            self._last_screenshot = screenshot
                        except Exception as e:
                            logger.warning(f"Failed to get new browser state: {e}")
                    continue

                # Execute code blocks sequentially
                all_blocks = self.namespace.get("_all_code_blocks", {})
                python_blocks = [k for k in sorted(all_blocks.keys()) if k.startswith("python_")]

                if len(python_blocks) > 1:
                    output = None
                    error = None

                    for i, block_key in enumerate(python_blocks):
                        logger.info(f"Executing Python block {i + 1}/{len(python_blocks)}")
                        block_code = all_blocks[block_key]
                        block_output, block_error, _ = await self._execute_code(block_code)

                        if block_output:
                            output = (output or "") + block_output
                        if block_error:
                            error = block_error
                            break
                else:
                    output, error, _ = await self._execute_code(code)

                # Track consecutive errors
                if error:
                    self._consecutive_errors += 1
                    logger.warning(f"Consecutive errors: {self._consecutive_errors}/{self.max_failures}")

                    if self._consecutive_errors >= self.max_failures:
                        logger.error(f"Terminating: {self.max_failures} consecutive errors reached.")
                        await self._add_step_to_complete_history(
                            model_output_code=code,
                            full_llm_response=f"[Terminated after {self.max_failures} consecutive errors]",
                            output=None,
                            error=f"Auto-terminated: {self.max_failures} consecutive errors without progress",
                            screenshot_path=None,
                        )
                        break
                else:
                    self._consecutive_errors = 0

                # Check if task is done
                if self._is_task_done():
                    final_result: str | None = self.namespace.get("_task_result")
                    if final_result:
                        output = final_result

                if output:
                    if self._is_task_done():
                        logger.info(f"✓ Task completed - Final output from done():\n{output[:300] if len(output) > 300 else output}")
                    else:
                        logger.info(f"Code output:\n{output}")

                # Take screenshot for tracking
                screenshot_path = await self._capture_screenshot(step + 1)

                # Add step to complete_history
                await self._add_step_to_complete_history(
                    model_output_code=code,
                    full_llm_response=full_llm_response,
                    output=output,
                    error=error,
                    screenshot_path=screenshot_path,
                )

                # Check if task is done
                if self._is_task_done():
                    final_result: str | None = self.namespace.get("_task_result", output)
                    logger.info("Task completed successfully")
                    if final_result:
                        logger.info(f"Final result: {final_result}")
                    break

                # Add result to LLM messages for next iteration
                result_message = self._format_execution_result(code, output, error, current_step=step + 1)
                truncated_result = truncate_message_content(result_message)
                self._llm_messages.append({"role": "user", "content": truncated_result})

            except Exception as e:
                logger.error(f"Error in step {step + 1}: {e}")
                traceback.print_exc()
                break
        else:
            logger.warning(f"Maximum steps ({self.max_steps}) reached without task completion")

        # Log final summary if task was completed
        if self._is_task_done():
            logger.info("\n" + "=" * 60)
            logger.info("TASK COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            final_result: str | None = self.namespace.get("_task_result")
            if final_result:
                logger.info(f"\nFinal Output:\n{final_result}")
            logger.info("=" * 60 + "\n")

        # Close browser
        await self.close()

        return self.session

    async def _get_code_from_llm(self) -> tuple[str, str]:
        """Get Python code from the LLM.

        Sends the current conversation history and browser state to the LLM
        and extracts Python code from the response. Also handles:
        - Token limit detection and recovery
        - Code block extraction (Python, JavaScript, bash)
        - Injecting non-Python blocks into the namespace

        Returns:
            A tuple of (extracted_code, full_llm_response) where:
            - extracted_code: The Python code to execute (may be empty)
            - full_llm_response: The complete LLM response text

        Raises:
            Exception: If the LLM call fails. The caller should handle
                this and increment the error counter.
        """
        # Prepare messages for this request
        messages_to_send = self._llm_messages.copy()

        if self._last_browser_state_text:
            if self.use_vision and self._last_screenshot:
                # Build content with text + screenshot
                content = [
                    {"type": "text", "text": self._last_browser_state_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self._last_screenshot}"},
                    },
                ]
                messages_to_send.append({"role": "user", "content": content})
            else:
                messages_to_send.append({"role": "user", "content": self._last_browser_state_text})

            # Clear browser state after including it
            self._last_browser_state_text = None
            self._last_screenshot = None

        # Call LLM with message history
        response = await self.llm.ainvoke(messages_to_send)

        # Handle different response types
        if hasattr(response, "content"):
            full_response = response.content
        elif isinstance(response, str):
            full_response = response
        else:
            full_response = str(response)

        logger.info(f"LLM Response:\n{full_response}")

        # Check for token limit issues
        max_tokens = getattr(self.llm, "max_tokens", None)
        completion_tokens = None
        stop_reason = None

        if hasattr(response, "usage") and response.usage:
            completion_tokens = getattr(response.usage, "completion_tokens", None)
        if hasattr(response, "stop_reason"):
            stop_reason = response.stop_reason

        is_problematic, issue_message = detect_token_limit_issue(
            completion=full_response,
            completion_tokens=completion_tokens,
            max_tokens=max_tokens,
            stop_reason=stop_reason,
        )

        if is_problematic:
            logger.warning(f"Token limit issue detected: {issue_message}")
            recovery_prompt = (
                f"Your previous response hit a token limit or became repetitive: {issue_message}\n\n"
                "Please write a SHORT plan (2 sentences) for what to do next, then execute ONE simple action."
            )
            self._llm_messages.append({"role": "user", "content": recovery_prompt})
            return "", f"[Token limit error: {issue_message}]"

        # Extract code blocks from response
        code_blocks = extract_code_blocks(full_response)

        # Inject non-python blocks into namespace as variables
        if "_code_block_vars" not in self.namespace:
            self.namespace["_code_block_vars"] = set()

        for block_type, block_content in code_blocks.items():
            if not block_type.startswith("python"):
                self.namespace[block_type] = block_content
                self.namespace["_code_block_vars"].add(block_type)
                print(f"→ Code block variable: {block_type} (str, {len(block_content)} chars)")
                logger.debug(f"Injected {block_type} block into namespace ({len(block_content)} chars)")

        # Store all code blocks for sequential execution
        self.namespace["_all_code_blocks"] = code_blocks

        # Get Python code if it exists
        code = code_blocks.get("python", full_response)

        # Add to LLM messages
        truncated_completion = truncate_message_content(full_response)
        self._llm_messages.append({"role": "assistant", "content": truncated_completion})

        return code, full_response

    async def _execute_code(self, code: str) -> tuple[str | None, str | None, str | None]:
        """Execute Python code in the namespace.

        Creates a new code cell, executes the provided Python code in the
        agent's namespace, and captures any output or errors. Handles both
        sync and async code, properly managing await expressions.

        Args:
            code: The Python code to execute. Can contain await expressions
                which will be automatically wrapped in an async function.

        Returns:
            A tuple of (output, error, browser_state) where:
            - output: Captured stdout from the execution, or None
            - error: Error message if execution failed, or None
            - browser_state: Always None (reserved for future use)

        Note:
            - Variables defined in the code are added to the namespace
            - Global declarations are properly handled
            - EvaluateError from JavaScript stops execution immediately
            - The cell is recorded in the session with its status
        """
        # Create new cell
        cell = self.session.add_cell(source=code)
        cell.status = ExecutionStatus.RUNNING
        cell.execution_count = self.session.increment_execution_count()

        output = None
        error = None

        try:
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Add asyncio to namespace if not already there
                if "asyncio" not in self.namespace:
                    self.namespace["asyncio"] = asyncio

                # Store the current code in namespace for done() validation
                self.namespace["_current_cell_code"] = code
                self.namespace["_consecutive_errors"] = self._consecutive_errors

                # Check if code contains await expressions
                try:
                    tree = ast.parse(code, mode="exec")
                    has_await = any(isinstance(node, (ast.Await, ast.AsyncWith, ast.AsyncFor)) for node in ast.walk(tree))
                except SyntaxError:
                    has_await = False

                if has_await:
                    # Wrap in async function
                    try:
                        assigned_names = set()
                        user_global_names = set()

                        for node in ast.walk(tree):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if isinstance(target, ast.Name):
                                        assigned_names.add(target.id)
                            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                                assigned_names.add(node.target.id)
                            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                                if hasattr(node, "target") and isinstance(node.target, ast.Name):
                                    assigned_names.add(node.target.id)
                            elif isinstance(node, ast.Global):
                                user_global_names.update(node.names)

                        # Pre-define globals that don't exist yet
                        for name in user_global_names:
                            if name not in self.namespace:
                                self.namespace[name] = None

                        existing_vars = {name for name in (assigned_names | user_global_names) if name in self.namespace}
                    except Exception:
                        existing_vars = set()

                    global_decl = ""
                    has_global_decl = False
                    if existing_vars:
                        vars_str = ", ".join(sorted(existing_vars))
                        global_decl = f"    global {vars_str}\n"
                        has_global_decl = True

                    indented_code = "\n".join("    " + line if line.strip() else line for line in code.split("\n"))
                    wrapped_code = f"""async def __code_exec__():
{global_decl}{indented_code}
    return locals()

__code_exec_coro__ = __code_exec__()
"""
                    self.namespace["_has_global_decl"] = has_global_decl

                    compiled_code = compile(wrapped_code, "<code>", "exec")
                    exec(compiled_code, self.namespace, self.namespace)

                    coro = self.namespace.get("__code_exec_coro__")
                    if coro:
                        result_locals = await coro
                        if result_locals:
                            for key, value in result_locals.items():
                                if not key.startswith("_"):
                                    self.namespace[key] = value

                        self.namespace.pop("__code_exec_coro__", None)
                        self.namespace.pop("__code_exec__", None)
                else:
                    # Execute directly at module level
                    compiled_code = compile(code, "<code>", "exec")
                    exec(compiled_code, self.namespace, self.namespace)

                # Get output
                output_value = sys.stdout.getvalue()
                if output_value:
                    output = output_value

            finally:
                sys.stdout = old_stdout

            # Wait for page to stabilize
            await asyncio.sleep(0.5)

            cell.status = ExecutionStatus.SUCCESS
            cell.output = output
            cell.browser_state = None

        except EvaluateError as e:
            error = str(e)
            cell.status = ExecutionStatus.ERROR
            cell.error = error
            logger.error(f"Code execution error: {error}")
            await asyncio.sleep(1)
            return output, error, None

        except SyntaxError as e:
            error_msg = e.msg if e.msg else str(e)
            error = f"{type(e).__name__}: {error_msg}"

            if e.text:
                error += f"\n{e.text}"
            elif e.lineno and code:
                lines = code.split("\n")
                if 0 < e.lineno <= len(lines):
                    error += f"\n{lines[e.lineno - 1]}"

            cell.status = ExecutionStatus.ERROR
            cell.error = error
            logger.error(f"Code execution error: {error}")
            await asyncio.sleep(1)

        except Exception as e:
            error_str = str(e)
            error = f"{type(e).__name__}: {error_str}" if error_str else f"{type(e).__name__} occurred"

            cell.status = ExecutionStatus.ERROR
            cell.error = error
            logger.error(f"Code execution error: {error}")
            await asyncio.sleep(1)

        return output, error, None

    async def _get_browser_state(self) -> tuple[str, str | None]:
        """Get the current browser state as formatted text.

        Retrieves the current page URL, title, DOM structure, and optionally
        a screenshot, then formats them for LLM consumption.

        Returns:
            A tuple of (browser_state_text, screenshot_base64) where:
            - browser_state_text: Markdown-formatted browser state
            - screenshot_base64: Base64-encoded JPEG screenshot, or None

        Note:
            Screenshot capture only occurs if use_vision is True.
            DOM structure is simplified for token efficiency.
        """
        if not self.browser_session:
            return "Browser state not available", None

        try:
            # Get current page info
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            # Get URL
            result = await client.send.Runtime.evaluate(
                params={"expression": "window.location.href", "returnByValue": True},
                session_id=session_id,
            )
            url = result.get("result", {}).get("value", "")

            # Get title
            result = await client.send.Runtime.evaluate(
                params={"expression": "document.title", "returnByValue": True},
                session_id=session_id,
            )
            title = result.get("result", {}).get("value", "")

            # Get DOM representation (simplified)
            result = await client.send.Runtime.evaluate(
                params={
                    "expression": """
                    (function() {
                        function getElements(el, depth) {
                            if (depth > 5) return '';
                            var result = '';
                            for (var child of el.children) {
                                var tag = child.tagName.toLowerCase();
                                var text = child.textContent?.trim()?.slice(0, 50) || '';
                                if (['script', 'style', 'noscript'].includes(tag)) continue;
                                result += '  '.repeat(depth) + '<' + tag + '>' + (text ? ' ' + text : '') + '\\n';
                                result += getElements(child, depth + 1);
                            }
                            return result;
                        }
                        return getElements(document.body, 0);
                    })()
                    """,
                    "returnByValue": True,
                },
                session_id=session_id,
            )
            dom_html = result.get("result", {}).get("value", "")

            # Take screenshot if vision enabled
            screenshot = None
            if self.use_vision:
                try:
                    result = await client.send.Page.captureScreenshot(
                        params={"format": "jpeg", "quality": 70},
                        session_id=session_id,
                    )
                    screenshot = result.get("data")
                except Exception as e:
                    logger.debug(f"Failed to capture screenshot: {e}")

            # Format browser state
            browser_state_text = await format_browser_state_for_llm(
                url=url,
                title=title,
                dom_html=dom_html,
                namespace=self.namespace,
                browser_session=self.browser_session,
                screenshot=screenshot,
            )

            return browser_state_text, screenshot

        except Exception as e:
            logger.error(f"Failed to get browser state: {e}")
            return f"Error getting browser state: {e}", None

    def _format_execution_result(self, code: str, output: str | None, error: str | None, current_step: int | None = None) -> str:
        """Format the execution result for the LLM.

        Creates a summary message of the code execution outcome to be
        added to the conversation history for the next LLM call.

        Args:
            code: The executed Python code (for context).
            output: Standard output from execution, if any.
            error: Error message from execution, if any.
            current_step: Current step number for progress display.

        Returns:
            Formatted string containing step progress, errors, and output.
            Output is truncated to 10000 characters if too long.
        """
        result = []

        if current_step is not None:
            progress_header = f"Step {current_step}/{self.max_steps} executed"
            if error and self._consecutive_errors > 0:
                progress_header += f" | Consecutive failures: {self._consecutive_errors}/{self.max_failures}"
            result.append(progress_header)

        if error:
            result.append(f"Error: {error}")

        if output:
            if len(output) > 10000:
                output = output[:9950] + "\n[Truncated after 10000 characters]"
            result.append(f"Output: {output}")
        if len(result) == 0:
            result.append("Executed")
        return "\n".join(result)

    def _is_task_done(self) -> bool:
        """Check if the task is marked as done in the namespace.

        The task is considered done when the done() action has been called
        successfully, which sets the _task_done flag in the namespace.

        Returns:
            True if done() has been called, False otherwise.
        """
        return self.namespace.get("_task_done", False)

    async def _capture_screenshot(self, step_number: int) -> str | None:
        """Capture and store a screenshot for tracking.

        Takes a screenshot of the current browser state and saves it to
        the agent's temporary directory for later review.

        Args:
            step_number: The current step number, used in the filename.

        Returns:
            The file path to the saved screenshot, or None if capture failed.
        """
        if not self.browser_session:
            return None

        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session()
            client = cdp_session.cdp_client
            session_id = cdp_session.session_id

            result = await client.send.Page.captureScreenshot(
                params={"format": "jpeg", "quality": 70},
                session_id=session_id,
            )
            screenshot_data = result.get("data")
            if screenshot_data:
                # Ensure directory exists
                self.agent_directory.mkdir(parents=True, exist_ok=True)
                screenshot_path = self.agent_directory / f"step_{step_number}.jpg"

                import base64

                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(screenshot_data))

                return str(screenshot_path)
        except Exception as e:
            logger.warning(f"Failed to capture screenshot for step {step_number}: {e}")
            return None

    async def _add_step_to_complete_history(
        self,
        model_output_code: str,
        full_llm_response: str,
        output: str | None,
        error: str | None,
        screenshot_path: str | None,
    ) -> None:
        """Add a step to the complete_history using type-safe models.

        Records a complete step of the agent's execution, including the
        LLM's response, execution results, and browser state.

        Args:
            model_output_code: The extracted Python code from the LLM.
            full_llm_response: The complete LLM response text.
            output: Standard output from code execution, if any.
            error: Error message from execution, if any.
            screenshot_path: Path to the captured screenshot, if any.
        """
        # Get current browser URL and title for state
        url: str | None = None
        title: str | None = None
        if self.browser_session:
            try:
                cdp_session = await self.browser_session.get_or_create_cdp_session()
                result = await cdp_session.cdp_client.send.Runtime.evaluate(
                    params={"expression": "window.location.href", "returnByValue": True},
                    session_id=cdp_session.session_id,
                )
                url = result.get("result", {}).get("value")
                result = await cdp_session.cdp_client.send.Runtime.evaluate(
                    params={"expression": "document.title", "returnByValue": True},
                    session_id=cdp_session.session_id,
                )
                title = result.get("result", {}).get("value")
            except Exception as e:
                logger.debug(f"Failed to get browser URL/title for history: {e}")

        # Check if this is a done result
        is_done = self._is_task_done()

        # Get self-reported success from done() call if task is done
        self_reported_success: bool | None = None
        if is_done:
            task_success = self.namespace.get("_task_success")
            self_reported_success = task_success if isinstance(task_success, bool) else None

        # Create result entry
        result_entry = CodeAgentResult(
            extracted_content=output if output else None,
            error=error if error else None,
            is_done=is_done,
            success=self_reported_success,
        )

        # Create state entry
        state_entry = CodeAgentState(url=url, title=title, screenshot_path=screenshot_path)

        # Create metadata entry
        step_end_time = datetime.datetime.now().timestamp()
        metadata_entry = CodeAgentStepMetadata(
            input_tokens=None,
            output_tokens=None,
            step_start_time=self._step_start_time,
            step_end_time=step_end_time,
        )

        # Create model output entry
        model_output_entry: CodeAgentModelOutput | None = None
        if model_output_code or full_llm_response:
            model_output_entry = CodeAgentModelOutput(
                model_output=model_output_code if model_output_code else "",
                full_response=full_llm_response if full_llm_response else "",
            )

        # Create history entry
        history_entry = CodeAgentHistory(
            model_output=model_output_entry,
            result=[result_entry],
            state=state_entry,
            metadata=metadata_entry,
            screenshot_path=screenshot_path,
        )

        self.complete_history.append(history_entry)

    def screenshot_paths(self, n_last: int | None = None) -> list[str | None]:
        """Get screenshot paths from the complete history.

        Retrieves file paths to screenshots captured during agent execution.

        Args:
            n_last: Optional limit to return only the last N screenshots.
                If None, returns all screenshot paths.

        Returns:
            List of screenshot file paths. May contain None values for
            steps where screenshot capture failed.
        """
        paths = [step.screenshot_path for step in self.complete_history]

        if n_last is not None:
            return paths[-n_last:] if len(paths) > n_last else paths

        return paths

    @property
    def history(self) -> Any:
        """Compatibility property for accessing execution history.

        Provides backward compatibility with code expecting a history
        object with a .history attribute containing the step list.

        Returns:
            A mock object with a .history attribute containing complete_history.
        """

        class MockAgentHistoryList:
            def __init__(self, complete_history: list[CodeAgentHistory]):
                self.history = complete_history

        return MockAgentHistoryList(self.complete_history)

    async def close(self) -> None:
        """Close the browser session and clean up resources.

        Should be called when the agent is no longer needed. This is
        automatically called by the run() method and the async context
        manager exit.
        """
        if self.browser_session:
            try:
                await self.browser_session.stop()
            except Exception as e:
                logger.warning(f"Error closing browser session: {e}")

    async def __aenter__(self) -> "CodeAgent":
        """Async context manager entry.

        Returns:
            The CodeAgent instance for use in the async with block.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.

        Ensures the browser session is closed even if an exception occurred.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        await self.close()

