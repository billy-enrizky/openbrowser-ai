"""Namespace initialization for code-use mode.

This module creates a namespace with all browser tools available as functions,
similar to a Jupyter notebook environment.
"""

import asyncio
import csv
import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.openbrowser.browser.session import BrowserSession
from src.openbrowser.tools.actions import Tools

logger = logging.getLogger(__name__)


def _strip_js_comments(js_code: str) -> str:
    """
    Remove JavaScript comments before CDP evaluation.
    CDP's Runtime.evaluate doesn't handle comments in all contexts.

    Args:
        js_code: JavaScript code potentially containing comments

    Returns:
        JavaScript code with comments stripped
    """
    # Remove multi-line comments (/* ... */)
    js_code = re.sub(r"/\*.*?\*/", "", js_code, flags=re.DOTALL)

    # Remove single-line comments - only lines that START with // (after whitespace)
    js_code = re.sub(r"^\s*//.*$", "", js_code, flags=re.MULTILINE)

    return js_code


class EvaluateError(Exception):
    """Special exception raised by evaluate() to stop Python execution immediately."""

    pass


async def evaluate(code: str, browser_session: BrowserSession) -> Any:
    """
    Execute JavaScript code in the browser and return the result.

    Args:
        code: JavaScript code to execute (must be wrapped in IIFE)

    Returns:
        The result of the JavaScript execution

    Raises:
        EvaluateError: If JavaScript execution fails. This stops Python execution immediately.

    Example:
        result = await evaluate('''
        (function(){
            return Array.from(document.querySelectorAll('.product')).map(p => ({
                name: p.querySelector('.name').textContent,
                price: p.querySelector('.price').textContent
            }))
        })()
        ''')
    """
    # Strip JavaScript comments before CDP evaluation
    code = _strip_js_comments(code)

    cdp_session = await browser_session.get_or_create_cdp_session()

    try:
        # Execute JavaScript with proper error handling
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": code, "returnByValue": True, "awaitPromise": True},
            session_id=cdp_session.session_id,
        )

        # Check for JavaScript execution errors
        if result.get("exceptionDetails"):
            exception = result["exceptionDetails"]
            error_text = exception.get("text", "Unknown error")

            # Try to get more details from the exception
            error_details = []
            if "exception" in exception:
                exc_obj = exception["exception"]
                if "description" in exc_obj:
                    error_details.append(exc_obj["description"])
                elif "value" in exc_obj:
                    error_details.append(str(exc_obj["value"]))

            # Build comprehensive error message
            error_msg = f"JavaScript execution error: {error_text}"
            if error_details:
                error_msg += f'\nDetails: {" | ".join(error_details)}'

            raise EvaluateError(error_msg)

        # Get the result data
        result_data = result.get("result", {})

        # Get the actual value
        value = result_data.get("value")

        # Return the value directly
        if value is None:
            return None if "value" in result_data else "undefined"
        elif isinstance(value, (dict, list)):
            # Complex objects - already deserialized by returnByValue
            return value
        else:
            # Primitive values
            return value

    except EvaluateError:
        # Re-raise EvaluateError as-is to stop Python execution
        raise
    except Exception as e:
        # Wrap other exceptions in EvaluateError
        raise EvaluateError(f"Failed to execute JavaScript: {type(e).__name__}: {e}") from e


def create_namespace(
    browser_session: BrowserSession,
    tools: Tools | None = None,
    sensitive_data: dict[str, str | dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Create a namespace with all browser tools available as functions.

    This function creates a dictionary of functions that can be used to interact
    with the browser, similar to a Jupyter notebook environment.

    Args:
        browser_session: The browser session to use
        tools: Optional Tools instance (will create default if not provided)
        sensitive_data: Optional sensitive data dictionary

    Returns:
        Dictionary containing all available functions and objects

    Example:
        namespace = create_namespace(browser_session)
        await namespace['navigate'](url='https://google.com')
        result = await namespace['evaluate']('document.title')
    """
    if tools is None:
        tools = Tools(browser_session)

    namespace: dict[str, Any] = {
        # Core objects
        "browser": browser_session,
        # Standard library modules (always available)
        "json": json,
        "asyncio": asyncio,
        "Path": Path,
        "csv": csv,
        "re": re,
        "datetime": datetime,
    }

    # Track failed evaluate() calls to detect repeated failed approaches
    if "_evaluate_failures" not in namespace:
        namespace["_evaluate_failures"] = []

    # Add custom evaluate function that returns values directly
    async def evaluate_wrapper(
        code: str | None = None, variables: dict[str, Any] | None = None, *_args: Any, **kwargs: Any
    ) -> Any:
        # Handle both positional and keyword argument styles
        if code is None:
            code = kwargs.get("code", kwargs.get("js_code", kwargs.get("expression", "")))
        if variables is None:
            variables = kwargs.get("variables")

        if not code:
            raise ValueError("No JavaScript code provided to evaluate()")

        # Inject variables if provided
        if variables:
            vars_json = json.dumps(variables)
            stripped = code.strip()

            # Check if code is already a function expression expecting params
            if re.match(r"\((?:async\s+)?function\s*\(\s*\w+\s*\)", stripped):
                code = f"(function(){{ const params = {vars_json}; return {stripped}(params); }})()"
            else:
                # Not a parameterized function, inject params in scope
                is_wrapped = (
                    (stripped.startswith("(function()") and "})())" in stripped[-10:])
                    or (stripped.startswith("(async function()") and "})())" in stripped[-10:])
                    or (stripped.startswith("(() =>") and ")())" in stripped[-10:])
                    or (stripped.startswith("(async () =>") and ")())" in stripped[-10:])
                )
                if is_wrapped:
                    match = re.match(r"(\((?:async\s+)?function\s*\(\s*\)\s*\{)", stripped)
                    if match:
                        prefix = match.group(1)
                        rest = stripped[len(prefix) :]
                        code = f"{prefix} const params = {vars_json}; {rest}"
                    else:
                        code = f"(function(){{ const params = {vars_json}; return {stripped}; }})()"
                else:
                    code = f"(function(){{ const params = {vars_json}; {code} }})()"
                    return await evaluate(code, browser_session)

        # Auto-wrap in IIFE if not already wrapped (and no variables were injected)
        if not variables:
            stripped = code.strip()
            is_wrapped = (
                (stripped.startswith("(function()") and "})())" in stripped[-10:])
                or (stripped.startswith("(async function()") and "})())" in stripped[-10:])
                or (stripped.startswith("(() =>") and ")())" in stripped[-10:])
                or (stripped.startswith("(async () =>") and ")())" in stripped[-10:])
            )
            if not is_wrapped:
                code = f"(function(){{{code}}})()"

        # Execute and track failures
        try:
            result = await evaluate(code, browser_session)

            # Print result structure for debugging
            if isinstance(result, list) and result and isinstance(result[0], dict):
                result_preview = f"list of dicts - len={len(result)}, example 1:\n"
                sample_result = result[0]
                for key, value in list(sample_result.items())[:10]:
                    value_str = str(value)[:10] if not isinstance(value, (int, float, bool, type(None))) else str(value)
                    result_preview += f"  {key}: {value_str}...\n"
                if len(sample_result) > 10:
                    result_preview += f"  ... {len(sample_result) - 10} more keys"
                print(result_preview)

            elif isinstance(result, list):
                if len(result) == 0:
                    print("type=list, len=0")
                else:
                    result_preview = str(result)[:100]
                    print(f"type=list, len={len(result)}, preview={result_preview}...")
            elif isinstance(result, dict):
                result_preview = f"type=dict, len={len(result)}, sample keys:\n"
                for key, value in list(result.items())[:10]:
                    value_str = str(value)[:10] if not isinstance(value, (int, float, bool, type(None))) else str(value)
                    result_preview += f"  {key}: {value_str}...\n"
                if len(result) > 10:
                    result_preview += f"  ... {len(result) - 10} more keys"
                print(result_preview)

            else:
                print(f"type={type(result).__name__}, value={repr(result)[:50]}")

            return result
        except Exception as e:
            namespace["_evaluate_failures"].append({"error": str(e), "type": "exception"})
            raise

    namespace["evaluate"] = evaluate_wrapper

    # Inject all tools as functions into the namespace
    for action_name, action in tools.registry.registry.actions.items():
        if action_name == "evaluate":
            continue  # Skip - use custom evaluate that returns Python objects directly
        param_model = action.param_model
        action_function = action.function

        # Create a closure to capture the current action_name, param_model, and action_function
        def make_action_wrapper(act_name, par_model, act_func):
            async def action_wrapper(*args, **kwargs):
                # Convert positional args to kwargs based on param model fields
                if args:
                    field_names = list(par_model.model_fields.keys())
                    for i, arg in enumerate(args):
                        if i < len(field_names):
                            kwargs[field_names[i]] = arg

                # Create params from kwargs
                try:
                    params = par_model(**kwargs)
                except Exception as e:
                    raise ValueError(f"Invalid parameters for {act_name}: {e}") from e

                # Special validation for done() - enforce minimal code cell
                if act_name == "done":
                    consecutive_failures = namespace.get("_consecutive_errors")
                    if not consecutive_failures or consecutive_failures <= 3:
                        # Check if there are multiple Python blocks in this response
                        all_blocks = namespace.get("_all_code_blocks", {})
                        python_blocks = [k for k in sorted(all_blocks.keys()) if k.startswith("python_")]

                        if len(python_blocks) > 1:
                            msg = (
                                "done() should be the ONLY code block in the response.\n"
                                "You have multiple Python blocks in this response."
                            )
                            print(msg)

                # Build special context
                special_context = {
                    "browser_session": browser_session,
                }

                # Execute the action
                result = await act_func(params=params, **special_context)

                # For code-use mode, we want to return the result directly
                if hasattr(result, "extracted_content"):
                    # Special handling for done action
                    if act_name == "done" and hasattr(result, "is_done") and result.is_done:
                        namespace["_task_done"] = True
                        if result.extracted_content:
                            namespace["_task_result"] = result.extracted_content
                        if hasattr(result, "success"):
                            namespace["_task_success"] = result.success

                    if result.extracted_content:
                        return result.extracted_content
                    if result.error:
                        raise RuntimeError(result.error)
                    return None
                return result

            return action_wrapper

        # Rename 'input' to 'input_text' to avoid shadowing Python's built-in input()
        namespace_action_name = "input_text" if action_name == "input" else action_name

        # Add the wrapper to the namespace
        namespace[namespace_action_name] = make_action_wrapper(action_name, param_model, action_function)

    return namespace


def get_namespace_documentation(namespace: dict[str, Any]) -> str:
    """
    Generate documentation for all available functions in the namespace.

    Args:
        namespace: The namespace dictionary

    Returns:
        Markdown-formatted documentation string
    """
    docs = ["# Available Functions\n"]

    # Document each function
    for name, obj in sorted(namespace.items()):
        if callable(obj) and not name.startswith("_"):
            # Get function signature and docstring
            if hasattr(obj, "__doc__") and obj.__doc__:
                docs.append(f"## {name}\n")
                docs.append(f"{obj.__doc__}\n")

    return "\n".join(docs)

