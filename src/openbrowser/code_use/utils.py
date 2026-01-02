"""Utility functions for code-use agent.

This module provides helper functions for text processing, token limit detection,
URL extraction, and code block parsing used by the CodeAgent.
"""

import re


def truncate_message_content(content: str, max_length: int = 10000) -> str:
    """Truncate message content to a maximum length for history storage.

    Used to prevent excessively long messages from being stored in the
    conversation history, which could cause context window issues.

    Args:
        content: The message content to potentially truncate.
        max_length: Maximum allowed length in characters. Defaults to 10000.

    Returns:
        The original content if within limits, or truncated content with
        a marker indicating how many characters were removed.

    Example:
        ```python
        long_text = "x" * 15000
        truncated = truncate_message_content(long_text)
        # Returns first 10000 chars + truncation notice
        ```
    """
    if len(content) <= max_length:
        return content
    # Truncate and add marker
    return content[:max_length] + f"\n\n[... truncated {len(content) - max_length} characters for history]"


def detect_token_limit_issue(
    completion: str,
    completion_tokens: int | None,
    max_tokens: int | None,
    stop_reason: str | None,
) -> tuple[bool, str | None]:
    """Detect if the LLM response hit token limits or is repetitive garbage.

    Checks multiple indicators that suggest the LLM's response was cut off
    or degraded due to token limits, including explicit stop reasons,
    token usage ratios, and repetitive output patterns.

    Args:
        completion: The full text of the LLM completion.
        completion_tokens: Number of tokens used in the completion, if known.
        max_tokens: Maximum tokens allowed for the completion, if known.
        stop_reason: The reason the LLM stopped generating, if available.

    Returns:
        A tuple of (is_problematic, error_message). If is_problematic is True,
        error_message contains a description of the detected issue.

    Example:
        ```python
        is_bad, msg = detect_token_limit_issue(
            completion=response.content,
            completion_tokens=response.usage.completion_tokens,
            max_tokens=4096,
            stop_reason=response.stop_reason
        )
        if is_bad:
            logger.warning(f"Token issue: {msg}")
        ```
    """
    # Check 1: Stop reason indicates max_tokens
    if stop_reason == "max_tokens":
        return True, f"Response terminated due to max_tokens limit (stop_reason: {stop_reason})"

    # Check 2: Used 90%+ of max_tokens (if we have both values)
    if completion_tokens is not None and max_tokens is not None and max_tokens > 0:
        usage_ratio = completion_tokens / max_tokens
        if usage_ratio >= 0.9:
            return True, f"Response used {usage_ratio:.1%} of max_tokens ({completion_tokens}/{max_tokens})"

    # Check 3: Last 6 characters repeat 40+ times (repetitive garbage)
    if len(completion) >= 6:
        last_6 = completion[-6:]
        repetition_count = completion.count(last_6)
        if repetition_count >= 40:
            return True, f'Repetitive output detected: last 6 chars "{last_6}" appears {repetition_count} times'

    return False, None


def extract_url_from_task(task: str) -> str | None:
    """Extract a URL from a task string using pattern matching.

    Attempts to find URLs in the task description for automatic navigation.
    Filters out email addresses and handles URLs with or without protocols.
    Returns None if multiple URLs are found to avoid ambiguity.

    Args:
        task: The task description string that may contain a URL.

    Returns:
        A single URL with https:// prefix if exactly one URL is found,
        or None if zero or multiple URLs are detected.

    Example:
        ```python
        url = extract_url_from_task("Go to google.com and search")
        # Returns "https://google.com"

        url = extract_url_from_task("Compare google.com and bing.com")
        # Returns None (multiple URLs)
        ```
    """
    # Remove email addresses from task before looking for URLs
    task_without_emails = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", task)

    # Look for common URL patterns
    patterns = [
        r"https?://[^\s<>\"']+",  # Full URLs with http/https
        r"(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}(?:/[^\s<>\"']*)?",  # Domain names
    ]

    found_urls = []
    for pattern in patterns:
        matches = re.finditer(pattern, task_without_emails)
        for match in matches:
            url = match.group(0)

            # Remove trailing punctuation that's not part of URLs
            url = re.sub(r"[.,;:!?()\[\]]+$", "", url)
            # Add https:// if missing
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            found_urls.append(url)

    unique_urls = list(set(found_urls))
    # If multiple URLs found, skip auto-navigation to avoid ambiguity
    if len(unique_urls) > 1:
        return None

    # If exactly one URL found, return it
    if len(unique_urls) == 1:
        return unique_urls[0]

    return None


def extract_code_blocks(text: str) -> dict[str, str]:
    """Extract all code blocks from a Markdown response.

    Parses Markdown-formatted text to extract fenced code blocks,
    supporting multiple languages and named blocks.

    Supported languages:
        - python: Python code for execution
        - js/javascript: JavaScript for browser evaluation
        - bash/sh/shell: Shell commands
        - markdown/md: Markdown content

    Special features:
        - Named blocks: ```js my_function → saved as 'my_function' in namespace
        - Nested blocks: Use 4+ backticks for outer block containing 3-backtick content
        - Multiple Python blocks: Each gets a unique key (python_0, python_1, etc.)

    Args:
        text: Markdown text containing fenced code blocks.

    Returns:
        Dictionary mapping block identifiers to their content.
        - 'python' key contains the first Python block (for backward compatibility)
        - 'python_N' keys for each Python block
        - Named blocks use their specified names as keys
        - Other languages use their normalized name as key

    Example:
        ```python
        text = '''Here's the code:
        ```python
        x = 42
        ```
        And some JavaScript:
        ```js extract_data
        document.querySelector('.price').textContent
        ```
        '''
        blocks = extract_code_blocks(text)
        # Returns {'python': 'x = 42', 'python_0': 'x = 42',
        #          'extract_data': 'document.querySelector...'}
        ```
    """
    # Pattern to match code blocks with language identifier and optional variable name
    pattern = r"(`{3,})(\w+)(?:\s+(\w+))?\n(.*?)\1(?:\n|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    blocks: dict[str, str] = {}
    python_block_counter = 0

    for backticks, lang, var_name, content in matches:
        lang = lang.lower()

        # Normalize language names
        if lang in ("javascript", "js"):
            lang_normalized = "js"
        elif lang in ("markdown", "md"):
            lang_normalized = "markdown"
        elif lang in ("sh", "shell"):
            lang_normalized = "bash"
        elif lang == "python":
            lang_normalized = "python"
        else:
            # Unknown language, skip
            continue

        # Only process supported types
        if lang_normalized in ("python", "js", "bash", "markdown"):
            content = content.rstrip()  # Only strip trailing whitespace
            if content:
                # Determine the key to use
                if var_name:
                    # Named block - use the variable name
                    block_key = var_name
                    blocks[block_key] = content
                elif lang_normalized == "python":
                    # Unnamed Python blocks - give each a unique key
                    block_key = f"python_{python_block_counter}"
                    blocks[block_key] = content
                    python_block_counter += 1
                else:
                    # Other unnamed blocks - keep last one only
                    blocks[lang_normalized] = content

    # If we have multiple python blocks, mark the first one as 'python' for backward compat
    if python_block_counter > 0:
        blocks["python"] = blocks["python_0"]

    # Fallback: if no python block but there's generic ``` block, treat as python
    if python_block_counter == 0 and "python" not in blocks:
        generic_pattern = r"```\n(.*?)```"
        generic_matches = re.findall(generic_pattern, text, re.DOTALL)
        if generic_matches:
            combined = "\n\n".join(m.strip() for m in generic_matches if m.strip())
            if combined:
                blocks["python"] = combined

    return blocks

