"""Namespace initialization for code-use mode.

This module creates a namespace with all browser tools available as functions,
similar to a Jupyter notebook environment.
"""

import asyncio
import base64
import csv
import datetime
import json
import logging
import re
import time
from inspect import iscoroutinefunction
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import requests
from pydantic import ValidationError

from openbrowser.browser import BrowserSession
from openbrowser.browser.navigation import WaitUntil

try:
	from openbrowser.filesystem.file_system import FileSystem
except ImportError:
	FileSystem = None  # type: ignore[assignment,misc]
from openbrowser.llm.base import BaseChatModel
from openbrowser.tools.action_availability import ActionAvailabilityCode, ActionAvailabilityError
from openbrowser.tools.action_readiness import (
	readiness_error_from_validation_error,
	validate_index_value,
)
from openbrowser.tools.registry.views import RegisteredAction
from openbrowser.tools.service import CodeAgentTools, Tools
from openbrowser.tools.views import IndexedAction

logger = logging.getLogger(__name__)

# Try to import optional data science libraries
try:
	import numpy as np  # type: ignore

	NUMPY_AVAILABLE = True
except ImportError:
	NUMPY_AVAILABLE = False

try:
	import pandas as pd  # type: ignore

	PANDAS_AVAILABLE = True
except ImportError:
	PANDAS_AVAILABLE = False

try:
	import matplotlib.pyplot as plt  # type: ignore

	MATPLOTLIB_AVAILABLE = True
except ImportError:
	MATPLOTLIB_AVAILABLE = False

try:
	from bs4 import BeautifulSoup  # type: ignore

	BS4_AVAILABLE = True
except ImportError:
	BS4_AVAILABLE = False

_PDF_INSTALL_HINT = "pypdf is not installed. Install it with: pip install 'openbrowser-ai[pdf]'"


class _MissingPdfReader:
	"""Actionable placeholder when the optional PDF dependency is absent."""

	def __init__(self, *args, **kwargs):
		raise ImportError(_PDF_INSTALL_HINT)


class _MissingPypdfModule:
	"""Module-like fallback so ``pypdf.PdfReader`` stays discoverable."""

	def __getattr__(self, name: str):
		if name == 'PdfReader':
			return _MissingPdfReader
		raise AttributeError(name)


try:
	import pypdf  # type: ignore
	from pypdf import PdfReader  # type: ignore

	PYPDF_AVAILABLE = True
except ImportError:
	pypdf = _MissingPypdfModule()  # type: ignore[assignment]
	PdfReader = _MissingPdfReader  # type: ignore[assignment,misc]
	PYPDF_AVAILABLE = False

try:
	from tabulate import tabulate  # type: ignore

	TABULATE_AVAILABLE = True
except ImportError:
	TABULATE_AVAILABLE = False


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
	js_code = re.sub(r'/\*.*?\*/', '', js_code, flags=re.DOTALL)

	# Remove single-line comments - only lines that START with // (after whitespace)
	# This avoids breaking XPath strings, URLs, regex patterns, etc.
	js_code = re.sub(r'^\s*//.*$', '', js_code, flags=re.MULTILINE)

	return js_code


class EvaluateError(Exception):
	"""Special exception raised by evaluate() to stop Python execution immediately."""

	pass


async def validate_task_completion(
	task: str,
	output: str | None,
	llm: BaseChatModel,
) -> tuple[bool, str]:
	"""
	Validate if task is truly complete by asking LLM without system prompt or history.

	Args:
		task: The original task description
		output: The output from the done() call
		llm: The LLM to use for validation

	Returns:
		Tuple of (is_complete, reasoning)
	"""
	from openbrowser.llm.messages import UserMessage

	# Build validation prompt
	validation_prompt = f"""You are a task completion validator. Analyze if the agent has truly completed the user's task.

**Original Task:**
{task}

**Agent's Output:**
{output[:100000] if output else '(No output provided)'}

**Your Task:**
Determine if the agent has successfully completed the user's task. Consider:
1. Has the agent delivered what the user requested?
2. If data extraction was requested, is there actual data?
3. If the task is impossible (e.g., localhost website, login required but no credentials), is it truly impossible?
4. Could the agent continue and make meaningful progress?

**Response Format:**
Reasoning: [Your analysis of whether the task is complete]
Verdict: [YES or NO]

YES = Task is complete OR truly impossible to complete
NO = Agent should continue working"""

	try:
		# Call LLM with just the validation prompt (no system prompt, no history)
		response = await llm.ainvoke([UserMessage(content=validation_prompt)])
		response_text = response.completion

		# Parse the response
		reasoning = ''
		verdict = 'NO'

		# Extract reasoning and verdict
		lines = response_text.split('\n')
		for line in lines:
			if line.strip().lower().startswith('reasoning:'):
				reasoning = line.split(':', 1)[1].strip()
			elif line.strip().lower().startswith('verdict:'):
				verdict_text = line.split(':', 1)[1].strip().upper()
				if 'YES' in verdict_text:
					verdict = 'YES'
				elif 'NO' in verdict_text:
					verdict = 'NO'

		# If we couldn't parse, try to find YES/NO in the response
		if not reasoning:
			reasoning = response_text

		is_complete = verdict == 'YES'

		logger.info(f'Task validation: {verdict}')
		logger.debug(f'Validation reasoning: {reasoning}')

		return is_complete, reasoning

	except Exception as e:
		logger.warning(f'Failed to validate task completion: {e}')
		# On error, assume the agent knows what they're doing
		return True, f'Validation failed: {e}'


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
	# Strip JavaScript comments before CDP evaluation (CDP doesn't support them in all contexts)
	code = _strip_js_comments(code)

	cdp_session = await browser_session.get_or_create_cdp_session()

	try:
		# Execute JavaScript with proper error handling
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': code, 'returnByValue': True, 'awaitPromise': True},
			session_id=cdp_session.session_id,
		)

		# Check for JavaScript execution errors
		if result.get('exceptionDetails'):
			exception = result['exceptionDetails']
			error_text = exception.get('text', 'Unknown error')

			# Try to get more details from the exception
			error_details = []
			if 'exception' in exception:
				exc_obj = exception['exception']
				if 'description' in exc_obj:
					error_details.append(exc_obj['description'])
				elif 'value' in exc_obj:
					error_details.append(str(exc_obj['value']))

			# Build comprehensive error message with full CDP context
			error_msg = f'JavaScript execution error: {error_text}'
			if error_details:
				error_msg += f'\nDetails: {" | ".join(error_details)}'

			# Raise special exception that will stop Python execution immediately
			raise EvaluateError(error_msg)

		# Get the result data
		result_data = result.get('result', {})

		# Get the actual value
		value = result_data.get('value')

		# Return the value directly
		if value is None:
			return None if 'value' in result_data else 'undefined'
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
		raise EvaluateError(f'Failed to execute JavaScript: {type(e).__name__}: {e}') from e


async def scrape_http(
	url: str,
	*,
	format: str,
	selector: str | None,
	extract_links: bool,
	timeout_ms: int | None,
	engine: str,
) -> dict[str, Any] | None:
	"""Fetch and parse a URL without a browser.

	Returns None when engine=auto finds page JavaScript and the browser path
	should handle the page.
	"""
	import httpx
	from bs4 import BeautifulSoup
	from markdownify import markdownify as _markdownify

	timeout = (timeout_ms / 1000) if timeout_ms else 30.0
	async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
		response = await client.get(url)
	if response.status_code >= 400:
		return None

	content_type = response.headers.get('content-type', '').split(';')[0].strip().lower()
	if 'json' in content_type:
		try:
			payload = response.json()
		except Exception:
			return None
		return {
			'url': str(response.url),
			'title': '',
			'format': format,
			'content_type': 'application/json',
			'content': payload,
			'stats': {'engine_used': 'http', 'status_code': response.status_code},
			'engine_used': 'http',
			'status_code': response.status_code,
		}

	if 'html' not in content_type and content_type not in ('', 'text/plain'):
		return None

	html = response.text
	soup = BeautifulSoup(html, 'html.parser')
	if engine == 'auto':
		# Use HTTP only when the server-rendered text is substantial. Thin
		# shells usually need JavaScript, so fall back to the browser.
		text_probe = soup.get_text(' ', strip=True)
		if len(text_probe) < 40:
			return None
		if soup.select_one('#root:empty, #app:empty, #__next:empty'):
			return None

	title = soup.title.get_text(strip=True) if soup.title else ''
	content: Any
	if format == 'markdown':
		for tag in soup(['script', 'style', 'noscript']):
			tag.decompose()
		content = _markdownify(
			str(soup),
			heading_style='ATX',
			bullets='-',
			escape_asterisks=False,
			escape_underscores=False,
		).strip()
	elif format == 'text':
		for tag in soup(['script', 'style', 'noscript']):
			tag.decompose()
		content = soup.get_text('\n', strip=True)
	elif format == 'html':
		content = html
	elif format == 'links':
		content = [
			{
				'text': (tag.get_text(' ', strip=True) or ''),
				'href': tag.get('href'),
				'title': tag.get('title'),
			}
			for tag in soup.find_all('a', href=True)
		]
	else:
		content = [
			{
				'text': tag.get_text(' ', strip=True),
				'html': str(tag),
				'attributes': dict(tag.attrs),
			}
			for tag in (soup.select(selector) if selector else [])
		]

	return {
		'url': str(response.url),
		'title': title,
		'format': format,
		'content_type': 'application/json' if format in {'links', 'json'} else f'text/{format}',
		'content': content,
		'stats': {
			'engine_used': 'http',
			'status_code': response.status_code,
			'html_chars': len(html),
		},
		'engine_used': 'http',
		'status_code': response.status_code,
	}


def _truncate_scrape_result(result: dict[str, Any], max_chars: int | None) -> dict[str, Any]:
	"""Apply the optional output cap to a scrape result."""
	if max_chars is not None:
		content = result.get('content')
		if isinstance(content, str) and len(content) > max_chars:
			result['content'] = content[:max_chars]
			result['truncated'] = True
			result['total_chars'] = len(content)
	return result


def create_namespace(
	browser_session: BrowserSession,
	tools: Tools | None = None,
	page_extraction_llm: BaseChatModel | None = None,
	file_system: FileSystem | None = None,
	available_file_paths: list[str] | None = None,
	sensitive_data: dict[str, str | dict[str, str]] | None = None,
	completion_callback: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
	"""
	Create a namespace with all browser tools available as functions.

	This function creates a dictionary of functions that can be used to interact
	with the browser, similar to a Jupyter notebook environment.

	Args:
		browser_session: The browser session to use
		tools: Optional Tools instance (will create default if not provided)
		page_extraction_llm: Optional LLM for page extraction
		file_system: Optional file system for file operations
		available_file_paths: Optional list of available file paths
		sensitive_data: Optional sensitive data dictionary
		completion_callback: Optional sink for a done action result without an error

	Returns:
		Dictionary containing all available functions and objects

	Example:
		namespace = create_namespace(browser_session)
		await namespace['navigate'](url='https://google.com')
		result = await namespace['evaluate']('document.title')
	"""
	if tools is None:
		# Use CodeAgentTools with default exclusions optimized for code-use mode
		# For code-use, we keep: navigate, evaluate, wait, done
		# and exclude: most browser interaction, file system actions (use Python instead)
		tools = CodeAgentTools()

	if available_file_paths is None:
		available_file_paths = []

	namespace: dict[str, Any] = {
		# Core objects
		'browser': browser_session,
		'file_system': file_system,
		# Standard library modules (always available)
		'json': json,
		'asyncio': asyncio,
		'Path': Path,
		'csv': csv,
		're': re,
		'datetime': datetime,
		'requests': requests,
	}

	# Add optional data science libraries if available
	if NUMPY_AVAILABLE:
		namespace['np'] = np
		namespace['numpy'] = np
	if PANDAS_AVAILABLE:
		namespace['pd'] = pd
		namespace['pandas'] = pd
	if MATPLOTLIB_AVAILABLE:
		namespace['plt'] = plt
		namespace['matplotlib'] = plt
	if BS4_AVAILABLE:
		namespace['BeautifulSoup'] = BeautifulSoup
		namespace['bs4'] = BeautifulSoup
	pdf_reader = PdfReader if PYPDF_AVAILABLE else _MissingPdfReader
	namespace['PdfReader'] = pdf_reader
	namespace['pypdf'] = pypdf if PYPDF_AVAILABLE else _MissingPypdfModule()
	if TABULATE_AVAILABLE:
		namespace['tabulate'] = tabulate

	# Track failed evaluate() calls to detect repeated failed approaches
	if '_evaluate_failures' not in namespace:
		namespace['_evaluate_failures'] = []

	# Add custom evaluate function that returns values directly
	async def evaluate_wrapper(
		code: str | None = None, variables: dict[str, Any] | None = None, *_args: Any, **kwargs: Any
	) -> Any:
		resolver = getattr(tools.registry, 'resolve_availability_page_url', None)
		if iscoroutinefunction(resolver):
			page_url = await resolver('evaluate', browser_session, None)
			checker = getattr(tools.registry, 'check_action_available', None)
			if callable(checker):
				checker('evaluate', page_url=page_url, display_name='evaluate')

		# Handle both positional and keyword argument styles
		if code is None:
			# Check if code was passed as keyword arg
			code = kwargs.get('code', kwargs.get('js_code', kwargs.get('expression', '')))
		# Extract variables if passed as kwarg
		if variables is None:
			variables = kwargs.get('variables')

		if not code:
			raise ValueError('No JavaScript code provided to evaluate()')

		# Inject variables if provided
		if variables:
			vars_json = json.dumps(variables)
			stripped = code.strip()

			# Check if code is already a function expression expecting params
			# Pattern: (function(params) { ... }) or (async function(params) { ... })
			if re.match(r'\((?:async\s+)?function\s*\(\s*\w+\s*\)', stripped):
				# Already expects params, wrap to call it with our variables
				code = f'(function(){{ const params = {vars_json}; return {stripped}(params); }})()'
			else:
				# Not a parameterized function, inject params in scope
				# Check if already wrapped in IIFE (including arrow function IIFEs)
				is_wrapped = (
					(stripped.startswith('(function()') and '})()' in stripped[-10:])
					or (stripped.startswith('(async function()') and '})()' in stripped[-10:])
					or (stripped.startswith('(() =>') and ')()' in stripped[-10:])
					or (stripped.startswith('(async () =>') and ')()' in stripped[-10:])
				)
				if is_wrapped:
					# Already wrapped, inject params at the start
					# Try to match regular function IIFE
					match = re.match(r'(\((?:async\s+)?function\s*\(\s*\)\s*\{)', stripped)
					if match:
						prefix = match.group(1)
						rest = stripped[len(prefix) :]
						code = f'{prefix} const params = {vars_json}; {rest}'
					else:
						# Try to match arrow function IIFE
						# Patterns: (() => expr)() or (() => { ... })() or (async () => ...)()
						arrow_match = re.match(r'(\((?:async\s+)?\(\s*\)\s*=>\s*\{)', stripped)
						if arrow_match:
							# Arrow function with block body: (() => { ... })()
							prefix = arrow_match.group(1)
							rest = stripped[len(prefix) :]
							code = f'{prefix} const params = {vars_json}; {rest}'
						else:
							# Arrow function with expression body or fallback: wrap in outer function
							code = f'(function(){{ const params = {vars_json}; return {stripped}; }})()'
				else:
					# Not wrapped, wrap with params
					code = f'(function(){{ const params = {vars_json}; {code} }})()'
					# Skip auto-wrap below
					return await evaluate(code, browser_session)

		# Auto-wrap in IIFE if not already wrapped (and no variables were injected)
		if not variables:
			stripped = code.strip()
			# Check for regular function IIFEs, async function IIFEs, and arrow function IIFEs
			is_wrapped = (
				(stripped.startswith('(function()') and '})()' in stripped[-10:])
				or (stripped.startswith('(async function()') and '})()' in stripped[-10:])
				or (stripped.startswith('(() =>') and ')()' in stripped[-10:])
				or (stripped.startswith('(async () =>') and ')()' in stripped[-10:])
			)
			if not is_wrapped:
				# For simple single-expression code (no newlines, no semicolons,
				# not a statement keyword), wrap with return so the value is
				# returned from the IIFE.  Multi-statement code gets wrapped
				# without return (caller should include explicit return).
				_statement_prefixes = (
					'var ', 'let ', 'const ', 'if ', 'if(', 'for ', 'for(',
					'while ', 'while(', 'switch ', 'switch(', 'try ', 'try{',
					'class ', 'function ', 'async function', 'throw ', 'return ',
				)
				is_single_expr = (
					'\n' not in stripped
					and ';' not in stripped
					and not stripped.startswith(_statement_prefixes)
				)
				if is_single_expr:
					code = f'(function(){{return {stripped}}})()'
				else:
					code = f'(function(){{{stripped}}})()'

		# Execute and track failures
		try:
			result = await evaluate(code, browser_session)
			return result
		except Exception as e:
			# Track errors for pattern detection
			namespace['_evaluate_failures'].append({'error': str(e), 'type': 'exception'})
			raise

	excluded_actions = getattr(tools.registry, 'exclude_actions', None)
	if isinstance(excluded_actions, (list, tuple, set, frozenset)) and 'evaluate' in excluded_actions:
		async def excluded_evaluate_wrapper(*_args: Any, **_kwargs: Any) -> Any:
			raise ActionAvailabilityError(
				ActionAvailabilityCode.ACTION_EXCLUDED,
				action='evaluate',
			)

		namespace['evaluate'] = excluded_evaluate_wrapper
	else:
		namespace['evaluate'] = evaluate_wrapper

	# Add get_selector_from_index helper for code_use mode
	async def get_selector_from_index_wrapper(index: int) -> str:
		"""
		Get the CSS selector for an element by its interactive index.

		This allows you to use the element's index from the browser state to get
		its CSS selector for use in JavaScript evaluate() calls.

		Args:
			index: The interactive index from the browser state (e.g., [123])

		Returns:
			str: CSS selector that can be used in JavaScript

		Example:
			selector = await get_selector_from_index(123)
			await evaluate(f'''
			(function(){{
				const el = document.querySelector({json.dumps(selector)});
				if (el) el.click();
			}})()
			''')
		"""
		from openbrowser.dom.utils import generate_css_selector_for_element

		# Get element by index from browser session
		node = await browser_session.get_element_by_index(index)
		if node is None:
			msg = f'Element index {index} not available - page may have changed. Try refreshing browser state.'
			logger.warning(f'⚠️ {msg}')
			raise RuntimeError(msg)

		# Check if element is in shadow DOM
		shadow_hosts = []
		current = node.parent_node
		while current:
			if current.shadow_root_type is not None:
				# This is a shadow host
				host_tag = current.tag_name.lower()
				host_id = current.attributes.get('id', '') if current.attributes else ''
				host_desc = f'{host_tag}#{host_id}' if host_id else host_tag
				shadow_hosts.insert(0, host_desc)
			current = current.parent_node

		# Check if in iframe
		in_iframe = False
		current = node.parent_node
		while current:
			if current.tag_name.lower() == 'iframe':
				in_iframe = True
				break
			current = current.parent_node

		# Use the robust selector generation function (now handles special chars in IDs)
		selector = generate_css_selector_for_element(node)

		# Log shadow DOM/iframe info if detected
		if shadow_hosts:
			shadow_path = ' > '.join(shadow_hosts)
			logger.info(f'Element [{index}] is inside Shadow DOM. Path: {shadow_path}')
			logger.info(f'    Selector: {selector}')
			logger.info(
				f'    To access: document.querySelector("{shadow_hosts[0].split("#")[0]}").shadowRoot.querySelector("{selector}")'
			)
		if in_iframe:
			logger.info(f"Element [{index}] is inside an iframe. Regular querySelector won't work.")

		if selector:
			return selector

		# Fallback: just use tag name if available
		if node.tag_name:
			return node.tag_name.lower()

		raise ValueError(f'Could not generate selector for element index {index}')

	namespace['get_selector_from_index'] = get_selector_from_index_wrapper

	# Add download_file function for downloading files via browser session
	async def download_file(url: str, filename: str | None = None) -> str:
		"""Download a file from a URL using the browser's cookies and session.

		Uses JavaScript fetch in the browser context to preserve authentication
		and cookies. Falls back to Python requests if browser fetch fails.

		Args:
			url: The URL to download
			filename: Optional filename. Auto-derived from URL if not provided.

		Returns:
			Absolute path to the downloaded file
		"""
		# Validate URL
		parsed_url = urlparse(url)
		if not parsed_url.scheme or not parsed_url.netloc:
			raise ValueError(f'Invalid URL: {url}')
		if parsed_url.scheme not in ('http', 'https'):
			raise ValueError(f'Unsupported URL scheme: {parsed_url.scheme}')

		# Determine downloads directory
		dl_path = browser_session.browser_profile.downloads_path
		if not dl_path:
			dl_path = str(Path.home() / 'Downloads' / 'openbrowser-mcp')
		dl_dir = Path(str(dl_path)).expanduser().resolve()

		registry = getattr(browser_session, 'downloads_registry', None)
		from openbrowser.browser.download_state import DownloadPhase, DownloadRegistry

		download_id = f'namespace:{url}:{time.time_ns()}'
		try:
			dl_dir.mkdir(parents=True, exist_ok=True)
		except OSError as error:
			if isinstance(registry, DownloadRegistry):
				registry.reject(
					download_id,
					stage='write',
					error=f'{type(error).__name__}: {error}',
					url=url,
					source='namespace',
				)
			raise

		# Determine filename from URL if not provided
		if not filename:
			filename = Path(unquote(parsed_url.path)).name or 'download'
			if '.' not in filename:
				filename = filename + '.pdf'

		# Sanitize filename: strip path components to prevent traversal
		filename = Path(filename).name
		if not filename or filename in ('.', '..'):
			filename = 'download.pdf'

		# Resolve and verify final path stays inside downloads directory
		final_path = (dl_dir / filename).resolve()
		if not str(final_path).startswith(str(dl_dir)):
			raise ValueError('Invalid filename: path traversal detected')

		# Handle filename conflicts
		if final_path.exists():
			stem = final_path.stem
			ext = final_path.suffix
			counter = 1
			while (dl_dir / f'{stem} ({counter}){ext}').exists():
				counter += 1
			filename = f'{stem} ({counter}){ext}'
			final_path = dl_dir / filename

		if isinstance(registry, DownloadRegistry):
			active = registry.active_count()
			limit = registry.max_active
			if isinstance(active, int) and isinstance(limit, int) and active >= limit:
				raise RuntimeError(
					f'Too many active downloads (limit {limit}). '
					'Wait for one to finish or inspect download_outcomes().'
				)
			registry.start(download_id, url=url, file_name=filename, source='namespace')

		# Strategy 1: Browser JS fetch (preserves cookies/auth)
		js_error = None
		try:
			cdp_session = await browser_session.get_or_create_cdp_session()
			escaped_url = json.dumps(url)

			result = await asyncio.wait_for(
				cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': f"""(async () => {{
	const response = await fetch({escaped_url}, {{ cache: 'force-cache' }});
	if (!response.ok) throw new Error('HTTP ' + response.status);
	const buf = await response.arrayBuffer();
	const bytes = new Uint8Array(buf);
	let binary = '';
	const chunkSize = 32768;
	for (let i = 0; i < bytes.length; i += chunkSize) {{
		const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
		let s = '';
		for (let j = 0; j < chunk.length; j++) s += String.fromCharCode(chunk[j]);
		binary += s;
	}}
	return {{ base64: btoa(binary), size: bytes.length, type: response.headers.get('content-type') || '' }};
}})()""",
						'awaitPromise': True,
						'returnByValue': True,
					},
					session_id=cdp_session.session_id,
				),
				timeout=120.0,
			)

			# Check for JS exception
			if result.get('exceptionDetails'):
				exc_details = result['exceptionDetails']
				js_error = exc_details.get('text', '') or str(
					exc_details.get('exception', {}).get('description', 'Unknown JS error')
				)
			else:
				data = result.get('result', {}).get('value', {})
				base64_value = data.get('base64') if isinstance(data, dict) else None
				# An empty string is a valid zero-byte file, not missing data.
				if isinstance(base64_value, str):
					try:
						file_bytes = base64.b64decode(base64_value)
					except Exception as b64_err:
						js_error = f'Invalid base64 encoding: {b64_err}'
					else:
						with open(final_path, 'wb') as f:
							f.write(file_bytes)
						if registry is not None:
							registry.complete(
								download_id,
								path=str(final_path),
								file_size=len(file_bytes),
								file_name=final_path.name,
								mime_type=data.get('type') or None,
								url=url,
								source='namespace',
							)
						size_kb = len(file_bytes) / 1024
						if isinstance(registry, DownloadRegistry):
							current = registry.get(download_id)
							if current is not None and current.phase is DownloadPhase.CANCELLED:
								raise asyncio.CancelledError('download cancelled during reset')
						logger.info(f'Downloaded {filename} ({size_kb:.1f} KB) via browser fetch')
						return str(final_path)
				else:
					js_error = 'No data received from browser fetch'
		except asyncio.CancelledError:
			if registry is not None:
				registry.cancel(download_id, detail='download_file cancelled')
			raise
		except asyncio.TimeoutError:
			js_error = 'Browser fetch timed out after 120s'
		except Exception as e:
			js_error = f'{type(e).__name__}: {e}'

		if isinstance(registry, DownloadRegistry):
			current = registry.get(download_id)
			if current is not None and current.phase is DownloadPhase.CANCELLED:
				raise asyncio.CancelledError('download cancelled during reset')

		# Strategy 2: Python requests fallback (no browser cookies)
		try:
			response = requests.get(url, timeout=120, stream=True)
			response.raise_for_status()
			with open(final_path, 'wb') as f:
				for chunk in response.iter_content(chunk_size=8192):
					f.write(chunk)
			file_size = final_path.stat().st_size
			if registry is not None:
				registry.complete(
					download_id,
					path=str(final_path),
					file_size=file_size,
					file_name=final_path.name,
					url=url,
					source='namespace',
					detail='browser fetch failed; python requests fallback succeeded',
				)
			size_kb = file_size / 1024
			if isinstance(registry, DownloadRegistry):
				current = registry.get(download_id)
				if current is not None and current.phase is DownloadPhase.CANCELLED:
					raise asyncio.CancelledError('download cancelled during reset')
			logger.info(f'Downloaded {filename} ({size_kb:.1f} KB) via Python requests (fallback)')
			return str(final_path)
		except asyncio.CancelledError:
			if registry is not None:
				registry.cancel(download_id, detail='download_file cancelled')
			raise
		except Exception as req_error:
			error_message = (
				f'Download failed.  Browser fetch error: {js_error}  Python requests error: {req_error}'
				if js_error
				else f'Download failed: {type(req_error).__name__}: {req_error}'
			)
			if registry is not None:
				registry.fail(
					download_id,
					stage='namespace_fetch',
					error=error_message,
					url=url,
					source='namespace',
				)
			if js_error:
				raise RuntimeError(
					f'Download failed.\n'
					f'  Browser fetch error: {js_error}\n'
					f'  Python requests error: {req_error}'
				)
			raise

	namespace['download_file'] = download_file

	# Add list_downloads helper
	def list_downloads() -> list[str]:
		"""List all files in the downloads directory.

		Returns:
			List of absolute paths to downloaded files, sorted by name.
		"""
		dl_path = browser_session.browser_profile.downloads_path
		if not dl_path:
			return []
		dl_dir = Path(str(dl_path)).expanduser().resolve()
		try:
			if not dl_dir.exists():
				return []
			return sorted([str(f) for f in dl_dir.iterdir() if f.is_file()])
		except (FileNotFoundError, PermissionError):
			return []

	namespace['list_downloads'] = list_downloads

	async def wait_for_download(
		timeout: float = 30.0,
		url: str | None = None,
		lookback: float = 0.0,
		download_ref: str | None = None,
	) -> dict[str, Any]:
		"""Wait for the next download to reach a terminal phase.

		Start the waiter before clicking a link or submitting a form so a fast
		download cannot finish first:
		`waiter = asyncio.create_task(wait_for_download(timeout=30))`.
		If the download may already have finished, pass `lookback=5` to accept
		a terminal outcome from the last five seconds. Returns the terminal
		outcome, including `path`, `file_size`, `phase`, `stage`, and `error`.
		If you have a prior outcome entry, pass its `download_ref` to wait for
		that specific attempt without exposing the internal download id.
		Raises TimeoutError with the last observed phase when no terminal
		outcome appears in time.
		"""
		return await browser_session.wait_for_download(
			timeout=timeout,
			url=url,
			lookback=lookback,
			download_ref=download_ref,
		)

	def download_outcomes() -> list[dict[str, Any]]:
		"""Return recent download attempts, oldest first, with their last phase."""
		return browser_session.download_outcomes()

	def download_failures() -> list[dict[str, Any]]:
		"""Return download attempts that ended in the failed phase."""
		return [outcome for outcome in browser_session.download_outcomes() if outcome.get('phase') == 'failed']

	namespace['wait_for_download'] = wait_for_download
	namespace['download_outcomes'] = download_outcomes
	namespace['download_failures'] = download_failures

	async def scrape_page(
		url: str | None = None,
		*,
		format: str = 'markdown',
		selector: str | None = None,
		wait_for: str | None = None,
		wait_until: WaitUntil = 'load',
		timeout_ms: int | None = None,
		extract_links: bool = False,
		max_chars: int | None = None,
		engine: str = 'auto',
	) -> dict[str, Any]:
		"""Scrape the current page or a URL without an LLM.

		Args:
			url: Optional URL to navigate to before scraping. Uses the current page when omitted.
			format: One of markdown, text, html, links, or json.
			selector: Optional CSS selector. Limits text, html, and json output to matching elements.
			wait_for: Optional CSS selector to wait for before scraping. Useful for SPAs.
			wait_until: Navigation readiness: load, domcontentloaded, networkidle, or commit.
			timeout_ms: Optional navigation and selector wait timeout in milliseconds.
			extract_links: Preserve links in markdown output.
			max_chars: Optional maximum content length. Adds truncated and total_chars fields.

		Returns:
			A dict with url, title, format, content, content_type, and optional stats.
		"""
		allowed_formats = {'markdown', 'text', 'html', 'links', 'json'}
		if format not in allowed_formats:
			raise ValueError(f'format must be one of {sorted(allowed_formats)}, got {format!r}')
		if engine not in {'auto', 'http', 'browser'}:
			raise ValueError(f"engine must be one of ['auto', 'browser', 'http'], got {engine!r}")

		if engine in {'auto', 'http'} and url:
			http_result = await scrape_http(
				url,
				format=format,
				selector=selector,
				extract_links=extract_links,
				timeout_ms=timeout_ms,
				engine=engine,
			)
			if http_result is not None:
				return _truncate_scrape_result(http_result, max_chars)
			if engine == 'http':
				raise RuntimeError(f'HTTP scrape failed for {url}')

		if url:
			await browser_session.navigate_to(url, wait_until=wait_until, timeout_ms=timeout_ms)

		selector_js = json.dumps(selector) if selector is not None else None
		wait_selector_js = json.dumps(wait_for) if wait_for is not None else None
		deadline = time.monotonic() + ((timeout_ms / 1000) if timeout_ms else 30.0)

		if wait_selector_js is not None:
			while True:
				found = await namespace['evaluate'](f'document.querySelector({wait_selector_js}) !== null')
				if found:
					break
				if time.monotonic() >= deadline:
					raise TimeoutError(f'wait_for selector did not appear within {timeout_ms or 30000} ms: {wait_for}')
				await asyncio.sleep(0.05)

		title = await namespace['evaluate']('document.title || ""')
		final_url = await browser_session.get_current_page_url()

		content: Any
		stats: dict[str, Any] = {}
		if format == 'markdown':
			if selector is None:
				from openbrowser.dom.markdown_extractor import extract_clean_markdown

				content, stats = await extract_clean_markdown(
					browser_session=browser_session,
					extract_links=extract_links,
				)
			else:
				html = await namespace['evaluate'](
					f'(document.querySelector({selector_js}) || document.body).outerHTML'
				)
				from markdownify import markdownify as _markdownify

				content = _markdownify(
					html or '',
					heading_style='ATX',
					strip=['script', 'style'],
					bullets='-',
					escape_asterisks=False,
					escape_underscores=False,
				).strip()
				stats = {'method': 'selector_html', 'selector': selector}
		elif format == 'text':
			expression = (
				f'(document.querySelector({selector_js}) || document.body)?.innerText || ""'
				if selector is not None
				else 'document.body ? document.body.innerText : ""'
			)
			content = await namespace['evaluate'](expression)
		elif format == 'html':
			expression = (
				f'(document.querySelector({selector_js}) || document.documentElement).outerHTML'
				if selector is not None
				else 'document.documentElement ? document.documentElement.outerHTML : ""'
			)
			content = await namespace['evaluate'](expression)
		elif format == 'links':
			content = await namespace['evaluate'](
				'Array.from(document.links).map(a => ('
				'{text: (a.innerText || "").trim(), href: a.href, title: a.title || null}'
				'))'
			)
		else:
			if selector is not None:
				content = await namespace['evaluate'](
					f'Array.from(document.querySelectorAll({selector_js})).map(el => ({{'
					'text: (el.innerText || "").trim(), '
					'html: el.outerHTML, '
					'attributes: Object.fromEntries(Array.from(el.attributes).map(a => [a.name, a.value]))'
					'}))'
				)
			else:
				from openbrowser.dom.markdown_extractor import extract_clean_markdown

				content, stats = await extract_clean_markdown(
					browser_session=browser_session,
					extract_links=extract_links,
				)

		result: dict[str, Any] = {
			'url': final_url,
			'title': title,
			'format': format,
			'content_type': 'application/json' if format in {'links', 'json'} else f'text/{format}',
			'content': content,
			'stats': {**stats, 'engine_used': 'browser'},
			'engine_used': 'browser',
		}
		if selector is not None:
			result['selector'] = selector
		return _truncate_scrape_result(result, max_chars)

	namespace['scrape_page'] = scrape_page
	namespace['scrape'] = scrape_page


	# Inject all tools as functions into the namespace
	# Skip 'evaluate' since we have a custom implementation above
	for action_name, action in tools.registry.registry.actions.items():
		if action_name == 'evaluate':
			continue  # Skip - use custom evaluate that returns Python objects directly
		param_model = action.param_model
		action_function = (
			action.normalized_function
			if isinstance(action, RegisteredAction) and action.normalized_function
			else action.function
		)
		namespace_action_name = 'input_text' if action_name == 'input' else action_name

		# Create a closure to capture the registry key, public alias, param model, and action function.
		def make_action_wrapper(registry_name, public_name, par_model, act_func):
			async def action_wrapper(*args, **kwargs):
				# Convert positional args to kwargs based on param model fields
				if args:
					# Get the field names from the pydantic model
					field_names = list(par_model.model_fields.keys())
					for i, arg in enumerate(args):
						if i < len(field_names):
							kwargs[field_names[i]] = arg

				# Availability is checked before parameter validation so a stale
				# or hidden action cannot report a misleading schema error.
				supplied_page_url = kwargs.pop('page_url', None)
				page_url = supplied_page_url
				resolver = getattr(tools.registry, 'resolve_availability_page_url', None)
				if iscoroutinefunction(resolver):
					page_url = await resolver(
						registry_name,
						browser_session,
						supplied_page_url,
					)
					checker = getattr(tools.registry, 'check_action_available', None)
					if callable(checker):
						checker(
							registry_name,
							page_url=page_url,
							display_name=public_name,
						)

				# Readiness guard: indexed actions fail with a typed error and a
				# refresh hint before Pydantic produces a raw schema error.
				if 'index' in kwargs:
					if isinstance(par_model, type) and issubclass(par_model, IndexedAction):
						kwargs['index'] = validate_index_value(kwargs['index'], action=public_name)
					elif kwargs['index'] is None:
						field_info = par_model.model_fields.get('index')
						if field_info and field_info.is_required():
							raise ValueError(
								f'{public_name}() requires a valid element index, got index=None. '
								f'Read the current browser state and use an [i_N] index from the DOM tree.'
							)

				# Create params from kwargs
				try:
					params = par_model(**kwargs)
				except ValidationError as e:
					readiness = readiness_error_from_validation_error(e)
					if readiness is not None:
						raise readiness.with_action(public_name) from e
					raise ValueError(f'Invalid parameters for {public_name}: {e}') from e
				except Exception as e:
					raise ValueError(f'Invalid parameters for {public_name}: {e}') from e

				# Special validation for done() - enforce minimal code cell
				if public_name == 'done':
					consecutive_failures = namespace.get('_consecutive_errors')
					if consecutive_failures and consecutive_failures > 3:
						pass

					else:
						# Check if there are multiple Python blocks in this response
						all_blocks = namespace.get('_all_code_blocks', {})
						python_blocks = [k for k in sorted(all_blocks.keys()) if k.startswith('python_')]

						if len(python_blocks) > 1:
							msg = (
								'done() should be the ONLY code block in the response.\n'
								'You have multiple Python blocks in this response. Consider calling done() in a separate response '
								'Now verify the last output and if it satisfies the task, call done(), else continue working.'
							)
							print(msg)

						# Get the current cell code from namespace (injected by service.py before execution)
						current_code = namespace.get('_current_cell_code')
						if current_code and isinstance(current_code, str):
							# Count non-empty, non-comment lines
							lines = [line.strip() for line in current_code.strip().split('\n')]
							code_lines = [line for line in lines if line and not line.startswith('#')]

							# Check if the line above await done() contains an if block
							done_line_index = -1
							for i, line in enumerate(reversed(code_lines)):
								if 'await done()' in line or 'await done(' in line:
									done_line_index = len(code_lines) - 1 - i
									break

							has_if_above = False
							has_else_above = False
							has_elif_above = False
							if done_line_index > 0:
								line_above = code_lines[done_line_index - 1]
								has_if_above = line_above.strip().startswith('if ') and line_above.strip().endswith(':')
								has_else_above = line_above.strip().startswith('else:')
								has_elif_above = line_above.strip().startswith('elif ')
							if has_if_above or has_else_above or has_elif_above:
								msg = (
									'done() should be called individually after verifying the result from any logic.\n'
									'Consider validating your output first, THEN call done() in a final step without if/else/elif blocks only if the task is truly complete.'
								)
								logger.error(msg)
								print(msg)
								raise RuntimeError(msg)

				# Build special context
				special_context = {
					'browser_session': browser_session,
					'page_extraction_llm': page_extraction_llm,
					'available_file_paths': available_file_paths,
					'has_sensitive_data': False,  # Can be handled separately if needed
					'file_system': file_system,
				}

				# Execute the action. Omit page_url when no value was supplied or
				# resolved so an action's own default remains effective.
				if page_url is not None:
					special_context['page_url'] = page_url
				result = await act_func(params=params, **special_context)

				# For code-use mode, we want to return the result directly
				# not wrapped in ActionResult
				if hasattr(result, 'extracted_content'):
					# Special handling for done action - mark task as complete
					if (
						public_name == 'done'
						and hasattr(result, 'is_done')
						and result.is_done
						and not getattr(result, 'error', None)
					):
						if completion_callback is not None:
							completion_callback(result)
						namespace['_task_done'] = True
						namespace['_task_done_by_action'] = True
						# Store the extracted content as the final result
						if result.extracted_content:
							namespace['_task_result'] = result.extracted_content
						# Store the self-reported success status
						if hasattr(result, 'success'):
							namespace['_task_success'] = result.success
						# Store file attachments if present
						if hasattr(result, 'attachments') and result.attachments:
							namespace['_task_attachments'] = result.attachments

					# If there's extracted content, return it
					if result.extracted_content:
						return result.extracted_content
					# If there's an error, raise it
					if result.error:
						raise RuntimeError(result.error)
					# Otherwise return None
					return None
				return result

			return action_wrapper

		# Add the wrapper under its public alias while retaining the registry key.
		namespace[namespace_action_name] = make_action_wrapper(
			action_name,
			namespace_action_name,
			param_model,
			action_function,
		)

	return namespace


def get_namespace_documentation(namespace: dict[str, Any]) -> str:
	"""
	Generate documentation for all available functions in the namespace.

	Args:
		namespace: The namespace dictionary

	Returns:
		Markdown-formatted documentation string
	"""
	docs = ['# Available Functions\n']

	# Document each function
	for name, obj in sorted(namespace.items()):
		if callable(obj) and not name.startswith('_'):
			# Get function signature and docstring
			if hasattr(obj, '__doc__') and obj.__doc__:
				docs.append(f'## {name}\n')
				docs.append(f'{obj.__doc__}\n')

	return '\n'.join(docs)
