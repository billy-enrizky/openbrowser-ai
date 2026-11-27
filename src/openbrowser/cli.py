# pyright: reportMissingImports=false

# Check for MCP mode early to prevent logging initialization
import contextlib
import logging
import sys
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from openbrowser.cli_modes import CliMode, select_cli_mode


def run_http_scrape_sync(*args, **kwargs):
	"""Resolve the HTTP scrape runner on first use."""
	from openbrowser.scrape.http import run_http_scrape_sync as implementation

	return implementation(*args, **kwargs)


def emit_scrape_payload(*args, **kwargs):
	"""Resolve the scrape payload writer on first use."""
	from openbrowser.scrape.output import emit_scrape_payload as implementation

	return implementation(*args, **kwargs)


def read_scrape_urls_file(*args, **kwargs):
	"""Resolve the URL file reader on first use."""
	from openbrowser.scrape.output import read_scrape_urls_file as implementation

	return implementation(*args, **kwargs)

# Fast paths execute only when this module is the process entry point. The
# __test__ seam keeps the existing source-executing coverage tests honest.
INIT_TEMPLATES = {
	'default': {
		'file': 'default_template.py',
		'description': 'Simplest setup - capable of any web task with minimal configuration',
	},
	'advanced': {
		'file': 'advanced_template.py',
		'description': 'All configuration options shown with defaults',
	},
	'tools': {
		'file': 'tools_template.py',
		'description': 'Custom action examples - extend the agent with your own functions',
	},
}

_RUN_FAST_PATHS = __name__ in {'__main__', '__test__'}
_mode = select_cli_mode(sys.argv[1:]) if _RUN_FAST_PATHS else None

if _mode is CliMode.MCP:
	import asyncio
	import logging
	import os

	os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'

	transport = os.environ.get('OPENBROWSER_MCP_TRANSPORT', 'lean').strip().lower()
	if transport not in {'lean', 'sdk'}:
		sys.stderr.write("ERROR: OPENBROWSER_MCP_TRANSPORT must be 'lean' or 'sdk'\n")
		sys.exit(2)
	if transport == 'lean':
		from openbrowser.mcp.lean import main as lean_mcp_main

		logging.disable(logging.CRITICAL)
		asyncio.run(lean_mcp_main())
		sys.exit(0)

	# Early exit: run MCP server directly without loading heavy CLI dependencies
	# (anthropic, openai, etc. are not needed for MCP server mode)
	try:
		from openbrowser.telemetry import CLITelemetryEvent, ProductTelemetry
		from openbrowser.utils import get_openbrowser_version

		telemetry = ProductTelemetry()
		telemetry.capture(
			CLITelemetryEvent(
				version=get_openbrowser_version(),
				action='start',
				mode='mcp_server',
			)
		)
	except Exception:
		pass

	try:
		from openbrowser.mcp.server import main as mcp_main
	except ImportError as e:
		sys.stderr.write(f"ERROR: mcp SDK not installed ({e}). Fix: pip install 'openbrowser-ai[mcp]'\n")
		sys.exit(1)

	logging.disable(logging.CRITICAL)
	asyncio.run(mcp_main())
	sys.exit(0)

# Check for -c (execute code) mode early -- minimal imports for fast startup
if _mode is CliMode.EXECUTE_CODE:
	import os
	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
	try:
		from dotenv import load_dotenv
		load_dotenv()
	except ImportError:
		pass

	import asyncio

	c_idx = sys.argv.index('-c')
	code = sys.argv[c_idx + 1] if c_idx + 1 < len(sys.argv) else None
	if code == '-':
		code = sys.stdin.read().strip()
	if code is None:
		# No code argument: print the function reference and auto-start daemon
		from openbrowser.code_use.descriptions import (
			EXECUTE_CODE_DESCRIPTION,
			EXECUTE_CODE_DESCRIPTION_COMPACT,
		)
		from openbrowser.daemon.client import DaemonClient

		client = DaemonClient()
		try:
			status = asyncio.run(client.status())
		except TimeoutError:
			# Daemon is alive but slow to respond -- show compact help
			print(EXECUTE_CODE_DESCRIPTION_COMPACT)
			sys.exit(0)
		except (OSError, asyncio.CancelledError, ValueError):
			status = None
		if status and status.success:
			# Daemon already running (warm) -- compact description
			print(EXECUTE_CODE_DESCRIPTION_COMPACT)
		elif status and status.error_code == 'daemon_timeout':
			# The daemon is alive but busy. Do not describe it as cold or try to start another.
			print(EXECUTE_CODE_DESCRIPTION_COMPACT)
		else:
			# Daemon not running (cold) -- verbose description, then start it
			print(EXECUTE_CODE_DESCRIPTION)
			try:
				asyncio.run(client._start_daemon())
			except (TimeoutError, OSError, asyncio.CancelledError):
				pass  # Best-effort; daemon may still be starting in background
		sys.exit(0)

	from openbrowser.daemon.client import execute_code_via_daemon

	result = asyncio.run(execute_code_via_daemon(code))
	if result.success:
		if result.output:
			print(result.output)
	else:
		print(result.output or result.error, file=sys.stderr)
		if result.error_code:
			print(f'Error code: {result.error_code}', file=sys.stderr)
		if result.hint:
			print(f'Hint: {result.hint}', file=sys.stderr)
		sys.exit(1)
	sys.exit(0)

# Check for config subcommand early
if _mode is CliMode.CONFIG:
	import json
	import os

	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
	try:
		from dotenv import load_dotenv
		load_dotenv()
	except ImportError:
		pass

	from openbrowser.config import CONFIG

	sub = sys.argv[2] if len(sys.argv) > 2 else 'status'
	if sub == 'status':
		result = CONFIG.load_result()
		print(json.dumps(result.snapshot.to_dict(), indent=2))
		sys.exit(0)
	print('Usage: openbrowser-ai config status', file=sys.stderr)
	sys.exit(2)

# Check for daemon and warm subcommands early
if _mode in (CliMode.DAEMON, CliMode.WARM):
	import os
	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
	try:
		from dotenv import load_dotenv
		load_dotenv()
	except ImportError:
		pass

	import asyncio
	import json
	import time
	from pathlib import Path

	from openbrowser.daemon.client import DaemonClient, execute_code_via_daemon
	from openbrowser.daemon import DAEMON_DIR, IS_WINDOWS, WINDOWS_PORT, get_log_path, get_pid_path, get_socket_path
	from openbrowser.daemon.diagnostics import daemon_processes, reap_orphan_processes, socket_candidates

	sub = 'warm' if _mode is CliMode.WARM else (sys.argv[2] if len(sys.argv) > 2 else 'status')
	client = DaemonClient()

	def _print_daemon_error(resp) -> None:
		print(resp.output or resp.error, file=sys.stderr)
		if resp.error_code:
			print(f'Error code: {resp.error_code}', file=sys.stderr)
		if resp.hint:
			print(f'Hint: {resp.hint}', file=sys.stderr)

	def _requested_headless() -> bool | None:
		raw = os.environ.get('OPENBROWSER_HEADLESS')
		if raw is None:
			return None
		# Mirror the daemon: only explicit false values disable headless mode.
		return raw.lower() not in ('false', '0', 'no')

	def _status_matches_request(resp) -> bool:
		if not resp.success:
			return False
		try:
			payload = json.loads(resp.output)
		except (TypeError, json.JSONDecodeError):
			return True
		if not isinstance(payload, dict):
			return True
		profile = payload.get('profile') or {}
		requested_headless = _requested_headless()
		if requested_headless is not None and profile.get('headless') != requested_headless:
			return False
		requested_profile = os.environ.get('OPENBROWSER_USER_DATA_DIR') or os.environ.get('OPENBROWSER_DAEMON_PROFILE_DIR')
		if requested_profile:
			current_profile = profile.get('user_data_dir')
			if not current_profile:
				return False
			try:
				if Path(current_profile).expanduser().resolve() != Path(requested_profile).expanduser().resolve():
					return False
			except OSError:
				return False
		custom_socket = os.environ.get('OPENBROWSER_SOCKET')
		isolate = os.environ.get('OPENBROWSER_DAEMON_PROFILE_ISOLATION', '').lower() in ('true', '1', 'yes')
		if not requested_profile and custom_socket and isolate:
			current_profile = profile.get('user_data_dir')
			if not current_profile:
				return False
			import hashlib

			expected_name = f'daemon-{hashlib.sha256(str(get_socket_path()).encode("utf-8")).hexdigest()[:12]}'
			if Path(current_profile).name != expected_name:
				return False
		return True

	if sub == 'start':
		async def _start():
			status = await client.status()
			if status.success and _status_matches_request(status):
				print('Daemon is already running')
				return
			if status.success:
				print('Daemon configuration differs; restarting')
				stop = await client.force_stop()
				if not stop.success:
					_print_daemon_error(stop)
					sys.exit(1)
			try:
				await client._start_daemon()
			except (TimeoutError, OSError) as error:
				print(f'Daemon failed to start: {error}', file=sys.stderr)
				sys.exit(1)
			print('Daemon started')
		asyncio.run(_start())
	elif sub == 'ensure':
		async def _ensure():
			status = await client.status()
			if status.success and _status_matches_request(status):
				print('Daemon already matches the requested configuration')
				return
			if status.success:
				print('Daemon configuration differs; restarting')
				stop = await client.force_stop()
				if not stop.success:
					_print_daemon_error(stop)
					sys.exit(1)
			try:
				await client._start_daemon()
			except (TimeoutError, OSError) as error:
				print(f'Daemon failed to start: {error}', file=sys.stderr)
				sys.exit(1)
			print('Daemon started')
		asyncio.run(_ensure())
	elif sub == 'warm':
		async def _warm():
			status = await client.status()
			if status.success and not _status_matches_request(status):
				stop = await client.force_stop()
				if not stop.success:
					_print_daemon_error(stop)
					sys.exit(1)
			started = time.monotonic()
			resp = await execute_code_via_daemon(
				'summary = await browser.get_browser_state_summary(include_screenshot=False); print(summary.url)'
			)
			elapsed_ms = (time.monotonic() - started) * 1000.0
			if not resp.success:
				_print_daemon_error(resp)
				sys.exit(1)
			output = (resp.output or '').strip()
			url = output.splitlines()[-1].strip() if output else ''
			print(json.dumps({'ok': True, 'elapsed_ms': round(elapsed_ms, 1), 'url': url}, indent=2))
		asyncio.run(_warm())
	elif sub == 'stop':
		resp = asyncio.run(client.stop())
		if not resp.success:
			_print_daemon_error(resp)
			sys.exit(1)
		print(resp.output or 'Daemon stopped')
	elif sub == 'force-stop':
		resp = asyncio.run(client.force_stop())
		if not resp.success:
			_print_daemon_error(resp)
			sys.exit(1)
		print(resp.output or 'Daemon force-stopped')
	elif sub == 'status':
		resp = asyncio.run(client.status())
		if not resp.success:
			_print_daemon_error(resp)
			sys.exit(1)
		print(resp.output)
	elif sub == 'reset':
		resp = asyncio.run(client.reset())
		if not resp.success:
			_print_daemon_error(resp)
			sys.exit(1)
		print(resp.output or 'Session reset')
	elif sub == 'restart':
		async def _restart():
			stop = await client.force_stop()
			if not stop.success:
				_print_daemon_error(stop)
				sys.exit(1)
			try:
				await client._start_daemon()
			except (TimeoutError, OSError) as error:
				print(f'Daemon failed to start: {error}', file=sys.stderr)
				sys.exit(1)
			print('Daemon restarted')
		asyncio.run(_restart())
	elif sub == 'list':
		async def _list():
			original_socket = os.environ.get('OPENBROWSER_SOCKET')
			results = []
			try:
				for socket_path in socket_candidates():
					os.environ['OPENBROWSER_SOCKET'] = str(socket_path)
					status = await DaemonClient().status()
					payload = None
					if status.success:
						try:
							payload = json.loads(status.output)
						except (TypeError, json.JSONDecodeError):
							payload = {'raw': status.output}
					results.append(
						{
							'socket': str(socket_path),
							'running': status.success,
							'status': payload,
							'error': status.error,
							'error_code': status.error_code,
						}
					)
			finally:
				if original_socket is None:
					os.environ.pop('OPENBROWSER_SOCKET', None)
				else:
					os.environ['OPENBROWSER_SOCKET'] = original_socket
			print(json.dumps({'sockets': results, 'processes': daemon_processes()}, indent=2))
		asyncio.run(_list())
	elif sub == 'reap':
		async def _reap():
			original_socket = os.environ.get('OPENBROWSER_SOCKET')
			results = []
			try:
				for socket_path in socket_candidates():
					os.environ['OPENBROWSER_SOCKET'] = str(socket_path)
					response = await DaemonClient().force_stop()
					results.append(
						{
							'socket': str(socket_path),
							'success': response.success,
							'output': response.output,
							'error': response.error,
							'error_code': response.error_code,
						}
					)
			finally:
				if original_socket is None:
					os.environ.pop('OPENBROWSER_SOCKET', None)
				else:
					os.environ['OPENBROWSER_SOCKET'] = original_socket
			orphans = reap_orphan_processes()
			print(json.dumps({'sockets': results, 'orphans': orphans}, indent=2))
			if any(not item['success'] for item in results) or any(
				not item['reaped']
				and item['reason'] not in ('socket is live or unknown', 'process is too young', 'process identity changed')
				for item in orphans
			):
				sys.exit(1)
		asyncio.run(_reap())
	elif sub == 'doctor':
		async def _doctor():
			status = await client.status()
			payload = None
			if status.success:
				try:
					payload = json.loads(status.output)
				except json.JSONDecodeError:
					payload = {'raw': status.output}

			socket_path = get_socket_path()
			pid_path = get_pid_path()
			log_path = get_log_path()
			pid = None
			if pid_path.exists():
				try:
					pid = int(pid_path.read_text().strip())
				except (OSError, ValueError):
					pid = None
			log_tail = ''
			if log_path.exists():
				try:
					log_tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-15:])
				except OSError:
					log_tail = ''

			healthy = bool(status.success and payload and payload.get('state') == 'ready')
			if IS_WINDOWS:
				listeners = [row for row in daemon_processes() if row.get('socket_live')]
				socket_report = {
					'transport': 'tcp',
					'host': '127.0.0.1',
					'port': WINDOWS_PORT,
					'listening': bool(listeners),
				}
			else:
				socket_report = {
					'transport': 'unix',
					'path': str(socket_path),
					'exists': socket_path.exists(),
				}
			report = {
				'healthy': healthy,
				'status': payload,
				'status_error': status.error,
				'error_code': status.error_code,
				'hint': status.hint,
				'socket': socket_report,
				'pid_file': {'path': str(pid_path), 'exists': pid_path.exists(), 'pid': pid},
				'log': {'path': str(log_path), 'tail': log_tail},
			}
			print(json.dumps(report, indent=2))
			if not healthy:
				sys.exit(1)
		asyncio.run(_doctor())
	else:
		print(f'Unknown daemon command: {sub}', file=sys.stderr)
		print(
			'Usage: openbrowser-ai warm | openbrowser-ai daemon '
			'[start|ensure|warm|stop|force-stop|status|reset|restart|list|reap|doctor]',
			file=sys.stderr,
		)
		sys.exit(1)
	sys.exit(0)

# Special case: install command doesn't need CLI dependencies
if _mode is CliMode.INSTALL:
	import platform
	import subprocess

	print('📦 Installing Chromium browser + system dependencies...')
	print('⏳ This may take a few minutes...\n')

	# Build command - only use --with-deps on Linux (it fails on Windows/macOS)
	cmd = ['uvx', 'playwright', 'install', 'chromium']
	if platform.system() == 'Linux':
		cmd.append('--with-deps')
	cmd.append('--no-shell')

	result = subprocess.run(cmd)

	if result.returncode == 0:
		print('\n✅ Installation complete!')
		print('Ready to use! Run: uvx openbrowser-ai')
	else:
		print('\n❌ Installation failed')
		sys.exit(1)
	sys.exit(0)

# Check for init subcommand early to avoid loading heavy dependencies
if _mode is CliMode.INIT:
	from openbrowser.init_cmd import INIT_TEMPLATES
	from openbrowser.init_cmd import main as init_main

	# Check if --template or -t flag is present without a value
	# If so, just remove it and let init_main handle interactive mode
	if '--template' in sys.argv or '-t' in sys.argv:
		try:
			template_idx = sys.argv.index('--template') if '--template' in sys.argv else sys.argv.index('-t')
			template = sys.argv[template_idx + 1] if template_idx + 1 < len(sys.argv) else None

			# If template is not provided or is another flag, remove the flag and use interactive mode
			if not template or template.startswith('-'):
				if '--template' in sys.argv:
					sys.argv.remove('--template')
				else:
					sys.argv.remove('-t')
		except (ValueError, IndexError):
			pass

	# Remove 'init' from sys.argv so click doesn't see it as an unexpected argument
	sys.argv.remove('init')
	init_main()
	sys.exit(0)

# Check for --template flag early to avoid loading heavy dependencies
if _mode is CliMode.TEMPLATE:
	from pathlib import Path

	import click

	# Parse template and output from sys.argv
	try:
		template_idx = sys.argv.index('--template')
		template = sys.argv[template_idx + 1] if template_idx + 1 < len(sys.argv) else None
	except (ValueError, IndexError):
		template = None

	# If template is not provided or is another flag, use interactive mode
	if not template or template.startswith('-'):
		# Redirect to init command with interactive template selection
		from openbrowser.init_cmd import main as init_main

		# Remove --template from sys.argv
		sys.argv.remove('--template')
		init_main()
		sys.exit(0)

	# Validate template name
	if template not in INIT_TEMPLATES:
		click.echo(f'❌ Invalid template. Choose from: {", ".join(INIT_TEMPLATES.keys())}', err=True)
		sys.exit(1)

	# Check for --output flag
	output = None
	if '--output' in sys.argv or '-o' in sys.argv:
		try:
			output_idx = sys.argv.index('--output') if '--output' in sys.argv else sys.argv.index('-o')
			output = sys.argv[output_idx + 1] if output_idx + 1 < len(sys.argv) else None
		except (ValueError, IndexError):
			pass

	# Check for --force flag
	force = '--force' in sys.argv or '-f' in sys.argv

	# Determine output path
	output_path = Path(output) if output else Path.cwd() / f'openbrowser_{template}.py'

	# Read and write template
	try:
		templates_dir = Path(__file__).parent / 'cli_templates'
		template_file = INIT_TEMPLATES[template]['file']
		template_path = templates_dir / template_file
		content = template_path.read_text(encoding='utf-8')

		# Write file with safety checks
		if output_path.exists() and not force:
			click.echo(f'⚠️  File already exists: {output_path}')
			if not click.confirm('Overwrite?', default=False):
				click.echo('❌ Cancelled')
				sys.exit(1)

		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(content, encoding='utf-8')

		click.echo(f'✅ Created {output_path}')
		click.echo('\nNext steps:')
		click.echo('  This generated Agent template uses ChatOpenAI and needs OPENAI_API_KEY.')
		click.echo('  MCP and openbrowser-ai -c need no LLM key.')
		click.echo('  1. Install openbrowser:')
		click.echo('     uv pip install openbrowser-ai')
		click.echo('  2. Set OPENAI_API_KEY in the .env file or environment:')
		click.echo('     OPENAI_API_KEY=your-key')
		click.echo('  3. Run your script:')
		click.echo(f'     python {output_path.name}')
	except Exception as e:
		click.echo(f'❌ Error: {e}', err=True)
		sys.exit(1)

	sys.exit(0)

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
	from openbrowser import Agent, Controller
	from openbrowser.agent.views import AgentSettings
except ImportError:
	Agent = None  # type: ignore[assignment,misc]
	Controller = None  # type: ignore[assignment,misc]
	AgentSettings = None  # type: ignore[assignment,misc]
from openbrowser.browser import BrowserProfile, BrowserSession
from openbrowser.telemetry import CLITelemetryEvent, ProductTelemetry
from openbrowser.utils import get_openbrowser_version

import click


os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'result'

from openbrowser.config import CONFIG, ConfigLoadResult, ConfigStartupError, apply_managed_browser_profile_defaults

# Set USER_DATA_DIR now that CONFIG is imported
USER_DATA_DIR = CONFIG.OPENBROWSER_PROFILES_DIR / 'cli'

# Ensure directories exist
CONFIG.OPENBROWSER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default User settings
MAX_HISTORY_LENGTH = 100


def get_default_config(config_result: ConfigLoadResult | None = None) -> dict[str, Any]:
	"""Return default configuration dictionary using the new config system."""
	# Use the exact load result when a start gate already approved it.
	config_data = CONFIG.config_dict_from_result(config_result) if config_result is not None else CONFIG.load_config()

	# Extract browser profile, llm, and agent configs
	browser_profile = config_data.get('browser_profile', {})
	llm_config = config_data.get('llm', {})
	agent_config = config_data.get('agent', {})

	return {
		'model': {
			'name': llm_config.get('model'),
			'temperature': llm_config.get('temperature', 0.0),
			'api_keys': {
				'OPENAI_API_KEY': llm_config.get('api_key', CONFIG.OPENAI_API_KEY),
				'ANTHROPIC_API_KEY': CONFIG.ANTHROPIC_API_KEY,
				'GOOGLE_API_KEY': CONFIG.GOOGLE_API_KEY,
				'DEEPSEEK_API_KEY': CONFIG.DEEPSEEK_API_KEY,
				'GROK_API_KEY': CONFIG.GROK_API_KEY,
			},
		},
		'agent': agent_config,
		'browser': {
			'headless': browser_profile.get('headless', True),
			'keep_alive': browser_profile.get('keep_alive', True),
			'ignore_https_errors': browser_profile.get('ignore_https_errors', False),
			'user_data_dir': browser_profile.get('user_data_dir'),
			'storage_state': browser_profile.get('storage_state'),
			'allowed_domains': browser_profile.get('allowed_domains'),
			'wait_between_actions': browser_profile.get('wait_between_actions'),
			'is_mobile': browser_profile.get('is_mobile'),
			'device_scale_factor': browser_profile.get('device_scale_factor'),
			'disable_security': browser_profile.get('disable_security'),
			'render_mode': browser_profile.get('render_mode'),
			'lightweight': browser_profile.get('lightweight', False),
		},
		'command_history': [],
	}


def load_user_config(config_result: ConfigLoadResult | None = None) -> dict[str, Any]:
	"""Load user configuration using the new config system."""
	# Just get the default config which already loads from the new system
	config = get_default_config(config_result)

	# Load command history from a separate file if it exists
	history_file = CONFIG.OPENBROWSER_CONFIG_DIR / 'command_history.json'
	if history_file.exists():
		try:
			with open(history_file) as f:
				config['command_history'] = json.load(f)
		except (FileNotFoundError, json.JSONDecodeError):
			config['command_history'] = []

	return config


def save_user_config(config: dict[str, Any]) -> None:
	"""Save command history only (config is saved via the new system)."""
	# Only save command history to a separate file
	if 'command_history' in config and isinstance(config['command_history'], list):
		# Ensure command history doesn't exceed maximum length
		history = config['command_history']
		if len(history) > MAX_HISTORY_LENGTH:
			history = history[-MAX_HISTORY_LENGTH:]

		# Save to separate history file
		history_file = CONFIG.OPENBROWSER_CONFIG_DIR / 'command_history.json'
		with open(history_file, 'w') as f:
			json.dump(history, f, indent=2)


def update_config_with_click_args(config: dict[str, Any], ctx: click.Context) -> dict[str, Any]:
	"""Update configuration with command-line arguments."""
	# Ensure required sections exist
	if 'model' not in config:
		config['model'] = {}
	if 'browser' not in config:
		config['browser'] = {}

	# Update configuration with command-line args if provided
	if ctx.params.get('model'):
		config['model']['name'] = ctx.params['model']
	if ctx.params.get('headless') is not None:
		config['browser']['headless'] = ctx.params['headless']
	if ctx.params.get('window_width'):
		config['browser']['window_width'] = ctx.params['window_width']
	if ctx.params.get('window_height'):
		config['browser']['window_height'] = ctx.params['window_height']
	if ctx.params.get('user_data_dir'):
		config['browser']['user_data_dir'] = ctx.params['user_data_dir']
	if ctx.params.get('profile_directory'):
		config['browser']['profile_directory'] = ctx.params['profile_directory']
	if ctx.params.get('cdp_url'):
		config['browser']['cdp_url'] = ctx.params['cdp_url']
	if ctx.params.get('render_mode'):
		config['browser']['render_mode'] = ctx.params['render_mode']
	if ctx.params.get('no_render'):
		config['browser']['render_mode'] = 'no-render'
	if ctx.params.get('lightweight'):
		config['browser']['render_mode'] = 'lightweight'
	if ctx.params.get('text_only'):
		config['browser']['text_only'] = True
	for name in ('block_css', 'block_images', 'block_fonts', 'block_media', 'block_scripts'):
		if ctx.params.get(name) is not None:
			config['browser'][name] = ctx.params[name]

	# Consolidated proxy dict
	proxy: dict[str, str] = {}
	if ctx.params.get('proxy_url'):
		proxy['server'] = ctx.params['proxy_url']
	if ctx.params.get('no_proxy'):
		# Store as comma-separated list string to match Chrome flag
		proxy['bypass'] = ','.join([p.strip() for p in ctx.params['no_proxy'].split(',') if p.strip()])
	if ctx.params.get('proxy_username'):
		proxy['username'] = ctx.params['proxy_username']
	if ctx.params.get('proxy_password'):
		proxy['password'] = ctx.params['proxy_password']
	if proxy:
		config['browser']['proxy'] = proxy

	return config



def get_llm(config: dict[str, Any]):
	"""Get the language model based on config and available API keys."""
	# Lazy imports: these are only needed when actually creating an LLM instance.
	# Importing at module level causes ModuleNotFoundError when optional deps
	# (e.g. anthropic) are not installed, breaking commands like --version.
	from openbrowser.llm.openai.chat import ChatOpenAI

	model_config = config.get('model', {})
	model_name = model_config.get('name')
	temperature = model_config.get('temperature', 0.0)

	# Get API key from config or environment
	api_key = model_config.get('api_keys', {}).get('OPENAI_API_KEY') or CONFIG.OPENAI_API_KEY

	if model_name:
		if model_name.startswith('gpt'):
			if not api_key and not CONFIG.OPENAI_API_KEY:
				print('OpenAI API key not found. Please update your config or set OPENAI_API_KEY environment variable.')
				sys.exit(1)
			return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key or CONFIG.OPENAI_API_KEY)
		elif model_name.startswith('claude'):
			from openbrowser.llm.anthropic.chat import ChatAnthropic

			if not CONFIG.ANTHROPIC_API_KEY:
				print('Anthropic API key not found. Please update your config or set ANTHROPIC_API_KEY environment variable.')
				sys.exit(1)
			return ChatAnthropic(model=model_name, temperature=temperature)
		elif model_name.startswith('gemini'):
			from openbrowser.llm.google.chat import ChatGoogle

			if not CONFIG.GOOGLE_API_KEY:
				print('Google API key not found. Please update your config or set GOOGLE_API_KEY environment variable.')
				sys.exit(1)
			return ChatGoogle(model=model_name, temperature=temperature)
		elif model_name.startswith('oci'):
			# OCI models require additional configuration
			print(
				'OCI models require manual configuration. Please use the ChatOCIRaw class directly with your OCI credentials.'
			)
			sys.exit(1)

	# Auto-detect based on available API keys
	if api_key or CONFIG.OPENAI_API_KEY:
		return ChatOpenAI(model='gpt-5-mini', temperature=temperature, api_key=api_key or CONFIG.OPENAI_API_KEY)
	elif CONFIG.ANTHROPIC_API_KEY:
		from openbrowser.llm.anthropic.chat import ChatAnthropic

		return ChatAnthropic(model='claude-4-sonnet', temperature=temperature)
	elif CONFIG.GOOGLE_API_KEY:
		from openbrowser.llm.google.chat import ChatGoogle

		return ChatGoogle(model='gemini-2.5-pro', temperature=temperature)
	else:
		print(
			'No API keys found. Please update your config or set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.'
		)
		sys.exit(1)





async def run_prompt_mode(prompt: str, ctx: click.Context, debug: bool = False):
	"""Run openbrowser in non-interactive mode with a single prompt."""
	# Import and call setup_logging to ensure proper initialization
	from openbrowser.logging_config import setup_logging

	# Set up logging to only show results by default
	os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'result'

	# Re-run setup_logging to apply the new log level
	setup_logging()

	# The logging is now properly configured by setup_logging()
	# No need to manually configure handlers since setup_logging() handles it

	# Initialize telemetry
	telemetry = ProductTelemetry()
	start_time = time.time()
	error_msg = None

	try:
		# Freeze one config result for the gate and for the browser settings.
		config_result = CONFIG.require_browser_start_allowed()
		config = load_user_config(config_result)
		config = update_config_with_click_args(config, ctx)

		# Get LLM
		llm = get_llm(config)

		# Capture telemetry for CLI start in oneshot mode
		telemetry.capture(
			CLITelemetryEvent(
				version=get_openbrowser_version(),
				action='start',
				mode='oneshot',
				model=llm.model if hasattr(llm, 'model') else None,
				model_provider=llm.__class__.__name__ if llm else None,
			)
		)

		# Get agent settings from config
		agent_settings = AgentSettings.model_validate(config.get('agent', {}))

		# Create browser session with config parameters
		browser_config = config.get('browser', {})
		# Remove None values from browser_config
		browser_config = {k: v for k, v in browser_config.items() if v is not None}
		browser_config = apply_managed_browser_profile_defaults(browser_config, USER_DATA_DIR)
		profile = BrowserProfile(**browser_config)
		browser_session = BrowserSession(
			browser_profile=profile,
		)

		# Create and run agent
		agent = Agent(
			task=prompt,
			llm=llm,
			browser_session=browser_session,
			source='cli',
			**agent_settings.model_dump(),
		)

		await agent.run()

		# Ensure the browser session is fully stopped
		# The agent's close() method only kills the browser if keep_alive=False,
		# but we need to ensure all background tasks are stopped regardless
		if browser_session:
			# Kill the browser session to stop all background tasks
			with contextlib.suppress(Exception):
				await browser_session.kill()

		# Capture telemetry for successful completion
		telemetry.capture(
			CLITelemetryEvent(
				version=get_openbrowser_version(),
				action='task_completed',
				mode='oneshot',
				model=llm.model if hasattr(llm, 'model') else None,
				model_provider=llm.__class__.__name__ if llm else None,
				duration_seconds=time.time() - start_time,
			)
		)

	except ConfigStartupError as error:
		error_msg = error.code
		telemetry.capture(
			CLITelemetryEvent(
				version=get_openbrowser_version(),
				action='error',
				mode='oneshot',
				model=None,
				model_provider=None,
				duration_seconds=time.time() - start_time,
				error_message=error_msg,
			)
		)
		print(f'Config error [{error.code}]: {error.path}', file=sys.stderr)
		print(error.hint, file=sys.stderr)
		sys.exit(1)

	except Exception as e:
		error_msg = str(e)
		# Capture telemetry for error
		telemetry.capture(
			CLITelemetryEvent(
				version=get_openbrowser_version(),
				action='error',
				mode='oneshot',
				model=llm.model if hasattr(llm, 'model') else None,
				model_provider=llm.__class__.__name__ if llm and 'llm' in locals() else None,
				duration_seconds=time.time() - start_time,
				error_message=error_msg,
			)
		)
		if debug:
			import traceback

			traceback.print_exc()
		else:
			print(f'Error: {str(e)}', file=sys.stderr)
		sys.exit(1)
	finally:
		# Ensure telemetry is flushed
		telemetry.flush()

		# Give a brief moment for cleanup to complete
		await asyncio.sleep(0.1)

		# Cancel any remaining tasks to ensure clean exit
		tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
		for task in tasks:
			task.cancel()

		# Wait for all tasks to be cancelled
		if tasks:
			await asyncio.gather(*tasks, return_exceptions=True)



@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Print version and exit')
@click.option(
	'--template',
	type=click.Choice(['default', 'advanced', 'tools'], case_sensitive=False),
	help='Generate a template file (default, advanced, or tools)',
)
@click.option('--output', '-o', type=click.Path(), help='Output file path for template (default: openbrowser_<template>.py)')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files without asking')
@click.option('--model', type=str, help='Model to use (e.g., gpt-5-mini, claude-4-sonnet, gemini-2.5-flash)')
@click.option('--debug', is_flag=True, help='Enable verbose startup logging')
@click.option('--headless', is_flag=True, help='Run browser in headless mode', default=None)
@click.option('--window-width', type=int, help='Browser window width')
@click.option('--window-height', type=int, help='Browser window height')
@click.option(
	'--user-data-dir', type=str, help='Path to Chrome user data directory (e.g. ~/Library/Application Support/Google/Chrome)'
)
@click.option('--profile-directory', type=str, help='Chrome profile directory name (e.g. "Default", "Profile 1")')
@click.option('--cdp-url', type=str, help='Connect to existing Chrome via CDP URL (e.g. http://localhost:9222)')
@click.option(
	'--render-mode',
	type=click.Choice(['full', 'lightweight', 'no-render']),
	help='Rendering policy: full, lightweight, or no-render.',
)
@click.option('--no-render', is_flag=True, help='Shortcut for --render-mode no-render.')
@click.option('--text-only', is_flag=True,
	help='Block external CSS, images, web fonts, media, and text tracks. Scripts and screenshots stay on.')
@click.option('--no-css/--css', 'block_css', default=None,
	help='Block or allow external stylesheets.')
@click.option('--no-images/--images', 'block_images', default=None,
	help='Block or allow image requests.')
@click.option('--no-fonts/--fonts', 'block_fonts', default=None,
	help='Block or allow web font requests.')
@click.option('--no-media/--media', 'block_media', default=None,
	help='Block or allow media and text track requests.')
@click.option('--no-scripts/--scripts', 'block_scripts', default=None,
	help='Block and disable scripts. Use only for server-rendered pages.')
@click.option('--proxy-url', type=str, help='Proxy server for Chromium traffic (e.g. http://host:8080 or socks5://host:1080)')
@click.option('--no-proxy', type=str, help='Comma-separated hosts to bypass proxy (e.g. localhost,127.0.0.1,*.internal)')
@click.option('--proxy-username', type=str, help='Proxy auth username')
@click.option('--proxy-password', type=str, help='Proxy auth password')
@click.option('-p', '--prompt', type=str, help='Run a single task in one-shot mode')
@click.option('--mcp', is_flag=True, help='Run as MCP server (exposes JSON RPC via stdin/stdout)')
@click.pass_context
def main(ctx: click.Context, debug: bool = False, **kwargs):
	"""OpenBrowser - AI Agent for Web Automation

	Examples:

	\b
	  openbrowser-ai -p "Search for the latest AI news"
	  openbrowser-ai -c "await navigate('https://example.com')"
	  openbrowser-ai --mcp
	  openbrowser-ai --template default
	"""

	# Handle template generation
	if kwargs.get('template'):
		_run_template_generation(kwargs['template'], kwargs.get('output'), kwargs.get('force', False))
		return

	if ctx.invoked_subcommand is None:
		# No subcommand, run the main interface
		run_main_interface(ctx, debug, **kwargs)


def run_main_interface(ctx: click.Context, debug: bool = False, **kwargs):
	"""Run the main openbrowser interface."""

	if kwargs['version']:
		from importlib.metadata import version

		print(version('openbrowser-ai'))
		sys.exit(0)

	# Check if MCP server mode is activated
	if kwargs.get('mcp'):
		transport = os.environ.get('OPENBROWSER_MCP_TRANSPORT', 'lean').strip().lower()
		if transport not in {'lean', 'sdk'}:
			print("ERROR: OPENBROWSER_MCP_TRANSPORT must be 'lean' or 'sdk'", file=sys.stderr)
			sys.exit(2)
		if transport == 'lean':
			from openbrowser.mcp.lean import main as lean_mcp_main

			asyncio.run(lean_mcp_main())
			return
		# Capture telemetry for MCP server mode via CLI (suppress any logging from this)
		try:
			telemetry = ProductTelemetry()
			telemetry.capture(
				CLITelemetryEvent(
					version=get_openbrowser_version(),
					action='start',
					mode='mcp_server',
				)
			)
		except Exception:
			# Ignore telemetry errors in MCP mode to prevent any stdout contamination
			pass
		# Run as MCP server
		from openbrowser.mcp.server import main as mcp_main

		asyncio.run(mcp_main())
		return

	# Check if prompt mode is activated
	if kwargs.get('prompt'):
		if Agent is None:
			print('Error: Agent dependencies not installed.')
			print('Install with: pip install "openbrowser-ai[agent]"')
			sys.exit(1)
		# Set environment variable for prompt mode before running
		os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'result'
		# Run in non-interactive mode
		asyncio.run(run_prompt_mode(kwargs['prompt'], ctx, debug))
		return

	# No specific mode requested -- show help
	click.echo(ctx.get_help())


@main.command()
def install():
	"""Install Chromium browser with system dependencies"""
	import platform
	import subprocess

	print('📦 Installing Chromium browser + system dependencies...')
	print('⏳ This may take a few minutes...\n')

	# Build command - only use --with-deps on Linux (it fails on Windows/macOS)
	cmd = ['uvx', 'playwright', 'install', 'chromium']
	if platform.system() == 'Linux':
		cmd.append('--with-deps')
	cmd.append('--no-shell')

	result = subprocess.run(cmd)

	if result.returncode == 0:
		print('\n✅ Installation complete!')
		print('Ready to use! Run: uvx openbrowser-ai')
	else:
		print('\n❌ Installation failed')
		sys.exit(1)


# ============================================================================
# Scrape - Fetch a page and return markdown, text, HTML, links, or JSON
# ============================================================================

def _quiet_stderr_logging() -> None:
	"""Send every log record to stderr so a scrape pipe receives only the result."""
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(logging.Formatter('%(levelname)s %(name)s %(message)s'))
	logging.root.handlers = [handler]
	logging.root.setLevel(logging.CRITICAL)
	for name in list(logging.root.manager.loggerDict):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = [handler]
		logger_obj.setLevel(logging.CRITICAL)
		logger_obj.propagate = False


def _render_web_search_table(results: list[Any]) -> str:
	"""Render search results as a plain-text table."""
	from io import StringIO

	from rich import box
	from rich.console import Console
	from rich.table import Table
	from rich.text import Text

	buffer = StringIO()
	console = Console(
		file=buffer,
		width=120,
		force_terminal=False,
		color_system=None,
		no_color=True,
		highlight=False,
	)
	table = Table(box=box.SIMPLE, pad_edge=False, show_header=True)
	table.add_column("#", justify="right", no_wrap=True)
	table.add_column("Title", overflow="fold")
	table.add_column("URL", overflow="fold")
	table.add_column("Snippet", overflow="fold")
	for index, result in enumerate(results, start=1):
		table.add_row(
			Text(str(index)),
			Text(result.title or "(untitled)"),
			Text(result.url),
			Text(result.snippet),
		)
	console.print(table)
	return buffer.getvalue()


_WEB_SEARCH_OPTIONS = (
	click.argument("query"),
	click.option(
		"--max-results",
		type=click.IntRange(min=1),
		default=5,
		show_default=True,
		help="Maximum OpenBrowser web search results to return. Values above 20 are capped at 20.",
	),
	click.option(
		"--backend",
		default="auto",
		show_default=True,
		help="DDGS backend or comma-delimited backends for OpenBrowser web search.",
	),
	click.option("--region", default="us-en", show_default=True, help="OpenBrowser web search region."),
	click.option(
		"--safe-search",
		type=click.Choice(["on", "moderate", "off"]),
		default="moderate",
		show_default=True,
		help="Safe search level for OpenBrowser web search.",
	),
	click.option(
		"--timelimit",
		type=str,
		help="Limit OpenBrowser web search results by age, for example d, w, m, or y.",
	),
	click.option("--json", "as_json", is_flag=True, help="Print a JSON array of OpenBrowser web search results."),
	click.option("--urls-only", is_flag=True, help="Print one OpenBrowser web search result URL per line."),
	click.option("--output", type=click.Path(dir_okay=False), help="Write OpenBrowser web search output to a file."),
)


def _web_search_options(func):
	for decorator in _WEB_SEARCH_OPTIONS:
		func = decorator(func)
	return func


def _run_web_search(
	query: str,
	max_results: int,
	backend: str,
	region: str,
	safe_search: Literal['on', 'moderate', 'off'],
	timelimit: str | None,
	as_json: bool,
	urls_only: bool,
	output: str | None,
) -> None:
	"""Run OpenBrowser web search. Read source pages before citing."""
	import json
	from pathlib import Path

	if as_json and urls_only:
		raise click.UsageError("--json and --urls-only cannot be used together.")

	from openbrowser.logging_config import setup_logging

	setup_logging(stream=sys.stderr, log_level='critical', force_setup=True)
	_quiet_stderr_logging()

	from openbrowser.web_search import WebSearchError, web_search

	try:
		results = web_search(
			query,
			max_results=max_results,
			backend=backend,
			region=region,
			safe_search=safe_search,
			timelimit=timelimit,
		)
	except WebSearchError as error:
		raise click.ClickException(str(error)) from error
	except (TypeError, ValueError) as error:
		raise click.BadParameter(str(error)) from error

	if as_json:
		rendered = json.dumps([result.as_dict() for result in results], indent=2, ensure_ascii=False)
	elif urls_only:
		rendered = "\n".join(result.url for result in results)
	else:
		rendered = _render_web_search_table(results) if results else "No OpenBrowser web search results."

	if output:
		path = Path(output)
		try:
			path.write_text(rendered + ("" if not rendered or rendered.endswith("\n") else "\n"), encoding="utf-8")
		except OSError as error:
			raise click.ClickException(f"Could not write OpenBrowser web search output to {path}: {error}") from error
		click.echo(f"Wrote OpenBrowser web search output to {path}", err=True)
	elif rendered:
		click.echo(rendered, nl=not rendered.endswith("\n"))


@main.command(name="web-search")
@_web_search_options
def web_search_command(
	query: str,
	max_results: int,
	backend: str,
	region: str,
	safe_search: Literal['on', 'moderate', 'off'],
	timelimit: str | None,
	as_json: bool,
	urls_only: bool,
	output: str | None,
) -> None:
	"""Run OpenBrowser web search. Read source pages before citing."""
	_run_web_search(query, max_results, backend, region, safe_search, timelimit, as_json, urls_only, output)


@main.command(name="search", hidden=True)
@_web_search_options
def search_compat_command(
	query: str,
	max_results: int,
	backend: str,
	region: str,
	safe_search: Literal['on', 'moderate', 'off'],
	timelimit: str | None,
	as_json: bool,
	urls_only: bool,
	output: str | None,
) -> None:
	"""Deprecated alias for web-search."""
	_run_web_search(query, max_results, backend, region, safe_search, timelimit, as_json, urls_only, output)


@main.command()
@click.argument('urls', nargs=-1, required=False, metavar='URL...')
@click.option(
	'--urls-file',
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	help='Read one URL per line from a file. Blank lines and lines starting with # are ignored.',
)
@click.option(
	'--format',
	'output_format',
	type=click.Choice(['markdown', 'text', 'html', 'links', 'json']),
	default='markdown',
	show_default=True,
	help='Output format.',
)
@click.option('--selector', type=str, help='CSS selector to limit text, HTML, or JSON output.')
@click.option(
	'--engine',
	type=click.Choice(['auto', 'http', 'browser']),
	default='auto',
	show_default=True,
	help='Fetch with HTTP first, always use HTTP, or always use the browser.',
)
@click.option(
	'--no-browser',
	is_flag=True,
	help='Fetch over HTTP only. Does not load Blink or V8 and never falls back to a browser.',
)
@click.option('--wait-for', type=str, help='CSS selector to wait for before scraping.')
@click.option(
	'--wait-until',
	type=click.Choice(['load', 'domcontentloaded', 'networkidle', 'commit']),
	default='load',
	show_default=True,
	help='Navigation readiness state.',
)
@click.option('--timeout-ms', type=int, help='Navigation and selector wait timeout in milliseconds.')
@click.option('--extract-links', is_flag=True, help='Preserve links in markdown output.')
@click.option('--max-chars', type=int, help='Truncate string content to this many characters.')
@click.option('--only-main-content', is_flag=True,
	help='Remove navigation, header, footer, aside, cookie banners, and ad blocks before extraction.')
@click.option('--include-tag', 'include_tags', multiple=True,
	help='Keep only the first matching selector. Repeat to try several selectors.')
@click.option('--exclude-tag', 'exclude_tags', multiple=True,
	help='Remove matching elements before extraction. Repeat for several selectors.')
@click.option('--lightweight', is_flag=True, help='Disable images, web fonts, GPU acceleration, and media.')
@click.option(
	'--render-mode',
	type=click.Choice(['full', 'lightweight', 'no-render']),
	help='Rendering policy. Overrides --lightweight.',
)
@click.option('--text-only', 'text_only', is_flag=True,
	help='Block external CSS, images, web fonts, media, and text tracks for the browser fallback.')
@click.option('--no-css/--css', 'block_css', default=None,
	help='Block or allow external stylesheets in the browser fallback.')
@click.option('--no-images/--images', 'block_images', default=None,
	help='Block or allow image requests in the browser fallback.')
@click.option('--no-fonts/--fonts', 'block_fonts', default=None,
	help='Block or allow web font requests in the browser fallback.')
@click.option('--no-media/--media', 'block_media', default=None,
	help='Block or allow media requests in the browser fallback.')
@click.option(
	'--concurrency',
	type=click.IntRange(min=1),
	default=4,
	show_default=True,
	help='Maximum concurrent HTTP requests in batch mode.',
)
@click.option('--output', '-o', type=click.Path(), help='Write the result to a file.')
@click.option(
	'--output-format',
	'batch_output_format',
	type=click.Choice(['json', 'jsonl']),
	help='Batch serialization: a JSON array or one JSON object per line. A .jsonl output path selects JSONL.',
)
@click.option('--jsonl', is_flag=True, help='Print one JSON object per line for a batch.')
@click.option('--json', 'as_json', is_flag=True, help='Print the full result object as JSON.')
@click.pass_context
def scrape(
	ctx: click.Context,
	urls: tuple[str, ...],
	urls_file: Path | None,
	output_format: str,
	selector: str | None,
	engine: str,
	no_browser: bool,
	wait_for: str | None,
	wait_until: str,
	timeout_ms: int | None,
	extract_links: bool,
	max_chars: int | None,
	only_main_content: bool,
	include_tags: tuple[str, ...],
	exclude_tags: tuple[str, ...],
	lightweight: bool,
	render_mode: str | None,
	text_only: bool,
	block_css: bool | None,
	block_images: bool | None,
	block_fonts: bool | None,
	block_media: bool | None,
	concurrency: int,
	output: str | None,
	batch_output_format: str | None,
	jsonl: bool,
	as_json: bool,
):
	"""Scrape one or more URLs without an LLM.

	Examples:

	\b
	# Markdown of a page
	openbrowser-ai scrape https://example.com

	\b
	# JSON list from a CSS selector
	openbrowser-ai scrape https://example.com --format json --selector "article"

	\b
	# Batch scrape with bounded HTTP concurrency and JSONL output
	openbrowser-ai scrape https://a.example https://b.example --jsonl --concurrency 8

	\b
	# Read URLs from a file and write a JSON array
	openbrowser-ai scrape --urls-file urls.txt --output results.json
	"""
	url_list = [url.strip() for url in urls if url.strip()]
	if urls_file is not None:
		try:
			url_list.extend(read_scrape_urls_file(urls_file))
		except ValueError as error:
			raise click.ClickException(str(error)) from error
	if not url_list:
		raise click.UsageError('Provide at least one URL or --urls-file.')
	if no_browser:
		if engine == 'browser':
			raise click.UsageError('--no-browser cannot be combined with --engine browser.')
		engine = 'http'
	batch_mode = len(url_list) > 1 or urls_file is not None

	if engine == 'http':
		try:
			http_payload = run_http_scrape_sync(
				url_list,
				batch_mode=batch_mode,
				output_format=output_format,
				selector=selector,
				extract_links=extract_links,
				timeout_ms=timeout_ms,
				max_chars=max_chars,
				concurrency=concurrency,
				only_main_content=only_main_content,
				include_tags=list(include_tags) or None,
				exclude_tags=list(exclude_tags) or None,
			)
		except Exception as error:
			click.echo(f'Scrape failed: {type(error).__name__}: {error}', err=True)
			sys.exit(1)
		if http_payload is None:
			click.echo('Scrape failed: the HTTP engine returned no payload', err=True)
			sys.exit(1)
		emit_scrape_payload(
			http_payload,
			batch_mode=batch_mode,
			jsonl=jsonl,
			batch_output_format=batch_output_format,
			output=output,
			as_json=as_json,
		)
		return

	from openbrowser.logging_config import setup_logging

	# Keep stdout clean: the scrape result is the only thing a pipe should receive.
	setup_logging(stream=sys.stderr, log_level='critical', force_setup=True)
	_quiet_stderr_logging()

	# The browser stack and the code-use namespace are imported inside the
	# browser fallback. A static page is served over HTTP, and loading those
	# modules costs RSS and import time the HTTP path never uses.
	from openbrowser.scrape.http import scrape_http

	parent = ctx.parent.params if ctx.parent is not None else {}
	config_result = CONFIG.require_browser_start_allowed()
	config = load_user_config(config_result)
	if ctx.parent is not None:
		config = update_config_with_click_args(config, ctx.parent)
	browser_config = {k: v for k, v in config.get('browser', {}).items() if v is not None}
	browser_config = apply_managed_browser_profile_defaults(browser_config, USER_DATA_DIR)
	# A one-shot scrape does not need extensions, and storage_state conflicts with a persistent profile.
	browser_config['enable_default_extensions'] = False
	browser_config.pop('storage_state', None)
	# Scrape does not use VNC, extensions, or exact screenshot bytes, so the
	# Playwright headless shell is safe here and uses about half the memory of
	# full Chrome. An explicit executable_path still wins.
	env_shell = os.environ.get('OPENBROWSER_USE_HEADLESS_SHELL', '').strip().lower()
	if env_shell in ('0', 'false', 'no'):
		browser_config['use_headless_shell'] = False
	elif env_shell in ('1', 'true', 'yes'):
		browser_config['use_headless_shell'] = True
	else:
		browser_config.setdefault('use_headless_shell', True)
	effective_render_mode = render_mode or parent.get('render_mode')
	if parent.get('no_render'):
		effective_render_mode = 'no-render'
	if effective_render_mode:
		browser_config['render_mode'] = effective_render_mode
	elif lightweight:
		browser_config['render_mode'] = 'lightweight'
	if text_only:
		browser_config['text_only'] = True
	for name, value in (
		('block_css', block_css),
		('block_images', block_images),
		('block_fonts', block_fonts),
		('block_media', block_media),
	):
		if value is not None:
			browser_config[name] = value

	async def _run_single_scrape(url: str) -> dict:
		if engine in {'auto', 'http'}:
			http_result = await scrape_http(
				url,
				format=output_format,
				selector=selector,
				extract_links=extract_links,
				timeout_ms=timeout_ms,
				engine=engine,
				only_main_content=only_main_content,
				include_tags=list(include_tags) or None,
				exclude_tags=list(exclude_tags) or None,
			)
			if http_result is not None:
				if max_chars is not None and isinstance(http_result.get('content'), str):
					content = http_result['content']
					if len(content) > max_chars:
						http_result['content'] = content[:max_chars]
						http_result['truncated'] = True
						http_result['total_chars'] = len(content)
				return http_result
			if engine == 'http':
				raise RuntimeError(f'HTTP scrape failed for {url}')

		from openbrowser.browser import BrowserProfile, BrowserSession
		from openbrowser.code_use.namespace import create_namespace
		from openbrowser.tools.service import CodeAgentTools

		session = BrowserSession(browser_profile=BrowserProfile(**browser_config))
		namespace = None
		try:
			await session.start()
			_quiet_stderr_logging()
			namespace = create_namespace(browser_session=session, tools=CodeAgentTools())
			return await namespace['scrape_page'](
				url,
				format=output_format,
				selector=selector,
				wait_for=wait_for,
				wait_until=wait_until,
				timeout_ms=timeout_ms,
				extract_links=extract_links,
				max_chars=max_chars,
				engine='browser',
				only_main_content=only_main_content,
				include_tags=list(include_tags) or None,
				exclude_tags=list(exclude_tags) or None,
			)
		finally:
			if namespace is not None:
				close_http = cast(
					Callable[[], Awaitable[None]] | None,
					namespace.get('_close_http_client'),
				)
				if close_http is not None:
					with contextlib.suppress(Exception):
						await close_http()
			with contextlib.suppress(Exception):
				await session.kill()

	async def _run_batch_scrape(batch_urls: list[str]) -> list[dict]:
		from openbrowser.browser import BrowserProfile, BrowserSession
		from openbrowser.code_use.namespace import create_namespace
		from openbrowser.tools.service import CodeAgentTools

		session = BrowserSession(browser_profile=BrowserProfile(**browser_config))
		namespace = None
		try:
			namespace = create_namespace(browser_session=session, tools=CodeAgentTools())
			return await namespace['scrape_pages'](
				batch_urls,
				concurrency=concurrency,
				format=output_format,
				selector=selector,
				wait_for=wait_for,
				wait_until=wait_until,
				timeout_ms=timeout_ms,
				extract_links=extract_links,
				max_chars=max_chars,
				engine=engine,
				only_main_content=only_main_content,
				include_tags=list(include_tags) or None,
				exclude_tags=list(exclude_tags) or None,
			)
		finally:
			if namespace is not None:
				close_http = cast(
					Callable[[], Awaitable[None]] | None,
					namespace.get('_close_http_client'),
				)
				if close_http is not None:
					with contextlib.suppress(Exception):
						await close_http()
			with contextlib.suppress(Exception):
				await session.kill()

	try:
		if batch_mode:
			batch_payload = asyncio.run(_run_batch_scrape(url_list))
		else:
			payload: dict[str, Any] = asyncio.run(_run_single_scrape(url_list[0]))
	except Exception as error:
		click.echo(f'Scrape failed: {type(error).__name__}: {error}', err=True)
		sys.exit(1)

	emit_scrape_payload(
		batch_payload if batch_mode else payload,
		batch_mode=batch_mode,
		jsonl=jsonl,
		batch_output_format=batch_output_format,
		output=output,
		as_json=as_json,
	)


@main.command()
@click.argument('url')
@click.option('--max-urls', type=click.IntRange(min=1), default=100, show_default=True,
	help='Maximum URLs to return.')
@click.option('--include', 'include_pattern', type=str,
	help='Only keep URLs that match this regular expression.')
@click.option('--exclude', 'exclude_pattern', type=str,
	help='Drop URLs that match this regular expression.')
@click.option('--no-sitemap', 'skip_sitemap', is_flag=True,
	help='Skip /sitemap.xml discovery and map only links on the page.')
@click.option('--timeout-ms', type=int, help='Per-request timeout in milliseconds.')
@click.option('--output', '-o', type=click.Path(), help='Write the JSON result to a file.')
def map(url: str, max_urls: int, include_pattern: str | None, exclude_pattern: str | None,
		skip_sitemap: bool, timeout_ms: int | None, output: str | None):
	"""List URLs discovered from URL and its sitemap. No browser starts."""
	from openbrowser.scrape.crawl import map_urls

	try:
		result = asyncio.run(
			map_urls(
				url,
				max_urls=max_urls,
				include=include_pattern,
				exclude=exclude_pattern,
				use_sitemap=not skip_sitemap,
				timeout_ms=timeout_ms,
			)
		)
	except Exception as error:
		click.echo(f'Map failed: {type(error).__name__}: {error}', err=True)
		sys.exit(1)
	rendered = json.dumps(result, indent=2, ensure_ascii=False)
	if output:
		Path(output).write_text(rendered + "\n", encoding="utf-8")
		click.echo(f'Wrote map output to {output}', err=True)
	else:
		click.echo(rendered)


@main.command()
@click.argument('url')
@click.option('--max-pages', type=click.IntRange(min=1), default=10, show_default=True,
	help='Maximum pages to fetch.')
@click.option('--max-depth', type=click.IntRange(min=0), default=2, show_default=True,
	help='Maximum link hops from the start URL.')
@click.option('--format', 'output_format', type=click.Choice(['markdown', 'text', 'html', 'links']),
	default='markdown', show_default=True, help='Output format for each page.')
@click.option('--selector', type=str, help='CSS selector to limit each page output.')
@click.option('--include', 'include_pattern', type=str,
	help='Only follow URLs that match this regular expression.')
@click.option('--exclude', 'exclude_pattern', type=str,
	help='Do not follow URLs that match this regular expression.')
@click.option('--concurrency', type=click.IntRange(min=1), default=4, show_default=True,
	help='Maximum concurrent requests per depth level.')
@click.option('--timeout-ms', type=int, help='Per-request timeout in milliseconds.')
@click.option('--output', '-o', type=click.Path(), help='Write the JSON result to a file.')
def crawl(url: str, max_pages: int, max_depth: int, output_format: str, selector: str | None,
		include_pattern: str | None, exclude_pattern: str | None, concurrency: int,
		timeout_ms: int | None, output: str | None):
	"""Crawl same-origin links from URL and return page content. No browser starts."""
	from openbrowser.scrape.crawl import crawl_site

	try:
		result = asyncio.run(
			crawl_site(
				url,
				max_pages=max_pages,
				max_depth=max_depth,
				format=output_format,
				selector=selector,
				include=include_pattern,
				exclude=exclude_pattern,
				concurrency=concurrency,
				timeout_ms=timeout_ms,
			)
		)
	except Exception as error:
		click.echo(f'Crawl failed: {type(error).__name__}: {error}', err=True)
		sys.exit(1)
	rendered = json.dumps(result, indent=2, ensure_ascii=False)
	if output:
		Path(output).write_text(rendered + "\n", encoding="utf-8")
		click.echo(f'Wrote crawl output to {output}', err=True)
	else:
		click.echo(rendered)


# ============================================================================
# Template Generation - Generate template files
# ============================================================================

# Template metadata


def _run_template_generation(template: str, output: str | None, force: bool):
	"""Generate a template file (called from main CLI)."""
	# Determine output path
	if output:
		output_path = Path(output)
	else:
		output_path = Path.cwd() / f'openbrowser_{template}.py'

	# Read template file
	try:
		templates_dir = Path(__file__).parent / 'cli_templates'
		template_file = INIT_TEMPLATES[template]['file']
		template_path = templates_dir / template_file
		content = template_path.read_text(encoding='utf-8')
	except Exception as e:
		click.echo(f'❌ Error reading template: {e}', err=True)
		sys.exit(1)

	# Write file
	if _write_init_file(output_path, content, force):
		click.echo(f'✅ Created {output_path}')
		click.echo('\nNext steps:')
		click.echo('  This generated Agent template uses ChatOpenAI and needs OPENAI_API_KEY.')
		click.echo('  MCP and openbrowser-ai -c need no LLM key.')
		click.echo('  1. Install openbrowser:')
		click.echo('     uv pip install openbrowser-ai')
		click.echo('  2. Set OPENAI_API_KEY in the .env file or environment:')
		click.echo('     OPENAI_API_KEY=your-key')
		click.echo('  3. Run your script:')
		click.echo(f'     python {output_path.name}')
	else:
		sys.exit(1)


def _write_init_file(output_path: Path, content: str, force: bool = False) -> bool:
	"""Write content to a file, with safety checks."""
	# Check if file already exists
	if output_path.exists() and not force:
		click.echo(f'⚠️  File already exists: {output_path}')
		if not click.confirm('Overwrite?', default=False):
			click.echo('❌ Cancelled')
			return False

	# Ensure parent directory exists
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Write file
	try:
		output_path.write_text(content, encoding='utf-8')
		return True
	except Exception as e:
		click.echo(f'❌ Error writing file: {e}', err=True)
		return False


@main.command('init')
@click.option(
	'--template',
	'-t',
	type=click.Choice(['default', 'advanced', 'tools'], case_sensitive=False),
	help='Template to use',
)
@click.option(
	'--output',
	'-o',
	type=click.Path(),
	help='Output file path (default: openbrowser_<template>.py)',
)
@click.option(
	'--force',
	'-f',
	is_flag=True,
	help='Overwrite existing files without asking',
)
@click.option(
	'--list',
	'-l',
	'list_templates',
	is_flag=True,
	help='List available templates',
)
def init(
	template: str | None,
	output: str | None,
	force: bool,
	list_templates: bool,
):
	"""
	Generate an openbrowser template file to get started quickly.

	Examples:

	\b
	# Interactive mode - prompts for template selection
	uvx openbrowser-ai init

	\b
	# Generate default template
	uvx openbrowser-ai init --template default

	\b
	# Generate advanced template with custom filename
	uvx openbrowser-ai init --template advanced --output my_script.py

	\b
	# List available templates
	uvx openbrowser-ai init --list
	"""

	# Handle --list flag
	if list_templates:
		click.echo('Available templates:\n')
		for name, info in INIT_TEMPLATES.items():
			click.echo(f'  {name:12} - {info["description"]}')
		return

	# Interactive template selection if not provided
	if not template:
		click.echo('Available templates:\n')
		for name, info in INIT_TEMPLATES.items():
			click.echo(f'  {name:12} - {info["description"]}')
		click.echo()

		template = click.prompt(
			'Which template would you like to use?',
			type=click.Choice(['default', 'advanced', 'tools'], case_sensitive=False),
			default='default',
		)

	# Template is guaranteed to be set at this point (either from option or prompt)
	assert template is not None

	# Determine output path
	if output:
		output_path = Path(output)
	else:
		output_path = Path.cwd() / f'openbrowser_{template}.py'

	# Read template file
	try:
		templates_dir = Path(__file__).parent / 'cli_templates'
		template_file = INIT_TEMPLATES[template]['file']
		template_path = templates_dir / template_file
		content = template_path.read_text(encoding='utf-8')
	except Exception as e:
		click.echo(f'❌ Error reading template: {e}', err=True)
		sys.exit(1)

	# Write file
	if _write_init_file(output_path, content, force):
		click.echo(f'✅ Created {output_path}')
		click.echo('\nNext steps:')
		click.echo('  This generated Agent template uses ChatOpenAI and needs OPENAI_API_KEY.')
		click.echo('  MCP and openbrowser-ai -c need no LLM key.')
		click.echo('  1. Install openbrowser:')
		click.echo('     uv pip install openbrowser-ai')
		click.echo('  2. Set OPENAI_API_KEY in the .env file or environment:')
		click.echo('     OPENAI_API_KEY=your-key')
		click.echo('  3. Run your script:')
		click.echo(f'     python {output_path.name}')
	else:
		sys.exit(1)


if __name__ == '__main__':
	main()
