# pyright: reportMissingImports=false

# Check for MCP mode early to prevent logging initialization
import sys

if '--mcp' in sys.argv:
	import asyncio
	import logging
	import os

	os.environ['OPENBROWSER_LOGGING_LEVEL'] = 'critical'
	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
	logging.disable(logging.CRITICAL)

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

	from openbrowser.mcp.server import main as mcp_main

	asyncio.run(mcp_main())
	sys.exit(0)

# Check for -c (execute code) mode early -- minimal imports for fast startup
if '-c' in sys.argv:
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
		sys.exit(1)
	sys.exit(0)

# Check for daemon subcommand early
if len(sys.argv) > 1 and sys.argv[1] == 'daemon':
	import os
	os.environ['OPENBROWSER_SETUP_LOGGING'] = 'false'
	try:
		from dotenv import load_dotenv
		load_dotenv()
	except ImportError:
		pass

	import asyncio

	from openbrowser.daemon.client import DaemonClient

	sub = sys.argv[2] if len(sys.argv) > 2 else 'status'
	client = DaemonClient()

	if sub == 'start':
		async def _start():
			status = await client.status()
			if status.success:
				print('Daemon is already running')
				return
			await client._start_daemon()
			print('Daemon started')
		asyncio.run(_start())
	elif sub == 'stop':
		resp = asyncio.run(client.stop())
		print(resp.output or resp.error)
		if not resp.success:
			sys.exit(1)
	elif sub == 'status':
		resp = asyncio.run(client.status())
		print(resp.output or resp.error)
		if not resp.success:
			sys.exit(1)
	elif sub == 'restart':
		import time as _time

		async def _restart():
			await client.stop()
			# Wait for old daemon to fully shut down
			stopped = False
			deadline = _time.time() + 5.0
			while _time.time() < deadline:
				resp = await client.status()
				if not resp.success:
					stopped = True
					break
				await asyncio.sleep(0.3)
			if not stopped:
				print('Old daemon did not stop within timeout', file=sys.stderr)
				sys.exit(1)
			await client._start_daemon()
			print('Daemon restarted')
		asyncio.run(_restart())
	else:
		print(f'Unknown daemon command: {sub}', file=sys.stderr)
		print('Usage: openbrowser-ai daemon [start|stop|status|restart]', file=sys.stderr)
		sys.exit(1)
	sys.exit(0)

# Special case: install command doesn't need CLI dependencies
if len(sys.argv) > 1 and sys.argv[1] == 'install':
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
if 'init' in sys.argv:
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
if '--template' in sys.argv:
	from pathlib import Path

	import click

	from openbrowser.init_cmd import INIT_TEMPLATES

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
		click.echo('  1. Install openbrowser:')
		click.echo('     uv pip install openbrowser-ai')
		click.echo('  2. Set up your API key in .env file or environment:')
		click.echo('     your llm API Key, e.g., OPENAI_API_KEY=your-key')
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
from typing import Any

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

from openbrowser.config import CONFIG

# Set USER_DATA_DIR now that CONFIG is imported
USER_DATA_DIR = CONFIG.OPENBROWSER_PROFILES_DIR / 'cli'

# Ensure directories exist
CONFIG.OPENBROWSER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default User settings
MAX_HISTORY_LENGTH = 100


def get_default_config() -> dict[str, Any]:
	"""Return default configuration dictionary using the new config system."""
	# Load config from the new config system
	config_data = CONFIG.load_config()

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
			'allowed_domains': browser_profile.get('allowed_domains'),
			'wait_between_actions': browser_profile.get('wait_between_actions'),
			'is_mobile': browser_profile.get('is_mobile'),
			'device_scale_factor': browser_profile.get('device_scale_factor'),
			'disable_security': browser_profile.get('disable_security'),
		},
		'command_history': [],
	}


def load_user_config() -> dict[str, Any]:
	"""Load user configuration using the new config system."""
	# Just get the default config which already loads from the new system
	config = get_default_config()

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
		# Load config
		config = load_user_config()
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
		# Create BrowserProfile with user_data_dir
		profile = BrowserProfile(user_data_dir=str(USER_DATA_DIR), **browser_config)
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
			try:
				# Kill the browser session to stop all background tasks
				await browser_session.kill()
			except Exception:
				# Ignore errors during cleanup
				pass

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
# Template Generation - Generate template files
# ============================================================================

# Template metadata
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
		click.echo('  1. Install openbrowser:')
		click.echo('     uv pip install openbrowser-ai')
		click.echo('  2. Set up your API key in .env file or environment:')
		click.echo('     your llm API Key, e.g., OPENAI_API_KEY=your-key')
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
		click.echo('  1. Install openbrowser:')
		click.echo('     uv pip install openbrowser-ai')
		click.echo('  2. Set up your API key in .env file or environment:')
		click.echo('     your llm API Key, e.g., OPENAI_API_KEY=your-key')
		click.echo('  3. Run your script:')
		click.echo(f'     python {output_path.name}')
	else:
		sys.exit(1)


if __name__ == '__main__':
	main()
