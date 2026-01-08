"""Configuration system for openbrowser with automatic migration support.

Performance optimizations:
- Cached environment variable lookups
- Lazy directory creation (only when needed)
- Cached config file loading
- Reduced redundant Path operations
"""

import json
import logging
import os
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


@cache
def is_running_in_docker() -> bool:
	"""Detect if we are running in a docker container, for the purpose of optimizing chrome launch flags (dev shm usage, gpu settings, etc.)."""
	try:
		if Path('/.dockerenv').exists() or 'docker' in Path('/proc/1/cgroup').read_text().lower():
			return True
	except Exception:
		pass

	try:
		# if init proc (PID 1) looks like uvicorn/python/uv/etc. then we're in Docker
		# if init proc (PID 1) looks like bash/systemd/init/etc. then we're probably NOT in Docker
		init_cmd = ' '.join(psutil.Process(1).cmdline())
		if ('py' in init_cmd) or ('uv' in init_cmd) or ('app' in init_cmd):
			return True
	except Exception:
		pass

	try:
		# if less than 10 total running procs, then we're almost certainly in a container
		if len(psutil.pids()) < 10:
			return True
	except Exception:
		pass

	return False


# Cached environment variable lookups for frequently accessed values
@lru_cache(maxsize=32)
def _get_env_cached(key: str, default: str = '') -> str:
	"""Get environment variable with caching."""
	return os.getenv(key, default)


@lru_cache(maxsize=16)
def _get_env_bool_cached(key: str, default: bool = False) -> bool:
	"""Get boolean environment variable with caching."""
	val = os.getenv(key, str(default)).lower()
	return val[:1] in 'ty1' if val else default


@lru_cache(maxsize=8)
def _get_path_cached(key: str, default: str) -> Path:
	"""Get path environment variable with caching."""
	return Path(os.getenv(key, default)).expanduser().resolve()


class OldConfig:
	"""Original lazy-loading configuration class for environment variables.
	
	Optimized with cached property access patterns.
	"""

	# Cache for directory creation tracking
	_dirs_created = False
	
	# Cached values (class-level for singleton-like behavior)
	_cached_values: dict[str, Any] = {}

	@property
	def OPENBROWSER_LOGGING_LEVEL(self) -> str:
		return _get_env_cached('OPENBROWSER_LOGGING_LEVEL', 'info').lower()

	@property
	def ANONYMIZED_TELEMETRY(self) -> bool:
		return _get_env_bool_cached('ANONYMIZED_TELEMETRY', True)

	@property
	def BROWSER_USE_CLOUD_SYNC(self) -> bool:
		return os.getenv('BROWSER_USE_CLOUD_SYNC', str(self.ANONYMIZED_TELEMETRY)).lower()[:1] in 'ty1'

	@property
	def BROWSER_USE_CLOUD_API_URL(self) -> str:
		url = _get_env_cached('BROWSER_USE_CLOUD_API_URL', 'https://api.browser-use.com')
		assert '://' in url, 'BROWSER_USE_CLOUD_API_URL must be a valid URL'
		return url

	@property
	def BROWSER_USE_CLOUD_UI_URL(self) -> str:
		url = _get_env_cached('BROWSER_USE_CLOUD_UI_URL', '')
		# Allow empty string as default, only validate if set
		if url and '://' not in url:
			raise AssertionError('BROWSER_USE_CLOUD_UI_URL must be a valid URL if set')
		return url

	# Path configuration - use cached path lookups
	@property
	def XDG_CACHE_HOME(self) -> Path:
		return _get_path_cached('XDG_CACHE_HOME', '~/.cache')

	@property
	def XDG_CONFIG_HOME(self) -> Path:
		return _get_path_cached('XDG_CONFIG_HOME', '~/.config')

	@property
	def OPENBROWSER_CONFIG_DIR(self) -> Path:
		config_dir_env = os.getenv('OPENBROWSER_CONFIG_DIR')
		if config_dir_env:
			path = Path(config_dir_env).expanduser().resolve()
		else:
			path = self.XDG_CONFIG_HOME / 'browseruse'
		self._ensure_dirs()
		return path

	@property
	def OPENBROWSER_CONFIG_FILE(self) -> Path:
		return self.OPENBROWSER_CONFIG_DIR / 'config.json'

	@property
	def OPENBROWSER_PROFILES_DIR(self) -> Path:
		path = self.OPENBROWSER_CONFIG_DIR / 'profiles'
		self._ensure_dirs()
		return path

	@property
	def OPENBROWSER_DEFAULT_USER_DATA_DIR(self) -> Path:
		return self.OPENBROWSER_PROFILES_DIR / 'default'

	@property
	def OPENBROWSER_EXTENSIONS_DIR(self) -> Path:
		path = self.OPENBROWSER_CONFIG_DIR / 'extensions'
		self._ensure_dirs()
		return path

	def _ensure_dirs(self) -> None:
		"""Create directories if they don't exist (only once)."""
		if not OldConfig._dirs_created:
			config_dir_env = os.getenv('OPENBROWSER_CONFIG_DIR')
			if config_dir_env:
				config_dir = Path(config_dir_env).expanduser().resolve()
			else:
				config_dir = self.XDG_CONFIG_HOME / 'browseruse'
			config_dir.mkdir(parents=True, exist_ok=True)
			(config_dir / 'profiles').mkdir(parents=True, exist_ok=True)
			(config_dir / 'extensions').mkdir(parents=True, exist_ok=True)
			OldConfig._dirs_created = True

	# LLM API key configuration - use cached lookups
	@property
	def OPENAI_API_KEY(self) -> str:
		return _get_env_cached('OPENAI_API_KEY', '')

	@property
	def ANTHROPIC_API_KEY(self) -> str:
		return _get_env_cached('ANTHROPIC_API_KEY', '')

	@property
	def GOOGLE_API_KEY(self) -> str:
		return _get_env_cached('GOOGLE_API_KEY', '')

	@property
	def DEEPSEEK_API_KEY(self) -> str:
		return _get_env_cached('DEEPSEEK_API_KEY', '')

	@property
	def GROK_API_KEY(self) -> str:
		return _get_env_cached('GROK_API_KEY', '')

	@property
	def NOVITA_API_KEY(self) -> str:
		return _get_env_cached('NOVITA_API_KEY', '')

	@property
	def AZURE_OPENAI_ENDPOINT(self) -> str:
		return _get_env_cached('AZURE_OPENAI_ENDPOINT', '')

	@property
	def AZURE_OPENAI_KEY(self) -> str:
		return _get_env_cached('AZURE_OPENAI_KEY', '')

	@property
	def SKIP_LLM_API_KEY_VERIFICATION(self) -> bool:
		return _get_env_bool_cached('SKIP_LLM_API_KEY_VERIFICATION', False)

	@property
	def DEFAULT_LLM(self) -> str:
		return _get_env_cached('DEFAULT_LLM', '')

	# Runtime hints
	@property
	def IN_DOCKER(self) -> bool:
		return _get_env_bool_cached('IN_DOCKER', False) or is_running_in_docker()

	@property
	def IS_IN_EVALS(self) -> bool:
		return _get_env_bool_cached('IS_IN_EVALS', False)

	@property
	def WIN_FONT_DIR(self) -> str:
		return _get_env_cached('WIN_FONT_DIR', 'C:\\Windows\\Fonts')


class FlatEnvConfig(BaseSettings):
	"""All environment variables in a flat namespace."""

	model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=True, extra='allow')

	# Logging and telemetry
	OPENBROWSER_LOGGING_LEVEL: str = Field(default='info')
	CDP_LOGGING_LEVEL: str = Field(default='WARNING')
	OPENBROWSER_DEBUG_LOG_FILE: str | None = Field(default=None)
	OPENBROWSER_INFO_LOG_FILE: str | None = Field(default=None)
	ANONYMIZED_TELEMETRY: bool = Field(default=True)
	BROWSER_USE_CLOUD_SYNC: bool | None = Field(default=None)
	BROWSER_USE_CLOUD_API_URL: str = Field(default='https://api.browser-use.com')
	BROWSER_USE_CLOUD_UI_URL: str = Field(default='')

	# Path configuration
	XDG_CACHE_HOME: str = Field(default='~/.cache')
	XDG_CONFIG_HOME: str = Field(default='~/.config')
	OPENBROWSER_CONFIG_DIR: str | None = Field(default=None)

	# LLM API keys
	OPENAI_API_KEY: str = Field(default='')
	ANTHROPIC_API_KEY: str = Field(default='')
	GOOGLE_API_KEY: str = Field(default='')
	DEEPSEEK_API_KEY: str = Field(default='')
	GROK_API_KEY: str = Field(default='')
	NOVITA_API_KEY: str = Field(default='')
	AZURE_OPENAI_ENDPOINT: str = Field(default='')
	AZURE_OPENAI_KEY: str = Field(default='')
	SKIP_LLM_API_KEY_VERIFICATION: bool = Field(default=False)
	DEFAULT_LLM: str = Field(default='')

	# Runtime hints
	IN_DOCKER: bool | None = Field(default=None)
	IS_IN_EVALS: bool = Field(default=False)
	WIN_FONT_DIR: str = Field(default='C:\\Windows\\Fonts')

	# MCP-specific env vars
	OPENBROWSER_CONFIG_PATH: str | None = Field(default=None)
	OPENBROWSER_HEADLESS: bool | None = Field(default=None)
	OPENBROWSER_ALLOWED_DOMAINS: str | None = Field(default=None)
	OPENBROWSER_LLM_MODEL: str | None = Field(default=None)

	# Proxy env vars
	OPENBROWSER_PROXY_URL: str | None = Field(default=None)
	OPENBROWSER_NO_PROXY: str | None = Field(default=None)
	OPENBROWSER_PROXY_USERNAME: str | None = Field(default=None)
	OPENBROWSER_PROXY_PASSWORD: str | None = Field(default=None)


class DBStyleEntry(BaseModel):
	"""Database-style entry with UUID and metadata."""

	id: str = Field(default_factory=lambda: str(uuid4()))
	default: bool = Field(default=False)
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class BrowserProfileEntry(DBStyleEntry):
	"""Browser profile configuration entry - accepts any BrowserProfile fields."""

	model_config = ConfigDict(extra='allow')

	# Common browser profile fields for reference
	headless: bool | None = None
	user_data_dir: str | None = None
	allowed_domains: list[str] | None = None
	downloads_path: str | None = None


class LLMEntry(DBStyleEntry):
	"""LLM configuration entry."""

	api_key: str | None = None
	model: str | None = None
	temperature: float | None = None
	max_tokens: int | None = None


class AgentEntry(DBStyleEntry):
	"""Agent configuration entry."""

	max_steps: int | None = None
	use_vision: bool | None = None
	system_prompt: str | None = None


class DBStyleConfigJSON(BaseModel):
	"""New database-style configuration format."""

	browser_profile: dict[str, BrowserProfileEntry] = Field(default_factory=dict)
	llm: dict[str, LLMEntry] = Field(default_factory=dict)
	agent: dict[str, AgentEntry] = Field(default_factory=dict)


def create_default_config() -> DBStyleConfigJSON:
	"""Create a fresh default configuration."""
	logger.debug('Creating fresh default config.json')

	new_config = DBStyleConfigJSON()

	# Generate default IDs
	profile_id = str(uuid4())
	llm_id = str(uuid4())
	agent_id = str(uuid4())

	# Create default browser profile entry
	new_config.browser_profile[profile_id] = BrowserProfileEntry(id=profile_id, default=True, headless=False, user_data_dir=None)

	# Create default LLM entry
	new_config.llm[llm_id] = LLMEntry(id=llm_id, default=True, model='gpt-4.1-mini', api_key='your-openai-api-key-here')

	# Create default agent entry
	new_config.agent[agent_id] = AgentEntry(id=agent_id, default=True)

	return new_config


# Cache for loaded config files (keyed by path)
_config_cache: dict[str, tuple[float, DBStyleConfigJSON]] = {}


def load_and_migrate_config(config_path: Path) -> DBStyleConfigJSON:
	"""Load config.json or create fresh one if old format detected.
	
	Uses caching to avoid repeated file reads.
	"""
	config_path_str = str(config_path)
	
	# Check cache first
	if config_path_str in _config_cache:
		cached_mtime, cached_config = _config_cache[config_path_str]
		try:
			current_mtime = config_path.stat().st_mtime
			if current_mtime == cached_mtime:
				return cached_config
		except FileNotFoundError:
			pass  # File was deleted, recreate it
	
	if not config_path.exists():
		# Create fresh config with defaults
		config_path.parent.mkdir(parents=True, exist_ok=True)
		new_config = create_default_config()
		with open(config_path, 'w') as f:
			json.dump(new_config.model_dump(), f, indent=2)
		# Cache the new config
		_config_cache[config_path_str] = (config_path.stat().st_mtime, new_config)
		return new_config

	try:
		with open(config_path) as f:
			data = json.load(f)

		# Check if it's already in DB-style format
		if all(key in data for key in ['browser_profile', 'llm', 'agent']) and all(
			isinstance(data.get(key, {}), dict) for key in ['browser_profile', 'llm', 'agent']
		):
			# Check if the values are DB-style entries (have UUIDs as keys)
			if data.get('browser_profile') and all(isinstance(v, dict) and 'id' in v for v in data['browser_profile'].values()):
				# Already in new format
				config = DBStyleConfigJSON(**data)
				_config_cache[config_path_str] = (config_path.stat().st_mtime, config)
				return config

		# Old format detected - delete it and create fresh config
		logger.debug(f'Old config format detected at {config_path}, creating fresh config')
		new_config = create_default_config()

		# Overwrite with new config
		with open(config_path, 'w') as f:
			json.dump(new_config.model_dump(), f, indent=2)

		logger.debug(f'Created fresh config.json at {config_path}')
		_config_cache[config_path_str] = (config_path.stat().st_mtime, new_config)
		return new_config

	except Exception as e:
		logger.error(f'Failed to load config from {config_path}: {e}, creating fresh config')
		# On any error, create fresh config
		new_config = create_default_config()
		try:
			with open(config_path, 'w') as f:
				json.dump(new_config.model_dump(), f, indent=2)
			_config_cache[config_path_str] = (config_path.stat().st_mtime, new_config)
		except Exception as write_error:
			logger.error(f'Failed to write fresh config: {write_error}')
		return new_config


class Config:
	"""Backward-compatible configuration class that merges all config sources.

	Optimized with caching for frequently accessed attributes.
	"""
	
	__slots__ = ('_dirs_created', '_old_config', '_env_config')

	def __init__(self):
		# Cache for directory creation tracking only
		self._dirs_created = False
		# Lazy-initialized config instances
		self._old_config: OldConfig | None = None
		self._env_config: FlatEnvConfig | None = None

	def _get_old_config(self) -> OldConfig:
		"""Get or create OldConfig instance."""
		if self._old_config is None:
			self._old_config = OldConfig()
		return self._old_config

	def _get_env_config(self) -> FlatEnvConfig:
		"""Get or create FlatEnvConfig instance."""
		if self._env_config is None:
			self._env_config = FlatEnvConfig()
		return self._env_config

	def __getattr__(self, name: str) -> Any:
		"""Dynamically proxy all attributes to cached instances."""
		# Special handling for internal attributes
		if name.startswith('_'):
			raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

		# Use cached old config instance
		old_config = self._get_old_config()

		# Always use old config for all attributes (it handles env vars with proper transformations)
		if hasattr(old_config, name):
			return getattr(old_config, name)

		# For new MCP-specific attributes not in old config
		env_config = self._get_env_config()
		if hasattr(env_config, name):
			return getattr(env_config, name)

		# Handle special methods
		if name == 'get_default_profile':
			return lambda: self._get_default_profile()
		elif name == 'get_default_llm':
			return lambda: self._get_default_llm()
		elif name == 'get_default_agent':
			return lambda: self._get_default_agent()
		elif name == 'load_config':
			return lambda: self._load_config()
		elif name == '_ensure_dirs':
			return lambda: old_config._ensure_dirs()

		raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

	def _get_config_path(self) -> Path:
		"""Get config path from fresh env config."""
		env_config = self._get_env_config()
		if env_config.OPENBROWSER_CONFIG_PATH:
			return Path(env_config.OPENBROWSER_CONFIG_PATH).expanduser()
		elif env_config.OPENBROWSER_CONFIG_DIR:
			return Path(env_config.OPENBROWSER_CONFIG_DIR).expanduser() / 'config.json'
		else:
			xdg_config = Path(env_config.XDG_CONFIG_HOME).expanduser()
			return xdg_config / 'browseruse' / 'config.json'

	def _get_db_config(self) -> DBStyleConfigJSON:
		"""Load and migrate config.json."""
		config_path = self._get_config_path()
		return load_and_migrate_config(config_path)

	def _get_default_profile(self) -> dict[str, Any]:
		"""Get the default browser profile configuration."""
		db_config = self._get_db_config()
		for profile in db_config.browser_profile.values():
			if profile.default:
				return profile.model_dump(exclude_none=True)

		# Return first profile if no default
		if db_config.browser_profile:
			return next(iter(db_config.browser_profile.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_llm(self) -> dict[str, Any]:
		"""Get the default LLM configuration."""
		db_config = self._get_db_config()
		for llm in db_config.llm.values():
			if llm.default:
				return llm.model_dump(exclude_none=True)

		# Return first LLM if no default
		if db_config.llm:
			return next(iter(db_config.llm.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_agent(self) -> dict[str, Any]:
		"""Get the default agent configuration."""
		db_config = self._get_db_config()
		for agent in db_config.agent.values():
			if agent.default:
				return agent.model_dump(exclude_none=True)

		# Return first agent if no default
		if db_config.agent:
			return next(iter(db_config.agent.values())).model_dump(exclude_none=True)

		return {}

	def _load_config(self) -> dict[str, Any]:
		"""Load configuration with env var overrides for MCP components."""
		config = {
			'browser_profile': self._get_default_profile(),
			'llm': self._get_default_llm(),
			'agent': self._get_default_agent(),
		}

		# Use cached env config for overrides
		env_config = self._get_env_config()

		# Apply MCP-specific env var overrides
		if env_config.OPENBROWSER_HEADLESS is not None:
			config['browser_profile']['headless'] = env_config.OPENBROWSER_HEADLESS

		if env_config.OPENBROWSER_ALLOWED_DOMAINS:
			domains = [d.strip() for d in env_config.OPENBROWSER_ALLOWED_DOMAINS.split(',') if d.strip()]
			config['browser_profile']['allowed_domains'] = domains

		# Proxy settings (Chromium) -> consolidated `proxy` dict
		proxy_dict: dict[str, Any] = {}
		if env_config.OPENBROWSER_PROXY_URL:
			proxy_dict['server'] = env_config.OPENBROWSER_PROXY_URL
		if env_config.OPENBROWSER_NO_PROXY:
			# store bypass as comma-separated string to match Chrome flag
			proxy_dict['bypass'] = ','.join([d.strip() for d in env_config.OPENBROWSER_NO_PROXY.split(',') if d.strip()])
		if env_config.OPENBROWSER_PROXY_USERNAME:
			proxy_dict['username'] = env_config.OPENBROWSER_PROXY_USERNAME
		if env_config.OPENBROWSER_PROXY_PASSWORD:
			proxy_dict['password'] = env_config.OPENBROWSER_PROXY_PASSWORD
		if proxy_dict:
			# ensure section exists
			config.setdefault('browser_profile', {})
			config['browser_profile']['proxy'] = proxy_dict

		if env_config.OPENAI_API_KEY:
			config['llm']['api_key'] = env_config.OPENAI_API_KEY

		if env_config.OPENBROWSER_LLM_MODEL:
			config['llm']['model'] = env_config.OPENBROWSER_LLM_MODEL

		return config


# Create singleton instance
CONFIG = Config()


# Helper functions for MCP components
def load_openbrowser_config() -> dict[str, Any]:
	"""Load openbrowser configuration for MCP components."""
	return CONFIG.load_config()


def get_default_profile(config: dict[str, Any]) -> dict[str, Any]:
	"""Get default browser profile from config dict."""
	return config.get('browser_profile', {})


def get_default_llm(config: dict[str, Any]) -> dict[str, Any]:
	"""Get default LLM config from config dict."""
	return config.get('llm', {})
