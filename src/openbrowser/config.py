"""Configuration system for openbrowser with automatic migration support.

Performance optimizations:
- Cached environment variable lookups
- Lazy directory creation (only when needed)
- Cached config file loading
- Reduced redundant Path operations
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from openbrowser.config_lifecycle import (
	ConfigFilePhase,
	ConfigFileSnapshot,
	ConfigLoadResult,
	ConfigStartupError,
	restart_required,
	start_allowed,
)

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
			path = self.XDG_CONFIG_HOME / 'openbrowser'
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
				config_dir = self.XDG_CONFIG_HOME / 'openbrowser'
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
	OPENBROWSER_USER_DATA_DIR: str | None = Field(default=None)
	OPENBROWSER_STORAGE_STATE: str | None = Field(default=None)

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
	new_config.browser_profile[profile_id] = BrowserProfileEntry(id=profile_id, default=True, headless=True, user_data_dir=None)

	# Create default LLM entry
	new_config.llm[llm_id] = LLMEntry(id=llm_id, default=True, model='gpt-4.1-mini', api_key='your-openai-api-key-here')

	# Create default agent entry
	new_config.agent[agent_id] = AgentEntry(id=agent_id, default=True)

	return new_config


# Cache for loaded config files (keyed by the user path)
FileIdentity = tuple[int, int, int, int, int]
_config_cache: dict[str, tuple[FileIdentity, ConfigLoadResult]] = {}


def _file_identity(path: Path) -> FileIdentity:
	stat = path.stat()
	return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _content_version(raw: bytes | None) -> str | None:
	if raw is None:
		return None
	return hashlib.sha256(raw).hexdigest()[:16]


def _config_hint(path: Path, phase: ConfigFilePhase) -> str | None:
	display_path = str(path)
	if phase is ConfigFilePhase.READ_FAILED:
		return (
			f'Fix the file permissions, or move {display_path} to a unique backup name and restart. '
			'Moving it aside creates a new default and does not migrate old settings.'
		)
	if phase in (ConfigFilePhase.INVALID, ConfigFilePhase.LEGACY_FORMAT):
		return (
			f'Inspect {display_path}, or move it to a unique backup name and restart. '
			'Moving it aside creates a new default and does not migrate old settings.'
		)
	if phase is ConfigFilePhase.WRITE_FAILED:
		return (
			f'Check the config directory permissions for {display_path}. '
			'The file may have been created; run `openbrowser-ai config status` before retrying.'
		)
	return None


def _snapshot(
	path: Path,
	resolved_path: Path,
	phase: ConfigFilePhase,
	*,
	file_version: str | None = None,
	error_code: str | None = None,
	hint: str | None = None,
) -> ConfigFileSnapshot:
	return ConfigFileSnapshot(
		path=str(path),
		resolved_path=str(resolved_path),
		phase=phase,
		error_code=error_code,
		hint=hint if hint is not None else _config_hint(path, phase),
		browser_start_allowed=phase in (ConfigFilePhase.READY, ConfigFilePhase.CREATED),
		file_version=file_version,
	)


def _blocked_result(path: Path, resolved_path: Path, phase: ConfigFilePhase, error_code: str) -> ConfigLoadResult:
	return ConfigLoadResult(
		config=create_default_config(),
		snapshot=_snapshot(path, resolved_path, phase, error_code=error_code),
	)


def _fsync_directory(path: Path) -> None:
	if os.name == 'nt':
		# Windows has no portable directory-fsync operation. The temporary
		# file's fsync is the available durability boundary before linking.
		return
	descriptor = os.open(path, os.O_RDONLY)
	try:
		os.fsync(descriptor)
	finally:
		os.close(descriptor)


def _publish_default(path: Path, resolved_path: Path) -> ConfigLoadResult:
	"""Create a missing config without exposing partial bytes or overwriting a winner."""
	try:
		resolved_path.parent.mkdir(parents=True, exist_ok=True)
	except OSError:
		return _blocked_result(path, resolved_path, ConfigFilePhase.WRITE_FAILED, 'config_write_failed')

	new_config = create_default_config()
	payload = json.dumps(new_config.model_dump(), indent=2).encode('utf-8')
	temp_name: str | None = None
	try:
		with tempfile.NamedTemporaryFile(
			prefix=f'.{resolved_path.name}.',
			suffix='.tmp',
			dir=resolved_path.parent,
			delete=False,
		) as handle:
			temp_name = handle.name
			handle.write(payload)
			handle.flush()
			os.fsync(handle.fileno())
	except OSError:
		if temp_name is not None:
			Path(temp_name).unlink(missing_ok=True)
		return _blocked_result(path, resolved_path, ConfigFilePhase.WRITE_FAILED, 'config_write_failed')

	assert temp_name is not None
	try:
		try:
			os.link(temp_name, resolved_path)
		except FileExistsError:
			winner, _ = _load_existing(path, resolved_path)
			return winner
		except OSError:
			return _blocked_result(path, resolved_path, ConfigFilePhase.WRITE_FAILED, 'config_write_failed')
		try:
			_fsync_directory(resolved_path.parent)
		except OSError:
			return _blocked_result(path, resolved_path, ConfigFilePhase.WRITE_FAILED, 'config_write_failed')
		return ConfigLoadResult(
			config=new_config,
			snapshot=_snapshot(
				path,
				resolved_path,
				ConfigFilePhase.CREATED,
				file_version=_content_version(payload),
			),
		)
	finally:
		Path(temp_name).unlink(missing_ok=True)


@cache
def _known_browser_profile_fields() -> frozenset[str]:
	"""Return the BrowserProfile fields that identify the old flat config shape."""
	try:
		from openbrowser.browser.profile import BrowserProfile
	except ImportError:
		return frozenset(BrowserProfileEntry.model_fields)
	return frozenset(BrowserProfile.model_fields)


def _legacy_browser_profile(data: dict[str, Any]) -> bool:
	browser_profile = data.get('browser_profile')
	if not isinstance(browser_profile, dict) or not browser_profile:
		return False
	if any(isinstance(value, dict) for value in browser_profile.values()):
		return False
	return any(key in _known_browser_profile_fields() for key in browser_profile)


def _parse_existing(
	path: Path,
	resolved_path: Path,
	raw: bytes,
) -> ConfigLoadResult:
	try:
		decoded = raw.decode('utf-8')
		data = json.loads(decoded)
	except (UnicodeDecodeError, json.JSONDecodeError):
		return ConfigLoadResult(
			config=create_default_config(),
			snapshot=_snapshot(
				path,
				resolved_path,
				ConfigFilePhase.INVALID,
				file_version=_content_version(raw),
				error_code='config_invalid_json',
			),
		)

	if not isinstance(data, dict):
		return ConfigLoadResult(
			config=create_default_config(),
			snapshot=_snapshot(
				path,
				resolved_path,
				ConfigFilePhase.INVALID,
				file_version=_content_version(raw),
				error_code='config_invalid_schema',
			),
		)

	known_keys = ('browser_profile', 'llm', 'agent')
	has_db_wrapper = all(key in data and isinstance(data[key], dict) for key in known_keys)
	if has_db_wrapper:
		browser_profile = data.get('browser_profile')
		if not isinstance(browser_profile, dict) or not browser_profile:
			return ConfigLoadResult(
				config=create_default_config(),
				snapshot=_snapshot(
					path,
					resolved_path,
					ConfigFilePhase.INVALID,
					file_version=_content_version(raw),
					error_code='config_invalid_schema',
				),
			)
		if not all(isinstance(value, dict) for value in browser_profile.values()):
			phase = ConfigFilePhase.LEGACY_FORMAT if _legacy_browser_profile(data) else ConfigFilePhase.INVALID
			return ConfigLoadResult(
				config=create_default_config(),
				snapshot=_snapshot(
					path,
					resolved_path,
					phase,
					file_version=_content_version(raw),
					error_code=(
						'config_legacy_format'
						if phase is ConfigFilePhase.LEGACY_FORMAT
						else 'config_invalid_schema'
					),
				),
			)
		try:
			config = DBStyleConfigJSON.model_validate(data)
		except ValidationError:
			return ConfigLoadResult(
				config=create_default_config(),
				snapshot=_snapshot(
					path,
					resolved_path,
					ConfigFilePhase.INVALID,
					file_version=_content_version(raw),
					error_code='config_invalid_schema',
				),
			)
		return ConfigLoadResult(
			config=config,
			snapshot=_snapshot(path, resolved_path, ConfigFilePhase.READY, file_version=_content_version(raw)),
		)

	return ConfigLoadResult(
		config=create_default_config(),
		snapshot=_snapshot(
			path,
			resolved_path,
			ConfigFilePhase.INVALID,
			file_version=_content_version(raw),
			error_code='config_invalid_schema',
		),
	)


def _load_existing(path: Path, resolved_path: Path) -> tuple[ConfigLoadResult, FileIdentity | None]:
	if not resolved_path.is_file():
		return _blocked_result(path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed'), None

	for _ in range(2):
		try:
			before = _file_identity(resolved_path)
		except FileNotFoundError:
			return _publish_default(path, resolved_path), None
		except (OSError, RuntimeError):
			return _blocked_result(path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed'), None
		try:
			raw = resolved_path.read_bytes()
		except FileNotFoundError:
			continue
		except OSError:
			return _blocked_result(path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed'), None
		try:
			after = _file_identity(resolved_path)
		except FileNotFoundError:
			continue
		except (OSError, RuntimeError):
			return _blocked_result(path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed'), None
		if before == after:
			return _parse_existing(path, resolved_path, raw), after
	return _blocked_result(path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed'), None


def load_and_migrate_config_result(config_path: Path) -> ConfigLoadResult:
	"""Load config without ever rewriting an existing file."""
	user_path = config_path.expanduser()
	try:
		resolved_path = user_path.resolve(strict=False)
	except (OSError, RuntimeError):
		return _blocked_result(user_path, user_path, ConfigFilePhase.READ_FAILED, 'config_read_failed')
	cache_key = str(user_path)
	for _ in range(2):
		try:
			identity = _file_identity(resolved_path)
		except FileNotFoundError:
			return _publish_default(user_path, resolved_path)
		except (OSError, RuntimeError):
			return _blocked_result(user_path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed')
		cached = _config_cache.get(cache_key)
		if cached is not None and cached[0] == identity:
			return cached[1]

		result, read_identity = _load_existing(user_path, resolved_path)
		if read_identity is None:
			return result
		try:
			current_identity = _file_identity(resolved_path)
		except FileNotFoundError:
			continue
		except (OSError, RuntimeError):
			return _blocked_result(user_path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed')
		if current_identity != read_identity:
			continue
		if result.snapshot.phase in (ConfigFilePhase.READY, ConfigFilePhase.LEGACY_FORMAT, ConfigFilePhase.INVALID):
			_config_cache[cache_key] = (read_identity, result)
		return result
	return _blocked_result(user_path, resolved_path, ConfigFilePhase.READ_FAILED, 'config_read_failed')


def load_and_migrate_config(config_path: Path) -> DBStyleConfigJSON:
	"""Return the parsed config, using an in-memory default for a blocked file."""
	return load_and_migrate_config_result(config_path).config


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
			return xdg_config / 'openbrowser' / 'config.json'

	def load_result(self) -> ConfigLoadResult:
		"""Load the config file once and return its lifecycle snapshot."""
		return load_and_migrate_config_result(self._get_config_path())

	def _get_db_config(self) -> DBStyleConfigJSON:
		"""Return the parsed config, using an in-memory default for a blocked file."""
		return self.load_result().config

	def _get_default_profile(self, db_config: DBStyleConfigJSON | None = None) -> dict[str, Any]:
		"""Get the default browser profile configuration."""
		db_config = db_config or self._get_db_config()
		for profile in db_config.browser_profile.values():
			if profile.default:
				return profile.model_dump(exclude_none=True)

		# Return first profile if no default
		if db_config.browser_profile:
			return next(iter(db_config.browser_profile.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_llm(self, db_config: DBStyleConfigJSON | None = None) -> dict[str, Any]:
		"""Get the default LLM configuration."""
		db_config = db_config or self._get_db_config()
		for llm in db_config.llm.values():
			if llm.default:
				return llm.model_dump(exclude_none=True)

		# Return first LLM if no default
		if db_config.llm:
			return next(iter(db_config.llm.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_agent(self, db_config: DBStyleConfigJSON | None = None) -> dict[str, Any]:
		"""Get the default agent configuration."""
		db_config = db_config or self._get_db_config()
		for agent in db_config.agent.values():
			if agent.default:
				return agent.model_dump(exclude_none=True)

		# Return first agent if no default
		if db_config.agent:
			return next(iter(db_config.agent.values())).model_dump(exclude_none=True)

		return {}

	def config_dict_from_result(self, result: ConfigLoadResult) -> dict[str, Any]:
		"""Convert one frozen load result into the legacy config dictionary."""
		config = {
			'browser_profile': self._get_default_profile(result.config),
			'llm': self._get_default_llm(result.config),
			'agent': self._get_default_agent(result.config),
		}
		return self._apply_env_overrides(config)

	def _load_config(self) -> dict[str, Any]:
		"""Load configuration with env var overrides for MCP components."""
		return self.config_dict_from_result(self.load_result())

	def _apply_env_overrides(self, config: dict[str, Any]) -> dict[str, Any]:
		"""Apply process environment overrides to one already-selected config."""
		env_config = self._get_env_config()

		if env_config.OPENBROWSER_HEADLESS is not None:
			config['browser_profile']['headless'] = env_config.OPENBROWSER_HEADLESS

		if env_config.OPENBROWSER_ALLOWED_DOMAINS:
			domains = [d.strip() for d in env_config.OPENBROWSER_ALLOWED_DOMAINS.split(',') if d.strip()]
			config['browser_profile']['allowed_domains'] = domains

		if env_config.OPENBROWSER_USER_DATA_DIR:
			config['browser_profile']['user_data_dir'] = env_config.OPENBROWSER_USER_DATA_DIR

		if env_config.OPENBROWSER_STORAGE_STATE:
			config['browser_profile']['storage_state'] = env_config.OPENBROWSER_STORAGE_STATE

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
			config.setdefault('browser_profile', {})
			config['browser_profile']['proxy'] = proxy_dict

		if env_config.OPENAI_API_KEY:
			config['llm']['api_key'] = env_config.OPENAI_API_KEY

		if env_config.OPENBROWSER_LLM_MODEL:
			config['llm']['model'] = env_config.OPENBROWSER_LLM_MODEL

		return config

	def require_browser_start_allowed(self, startup_result: ConfigLoadResult | None = None) -> ConfigLoadResult:
		"""Return the frozen config result or raise a typed startup error."""
		if startup_result is not None:
			if not start_allowed(startup_result.snapshot):
				raise ConfigStartupError(
					startup_result.snapshot,
					code='config_restart_required',
					hint='Fix the config, then restart the daemon or MCP server.',
				)
			current = self.load_result()
			if restart_required(startup_result.snapshot, current.snapshot):
				raise ConfigStartupError(
					startup_result.snapshot,
					code='config_restart_required',
					hint='The config changed or became unavailable. Restart the daemon or MCP server.',
				)
			return startup_result

		result = self.load_result()
		if not start_allowed(result.snapshot):
			raise ConfigStartupError(result.snapshot)
		return result


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


def is_openbrowser_managed_profile_dir(path: str | Path | None) -> bool:
	"""Return True when the path lives under OpenBrowser's managed profiles dir."""
	if not path:
		return False

	try:
		resolved = Path(path).expanduser().resolve()
	except OSError:
		resolved = Path(path).expanduser()

	try:
		return resolved.is_relative_to(CONFIG.OPENBROWSER_PROFILES_DIR)
	except ValueError:
		return False


def apply_managed_browser_profile_defaults(
	profile_data: dict[str, Any], default_user_data_dir: str | Path | None = None
) -> dict[str, Any]:
	"""Fill in managed persistence defaults for OpenBrowser-owned browser profiles."""
	resolved_profile = dict(profile_data)

	if default_user_data_dir and not resolved_profile.get('user_data_dir'):
		resolved_profile['user_data_dir'] = str(Path(default_user_data_dir).expanduser())

	if not resolved_profile.get('storage_state') and is_openbrowser_managed_profile_dir(
		resolved_profile.get('user_data_dir')
	):
		user_data_dir = Path(str(resolved_profile['user_data_dir'])).expanduser()
		resolved_profile['storage_state'] = str(user_data_dir / 'storage_state.json')

	return resolved_profile
