"""Configuration system for openbrowser with automatic migration support."""

import json
import logging
import os
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    HAS_PSUTIL = False

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


@cache
def is_running_in_docker() -> bool:
    """Detect if we are running in a docker container.
    
    Used for optimizing chrome launch flags (dev shm usage, gpu settings, etc.)
    """
    try:
        if Path('/.dockerenv').exists():
            return True
        cgroup_path = Path('/proc/1/cgroup')
        if cgroup_path.exists() and 'docker' in cgroup_path.read_text().lower():
            return True
    except Exception:
        pass

    if HAS_PSUTIL:
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


class EnvConfig(BaseSettings):
    """Environment variable configuration using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='allow'
    )

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
    GROQ_API_KEY: str = Field(default='')
    DEEPSEEK_API_KEY: str = Field(default='')
    OPENROUTER_API_KEY: str = Field(default='')
    AZURE_OPENAI_ENDPOINT: str = Field(default='')
    AZURE_OPENAI_KEY: str = Field(default='')
    AWS_ACCESS_KEY_ID: str = Field(default='')
    AWS_SECRET_ACCESS_KEY: str = Field(default='')
    AWS_REGION: str = Field(default='us-east-1')
    OCI_API_KEY: str = Field(default='')
    CEREBRAS_API_KEY: str = Field(default='')
    SKIP_LLM_API_KEY_VERIFICATION: bool = Field(default=False)
    DEFAULT_LLM: str = Field(default='')

    # Runtime hints
    IN_DOCKER: bool | None = Field(default=None)
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
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BrowserProfileEntry(DBStyleEntry):
    """Browser profile configuration entry."""

    model_config = ConfigDict(extra='allow')

    # Common browser profile fields
    headless: bool | None = None
    user_data_dir: str | None = None
    allowed_domains: list[str] | None = None
    downloads_path: str | None = None
    disable_security: bool | None = None
    window_width: int | None = None
    window_height: int | None = None
    proxy: dict[str, Any] | None = None


class LLMEntry(DBStyleEntry):
    """LLM configuration entry."""

    provider: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    base_url: str | None = None


class AgentEntry(DBStyleEntry):
    """Agent configuration entry."""

    max_steps: int | None = None
    use_vision: bool | None = None
    system_prompt: str | None = None
    llm: str | None = None  # Reference to LLM entry ID
    browser_profile: str | None = None  # Reference to browser profile entry ID


class ConfigJSON(BaseModel):
    """Configuration file format."""

    browser_profiles: dict[str, BrowserProfileEntry] = Field(default_factory=dict)
    llms: dict[str, LLMEntry] = Field(default_factory=dict)
    agents: dict[str, AgentEntry] = Field(default_factory=dict)


def create_default_config() -> ConfigJSON:
    """Create a fresh default configuration."""
    logger.debug('Creating fresh default config.json')

    new_config = ConfigJSON()

    # Generate default IDs
    profile_id = str(uuid4())
    llm_id = str(uuid4())
    agent_id = str(uuid4())

    # Create default browser profile entry
    new_config.browser_profiles[profile_id] = BrowserProfileEntry(
        id=profile_id,
        default=True,
        headless=False,
        user_data_dir=None
    )

    # Create default LLM entry
    new_config.llms[llm_id] = LLMEntry(
        id=llm_id,
        default=True,
        provider='openai',
        model='gpt-4o-mini',
        api_key=None  # Will be read from env
    )

    # Create default agent entry
    new_config.agents[agent_id] = AgentEntry(
        id=agent_id,
        default=True,
        llm=llm_id,
        browser_profile=profile_id
    )

    return new_config


def load_and_migrate_config(config_path: Path) -> ConfigJSON:
    """Load config.json or create fresh one if old format detected."""
    if not config_path.exists():
        # Create fresh config with defaults
        config_path.parent.mkdir(parents=True, exist_ok=True)
        new_config = create_default_config()
        with open(config_path, 'w') as f:
            json.dump(new_config.model_dump(), f, indent=2)
        return new_config

    try:
        with open(config_path) as f:
            data = json.load(f)

        # Check if it's already in expected format
        if all(key in data for key in ['browser_profiles', 'llms', 'agents']):
            # Check if values are DB-style entries (have UUIDs as keys)
            if data.get('browser_profiles') and all(
                isinstance(v, dict) and 'id' in v for v in data['browser_profiles'].values()
            ):
                return ConfigJSON(**data)

        # Old format detected - delete it and create fresh config
        logger.debug(f'Old config format detected at {config_path}, creating fresh config')
        new_config = create_default_config()

        # Overwrite with new config
        with open(config_path, 'w') as f:
            json.dump(new_config.model_dump(), f, indent=2)

        logger.debug(f'Created fresh config.json at {config_path}')
        return new_config

    except Exception as e:
        logger.error(f'Failed to load config from {config_path}: {e}, creating fresh config')
        # On any error, create fresh config
        new_config = create_default_config()
        try:
            with open(config_path, 'w') as f:
                json.dump(new_config.model_dump(), f, indent=2)
        except Exception as write_error:
            logger.error(f'Failed to write fresh config: {write_error}')
        return new_config


class Config:
    """Configuration class that merges environment and file config.

    Re-reads environment variables on every access for flexibility.
    """

    _instance: 'Config | None' = None
    _dirs_created: bool = False

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def LOGGING_LEVEL(self) -> str:
        return os.getenv('OPENBROWSER_LOGGING_LEVEL', 'info').lower()

    @property
    def ANONYMIZED_TELEMETRY(self) -> bool:
        return os.getenv('ANONYMIZED_TELEMETRY', 'true').lower()[:1] in 'ty1'

    @property
    def XDG_CACHE_HOME(self) -> Path:
        return Path(os.getenv('XDG_CACHE_HOME', '~/.cache')).expanduser().resolve()

    @property
    def XDG_CONFIG_HOME(self) -> Path:
        return Path(os.getenv('XDG_CONFIG_HOME', '~/.config')).expanduser().resolve()

    @property
    def CONFIG_DIR(self) -> Path:
        path = Path(
            os.getenv('OPENBROWSER_CONFIG_DIR', str(self.XDG_CONFIG_HOME / 'openbrowser'))
        ).expanduser().resolve()
        self._ensure_dirs()
        return path

    @property
    def CONFIG_FILE(self) -> Path:
        return self.CONFIG_DIR / 'config.json'

    @property
    def PROFILES_DIR(self) -> Path:
        path = self.CONFIG_DIR / 'profiles'
        self._ensure_dirs()
        return path

    @property
    def DEFAULT_USER_DATA_DIR(self) -> Path:
        return self.PROFILES_DIR / 'default'

    @property
    def EXTENSIONS_DIR(self) -> Path:
        path = self.CONFIG_DIR / 'extensions'
        self._ensure_dirs()
        return path

    def _ensure_dirs(self) -> None:
        """Create directories if they don't exist (only once)"""
        if not self._dirs_created:
            config_dir = Path(
                os.getenv('OPENBROWSER_CONFIG_DIR', str(self.XDG_CONFIG_HOME / 'openbrowser'))
            ).expanduser().resolve()
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / 'profiles').mkdir(parents=True, exist_ok=True)
            (config_dir / 'extensions').mkdir(parents=True, exist_ok=True)
            Config._dirs_created = True

    # LLM API key configuration
    @property
    def OPENAI_API_KEY(self) -> str:
        return os.getenv('OPENAI_API_KEY', '')

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return os.getenv('ANTHROPIC_API_KEY', '')

    @property
    def GOOGLE_API_KEY(self) -> str:
        return os.getenv('GOOGLE_API_KEY', '')

    @property
    def GROQ_API_KEY(self) -> str:
        return os.getenv('GROQ_API_KEY', '')

    @property
    def DEEPSEEK_API_KEY(self) -> str:
        return os.getenv('DEEPSEEK_API_KEY', '')

    @property
    def OPENROUTER_API_KEY(self) -> str:
        return os.getenv('OPENROUTER_API_KEY', '')

    @property
    def AZURE_OPENAI_ENDPOINT(self) -> str:
        return os.getenv('AZURE_OPENAI_ENDPOINT', '')

    @property
    def AZURE_OPENAI_KEY(self) -> str:
        return os.getenv('AZURE_OPENAI_KEY', '')

    @property
    def AWS_ACCESS_KEY_ID(self) -> str:
        return os.getenv('AWS_ACCESS_KEY_ID', '')

    @property
    def AWS_SECRET_ACCESS_KEY(self) -> str:
        return os.getenv('AWS_SECRET_ACCESS_KEY', '')

    @property
    def AWS_REGION(self) -> str:
        return os.getenv('AWS_REGION', 'us-east-1')

    @property
    def OCI_API_KEY(self) -> str:
        return os.getenv('OCI_API_KEY', '')

    @property
    def CEREBRAS_API_KEY(self) -> str:
        return os.getenv('CEREBRAS_API_KEY', '')

    @property
    def SKIP_LLM_API_KEY_VERIFICATION(self) -> bool:
        return os.getenv('SKIP_LLM_API_KEY_VERIFICATION', 'false').lower()[:1] in 'ty1'

    @property
    def DEFAULT_LLM(self) -> str:
        return os.getenv('DEFAULT_LLM', '')

    # Runtime hints
    @property
    def IN_DOCKER(self) -> bool:
        return os.getenv('IN_DOCKER', 'false').lower()[:1] in 'ty1' or is_running_in_docker()

    @property
    def WIN_FONT_DIR(self) -> str:
        return os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts')

    def _get_config_path(self) -> Path:
        """Get config path from env config."""
        env_config = EnvConfig()
        if env_config.OPENBROWSER_CONFIG_PATH:
            return Path(env_config.OPENBROWSER_CONFIG_PATH).expanduser()
        elif env_config.OPENBROWSER_CONFIG_DIR:
            return Path(env_config.OPENBROWSER_CONFIG_DIR).expanduser() / 'config.json'
        else:
            xdg_config = Path(env_config.XDG_CONFIG_HOME).expanduser()
            return xdg_config / 'openbrowser' / 'config.json'

    def _get_db_config(self) -> ConfigJSON:
        """Load and migrate config.json."""
        config_path = self._get_config_path()
        return load_and_migrate_config(config_path)

    def get_default_profile(self) -> dict[str, Any]:
        """Get the default browser profile configuration."""
        db_config = self._get_db_config()
        for profile in db_config.browser_profiles.values():
            if profile.default:
                return profile.model_dump(exclude_none=True)

        # Return first profile if no default
        if db_config.browser_profiles:
            return next(iter(db_config.browser_profiles.values())).model_dump(exclude_none=True)

        return {}

    def get_default_llm(self) -> dict[str, Any]:
        """Get the default LLM configuration."""
        db_config = self._get_db_config()
        for llm in db_config.llms.values():
            if llm.default:
                return llm.model_dump(exclude_none=True)

        # Return first LLM if no default
        if db_config.llms:
            return next(iter(db_config.llms.values())).model_dump(exclude_none=True)

        return {}

    def get_default_agent(self) -> dict[str, Any]:
        """Get the default agent configuration."""
        db_config = self._get_db_config()
        for agent in db_config.agents.values():
            if agent.default:
                return agent.model_dump(exclude_none=True)

        # Return first agent if no default
        if db_config.agents:
            return next(iter(db_config.agents.values())).model_dump(exclude_none=True)

        return {}

    def load_config(self) -> dict[str, Any]:
        """Load configuration with env var overrides."""
        config = {
            'browser_profile': self.get_default_profile(),
            'llm': self.get_default_llm(),
            'agent': self.get_default_agent(),
        }

        # Fresh env config for overrides
        env_config = EnvConfig()

        # Apply MCP-specific env var overrides
        if env_config.OPENBROWSER_HEADLESS is not None:
            config['browser_profile']['headless'] = env_config.OPENBROWSER_HEADLESS

        if env_config.OPENBROWSER_ALLOWED_DOMAINS:
            domains = [d.strip() for d in env_config.OPENBROWSER_ALLOWED_DOMAINS.split(',') if d.strip()]
            config['browser_profile']['allowed_domains'] = domains

        # Proxy settings
        proxy_dict: dict[str, Any] = {}
        if env_config.OPENBROWSER_PROXY_URL:
            proxy_dict['server'] = env_config.OPENBROWSER_PROXY_URL
        if env_config.OPENBROWSER_NO_PROXY:
            proxy_dict['bypass'] = ','.join(
                [d.strip() for d in env_config.OPENBROWSER_NO_PROXY.split(',') if d.strip()]
            )
        if env_config.OPENBROWSER_PROXY_USERNAME:
            proxy_dict['username'] = env_config.OPENBROWSER_PROXY_USERNAME
        if env_config.OPENBROWSER_PROXY_PASSWORD:
            proxy_dict['password'] = env_config.OPENBROWSER_PROXY_PASSWORD
        if proxy_dict:
            config.setdefault('browser_profile', {})
            config['browser_profile']['proxy'] = proxy_dict

        # LLM API key overrides
        if env_config.OPENAI_API_KEY:
            config['llm']['api_key'] = env_config.OPENAI_API_KEY

        if env_config.OPENBROWSER_LLM_MODEL:
            config['llm']['model'] = env_config.OPENBROWSER_LLM_MODEL

        return config


# Create singleton instance
CONFIG = Config()


# Helper functions
def load_openbrowser_config() -> dict[str, Any]:
    """Load openbrowser configuration."""
    return CONFIG.load_config()


def get_default_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Get default browser profile from config dict."""
    return config.get('browser_profile', {})


def get_default_llm(config: dict[str, Any]) -> dict[str, Any]:
    """Get default LLM config from config dict."""
    return config.get('llm', {})


def get_api_key_for_provider(provider: str) -> str:
    """Get API key for a specific LLM provider."""
    provider_lower = provider.lower()
    
    if provider_lower == 'openai':
        return CONFIG.OPENAI_API_KEY
    elif provider_lower in ('anthropic', 'claude'):
        return CONFIG.ANTHROPIC_API_KEY
    elif provider_lower == 'google':
        return CONFIG.GOOGLE_API_KEY
    elif provider_lower == 'groq':
        return CONFIG.GROQ_API_KEY
    elif provider_lower == 'deepseek':
        return CONFIG.DEEPSEEK_API_KEY
    elif provider_lower == 'openrouter':
        return CONFIG.OPENROUTER_API_KEY
    elif provider_lower in ('azure', 'azure_openai'):
        return CONFIG.AZURE_OPENAI_KEY
    elif provider_lower == 'oci':
        return CONFIG.OCI_API_KEY
    elif provider_lower == 'cerebras':
        return CONFIG.CEREBRAS_API_KEY
    else:
        return ''

