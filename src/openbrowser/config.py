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
    """Detect if we are running in a Docker container.
    
    Uses multiple detection strategies to determine if the current process
    is running inside a Docker container. This information is used for
    optimizing Chrome launch flags (dev shm usage, GPU settings, etc.).
    
    Detection strategies (in order):
        1. Check for /.dockerenv file existence
        2. Check /proc/1/cgroup for 'docker' string
        3. Check if PID 1 is python/uvicorn/uv (using psutil)
        4. Check if total process count < 10 (using psutil)
    
    Returns:
        True if running in Docker, False otherwise.
        
    Note:
        The psutil-based checks only run if psutil is installed.
        Results are cached using functools.cache for performance.
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
    """Environment variable configuration using pydantic-settings.
    
    This class reads configuration from environment variables and .env files,
    providing type-safe access to all configuration options. It supports
    automatic type conversion and validation.
    
    Attributes:
        OPENBROWSER_LOGGING_LEVEL: Logging level (default: 'info').
        CDP_LOGGING_LEVEL: Chrome DevTools Protocol logging level (default: 'WARNING').
        ANONYMIZED_TELEMETRY: Enable anonymized telemetry (default: True).
        XDG_CACHE_HOME: Cache directory path (default: '~/.cache').
        XDG_CONFIG_HOME: Config directory path (default: '~/.config').
        OPENBROWSER_CONFIG_DIR: Override config directory location.
        OPENAI_API_KEY: OpenAI API key.
        ANTHROPIC_API_KEY: Anthropic API key.
        GOOGLE_API_KEY: Google API key.
        GROQ_API_KEY: Groq API key.
        IN_DOCKER: Override Docker detection.
        
    Example:
        >>> config = EnvConfig()
        >>> print(config.OPENAI_API_KEY)
    """

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
    """Database-style entry with UUID and metadata.
    
    Base class for configuration entries that need unique identification
    and tracking. Each entry gets a UUID and creation timestamp.
    
    Attributes:
        id: Unique identifier (auto-generated UUIDv4).
        default: Whether this is the default entry of its type.
        created_at: ISO format timestamp of creation.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    default: bool = Field(default=False)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BrowserProfileEntry(DBStyleEntry):
    """Browser profile configuration entry.
    
    Stores configuration for a browser profile including window settings,
    security options, and proxy configuration.
    
    Attributes:
        headless: Run browser without visible window.
        user_data_dir: Path to browser user data directory.
        allowed_domains: List of domains the browser can access.
        downloads_path: Directory for downloaded files.
        disable_security: Disable browser security features (not recommended).
        window_width: Browser window width in pixels.
        window_height: Browser window height in pixels.
        proxy: Proxy server configuration dict.
    """

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
    """LLM configuration entry.
    
    Stores configuration for a language model including provider details,
    authentication, and model parameters.
    
    Attributes:
        provider: LLM provider name (openai, anthropic, google, etc.).
        api_key: API key for authentication.
        model: Model name/identifier.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in response.
        base_url: Custom API endpoint URL.
    """

    provider: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    base_url: str | None = None


class AgentEntry(DBStyleEntry):
    """Agent configuration entry.
    
    Stores configuration for a browser automation agent including
    behavior settings and references to other configuration entries.
    
    Attributes:
        max_steps: Maximum steps before agent stops.
        use_vision: Enable screenshot analysis.
        system_prompt: Custom system prompt for the agent.
        llm: Reference to LLM entry ID.
        browser_profile: Reference to browser profile entry ID.
    """

    max_steps: int | None = None
    use_vision: bool | None = None
    system_prompt: str | None = None
    llm: str | None = None  # Reference to LLM entry ID
    browser_profile: str | None = None  # Reference to browser profile entry ID


class ConfigJSON(BaseModel):
    """Configuration file format.
    
    Represents the structure of the config.json file that stores
    all persistent configuration for OpenBrowser.
    
    Attributes:
        browser_profiles: Dict of browser profile entries keyed by ID.
        llms: Dict of LLM configuration entries keyed by ID.
        agents: Dict of agent configuration entries keyed by ID.
        
    Example:
        >>> config = ConfigJSON()
        >>> config.browser_profiles['uuid'] = BrowserProfileEntry(headless=True)
    """

    browser_profiles: dict[str, BrowserProfileEntry] = Field(default_factory=dict)
    llms: dict[str, LLMEntry] = Field(default_factory=dict)
    agents: dict[str, AgentEntry] = Field(default_factory=dict)


def create_default_config() -> ConfigJSON:
    """Create a fresh default configuration.
    
    Generates a new ConfigJSON with default browser profile, LLM, and agent
    entries. Each entry is assigned a new UUID and the default flag is set.
    
    Returns:
        A ConfigJSON instance with default entries for:
            - Browser profile: headless=False, no user data dir
            - LLM: OpenAI provider with gpt-4o-mini model
            - Agent: References the default LLM and browser profile
            
    Example:
        >>> config = create_default_config()
        >>> print(len(config.browser_profiles))  # 1
    """
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
    """Load config.json or create fresh one if old format detected.
    
    Attempts to load the configuration file at the given path. If the file
    doesn't exist, uses an old format, or is corrupted, a fresh default
    configuration is created and saved.
    
    Args:
        config_path: Path to the config.json file.
        
    Returns:
        A valid ConfigJSON instance, either loaded from file or freshly created.
        
    Note:
        This function handles migration from older config formats by detecting
        the format and replacing with a new default configuration if needed.
    """
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
    
    Singleton class that provides unified access to all configuration
    settings. Re-reads environment variables on every property access
    for flexibility in dynamic environments.
    
    This class:
        - Provides type-safe access to environment variables
        - Manages configuration file loading and caching
        - Creates necessary directories on first access
        - Merges file-based and environment-based settings
        
    Attributes:
        LOGGING_LEVEL: Current logging level from environment.
        CONFIG_DIR: Path to configuration directory.
        CONFIG_FILE: Path to config.json file.
        PROFILES_DIR: Path to browser profiles directory.
        OPENAI_API_KEY: OpenAI API key from environment.
        IN_DOCKER: Whether running in Docker container.
        
    Example:
        >>> config = Config()
        >>> print(config.OPENAI_API_KEY)
        >>> print(config.CONFIG_DIR)
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
        """Create directories if they don't exist (only once).
        
        Creates the configuration directory structure including:
            - Main config directory
            - Profiles subdirectory
            - Extensions subdirectory
            
        This method is idempotent - directories are only created on first call.
        """
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
        """Get config path from env config.
        
        Determines the configuration file path using the following priority:
            1. OPENBROWSER_CONFIG_PATH environment variable
            2. OPENBROWSER_CONFIG_DIR/config.json
            3. XDG_CONFIG_HOME/openbrowser/config.json
            
        Returns:
            Path to the configuration JSON file.
        """
        env_config = EnvConfig()
        if env_config.OPENBROWSER_CONFIG_PATH:
            return Path(env_config.OPENBROWSER_CONFIG_PATH).expanduser()
        elif env_config.OPENBROWSER_CONFIG_DIR:
            return Path(env_config.OPENBROWSER_CONFIG_DIR).expanduser() / 'config.json'
        else:
            xdg_config = Path(env_config.XDG_CONFIG_HOME).expanduser()
            return xdg_config / 'openbrowser' / 'config.json'

    def _get_db_config(self) -> ConfigJSON:
        """Load and migrate config.json.
        
        Loads the configuration file, handling migration from old formats
        and creation of new config if needed.
        
        Returns:
            ConfigJSON instance with all configuration entries.
        """
        config_path = self._get_config_path()
        return load_and_migrate_config(config_path)

    def get_default_profile(self) -> dict[str, Any]:
        """Get the default browser profile configuration.
        
        Retrieves the browser profile marked as default from the config file.
        Falls back to the first profile if no default is set.
        
        Returns:
            Dict with browser profile settings, or empty dict if none exist.
        """
        db_config = self._get_db_config()
        for profile in db_config.browser_profiles.values():
            if profile.default:
                return profile.model_dump(exclude_none=True)

        # Return first profile if no default
        if db_config.browser_profiles:
            return next(iter(db_config.browser_profiles.values())).model_dump(exclude_none=True)

        return {}

    def get_default_llm(self) -> dict[str, Any]:
        """Get the default LLM configuration.
        
        Retrieves the LLM entry marked as default from the config file.
        Falls back to the first LLM entry if no default is set.
        
        Returns:
            Dict with LLM settings (provider, model, etc.), or empty dict if none exist.
        """
        db_config = self._get_db_config()
        for llm in db_config.llms.values():
            if llm.default:
                return llm.model_dump(exclude_none=True)

        # Return first LLM if no default
        if db_config.llms:
            return next(iter(db_config.llms.values())).model_dump(exclude_none=True)

        return {}

    def get_default_agent(self) -> dict[str, Any]:
        """Get the default agent configuration.
        
        Retrieves the agent entry marked as default from the config file.
        Falls back to the first agent entry if no default is set.
        
        Returns:
            Dict with agent settings (max_steps, use_vision, etc.), or empty dict if none exist.
        """
        db_config = self._get_db_config()
        for agent in db_config.agents.values():
            if agent.default:
                return agent.model_dump(exclude_none=True)

        # Return first agent if no default
        if db_config.agents:
            return next(iter(db_config.agents.values())).model_dump(exclude_none=True)

        return {}

    def load_config(self) -> dict[str, Any]:
        """Load configuration with env var overrides.
        
        Loads the complete configuration by merging file-based settings
        with environment variable overrides. Environment variables take
        precedence over file settings.
        
        Returns:
            Dict containing:
                - browser_profile: Browser settings with env overrides
                - llm: LLM settings with API key overrides
                - agent: Agent configuration
                
        Note:
            MCP-specific environment variables (OPENBROWSER_HEADLESS,
            OPENBROWSER_ALLOWED_DOMAINS, etc.) are applied as overrides.
        """
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
    """Load openbrowser configuration.
    
    Convenience function that loads the complete configuration using
    the singleton Config instance.
    
    Returns:
        Dict containing browser_profile, llm, and agent configuration.
        
    Example:
        >>> config = load_openbrowser_config()
        >>> headless = config['browser_profile'].get('headless', False)
    """
    return CONFIG.load_config()


def get_default_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Get default browser profile from config dict.
    
    Extracts the browser profile section from a loaded configuration.
    
    Args:
        config: Configuration dict from load_openbrowser_config().
        
    Returns:
        Browser profile settings dict, or empty dict if not present.
    """
    return config.get('browser_profile', {})


def get_default_llm(config: dict[str, Any]) -> dict[str, Any]:
    """Get default LLM config from config dict.
    
    Extracts the LLM section from a loaded configuration.
    
    Args:
        config: Configuration dict from load_openbrowser_config().
        
    Returns:
        LLM settings dict (provider, model, etc.), or empty dict if not present.
    """
    return config.get('llm', {})


def get_api_key_for_provider(provider: str) -> str:
    """Get API key for a specific LLM provider.
    
    Retrieves the appropriate API key from environment variables
    based on the provider name.
    
    Args:
        provider: LLM provider name (case-insensitive). Supported values:
            - 'openai': Returns OPENAI_API_KEY
            - 'anthropic' or 'claude': Returns ANTHROPIC_API_KEY
            - 'google': Returns GOOGLE_API_KEY
            - 'groq': Returns GROQ_API_KEY
            - 'deepseek': Returns DEEPSEEK_API_KEY
            - 'openrouter': Returns OPENROUTER_API_KEY
            - 'azure' or 'azure_openai': Returns AZURE_OPENAI_KEY
            - 'oci': Returns OCI_API_KEY
            - 'cerebras': Returns CEREBRAS_API_KEY
            
    Returns:
        API key string, or empty string if provider not found or key not set.
        
    Example:
        >>> key = get_api_key_for_provider('openai')
        >>> if key:
        ...     llm = ChatOpenAI(api_key=key)
    """
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

