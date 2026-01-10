"""Core configuration settings."""

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Allow extra env vars like API keys
    )

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # CORS settings
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "https://openbrowser.me", "https://www.openbrowser.me"],
        description="Allowed CORS origins"
    )
    
    # OpenBrowser settings
    OPENBROWSER_DATA_DIR: Path = Field(
        default=Path.home() / ".openbrowser",
        description="Directory for OpenBrowser data"
    )
    
    # Agent settings
    DEFAULT_MAX_STEPS: int = Field(default=50, description="Default max steps for agent")
    DEFAULT_AGENT_TYPE: Literal["browser", "code"] = Field(
        default="code",
        description="Default agent type (browser=Agent, code=CodeAgent)"
    )
    DEFAULT_LLM_MODEL: str = Field(
        default="gemini-3-flash-preview",
        description="Default LLM model to use for agents"
    )
    
    # Redis settings (optional, for session persistence)
    REDIS_URL: str | None = Field(default=None, description="Redis URL for session storage")
    
    # Rate limiting
    MAX_CONCURRENT_AGENTS: int = Field(default=10, description="Max concurrent agent sessions")
    
    # LLM API Keys (optional - can be set via environment)
    GOOGLE_API_KEY: str | None = Field(default=None, description="Google API key for Gemini")
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: str | None = Field(default=None, description="Anthropic API key")


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

