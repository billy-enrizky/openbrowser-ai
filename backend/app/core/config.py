"""Core configuration settings."""

import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables with override=True to ensure .env values take precedence
load_dotenv(override=True)

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
        default=["http://localhost:3000", "http://localhost:3001", "https://openbrowser.me", "https://www.openbrowser.me"],
        description="Allowed CORS origins"
    )
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
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
    
    # VNC settings
    VNC_ENABLED: bool = Field(default=True, description="Enable VNC browser viewing")
    VNC_PASSWORD: str | None = Field(default=None, description="VNC password (auto-generated if not set)")
    VNC_BASE_DISPLAY: int = Field(default=99, description="Base X11 display number for VNC")
    VNC_BASE_PORT: int = Field(default=5900, description="Base VNC port")
    WEBSOCKIFY_BASE_PORT: int = Field(default=6080, description="Base websockify port for WebSocket-to-VNC bridging")
    VNC_WIDTH: int = Field(default=1920, description="Default VNC display width")
    VNC_HEIGHT: int = Field(default=1080, description="Default VNC display height")
    
    # LLM API Keys (optional - can be set via environment)
    GOOGLE_API_KEY: str | None = Field(default=None, description="Google API key for Gemini")
    GEMINI_API_KEY: str | None = Field(default=None, description="Gemini API key (alias for GOOGLE_API_KEY)")
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: str | None = Field(default=None, description="Anthropic API key")

    # Authentication settings
    AUTH_ENABLED: bool = Field(default=False, description="Enable Cognito authentication")
    COGNITO_REGION: str | None = Field(default=None, description="AWS region of the Cognito User Pool")
    COGNITO_USER_POOL_ID: str | None = Field(default=None, description="Cognito User Pool ID")
    COGNITO_APP_CLIENT_ID: str | None = Field(default=None, description="Cognito App Client ID")
    COGNITO_ISSUER: str | None = Field(
        default=None,
        description="Optional Cognito issuer override. If not set, built from region + user pool id.",
    )
    
    def get_google_api_key(self) -> str | None:
        """Get Google/Gemini API key (supports both GOOGLE_API_KEY and GEMINI_API_KEY)."""
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY

    def get_cognito_issuer(self) -> str | None:
        """Get Cognito issuer URL for JWT verification."""
        if self.COGNITO_ISSUER:
            return self.COGNITO_ISSUER.rstrip("/")

        if self.COGNITO_REGION and self.COGNITO_USER_POOL_ID:
            return f"https://cognito-idp.{self.COGNITO_REGION}.amazonaws.com/{self.COGNITO_USER_POOL_ID}"

        return None
    
    def get_available_providers(self) -> list[str]:
        """Get list of available LLM providers based on configured API keys."""
        providers = []
        if self.get_google_api_key():
            providers.append("google")
        if self.OPENAI_API_KEY:
            providers.append("openai")
        if self.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        return providers
    
    def get_available_models(self) -> list[dict]:
        """Get list of available models based on configured API keys."""
        # Define all models by provider
        google_models = [
            {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash", "provider": "google"},
            {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro", "provider": "google"},
            {"id": "gemini-3-pro-image-preview", "name": "Gemini 3 Pro Image", "provider": "google"},
            {"id": "gemini-2.5-flash-preview-05-20", "name": "Gemini 2.5 Flash", "provider": "google"},
            {"id": "gemini-2.5-pro-preview-05-06", "name": "Gemini 2.5 Pro", "provider": "google"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "google"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "provider": "google"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "google"},
        ]
        
        openai_models = [
            {"id": "gpt-5.2", "name": "GPT-5.2", "provider": "openai"},
            {"id": "gpt-5.2-pro", "name": "GPT-5.2 Pro", "provider": "openai"},
            {"id": "gpt-5-mini", "name": "GPT-5 Mini", "provider": "openai"},
            {"id": "gpt-4.1", "name": "GPT-4.1", "provider": "openai"},
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai"},
            {"id": "o4-mini", "name": "o4 Mini", "provider": "openai"},
        ]
        
        anthropic_models = [
            {"id": "claude-opus-4", "name": "Claude Opus 4", "provider": "anthropic"},
            {"id": "claude-sonnet-4", "name": "Claude Sonnet 4", "provider": "anthropic"},
            {"id": "claude-3-7-sonnet", "name": "Claude 3.7 Sonnet", "provider": "anthropic"},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "provider": "anthropic"},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "provider": "anthropic"},
        ]
        
        # Return only models for providers with configured API keys
        models = []
        if self.get_google_api_key():
            models += google_models
        if self.OPENAI_API_KEY:
            models += openai_models
        if self.ANTHROPIC_API_KEY:
            models += anthropic_models
        
        return models


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
