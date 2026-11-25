"""Application configuration settings."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    ENVIRONMENT: str = "development"
    APP_NAME: str = "Voice Splitwise Backend"
    DEBUG: bool = True
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8081", "*"]
    )

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./voicesplitwise.db"
    DATABASE_URL_SYNC: str = "sqlite:///./voicesplitwise.db"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        """Ensure PostgreSQL connection strings use the asyncpg driver on Render/Heroku/AWS."""
        if not isinstance(v, str):
            return v
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # AI Configuration
    GOOGLE_GENAI_USE_VERTEXAI: bool = True
    GOOGLE_CLOUD_PROJECT: str = "project-1b52589d-0827-46ab-9be"
    GOOGLE_CLOUD_LOCATION: str = "global"
    GEMINI_API_KEY: str = ""

    # Single-pass Multimodal Gemini Model (with STAGE1/STAGE2 aliases for backward compatibility)
    GEMINI_MODEL: str = "gemini-4-hydrogen"
    STAGE1_MODEL: str = "gemini-4-hydrogen"
    STAGE2_MODEL: str = "gemini-4-hydrogen"

    # Default Mock Auth (non-production only)
    DEFAULT_USER_EMAIL: str = "testuser@example.com"
    DEFAULT_USER_NAME: str = "Test User"

    # Firebase Auth Configuration
    FIREBASE_CREDENTIALS_PATH: str | None = None
    FIREBASE_CREDENTIALS_JSON: str | None = None
    FIREBASE_CREDENTIALS_BASE64: str | None = None
    FIREBASE_PROJECT_ID: str | None = None


settings = Settings()
