"""
Application Settings Management

Uses pydantic-settings for type-safe configuration with environment variable support.
All settings can be overridden via environment variables or .env file.
"""

from typing import Optional, List
from pathlib import Path
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables.
    Loads from .env file in project root by default.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',  # Ignore extra fields from .env
        validate_assignment=True,
        use_enum_values=True
    )

    # Server Configuration
    port: int = Field(default=8084, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    secret_key: SecretStr = Field(
        default="change-me-in-production",
        description="Flask secret key for sessions"
    )

    # Model Configuration
    embedding_model: str = Field(
        default="mistral",
        description="Ollama embedding model name"
    )
    llm_model: str = Field(
        default="sixthwood",
        description="Ollama LLM model name"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation"
    )

    # Output Configuration
    max_output_tokens: int = Field(
        default=16384,
        ge=1,
        description="Maximum output tokens"
    )
    repeat_penalty: float = Field(
        default=1.1,
        ge=0.0,
        description="Repeat penalty for generation"
    )
    top_k: int = Field(default=40, ge=1, description="Top-K sampling")
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-P sampling"
    )

    # Vector Database Configuration
    chroma_host: str = Field(default="localhost", description="Chroma server host")
    chroma_port: int = Field(default=8000, description="Chroma server port")
    default_collection: str = Field(
        default="langchain",
        description="Default collection name"
    )

    # Document Processing
    chunk_size: int = Field(
        default=7500,
        ge=100,
        description="Document chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Chunk overlap in characters"
    )

    # Retrieval Configuration
    default_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default number of documents to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval"
    )

    # ConPort Configuration
    conport_workspace_id: Optional[str] = Field(
        default=None,
        description="ConPort workspace ID (usually project root path)"
    )

    # Feedback Configuration
    positive_feedback_threshold: int = Field(
        default=4,
        ge=1,
        le=5,
        description="Rating threshold for positive feedback"
    )
    negative_feedback_threshold: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Rating threshold for negative feedback"
    )

    # Training Configuration
    min_positive_pairs: int = Field(
        default=50,
        ge=1,
        description="Minimum positive training pairs required"
    )
    min_negative_pairs: int = Field(
        default=50,
        ge=1,
        description="Minimum negative training pairs required"
    )
    hard_negative_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for hard negatives"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="json",
        description="Log format (json or console)"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Optional log file path"
    )

    # Performance Configuration
    enable_caching: bool = Field(
        default=True,
        description="Enable query result caching"
    )
    cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Cache TTL in seconds"
    )
    max_cache_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum cache entries"
    )

    @field_validator('conport_workspace_id', mode='before')
    @classmethod
    def set_default_workspace_id(cls, v):
        """Set default workspace ID to project root if not provided"""
        if v is None:
            return str(Path.cwd())
        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @field_validator('log_format')
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is valid"""
        valid_formats = ['json', 'console']
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"log_format must be one of {valid_formats}")
        return v_lower

    def get_secret_key_str(self) -> str:
        """Get the secret key as a plain string"""
        return self.secret_key.get_secret_value()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Creates the instance on first call and caches it.
    Useful for dependency injection.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment.

    Useful for testing or when environment changes.
    """
    global _settings
    _settings = Settings()
    return _settings