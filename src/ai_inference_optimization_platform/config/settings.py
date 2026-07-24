from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    app_name: str = "AI Inference Optimization Platform"
    app_version: str = "0.1.0"

    host: str = "127.0.0.1"
    port: int = 8000

    debug: bool = True

    log_level: str = "INFO"

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    redis_url: str = "redis://localhost:6379"

    cache_ttl: int = 3600

    default_model: str = "qwen2.5:3b"
    default_provider: str = "ollama"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """
    return Settings()


settings = get_settings()