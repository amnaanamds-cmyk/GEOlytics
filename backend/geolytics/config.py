"""Application settings, loaded from environment / .env.

Every knob that affects an experiment lives here so that a run can record the
exact configuration it executed under (see `evaluation.harness.RunConfig`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEOLYTICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # Storage
    postgres_dsn: str = "postgresql+psycopg://geolytics:geolytics@localhost:5432/geolytics"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    redis_url: str = "redis://localhost:6379/0"

    # Embeddings
    embedding_backend: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Crawler
    crawl_user_agent: str = "GEOlyticsBot/0.1 (+https://github.com/amnaanamds-cmyk/GEOlytics)"
    crawl_max_pages: int = 100
    crawl_delay_seconds: float = 1.0
    crawl_respect_robots: bool = True
    crawl_timeout_seconds: float = 30.0

    # LLM
    llm_backend: Literal["ollama", "none"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
