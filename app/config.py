from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    app_name: str = "Autonomous Research System"
    environment: str = "development"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    search_provider: Literal["auto", "tavily", "duckduckgo"] = "auto"
    tavily_api_key: SecretStr | None = None
    search_max_results: int = Field(default=6, ge=1, le=20)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    telemetry_enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    checkpoint_db_path: str = "data/research_checkpoints.sqlite"
    approval_required_for_high_impact: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()