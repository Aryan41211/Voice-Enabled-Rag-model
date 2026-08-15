"""Application configuration loaded from environment / .env.

Settings are read once and reused across modules. Keys come from ``.env``
(see ``.env.example``). No secrets are hardcoded or committed.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Dataset
    data_lang: str = "hi"
    data_split: str = "validation"

    # Embedding / retrieval
    embedding_model: str = "intfloat/multilingual-e5-small"
    device: str = "auto"  # auto | cpu | cuda
    index_dir: str = "./data/index"
    embedding_batch_size: int = 64

    # Generation
    llm_provider: str = "extractive"  # extractive | groq
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_max_tokens: int = 300
    llm_temperature: float = 0.2
    llm_timeout_s: float = 10.0

    # STT
    sarvam_api_key: str = ""
    stt_timeout_s: float = 10.0

    # Retrieval (generous: covers one-time embedder cold load)
    retrieval_timeout_s: float = 30.0

    # Harness
    max_retries: int = 2
    retry_base_delay_s: float = 0.5
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_s: float = 30.0

    # Server
    log_level: str = "INFO"

    @property
    def device_resolved(self) -> str:
        """Resolve ``auto`` to a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
