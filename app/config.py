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
    embedding_model: str = "intfloat/multilingual-e5-base"
    device: str = "auto"  # auto | cpu | cuda
    index_dir: str = "./data/index"
    embedding_batch_size: int = 64

    # Cross-encoder reranking
    rerank_enabled: bool = True
    rerank_candidates: int = 10
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_adaptive: bool = True  # skip rerank when top-1 confidence is high

    # Query expansion
    query_expansion_enabled: bool = False
    expansion_k: int = 15
    max_paraphrases: int = 2

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
    # Total budget for one STT call. Half goes to the realtime WS attempt;
    # the rest is reserved for the REST fallback, which is slower but handles
    # longer files (up to 30 s) and multi-turn audio reliably.
    stt_timeout_s: float = 20.0
    stt_provider: str = "sarvam"  # sarvam | fake (keyless local dev)
    stt_min_confidence: float = 0.4
    # Language sent to Sarvam. "auto" uses Sarvam's native language
    # identification (language_code="unknown", supported on the realtime WS,
    # REST and batch endpoints); each response then reports the detected
    # BCP-47 code + confidence. Override with a LANGUAGE_CODES key ("hi",
    # "bn", ...) or a full BCP-47 code to pin one language.
    stt_language: str = "auto"

    # Index / pipeline serving
    data_strategy: str = "metadata"

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
