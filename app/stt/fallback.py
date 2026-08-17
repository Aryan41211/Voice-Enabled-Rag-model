"""Fallback STT provider chain with circuit breaker routing."""
from __future__ import annotations

import logging
from typing import Any, Protocol

from app.harness.schemas import STTError, Transcript

logger = logging.getLogger(__name__)


class STTProvider(Protocol):
    async def transcribe(self, audio_path: str | Any) -> Transcript: ...


class FallbackSTTChain:
    def __init__(self, providers: list, circuit_breakers: list[Any | None] | None = None) -> None:
        self._providers = providers
        self._breakers = circuit_breakers or [None] * len(providers)

    async def transcribe(self, audio_path: str) -> Transcript:
        last_error: STTError | None = None
        for provider, breaker in zip(self._providers, self._breakers):
            if breaker is not None and breaker.is_open:
                continue
            try:
                result = await provider.transcribe(audio_path)
                if breaker is not None:
                    breaker.record_success()
                return result
            except STTError as e:
                last_error = e
                if breaker is not None:
                    breaker.record_failure()
                continue
        raise last_error or STTError("all STT providers failed")
