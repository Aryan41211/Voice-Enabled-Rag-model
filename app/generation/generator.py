"""Generation stage.

Two providers behind one interface:

* ``ExtractiveGenerator`` — offline default. Returns the highest-scoring
  retrieved passage (truncated) verbatim. Zero API keys, zero network, fully
  grounded by construction. This is what runs in the live demo when no LLM key
  is configured.
* ``LLMGenerator`` — optional hosted LLM (OpenAI-compatible, e.g. Groq).
  Streams the response to measure time-to-first-token (TTFT) and is prompted
  to cite passages by index. Failures raise ``GenerationError``; the harness
  falls back to the extractive generator.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from app.config import get_settings
from app.harness.schemas import (
    Answer,
    GenerationError,
    RetrievedChunk,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant for a voice search system over an Indian "
    "language dataset. Answer ONLY using the numbered passages below. "
    "Answer in the same language as the question. Keep the answer to 2-3 "
    "sentences. If the passages do not contain the answer, say you could not "
    "find the answer. End your reply with a line in the exact format: "
    "Cited: [comma separated passage numbers used]"
)

CITED_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]\s*$")

DEFAULT_MAX_ANSWER_CHARS = 800


def _truncate(text: str, max_chars: int = DEFAULT_MAX_ANSWER_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_chars: int = DEFAULT_MAX_ANSWER_CHARS) -> None:
        self.max_chars = max_chars

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        t0 = time.perf_counter()
        if not chunks:
            raise GenerationError("no retrieved chunks to extract from")
        best = max(chunks, key=lambda c: c.score)
        text = _truncate(best.text, self.max_chars)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return Answer(
            text=text,
            cited_chunk_ids=[best.chunk_id],
            ttft_ms=elapsed_ms,
            full_generation_ms=elapsed_ms,
            grounded=True,
        )


class LLMGenerator:
    name = "llm"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model or "llama-3.3-70b-versatile"
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        self.timeout_s = timeout_s or settings.llm_timeout_s

    @property
    def _chat_url(self) -> str:
        if self.base_url:
            return f"{self.base_url}/chat/completions"
        return "https://api.groq.com/openai/v1/chat/completions"

    @staticmethod
    def _messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
        numbered = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
        user_prompt = (
            f"Question: {query}\n\nPassages:\n{numbered}\n\n"
            "Answer the question using only the passages above."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _parse_citations(text: str, chunks: list[RetrievedChunk]) -> list[str]:
        match = CITED_RE.search(text)
        if not match:
            return [chunks[0].chunk_id] if chunks else []
        indices = [int(i) for i in re.split(r"\s*,\s*", match.group(1)) if i]
        return [chunks[i - 1].chunk_id for i in indices if 0 < i <= len(chunks)]

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        if not self.api_key:
            raise GenerationError("LLM provider selected but no API key configured")
        if not chunks:
            raise GenerationError("no retrieved chunks to generate from")

        payload = {
            "model": self.model,
            "messages": self._messages(query, chunks),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        t_start = time.perf_counter()
        ttft_ms: float | None = None
        parts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream(
                    "POST", self._chat_url, json=payload, headers=headers
                ) as resp:
                    if resp.status_code >= 400:
                        raise GenerationError(
                            f"LLM HTTP {resp.status_code}: {await resp.aread()}"
                        )
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        )
                        if not delta:
                            continue
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        parts.append(delta)
        except httpx.HTTPError as exc:
            raise GenerationError(f"LLM request failed: {exc}", retryable=True)

        text = "".join(parts).strip()
        if not text:
            raise GenerationError("LLM returned an empty response", retryable=True)

        full_ms = (time.perf_counter() - t_start) * 1000
        return Answer(
            text=text,
            cited_chunk_ids=self._parse_citations(text, chunks),
            ttft_ms=ttft_ms or full_ms,
            full_generation_ms=full_ms,
            grounded=None,  # output guardrail verifies grounding
        )


def make_generator() -> ExtractiveGenerator | LLMGenerator:
    """Return the generator selected by ``LLM_PROVIDER`` config."""
    provider = get_settings().llm_provider
    if provider == "llm":
        return LLMGenerator()
    return ExtractiveGenerator()
