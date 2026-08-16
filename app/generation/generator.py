"""Generation stage.

Two providers behind one interface:

* ``ExtractiveGenerator`` — offline default. Returns the most
  query-relevant **sentence** from the retrieved passages (token-overlap
  scoring across the top chunks), keeping the answer crisp instead of
  dumping a whole passage verbatim. Zero API keys, zero network, fully
  grounded by construction. This is what runs in the live demo when no LLM
  key is configured.
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

# Query words that carry no retrieval signal for sentence matching. Kept small
# and language-agnostic-ish (Devanagari + English function words).
_QUERY_STOPWORDS = {
    "का",
    "की",
    "के",
    "को",
    "से",
    "में",
    "पर",
    "और",
    "है",
    "हैं",
    "था",
    "थी",
    "क्या",
    "कौन",
    "सा",
    "सी",
    "से",
    "कब",
    "कहाँ",
    "कहां",
    "जब",
    "तब",
    "एक",
    "यह",
    "ये",
    "वह",
    "वे",
    "किस",
    "क्या",
    "कैसे",
    "why",
    "what",
    "who",
    "when",
    "where",
    "how",
    "which",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
}

_SENTENCE_BREAK = re.compile(r"(?<=[।.!?])\s+|(?<=\n)\s*")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the trailing punctuation attached."""
    return [s.strip() for s in _SENTENCE_BREAK.split(text.strip()) if s.strip()]


def _query_tokens(query: str) -> list[str]:
    """Lowercased content tokens of the query (stopwords removed)."""
    tokens = [t for t in re.split(r"[\W_]+", query.lower()) if t]
    return [t for t in tokens if t not in _QUERY_STOPWORDS]


def _truncate(text: str, max_chars: int = DEFAULT_MAX_ANSWER_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


class ExtractiveGenerator:
    name = "extractive"

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_ANSWER_CHARS,
        max_context_chars: int = 140,
    ) -> None:
        self.max_chars = max_chars
        self.max_context_chars = max_context_chars

    @staticmethod
    def _best_sentence(query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
        """Return (answer_text, chunk_id) for the most query-relevant sentence.

        Scores sentences across the top retrieved chunks by query-token
        overlap, so the answer is a crisp sentence rather than a verbatim
        passage dump. Falls back to the top chunk's opening when nothing
        overlaps (still verbatim-grounded).
        """
        tokens = _query_tokens(query)
        best_sentence = ""
        best_chunk_id = ""
        best_matches = -1
        best_score = float("-inf")

        for chunk in chunks:
            for sentence in _split_sentences(chunk.text):
                matches = sum(1 for t in tokens if t in sentence.lower())
                if matches < best_matches:
                    continue
                # Prefer more matches; tie-break by chunk score.
                if matches > best_matches or (
                    matches == best_matches and chunk.score > best_score
                ):
                    best_matches = matches
                    best_sentence = sentence
                    best_chunk_id = chunk.chunk_id
                    best_score = chunk.score

        if not best_sentence or best_matches <= 0:
            top = max(chunks, key=lambda c: c.score)
            opening = _split_sentences(top.text)
            return " ".join(opening[:2]), top.chunk_id
        return best_sentence, best_chunk_id

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        t0 = time.perf_counter()
        if not chunks:
            raise GenerationError("no retrieved chunks to extract from")

        sentence, chunk_id = self._best_sentence(query, chunks)

        # Pull in the next sentence as context when the hit is short, keeping
        # the answer tight (a fact alone reads better with a follow-on clause).
        if len(sentence) < self.max_context_chars:
            source = next((c for c in chunks if c.chunk_id == chunk_id), None)
            sentences = _split_sentences(source.text) if source else []
            for i, s in enumerate(sentences):
                if s == sentence and i + 1 < len(sentences):
                    following = sentences[i + 1]
                    candidate = f"{sentence} {following}"
                    if len(candidate) <= self.max_chars:
                        sentence = candidate
                    break

        text = _truncate(sentence, self.max_chars)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return Answer(
            text=text,
            cited_chunk_ids=[chunk_id],
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
