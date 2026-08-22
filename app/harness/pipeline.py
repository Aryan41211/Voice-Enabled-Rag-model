"""Pipeline orchestration: typing, timeouts, retries, circuit breakers.

The harness wires the stages together and owns all resilience concerns so the
stages themselves stay pure. Every path — happy, refused, clarify, or provider
failure — returns a typed ``QueryResponse`` (never an unhandled 500).

Resilience policy (ARCHITECTURE.md):
* per-stage timeouts via ``asyncio.wait_for``;
* retries with exponential backoff on transient stage failures (cap = config);
* a circuit breaker around the LLM provider that trips after repeated
  failures and degrades to the extractive generator;
* retrieval guardrail refusal short-circuits generation entirely.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.generation.generator import ExtractiveGenerator, make_generator
from app.guardrails.guardrails import GuardrailPipeline
from app.harness.schemas import (
    Answer,
    GenerationError,
    PipelineStageError,
    QueryResponse,
    RetrievalError,
    RetrievalResult,
    Source,
    Transcript,
)
from app.ingestion.embed import Embedder
from app.retrieval.query_expansion import expand_and_retrieve
from app.session.rewriter import rewrite_query

BACKGROUND_RANK = 20
GENERATION_CHUNKS = 5

logger = logging.getLogger("voice-rag")


class _EmbeddingCache:
    """LRU cache for query embeddings to skip redundant encoding in multi-turn."""

    def __init__(self, max_size: int = 128) -> None:
        self._cache: dict[str, Any] = {}
        self._max_size = max_size

    def get(self, query: str) -> Any | None:
        return self._cache.get(query)

    def put(self, query: str, vec: Any) -> None:
        if len(self._cache) >= self._max_size:
            # Evict oldest (dict maintains insertion order in Python 3.7+)
            self._cache.pop(next(iter(self._cache)))
        self._cache[query] = vec


class _RerankedDense:
    """Wraps a dense retriever with adaptive cross-encoder reranking.

    Preserves the original dense score on each chunk so retrieval guardrails
    (which compare against background_score) remain valid.

    Adaptive: skips reranking when the top-1 dense score exceeds a
    high-confidence threshold, only reranking ambiguous cases.
    """

    def __init__(
        self,
        dense_retriever,
        reranker,
        top_n: int = 10,
        adaptive: bool = True,
    ) -> None:
        self._dense = dense_retriever
        self._reranker = reranker
        self._top_n = top_n
        self._adaptive = adaptive

    def search(self, query_vector, k: int = 20, query_text: str = "") -> list:
        candidates = self._dense.search(query_vector, k=k, query_text=query_text)
        if not candidates or not query_text:
            return candidates
        if self._adaptive and not self._reranker.should_rerank(candidates):
            return candidates[: self._top_n]
        reranked = self._reranker.rerank(query_text, candidates, top_n=self._top_n)
        return reranked


class CircuitBreaker:
    """Trips after ``threshold`` consecutive failures; stays open for ``reset_s``."""

    def __init__(self, threshold: int = 5, reset_s: float = 30.0) -> None:
        self.threshold = threshold
        self.reset_s = reset_s
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_s:
            self._opened_at = None
            self._failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()


class Pipeline:
    def __init__(
        self,
        *,
        embedder: Embedder,
        retriever,
        guardrails: GuardrailPipeline,
        generator=None,
        stt=None,
        dense_retriever=None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.retriever = retriever
        self.guardrails = guardrails
        self.generator = generator or make_generator()
        self.stt = stt
        self._dense_retriever = dense_retriever
        self._generation_breaker = CircuitBreaker(
            self.settings.circuit_breaker_threshold,
            self.settings.circuit_breaker_reset_s,
        )
        self._embedding_cache = _EmbeddingCache()
        self._log_store = None
        self._session_store = None

    @classmethod
    def from_index(
        cls,
        lang: str = "hi",
        strategy: str = "metadata",
        index_dir: str | Path | None = None,
    ) -> "Pipeline":
        from app.retrieval.load import load_chunks, load_dense

        settings = get_settings()
        chunks = load_chunks(lang, strategy, index_dir)
        dense = load_dense(lang, strategy, chunks, index_dir)
        retriever = dense
        if settings.rerank_enabled:
            from app.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name=settings.rerank_model)
            retriever = _RerankedDense(
                dense, reranker, top_n=settings.rerank_candidates,
                adaptive=settings.rerank_adaptive,
            )
        embedder = Embedder()
        guardrails = GuardrailPipeline()
        return cls(
            embedder=embedder,
            retriever=retriever,
            guardrails=guardrails,
            generator=make_generator(),
            dense_retriever=dense,
        )

    def _refuse(
        self,
        reason: str,
        request_id: str,
        timings: dict,
        transcript_text: str = "",
        language: str | None = None,
    ) -> QueryResponse:
        return QueryResponse(
            request_id=request_id,
            transcript=transcript_text,
            transcript_language=language,
            refused=True,
            refusal_reason=reason,
            timings_ms=timings,
        )

    def warmup(self) -> None:
        """Pre-load models so the first live request doesn't pay cold-start."""
        try:
            self.embedder.encode_query("warmup")
        except Exception:
            pass

    async def process_audio(self, audio_path: str | Path) -> QueryResponse:
        request_id = uuid.uuid4().hex[:12]
        timings: dict = {}
        t0 = time.perf_counter()
        if self.stt is None:
            return self._refuse("speech-to-text is not configured", request_id, timings)

        transcript: Transcript | None = None
        last_error: str | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                transcript = await asyncio.wait_for(
                    self.stt.transcribe(audio_path),
                    timeout=self.settings.stt_timeout_s,
                )
                break
            except PipelineStageError as exc:
                last_error = str(exc)
                if not getattr(exc, "retryable", False):
                    break
            except Exception as exc:
                last_error = str(exc)
            await self._backoff(attempt)

        timings["stt_ms"] = (time.perf_counter() - t0) * 1000
        if transcript is None:
            return self._refuse(
                f"could not transcribe audio: {last_error or 'unknown error'}",
                request_id,
                timings,
            )
        return await self.query_async(
            transcript, request_id=request_id, timings=timings
        )

    def query(self, text: str, request_id: str | None = None, session_id: str | None = None) -> QueryResponse:
        """Synchronous convenience wrapper for scripts / tests."""
        return asyncio.run(
            self.query_async(Transcript(text=text), request_id=request_id, session_id=session_id)
        )

    async def query_async(
        self,
        transcript: Transcript,
        request_id: str | None = None,
        timings: dict | None = None,
        session_id: str | None = None,
    ) -> QueryResponse:
        request_id = request_id or uuid.uuid4().hex[:12]
        timings = timings or {}
        sources: list[Source] = []

        # --- Pre-guardrail: STT confidence check ------------------------
        if transcript.confidence < self.settings.stt_min_confidence:
            return QueryResponse(
                request_id=request_id,
                transcript=transcript.text,
                transcript_language=transcript.language,
                refused=True,
                refusal_reason=f"low_stt_confidence:{transcript.confidence:.2f}",
                timings_ms=timings,
            )

        # --- Layer 1: input guardrail -----------------------------------
        conversation_context: list[dict] | None = None
        session_history: list | None = None
        if self._session_store is not None and session_id:
            session = self._session_store.get_or_create(session_id)
            session_history = session.history
            if session_history:
                conversation_context = [
                    {"role": t.role, "text": t.text} for t in session_history
                ]

        t1 = time.perf_counter()
        gr = self.guardrails.check_input(transcript, conversation_context=conversation_context)
        timings["input_guardrail_ms"] = (time.perf_counter() - t1) * 1000
        timings["input_gr_action"] = gr.action
        logger.info(
            "input_guardrail",
            extra={"stage": "input_guardrail", "success": gr.passed, "latency_ms": timings["input_guardrail_ms"]},
        )
        if gr.action != "proceed":
            return self._refuse(
                gr.reason or gr.action,
                request_id,
                timings,
                transcript_text=transcript.text,
                language=transcript.language,
            )

        # --- Query rewriting for multi-turn --------------------------------
        query_text = transcript.text
        if session_history:
            query_text = rewrite_query(transcript.text, session_history)

        # --- Retrieval ---------------------------------------------------
        t2 = time.perf_counter()
        try:
            hits = await asyncio.wait_for(
                self._retrieve(query_text),
                timeout=self.settings.retrieval_timeout_s,
            )
        except (RetrievalError, asyncio.TimeoutError) as exc:
            return self._refuse(
                f"retrieval failed: {exc}",
                request_id,
                timings,
                transcript_text=transcript.text,
                language=transcript.language,
            )
        background = (
            hits[BACKGROUND_RANK - 1].score
            if len(hits) >= BACKGROUND_RANK
            else (hits[-1].score if hits else None)
        )
        retrieval = RetrievalResult(
            query=query_text,
            chunks=hits[:GENERATION_CHUNKS],
            retrieval_latency_ms=(time.perf_counter() - t2) * 1000,
            background_score=background,
        )
        timings["retrieval_ms"] = retrieval.retrieval_latency_ms
        sources = [
            Source(
                chunk_id=c.chunk_id,
                passage=c.text,
                score=c.score,
                strategy=c.strategy,
            )
            for c in retrieval.chunks
        ]
        logger.info(
            "retrieval",
            extra={"stage": "retrieval", "success": True, "latency_ms": timings["retrieval_ms"]},
        )

        # --- Layer 2: retrieval guardrail -------------------------------
        t3 = time.perf_counter()
        gr = self.guardrails.check_retrieval(retrieval)
        timings["retrieval_guardrail_ms"] = (time.perf_counter() - t3) * 1000
        timings["retrieval_gr_action"] = gr.action
        if gr.action != "proceed":
            return QueryResponse(
                request_id=request_id,
                transcript=transcript.text,
                transcript_language=transcript.language,
                refused=True,
                refusal_reason=gr.reason or gr.action,
                sources=sources,
                timings_ms=timings,
            )

        # --- Generation (retries + circuit breaker + fallback) ----------
        t4 = time.perf_counter()
        answer = await self._generate_with_resilience(query_text, retrieval.chunks)
        timings["generation_ms"] = (time.perf_counter() - t4) * 1000
        timings["ttft_ms"] = answer.ttft_ms
        logger.info(
            "generation",
            extra={"stage": "generation", "success": True, "latency_ms": timings["generation_ms"], "ttft_ms": timings.get("ttft_ms")},
        )

        # --- Layer 3: output guardrail ----------------------------------
        t5 = time.perf_counter()
        gr, answer = self.guardrails.check_output(
            query_text, retrieval.chunks, answer
        )
        timings["output_guardrail_ms"] = (time.perf_counter() - t5) * 1000
        timings["output_gr_action"] = gr.action
        logger.info(
            "output_guardrail",
            extra={"stage": "output_guardrail", "success": gr.passed, "latency_ms": timings["output_guardrail_ms"]},
        )
        if gr.action != "proceed":
            # Fail closed: fall back to the top passage verbatim instead of
            # surfacing an ungrounded or uncited answer.
            answer = ExtractiveGenerator().generate(query_text, retrieval.chunks)

        timings["total_ms"] = sum(v for v in timings.values() if isinstance(v, (int, float)))

        resp = QueryResponse(
            request_id=request_id,
            transcript=transcript.text,
            transcript_language=transcript.language,
            answer=answer.text,
            refused=False,
            sources=sources,
            timings_ms=timings,
        )
        logger.info(
            "request_complete",
            extra={"stage": "total", "success": not resp.refused, "latency_ms": timings["total_ms"]},
        )

        if self._log_store is not None:
            from app.observability.store import RequestLogEntry

            self._log_store.log_request(RequestLogEntry(
                request_id=request_id,
                timestamp=time.time(),
                transcript=transcript.text,
                language=transcript.language,
                answer=resp.answer,
                refused=resp.refused,
                refusal_reason=resp.refusal_reason,
                chunk_ids=[s.chunk_id for s in resp.sources],
                guardrail_input=timings.get("input_gr_action", "proceed"),
                guardrail_retrieval=timings.get("retrieval_gr_action", "proceed"),
                guardrail_output=timings.get("output_gr_action", "proceed"),
                stt_latency_ms=timings.get("stt_ms", 0),
                retrieval_latency_ms=timings.get("retrieval_ms", 0),
                generation_latency_ms=timings.get("generation_ms", 0),
                total_latency_ms=timings.get("total_ms", 0),
                top_retrieval_score=timings.get("top_score"),
            ))

        if self._session_store is not None and session_id:
            from app.harness.schemas import ConversationTurn

            session = self._session_store.get_or_create(session_id)
            session.add_turn(ConversationTurn(
                role="user", text=transcript.text, timestamp=time.time(),
                chunks_used=[s.chunk_id for s in resp.sources],
            ))
            if resp.answer:
                session.add_turn(ConversationTurn(
                    role="assistant", text=resp.answer, timestamp=time.time(),
                ))

        return resp

    async def _retrieve(self, query: str) -> list:
        if self.settings.query_expansion_enabled and self._dense_retriever is not None:
            return await asyncio.to_thread(
                expand_and_retrieve,
                query, self.embedder, self._dense_retriever,
                k=BACKGROUND_RANK,
                expansion_k=self.settings.expansion_k,
                max_paraphrases=self.settings.max_paraphrases,
            )
        cached = self._embedding_cache.get(query)
        if cached is not None:
            qv = cached
        else:
            qv = self.embedder.encode_query(query)
            self._embedding_cache.put(query, qv)
        hits = await asyncio.to_thread(
            self.retriever.search, qv, k=BACKGROUND_RANK, query_text=query
        )
        return hits

    async def _generate_with_resilience(self, query: str, chunks: list) -> Answer:
        """Run generation with retries; degrade to extractive on hard failure."""
        if isinstance(self.generator, ExtractiveGenerator):
            return self.generator.generate(query, chunks)

        if self._generation_breaker.is_open:
            return ExtractiveGenerator().generate(query, chunks)

        for attempt in range(self.settings.max_retries + 1):
            try:
                answer = await asyncio.wait_for(
                    self.generator.generate(query, chunks),
                    timeout=self.settings.llm_timeout_s,
                )
                self._generation_breaker.record_success()
                return answer
            except GenerationError as exc:
                self._generation_breaker.record_failure()
                if not exc.retryable:
                    break
            except Exception as exc:
                self._generation_breaker.record_failure()
                print(f"[pipeline] unexpected generation error: {exc}")
            await self._backoff(attempt)

        return ExtractiveGenerator().generate(query, chunks)

    @staticmethod
    async def _backoff(attempt: int) -> None:
        if attempt > 0:
            delay = get_settings().retry_base_delay_s * (2 ** (attempt - 1))
            await asyncio.sleep(delay)
