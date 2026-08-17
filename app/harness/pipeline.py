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

BACKGROUND_RANK = 20
GENERATION_CHUNKS = 5

logger = logging.getLogger("voice-rag")


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
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.retriever = retriever
        self.guardrails = guardrails
        self.generator = generator or make_generator()
        self.stt = stt
        self._generation_breaker = CircuitBreaker(
            self.settings.circuit_breaker_threshold,
            self.settings.circuit_breaker_reset_s,
        )
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

        chunks = load_chunks(lang, strategy, index_dir)
        dense = load_dense(lang, strategy, chunks, index_dir)
        embedder = Embedder()
        guardrails = GuardrailPipeline()
        return cls(
            embedder=embedder,
            retriever=dense,
            guardrails=guardrails,
            generator=make_generator(),
        )

    def _refuse(self, reason: str, request_id: str, timings: dict) -> QueryResponse:
        return QueryResponse(
            request_id=request_id,
            transcript="",
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

        # --- Layer 1: input guardrail -----------------------------------
        t1 = time.perf_counter()
        gr = self.guardrails.check_input(transcript)
        timings["input_guardrail_ms"] = (time.perf_counter() - t1) * 1000
        timings["input_gr_action"] = gr.action
        logger.info(
            "input_guardrail",
            extra={"stage": "input_guardrail", "success": gr.passed, "latency_ms": timings["input_guardrail_ms"]},
        )
        if gr.action != "proceed":
            return self._refuse(gr.reason or gr.action, request_id, timings)

        # --- Retrieval ---------------------------------------------------
        t2 = time.perf_counter()
        try:
            hits = await asyncio.wait_for(
                self._retrieve(transcript.text),
                timeout=self.settings.retrieval_timeout_s,
            )
        except (RetrievalError, asyncio.TimeoutError) as exc:
            return self._refuse(f"retrieval failed: {exc}", request_id, timings)
        background = (
            hits[BACKGROUND_RANK - 1].score
            if len(hits) >= BACKGROUND_RANK
            else (hits[-1].score if hits else None)
        )
        retrieval = RetrievalResult(
            query=transcript.text,
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
                refused=True,
                refusal_reason=gr.reason or gr.action,
                sources=sources,
                timings_ms=timings,
            )

        # --- Generation (retries + circuit breaker + fallback) ----------
        t4 = time.perf_counter()
        answer = await self._generate_with_resilience(transcript.text, retrieval.chunks)
        timings["generation_ms"] = (time.perf_counter() - t4) * 1000
        timings["ttft_ms"] = answer.ttft_ms
        logger.info(
            "generation",
            extra={"stage": "generation", "success": True, "latency_ms": timings["generation_ms"], "ttft_ms": timings.get("ttft_ms")},
        )

        # --- Layer 3: output guardrail ----------------------------------
        t5 = time.perf_counter()
        gr, answer = self.guardrails.check_output(
            transcript.text, retrieval.chunks, answer
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
            answer = ExtractiveGenerator().generate(transcript.text, retrieval.chunks)

        timings["total_ms"] = sum(v for v in timings.values() if isinstance(v, (int, float)))

        resp = QueryResponse(
            request_id=request_id,
            transcript=transcript.text,
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
        qv = self.embedder.encode_query(query)
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
