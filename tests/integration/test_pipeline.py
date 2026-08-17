"""End-to-end harness tests with an in-memory index and fake components."""

import asyncio

import faiss
import numpy as np

from app.generation.generator import ExtractiveGenerator, LLMGenerator
from app.guardrails.guardrails import (
    GuardrailPipeline,
    InputGuardrail,
    OutputGuardrail,
    RetrievalGuardrail,
)
from app.harness.pipeline import CircuitBreaker, Pipeline
from app.harness.schemas import (
    Answer,
    GenerationError,
    RetrievedChunk,
    RetrievalError,
    STTError,
    Transcript,
)
from app.ingestion.chunking import Chunk
from app.retrieval.retrievers import DenseRetriever
from app.stt.client import FakeSTT
from tests.integration.test_index_load import _write_index


class FakeEmbedder:
    @property
    def dimension(self):
        return 4

    def encode_query(self, text):
        if "बेटिंग" in text:
            return np.asarray([-1, 0, 0, 0], dtype="float32")
        return np.asarray([1, 0, 0, 0], dtype="float32")

    def encode_query_batch(self, queries):
        return np.asarray([[1, 0, 0, 0]] * len(queries), dtype="float32")

    def encode_passages(self, texts):
        return np.asarray([[1, 0, 0, 0]] * len(texts), dtype="float32")


def _dense_retriever():
    chunks = [
        Chunk(
            chunk_id="c1",
            text="दिल्ली भारत की राजधानी है",
            context="दिल्ली भारत की राजधानी है",
            source_query_id=1,
            passage_index=0,
            language="hi",
            strategy="metadata",
            position=0,
        ),
        Chunk(
            chunk_id="c2",
            text="ताजमहल आगरा में है",
            context="ताजमहल आगरा में है",
            source_query_id=2,
            passage_index=0,
            language="hi",
            strategy="metadata",
            position=0,
        ),
    ]
    vecs = np.asarray([[1, 0, 0, 0], [0.9, 0.1, 0, 0]], dtype="float32")
    index = faiss.IndexFlatIP(4)
    index.add(vecs)
    return DenseRetriever(index, chunks)


def _guardrails():
    return GuardrailPipeline(
        input_gr=InputGuardrail(),
        retrieval_gr=RetrievalGuardrail(
            min_top_score=0.3, min_margin=0.0, ambiguous_gap=0.0
        ),
        output_gr=OutputGuardrail(embedder=FakeEmbedder(), groundedness_threshold=0.0),
    )


def _pipeline(stt=None, generator=None, settings=None):
    return Pipeline(
        embedder=FakeEmbedder(),
        retriever=_dense_retriever(),
        guardrails=_guardrails(),
        generator=generator or ExtractiveGenerator(),
        stt=stt,
        settings=settings,
    )


def _settings(**overrides):
    from app.config import Settings

    return Settings(_env_file=None, **overrides)


def test_pipeline_answers_end_to_end():
    p = _pipeline(stt=FakeSTT({"audio_hi": "दिल्ली कहाँ है"}))
    resp = asyncio.run(p.process_audio("audio_hi.wav"))
    assert resp.refused is False
    assert "दिल्ली" in resp.answer
    assert resp.sources
    assert resp.timings_ms.get("total_ms", 0) > 0
    assert resp.timings_ms.get("retrieval_ms") >= 0


def test_pipeline_off_topic_refuses():
    p = _pipeline()
    resp = p.query("बेटिंग के नियम क्या हैं")
    assert resp.refused is True
    assert "off-topic" in resp.refusal_reason


def test_pipeline_retrieval_refusal_path():
    p = _pipeline()

    class WeakRetriever:
        def search(self, query_vec, k=5, query_text=""):
            from app.harness.schemas import RetrievedChunk

            return [
                RetrievedChunk(
                    chunk_id="cx",
                    text="असंबंधित पाठ",
                    score=0.1,
                    source="dense",
                    strategy="metadata",
                )
            ]

    p.retriever = WeakRetriever()
    resp = p.query("कुछ भी")
    assert resp.refused is True
    assert "score" in resp.refusal_reason


def test_pipeline_llm_failure_falls_back_to_extractive():
    p = _pipeline(
        generator=LLMGenerator(api_key=""),
        stt=None,
        settings=_settings(max_retries=0),
    )
    resp = p.query("दिल्ली कहाँ है")
    assert resp.refused is False
    assert resp.answer  # extractive fallback, no 500


def test_pipeline_no_stt_refuses_audio():
    p = _pipeline()
    resp = asyncio.run(p.process_audio("whatever.wav"))
    assert resp.refused is True


def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker(threshold=3, reset_s=60)
    assert not cb.is_open
    cb.record_failure()
    cb.record_failure()
    assert not cb.is_open
    cb.record_failure()
    assert cb.is_open
    cb.record_success()
    assert not cb.is_open


def test_circuit_breaker_half_open_after_reset():
    cb = CircuitBreaker(threshold=1, reset_s=0.2)
    cb.record_failure()
    assert cb.is_open
    import time

    time.sleep(0.3)
    assert not cb.is_open


class FlakySTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, audio_path):
        self.calls += 1
        if self.calls == 1:
            raise STTError("websocket dropped", retryable=True)
        return Transcript(text=self.text)


class FailingSTT:
    async def transcribe(self, audio_path):
        raise STTError("invalid audio", retryable=False)


class FailingRetriever:
    def search(self, query_vec, k=5, query_text=""):
        raise RetrievalError("index file missing")


class FakeLLMGenerator:
    name = "fake-llm"

    def __init__(self, answer: Answer | None = None, exc: Exception | None = None):
        self.answer = answer
        self.exc = exc
        self.calls = 0

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if self.answer is None:
            raise GenerationError("no fake answer configured")
        return self.answer


def test_pipeline_from_index_wiring(tmp_path, monkeypatch):
    _write_index(tmp_path)
    monkeypatch.setattr("app.harness.pipeline.Embedder", FakeEmbedder)
    monkeypatch.setattr("app.harness.pipeline.make_generator", ExtractiveGenerator)
    monkeypatch.setattr("app.guardrails.guardrails.Embedder", FakeEmbedder)
    monkeypatch.setattr(
        "app.harness.pipeline.get_settings",
        lambda: _settings(rerank_enabled=False),
    )

    p = Pipeline.from_index(lang="hi", strategy="metadata", index_dir=tmp_path)
    resp = p.query("दिल्ली कहाँ है")
    assert resp.refused is False
    assert "दिल्ली" in resp.answer
    assert resp.sources and resp.sources[0].chunk_id == "c1"


def test_pipeline_stt_retry_then_succeeds(monkeypatch):
    flaky = FlakySTT("दिल्ली कहाँ है")
    monkeypatch.setattr(
        "app.harness.pipeline.get_settings",
        lambda: _settings(max_retries=2, retry_base_delay_s=0.01),
    )
    p = _pipeline(stt=flaky)
    resp = asyncio.run(p.process_audio("x.wav"))
    assert flaky.calls == 2
    assert resp.refused is False
    assert "दिल्ली" in resp.answer


def test_pipeline_stt_nonretryable_failure_refuses():
    p = _pipeline(stt=FailingSTT())
    resp = asyncio.run(p.process_audio("x.wav"))
    assert resp.refused is True
    assert "transcribe" in resp.refusal_reason


def test_pipeline_retrieval_failure_refuses():
    p = _pipeline()
    p.retriever = FailingRetriever()
    resp = p.query("दिल्ली कहाँ है")
    assert resp.refused is True
    assert "retrieval failed" in resp.refusal_reason


def test_pipeline_generation_unexpected_error_falls_back():
    rude = FakeLLMGenerator(exc=RuntimeError("boom"))
    p = _pipeline(generator=rude, settings=_settings(max_retries=0))
    resp = p.query("दिल्ली कहाँ है")
    assert resp.refused is False
    assert "दिल्ली" in resp.answer
    assert p._generation_breaker._failures == 1


def test_pipeline_breaker_open_skips_generation():
    fake = FakeLLMGenerator(exc=RuntimeError("never reached"))
    p = _pipeline(generator=fake, settings=_settings(circuit_breaker_threshold=2))
    p._generation_breaker.record_failure()
    p._generation_breaker.record_failure()
    assert p._generation_breaker.is_open
    resp = p.query("दिल्ली कहाँ है")
    assert resp.refused is False
    assert "दिल्ली" in resp.answer
    assert fake.calls == 0


def test_pipeline_generation_success_records_success():
    good = FakeLLMGenerator(
        answer=Answer(
            text="दिल्ली दिल का शहर है",
            cited_chunk_ids=["c1"],
            grounded=True,
            ttft_ms=5.0,
        )
    )
    p = _pipeline(generator=good)
    p._generation_breaker.record_failure()
    resp = p.query("दिल्ली कहाँ है")
    assert resp.refused is False
    assert resp.answer == "दिल्ली दिल का शहर है"
    assert p._generation_breaker._failures == 0


def test_pipeline_uncited_answer_falls_back_to_extractive():
    uncited = FakeLLMGenerator(
        answer=Answer(text="बिना उद्धरण", cited_chunk_ids=[], grounded=True)
    )
    p = _pipeline(generator=uncited)
    resp = p.query("दिल्ली कहाँ है")
    assert resp.refused is False
    assert resp.answer == "दिल्ली भारत की राजधानी है"


def test_pipeline_logs_to_store(tmp_path):
    from app.observability.store import LogStore

    store = LogStore(tmp_path / "logs.db")
    p = _pipeline(stt=FakeSTT({"audio_hi": "दिल्ली कहाँ है"}))
    p._log_store = store
    resp = p.query("दिल्ली कहाँ है", request_id="test123")
    assert resp.refused is False
    entries = store.query()
    assert len(entries) == 1
    assert entries[0].request_id == "test123"
    assert entries[0].transcript == "दिल्ली कहाँ है"


def test_pipeline_session_turns_recorded(tmp_path):
    from app.session.state import SessionStore

    session_store = SessionStore()
    p = _pipeline()
    p._session_store = session_store
    p.query("दिल्ली कहाँ है", session_id="s1")
    p.query("वहाँ क्या है", session_id="s1")
    session = session_store.get_or_create("s1")
    assert session.turn_count >= 2
