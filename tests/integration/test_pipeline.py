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
from app.ingestion.chunking import Chunk
from app.retrieval.retrievers import DenseRetriever
from app.stt.client import FakeSTT


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
