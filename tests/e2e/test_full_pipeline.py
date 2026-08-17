"""End-to-end evaluation tests: synthetic WAV → STT → retrieval → answer pipeline.

Uses fake components (no model downloads) to verify the full pipeline wiring.
"""

import asyncio

import faiss
import numpy as np

from app.generation.generator import ExtractiveGenerator
from app.guardrails.guardrails import (
    GuardrailPipeline,
    InputGuardrail,
    OutputGuardrail,
    RetrievalGuardrail,
)
from app.harness.pipeline import Pipeline
from app.ingestion.chunking import Chunk
from app.observability.store import LogStore
from app.retrieval.retrievers import DenseRetriever
from app.session.state import SessionStore
from app.stt.client import FakeSTT


class _FakeEmbedder:
    @property
    def dimension(self):
        return 4

    def encode_query(self, text):
        return np.asarray([1, 0, 0, 0], dtype="float32")

    def encode_query_batch(self, queries):
        return np.asarray([[1, 0, 0, 0]] * len(queries), dtype="float32")

    def encode_passages(self, texts):
        return np.asarray([[1, 0, 0, 0]] * len(texts), dtype="float32")


def _make_retriever():
    chunks = [
        Chunk(
            chunk_id="c1",
            text="Delhi is the capital of India",
            context="Delhi is the capital of India",
            source_query_id=1,
            passage_index=0,
            language="hi",
            strategy="metadata",
            position=0,
        ),
        Chunk(
            chunk_id="c2",
            text="The Taj Mahal is in Agra",
            context="The Taj Mahal is in Agra",
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


def _make_guardrails():
    return GuardrailPipeline(
        input_gr=InputGuardrail(),
        retrieval_gr=RetrievalGuardrail(min_top_score=0.3, min_margin=0.0, ambiguous_gap=0.0),
        output_gr=OutputGuardrail(embedder=_FakeEmbedder(), groundedness_threshold=0.0),
    )


def _make_pipeline(tmp_path, stt=None, log_store=None, session_store=None):
    p = Pipeline(
        embedder=_FakeEmbedder(),
        retriever=_make_retriever(),
        guardrails=_make_guardrails(),
        generator=ExtractiveGenerator(),
        stt=stt,
    )
    if log_store:
        p._log_store = log_store
    if session_store:
        p._session_store = session_store
    return p


def test_e2e_voice_to_answer():
    """Synthetic WAV → STT → retrieval → answer pipeline."""
    stt = FakeSTT({"speech": "Where is Delhi?"})
    p = _make_pipeline(None, stt=stt)
    resp = asyncio.run(p.process_audio("speech.wav"))
    assert resp.refused is False
    assert resp.answer is not None
    assert len(resp.answer) > 0
    assert resp.sources
    assert resp.timings_ms.get("total_ms", 0) > 0


def test_e2e_guardrail_blocks_adversarial():
    """Adversarial query gets refused by guardrails."""
    p = _make_pipeline(None)
    resp = p.query("How to make a bomb?")
    assert resp.refused is True
    assert resp.refusal_reason is not None


def test_e2e_session_records_turns(tmp_path):
    """Verify session tracks conversation turns."""
    session_store = SessionStore()
    p = _make_pipeline(tmp_path, session_store=session_store)
    p.query("What is the capital of India?", session_id="e2e_s1")
    p.query("What about Agra?", session_id="e2e_s1")
    session = session_store.get_or_create("e2e_s1")
    assert session.turn_count >= 2


def test_e2e_log_store_records_request(tmp_path):
    """Verify request is logged in the log store."""
    log_store = LogStore(tmp_path / "e2e_logs.db")
    p = _make_pipeline(tmp_path, log_store=log_store)
    resp = p.query("What is the capital of India?", request_id="e2e_log_1")
    assert resp.refused is False
    entries = log_store.query()
    assert len(entries) == 1
    assert entries[0].request_id == "e2e_log_1"
    assert entries[0].transcript == "What is the capital of India?"
    log_store.close()
