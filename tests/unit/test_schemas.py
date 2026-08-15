from app.harness.schemas import (
    Answer,
    GuardrailResult,
    PipelineStageError,
    QueryResponse,
    RetrievedChunk,
    RetrievalResult,
    STTError,
    Transcript,
)


def test_transcript_valid():
    t = Transcript(text="क्या है", language="hi", confidence=0.9)
    assert t.is_final is True
    assert t.stt_latency_ms == 0.0


def test_transcript_rejects_out_of_range_confidence():
    try:
        Transcript(text="x", confidence=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected validation error")


def test_retrieved_chunk_defaults():
    c = RetrievedChunk(chunk_id="c1", text="text")
    assert c.source == "hybrid"
    assert c.score == 0.0
    assert c.metadata == {}


def test_retrieval_result_builds():
    r = RetrievalResult(
        query="q",
        chunks=[RetrievedChunk(chunk_id="c1", text="t")],
        retrieval_latency_ms=5.0,
    )
    assert len(r.chunks) == 1
    assert r.retrieval_latency_ms == 5.0


def test_answer_grounded_default_none():
    a = Answer(text="answer")
    assert a.grounded is None
    assert a.cited_chunk_ids == []


def test_guardrail_result_actions():
    assert GuardrailResult(passed=True, layer="input").action == "proceed"
    r = GuardrailResult(passed=False, layer="input", action="refuse", reason="x")
    assert r.action == "refuse"


def test_query_response_roundtrip():
    q = QueryResponse(
        request_id="rid",
        transcript="t",
        answer="a",
        sources=[{"chunk_id": "c", "passage": "p", "score": 0.9, "strategy": "hybrid"}],
        timings_ms={"total": 1},
    )
    assert q.refused is False
    assert q.schema_version == "1.0"
    assert q.sources[0].strategy == "hybrid"


def test_pipeline_stage_error_fields():
    err = PipelineStageError(stage="stt", detail="boom", retryable=True)
    assert err.stage == "stt"
    assert err.retryable is True
    assert str(err) == "[stt] boom"


def test_stt_error_subclass():
    err = STTError("timeout")
    assert err.stage == "stt"
    assert err.retryable is False


def test_stt_error_retryable():
    err = STTError("timeout", retryable=True)
    assert err.retryable is True
