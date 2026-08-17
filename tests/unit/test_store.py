import time

from app.observability.store import LogStore, RequestLogEntry


def test_log_store_roundtrip(tmp_path):
    db = tmp_path / "logs.db"
    store = LogStore(db)
    entry = RequestLogEntry(
        request_id="abc123",
        timestamp=time.time(),
        transcript="दिल्ली कहाँ है",
        language="hi",
        answer="दिल्ली भारत की राजधानी है",
        refused=False,
        refusal_reason=None,
        chunk_ids=["c1", "c2"],
        guardrail_input="proceed",
        guardrail_retrieval="proceed",
        guardrail_output="proceed",
        stt_latency_ms=150.0,
        retrieval_latency_ms=80.0,
        generation_latency_ms=200.0,
        total_latency_ms=430.0,
        top_retrieval_score=0.85,
        metadata={"source": "test"},
    )
    store.log_request(entry)
    results = store.query()
    assert len(results) == 1
    r = results[0]
    assert r.request_id == "abc123"
    assert r.transcript == "दिल्ली कहाँ है"
    assert r.language == "hi"
    assert r.answer == "दिल्ली भारत की राजधानी है"
    assert r.refused is False
    assert r.refusal_reason is None
    assert r.chunk_ids == ["c1", "c2"]
    assert r.guardrail_input == "proceed"
    assert r.guardrail_retrieval == "proceed"
    assert r.guardrail_output == "proceed"
    assert r.stt_latency_ms == 150.0
    assert r.retrieval_latency_ms == 80.0
    assert r.generation_latency_ms == 200.0
    assert r.total_latency_ms == 430.0
    assert r.top_retrieval_score == 0.85
    assert r.metadata == {"source": "test"}


def test_log_store_filter_by_refused(tmp_path):
    store = LogStore(tmp_path / "logs.db")
    store.log_request(RequestLogEntry(
        request_id="r1", timestamp=time.time(), transcript="q1",
        language="hi", refused=False, guardrail_input="proceed",
        guardrail_retrieval="proceed", guardrail_output="proceed",
    ))
    store.log_request(RequestLogEntry(
        request_id="r2", timestamp=time.time(), transcript="q2",
        language="hi", refused=True, refusal_reason="off-topic",
        guardrail_input="refuse", guardrail_retrieval="proceed",
        guardrail_output="proceed",
    ))
    refused = store.query(refused_only=True)
    assert len(refused) == 1
    assert refused[0].request_id == "r2"


def test_log_store_recent(tmp_path):
    store = LogStore(tmp_path / "logs.db")
    for i in range(5):
        store.log_request(RequestLogEntry(
            request_id=f"r{i}", timestamp=time.time() + i, transcript=f"q{i}",
            language="hi", refused=False, guardrail_input="proceed",
            guardrail_retrieval="proceed", guardrail_output="proceed",
        ))
    recent = store.recent(n=3)
    assert len(recent) == 3
    assert recent[0].request_id == "r2"


def test_log_store_set_feedback(tmp_path):
    store = LogStore(tmp_path / "logs.db")
    store.log_request(RequestLogEntry(
        request_id="fb1", timestamp=time.time(), transcript="q",
        language="hi", refused=False, guardrail_input="proceed",
        guardrail_retrieval="proceed", guardrail_output="proceed",
    ))
    results = store.query()
    assert results[0].explicit_feedback is None

    store.set_feedback("fb1", 1)
    results = store.query()
    assert results[0].explicit_feedback == 1

    store.set_feedback("fb1", -1)
    results = store.query()
    assert results[0].explicit_feedback == -1


def test_log_store_stats(tmp_path):
    store = LogStore(tmp_path / "logs.db")
    store.log_request(RequestLogEntry(
        request_id="s1", timestamp=time.time(), transcript="q1",
        language="hi", refused=False, total_latency_ms=100.0,
        top_retrieval_score=0.9, guardrail_input="proceed",
        guardrail_retrieval="proceed", guardrail_output="proceed",
    ))
    store.log_request(RequestLogEntry(
        request_id="s2", timestamp=time.time(), transcript="q2",
        language="hi", refused=True, refusal_reason="bad",
        total_latency_ms=200.0, top_retrieval_score=0.5,
        guardrail_input="refuse", guardrail_retrieval="proceed",
        guardrail_output="proceed",
    ))
    s = store.stats()
    assert s["total"] == 2
    assert s["refused_count"] == 1
    assert s["avg_latency"] == 150.0
    assert s["avg_score"] == 0.7


def test_log_store_context_manager(tmp_path):
    with LogStore(tmp_path / "logs.db") as store:
        store.log_request(RequestLogEntry(
            request_id="cm1", timestamp=time.time(), transcript="q",
            language="hi", refused=False, guardrail_input="proceed",
            guardrail_retrieval="proceed", guardrail_output="proceed",
        ))
        assert len(store.query()) == 1


def test_log_store_duplicate_ignored(tmp_path):
    store = LogStore(tmp_path / "logs.db")
    store.log_request(RequestLogEntry(
        request_id="dup1", timestamp=100.0, transcript="first",
        language="hi", refused=False, guardrail_input="proceed",
        guardrail_retrieval="proceed", guardrail_output="proceed",
    ))
    store.log_request(RequestLogEntry(
        request_id="dup1", timestamp=200.0, transcript="second",
        language="hi", refused=False, guardrail_input="proceed",
        guardrail_retrieval="proceed", guardrail_output="proceed",
    ))
    results = store.query()
    assert len(results) == 1
    assert results[0].transcript == "first"
    assert results[0].timestamp == 100.0
