"""Unit tests for API rate limiting."""

import types

from fastapi.testclient import TestClient

from app.api import server


def _fake_pipeline():
    class FakePipeline:
        generator = types.SimpleNamespace(name="extractive")
        stt = types.SimpleNamespace()

        async def query_async(self, transcript, request_id=None, timings=None, session_id=None):
            from app.harness.schemas import QueryResponse
            return QueryResponse(
                request_id=request_id or "test",
                transcript=transcript.text,
                answer="answer",
                sources=[],
                timings_ms={},
            )

        async def process_audio(self, path):
            from app.harness.schemas import QueryResponse
            return QueryResponse(request_id="test", transcript="x", answer="y", timings_ms={})

    return FakePipeline()


def _setup_app(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    return TestClient(server.create_app())


def test_rate_limit_allows_normal_traffic(monkeypatch):
    client = _setup_app(monkeypatch)
    for _ in range(5):
        r = client.post("/query", json={"text": "hello", "language": "hi"})
        assert r.status_code == 200


def test_rate_limit_rejects_after_threshold(monkeypatch):
    client = _setup_app(monkeypatch)
    # Default limit is 30 req/min; send 31 requests from same IP
    for i in range(30):
        r = client.post("/query", json={"text": f"q{i}", "language": "hi"})
        assert r.status_code == 200, f"request {i+1} should succeed"
    r = client.post("/query", json={"text": "q31", "language": "hi"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_rate_limit_exempts_health(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    monkeypatch.setattr(server.state, "log_store", None)
    with TestClient(server.create_app()) as client:
        for _ in range(35):
            r = client.get("/health")
            assert r.status_code == 200
