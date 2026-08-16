"""API endpoint tests using a stubbed pipeline (no model download in CI)."""

import types

from fastapi.testclient import TestClient

from app.api import server
from app.harness.schemas import QueryResponse


def _fake_pipeline():
    class FakePipeline:
        generator = types.SimpleNamespace(name="extractive")
        stt = types.SimpleNamespace()

        async def query_async(self, transcript, request_id=None, timings=None):
            return QueryResponse(
                request_id=request_id or "test",
                transcript=transcript.text,
                answer="नमस्ते उत्तर",
                sources=[],
                timings_ms={"total_ms": 5.0},
            )

        async def process_audio(self, path):
            return QueryResponse(
                request_id="test",
                transcript="दिल्ली कहाँ है",
                answer="दिल्ली",
                timings_ms={},
            )

    return FakePipeline()


def test_health_ok(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    with TestClient(server.create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["generation"] == "extractive"


def test_query_returns_response(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    with TestClient(server.create_app()) as client:
        r = client.post("/query", json={"text": "प्रश्न क्या है", "language": "hi"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "नमस्ते उत्तर"
        assert body["timings_ms"]["total_ms"] == 5.0


def test_index_serves_voice_ui(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    with TestClient(server.create_app()) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "रिकॉर्ड करें" in r.text
        assert "/v1/voice" in r.text


def test_voice_endpoint_returns_response(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    with TestClient(server.create_app()) as client:
        r = client.post(
            "/v1/voice",
            files={"audio": ("speech.wav", b"fake-wav-bytes", "audio/wav")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["transcript"] == "दिल्ली कहाँ है"
        assert body["answer"] == "दिल्ली"


def test_health_503_when_not_loaded(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", None)
    monkeypatch.setattr(server.state, "ready", False)
    with TestClient(server.create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/query", json={"text": "प्रश्न"}).status_code == 503
