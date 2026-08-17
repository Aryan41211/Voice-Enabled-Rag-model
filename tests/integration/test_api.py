"""API endpoint tests using a stubbed pipeline (no model download in CI)."""

import types

from fastapi.testclient import TestClient

from app.api import server
from app.harness.schemas import QueryResponse


def _fake_pipeline():
    class FakePipeline:
        generator = types.SimpleNamespace(name="extractive")
        stt = types.SimpleNamespace()

        async def query_async(self, transcript, request_id=None, timings=None, session_id=None):
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


def test_voice_sanitizes_upload_filename(monkeypatch):
    import pathlib

    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)

    written = []
    real_write = pathlib.Path.write_bytes

    def capturing_write(self, data):
        written.append(str(self))
        return real_write(self, data)

    monkeypatch.setattr(pathlib.Path, "write_bytes", capturing_write)
    with TestClient(server.create_app()) as client:
        r = client.post(
            "/v1/voice",
            files={"audio": ("..\\..\\evil.wav", b"fake", "audio/wav")},
        )
        assert r.status_code == 200
    assert written
    # basename is kept, traversal components are stripped
    assert written[0].endswith("evil.wav")
    assert ".." not in written[0].split("tmp")[-1]


def test_health_503_when_not_loaded(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", None)
    monkeypatch.setattr(server.state, "ready", False)
    with TestClient(server.create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/query", json={"text": "प्रश्न"}).status_code == 503


def test_query_accepts_session_id(monkeypatch):
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    with TestClient(server.create_app()) as client:
        r = client.post(
            "/query",
            json={"text": "प्रश्न", "language": "hi", "session_id": "s1"},
        )
        assert r.status_code == 200


def test_feedback_sets_feedback(monkeypatch, tmp_path):
    from app.observability.store import LogStore

    store = LogStore(tmp_path / "logs.db")
    monkeypatch.setattr(server.state, "log_store", store)
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    with TestClient(server.create_app()) as client:
        r = client.post("/feedback", json={"request_id": "r1", "feedback": 1})
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_health_includes_log_stats(monkeypatch, tmp_path):
    from app.observability.store import LogStore

    store = LogStore(tmp_path / "logs.db")
    monkeypatch.setattr(server.state, "log_store", store)
    monkeypatch.setattr(server.state, "load", lambda: None)
    monkeypatch.setattr(server.state, "pipeline", _fake_pipeline())
    monkeypatch.setattr(server.state, "ready", True)
    with TestClient(server.create_app()) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "log_stats" in body
