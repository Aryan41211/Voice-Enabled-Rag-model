import numpy as np

from app.harness.schemas import RetrievedChunk
from app.retrieval.rerank import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        # higher score for chunks that contain the query token
        return np.array([1.0 if "दस्तावेज़" in p[1] else 0.0 for p in pairs])


def _hit(cid, text):
    return RetrievedChunk(chunk_id=cid, text=text, source="dense", strategy="metadata")


def test_rerank_reorders(monkeypatch):
    reranker = CrossEncoderReranker(model_name="fake")
    monkeypatch.setattr(reranker, "_model", FakeCrossEncoder())

    hits = [
        _hit("c1", "अन्य पाठ"),
        _hit("c2", "दस्तावेज़ मैच"),
        _hit("c3", "कुछ और"),
    ]
    out = reranker.rerank("दस्तावेज़", hits)
    assert out[0].chunk_id == "c2"
    assert out[0].score == 1.0


def test_rerank_empty():
    reranker = CrossEncoderReranker(model_name="fake")
    assert reranker.rerank("q", []) == []


def test_rerank_top_n(monkeypatch):
    reranker = CrossEncoderReranker(model_name="fake")
    monkeypatch.setattr(reranker, "_model", FakeCrossEncoder())
    hits = [
        _hit(f"c{i}", "दस्तावेज़" if i % 2 else "अन्य") for i in range(6)
    ]
    out = reranker.rerank("दस्तावेज़", hits, top_n=2)
    assert len(out) == 2
    assert all(h.score == 1.0 for h in out)
