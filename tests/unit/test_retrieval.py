from app.retrieval.tokenize import tokenize
from app.retrieval.retrievers import (
    HybridRetriever,
    SparseRetriever,
    reciprocal_rank_fusion,
)
from app.ingestion.chunking import Chunk


def _chunk(text, cid="c1", qid=1, pidx=0):
    return Chunk(
        chunk_id=cid,
        text=text,
        context=text,
        source_query_id=qid,
        passage_index=pidx,
        language="hin_Deva",
        strategy="metadata",
        position=0,
        passage_is_selected=1,
    )


def test_tokenize_hindi_and_english():
    assert tokenize("कॉर्पोरेशन क्या है?") == ["कॉर्पोरेशन", "क्या", "है"]
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("a_b") == ["a_b"]


def test_rrf_hand_computed():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert fused["x"] == 1 / 61 + 1 / 62
    assert fused["y"] == 1 / 62 + 1 / 61
    assert fused["w"] == 1 / 63
    assert fused["z"] == 1 / 63


def test_rrf_rank_order():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "b"]])
    ordered = sorted(fused, key=fused.get, reverse=True)
    assert ordered[:2] == ["a", "b"]


def test_dense_retriever_simple():
    import numpy as np

    chunks = [
        _chunk("मैकडॉनल्ड कॉर्पोरेशन", "c1"),
        _chunk("भारत की राजधानी", "c2"),
        _chunk("जलवायु परिवर्तन", "c3"),
    ]
    vecs = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32"
    )
    from app.retrieval.retrievers import make_retrievers

    dense, _ = make_retrievers(chunks, vectors=vecs)
    hits = dense.search(np.array([1.0, 0.0, 0.0], dtype="float32"), k=2)
    assert hits[0].chunk_id == "c1"
    assert hits[0].source == "dense"
    assert len(hits) == 2


def test_sparse_retriever_simple():
    from rank_bm25 import BM25Okapi

    chunks = [
        _chunk("मैकडॉनल्ड कॉर्पोरेशन एक कंपनी है", "c1"),
        _chunk("भारत की राजधानी नई दिल्ली है", "c2"),
    ]
    from app.retrieval.tokenize import tokenize

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    sparse = SparseRetriever(bm25, chunks)
    hits = sparse.search("भारत राजधानी", k=1)
    assert hits[0].chunk_id == "c2"


def test_hybrid_fuses_dense_and_sparse():
    import numpy as np

    chunks = [
        _chunk("पहला दस्तावेज़ अल्फा", "c1"),
        _chunk("दूसरा दस्तावेज़ बीटा", "c2"),
    ]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    from app.retrieval.tokenize import tokenize
    from rank_bm25 import BM25Okapi

    from app.retrieval.retrievers import make_retrievers

    d, s = make_retrievers(
        chunks, vectors=vecs, bm25=BM25Okapi([tokenize(c.text) for c in chunks])
    )
    hybrid = HybridRetriever(d, s)
    hits = hybrid.search(np.array([1.0, 0.0], dtype="float32"), "पहला दस्तावेज़", k=2)
    assert len(hits) == 2
    assert all(h.source == "hybrid" for h in hits)
