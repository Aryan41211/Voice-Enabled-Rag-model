"""On-disk index loading (app/retrieval/load.py) against a hand-built tiny index.

Proves the exact fresh-clone path — artifacts written by
``build_strategy()`` load back into usable retrievers — without needing the
embedding model or any network access.
"""

import json
import pickle

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from app.ingestion.chunking import Chunk
from app.retrieval.load import (
    IndexNotFoundError,
    load_chunks,
    load_dense,
    load_hybrid,
    load_manifest,
    load_sparse,
    strategy_dir,
)
from app.retrieval.tokenize import tokenize


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c1",
            text="दिल्ली भारत की राजधानी है",
            context="दिल्ली भारत की राजधानी है",
            source_query_id=1,
            passage_index=0,
            language="hi",
            strategy="metadata",
            position=0,
            passage_is_selected=1,
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
            passage_is_selected=1,
        ),
        Chunk(
            chunk_id="c3",
            text="ताजमहल विश्व का सातवाँ अजूबा है",
            context="ताजमहल विश्व का सातवाँ अजूबा है",
            source_query_id=3,
            passage_index=0,
            language="hi",
            strategy="metadata",
            position=0,
            passage_is_selected=1,
        ),
    ]


def _write_index(tmp_path) -> list[Chunk]:
    out = strategy_dir("hi", "metadata", tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    chunks = _chunks()

    with open(out / "chunks.pkl", "wb") as fh:
        pickle.dump([c.model_dump() for c in chunks], fh)

    vecs = np.asarray([[1, 0, 0, 0], [0.8, 0.2, 0, 0], [0, 0, 1, 0]], dtype="float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(4)
    index.add(vecs)
    faiss.write_index(index, str(out / "dense.faiss"))

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    with open(out / "sparse.pkl", "wb") as fh:
        pickle.dump(bm25, fh)

    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"strategy": "metadata", "n_chunks": len(chunks)}, fh)
    return chunks


def test_load_chunks_roundtrips(tmp_path):
    expected = _write_index(tmp_path)
    loaded = load_chunks("hi", "metadata", tmp_path)
    assert [c.chunk_id for c in loaded] == [c.chunk_id for c in expected]
    assert loaded[0].passage_is_selected == 1


def test_load_dense_searches(tmp_path):
    _write_index(tmp_path)
    retriever = load_dense("hi", "metadata", index_dir=tmp_path)
    hits = retriever.search(np.asarray([1, 0, 0, 0], dtype="float32"), k=2)
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].source == "dense"
    assert hits[0].metadata["source_query_id"] == 1


def test_load_sparse_searches(tmp_path):
    _write_index(tmp_path)
    retriever = load_sparse("hi", "metadata", index_dir=tmp_path)
    hits = retriever.search("दिल्ली", k=1)
    assert hits and hits[0].chunk_id == "c1"


def test_load_hybrid_fuses(tmp_path):
    _write_index(tmp_path)
    retriever = load_hybrid("hi", "metadata", index_dir=tmp_path)
    # Dense favors c1 (unit vector [1,0,0,0]); sparse favors c2/c3 ("ताजमहल").
    # Only c2 appears in both top-2 lists, so RRF promotes it to rank 1.
    hits = retriever.search(
        np.asarray([1, 0, 0, 0], dtype="float32"), query_text="ताजमहल", k=2
    )
    assert hits[0].chunk_id == "c2"
    assert hits[0].source == "hybrid"


def test_load_manifest(tmp_path):
    _write_index(tmp_path)
    m = load_manifest("hi", "metadata", tmp_path)
    assert m["strategy"] == "metadata"
    assert load_manifest("hi", "missing", tmp_path) == {}


def test_missing_index_raises(tmp_path):
    _write_index(tmp_path)
    with pytest.raises(IndexNotFoundError):
        load_chunks("hi", "no-such-strategy", tmp_path)
    with pytest.raises(IndexNotFoundError):
        load_dense("hi", "no-such-strategy", tmp_path)
    with pytest.raises(IndexNotFoundError):
        load_sparse("hi", "no-such-strategy", tmp_path)
