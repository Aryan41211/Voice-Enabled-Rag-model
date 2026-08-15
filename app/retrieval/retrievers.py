"""Dense, sparse and hybrid retrievers over built indexes.

* ``DenseRetriever`` — FAISS inner-product (cosine) search over chunk vectors.
* ``SparseRetriever`` — BM25 over tokenized chunk text.
* ``HybridRetriever`` — Reciprocal Rank Fusion of dense + sparse rankings.
"""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.harness.schemas import RetrievedChunk
from app.ingestion.chunking import Chunk
from app.retrieval.tokenize import tokenize


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked lists of chunk ids via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class DenseRetriever:
    def __init__(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
    ) -> None:
        self.index = index
        self.chunks = chunks

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        query_text: str = "",
    ) -> list[RetrievedChunk]:
        query_vec = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, idxs = self.index.search(query_vec, int(k))
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            c = self.chunks[int(idx)]
            out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.context,
                    score=float(score),
                    source="dense",
                    strategy=c.strategy,
                    metadata=self._metadata(c),
                )
            )
        return out

    @staticmethod
    def _metadata(chunk: Chunk) -> dict:
        return {
            "source_query_id": chunk.source_query_id,
            "passage_index": chunk.passage_index,
            "language": chunk.language,
            "passage_is_selected": chunk.passage_is_selected,
            "parent_chunk_id": chunk.parent_chunk_id,
        }


class SparseRetriever:
    def __init__(
        self,
        bm25: BM25Okapi,
        chunks: list[Chunk],
    ) -> None:
        self.bm25 = bm25
        self.chunks = chunks

    def search(self, query_text: str, k: int = 5) -> list[RetrievedChunk]:
        query_tokens = tokenize(query_text)
        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype="float64")
        top = np.argsort(scores)[::-1][:k]
        out: list[RetrievedChunk] = []
        for idx in top:
            c = self.chunks[int(idx)]
            out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.context,
                    score=float(scores[idx]),
                    source="sparse",
                    strategy=c.strategy,
                    metadata=DenseRetriever._metadata(c),
                )
            )
        return out


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = 60,
        dense_k: int = 50,
        sparse_k: int = 50,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.dense_k = dense_k
        self.sparse_k = sparse_k

    def search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        k: int = 5,
        reranker: Any | None = None,
    ) -> list[RetrievedChunk]:
        dense_hits = self.dense.search(query_vec, k=self.dense_k, query_text=query_text)
        sparse_hits = self.sparse.search(query_text, k=self.sparse_k)

        by_id: dict[str, RetrievedChunk] = {}
        for hit in (*dense_hits, *sparse_hits):
            by_id.setdefault(hit.chunk_id, hit)

        fused = reciprocal_rank_fusion(
            [[h.chunk_id for h in dense_hits], [h.chunk_id for h in sparse_hits]],
            k=self.rrf_k,
        )
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = [
            by_id[cid].model_copy(update={"score": score, "source": "hybrid"})
            for cid, score in ranked
        ]

        if reranker is not None:
            out = reranker.rerank(query_text, out)
        return out


def make_retrievers(
    chunks: list[Chunk],
    vectors: np.ndarray | None = None,
    bm25: BM25Okapi | None = None,
) -> tuple[DenseRetriever | None, SparseRetriever | None]:
    """Build dense + sparse retrievers from chunk lists and prebuilt data.

    Returns ``None`` for either retriever when its data is not provided.
    """
    dense: DenseRetriever | None = None
    if vectors is not None:
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        vecs = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(vecs)
        index.add(vecs)
        dense = DenseRetriever(index, chunks)

    sparse: SparseRetriever | None = None
    if bm25 is not None:
        sparse = SparseRetriever(bm25, chunks)

    return dense, sparse
