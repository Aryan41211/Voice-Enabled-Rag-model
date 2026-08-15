"""Load built indexes from disk into ready-to-use retrievers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss

from app.config import get_settings
from app.ingestion.chunking import Chunk
from app.retrieval.retrievers import DenseRetriever, HybridRetriever, SparseRetriever


class IndexNotFoundError(FileNotFoundError):
    """Raised when a requested strategy index is missing on disk."""


def strategy_dir(lang: str, strategy: str, index_dir: str | Path | None = None) -> Path:
    base = Path(index_dir or get_settings().index_dir)
    return base / lang / strategy


def load_chunks(lang: str, strategy: str, index_dir: str | Path | None = None) -> list[Chunk]:
    path = strategy_dir(lang, strategy, index_dir) / "chunks.pkl"
    if not path.exists():
        raise IndexNotFoundError(
            f"missing index chunks at {path} — run `python -m app.ingestion.build_index`"
        )
    with open(path, "rb") as fh:
        return [Chunk.model_validate(row) for row in pickle.load(fh)]


def load_dense(
    lang: str,
    strategy: str,
    chunks: list[Chunk] | None = None,
    index_dir: str | Path | None = None,
) -> DenseRetriever:
    base = strategy_dir(lang, strategy, index_dir)
    index_path = base / "dense.faiss"
    if not index_path.exists():
        raise IndexNotFoundError(f"missing dense index at {index_path}")
    index = faiss.read_index(str(index_path))
    chunks = chunks or load_chunks(lang, strategy, index_dir)
    return DenseRetriever(index, chunks)


def load_sparse(
    lang: str,
    strategy: str,
    chunks: list[Chunk] | None = None,
    index_dir: str | Path | None = None,
) -> SparseRetriever:
    base = strategy_dir(lang, strategy, index_dir)
    bm25_path = base / "sparse.pkl"
    if not bm25_path.exists():
        raise IndexNotFoundError(f"missing sparse index at {bm25_path}")
    with open(bm25_path, "rb") as fh:
        bm25 = pickle.load(fh)
    chunks = chunks or load_chunks(lang, strategy, index_dir)
    return SparseRetriever(bm25, chunks)


def load_hybrid(
    lang: str,
    strategy: str,
    index_dir: str | Path | None = None,
) -> HybridRetriever:
    chunks = load_chunks(lang, strategy, index_dir)
    dense = load_dense(lang, strategy, chunks, index_dir)
    sparse = load_sparse(lang, strategy, chunks, index_dir)
    return HybridRetriever(dense, sparse)


def load_manifest(
    lang: str,
    strategy: str,
    index_dir: str | Path | None = None,
) -> dict:
    path = strategy_dir(lang, strategy, index_dir) / "manifest.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
