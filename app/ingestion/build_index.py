"""Build FAISS (dense) + BM25 (sparse) indexes per chunking strategy.

Offline stage — not part of the query-time latency budget (see ARCHITECTURE.md
Stage 0). Saves everything under ``data/index/{lang}/{strategy}/``:

* ``dense.faiss``  — FAISS inner-product index over chunk vectors
* ``chunks.pkl``   — aligned list of :class:`Chunk` objects
* ``sparse.pkl``   — BM25 (rank_bm25) over the same chunks
* ``manifest.json``— build metadata (strategy, counts, model, timings)

Also writes ``data/index/{lang}/eval_gold.jsonl`` — the sampled eval queries
with their gold ``(query_id, passage_index)`` pairs.

Usage:
    python -m app.ingestion.build_index --lang hi --corpus-queries 2000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from datasets import Dataset
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.ingestion.chunking import STRATEGIES, Chunk, chunk_examples
from app.ingestion.dataset import load as load_dataset
from app.ingestion.embed import Embedder
from app.retrieval.load import strategy_dir
from app.retrieval.tokenize import tokenize

DEFAULT_SEED = 42


def has_selected(ex: dict) -> bool:
    return sum(ex["passages"]["is_selected"]) >= 1


def sample_queries(
    ds: Dataset,
    corpus_queries: int,
    eval_queries: int,
    seed: int = DEFAULT_SEED,
    max_rows: int | None = None,
) -> list[dict]:
    """Sample corpus queries (each with >=1 gold passage); first N are eval."""
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    filtered = ds.filter(has_selected, num_proc=1)
    sampled = filtered.shuffle(seed=seed).select(range(corpus_queries))
    return [ex for ex in sampled]


def write_eval_gold(examples: list[dict], path: Path, n_eval: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples[:n_eval]:
            gold = [
                [ex["query_id"], pidx]
                for pidx, sel in enumerate(ex["passages"]["is_selected"])
                if sel
            ]
            fh.write(
                json.dumps(
                    {
                        "query_id": ex["query_id"],
                        "query": ex["query"],
                        "gold": gold,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_strategy(
    chunks: list[Chunk],
    strategy: str,
    out_dir: Path,
    embedder: Embedder,
    skip_dense: bool,
    skip_sparse: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [c.text for c in chunks]
    timings: dict[str, float] = {}

    with open(out_dir / "chunks.pkl", "wb") as fh:
        pickle.dump([c.model_dump() for c in chunks], fh)

    if not skip_dense:
        t0 = time.perf_counter()
        vectors = embedder.encode_passages(texts)
        normed = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(normed)
        index = faiss.IndexFlatIP(int(normed.shape[1]))
        index.add(normed)
        faiss.write_index(index, str(out_dir / "dense.faiss"))
        timings["dense_build_s"] = round(time.perf_counter() - t0, 2)
    else:
        timings["dense_build_s"] = 0.0

    if not skip_sparse:
        t0 = time.perf_counter()
        tokenized = [tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open(out_dir / "sparse.pkl", "wb") as fh:
            pickle.dump(bm25, fh)
        timings["sparse_build_s"] = round(time.perf_counter() - t0, 2)
    else:
        timings["sparse_build_s"] = 0.0

    return timings


def manifest(
    strategy: str,
    chunks: list[Chunk],
    timings: dict,
    model_name: str,
    n_queries: int,
) -> dict:
    return {
        "strategy": strategy,
        "n_chunks": len(chunks),
        "n_queries": n_queries,
        "n_passages": len({(c.source_query_id, c.passage_index) for c in chunks}),
        "embedding_model": model_name,
        "build_timings_s": timings,
        "dense_index": "IndexFlatIP",
        "sparse_index": "BM25Okapi",
    }


def build(
    lang: str,
    split: str,
    corpus_queries: int,
    eval_queries: int,
    strategies: list[str],
    index_dir: str | Path,
    seed: int = DEFAULT_SEED,
    max_rows: int | None = None,
    skip_dense: bool = False,
    skip_sparse: bool = False,
    model_name: str | None = None,
) -> None:
    settings = get_settings()
    index_dir = Path(index_dir)
    embedder = Embedder(model_name=model_name) if not skip_dense else None

    print(f"[build] loading {lang}/{split} dataset...")
    ds = load_dataset(lang, split)
    examples = sample_queries(
        ds, corpus_queries, eval_queries, seed=seed, max_rows=max_rows
    )
    print(f"[build] sampled {len(examples)} corpus queries "
          f"({eval_queries} eval)")

    write_eval_gold(examples, index_dir / lang / "eval_gold.jsonl", eval_queries)

    for strategy in strategies:
        print(f"[build] chunking strategy={strategy} ...")
        t0 = time.perf_counter()
        chunks = chunk_examples(examples, strategy, embedder=embedder)
        chunk_s = round(time.perf_counter() - t0, 2)

        out_dir = strategy_dir(lang, strategy, index_dir)
        timings = build_strategy(
            chunks,
            strategy,
            out_dir,
            embedder,
            skip_dense,
            skip_sparse,
        )
        timings["chunking_s"] = chunk_s

        m = manifest(
            strategy, chunks, timings, embedder.model_name if embedder else "skipped",
            len(examples),
        )
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=2)
        print(f"[build] {strategy}: {len(chunks)} chunks -> {out_dir}")
        for k, v in timings.items():
            print(f"    {k}: {v}")

    print("[build] done")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.ingestion.build_index")
    p.add_argument("--lang", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--corpus-queries", type=int, default=2000)
    p.add_argument("--eval-queries", type=int, default=200)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    p.add_argument("--index-dir", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap dataset rows scanned (fast iteration)")
    p.add_argument("--model", default=None, help="override EMBEDDING_MODEL")
    p.add_argument("--skip-dense", action="store_true")
    p.add_argument("--skip-sparse", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    build(
        lang=args.lang or settings.data_lang,
        split=args.split or settings.data_split,
        corpus_queries=args.corpus_queries,
        eval_queries=args.eval_queries,
        strategies=args.strategies,
        index_dir=args.index_dir or settings.index_dir,
        seed=args.seed,
        max_rows=args.max_rows,
        skip_dense=args.skip_dense,
        skip_sparse=args.skip_sparse,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
