"""Retrieval evaluation against MSMARCO-XI labeled passages.

For each chunking strategy's built index, runs the eval queries from
``data/index/{lang}/eval_gold.jsonl`` through dense, sparse and hybrid (RRF)
retrieval and reports Recall@3, Recall@5, MRR and per-query latency.

Usage:
    python benchmarks/run_retrieval_eval.py --lang hi --strategies all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.ingestion.embed import Embedder
from app.ingestion.chunking import STRATEGIES
from app.retrieval.load import (
    IndexNotFoundError,
    load_chunks,
    load_dense,
    load_manifest,
    load_sparse,
)
from app.retrieval.retrievers import DenseRetriever, HybridRetriever
from benchmarks.metrics import mrr, percentile, recall_at_k


def load_eval_gold(lang: str, index_dir: str | Path) -> list[dict]:
    path = Path(index_dir) / lang / "eval_gold.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing eval gold at {path}")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def gold_ids(record: dict) -> set[str]:
    return {f"{qid}:{pidx}" for qid, pidx in record["gold"]}


def match_key(chunk) -> str:
    return (
        f"{chunk.metadata.get('source_query_id')}:{chunk.metadata.get('passage_index')}"
    )


def evaluate_retriever(
    retriever,
    query: str,
    embedder: Embedder,
    gold: set[str],
    k: int,
    use_sparse: bool = True,
    reranker=None,
    rerank_candidates: int = 20,
) -> tuple[dict, float]:
    t0 = time.perf_counter()
    query_vec = embedder.encode_query(query)
    if isinstance(retriever, HybridRetriever):
        hits = retriever.search(query_vec, query, k=k)
    elif isinstance(retriever, DenseRetriever) or not use_sparse:
        hits = retriever.search(
            query_vec, k=k if reranker is None else rerank_candidates, query_text=query
        )
    else:
        hits = retriever.search(query, k=k)
    if reranker is not None and hits:
        hits = reranker.rerank(query, hits, top_n=k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ranked = [match_key(h) for h in hits]
    return (
        {
            "recall_3": recall_at_k(ranked, gold, 3),
            "recall_5": recall_at_k(ranked, gold, 5),
            "mrr": mrr(ranked, gold),
        },
        elapsed_ms,
    )


def run(
    lang: str,
    strategies: list[str],
    index_dir: str | Path,
    k: int = 5,
    model_name: str | None = None,
    rerank: bool = False,
    max_queries: int | None = None,
) -> dict:
    index_dir = Path(index_dir)
    gold_records = load_eval_gold(lang, index_dir)
    if max_queries is not None:
        gold_records = gold_records[:max_queries]
    embedder = Embedder(model_name=model_name)
    # warm up the model so the first timed query doesn't pay load cost
    embedder.encode_query("warmup")

    reranker = None
    if rerank:
        from app.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker()

    results = {}
    for strategy in strategies:
        try:
            chunks = load_chunks(lang, strategy, index_dir)
            dense = load_dense(lang, strategy, chunks, index_dir)
            sparse = load_sparse(lang, strategy, chunks, index_dir)
        except IndexNotFoundError as exc:
            print(f"[eval] skipping {strategy}: {exc}")
            continue
        hybrid = HybridRetriever(dense, sparse)

        agg = {
            "dense": {"recall_3": [], "recall_5": [], "mrr": [], "latency_ms": []},
            "hybrid": {"recall_3": [], "recall_5": [], "mrr": [], "latency_ms": []},
            "rerank": {"recall_3": [], "recall_5": [], "mrr": [], "latency_ms": []},
        }
        for rec in gold_records:
            gold = gold_ids(rec)
            for name, retriever in (
                ("dense", dense),
                ("hybrid", hybrid),
                ("rerank", dense),
            ):
                metrics, elapsed = evaluate_retriever(
                    retriever,
                    rec["query"],
                    embedder,
                    gold,
                    k,
                    reranker=reranker if name == "rerank" else None,
                )
                for mk, mv in metrics.items():
                    agg[name][mk].append(mv)
                agg[name]["latency_ms"].append(elapsed)

        m = load_manifest(lang, strategy, index_dir)
        per_retriever = {}
        for name in ("dense", "hybrid", "rerank"):
            if name == "rerank" and reranker is None:
                continue
            a = agg[name]
            n = max(len(a["latency_ms"]), 1)
            per_retriever[name] = {
                "recall_3": round(sum(a["recall_3"]) / n, 4),
                "recall_5": round(sum(a["recall_5"]) / n, 4),
                "mrr": round(sum(a["mrr"]) / n, 4),
                "p50_latency_ms": round(percentile(a["latency_ms"], 50), 2),
                "p70_latency_ms": round(percentile(a["latency_ms"], 70), 2),
                "p100_latency_ms": round(percentile(a["latency_ms"], 100), 2),
            }
        results[strategy] = {
            "n_chunks": m.get("n_chunks"),
            "n_queries": m.get("n_queries"),
            "build_timings_s": m.get("build_timings_s"),
            "model": m.get("embedding_model"),
            **per_retriever,
        }
        print(f"[eval] {strategy}: {results[strategy]}")

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_retrieval_eval")
    p.add_argument("--lang", default=None)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    p.add_argument("--index-dir", default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--model", default=None)
    p.add_argument(
        "--rerank", action="store_true", help="enable cross-encoder rerank ablation"
    )
    p.add_argument(
        "--max-queries", type=int, default=None, help="cap number of eval queries"
    )
    p.add_argument("--out", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    results = run(
        lang=args.lang or settings.data_lang,
        strategies=args.strategies,
        index_dir=args.index_dir or settings.index_dir,
        k=args.topk,
        model_name=args.model,
        rerank=args.rerank,
        max_queries=args.max_queries,
    )
    out_path = args.out or (
        Path(settings.index_dir).parent.parent
        / "benchmarks"
        / "results"
        / "retrieval_eval.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"lang": args.lang or settings.data_lang, "results": results},
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
