"""Enhanced retrieval evaluation: RRF weight tuning + negative mining analysis.

1. Sweep dense/sparse weight ratios for RRF fusion to find optimal mix
2. Analyze negative mining failures — which queries still fail and why
3. Reports per-category breakdown for targeted improvement
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.ingestion.embed import Embedder
from app.retrieval.load import IndexNotFoundError, load_chunks, load_dense, load_sparse
from app.retrieval.retrievers import DenseRetriever, SparseRetriever, reciprocal_rank_fusion
from benchmarks.metrics import mrr, percentile, recall_at_k


def load_eval_gold(lang: str, index_dir: Path) -> list[dict]:
    path = index_dir / lang / "eval_gold.jsonl"
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
    return f"{chunk.metadata.get('source_query_id')}:{chunk.metadata.get('passage_index')}"


def weighted_rrf(
    dense_hits: list,
    sparse_hits: list,
    dense_weight: float,
    sparse_weight: float,
    k: int = 60,
) -> dict[str, float]:
    """Weighted RRF: apply per-ranker weight to the reciprocal score."""
    scores: dict[str, float] = {}
    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + dense_weight / (k + rank)
    for rank, hit in enumerate(sparse_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + sparse_weight / (k + rank)
    return scores


def rerank_by_score(
    dense_hits: list,
    sparse_hits: list,
    fused_scores: dict[str, float],
    k: int,
) -> list[str]:
    """Sort RetrievedChunks by fused score and return match_keys."""
    by_id = {}
    for hit in (*dense_hits, *sparse_hits):
        by_id.setdefault(hit.chunk_id, hit)
    ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [match_key(by_id[cid]) for cid, _ in ranked if cid in by_id]


def sweep_rrf_weights(
    dense: DenseRetriever,
    sparse: SparseRetriever,
    embedder: Embedder,
    eval_gold: list[dict],
    k: int = 5,
) -> dict:
    """Sweep dense/sparse weight ratios for RRF and report R@5 and MRR."""
    dense_weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = {}

    for w in dense_weights:
        r5_scores = []
        mrr_scores = []
        for rec in eval_gold:
            gold = gold_ids(rec)
            qv = embedder.encode_query(rec["query"])

            dense_hits = dense.search(qv, k=10, query_text=rec["query"])
            sparse_hits = sparse.search(rec["query"], k=10)

            # Weighted RRF fusion
            fused_scores = weighted_rrf(
                dense_hits, sparse_hits,
                dense_weight=w, sparse_weight=1.0 - w,
            )
            keys = rerank_by_score(dense_hits, sparse_hits, fused_scores, k)
            r5_scores.append(recall_at_k(keys, gold, k))
            mrr_scores.append(mrr(keys, gold))

        avg_r5 = sum(r5_scores) / len(r5_scores) if r5_scores else 0
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        label = f"dense={w:.1f}/sparse={1.0 - w:.1f}"
        results[label] = {
            "r5": round(avg_r5, 4),
            "mrr": round(avg_mrr, 4),
        }

    return results


def analyze_failures(
    dense: DenseRetriever,
    sparse: SparseRetriever,
    embedder: Embedder,
    eval_gold: list[dict],
    k: int = 5,
) -> dict:
    """Identify queries where retrieval fails — category breakdown + failure patterns."""
    failures = []
    for rec in eval_gold:
        gold = gold_ids(rec)
        qv = embedder.encode_query(rec["query"])
        dense_hits = dense.search(qv, k=k, query_text=rec["query"])
        keys = [match_key(h) for h in dense_hits]
        r5 = recall_at_k(keys, gold, k)

        if r5 < 0.5:
            failures.append({
                "query": rec["query"],
                "r5": r5,
                "gold_count": len(gold),
                "retrieved_keys": keys[:k],
                "query_length": len(rec["query"]),
                "lang": rec.get("lang", "unknown"),
            })

    # Categorize failures
    categories = {"short_query": 0, "long_query": 0, "low_recall": 0, "total": len(failures)}
    for f in failures:
        if f["query_length"] < 20:
            categories["short_query"] += 1
        elif f["query_length"] > 60:
            categories["long_query"] += 1
        if f["r5"] == 0:
            categories["low_recall"] += 1

    return {
        "total_failures": len(failures),
        "failure_rate": round(len(failures) / len(eval_gold), 4) if eval_gold else 0,
        "categories": categories,
        "top_failures": failures[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Enhanced retrieval eval with RRF tuning")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = get_settings()
    index_dir = Path(args.index_dir or settings.index_dir)

    gold_records = load_eval_gold(args.lang, index_dir)
    embedder = Embedder()
    embedder.encode_query("warmup")

    try:
        chunks = load_chunks(args.lang, "metadata", index_dir)
        dense = load_dense(args.lang, "metadata", chunks, index_dir)
        sparse = load_sparse(args.lang, "metadata", chunks, index_dir)
    except IndexNotFoundError as e:
        print(f"[eval-enhanced] {e}")
        sys.exit(1)

    print(f"[eval-enhanced] running on {len(gold_records)} queries...")

    results = {
        "rrf_weight_sweep": sweep_rrf_weights(dense, sparse, embedder, gold_records, args.topk),
        "failure_analysis": analyze_failures(dense, sparse, embedder, gold_records, args.topk),
    }

    out_path = args.out or str(
        Path(settings.index_dir).parent.parent / "benchmarks" / "results" / "retrieval_eval_enhanced.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"[eval-enhanced] wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
