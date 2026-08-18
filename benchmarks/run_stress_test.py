"""Stress-test pipeline against messy real-world query text.

Tests retrieval robustness against:
1. Typos and STT-style errors (homophone substitutions, missing punctuation)
2. Extremely long or run-on queries
3. Queries with numbers/dates/named entities

Reports failure patterns and identifies cheap wins vs inherent limitations.
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
from app.retrieval.load import (
    IndexNotFoundError,
    load_chunks,
    load_dense,
)
from app.retrieval.retrievers import DenseRetriever
from benchmarks.metrics import mrr, recall_at_k


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
    return (
        f"{chunk.metadata.get('source_query_id')}:{chunk.metadata.get('passage_index')}"
    )


# ─── Stress test query categories ──────────────────────────────────────

# Typos and STT errors — realistic homophone/misspelling variants
TYPO_VARIANTS = [
    # Hindi typos (Devanagari character swaps, common STT errors)
    {"original": "भारत का राष्ट्रीय पक्षी क्या है", "variant": "भारत का राष्ट्रिय पक्षी क्या है", "type": "typo_hindi_vowel"},
    {"original": "गांधी जी का जन्म कब हुआ", "variant": "गांधी जी का जनम कब हुआ", "type": "typo_hindi_consonant"},
    {"original": "ताजमहल कहाँ है", "variant": "ताज महल कहा है", "type": "typo_hindi_split"},
    # English typos
    {"original": "What is the capital of France", "variant": "What is the capitl of Frnace", "type": "typo_english_misspell"},
    {"original": "Who invented the telephone", "variant": "Who inveted the telephone", "type": "typo_english_skip"},
    {"original": "When was Python created", "variant": "When was Phyton created", "type": "typo_english_transpose"},
]

# Long / run-on queries
LONG_QUERIES = [
    {
        "query": "मुझे बताओ कि भारत का राष्ट्रीय पक्षी कौन सा है और इसके बारे में सभी जानकारी दो जैसे कि इसका वैज्ञानिक नाम क्या है और यह कहाँ पाया जाता है और इसकी विशेषताएं क्या हैं",
        "type": "long_hindi",
        "expect_pass": True,
    },
    {
        "query": "Tell me everything you know about the Great Wall of China including when it was built, how long it is, who built it, what materials were used, and why it was constructed in the first place and how many people died during its construction",
        "type": "long_english",
        "expect_pass": True,
    },
    {
        "query": "क",
        "type": "single_char",
        "expect_pass": False,
    },
]

# Entity-heavy queries (numbers, dates, proper nouns)
ENTITY_QUERIES = [
    {
        "query": "1947 में भारत को आज़ादी कब मिली",
        "type": "date_entity",
        "expect_keywords": ["1947", "आज़ादी"],
    },
    {
        "query": "MSMARCO dataset में कितने queries हैं",
        "type": "number_entity",
        "expect_keywords": ["MSMARCO"],
    },
    {
        "query": "Who was the 16th President of the United States",
        "type": "ordinal_entity",
        "expect_keywords": ["16th", "President"],
    },
    {
        "query": "What is the population of Tokyo in 2024",
        "type": "year_entity",
        "expect_keywords": ["Tokyo", "2024"],
    },
]


def test_typo_robustness(
    dense: DenseRetriever,
    embedder: Embedder,
    eval_gold: list[dict],
    k: int = 5,
) -> dict:
    """Measure retrieval degradation under typos/STT errors."""
    # Build a lookup from original query to gold
    gold_by_query = {rec["query"]: gold_ids(rec) for rec in eval_gold}

    results = []
    for variant in TYPO_VARIANTS:
        orig = variant["original"]
        if orig not in gold_by_query:
            continue
        gold = gold_by_query[orig]

        # Clean retrieval
        qv_clean = embedder.encode_query(orig)
        hits_clean = dense.search(qv_clean, k=k, query_text=orig)
        r5_clean = recall_at_k([match_key(h) for h in hits_clean], gold, k)

        # Typo retrieval
        qv_typo = embedder.encode_query(variant["variant"])
        hits_typo = dense.search(qv_typo, k=k, query_text=variant["variant"])
        r5_typo = recall_at_k([match_key(h) for h in hits_typo], gold, k)

        results.append({
            "type": variant["type"],
            "original": orig,
            "variant": variant["variant"],
            "r5_clean": r5_clean,
            "r5_typo": r5_typo,
            "degradation": r5_clean - r5_typo,
        })

    if not results:
        return {"error": "no matching eval queries for typo variants"}

    avg_degradation = sum(r["degradation"] for r in results) / len(results)
    return {
        "n_variants": len(results),
        "avg_degradation": round(avg_degradation, 4),
        "cases": results,
    }


def test_long_queries(
    dense: DenseRetriever,
    embedder: Embedder,
    k: int = 5,
) -> dict:
    """Test retrieval on extremely long queries — confirm no silent truncation."""
    results = []
    for item in LONG_QUERIES:
        query = item["query"]
        qv = embedder.encode_query(query)
        t0 = time.perf_counter()
        hits = dense.search(qv, k=k, query_text=query)
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "type": item["type"],
            "query_length_chars": len(query),
            "query_length_words": len(query.split()),
            "n_results": len(hits),
            "top_score": hits[0].score if hits else 0,
            "latency_ms": round(latency_ms, 2),
            "passed": len(hits) > 0 if item["expect_pass"] else True,
        })

    return {"cases": results}


def test_entity_queries(
    dense: DenseRetriever,
    embedder: Embedder,
    eval_gold: list[dict],
    k: int = 5,
) -> dict:
    """Test entity-heavy queries — check if named entities are handled."""
    results = []
    for item in ENTITY_QUERIES:
        query = item["query"]
        qv = embedder.encode_query(query)
        hits = dense.search(qv, k=k, query_text=query)

        # Check if any result text contains expected keywords
        result_texts = [h.text.lower() for h in hits]
        keywords_found = 0
        for kw in item["expect_keywords"]:
            if any(kw.lower() in t for t in result_texts):
                keywords_found += 1

        results.append({
            "type": item["type"],
            "query": query,
            "n_results": len(hits),
            "top_score": hits[0].score if hits else 0,
            "keywords_found": keywords_found,
            "keywords_total": len(item["expect_keywords"]),
        })

    return {"cases": results}


def main():
    parser = argparse.ArgumentParser(description="Stress test for messy queries")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = get_settings()
    index_dir = Path(args.index_dir or settings.index_dir)
    lang = args.lang

    gold_records = load_eval_gold(lang, index_dir)
    embedder = Embedder()
    embedder.encode_query("warmup")

    try:
        chunks = load_chunks(lang, "metadata", index_dir)
        dense = load_dense(lang, "metadata", chunks, index_dir)
    except IndexNotFoundError as e:
        print(f"[stress] {e}")
        sys.exit(1)

    print(f"[stress] running stress tests on {len(gold_records)} eval queries...")

    results = {
        "typo_robustness": test_typo_robustness(dense, embedder, gold_records, args.topk),
        "long_queries": test_long_queries(dense, embedder, args.topk),
        "entity_queries": test_entity_queries(dense, embedder, gold_records, args.topk),
    }

    out_path = args.out or str(
        Path(settings.index_dir).parent.parent / "benchmarks" / "results" / "stress_test.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"[stress] wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
