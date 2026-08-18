"""Query expansion via paraphrase generation.

Generates 2-3 paraphrases of the input query (synonyms, reordering) and
merges retrieval results across all variants using RRF. This recovers cases
where the literal query wording doesn't match passage phrasing.

Two modes:
* ``rule_based`` — fast, deterministic, no LLM; rewrites via Hindi/English
  synonym banks and word reordering.
* ``llm_based`` — uses a configured LLM to generate paraphrases ( richer
  but adds latency; optional toggle).
"""

from __future__ import annotations

import re
from typing import Any

from app.retrieval.retrievers import DenseRetriever, reciprocal_rank_fusion


# Hindi synonym/paraphrase bank for common query patterns
_HINDI_SYNONYMS: dict[str, list[str]] = {
    "कौन है": ["कौन था", "किसका नाम है"],
    "क्या है": ["किसे कहते है", "किसको बोलते है"],
    "कब हुआ": ["किस समय हुआ", "कब का है"],
    "कहाँ है": ["किस जगह है", "कहाँ पर स्थित है"],
    "क्यों है": ["किस कारण से है", "वजह क्या है"],
    "कैसे करें": ["किस तरह से करें", "क्या तरीका है"],
}

# English synonym bank
_ENGLISH_SYNONYMS: dict[str, list[str]] = {
    "what is": ["what does", "define"],
    "who is": ["who was", "name the"],
    "when did": ["when was", "at what time"],
    "where is": ["where can I find", "location of"],
    "why does": ["why is", "reason for"],
    "how to": ["how can I", "steps to"],
}


def _hindi_paraphrases(query: str) -> list[str]:
    """Generate Hindi paraphrases via synonym substitution."""
    paraphrases = []
    for pattern, replacements in _HINDI_SYNONYMS.items():
        if pattern in query:
            for repl in replacements[:1]:  # take first replacement only
                paraphrases.append(query.replace(pattern, repl, 1))
            break
    # Also try reordering: move the question word to the end
    question_words = ("कौन", "क्या", "कब", "कहाँ", "क्यों", "कैसे")
    for qw in question_words:
        if query.startswith(qw):
            rest = query[len(qw):].strip()
            if rest:
                paraphrases.append(f"{rest} {qw}")
            break
    return paraphrases[:2]


def _english_paraphrases(query: str) -> list[str]:
    """Generate English paraphrases via synonym substitution."""
    lower = query.lower()
    paraphrases = []
    for pattern, replacements in _ENGLISH_SYNONYMS.items():
        if lower.startswith(pattern):
            for repl in replacements[:1]:
                paraphrases.append(repl + query[len(pattern):])
            break
    return paraphrases[:2]


def generate_paraphrases(query: str, max_paraphrases: int = 2) -> list[str]:
    """Generate paraphrases of the query.

    Returns up to ``max_paraphrases`` variants. Deterministic, no LLM,
    <1ms latency.
    """
    # Detect language by script
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", query))
    if has_devanagari:
        return _hindi_paraphrases(query)[:max_paraphrases]
    return _english_paraphrases(query)[:max_paraphrases]


def expand_and_retrieve(
    query: str,
    embedder: Any,
    dense: DenseRetriever,
    k: int = 5,
    expansion_k: int = 15,
    max_paraphrases: int = 2,
) -> list:
    """Retrieve for the original query + paraphrases, merge with RRF.

    Returns top-k results after fusion and deduplication.
    """
    paraphrases = generate_paraphrases(query, max_paraphrases)
    all_queries = [query] + paraphrases

    # Collect rankings from all query variants
    rankings: list[list[str]] = []
    all_hits: dict[str, Any] = {}

    for q in all_queries:
        qv = embedder.encode_query(q)
        hits = dense.search(qv, k=expansion_k, query_text=q)
        ranking = [h.chunk_id for h in hits]
        rankings.append(ranking)
        for h in hits:
            all_hits[h.chunk_id] = h

    # RRF fusion
    fused = reciprocal_rank_fusion(rankings, k=60)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]

    return [all_hits[cid] for cid, _ in ranked if cid in all_hits]
