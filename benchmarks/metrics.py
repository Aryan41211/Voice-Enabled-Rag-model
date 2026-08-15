"""Retrieval metrics and latency percentiles.

* ``percentile`` — matches the definition in LATENCY_BENCHMARK.md.
* ``recall_at_k`` / ``mrr`` — standard retrieval metrics against gold sets.
"""

from __future__ import annotations


def percentile(data: list[float], p: float) -> float:
    """Linear-interpolated percentile (LATENCY_BENCHMARK.md definition)."""
    if not data:
        raise ValueError("percentile requires non-empty data")
    ordered = sorted(data)
    k = (len(ordered) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold ids found in the top-k of ``ranked``."""
    if not gold:
        return 0.0
    found = set(ranked[:k]) & gold
    return len(found) / len(gold)


def mrr(ranked: list[str], gold: set[str]) -> float:
    """Mean reciprocal rank — 1/first-gold-rank if found, else 0."""
    for rank, cid in enumerate(ranked, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0
