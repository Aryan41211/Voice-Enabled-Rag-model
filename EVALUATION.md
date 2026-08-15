# Evaluation

This is where "we tried multiple approaches" turns into evidence. Fill in real numbers before submission — a table of blanks is worse than no table.

## 1. Retrieval Quality — Chunking Strategy Comparison

Sample: `<N>` queries from MSMARCO-XI (`<language>`), seed `<seed>`. Ground truth: MS MARCO's labeled relevant passage(s) per query.

| Strategy | Recall@3 | Recall@5 | MRR | Avg chunks/doc | Index build time |
|---|---|---|---|---|---|
| Fixed-size + overlap | | | | | |
| Semantic chunking | | | | | |
| Sentence-window | | | | | |
| Metadata-aware | | | | | |
| Hierarchical (parent-child) | | | | | |
| **Hybrid (dense + BM25, best strategy)** | | | | | |

**Winner:** `<strategy name>` — reasoning: `<why, tied to the numbers above>`

## 2. Re-ranking Ablation (if implemented)

| Config | Recall@5 | MRR | Added latency (ms) | Worth it? |
|---|---|---|---|---|
| No re-rank | | | 0 | — |
| Cross-encoder re-rank | | | | Y/N + why |

## 3. Answer Quality (small manual eval, ~20–30 queries)

Rate each on a 1–5 scale, or pass/fail — pick one and be consistent.

| Metric | Definition | Result |
|---|---|---|
| Groundedness | Does every claim in the answer trace back to a retrieved passage? | |
| Correctness vs. gold answer | Does it match/entail the MS MARCO gold answer? | |
| Appropriate refusal rate | Of queries the system *should* refuse (off-topic/ungrounded test set), % correctly refused | |
| False refusal rate | Of answerable queries, % incorrectly refused | |

## 4. Latency Summary (pulled from LATENCY_BENCHMARK.md)

| Measurement | P50 | P70 | P100 |
|---|---|---|---|
| Retrieval only | | | |
| Retrieval + TTFT | | | |
| Retrieval + full generation | | | |
| STT | | | |

## 5. Known Limitations
- `<e.g. only tested on Hindi subset, generalization to other 12 languages not verified>`
- `<e.g. guardrail off-topic classifier has X% false-positive rate on ambiguous queries>`
- `<be honest here — judges notice fabricated perfection faster than they notice honest gaps>`
