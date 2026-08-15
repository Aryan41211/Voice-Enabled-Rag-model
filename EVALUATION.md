# Evaluation

This is where "we tried multiple approaches" turns into evidence. Fill in real numbers before submission — a table of blanks is worse than no table.

## 1. Retrieval Quality — Chunking Strategy Comparison

Sample: **300 queries** from MSMARCO-XI (`hi`, validation), seed **42**, corpus of **1,500 queries (~15K passages)**. Ground truth: MS MARCO's labeled relevant passage(s) per query. Embedding model: `intfloat/multilingual-e5-small`. Dense = FAISS inner-product; Hybrid = RRF fusion of dense + BM25.

| Strategy | Recall@3 | Recall@5 | MRR | Hybrid R@5 | Hybrid MRR | Avg chunks/doc | Index build time |
|---|---|---|---|---|---|---|---|
| Fixed-size + overlap | 0.583 | 0.709 | 0.449 | 0.594 | 0.391 | 10.1 | 65 s |
| Semantic chunking | 0.583 | 0.702 | 0.442 | 0.606 | 0.399 | 10.8 | 197 s |
| Sentence-window | 0.511 | 0.601 | 0.404 | 0.514 | 0.328 | 35.1 | 125 s |
| Metadata-aware | **0.583** | **0.709** | **0.452** | 0.601 | 0.397 | 10.0 | 115 s |
| Hierarchical (parent-child) | 0.511 | 0.601 | 0.404 | 0.514 | 0.328 | 35.1 | 137 s |
| **Hybrid (dense + BM25, on winner)** | — | 0.601 | 0.397 | — | — | — | — |

**Winner:** `metadata` (metadata-aware, passage-level) chunking **with dense-only retrieval** — reasoning: MS MARCO passages are short, self-contained snippets, so passage-level chunks (metadata = full passage + metadata) preserve the whole answer signal, while sentence-level strategies (sentence-window, hierarchical) fragment it and drop Recall@5 by ~11 points. `fixed` and `semantic` converge to the same passage-level units for these short texts but carry no filtering metadata. MRR is best for `metadata` (0.452).

Notable: **BM25 fusion *hurt*** dense-only results (MRR 0.452 → 0.397). On Hindi, surface-form matching is weak (morphology/transliteration), so RRF down-ranks good dense hits with poor sparse ones. The live demo therefore uses **dense retrieval over metadata chunks**; the hybrid path stays available behind a toggle.

## 2. Re-ranking Ablation

Cross-encoder: `BAAI/bge-reranker-v2-m3` over the top-20 dense candidates (60 queries).

| Config | Recall@5 | MRR | Added latency (ms) | Worth it? |
|---|---|---|---|---|
| No re-rank | 0.719 | 0.476 | 0 | — |
| Cross-encoder re-rank | 0.803 | 0.574 | ~345 (p50 total 359 ms) | **No for live demo** — +8.4 R@5 pts and +0.10 MRR are real, but p50 total ≈ 360 ms blows the sub-200 ms budget. Keep as a toggle/offline-only stage. |

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
| Retrieval only | 16.8 ms | 17.7 ms | 52.3 ms |
| Retrieval + TTFT | | | |
| Retrieval + full generation | | | |
| STT | | | |

## 5. Known Limitations
- Retrieval corpus is a 1,500-query sample (~15K passages) of the Hindi validation split — generalization to other 12 languages and the full corpus is not yet verified.
- Hybrid (dense + BM25) hurt results on this Hindi subset; BM25 is known to be weak on highly inflected Hindi. Re-evaluate if supporting a more analytic language (e.g. English source passages).
- Semantic chunking adds ~98 s of offline build time for no retrieval gain at this corpus size (passages are already short).
