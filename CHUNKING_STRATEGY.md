# Chunking & Retrieval Strategy

The brief explicitly penalizes "a single naive fixed-size chunking approach." This doc defines **five** strategies to implement, compare, and pick a winner (or ensemble) from — with evidence, not vibes.

## Why chunking is non-trivial here
`MSMARCO-XI` passages are already short, self-contained snippets (typical MS MARCO passage), so naive re-chunking risks either (a) doing nothing useful because passages are already small, or (b) fragmenting them and destroying context. Design chunking around **combining/organizing** short passages meaningfully, not just splitting long ones.

## Strategy 1 — Fixed-size with overlap (baseline)
- Chunk size: ~150–250 tokens, overlap: ~20–30 tokens.
- Purpose: baseline to benchmark everything else against.
- Weakness: arbitrary boundaries, no semantic awareness — expect this to be your worst performer; that's the point of including it.

## Strategy 2 — Semantic chunking
- Split on sentence boundaries, then merge adjacent sentences into a chunk while cosine similarity between consecutive sentence embeddings stays above a threshold; start a new chunk on a semantic "break."
- Purpose: keeps topically coherent spans together instead of cutting mid-idea.
- Implementation: `sentence-transformers` embeddings + a sliding similarity check (the "semantic chunking" pattern from LlamaIndex/LangChain, reimplemented directly — don't hide it inside a framework call in your report, show the logic).

## Strategy 3 — Sentence-window retrieval
- Index individual sentences, but retrieve with their surrounding N sentences ("window") as context sent to the LLM.
- Purpose: precise retrieval matching (small unit = better recall for exact-fact queries) + enough context for coherent generation.

## Strategy 4 — Metadata-aware chunking
- Attach metadata to every chunk at index time: `language`, `source_query_id`, `passage_is_selected` (check the MSMARCO-XI schema for a "does this passage answer the query" label, and use it if present), `chunk_strategy`.
- Purpose: enables **filtered retrieval** (e.g. restrict to the detected query language) and lets you measure whether metadata filtering improves precision — report the delta.

## Strategy 5 — Hierarchical / parent-child chunking
- Small child chunks (sentence-level) used for the actual similarity search (better matching), but when a child is retrieved, return its **parent** (the full original passage or a passage cluster) to the LLM for generation.
- Purpose: best of both — precise retrieval, full context for grounded generation.

## Hybrid Retrieval (applies across all strategies)
- Combine **dense** (embedding ANN) + **sparse** (BM25) results via Reciprocal Rank Fusion:
  `score(d) = Σ 1 / (k + rank_i(d))` across each retriever's ranked list, `k≈60`.
- Optionally re-rank the fused top-N with a cross-encoder — report the latency cost vs. accuracy gain explicitly (this is exactly the kind of tradeoff analysis the brief is asking for).

## Evaluation Plan
For each strategy (and the hybrid retriever), measure on a held-out sample of MSMARCO-XI queries:
- **Retrieval accuracy**: Recall@k / MRR against the passage(s) MS MARCO marks as the answer source.
- **Latency**: index build time (offline, not counted) + per-query retrieval time (counted).
- **Chunk count / index size**: more chunks ≠ better; report the tradeoff.

Record results in `EVALUATION.md` using the table there, and pick the winner for your live demo based on that data — say so explicitly in your README/demo. That explicit, evidence-backed decision is the "real thought" the evaluators are scoring for.
