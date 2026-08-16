# Architecture

## 1. Pipeline Stages

### Stage 0 — Offline Indexing (not part of the latency budget)
1. Load `ai4bharat/MSMARCO-XI` (pick language(s) your demo will support — start with 1, add more if time allows).
2. Apply **all** chunking strategies from `CHUNKING_STRATEGY.md` to build multiple candidate indexes.
3. Embed chunks with a multilingual embedding model.
4. Build a FAISS HNSW (or Qdrant) index per strategy, with metadata (source query id, language, chunk strategy, position).
5. Also build a BM25 sparse index over the same chunks for hybrid retrieval.

### Stage 1 — Voice Input → Transcript
- Browser mic → `MediaRecorder` → decoded + downsampled to 16 kHz mono PCM WAV in JS (`app/api/static/index.html`) → uploaded to `POST /v1/voice`.
- Server converts the WAV to raw linear16 PCM and streams it to Sarvam `saaras:v3-realtime` over WebSocket (`vad` endpointing, `stream_type=balanced`).
- Fallback: if the Sarvam WebSocket fails or times out, fall back to Sarvam's REST batch STT endpoint (harness responsibility, see §3).
- Only the **final** transcript goes downstream into retrieval/guardrails.

### Stage 2 — Query Guardrail (pre-retrieval)
- Cheap, fast checks *before* spending retrieval/generation budget:
  - Empty/garbage/too-short transcript → ask user to repeat.
  - Unsafe-content keyword gate + out-of-scope domain keyword gate → refuse.
  - Retrieval-side (Layer 2) scoring checks: top-score floor and top-1-vs-background isolation margin, calibrated on eval gold → refuse when nothing relevant.
- An embedding-based off-topic classifier was built and **removed after evaluation proved it useless** (see GUARDRAILS.md / EVALUATION.md §6).
- See `GUARDRAILS.md` for full detail.

### Stage 3 — Retrieval
- Embed the query with the same embedding model used at indexing.
- **Dense ANN retrieval (FAISS)** over `metadata`-aware passage chunks is the live path — it won the 300-query eval (MRR 0.452, R@5 0.709).
- Hybrid dense+BM25 (RRF) and cross-encoder re-rank were implemented and **measured, then excluded from the live path**: RRF *hurt* Hindi retrieval (MRR 0.452 → 0.397) and the reranker added ~345 ms for +8.4 R@5 — both stay available behind toggles (`hybrid`, `rerank`).
- Return passages + metadata + retrieval scores.

### Stage 4 — Generation
- Prompt template that **forces citation** to retrieved passage IDs and instructs the model to say "I don't have enough information" if passages are insufficient.
- Stream tokens; measure **time-to-first-token (TTFT)** and full completion time separately.
- Use a fast-inference backend (Groq / Cerebras / local small model) if you want any real shot at a low end-to-end number.

### Stage 5 — Answer Guardrail (post-generation)
- Groundedness check: does the answer's claims overlap/entail from the retrieved passages? (embedding similarity or NLI model)
- If grounding score < threshold → don't return the generated answer; return "not grounded, cannot answer confidently" and show the closest passage instead.
- See `GUARDRAILS.md`.

## 2. Latency Budget (target, per stage — tune after benchmarking)

| Stage | Budget | Notes |
|---|---|---|
| Query embedding | ~10–20ms | Small model, batch size 1 |
| ANN search | ~5–30ms | Depends on index size/type (HNSW ef_search) |
| BM25 search | ~5–15ms | In-memory index |
| Fusion/merge | ~1–2ms | Pure Python, negligible |
| Re-rank (optional) | ~30–80ms | **Cut first** if you're over budget |
| Guardrail checks | ~5–15ms | Rule-based + small classifier, not another LLM call if avoidable |
| Generation (TTFT only) | ~50–150ms | Depends entirely on inference backend — this is the hard part |
| **Total (retrieval + TTFT)** | **~150–250ms** | Report honestly; see README §2 |

STT latency is reported **separately** — it's a distinct upstream stage with its own claimed <150ms TTFT (Sarvam), not part of the "chunking→retrieval→output" budget the brief describes.

**Measured (110 real Hindi queries, extractive generation):** retrieval-only P50 15.2 / P70 19.4 / P100 40.2 ms; retrieval+full generation P50 15.0 ms. Real voice round-trip over the live link: ~1.0 s total (STT-bound). Full tables in `LATENCY_BENCHMARK.md`.

## 3. Harness (Orchestration Layer)

This is what makes it a "real pipeline" instead of a script. Implement as an explicit orchestrator (plain `asyncio` state machine, or LangGraph/LlamaIndex workflow if your team prefers a framework):

- **Structured I/O**: every stage takes/returns a typed schema (pydantic models) — `Transcript`, `RetrievedChunk[]`, `GuardrailResult`, `Answer` — not raw strings passed around. See `API.md` for the exact schemas.
- **Timeouts** per stage (STT, retrieval, generation) so one slow call doesn't hang the whole request.
- **Retries with backoff** on transient failures (STT websocket drop, LLM API 5xx) — cap at 2 retries, then fail gracefully.
- **Circuit breaker** on the LLM/STT provider if it's erroring repeatedly — fall back to a secondary path (e.g. cached "service unavailable, try again" response) rather than hanging.
- **Tracing/logging**: log stage-by-stage timings for every request — this *is* your latency benchmark data source, don't build it separately.
- **Error recovery**: if retrieval returns 0 results, don't call the LLM at all — go straight to the guardrail's "cannot answer" path.

## 4. Component Choices — Rationale

- **Sarvam over ElevenLabs**: brief allows either; Sarvam is purpose-built for Indian languages and pairs naturally with MSMARCO-XI's Indic language set — an easier story to tell in your demo video than routing Indic audio through a Western-language-first STT.
- **FAISS over a hosted vector DB for the demo**: no network hop = lower latency, no external dependency to fail during judging. Use Qdrant only if you need persistence/filtering across restarts and are comfortable running it locally/dockerized (still on localhost, not a remote hosted instance, to protect latency).
- **Dense-only over hybrid (dense+BM25)**: expected MS MARCO to be lexical-heavy, but our 300-query Hindi eval showed RRF *hurting* dense-only results (MRR 0.452 → 0.397) — surface-form matching is weak in Hindi (morphology/transliteration), so sparse hits down-rank good dense ones. Dense-only is the live path; hybrid stays behind a toggle. See EVALUATION.md.
