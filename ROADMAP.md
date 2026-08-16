# Roadmap — 8 Days (Aug 15 → Aug 22, 11:59 PM IST)

No resubmissions allowed — build with a "final by Day 7, polish on Day 8" mindset, not "finish on Day 8."

## Day 1 (Aug 15) — Setup & Data
- [x] Repo scaffold (structure from README §7), `.env.example`, CI lint/test skeleton
- [x] Pick target language(s) from MSMARCO-XI, download & explore schema
- [x] Get Sarvam API key, test a basic streaming STT call end-to-end
- [ ] Assign owners to: STT, Retrieval/Chunking, Guardrails/Generation, Harness/Infra, Demo/Video

## Day 2 (Aug 16) — Chunking & Indexing
- [x] Implement all 5 chunking strategies (CHUNKING_STRATEGY.md)
- [x] Build FAISS + BM25 indexes per strategy
- [x] Write the retrieval eval script (Recall@k, MRR) against MSMARCO-XI's labeled answer passages

## Day 3 (Aug 17) — Retrieval Evaluation & Hybrid Search
- [x] Run eval across all strategies, pick winner(s)/ensemble, log results in EVALUATION.md
- [x] Implement hybrid dense+BM25 fusion (RRF)
- [x] Optional: cross-encoder re-rank, measure latency cost

## Day 4 (Aug 18) — Generation & Guardrails
- [x] Wire generation LLM (fast-inference backend), prompt template with forced citation
- [x] Implement Layer 1–3 guardrails (GUARDRAILS.md)
- [x] Build the guardrail adversarial test set (TESTING.md §3), write unit tests

## Day 5 (Aug 19) — Harness & STT Integration
- [x] Orchestrator: structured I/O schemas (API.md), timeouts, retries, circuit breaker, tracing
- [x] Full voice → transcript → retrieval → guardrail → generation → answer wired end-to-end
- [x] STT fallback path (WebSocket fail → REST batch)
- [x] Integration tests (TESTING.md §2)

## Day 6 (Aug 20) — Latency Benchmarking, Deployment & Hardening
- [x] Run the full latency benchmark script (LATENCY_BENCHMARK.md) on ≥100 real queries
- [x] Fix the biggest latency offenders (usually: rerank, network hops, cold embedding model)
- [x] Deploy live link per DEPLOYMENT.md, verify WebSocket support on your host
- [x] Freeze feature scope — **no new features after today**

## Day 7 (Aug 21) — Demo Prep & Polish
- [ ] Record Video 1 (process, 90s) — show the team actually working, not the product
- [ ] Record Video 2 (full demo) — include a live "guardrail refuses to answer" moment, live latency numbers on screen
- [x] Finalize README + EVALUATION.md with real numbers filled in (not placeholders)
- [x] Internal dry-run: fresh clone → README Option B works end-to-end (venv install, tiny index build, 6/6 smoke, bench, server boot + /health /query /)
- [ ] Live link tested from a different network (needs a phone — user task)

## Day 8 (Aug 22) — Submission Day
- [~] Final GitHub push, tag a release, update CHANGELOG.md (in progress this sweep)
- [ ] Confirm live link is stable (test from a different network/device one more time)
- [ ] Upload both videos to Instagram, X, LinkedIn — **every team member individually**, `#RAGInGoa` on every post, ≥1 public Instagram account
- [ ] Fill submission form: https://forms.gle/MNvCjcv23Hn2Eeu58
- [ ] Submit **once**, after everything above is verified — no resubmissions allowed

## Ownership Template (fill in)

| Workstream | Owner | Backup |
|---|---|---|
| STT integration | | |
| Chunking & indexing | | |
| Retrieval & re-ranking | | |
| Generation & prompting | | |
| Guardrails | | |
| Harness/orchestration | | |
| Latency benchmarking | | |
| Deployment / live link | | |
| Testing | | |
| Videos & promotion | | |
