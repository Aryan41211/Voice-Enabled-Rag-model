# Implementation Progress

Live tracker for the autonomous execution loop. Mirrors ROADMAP.md with
machine-checkable status.

## Legend
- [x] done & verified
- [~] in progress
- [ ] not started

## Day 1 — Setup & Data (Aug 15)
- [x] Repo scaffold, `.env.example`, `.gitignore`, requirements
- [x] Language picked (`hi`) + MSMARCO-XI downloaded (`data/raw/validation/hinval.parquet`, 97,941 rows) & schema explored
- [x] Dataset loader + CLI + unit tests (`app/ingestion/dataset.py`)
- [ ] CI lint/test skeleton (GitHub Actions)

## Day 2 — Chunking & Indexing
- [~] Pipeline schemas per API.md (`app/harness/schemas.py`)
- [ ] 5 chunking strategies (`app/ingestion/chunking.py`)
- [ ] Embedding wrapper (`app/ingestion/embed.py`)
- [ ] FAISS + BM25 index build per strategy (`app/ingestion/build_index.py`)
- [ ] Retrieval eval script (Recall@k, MRR) — `benchmarks/run_retrieval_eval.py`

## Day 3 — Retrieval Evaluation & Hybrid Search
- [ ] Dense + sparse + hybrid (RRF) retrievers (`app/retrieval/`)
- [ ] Run eval across strategies, pick winner → EVALUATION.md
- [ ] Optional cross-encoder rerank + latency cost

## Day 4 — Generation & Guardrails
- [ ] Generation stage: extractive fallback + hosted LLM, forced citation
- [ ] 3-layer guardrails + adversarial test set
- [ ] Guardrail unit tests

## Day 5 — Harness & STT Integration
- [ ] Orchestrator: typed I/O, timeouts, retries, circuit breaker, tracing
- [ ] STT client (Sarvam) + mock tests + fallback
- [ ] End-to-end text pipeline wired

## Day 6 — Latency Benchmark, Deployment & Hardening
- [ ] P50/P70/P100 on >=100 queries → LATENCY_BENCHMARK.md
- [ ] FastAPI backend + minimal frontend
- [ ] Dockerfile + docker-compose.yml
- [ ] Deploy live link (needs cloud credentials)

## Day 7-8 — Demo Prep & Submission
- [ ] Real numbers in EVALUATION.md / LATENCY_BENCHMARK.md
- [ ] Final validation sweep (tests, lint, secrets, TODOs, docs)

## Environment
- GPU: RTX 4050 Laptop (CUDA available), 16.8 GB RAM
- Embedding model: TBD (e5-small for speed vs bge-m3 for quality)
