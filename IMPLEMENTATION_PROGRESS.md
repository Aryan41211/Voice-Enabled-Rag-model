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
- [x] CI test skeleton (GitHub Actions) — `.github/workflows/ci.yml`

## Day 2 — Chunking & Indexing
- [x] Pipeline schemas per API.md (`app/harness/schemas.py`)
- [x] 5 chunking strategies (`app/ingestion/chunking.py`)
- [x] Embedding wrapper (`app/ingestion/embed.py`)
- [x] FAISS + BM25 index build per strategy (`app/ingestion/build_index.py`)
- [x] Retrieval eval script (Recall@k, MRR) — `benchmarks/run_retrieval_eval.py`

## Day 3 — Retrieval Evaluation & Hybrid Search
- [x] Dense + sparse + hybrid (RRF) retrievers (`app/retrieval/`)
- [x] Run eval across strategies, pick winner → EVALUATION.md (winner: metadata-aware + dense, MRR 0.452)
- [x] Cross-encoder rerank ablation → EVALUATION.md (gains R@5 +8.4 pts but +345 ms; excluded from live path)

## Day 4 — Generation & Guardrails
- [x] Generation stage: extractive fallback + hosted LLM, forced citation (`app/generation/generator.py`)
- [x] 3-layer guardrail stack (`app/guardrails/guardrails.py`) — input (garbage/unsafe/off-topic), retrieval (score floor + isolation margin), output (citation/groundedness)
- [x] Guardrail unit tests + adversarial set (`tests/adversarial/`) with expected refusal actions
- [x] Retrieval guardrail threshold calibrated on eval gold (margin ≥ 0.03 → refuse) → EVALUATION.md

## Day 5 — Harness & STT Integration
- [x] Orchestrator: typed I/O, timeouts, retries, circuit breaker, tracing (`app/harness/pipeline.py`)
- [x] STT client (Sarvam WS + REST fallback) + FakeSTT + tests (`app/stt/client.py`)
- [x] End-to-end text pipeline wired + smoke-tested

## Day 6 — Latency Benchmark, Deployment & Hardening
- [x] P50/P70/P100 on 110 real queries → LATENCY_BENCHMARK.md (retrieval P50 15.2 / P70 19.4 / P100 40.2 ms)
- [x] FastAPI backend (`app/api/server.py`: /health, /query, /v1/voice) + browser voice UI at `/` + API tests
- [x] Dockerfile + docker-compose.yml + entrypoint + .dockerignore + CI
- [x] Live link via free cloudflared tunnel (`scripts/start_demo.ps1`) — verified /health + /query over public https
- [~] HF Spaces deploy staged (`space/`) but Docker Spaces now need PRO subscription — tunnel chosen instead

## Day 7-8 — Demo Prep & Submission
- [x] Real numbers in EVALUATION.md / LATENCY_BENCHMARK.md
- [x] Final validation sweep (tests, lint, secrets, TODOs, docs)
- [~] Fresh-clone dry-run (README Option B) — in progress
- [~] Release: tag + GitHub Release + CHANGELOG — in progress

## Environment
- GPU: RTX 4050 Laptop (CUDA available), 16.8 GB RAM
- Embedding model: `intfloat/multilingual-e5-small` (384-d) — speed/quality sweet spot for a hackathon corpus
