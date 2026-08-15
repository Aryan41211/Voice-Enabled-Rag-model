# Changelog

All notable changes to this project during the HH Goa 2026 sprint. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Added
- Initial project documentation set (README, ARCHITECTURE, CHUNKING_STRATEGY, GUARDRAILS, LATENCY_BENCHMARK, EVALUATION, API, TESTING, DEPLOYMENT, ROADMAP, SUBMISSION_CHECKLIST, CONTRIBUTING)
- Shared pipeline schemas (`app/harness/schemas.py`) per API.md
- 5 chunking strategies (`app/ingestion/chunking.py`): fixed+overlap, semantic, sentence-window, metadata-aware, hierarchical
- FAISS + BM25 index builder (`app/ingestion/build_index.py`) per strategy
- Dense / sparse / hybrid (RRF) retrievers (`app/retrieval/`)
- Retrieval evaluation script (Recall@3/@5, MRR) + results in EVALUATION.md
- Selection: **metadata-aware chunking + dense retrieval** (MRR 0.452, R@5 0.709, retrieval p50 ≈ 17 ms)
- Cross-encoder reranker (`app/retrieval/rerank.py`) — ablation in EVALUATION.md
- Generation stage (`app/generation/`): extractive fallback + optional streaming LLM with TTFT + forced citations
- 3-layer guardrail stack (`app/guardrails/`): input (garbage/unsafe/off-topic), retrieval (score floor + isolation margin), output (citation + groundedness)
- Guardrail adversarial test set (`tests/adversarial/`)

### Planned
- STT integration (Sarvam `saaras:v3-realtime`)
- Generation with forced citation + streaming TTFT measurement
- 3-layer guardrail stack
- Harness with retries/timeouts/circuit breaker
- Latency benchmark run on ≥100 real queries
- Live deployment

---

## How to use this file
Add an entry under `[Unreleased]` every time you merge something notable to `dev`/`main`. On submission day, rename `[Unreleased]` to `[1.0.0] - 2026-08-22` (or your actual submit date) so the repo has a clear "this is what we shipped" marker for judges.

Example future entry:
```
## [1.0.0] - 2026-08-22
### Added
- Full voice-to-answer pipeline live at <url>
- Hybrid retrieval (dense + BM25 + RRF), semantic chunking selected as primary strategy
- P50/P70/P100 latency benchmarks (see EVALUATION.md)
### Fixed
- STT WebSocket reconnect bug under intermittent network conditions
### Known Issues
- <anything you're disclosing rather than hiding>
```
