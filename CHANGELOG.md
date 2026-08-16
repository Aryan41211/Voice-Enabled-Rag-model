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
### Fixed
- `python-multipart` missing from `requirements.txt` — `/v1/voice` (multipart upload route) failed to boot on fresh installs; caught by the fresh-clone dry-run and fixed

### Added
- Fresh-clone dry-run verified (README Option B end-to-end on a clean checkout): venv install → tiny index build → 6/6 gold smoke queries → latency bench → server boots → `/health`, `/query`, `/` all 200
- Live link re-verified over a public tunnel: real Sarvam voice round-trip (exact transcript, ~994 ms total), Hindi text query answered, out-of-scope query refused honestly
- `v0.1.0` release tagged + GitHub Release published
- Test coverage raised to ~80% (116 tests, up from 90/70%): on-disk index load path, `Pipeline.from_index` wiring, STT retry+backoff and hard-failure refusals, circuit-breaker trip + success reset, generation error/fallback paths, uncited-answer fail-closed fallback, semantic batched chunking, guardrail edge cases
- Docs synced to implemented reality: `API.md` timings schema, `ARCHITECTURE.md` (actual STT flow, dense-only rationale), LICENSE holder filled

---

## [0.1.0] - 2026-08-15
### Added
- STT client (`app/stt/client.py`): Sarvam WebSocket realtime (`saaras:v3-realtime`) with REST fallback (`saaras:v3`), WAV→PCM conversion, language-code map; `FakeSTT` for keyless dev
- Harness orchestrator (`app/harness/pipeline.py`): typed I/O, per-stage retries + backoff, circuit breaker, timeouts, refusal→`refused:true` mapping, LLM→extractive degradation, `warmup()` for latency-stable measurement
- FastAPI server (`app/api/server.py`): `GET /health`, `POST /query` (text), `POST /v1/voice` (WAV audio → STT → pipeline), startup warm-up, Swagger UI
- Retrieval guardrail calibrated on real data: isolation margin (top1 − rank-20 cosine) ≥ 0.03 → refuse; ambiguity-gap check disabled (no signal); embedding-centroid off-topic check removed (provably useless) in favor of keyword gate
- Expanded unsafe-keyword coverage incl. self-harm variants + standalone `बम`
- Latency benchmark (`benchmarks/run_latency_bench.py`) + results: retrieval P50 15.2 / P70 19.4 / P100 40.2 ms on 110 real queries → LATENCY_BENCHMARK.md
- Dockerfile + docker-compose.yml + `scripts/entrypoint.py` (auto-builds index on first boot) + `.dockerignore`
- GitHub Actions CI (`ci.yml`) running the full 90-test suite on Ubuntu (no model downloads in tests)
- API tests (`tests/integration/test_api.py`), eval script bootstrap + warm-up fix
- Browser voice UI served at `/` (self-contained HTML: mic → 16 kHz WAV → `/v1/voice`, text fallback → `/query`)
- Live demo launcher (`scripts/start_demo.ps1`): starts the API and opens a free cloudflared https tunnel — public link verified end-to-end
- HF Spaces deploy staged (`space/` + `scripts/deploy_space.py`); Docker Spaces now require HF PRO, so the tunnel is the demo path

### Notes
- Chosen stack: `metadata` chunking + dense-only retrieval (hybrid/RRF hurt Hindi retrieval; reranker excluded — too slow for budget)
- Live STT/hosted-LLM paths are key-gated (`STT_PROVIDER`, `LLM_PROVIDER`); extractive generation is the offline default
- CI fully green on GitHub Actions: lint (ruff) + 116 tests (~80% coverage) + real-stack e2e smoke + **real Sarvam STT round-trip** (hi-IN, exact transcript match) using the `SARVAM_API_KEY` repo secret

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
