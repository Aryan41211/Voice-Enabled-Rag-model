# Changelog

All notable changes to this project during the HH Goa 2026 sprint. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
### Added
- Initial project documentation set (README, ARCHITECTURE, CHUNKING_STRATEGY, GUARDRAILS, LATENCY_BENCHMARK, EVALUATION, API, TESTING, DEPLOYMENT, ROADMAP, SUBMISSION_CHECKLIST, CONTRIBUTING)

### Planned
- STT integration (Sarvam `saaras:v3-realtime`)
- 5 chunking strategies + hybrid dense/BM25 retrieval
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
