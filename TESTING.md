# Testing

A hackathon repo with zero tests reads as "we didn't have time to make sure it works." A short, targeted test suite is cheap insurance against a live-demo failure in front of judges.

## 1. Unit Tests (`tests/unit/`)

| Component | What to test |
|---|---|
| Chunking strategies | Each strategy produces expected chunk boundaries on a fixed sample input; overlap counts are correct |
| RRF fusion | Given known dense/sparse rankings, fused order matches hand-computed expectation |
| Guardrail rules | Each rule in `GUARDRAILS.md` has ≥1 test that triggers it and ≥1 that correctly passes through |
| Schema validation | `API.md` pydantic models reject malformed input |
| Percentile function | `LATENCY_BENCHMARK.md` script's `percentile()` matches known values (e.g. `numpy.percentile`) |

## 2. Integration Tests (`tests/integration/`)

| Flow | What to test |
|---|---|
| End-to-end (text query, bypassing STT) | Query in → answer or refusal out, with sources populated |
| STT → Retrieval handoff | A pre-recorded audio clip transcribes and successfully triggers retrieval |
| Retry/timeout behavior | Mock a slow/failing STT or LLM call, confirm the harness retries then fails gracefully (not hangs) |
| Circuit breaker | Simulate repeated provider failures, confirm fallback path activates |

## 3. Guardrail Adversarial Test Set (`tests/adversarial/`)

Build a small fixed set (~15–20 queries) covering:
- Off-topic questions (e.g. general chit-chat unrelated to the dataset domain)
- Queries with no relevant passages in the index
- Ambiguous queries where retrieval scores are flat/low-confidence
- Unsafe/inappropriate input
- Empty or nonsense transcripts (simulating bad STT output)

Assert the expected `GuardrailResult.action` for each. This set doubles as your demo video's refusal examples — reuse it.

## 4. Latency Regression Check
- Run `benchmarks/run_latency_bench.py` on a small sample (~20 queries) as part of CI/pre-submit check.
- Fail (or at least flag loudly) if P50 retrieval-only latency regresses by more than ~30% from the last recorded baseline in `EVALUATION.md`.

## 5. Manual Pre-Demo Checklist
- [ ] Fresh clone + documented setup steps actually work, on a machine that isn't the dev's own laptop
- [ ] Live link accessed from a different network (e.g. phone hotspot) — catches localhost-only bugs
- [ ] Full run-through of the demo script including at least one guardrail refusal, timed
- [ ] Audio input tested with a real microphone, not just pre-recorded files
