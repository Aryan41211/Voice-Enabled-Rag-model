# 🎙️ Voice-RAG — Voice-Enabled RAG Model
**HH Goa 2026 · Task #2** · Deadline: **Aug 22, 2026, 11:59 PM IST**

Speak a question → get a grounded, cited answer. A full voice-to-answer pipeline: **STT → engineered chunking/retrieval → guardrailed generation**, running inside a real harness, benchmarked, and fast.

```
🎤 Voice → 📝 Transcript → 🔍 Retrieve → 🛡️ Guardrail → 💬 Answer
```

---

## 1. Requirement → Deliverable Map

| # | Requirement (from task brief) | How we satisfy it | Doc |
|---|---|---|---|
| 1 | STT via Sarvam **or** ElevenLabs | Sarvam `saaras:v3-realtime` WebSocket streaming STT | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| 2 | Chunking must be "vast" — multiple strategies | Fixed+overlap, semantic, sentence-window, metadata-aware, hierarchical, hybrid BM25+dense | [CHUNKING_STRATEGY.md](./CHUNKING_STRATEGY.md) |
| 3 | Full pipeline < 200ms | Precomputed embeddings + ANN index + fast LLM w/ TTFT measurement (see honest caveat below) | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| 4 | P50/P70/P100 latency, real queries | Benchmark harness + methodology + report template | [LATENCY_BENCHMARK.md](./LATENCY_BENCHMARK.md) |
| 5 | Proper harness (retries, structured I/O, error recovery) | Orchestrator layer w/ typed schemas, retry/backoff, circuit breaker | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| 6 | Guardrail your model | Multi-layer guardrail stack, refuse-to-answer policy | [GUARDRAILS.md](./GUARDRAILS.md) |
| — | GitHub repo + live link + 2 videos + form | Submission checklist, `#RAGInGoa` on every post | [SUBMISSION_CHECKLIST.md](./SUBMISSION_CHECKLIST.md) |
| — | 8-day execution plan | Day-by-day roadmap | [ROADMAP.md](./ROADMAP.md) |
| — | Test coverage & QA | Test plan by layer | [TESTING.md](./TESTING.md) |
| — | How to deploy the live link | Deployment options + latency tradeoffs | [DEPLOYMENT.md](./DEPLOYMENT.md) |
| — | API contract between stages | Request/response schemas | [API.md](./API.md) |
| — | Reporting retrieval/answer quality | Results table template | [EVALUATION.md](./EVALUATION.md) |
| — | Version history | Release log | [CHANGELOG.md](./CHANGELOG.md) |

---

## 2. ⚠️ Read this before you build: the 200ms target, honestly

The brief says "the full process — chunking + vector DB retrieval + everything through to final output — should complete in under 200ms." Two things to get right, or you will fail this requirement without realizing it:

- **"Chunking" at query time ≠ chunking at index time.** You cannot re-chunk the corpus on every query. Chunking/embedding/indexing happens **offline, once**, at ingestion. What runs at query time — and what your 200ms budget actually covers — is: query embedding → ANN search → (optional) re-rank → guardrail checks → generation. State this explicitly in your submission so evaluators know you understood the distinction.
- **"Final output" for an LLM is ambiguous — decide which one you mean, disclose it, then measure it.** A full generated paragraph from *any* hosted LLM (GPT, Claude, Gemini, Sarvam LLM) essentially never completes in 200ms — network + prefill + decode for even 100 output tokens is typically 500ms–2s+. What **is** realistically achievable in ~200ms:
  - Retrieval-only latency (embed query + ANN search + rerank): sub-100ms with a modest local corpus and FAISS/HNSW.
  - **Time-to-first-token (TTFT)** of a streamed answer on a fast inference stack (Groq/Cerebras-class hardware, or a small local model) can land retrieval+TTFT combined under 200ms.
  - A **full non-LLM extractive answer** (return the best-ranked passage/sentence span directly, no generation) trivially hits <200ms.

**Recommended approach:** report latency for **both** interpretations transparently — (a) retrieval pipeline alone, and (b) retrieval + TTFT of a streamed answer — and state clearly in your README/demo which one hits sub-200ms and why full multi-sentence generation cannot honestly claim to. Judges will trust a well-reasoned, honest breakdown far more than a suspicious "180ms end-to-end including generation" claim that can't survive a follow-up question.

---

## 3. Pipeline Shape

```
┌──────────┐   ┌───────────────┐   ┌─────────────────────┐   ┌───────────────┐   ┌──────────┐
│  Mic /   │──▶│  STT (Sarvam  │──▶│  Query Guardrail     │──▶│  Retrieval    │──▶│Generation│
│  Audio   │   │  saaras v3)   │   │  (topic/safety check)│   │ (hybrid dense │   │ + Answer │
│  Input   │   │  streaming    │   │                      │   │ + BM25 + rerank)│  │Guardrail │
└──────────┘   └───────────────┘   └─────────────────────┘   └───────────────┘   └──────────┘
                                                                        │
                                                              ┌─────────▼─────────┐
                                                              │  Vector DB (FAISS │
                                                              │  / Qdrant) built  │
                                                              │  offline from     │
                                                              │  MSMARCO-XI       │
                                                              └────────────────────┘
```

## 4. Dataset

[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — a multilingual (13 Indian languages) translated variant of MS MARCO, with `query`, `answers`, and `passages` fields per example. Use it as both the retrieval corpus (index the `passages`) and as a natural eval set (the `query`/`answers` pairs double as test queries and gold answers for the latency + accuracy benchmark).

```python
from datasets import load_dataset
ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")  # swap "hi" for any of the 13 language codes
```

## 5. Quick Start

```bash
git clone https://github.com/Aryan41211/Voice-Enabled-Rag-model.git
cd "Voice-Enabled Rag model"
cp .env.example .env        # fill SARVAM_API_KEY (STT) and optionally a Groq/OpenAI key
```

**Option A — Docker (recommended):**
```bash
docker compose up --build        # builds index on first boot, serves API on :8000
```

**Option B — local:**
```bash
pip install -r requirements.txt
python -m app.ingestion.build_index --lang hi --strategies metadata   # offline, once
python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

Then:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"text": "भारत का राष्ट्रीय पक्षी कौन सा है", "language": "hi"}'
```
Voice: `POST /v1/voice` with a multipart `audio` WAV file. Open http://localhost:8000/docs for the interactive Swagger UI.

## 6. Tech Stack (proposed — adjust to your team's comfort)

| Layer | Choice | Why |
|---|---|---|
| STT | Sarvam `saaras:v3-realtime` (WebSocket, REST fallback) | Built for Indian languages, matches MSMARCO-XI's Indic focus |
| Embeddings | `intfloat/multilingual-e5-small` (384-d) | Strong multilingual/Indic retrieval, fast on CPU/RTX |
| Vector DB | FAISS in-process | Fastest for a hackathon-sized corpus; no network hop |
| Chunking | **metadata**-aware passage chunks (measured winner) | Best MRR/R@5 in our 300-query eval — see EVALUATION.md |
| Sparse retrieval | BM25 (`rank_bm25`) — evaluated, **disabled** in prod | Hybrid (RRF) *hurt* retrieval quality on Hindi in our eval |
| Re-ranker | `BAAI/bge-reranker-v2-m3` — optional toggle, **not** in live path | +8.4pts R@5 but adds ~345ms → blows the 200ms budget |
| Generation | Extractive (offline default) or streaming LLM via Groq/OpenAI | Extractive = always under budget; LLM = richer answers with TTFT tracking |
| Guardrails | 3-layer rule + embedding-similarity stack, calibrated on real data | See GUARDRAILS.md |
| Harness | Custom `asyncio` orchestrator | Structured I/O, retries, timeouts, fallbacks, circuit breaker |

## 7. Repo Structure (suggested)

```
voice-rag/
├── app/
│   ├── stt/                # Sarvam client (WS realtime + REST fallback), FakeSTT
│   ├── ingestion/          # chunking strategies, embedding, index build
│   ├── retrieval/          # dense + sparse + hybrid + rerank
│   ├── guardrails/         # 3-layer guardrail stack
│   ├── generation/         # extractive + LLM generators
│   ├── harness/            # orchestrator: retries, timeouts, circuit breaker, schemas
│   └── api/server.py       # FastAPI: /health, /query, /v1/voice
├── benchmarks/
│   ├── run_latency_bench.py
│   ├── run_retrieval_eval.py
│   └── results/
├── data/                   # cached MSMARCO-XI subset + built index
├── scripts/entrypoint.py   # container entrypoint (builds index if missing)
├── tests/                  # 88 passing: unit + integration + adversarial
├── Dockerfile / docker-compose.yml / .github/workflows/ci.yml
├── requirements.txt
└── .env.example
```

## 8. Team

| Name | Role |
|---|---|
| — | STT + Harness |
| — | Retrieval + Chunking |
| — | Guardrails + Generation |
| — | Frontend / Demo / Video |

## 9. Links

- Live demo: `<TBD>`
- Video 1 (process, 90s): `<TBD>`
- Video 2 (demo): `<TBD>`
- Submission form: https://forms.gle/MNvCjcv23Hn2Eeu58

Every social post → tag **#RAGInGoa**, by **every** team member, on IG + X + LinkedIn. See [SUBMISSION_CHECKLIST.md](./SUBMISSION_CHECKLIST.md).

## 10. Full Doc Index

| File | Purpose |
|---|---|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Pipeline stages, latency budget, harness design, component rationale |
| [CHUNKING_STRATEGY.md](./CHUNKING_STRATEGY.md) | 5 chunking strategies + hybrid retrieval + evaluation plan |
| [GUARDRAILS.md](./GUARDRAILS.md) | 3-layer guardrail stack, refusal templates |
| [LATENCY_BENCHMARK.md](./LATENCY_BENCHMARK.md) | Benchmark methodology, report template, script skeleton |
| [EVALUATION.md](./EVALUATION.md) | Retrieval/answer quality results template |
| [API.md](./API.md) | Internal request/response schemas between pipeline stages |
| [TESTING.md](./TESTING.md) | Test plan per layer (unit, integration, guardrail adversarial tests) |
| [DEPLOYMENT.md](./DEPLOYMENT.md) | How to stand up the live demo link, latency-safe hosting choices |
| [ROADMAP.md](./ROADMAP.md) | Day-by-day 8-day execution plan |
| [SUBMISSION_CHECKLIST.md](./SUBMISSION_CHECKLIST.md) | Every deliverable + promotion requirement, tracked |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Team git workflow |
| [CHANGELOG.md](./CHANGELOG.md) | Release/version log |
| [LICENSE](./LICENSE) | Project license |
