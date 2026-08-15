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
git clone <your-repo-url>
cd voice-rag
cp .env.example .env        # add SARVAM_API_KEY / ELEVENLABS_API_KEY, LLM key, vector DB config
docker compose up --build   # or: pip install -r requirements.txt && python -m app.main
```

## 6. Tech Stack (proposed — adjust to your team's comfort)

| Layer | Choice | Why |
|---|---|---|
| STT | Sarvam `saaras:v3-realtime` (WebSocket) | Built for Indian languages, <150ms TTFT claimed, matches MSMARCO-XI's Indic focus |
| Embeddings | `bge-m3` or `intfloat/multilingual-e5-large` | Strong multilingual/Indic retrieval performance |
| Vector DB | FAISS (HNSW) locally, or Qdrant if you want a server + filtering | FAISS = fastest for a hackathon-sized corpus; Qdrant = easier metadata filtering + persistence |
| Sparse retrieval | BM25 (`rank_bm25` or Elasticsearch) | For hybrid retrieval, keyword-exact matches embeddings miss |
| Re-ranker | Cross-encoder (`bge-reranker`) — optional, adds latency | Only if you have budget left in the 200ms window |
| Generation | Fast-inference LLM (Groq-hosted Llama/Mixtral, or Sarvam LLM) | Needed to have any shot at low TTFT |
| Guardrails | Rule-based topic classifier + embedding-similarity grounding check + NLI entailment | See GUARDRAILS.md |
| Harness | Custom orchestrator (Python `asyncio` or LangGraph/LlamaIndex workflow) | Structured I/O, retries, timeouts, fallbacks |

## 7. Repo Structure (suggested)

```
voice-rag/
├── app/
│   ├── stt/                # Sarvam/ElevenLabs client + streaming handler
│   ├── ingestion/           # chunking strategies, embedding, index build
│   ├── retrieval/           # dense + sparse + hybrid + rerank
│   ├── guardrails/          # query & answer guardrails
│   ├── generation/          # LLM client, prompt templates
│   ├── harness/             # orchestrator: retries, timeouts, structured I/O, tracing
│   └── main.py
├── benchmarks/
│   ├── run_latency_bench.py
│   └── results/
├── data/                    # cached MSMARCO-XI subset + built index
├── docs/                    # this folder
├── tests/
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── LICENSE
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
