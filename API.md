# API / Internal Schemas

Structured I/O is a scored harness requirement — this file is the contract every stage in `app/harness/` must honor. Define these as pydantic models (or equivalent) in code; this doc is the source of truth they should match.

## External Endpoint (live demo — `app/api/server.py`)

FastAPI app exposed at port `8000` (uvicorn). Startup builds/warm-loads the pipeline, so first request is fast.

### `GET /health`
```json
{"status": "ok", "version": "0.1.0", "strategy": "metadata", "generation": "extractive", "stt_provider": "sarvam"}
```

### `POST /query` — text-in / text-out
Request:
```json
{"text": "भारत का राष्ट्रीय पक्षी कौन सा है", "language": "hi"}
```
Response: the `QueryResponse` contract below.

### `POST /v1/voice` — audio-in / text-out
Multipart `audio` file (WAV; STT converts to PCM). Response is the same `QueryResponse` contract with `timings_ms.stt` populated.

### `QueryResponse`
```json
{
  "request_id": "uuid",
  "transcript": "string",
  "answer": "string | null",
  "refused": false,
  "refusal_reason": "string | null",
  "sources": [
    {"chunk_id": "string", "passage": "string", "score": 0.0, "strategy": "metadata"}
  ],
  "timings_ms": {
    "stt": 0, "input_guardrail": 0, "retrieval": 0,
    "retrieval_guardrail": 0, "generation": 0, "output_guardrail": 0, "total_ms": 0
  },
  "schema_version": "1.0"
}
```

## Internal Stage Contracts

### `Transcript` (STT → Guardrail)
```python
class Transcript(BaseModel):
    text: str
    language: str
    confidence: float
    is_final: bool
    stt_latency_ms: float
```

### `GuardrailResult` (Guardrail → Retrieval, or Guardrail → short-circuit response)
```python
class GuardrailResult(BaseModel):
    passed: bool
    layer: Literal["input", "retrieval", "output"]
    reason: str | None
    action: Literal["proceed", "refuse", "clarify"]
```

### `RetrievedChunk` (Retrieval → Generation)
```python
class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: Literal["dense", "sparse", "hybrid"]
    strategy: str  # which chunking strategy produced this chunk
    metadata: dict
```

### `RetrievalResult` (Retrieval → Guardrail Layer 2 → Generation)
```python
class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
    background_score: float | None  # cosine at deep rank (rank 20); top-1 minus this ≈ isolation margin
```

### `Answer` (Generation → Output Guardrail → API response)
```python
class Answer(BaseModel):
    text: str
    cited_chunk_ids: list[str]
    ttft_ms: float
    full_generation_ms: float
    grounded: bool | None  # set by output guardrail
```

## Error Contract
Every stage raises a typed exception (`STTError`, `RetrievalError`, `GenerationError`, `GuardrailError`) caught by the harness, which converts it into a graceful `refused: true` response with `refusal_reason` — **never** an unhandled 500 in the demo.

```python
class PipelineStageError(Exception):
    stage: str
    retryable: bool
    detail: str
```

## Versioning
If the schema changes mid-hackathon, bump a `schema_version` field in the response and note it in `CHANGELOG.md` — small thing, but shows engineering discipline in the repo judges will skim.
