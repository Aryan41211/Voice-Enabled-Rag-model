# API / Internal Schemas

Structured I/O is a scored harness requirement — this file is the contract every stage in `app/harness/` must honor. Define these as pydantic models (or equivalent) in code; this doc is the source of truth they should match.

## External Endpoint (if you expose a REST/WebSocket API for the live demo)

### `POST /query/audio` (or WebSocket equivalent for streaming)
Request: multipart audio blob or streamed PCM/WAV chunks.

Response:
```json
{
  "request_id": "uuid",
  "transcript": "string",
  "answer": "string | null",
  "refused": false,
  "refusal_reason": "string | null",
  "sources": [
    {"chunk_id": "string", "passage": "string", "score": 0.0, "strategy": "hybrid"}
  ],
  "timings_ms": {
    "stt": 0,
    "retrieval": 0,
    "ttft": 0,
    "full_generation": 0,
    "guardrails": 0,
    "total": 0
  }
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
