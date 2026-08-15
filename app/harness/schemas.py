"""Typed contracts shared across pipeline stages.

Mirrors ``API.md`` — this file is the machine-readable version of that
contract. Every stage in ``app/`` takes and returns these pydantic models
instead of raw strings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class Transcript(BaseModel):
    text: str
    language: str = "hi"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_final: bool = True
    stt_latency_ms: float = Field(default=0.0, ge=0.0)


class GuardrailResult(BaseModel):
    passed: bool
    layer: Literal["input", "retrieval", "output"]
    reason: str | None = None
    action: Literal["proceed", "refuse", "clarify"] = "proceed"


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = 0.0
    source: Literal["dense", "sparse", "hybrid"] = "hybrid"
    strategy: str = ""
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval_latency_ms: float = Field(default=0.0, ge=0.0)


class Answer(BaseModel):
    text: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    ttft_ms: float = Field(default=0.0, ge=0.0)
    full_generation_ms: float = Field(default=0.0, ge=0.0)
    grounded: bool | None = None


class Source(BaseModel):
    chunk_id: str
    passage: str
    score: float = 0.0
    strategy: str = ""


class QueryResponse(BaseModel):
    request_id: str
    transcript: str
    answer: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    sources: list[Source] = Field(default_factory=list)
    timings_ms: dict = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


class PipelineStageError(Exception):
    """Typed stage failure caught by the harness.

    The harness converts any instance into a graceful ``refused: true``
    response with ``refusal_reason`` — never an unhandled 500.
    """

    def __init__(
        self,
        stage: str,
        detail: str,
        retryable: bool = False,
    ) -> None:
        self.stage = stage
        self.detail = detail
        self.retryable = retryable
        super().__init__(f"[{stage}] {detail}")


class STTError(PipelineStageError):
    def __init__(self, detail: str, retryable: bool = False) -> None:
        super().__init__(stage="stt", detail=detail, retryable=retryable)


class RetrievalError(PipelineStageError):
    def __init__(self, detail: str, retryable: bool = False) -> None:
        super().__init__(stage="retrieval", detail=detail, retryable=retryable)


class GenerationError(PipelineStageError):
    def __init__(self, detail: str, retryable: bool = False) -> None:
        super().__init__(stage="generation", detail=detail, retryable=retryable)


class GuardrailError(PipelineStageError):
    def __init__(self, detail: str, retryable: bool = False) -> None:
        super().__init__(stage="guardrail", detail=detail, retryable=retryable)
