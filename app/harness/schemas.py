"""Typed contracts shared across pipeline stages.

Mirrors ``API.md`` — this file is the machine-readable version of that
contract. Every stage in ``app/`` takes and returns these pydantic models
instead of raw strings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    background_score: float | None = Field(
        default=None,
        description="Cosine at a deep rank (e.g. rank 20); top-1 minus this "
        "estimates how isolated the best match is.",
    )


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


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    text: str
    timestamp: float
    chunks_used: list[str] = field(default_factory=list)
    guardrail_actions: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    session_id: str
    max_turns: int = 10
    language: str = "hi"
    history: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return len(self.history)

    def add_turn(self, turn: ConversationTurn) -> None:
        self.history.append(turn)

    def recent_history(self, n: int = 3) -> list[ConversationTurn]:
        return self.history[-n:]

    def last_user_query(self) -> str | None:
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.text
        return None

    def get_context_summary(self) -> str:
        recent = self.recent_history(n=3)
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.text}")
        return "\n".join(lines)
