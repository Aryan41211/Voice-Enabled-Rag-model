"""SQLite-backed structured logging store for production request data."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RequestLogEntry(BaseModel):
    request_id: str
    timestamp: float
    transcript: str = ""
    language: str = "hi"
    answer: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    guardrail_input: str = "proceed"
    guardrail_retrieval: str = "proceed"
    guardrail_output: str = "proceed"
    stt_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    top_retrieval_score: float | None = None
    explicit_feedback: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogStore:
    """Append-only structured log backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                request_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                transcript TEXT,
                language TEXT,
                answer TEXT,
                refused INTEGER,
                refusal_reason TEXT,
                chunk_ids TEXT,
                guardrail_input TEXT,
                guardrail_retrieval TEXT,
                guardrail_output TEXT,
                stt_latency_ms REAL,
                retrieval_latency_ms REAL,
                generation_latency_ms REAL,
                total_latency_ms REAL,
                top_retrieval_score REAL,
                explicit_feedback INTEGER,
                metadata TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON request_logs(timestamp)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_refused ON request_logs(refused)")
        self._conn.commit()

    def log_request(self, entry: RequestLogEntry) -> None:
        self._conn.execute(
            """INSERT INTO request_logs VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(request_id) DO NOTHING""",
            (
                entry.request_id, entry.timestamp, entry.transcript, entry.language,
                entry.answer, int(entry.refused), entry.refusal_reason,
                json.dumps(entry.chunk_ids), entry.guardrail_input,
                entry.guardrail_retrieval, entry.guardrail_output,
                entry.stt_latency_ms, entry.retrieval_latency_ms,
                entry.generation_latency_ms, entry.total_latency_ms,
                entry.top_retrieval_score, entry.explicit_feedback,
                json.dumps(entry.metadata) if entry.metadata else None,
            ),
        )
        self._conn.commit()

    def query(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        refused_only: bool = False,
        since: float | None = None,
    ) -> list[RequestLogEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if refused_only:
            clauses.append("refused = 1")
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM request_logs{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def recent(self, n: int = 10) -> list[RequestLogEntry]:
        rows = self._conn.execute(
            "SELECT * FROM request_logs ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_entry(r) for r in reversed(rows)]

    def set_feedback(self, request_id: str, feedback: int) -> None:
        self._conn.execute(
            "UPDATE request_logs SET explicit_feedback = ? WHERE request_id = ?",
            (feedback, request_id),
        )
        self._conn.commit()

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN refused=1 THEN 1 ELSE 0 END) as refused_count,
                      AVG(total_latency_ms) as avg_latency,
                      AVG(top_retrieval_score) as avg_score
               FROM request_logs"""
        ).fetchone()
        return dict(row)

    def _row_to_entry(self, row: sqlite3.Row) -> RequestLogEntry:
        return RequestLogEntry(
            request_id=row["request_id"],
            timestamp=row["timestamp"],
            transcript=row["transcript"] or "",
            language=row["language"] or "hi",
            answer=row["answer"],
            refused=bool(row["refused"]),
            refusal_reason=row["refusal_reason"],
            chunk_ids=json.loads(row["chunk_ids"]) if row["chunk_ids"] else [],
            guardrail_input=row["guardrail_input"] or "proceed",
            guardrail_retrieval=row["guardrail_retrieval"] or "proceed",
            guardrail_output=row["guardrail_output"] or "proceed",
            stt_latency_ms=row["stt_latency_ms"] or 0.0,
            retrieval_latency_ms=row["retrieval_latency_ms"] or 0.0,
            generation_latency_ms=row["generation_latency_ms"] or 0.0,
            total_latency_ms=row["total_latency_ms"] or 0.0,
            top_retrieval_score=row["top_retrieval_score"],
            explicit_feedback=row["explicit_feedback"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LogStore:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()
