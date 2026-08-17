"""Structured JSON logging with request-scoped context via contextvars."""

from __future__ import annotations

import contextvars
import json
import logging
from contextlib import contextmanager
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_extra_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "extra_context", default={}
)


class RequestContextFilter(logging.Filter):
    """Inject request_id and extra context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or ""  # type: ignore[attr-defined]
        extra = _extra_context.get()
        for k, v in extra.items():
            setattr(record, k, v)
        return True


class JsonFormatter(logging.Formatter):
    """Emit each log line as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", None)
        if rid:
            entry["request_id"] = rid
        for key in ("stage", "latency_ms", "success", "error", "detail"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    handler.addFilter(RequestContextFilter())
    root.addHandler(handler)


@contextmanager
def request_context(request_id: str, **extra: Any):  # type: ignore[return]
    """Context manager that sets request_id and extra fields for all logs within."""
    token_id = _request_id.set(request_id)
    token_extra = _extra_context.set({**_extra_context.get(), **extra})
    try:
        yield
    finally:
        _request_id.reset(token_id)
        _extra_context.reset(token_extra)
