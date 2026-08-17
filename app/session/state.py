"""In-memory session store with LRU eviction for multi-turn conversation state."""
from __future__ import annotations

import threading
from collections import OrderedDict

from app.harness.schemas import ConversationTurn, SessionState

__all__ = ["ConversationTurn", "SessionState", "SessionStore"]


class SessionStore:
    """Thread-safe LRU store of session states."""

    def __init__(self, max_sessions: int = 100) -> None:
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]
            while len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            session = SessionState(session_id=session_id)
            self._sessions[session_id] = session
            return session

    def clear(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)
