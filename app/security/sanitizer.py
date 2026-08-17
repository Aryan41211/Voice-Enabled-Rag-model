"""Input sanitization for SQL injection and prompt injection protection."""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"forget\s+(your|all)\s+(rules|instructions)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?safety", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are\s+an?\s+unrestricted", re.IGNORECASE),
]


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Strip control characters, limit length, detect prompt injection.

    Returns empty string if a prompt injection pattern is detected.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return ""

    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def sanitize_for_sql(text: str) -> str:
    """Escape single quotes for safe SQLite queries."""
    return text.replace("'", "''")
