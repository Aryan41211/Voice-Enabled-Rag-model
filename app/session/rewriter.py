"""Rule-based query rewriter for multi-turn conversations.

Rewrites follow-up questions containing referential pronouns into standalone
queries by prepending the topic from the last assistant response. Pure string
logic, no LLM calls, <1 ms latency.
"""
from __future__ import annotations

import re

from app.harness.schemas import ConversationTurn

# Referential words that signal a follow-up needing rewriting
_HINDI_PRONOUNS = re.compile(
    r"(वहाँ|उसमें|यहाँ|इसमें|उसका|इसका|उसकी|इसकी|उसके|इसके|उनका|उनकी|वह|यह|इसे|उसे)"
)
_ENGLISH_PRONOUNS = re.compile(
    r"\b(it|that|there|this|those|them|they|its|their|his|her)\b", re.IGNORECASE
)

# Topic extraction: find the first proper noun / domain keyword in assistant text
_TOPIC_PATTERN = re.compile(
    r"([A-Z][a-zA-Z]*(?:\s*[-–]\s*\d+)?"  # English proper nouns like Chandrayaan-3
    r"|[\u0900-\u097F]{2,})"  # Hindi words ≥2 chars
)

# English stopwords to skip when extracting topic
_EN_STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was",
    "are", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "must", "it", "its", "this", "that", "these",
    "those", "i", "we", "you", "he", "she", "they", "my", "your", "his",
    "her", "our", "their", "what", "which", "who", "whom", "where",
    "when", "how", "why", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "as", "until",
    "while", "about", "between", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once",
})


def _extract_topic(assistant_text: str) -> str | None:
    """Extract the first significant topic word from assistant text."""
    # Try Hindi first (more common in this project's domain)
    hindi_match = re.search(r"([\u0900-\u097F]{3,})", assistant_text)
    if hindi_match:
        return hindi_match.group(1)
    # Fall back to English proper noun, skipping stopwords
    for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s*[-–]\s*\d+)?)\b", assistant_text):
        word = match.group(1)
        if word.lower() not in _EN_STOPWORDS:
            return word
    return None


def _last_assistant_text(history: list[ConversationTurn]) -> str | None:
    """Get the text of the most recent assistant turn."""
    for turn in reversed(history):
        if turn.role == "assistant":
            return turn.text
    return None


def rewrite_query(
    current_query: str, conversation_history: list[ConversationTurn]
) -> str:
    """Rewrite a follow-up query into a standalone query.

    If the query contains referential pronouns (Hindi or English) and there is
    conversation history, prepend the topic from the last assistant response so
    the query is self-contained for retrieval.

    Returns the original query unchanged if no rewriting is needed.
    """
    if not conversation_history:
        return current_query

    has_pronoun = _HINDI_PRONOUNS.search(current_query) or _ENGLISH_PRONOUNS.search(
        current_query
    )
    if not has_pronoun:
        return current_query

    last_text = _last_assistant_text(conversation_history)
    if not last_text:
        return current_query

    topic = _extract_topic(last_text)
    if not topic:
        return current_query

    return f"{topic} {current_query}"
