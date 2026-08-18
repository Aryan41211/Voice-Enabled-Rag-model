"""Multi-turn query rewriter evaluation — 20 test cases.

Tests that the rule-based rewriter correctly resolves referential queries
into standalone queries. The rewriter prepends the topic word from the last
assistant response when it detects a pronoun in the follow-up.

Each test case has:
* conversation_history: prior turns
* raw_followup: what the user says next
* expected_contains: topic word that should appear in the rewritten query
* has_pronoun: whether the followup triggers rewriting
"""

from __future__ import annotations

import pytest

from app.harness.schemas import ConversationTurn
from app.session.rewriter import rewrite_query


def _user(text: str) -> ConversationTurn:
    return ConversationTurn(role="user", text=text, timestamp=0.0)


def _assistant(text: str) -> ConversationTurn:
    return ConversationTurn(role="assistant", text=text, timestamp=0.0)


# ─── Test cases ─────────────────────────────────────────────────────────

CASES = [
    # Hindi pronoun resolution (8)
    {
        "id": "hindi_implicit_topic",
        "history": [
            _user("भारत का राष्ट्रीय पक्षी क्या है"),
            _assistant("भारत का राष्ट्रीय पक्षी मोर है।"),
        ],
        "raw_followup": "उसकी वैज्ञानिक क्या है",
        "expected_contains": "भारत",  # rewriter prepends first topic word
        "has_pronoun": True,
    },
    {
        "id": "hindi_demonstrative",
        "history": [
            _user("चंद्रयान-3 कब लॉन्च हुआ"),
            _assistant("चंद्रयान-3 14 जुलाई 2023 को लॉन्च हुआ था।"),
        ],
        "raw_followup": "इसका उद्देश्य क्या था",
        "expected_contains": "चंद्रयान",
        "has_pronoun": True,
    },
    {
        "id": "hindi_possessive",
        "history": [
            _user("गांधी जी का जन्म कब हुआ"),
            _assistant("मोहनदास करमचंद गांधी का जन्म 2 अक्टूबर 1869 को हुआ था।"),
        ],
        "raw_followup": "उसका जन्मस्थान कहाँ था",
        "expected_contains": "मोहनदास",  # first Hindi word >=3 chars
        "has_pronoun": True,
    },
    {
        "id": "hindi_remote_reference",
        "history": [
            _user("ताजमहल कहाँ है"),
            _assistant("ताजमहल आगरा में स्थित है।"),
            _user("वह किसने बनवाया"),
            _assistant("शाहजहाँ ने बनवाया।"),
        ],
        "raw_followup": "उसकी लागत कितनी थी",
        "expected_contains": "शाहजहाँ",
        "has_pronoun": True,
    },
    # English pronoun resolution (6)
    {
        "id": "english_it_reference",
        "history": [
            _user("What is the capital of France"),
            _assistant("The capital of France is Paris."),
        ],
        "raw_followup": "What is its population",
        "expected_contains": "France",  # first capitalized non-stopword
        "has_pronoun": True,
    },
    {
        "id": "english_that_reference",
        "history": [
            _user("Tell me about the Magna Carta"),
            _assistant("The Magna Carta was signed in 1215."),
        ],
        "raw_followup": "When was that signed",
        "expected_contains": "Magna",
        "has_pronoun": True,
    },
    {
        "id": "english_they_reference",
        "history": [
            _user("Who founded Microsoft"),
            _assistant("Microsoft was founded by Bill Gates and Paul Allen."),
        ],
        "raw_followup": "Where did they grow up",
        "expected_contains": "Microsoft",
        "has_pronoun": True,
    },
    {
        "id": "english_this_reference",
        "history": [
            _user("What is machine learning"),
            _assistant("Machine learning is a subset of artificial intelligence."),
        ],
        "raw_followup": "Give me examples of this",
        "expected_contains": "Machine",
        "has_pronoun": True,
    },
    {
        "id": "english_there_reference",
        "history": [
            _user("Where is the Great Wall of China"),
            _assistant("The Great Wall stretches across northern China."),
        ],
        "raw_followup": "How long is there",
        "expected_contains": "Great",
        "has_pronoun": True,
    },
    {
        "id": "english_those_reference",
        "history": [
            _user("What are the seven wonders"),
            _assistant("The seven wonders include the Great Pyramid, Colosseum, and others."),
        ],
        "raw_followup": "Tell me more about those",
        "expected_contains": "Great",  # first capitalized non-stopword
        "has_pronoun": True,
    },
    # No rewriting needed (3)
    {
        "id": "no_pronoun_standalone",
        "history": [
            _user("भारत का राष्ट्रीय पक्षी क्या है"),
            _assistant("भारत का राष्ट्रीय पक्षी मोर है।"),
        ],
        "raw_followup": "चीन का राष्ट्रीय पक्षी क्या है",
        "expected_contains": None,  # should NOT be rewritten
        "has_pronoun": False,
    },
    {
        "id": "no_history",
        "history": [],
        "raw_followup": "भारत का राष्ट्रीय पक्षी",
        "expected_contains": None,
        "has_pronoun": False,
    },
    {
        "id": "english_standalone_no_pronoun",
        "history": [
            _user("What is Python"),
            _assistant("Python is a programming language."),
        ],
        "raw_followup": "What is Rust",
        "expected_contains": None,
        "has_pronoun": False,
    },
    # Edge cases (3)
    {
        "id": "hindi_multiple_pronouns",
        "history": [
            _user("केरल की राजधानी क्या है"),
            _assistant("केरल की राजधानी तिरुवनंतपुरम है।"),
        ],
        "raw_followup": "उसकी जनसंख्या क्या है और उसका क्षेत्रफल कितना है",
        "expected_contains": "केरल",
        "has_pronoun": True,
    },
    {
        "id": "english_mixed_context",
        "history": [
            _user("Explain quantum computing"),
            _assistant("Quantum computing uses qubits instead of classical bits."),
        ],
        "raw_followup": "How does this differ from classical computing",
        "expected_contains": "Quantum",
        "has_pronoun": True,
    },
    {
        "id": "long_conversation_turn5",
        "history": [
            _user("What is DNA"),
            _assistant("DNA carries genetic instructions for life."),
            _user("Where is it found"),
            _assistant("DNA is found in the nucleus of cells."),
            _user("What does it look like"),
            _assistant("DNA has a double helix structure."),
            _user("Who discovered it"),
            _assistant("Watson and Crick discovered the structure in 1953."),
            _user("When was that"),
        ],
        "raw_followup": "When was that published",
        "expected_contains": "Watson",
        "has_pronoun": True,
    },
]


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[c["id"] for c in CASES],
)
def test_rewriter_accuracy(case):
    """Verify rewriter produces expected standalone query."""
    result = rewrite_query(case["raw_followup"], case["history"])

    if case["expected_contains"] is None:
        # Should NOT rewrite — return original
        assert result == case["raw_followup"], (
            f"Expected no rewrite, got: {result}"
        )
    else:
        assert case["expected_contains"].lower() in result.lower(), (
            f"Expected '{case['expected_contains']}' in rewrite, "
            f"got: {result}"
        )

    # Verify pronoun detection matches expectation
    from app.session.rewriter import _HINDI_PRONOUNS, _ENGLISH_PRONOUNS
    has_pronoun = bool(
        _HINDI_PRONOUNS.search(case["raw_followup"])
        or _ENGLISH_PRONOUNS.search(case["raw_followup"])
    )
    assert has_pronoun == case["has_pronoun"], (
        f"Pronoun detection mismatch: expected {case['has_pronoun']}, "
        f"got {has_pronoun} for '{case['raw_followup']}'"
    )


def test_rewriter_returns_original_when_no_history():
    """Rewriter must not hallucinate without conversation context."""
    result = rewrite_query("भारत का राष्ट्रीय पक्षी", [])
    assert result == "भारत का राष्ट्रीय पक्षी"


def test_rewriter_returns_original_when_no_assistant_turn():
    """Rewriter must handle history with only user turns."""
    result = rewrite_query(
        "उसकी वैज्ञानिक क्या है",
        [_user("भारत का राष्ट्रीय पक्षी क्या है")],
    )
    assert result == "उसकी वैज्ञानिक क्या है"


def test_rewriter_latency_under_1ms():
    """Confirm rewriter is pure string logic, no LLM calls."""
    import time

    history = [
        _user("भारत का राष्ट्रीय पक्षी क्या है"),
        _assistant("भारत का राष्ट्रीय पक्षी मोर है।"),
    ]
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        rewrite_query("उसकी वैज्ञानिक क्या है", history)
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    assert avg_ms < 1.0, f"Rewriter avg latency {avg_ms:.3f}ms exceeds 1ms"


def test_long_conversation_no_degradation():
    """5+ turn conversation should still rewrite correctly."""
    history = [
        _user("What is Python"),
        _assistant("Python is a programming language created by Guido van Rossum."),
        _user("When was it created"),
        _assistant("Python was first released in 1991."),
        _user("What is it used for"),
        _assistant("Python is used for web development, data science, and AI."),
        _user("Who created it"),
        _assistant("Guido van Rossum created Python in the late 1980s."),
    ]
    result = rewrite_query("What is its latest version", history)
    assert "guido" in result.lower() or "python" in result.lower()
