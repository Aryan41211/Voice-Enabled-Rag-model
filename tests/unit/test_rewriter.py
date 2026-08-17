"""Tests for query rewriting (Task 3.1)."""
import time

from app.harness.schemas import ConversationTurn
from app.session.rewriter import rewrite_query


def _user_turn(text: str) -> ConversationTurn:
    return ConversationTurn(role="user", text=text, timestamp=time.time())


def _assistant_turn(text: str) -> ConversationTurn:
    return ConversationTurn(role="assistant", text=text, timestamp=time.time())


def test_rewrite_standalone_query():
    """No history → query returned unchanged."""
    result = rewrite_query("दिल्ली कहाँ है", [])
    assert result == "दिल्ली कहाँ है"


def test_rewrite_followup_with_pronoun():
    """Hindi pronoun 'वहाँ' after Delhi context → contains 'दिल्ली'."""
    history = [
        _user_turn("दिल्ली के बारे में बताओ"),
        _assistant_turn("दिल्ली भारत की राजधानी है और यह यमुना नदी के किनारे बसा है।"),
    ]
    result = rewrite_query("वहाँ का मौसम कैसा है", history)
    assert "दिल्ली" in result


def test_rewrite_followup_with_reference():
    """Hindi reference word after Chandrayaan context → contains topic."""
    history = [
        _user_turn("चंद्रयान-3 के बारे में बताओ"),
        _assistant_turn(
            "चंद्रयान-3 भारत का चंद्र मिशन है जिसने चंद्रयान-2 की असफलता के बाद सफलता प्राप्त की।"
        ),
    ]
    result = rewrite_query("इसकी लागत कितनी थी", history)
    assert "चंद्रयान" in result


def test_rewrite_no_context_needed():
    """New topic query → no rewriting."""
    history = [
        _user_turn("दिल्ली के बारे में बताओ"),
        _assistant_turn("दिल्ली भारत की राजधानी है।"),
    ]
    result = rewrite_query("ISRO का मुख्यालय कहाँ है", history)
    assert result == "ISRO का मुख्यालय कहाँ है"
