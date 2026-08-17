"""Multi-turn conversation evaluation tests (Task 3.3)."""
import time

from app.harness.schemas import ConversationTurn
from app.session.rewriter import rewrite_query
from app.session.state import SessionStore


def _user_turn(text: str) -> ConversationTurn:
    return ConversationTurn(role="user", text=text, timestamp=time.time())


def _assistant_turn(text: str) -> ConversationTurn:
    return ConversationTurn(role="assistant", text=text, timestamp=time.time())


def test_followup_resolves_topic():
    """Rewriting resolves referential queries into standalone form."""
    history = [
        _user_turn("दिल्ली के बारे में बताओ"),
        _assistant_turn("दिल्ली भारत की राजधानी है।"),
    ]
    result = rewrite_query("वहाँ का मौसम कैसा है", history)
    assert "दिल्ली" in result
    assert len(result) > len("वहाँ का मौसम कैसा है")


def test_topic_switch_does_not_carry_context():
    """New topic query is not polluted with old context."""
    history = [
        _user_turn("दिल्ली के बारे में बताओ"),
        _assistant_turn("दिल्ली भारत की राजधानी है।"),
    ]
    result = rewrite_query("मुंबई कहाँ है", history)
    assert result == "मुंबई कहाँ है"
    assert "दिल्ली" not in result


def test_session_tracks_conversation():
    """Session records all turns in order."""
    store = SessionStore()
    session = store.get_or_create("test-session")
    session.add_turn(_user_turn("पहला सवाल"))
    session.add_turn(_assistant_turn("पहला उत्तर"))
    session.add_turn(_user_turn("दूसरा सवाल"))
    session.add_turn(_assistant_turn("दूसरा उत्तर"))
    assert session.turn_count == 4
    assert session.history[0].role == "user"
    assert session.history[1].role == "assistant"
    assert session.history[2].role == "user"
    assert session.history[3].role == "assistant"
    assert session.recent_history(n=2)[0].text == "दूसरा सवाल"


def test_multi_turn_pipeline_session():
    """End-to-end: rewrite + session tracking with multiple turns."""
    store = SessionStore()
    session = store.get_or_create("e2e-session")

    # Turn 1: user asks about Delhi
    session.add_turn(_user_turn("दिल्ली के बारे में बताओ"))
    session.add_turn(_assistant_turn("दिल्ली भारत की राजधानी है।"))

    # Turn 2: follow-up with pronoun
    rewritten = rewrite_query("वहाँ का मौसम कैसा है", session.history)
    assert "दिल्ली" in rewritten

    # Simulate processing: record the rewritten query and answer
    session.add_turn(_user_turn("वहाँ का मौसम कैसा है"))
    session.add_turn(_assistant_turn("दिल्ली में गर्मियों में तापमान बहुत ऊँचा होता है।"))

    assert session.turn_count == 4
    assert session.last_user_query() == "वहाँ का मौसम कैसा है"
