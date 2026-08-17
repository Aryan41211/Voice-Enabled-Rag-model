import time

from app.session.state import ConversationTurn, SessionState, SessionStore


def test_session_state_new_turn():
    s = SessionState(session_id="s1")
    s.add_turn(ConversationTurn(role="user", text="दिल्ली कहाँ है", timestamp=time.time()))
    assert s.turn_count == 1
    assert s.history[0].text == "दिल्ली कहाँ है"


def test_session_state_context_window():
    s = SessionState(session_id="s1", max_turns=3)
    for i in range(5):
        s.add_turn(ConversationTurn(role="user", text=f"q{i}", timestamp=time.time() + i))
    assert s.turn_count == 5
    assert len(s.recent_history(n=3)) == 3
    assert s.recent_history(n=3)[0].text == "q2"


def test_session_store_create_get():
    store = SessionStore(max_sessions=10)
    s = store.get_or_create("s1")
    s.add_turn(ConversationTurn(role="user", text="hello", timestamp=time.time()))
    s2 = store.get_or_create("s1")
    assert s2.turn_count == 1
    assert s is s2


def test_session_store_evicts_lru():
    store = SessionStore(max_sessions=2)
    s1 = store.get_or_create("s1")
    store.get_or_create("s2")
    # s1 is LRU (accessed first)
    store.get_or_create("s3")  # evicts s1
    assert store.active_count() == 2
    s1_fresh = store.get_or_create("s1")
    assert s1_fresh is not s1  # new object (was evicted)
    assert s1_fresh.turn_count == 0


def test_session_clear():
    store = SessionStore()
    s = store.get_or_create("s1")
    s.add_turn(ConversationTurn(role="user", text="q", timestamp=time.time()))
    store.clear("s1")
    assert store.get_or_create("s1").turn_count == 0
