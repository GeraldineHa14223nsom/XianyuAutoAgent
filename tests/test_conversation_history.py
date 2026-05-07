import time
import pytest
from conversation_history import ConversationHistory, Message


@pytest.fixture
def history():
    return ConversationHistory(max_messages=5, ttl_seconds=60)


def test_add_and_get_messages(history):
    history.add_message("item1", "user1", "user", "Hello")
    history.add_message("item1", "user1", "assistant", "Hi there!")
    msgs = history.get_messages("item1", "user1")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "Hello"}
    assert msgs[1] == {"role": "assistant", "content": "Hi there!"}


def test_separate_sessions_are_isolated(history):
    history.add_message("item1", "user1", "user", "Message A")
    history.add_message("item2", "user2", "user", "Message B")
    assert len(history.get_messages("item1", "user1")) == 1
    assert len(history.get_messages("item2", "user2")) == 1
    assert history.get_messages("item1", "user2") == []


def test_max_messages_trim(history):
    for i in range(10):
        history.add_message("item1", "user1", "user", f"msg {i}")
    msgs = history.get_messages("item1", "user1")
    assert len(msgs) == 5
    assert msgs[0]["content"] == "msg 5"


def test_clear_removes_history(history):
    history.add_message("item1", "user1", "user", "Hello")
    history.clear("item1", "user1")
    assert history.get_messages("item1", "user1") == []
    assert history.session_count() == 0


def test_evict_expired(history):
    # Using a short TTL of 1 second to verify expired sessions are cleaned up
    short_ttl_history = ConversationHistory(max_messages=10, ttl_seconds=1)
    short_ttl_history.add_message("item1", "user1", "user", "Hello")
    short_ttl_history.add_message("item2", "user2", "user", "Hi")
    time.sleep(1.1)
    removed = short_ttl_history.evict_expired()
    assert removed == 2
    assert short_ttl_history.session_count() == 0


def test_evict_keeps_active_sessions(history):
    short_ttl_history = ConversationHistory(max_messages=10, ttl_seconds=2)
    short_ttl_history.add_message("item1", "user1", "user", "Hello")
    time.sleep(1)
    # Access item1 to refresh its TTL
    short_ttl_history.get_messages("item1", "user1")
    short_ttl_history.add_message("item2", "user2", "user", "Hi")
    time.sleep(1.1)
    removed = short_ttl_history.evict_expired()
    # item2 was added after sleep, item1 was accessed after sleep — both still fresh
    assert removed == 0


def test_session_count(history):
    assert history.session_count() == 0
    history.add_message("item1", "user1", "user", "a")
    history.add_message("item2", "user2", "user", "b")
    assert history.session_count() == 2


def test_get_messages_nonexistent_session(history):
    # Retrieving messages for a session that was never created should return an empty list
    assert history.get_messages("ghost_item", "ghost_user") == []
