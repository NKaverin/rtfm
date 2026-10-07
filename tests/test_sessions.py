import pytest

import answer
import sessions
import store
from embeddings import embed

pytestmark = pytest.mark.usefixtures("isolated_sessions")


def test_history_returns_messages_in_order():
    sessions.append("s1", "user", "first")
    sessions.append("s1", "assistant", "second")
    assert sessions.history("s1") == [
        {"role": "user", "text": "first"},
        {"role": "assistant", "text": "second"},
    ]


def test_history_keeps_only_the_most_recent_messages():
    for number in range(10):
        sessions.append("s1", "user", f"message {number}")
    assert [m["text"] for m in sessions.history("s1", limit=3)] == ["message 7", "message 8", "message 9"]


def test_sessions_do_not_see_each_other():
    sessions.append("s1", "user", "only in s1")
    assert sessions.history("s2") == []


def test_every_message_resets_the_expiry():
    sessions.append("s1", "user", "first")
    store.r.expire(sessions.PREFIX + "s1", 100)
    sessions.append("s1", "user", "second")
    assert store.r.ttl(sessions.PREFIX + "s1") == sessions.TTL_SECONDS


def test_new_id_gives_a_different_id_each_time():
    assert sessions.new_id() != sessions.new_id()


def save_tag_chunk():
    text = "git tag -d deletes a tag"
    store.save_chunk("tags", text, "tagging.asc", "Deleting Tags", 0, embed([text])[0])


def test_follow_up_question_gets_the_conversation_and_skips_the_cache(monkeypatch, isolated_store, isolated_cache):
    save_tag_chunk()
    prompts = []

    def fake_generate(system, prompt):
        prompts.append(prompt)
        return "Use git tag [tagging.asc]."

    monkeypatch.setattr(answer, "generate", fake_generate)
    answer.answer("How do I create a tag?", "s1")
    second = answer.answer("How do I delete it?", "s1")
    third = answer.answer("How do I delete it?", "s1")

    assert "<conversation>" not in prompts[0]
    assert "User: How do I create a tag?\nAssistant: Use git tag [tagging.asc]." in prompts[1]
    assert second["cached"] is False
    assert third["cached"] is False
    assert len(prompts) == 3
    assert [m["role"] for m in sessions.history("s1")] == ["user", "assistant"] * 3


def test_first_question_of_a_new_session_can_use_the_cache(monkeypatch, isolated_store, isolated_cache):
    save_tag_chunk()
    monkeypatch.setattr(answer, "generate", lambda system, prompt: "Use git tag [tagging.asc].")
    answer.answer("How do I create a tag?", "s1")
    result = answer.answer("How do I create a tag?", "s2")
    assert result["cached"] is True
    assert [m["text"] for m in sessions.history("s2")] == ["How do I create a tag?", "Use git tag [tagging.asc]."]