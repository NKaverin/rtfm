import pytest

import answer
import sessions
import store
from embeddings import embed

pytestmark = pytest.mark.usefixtures("isolated_sessions")

LONG = "word " * 600

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

def test_compact_leaves_a_short_conversation_alone():
    sessions.append("s1", "user", "short question")
    sessions.append("s1", "assistant", "short answer")
    assert sessions.compact("s1", lambda previous, messages: "never used") is False
    assert sessions.summary("s1") == ""
    assert len(sessions.history("s1")) == 2


def test_compact_replaces_old_messages_with_a_summary():
    for text in ["first question", LONG, "second question", LONG]:
        sessions.append("s1", "user", text)
    seen = []

    def fake_summarise(previous, messages):
        seen.append((previous, [message["text"] for message in messages]))
        return "They talked about words."

    assert sessions.compact("s1", fake_summarise) is True
    assert seen == [("", ["first question", LONG])]
    assert sessions.summary("s1") == "They talked about words."
    assert [m["text"] for m in sessions.history("s1")] == ["second question", LONG]
    assert store.r.ttl(sessions.PREFIX + "s1:summary") == sessions.TTL_SECONDS


def test_compact_hands_the_previous_summary_to_the_summariser():
    store.r.set(sessions.PREFIX + "s1:summary", "Earlier: tags.")
    for text in [LONG, LONG, "latest question", "latest answer"]:
        sessions.append("s1", "user", text)
    sessions.compact("s1", lambda previous, messages: f"{previous} Later: branches.")
    assert sessions.summary("s1") == "Earlier: tags. Later: branches."


def test_every_message_also_resets_the_expiry_of_the_summary():
    store.r.set(sessions.PREFIX + "s1:summary", "Earlier: tags.", ex=100)
    sessions.append("s1", "user", "next question")
    assert store.r.ttl(sessions.PREFIX + "s1:summary") == sessions.TTL_SECONDS

def test_a_long_conversation_is_summarised_and_the_summary_reaches_the_prompt(
    monkeypatch, isolated_store, isolated_cache
):
    save_tag_chunk()
    monkeypatch.setattr(sessions, "SUMMARY_TOKENS", 20)
    prompts = []

    def fake_generate(system, prompt, model=answer.LLM_MODEL):
        if system == answer.SUMMARY_PROMPT:
            return "The user asked how to create a tag."
        prompts.append(prompt)
        return "Use git tag with a name and it creates a tag for that commit [tagging.asc]."

    monkeypatch.setattr(answer, "generate", fake_generate)
    answer.answer("How do I create a tag?", "s1")
    answer.answer("How do I delete it?", "s1")
    answer.answer("And how do I list them?", "s1")

    assert sessions.summary("s1") == "The user asked how to create a tag."
    assert len(sessions.history("s1")) == 2
    assert "Summary of earlier messages: The user asked how to create a tag." in prompts[2]


def test_a_failing_summary_keeps_the_messages(monkeypatch, isolated_store, isolated_cache, caplog):
    save_tag_chunk()
    monkeypatch.setattr(sessions, "SUMMARY_TOKENS", 1)

    def fake_generate(system, prompt, model=answer.LLM_MODEL):
        if system == answer.SUMMARY_PROMPT:
            raise answer.ModelError("summary model is gone")
        return "Use git tag [tagging.asc]."

    monkeypatch.setattr(answer, "generate", fake_generate)
    answer.answer("How do I create a tag?", "s1")
    result = answer.answer("How do I delete it?", "s1")

    assert result["answer"] == "Use git tag [tagging.asc]."
    assert len(sessions.history("s1")) == 4
    assert "save_turn failed" in caplog.text