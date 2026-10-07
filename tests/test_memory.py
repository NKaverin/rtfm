import pytest

import answer
import memory
import store
from embeddings import embed

pytestmark = pytest.mark.usefixtures("isolated_memory")

GO = {"text": "The user is building a payment service in Go.", "topics": ["project"], "entities": ["Go"]}
WINDOWS = {"text": "The user works on Windows.", "topics": ["environment"], "entities": ["Windows"]}


def recall(user_id, question):
    return [item["text"] for item in memory.recall(user_id, embed([question])[0])]


def test_remember_then_recall_returns_the_facts():
    assert memory.remember("nik", [GO, WINDOWS]) == [GO["text"], WINDOWS["text"]]
    assert sorted(recall("nik", "What do you know about my project?")) == sorted([GO["text"], WINDOWS["text"]])


def test_users_do_not_see_each_others_memories():
    memory.remember("nik", [GO])
    assert recall("someone_else", "What do you know about my project?") == []


def test_exact_and_near_duplicates_are_not_stored_twice():
    memory.remember("nik", [GO])
    assert memory.remember("nik", [GO]) == []
    assert memory.remember("nik", [{"text": "The user is building a payments service in Go."}]) == []
    assert len(memory.all_memories("nik")) == 1


def test_all_memories_returns_topics_and_entities():
    memory.remember("nik", [GO])
    assert memory.all_memories("nik") == [{"text": GO["text"], "topics": ["project"], "entities": ["Go"]}]


def test_forget_removes_only_that_users_memories():
    memory.remember("nik", [GO])
    memory.remember("someone_else", [WINDOWS])
    assert memory.forget("nik") == 1
    assert memory.all_memories("nik") == []
    assert len(memory.all_memories("someone_else")) == 1


def fake_model(calls):
    def fake_generate(system, prompt, model=answer.LLM_MODEL):
        calls.append({"system": system, "prompt": prompt, "model": model})
        if system == answer.EXTRACTION_PROMPT:
            if "Go" in prompt:
                return '[{"text": "The user is building a payment service in Go.", "topics": ["project"], "entities": ["Go"]}]'
            return "[]"
        return "Noted." if "Go" in prompt.split("Question:")[-1] else "Use git revert [undoing.asc]."

    return fake_generate


def save_revert_chunk():
    text = "git revert undoes an earlier commit"
    store.save_chunk("revert", text, "undoing.asc", "H", 0, embed([text])[0])


def test_a_fact_told_in_one_session_reaches_the_prompt_of_another(
    monkeypatch, isolated_store, isolated_cache, isolated_sessions
):
    save_revert_chunk()
    calls = []
    monkeypatch.setattr(answer, "generate", fake_model(calls))

    first = answer.answer("I'm building a payment service in Go.", "session-1", "nik")
    second = answer.answer("How do I undo a commit?", "session-2", "nik")

    assert first["remembered"] == ["The user is building a payment service in Go."]
    assert second["remembered"] == []
    answer_prompts = [c["prompt"] for c in calls if c["system"] == answer.SYSTEM_PROMPT]
    assert "<memories>" not in answer_prompts[0]
    assert "<memories>\n- The user is building a payment service in Go.\n</memories>" in answer_prompts[1]
    assert "<conversation>" not in answer_prompts[1]
    extraction_models = {c["model"] for c in calls if c["system"] == answer.EXTRACTION_PROMPT}
    assert extraction_models == {answer.EXTRACTION_MODEL}


def test_answers_shaped_by_memories_are_not_cached(monkeypatch, isolated_store, isolated_cache):
    save_revert_chunk()
    memory.remember("nik", [GO])
    monkeypatch.setattr(answer, "generate", fake_model([]))

    answer.answer("How do I undo a commit?", user_id="nik")
    again = answer.answer("How do I undo a commit?", user_id="nik")
    stranger = answer.answer("How do I undo a commit?")

    assert again["cached"] is False
    assert stranger["cached"] is False


def test_without_a_user_id_nothing_is_extracted(monkeypatch, isolated_store, isolated_cache):
    save_revert_chunk()
    calls = []
    monkeypatch.setattr(answer, "generate", fake_model(calls))
    result = answer.answer("I'm building a payment service in Go.")
    assert result["remembered"] == []
    assert [c["system"] for c in calls] == [answer.SYSTEM_PROMPT]