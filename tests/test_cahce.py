import pytest

import answer
import cache
import store
from answer import REFUSAL
from embeddings import embed

pytestmark = pytest.mark.usefixtures("isolated_cache")

ENTRY = {"result": {"answer": "Use git commit --amend.", "sources": ["undoing.asc"], "refused": False}, "prompt_tokens": 1200}


def save(question):
    cache.save(question, embed([question])[0], ENTRY)


def lookup(question):
    return cache.lookup(embed([question])[0])


def test_lookup_returns_none_when_the_cache_is_empty():
    assert lookup("How do I undo my last commit?") is None


def test_lookup_finds_the_same_question_and_a_close_rewording():
    save("How do I undo my last commit?")
    assert lookup("How do I undo my last commit?") == ENTRY
    assert lookup("How can I undo the last commit?") == ENTRY


def test_lookup_misses_a_different_question_with_similar_words():
    save("How do I undo my last commit?")
    assert lookup("How do I see my last commit?") is None
    assert lookup("How do I undo my last push?") is None


def test_saving_the_same_question_twice_keeps_one_entry():
    save("How do I undo my last commit?")
    save("How do I undo my last commit?")
    assert len(list(store.r.scan_iter(cache.PREFIX + "*"))) == 1


def test_flush_removes_every_entry_and_reports_the_count():
    save("How do I undo my last commit?")
    save("How do I create a branch?")
    assert cache.flush() == 2
    assert lookup("How do I undo my last commit?") is None


def save_revert_chunk():
    text = "git revert undoes an earlier commit"
    store.save_chunk("revert", text, "undoing.asc", "H", 0, embed([text])[0])


def test_second_identical_question_is_served_from_cache(monkeypatch, isolated_store):
    save_revert_chunk()
    calls = []

    def fake_generate(system, prompt):
        calls.append(prompt)
        return "Use git revert [undoing.asc]."

    monkeypatch.setattr(answer, "generate", fake_generate)
    first = answer.answer("How do I undo a commit?")
    second = answer.answer("How do I undo a commit?")

    assert len(calls) == 1
    assert first["cached"] is False
    assert second == {**first, "cached": True}
    numbers = cache.metrics()
    assert (numbers["hits"], numbers["misses"], numbers["hit_rate"]) == (1, 1, 0.5)
    assert numbers["tokens_saved"] > 0


def test_refusals_are_not_cached(monkeypatch, isolated_store):
    save_revert_chunk()
    calls = []

    def fake_generate(system, prompt):
        calls.append(prompt)
        return REFUSAL

    monkeypatch.setattr(answer, "generate", fake_generate)
    answer.answer("How do I bake bread?")
    result = answer.answer("How do I bake bread?")

    assert len(calls) == 2
    assert result["cached"] is False