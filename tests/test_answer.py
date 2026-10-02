import pytest

import store
from answer import REFUSAL, SYSTEM_PROMPT, build_prompt, format_context, retrieve
from embeddings import embed

HITS = [
    {"source": "a.asc", "heading": "Undoing Things", "text": "Use git commit --amend.", "distance": 0.1},
    {"source": "b.asc", "heading": "A diff", "text": "--- a/file\n+++ b/file", "distance": 0.2},
]


def test_format_context_wraps_each_hit_in_a_labelled_excerpt():
    assert format_context(HITS) == (
        '<excerpt source="a.asc" section="Undoing Things">\n'
        "Use git commit --amend.\n"
        "</excerpt>\n"
        "\n"
        '<excerpt source="b.asc" section="A diff">\n'
        "--- a/file\n"
        "+++ b/file\n"
        "</excerpt>"
    )


def test_format_context_says_so_when_there_are_no_hits():
    assert format_context([]) == "No excerpts were found."


def test_build_prompt_puts_the_question_after_the_excerpts():
    prompt = build_prompt("  How do I amend?  ", HITS)
    assert prompt.startswith('<excerpt source="a.asc"')
    assert prompt.endswith("</excerpt>\n\nQuestion: How do I amend?")


def test_system_prompt_contains_the_refusal_sentence():
    assert f'reply exactly: "{REFUSAL}"' in SYSTEM_PROMPT


def test_retrieve_returns_nearest_chunk_first(isolated_store):
    for chunk_id, text in [("revert", "git revert undoes an earlier commit"), ("banana", "Bananas are rich in potassium")]:
        store.save_chunk(chunk_id, text, "x.asc", "H", 0, embed([text])[0])
    hits = retrieve("How do I undo a commit with git revert?", k=2)
    assert [hit["id"] for hit in hits] == ["revert", "banana"]


def test_retrieve_returns_nothing_from_an_empty_index(isolated_store):
    assert retrieve("anything") == []


def test_retrieve_rejects_a_blank_question():
    with pytest.raises(ValueError):
        retrieve("   ")