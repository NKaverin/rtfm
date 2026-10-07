import os
from types import SimpleNamespace

import pytest

import answer
import store
from answer import REFUSAL, SYSTEM_PROMPT, build_prompt, format_context, retrieve, search_text, parse_facts
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

class FakeGeminiClient:
    """Stands in for genai.Client: records the request, returns a canned reply."""

    def __init__(self, text, status="completed"):
        self.reply = SimpleNamespace(output_text=text, status=status)
        self.requests = []
        self.interactions = self

    def create(self, model, system_instruction, input):
        self.requests.append({"model": model, "system_instruction": system_instruction, "input": input})
        return self.reply


def test_generate_sends_system_and_prompt_and_returns_text(monkeypatch):
    fake = FakeGeminiClient("  Use git revert.\n")
    monkeypatch.setattr(answer.genai, "Client", lambda: fake)
    assert answer.generate("the rules", "the prompt") == "Use git revert."
    assert fake.requests == [
        {"model": "gemini-3.8-flash", "system_instruction": "the rules", "input": "the prompt"}
    ]


def test_generate_raises_when_the_model_returns_nothing(monkeypatch):
    monkeypatch.setattr(answer.genai, "Client", lambda: FakeGeminiClient("", status="failed"))
    with pytest.raises(RuntimeError):
        answer.generate("the rules", "the prompt")


def save_two_chunks():
    for chunk_id, source, text in [
        ("revert", "undoing.asc", "git revert undoes an earlier commit"),
        ("banana", "fruit.asc", "Bananas are rich in potassium"),
    ]:
        store.save_chunk(chunk_id, text, source, "H", 0, embed([text])[0])


def test_answer_lists_only_sources_the_model_cited(monkeypatch, isolated_store, isolated_cache):
    save_two_chunks()
    prompts = []

    def fake_generate(system, prompt):
        prompts.append((system, prompt))
        return "Use git revert [undoing.asc]. It adds a new commit [undoing.asc]."

    monkeypatch.setattr(answer, "generate", fake_generate)
    result = answer.answer("How do I undo a commit?")

    assert result == {
        "answer": "Use git revert [undoing.asc]. It adds a new commit [undoing.asc].",
        "sources": ["undoing.asc"],
        "refused": False,
        "cached": False,
        "remembered": [],
    }
    system, prompt = prompts[0]
    assert system == SYSTEM_PROMPT
    assert 'source="undoing.asc"' in prompt
    assert prompt.endswith("Question: How do I undo a commit?")


def test_answer_marks_a_refusal(monkeypatch, isolated_store, isolated_cache):
    save_two_chunks()
    monkeypatch.setattr(answer, "generate", lambda system, prompt: REFUSAL)
    result = answer.answer("How do I bake bread?")
    assert result == {"answer": REFUSAL, "sources": [], "refused": True, "cached": False, "remembered": []}


@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="needs GEMINI_API_KEY")
def test_real_model_refuses_when_the_excerpts_do_not_cover_the_question(isolated_store, isolated_cache):
    save_two_chunks()
    assert answer.answer("What is the capital of France?")["refused"] is True

class FakeStreamingClient:
    """Stands in for genai.Client when stream=True: returns a list of events."""

    def __init__(self, events):
        self.events = events
        self.requests = []
        self.interactions = self

    def create(self, model, system_instruction, input, stream):
        self.requests.append({"model": model, "stream": stream})
        return self.events


def text_event(text):
    return SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text=text))


def test_generate_stream_yields_only_text_pieces(monkeypatch):
    events = [
        SimpleNamespace(event_type="interaction.created"),
        text_event("Use "),
        SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="thought_summary")),
        text_event("git revert."),
        SimpleNamespace(event_type="interaction.completed"),
    ]
    fake = FakeStreamingClient(events)
    monkeypatch.setattr(answer.genai, "Client", lambda: fake)
    assert list(answer.generate_stream("rules", "prompt")) == ["Use ", "git revert."]
    assert fake.requests == [{"model": "gemini-3.8-flash", "stream": True}]


def test_answer_events_ends_with_sources_and_refusal_flag(monkeypatch):
    hits = [{"source": "undoing.asc", "heading": "H", "text": "git revert undoes a commit"}]
    monkeypatch.setattr(answer, "generate_stream", lambda system, prompt: iter(["Use git revert ", "[undoing.asc]."]))
    assert list(answer.answer_events("How do I undo a commit?", hits)) == [
        {"type": "token", "text": "Use git revert "},
        {"type": "token", "text": "[undoing.asc]."},
        {"type": "done", "sources": ["undoing.asc"], "refused": False},
    ]


def test_answer_events_detects_a_refusal_split_across_pieces(monkeypatch):
    half = len(REFUSAL) // 2
    monkeypatch.setattr(answer, "generate_stream", lambda system, prompt: iter([REFUSAL[:half], REFUSAL[half:] + "\n"]))
    events = list(answer.answer_events("How do I bake bread?", []))
    assert events[-1] == {"type": "done", "sources": [], "refused": True}    

HISTORY = [
    {"role": "user", "text": "How do I create a tag?"},
    {"role": "assistant", "text": "Use git tag [tagging.asc]."},
]


def test_build_prompt_puts_the_conversation_between_excerpts_and_question():
    prompt = build_prompt("How do I delete it?", HITS, HISTORY)
    assert prompt.endswith(
        "</excerpt>\n\n"
        "<conversation>\n"
        "User: How do I create a tag?\n"
        "Assistant: Use git tag [tagging.asc].\n"
        "</conversation>\n\n"
        "Question: How do I delete it?"
    )


def test_build_prompt_without_history_has_no_conversation_block():
    assert "<conversation>" not in build_prompt("How do I amend?", HITS)
    assert "<conversation>" not in build_prompt("How do I amend?", HITS, [])


def test_search_text_adds_earlier_user_questions_but_not_answers():
    assert search_text("How do I delete it?", HISTORY) == "How do I create a tag?\nHow do I delete it?"
    assert search_text("How do I delete it?", []) == "How do I delete it?"

def test_build_prompt_puts_memories_after_the_excerpts():
    memories = [{"text": "The user writes Go."}, {"text": "The user works on Windows."}]
    prompt = build_prompt("How do I amend?", HITS, HISTORY, memories)
    assert "</excerpt>\n\n<memories>\n- The user writes Go.\n- The user works on Windows.\n</memories>\n\n<conversation>" in prompt
    assert "<memories>" not in build_prompt("How do I amend?", HITS, HISTORY, [])


def test_parse_facts_reads_a_json_list():
    assert parse_facts('[{"text": "The user writes Go.", "topics": ["language"]}]') == [
        {"text": "The user writes Go.", "topics": ["language"]}
    ]


def test_parse_facts_accepts_a_fenced_code_block():
    assert parse_facts('```json\n[{"text": "The user writes Go."}]\n```') == [{"text": "The user writes Go."}]


def test_parse_facts_returns_nothing_for_anything_else():
    assert parse_facts("I found no facts.") == []
    assert parse_facts('{"text": "not a list"}') == []
    assert parse_facts('["just a string", {"no_text": 1}]') == []
    assert parse_facts("[]") == []