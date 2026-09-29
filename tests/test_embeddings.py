import os
from types import SimpleNamespace

import pytest

import embeddings
from embeddings import DIM, cosine_similarity, embed


def test_same_direction_is_one():
    assert cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)

def test_opposite_direction_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

def test_unrelated_directions_are_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

def test_zero_vector_is_rejected():
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) 

def test_embed_returns_one_row_per_text():
    vectors = embed(["first text", "second text"])
    assert vectors.shape == (2, DIM)

def test_related_sentence_scores_higher_than_unrelated():
    question, related, unrelated = embed([
        "How do I undo my last commit?",
        "git revert creates a new commit that undoes an erlier one",
        "Bananas are rich in potassium",
    ])
    assert cosine_similarity(question, related) > cosine_similarity(question, unrelated)

class FakeGeminiClient:
    """Stands in for genai, return made-up vectors"""

    def __init__(self):
        self.requests = []
        self.models = self

    def embed_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "dim": config.output_dimensionality})
        vectors = [SimpleNamespace(values=[float(i)] * 768) for i in range(len(contents))]
        return SimpleNamespace(embeddings=vectors)

def test_gemini_backend_turns_response_into_float32_array(monkeypatch):
    fake = FakeGeminiClient()
    monkeypatch.setattr(embeddings.genai, "Client", lambda: fake)
    monkeypatch.setattr(embeddings, "BACKEND", "gemini")

    vectors = embeddings.embed(["first", "second"])

    assert vectors.shape == (2, 768)
    assert vectors.dtype == "float32"
    assert vectors[1][0] == 1.0
    assert fake.requests == [
      {"model": "gemini-embedding-001", "contents": ["first", "second"], "dim": 768}      
    ]

@pytest.mark.skipif("GEMINI_API_KEY" not in os.environ, reason="needs GEMINI_API_KEY")
def test_gemini_real_call_related_beats_unrelated(monkeypatch):
    monkeypatch.setattr(embeddings, "BACKEND", "gemini")
    question, related, unrelated = embeddings.embed([
        "How do I undo my last commit?",
        "git revert creates a new commit that undoes an earlier one",
        "Bananas are rich in potasium",
    ])
    assert cosine_similarity(question, related) > cosine_similarity(question, unrelated)