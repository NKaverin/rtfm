import pytest

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
