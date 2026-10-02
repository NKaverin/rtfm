import numpy as np
import pytest

import store
from embeddings import DIM, embed

pytestmark = pytest.mark.usefixtures("isolated_store")

def save_test_chunk(chunk_id: str, text: str, source: str = "default") -> None:
    store.save_chunk(chunk_id, text, source, "Test heading", 0, embed([text])[0])

def test_save_then_load_returns_same_fields():
    save_test_chunk("test1", "foo", "bar.foo")
    assert store.load_chunk("test1") == {
        "text": "foo", 
        "source": "bar.foo",
        "heading": "Test heading", 
        "position": 0,
    }

def test_load_missing_chunk_returns_none():
    assert store.load_chunk("test-does-not-exist") is None

def test_chunk_ids_lists_saved_chunks():
    save_test_chunk("test1", "foo", "foo.bar")
    save_test_chunk("test2", "bar", "bar.foo")
    ids = store.chunk_ids()
    assert "test1" in ids
    assert "test2" in ids

def test_delete_chunk_removes_it():
    save_test_chunk("test1", "foo", "foo.bar")
    store.delete_chunk("test1")
    assert store.load_chunk("test1") is None

def test_save_rejects_wrong_vector_type():
    with pytest.raises(ValueError):
        store.save_chunk("test-bad", "x", "foo.bar", "Test heading", 0, np.zeros(DIM, dtype=np.float64))

def test_save_rejects_wrong_vector_length():
    with pytest.raises(ValueError):
        store.save_chunk("test-bad", "x", "foo.bar", "Test heading", 0, np.zeros(DIM + 1, dtype=np.float32))

def test_search_finds_nearest_chunk_first():
    save_test_chunk("test-revert", "git revert creates a new commit that undoes an earlier one")
    save_test_chunk("test-banana", "Babnanas are rich in potassium")
    question = embed(["git revert creates a new commit that undoes an earlier one"])
    hits = store.search(question, k=2)
    assert hits[0]["id"] == "test-revert"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)