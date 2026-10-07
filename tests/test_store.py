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


def save_three_sources():
    vector = embed(["git tag"])[0]
    store.save_chunk("a", "git tag", "02-git-basics/sections/tagging.asc", "H", 0, vector)
    store.save_chunk("b", "git tag", "02-git-basics/sections/undoing.asc", "H", 0, vector)
    store.save_chunk("c", "git tag", "03-git-branching/sections/basic.asc", "H", 0, vector)
    return vector


def test_escape_tag_puts_a_backslash_before_punctuation():
    assert store.escape_tag("02-git/a_b.asc") == "02\\-git\\/a_b\\.asc"


def test_search_can_be_limited_to_one_file_or_one_chapter():
    vector = save_three_sources()
    one_file = store.search(vector, k=3, source="02-git-basics/sections/tagging.asc")
    one_chapter = store.search(vector, k=3, source="02-git-basics")
    assert [hit["id"] for hit in one_file] == ["a"]
    assert sorted(hit["id"] for hit in one_chapter) == ["a", "b"]
    assert store.search(vector, k=3, source="no-such-chapter") == []


def test_sources_lists_each_file_once_in_order():
    save_three_sources()
    store.save_chunk("d", "more", "02-git-basics/sections/tagging.asc", "H", 1, embed(["more"])[0])
    assert store.sources() == [
        "02-git-basics/sections/tagging.asc",
        "02-git-basics/sections/undoing.asc",
        "03-git-branching/sections/basic.asc",
    ]