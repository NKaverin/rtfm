import pytest

import store

@pytest.fixture(autouse=True)
def remove_test_chunks():
    yield
    for chunk_id in store.chunk_ids():
        if chunk_id.startswith("test"):
            store.delete_chunk(chunk_id)

def test_save_then_load_returns_same_fields():
    store.save_chunk("test1", "foo", "bar.foo")
    assert store.load_chunk("test1") == {"text": "foo", "source": "bar.foo"}

def test_load_missing_chunk_returns_none():
    assert store.load_chunk("test-does-not-exist") is None

def test_chunk_ids_lists_saved_chunks():
    store.save_chunk("test1", "foo", "foo.bar")
    store.save_chunk("test2", "bar", "bar.foo")
    ids = store.chunk_ids()
    assert "test1" in ids
    assert "test2" in ids

def test_delete_chunk_removes_it():
    store.save_chunk("test1", "foo", "foo.bar")
    store.delete_chunk("test1")
    assert store.load_chunk("test1") is None