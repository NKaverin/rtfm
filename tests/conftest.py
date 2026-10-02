import pytest

import store

@pytest.fixture
def isolated_store(monkeypatch):
    monkeypatch.setattr(store, "PREFIX", "test-chunk:")
    monkeypatch.setattr(store, "INDEX", "idx:test-chunks")
    store.ensure_index()
    yield
    for chunk_id in store.chunk_ids():
        store.delete_chunk(chunk_id)
    store.r.ft(store.INDEX).dropindex()