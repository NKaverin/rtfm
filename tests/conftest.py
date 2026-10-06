import pytest

import cache
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

@pytest.fixture
def isolated_cache(monkeypatch):
    monkeypatch.setattr(cache, "PREFIX", "test-cache:")
    monkeypatch.setattr(cache, "INDEX", "idx:test-cache")
    monkeypatch.setattr(cache, "METRICS", "test-metrics:")
    cache.ensure_index()
    yield
    cache.flush()
    for key in store.r.scan_iter("test-metrics:*"):
        store.r.delete(key)
    store.r.ft(cache.INDEX).dropindex()