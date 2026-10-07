import pytest

import cache
import memory
import sessions
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

@pytest.fixture
def isolated_sessions(monkeypatch):
    monkeypatch.setattr(sessions, "PREFIX", "test-session:")
    yield
    for key in store.r.scan_iter("test-session:*"):
        store.r.delete(key)

@pytest.fixture
def isolated_memory(monkeypatch):
    monkeypatch.setattr(memory, "PREFIX", "test-memory:")
    monkeypatch.setattr(memory, "INDEX", "idx:test-memory")
    memory.ensure_index()
    yield
    for key in store.r.scan_iter("test-memory:*"):
        store.r.delete(key)
    store.r.ft(memory.INDEX).dropindex()