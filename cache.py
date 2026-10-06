import hashlib
import json

import numpy as np
from redis.commands.search.field import TextField
from redis.commands.search.query import Query

import store
from embeddings import BACKEND

PREFIX = f"cache:{BACKEND}:"
INDEX = f"idx:cache:{BACKEND}"
MAX_DISTANCE = 0.15
METRICS = "metrics:"


def ensure_index() -> None:
    store.create_index(INDEX, PREFIX, (TextField("question"), store.vector_field()))

def save(question: str, vector: np.ndarray, entry: dict) -> None:
    ensure_index()
    entry_id = hashlib.sha256(question.encode()).hexdigest()[:16]
    fields = {"question": question, "result": json.dumps(entry), "vector": vector.tobytes()}
    store.r.hset(PREFIX + entry_id, mapping=fields)


def lookup(vector: np.ndarray) -> dict | None:
    ensure_index()
    query = (
        Query("*=>[KNN 1 @vector $vec AS distance]")
        .return_fields("question", "result", "distance")
        .paging(0, 1)
    )
    docs = store.r.ft(INDEX).search(query, query_params={"vec": vector.tobytes()}).docs
    if not docs or float(docs[0].distance) > MAX_DISTANCE:
        return None
    return json.loads(docs[0].result)

def flush() -> int:
    keys = list(store.r.scan_iter(PREFIX + "*"))
    if keys:
        store.r.delete(*keys)
    return len(keys)


def record(outcome: str, seconds: float, tokens_saved: int = 0) -> None:
    store.r.incr(f"{METRICS}{outcome}:count")
    store.r.incrbyfloat(f"{METRICS}{outcome}:seconds", seconds)
    if tokens_saved:
        store.r.incrby(f"{METRICS}tokens_saved", tokens_saved)


def _number(name: str) -> float:
    return float(store.r.get(METRICS + name) or 0)


def metrics() -> dict:
    hits, misses = int(_number("hit:count")), int(_number("miss:count"))
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 3) if total else 0.0,
        "avg_hit_seconds": round(_number("hit:seconds") / hits, 3) if hits else 0.0,
        "avg_miss_seconds": round(_number("miss:seconds") / misses, 3) if misses else 0.0,
        "tokens_saved": int(_number("tokens_saved")),
    }
