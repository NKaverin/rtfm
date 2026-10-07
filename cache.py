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

PRICES = {  # US dollars per million tokens: (input, output)
    "gemini-3.8-flash": (0.75, 3.75),
}

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

def record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    store.r.incrby(f"{METRICS}usage:{model}:input", input_tokens)
    store.r.incrby(f"{METRICS}usage:{model}:output", output_tokens)

def _number(name: str) -> float:
    return float(store.r.get(METRICS + name) or 0)

def usage() -> dict:
    totals = {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}
    for model, (input_price, output_price) in PRICES.items():
        used_in, used_out = int(_number(f"usage:{model}:input")), int(_number(f"usage:{model}:output"))
        totals["input_tokens"] += used_in
        totals["output_tokens"] += used_out
        totals["estimated_cost_usd"] += (used_in * input_price + used_out * output_price) / 1_000_000
    totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 6)
    return totals

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
        **usage(),
    }
