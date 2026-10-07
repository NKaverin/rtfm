import hashlib
import time

import numpy as np
from redis.commands.search.field import NumericField, TagField, TextField
from redis.commands.search.query import Query

import store
from embeddings import BACKEND, embed

PREFIX = f"memory:{BACKEND}"
INDEX = f"idx:memory:{BACKEND}"
DUPLICATE_DISTANCE = 0.1

def ensure_index() -> None:
    schema = (
        TextField("text"),
        TagField("user_id"),
        TagField("topics"),
        TagField("entities"),
        NumericField("created"),
        store.vector_field(),
    )
    store.create_index(INDEX, PREFIX, schema)


def _search(user_id: str, vector: np.ndarray, k: int) -> list[dict]:
    ensure_index()
    query = (
        Query(f"(@user_id:{{{user_id}}})=>[KNN {k} @vector $vec AS distance]")
        .sort_by("distance")
        .return_fields("text", "topics", "entities", "distance")
        .paging(0, k)
    )
    docs = store.r.ft(INDEX).search(query, query_params={"vec": vector.tobytes()}).docs
    return [
        {
            "text": doc.text,
            "topics": [t for t in doc.topics.split(",") if t],
            "entities": [e for e in doc.entities.split(",") if e],
            "distance": float(doc.distance),
        }
        for doc in docs
    ]


def remember(user_id: str, facts: list[dict]) -> list[str]:
    ensure_index()
    stored = []
    for fact in facts:
        vector = embed([fact["text"]])[0]
        nearest = _search(user_id, vector, 1)
        if nearest and nearest[0]["distance"] < DUPLICATE_DISTANCE:
            continue
        entry_id = hashlib.sha256(f"{user_id}\n{fact['text']}".encode()).hexdigest()[:16]
        fields = {
            "text": fact["text"],
            "user_id": user_id,
            "topics": ",".join(fact.get("topics", [])),
            "entities": ",".join(fact.get("entities", [])),
            "created": int(time.time()),
            "vector": vector.tobytes(),
        }
        store.r.hset(f"{PREFIX}{user_id}:{entry_id}", mapping=fields)
        stored.append(fact["text"])
    return stored

def recall(user_id: str, vector: np.ndarray, k: int = 3) -> list[dict]:
    return _search(user_id, vector, k)

def all_memories(user_id: str) -> list[dict]:
    ensure_index()
    query = Query(f"@user_id:{{{user_id}}}").sort_by("created").return_fields("text", "topics", "entities").paging(0, 100)
    docs = store.r.ft(INDEX).search(query).docs
    return [
        {"text": d.text, "topics": [t for t in d.topics.split(",") if t], "entities": [e for e in d.entities.split(",") if e]}
        for d in docs
    ]


def forget(user_id: str) -> int:
    keys = list(store.r.scan_iter(f"{PREFIX}{user_id}:*"))
    if keys:
        store.r.delete(*keys)
    return len(keys)    