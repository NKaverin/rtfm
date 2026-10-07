import re
import time

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from embeddings import BACKEND, DIM

r = redis.Redis(host="localhost", port=6379)

PREFIX = f"chunk:{BACKEND}"
INDEX = f"idx:chunks:{BACKEND}"

def _key(chunk_id: str) -> str:
    return PREFIX + chunk_id

def create_index(index: str, prefix: str, schema: tuple) -> None:
    if index.encode() in r.execute_command("FT._LIST"):
        return
    definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
    r.ft(index).create_index(schema, definition=definition)
    while int(r.ft(index).info()["indexing"]):
        time.sleep(0.01)

def vector_field() -> VectorField:
    return VectorField("vector", "FLAT", {"TYPE": "FLOAT32", "DIM": DIM, "DISTANCE_METRIC": "COSINE"})

def ensure_index() -> None:
    schema = (
        TextField("text"),
        TagField("source"),
        TextField("heading"),
        NumericField("position"),
        vector_field(),
    )
    create_index(INDEX, PREFIX, schema)

def chunk_ids() -> list[str]:
    return sorted(key.decode().removeprefix(PREFIX) for key in r.scan_iter(PREFIX + "*"))

def save_chunk(
        chunk_id: str, 
        text: str, 
        source: str,
        heading: str,
        position: int, 
        vector: np.ndarray
) -> None:
    if vector.dtype != np.float32 or vector.shape != (DIM, ):
        raise ValueError(f"vector must be float32 with shape ({DIM},), got {vector.dtype} {vector.shape}")    
    fields = {
        "text": text,
        "source": source,
        "heading": heading,
        "position": position,
        "vector": vector.tobytes(),
    }
    r.hset(_key(chunk_id), mapping=fields)
    
def load_chunk(chunk_id: str) -> dict | None:
    raw = r.hgetall(_key(chunk_id))
    if not raw: # missing key returns empty dict {}
        return None
    chunk = {k.decode(): v.decode() for k, v in raw.items() if k != b"vector"}
    chunk["position"] = int(chunk["position"])
    return chunk

def delete_chunk(chunk_id: str) -> None:
    r.delete(_key(chunk_id))

def escape_tag(value: str) -> str:
    return re.sub(r"([^A-Za-z0-9_])", r"\\\1", value)


def sources() -> list[str]:
    return sorted(value.decode() for value in r.execute_command("FT.TAGVALS", INDEX, "source"))

def search(vector: np.ndarray, k: int = 3, source: str | None = None) -> list[dict]:
    candidates = f"(@source:{{{escape_tag(source)}*}})" if source else "*"
    query = (
        Query(f"{candidates}=>[KNN {k} @vector $vec AS distance]")
        .sort_by("distance")
        .return_fields("text", "source", "heading", "position", "distance")
        .paging(0, k)
    )
    result = r.ft(INDEX).search(query, query_params={"vec": vector.tobytes()})
    return [
        {
            "id": doc.id.removeprefix(PREFIX),
            "text": doc.text,
            "source": doc.source,
            "heading": doc.heading,
            "position": int(doc.position),
            "distance": float(doc.distance),
        }
        for doc in result.docs
    ]

if __name__ == "__main__":
    from embeddings import embed

    print(escape_tag("02-git-basics/sections/tagging.asc"))
    print(len(sources()), "files, first three:", sources()[:3])
    vector = embed(["How do I delete something?"])[0]
    for source in [None, "02-git-basics", "02-git-basics/sections/tagging.asc", "no-such-chapter"]:
        print("source =", source)
        for hit in search(vector, 3, source):
            print(f"   {hit['distance']:.3f}  {hit['source']}  ({hit['heading']})")
