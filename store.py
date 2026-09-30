import numpy as np
import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from embeddings import BACKEND, DIM

r = redis.Redis(host="localhost", port=6379)

PREFIX = "chunk:{BACKEND}"
INDEX = "idx:chunks:{BACKEND}"

def _key(chunk_id: str) -> str:
    return PREFIX + chunk_id

def ensure_index() -> None:
    if INDEX.encode() in r.execute_command("FT._LIST"):
        return
    schema = (
        TextField("text"),
        TagField("source"),
        VectorField("vector", "FLAT", {"TYPE": "FLOAT32", "DIM": DIM, "DISTANCE_METRIC": "COSINE"}),
    )
    definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    r.ft(INDEX).create_index(schema, definition=definition)

def chunk_ids() -> list[str]:
    return sorted(key.decode().removeprefix(PREFIX) for key in r.scan_iter(PREFIX + "*"))

def save_chunk(chunk_id: str, text: str, source: str, vector: np.ndarray) -> None:
    if vector.dtype != np.float32 or vector.shape != (DIM, ):
        raise ValueError(f"vector must be float32 with shape ({DIM},), got {vector.dtype} {vector.shape}")    
    r.hset(_key(chunk_id), mapping={"text": text, "source": source, "vector": vector.tobytes()})
    
def load_chunk(chunk_id: str) -> dict | None:
    raw = r.hgetall(_key(chunk_id))
    if not raw: # missing key returns empty dict {}
        return None
    return {k.decode(): v.decode() for k, v in raw.items() if k != b"vector"}

def delete_chunk(chunk_id: str) -> None:
    r.delete(_key(chunk_id))

def search(vector: np.ndarray, k: int = 3) -> list[dict]:
    query = (
        Query(f"*=>[KNN {k} @vector $vec AS distance]")
        .sort_by("distance")
        .return_fields("text", "source", "distance")
        .paging(0, k)
    )
    result = r.ft(INDEX).search(query, query_params={"vec": vector.tobytes()})
    return [
        {
            "id": doc.id.removeprefix(PREFIX),
            "text": doc.text,
            "source": doc.source,
            "distance": float(doc.distance),
        }
        for doc in result.docs
    ]

if __name__ == "__main__":
    from embeddings import embed
