import redis

r = redis.Redis(host="localhost", port=6379)

PREFIX = "chunk: "

def _key(chunk_id: str) -> str:
    return PREFIX + chunk_id

def save_chunk(chunk_id: str, text: str, source: str) -> None:
    r.hset(_key(chunk_id), mapping={"text": text, "source": source})
    
def load_chunk(chunk_id: str) -> dict | None:
    raw = r.hgetall(_key(chunk_id))
    if not raw:
        return None
    return {k.decode(): v.decode() for k, v in raw.items()}

if __name__ == "__main__":
    save_chunk("foo", "bar", "ch")
    print(load_chunk("foo"))
    print(load_chunk("foo")["text"])
    print(load_chunk("bar"))