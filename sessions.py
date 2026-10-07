import json
import uuid

import store
from embeddings import count_tokens

PREFIX = "session:"
TTL_SECONDS = 24 * 60 * 60
MAX_MESSAGES = 20
SUMMARY_TOKENS = 1000
KEEP_MESSAGES = 2

def new_id() -> str:
    return uuid.uuid4().hex


def append(session_id: str, role: str, text: str) -> None:
    key = PREFIX + session_id
    store.r.rpush(key, json.dumps({"role": role, "text": text}))
    store.r.expire(key, TTL_SECONDS)
    store.r.expire(key + ":summary", TTL_SECONDS)

def history(session_id: str, limit: int = MAX_MESSAGES) -> list[dict]:
    raw = store.r.lrange(PREFIX + session_id, -limit, -1)
    return [json.loads(item) for item in raw]

def summary(session_id: str) -> str:
    return (store.r.get(f"{PREFIX}{session_id}:summary") or b"").decode()


def compact(session_id: str, summarise) -> bool:
    messages = history(session_id)
    old = messages[:-KEEP_MESSAGES]
    if not old or sum(count_tokens(message["text"]) for message in messages) <= SUMMARY_TOKENS:
        return False
    text = summarise(summary(session_id), old)
    store.r.set(f"{PREFIX}{session_id}:summary", text, ex=TTL_SECONDS)
    store.r.ltrim(PREFIX + session_id, -KEEP_MESSAGES, -1)
    return True