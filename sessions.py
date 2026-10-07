import json
import uuid

import store

PREFIX = "session:"
TTL_SECONDS = 24 * 60 * 60
MAX_MESSAGES = 6

def new_id() -> str:
    return uuid.uuid4().hex


def append(session_id: str, role: str, text: str) -> None:
    key = PREFIX + session_id
    store.r.rpush(key, json.dumps({"role": role, "text": text}))
    store.r.expire(key, TTL_SECONDS)


def history(session_id: str, limit: int = MAX_MESSAGES) -> list[dict]:
    raw = store.r.lrange(PREFIX + session_id, -limit, -1)
    return [json.loads(item) for item in raw]

def main():
    session_id = new_id()
    print(session_id)
    append(session_id, "user", "How do I create a tag?")
    append(session_id, "assistant", "Use git tag -a v1.0 [tagging.asc].")
    append(session_id, "user", "How do I delete it?")
    for message in history(session_id):
        print(message)
    print("last 2:", [m["text"] for m in history(session_id, limit=2)])
    print("ttl:", store.r.ttl(PREFIX + session_id), "| other session:", history("nobody"))


if __name__ == "__main__":
    main()